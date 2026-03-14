import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import

from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class RequestType(str, Enum):
    """Types of vector database operations"""

    INGESTION = "ingestion"  # Insert vectors and texts
    SEARCHING = "searching"  # Search for similar vectors


@dataclass
class VectorDBRequest:
    """Data class for vector database requests"""

    request_id: str
    query_id: str
    request_type: RequestType
    data: Union[
        List[Tuple[Union[np.ndarray, list], str]],
        Tuple[List[Union[np.ndarray, list]], int],
    ]  # (embeddings, texts) for ingestion or (query_vectors, top_k) for searching
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = field(default_factory=time.time)


@ray.remote
class VectorDBEngine:
    """Ray Actor for serving vector database operations using FAISS

    Features:
    - Async request handling
    - Batched insertions
    - Concurrent read/write operations
    - Global FAISS index
    """

    def __init__(
        self,
        vector_dim: int = 768,
        index_path: Optional[str] = None,
        max_batch_size: int = 1000,
        max_queue_size: int = 2000,
        normalize_L2: bool = False,
        distance_strategy: str = "MAX_INNER_PRODUCT",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        """
        Initialize FAISS-based VectorDBEngine

        Args:
            vector_dim: Dimension of vectors
            index_path: Optional path to load existing FAISS index from
            max_batch_size: Maximum batch size for insertions
            max_queue_size: Maximum queue size for requests
            normalize_L2: Whether to normalize vectors
            distance_strategy: Distance strategy for similarity search
            relevance_score_fn: Function to convert raw scores to relevance scores
            scheduler_ref: Reference to scheduler actor
        """
        self.vector_dim = vector_dim
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.index_path = index_path
        self.normalize_L2 = normalize_L2
        self.distance_strategy = distance_strategy
        self.relevance_score_fn = relevance_score_fn

        self.name = kwargs.get("name", None)

        # Async queues for different operations
        self.insert_queue = asyncio.Queue(maxsize=max_queue_size)
        self.search_queue = asyncio.Queue(maxsize=max_queue_size)

        # FAISS index and related components
        self.faiss_index = None
        self.docstore = None
        self.index_to_docstore_id = {}
        self._next_doc_id = 0

        # Track active requests
        self.query_requests: Dict[str, List[VectorDBRequest]] = {}

        # Start processing tasks
        self.running = True
        self.tasks = []
        self._initialized = False
        self._init_error: Optional[Exception] = None
        self._ready_event = asyncio.Event()
        self._init_task = asyncio.create_task(self._initialize())

        self.scheduler_ref = scheduler_ref

    async def is_ready(self, timeout_s: float = 120.0):
        """Block until initialization completes (or fails)."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"VectorDBEngine initialization timed out after {timeout_s}s"
            ) from e

        if self._init_error is not None:
            raise RuntimeError(
                f"VectorDBEngine failed to initialize: {self._init_error}"
            ) from self._init_error

        return self._initialized

    async def _initialize(self):
        """Initialize FAISS vector store and start processing tasks"""
        try:
            # Initialize FAISS vector store
            if self.index_path:
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    embeddings=None,
                    allow_dangerous_deserialization=True,
                    normalize_L2=self.normalize_L2,
                    distance_strategy=self.distance_strategy,
                    relevance_score_fn=self.relevance_score_fn,
                )
            else:
                logger.info("Creating new FAISS index")
                # Create empty FAISS index - will be populated on first insert
                # We need a dummy text to initialize the index

                faiss = dependable_faiss_import()

                # Create appropriate index type based on distance strategy
                if self.distance_strategy == "MAX_INNER_PRODUCT":
                    # Inner product - used for cosine similarity with normalized vectors
                    index = faiss.IndexFlatIP(self.vector_dim)
                    logger.info(f"Created FAISS IndexFlatIP with dim={self.vector_dim}")
                else:
                    # Default to L2 distance
                    index = faiss.IndexFlatL2(self.vector_dim)
                    logger.info(f"Created FAISS IndexFlatL2 with dim={self.vector_dim}")

                docstore = InMemoryDocstore()
                index_to_docstore_id = {}

                self.vector_store = FAISS(
                    index=index,
                    embedding_function=None,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                    normalize_L2=self.normalize_L2,
                    distance_strategy=self.distance_strategy,
                    relevance_score_fn=self.relevance_score_fn,
                )

            # Start processing tasks
            self.tasks = [
                asyncio.create_task(self._process_inserts()),
                asyncio.create_task(self._process_searches()),
            ]
            self._initialized = True
            logger.info("VectorDBEngine initialized successfully")
        except Exception as e:
            self._init_error = e
            self.running = False
            logger.error(f"VectorDBEngine initialization failed: {e}")
            raise
        finally:
            self._ready_event.set()

    async def _finish_request(self, request: VectorDBRequest, result: Any) -> None:
        """Send result to scheduler and release tracking state."""
        result_ref = ray.put(result)
        if self.scheduler_ref is not None:
            await self.scheduler_ref.on_result.remote(
                request.request_id, request.query_id, result_ref
            )

        if request.query_id in self.query_requests:
            try:
                self.query_requests[request.query_id].remove(request)
            except ValueError:
                pass
            if len(self.query_requests[request.query_id]) == 0:
                del self.query_requests[request.query_id]

    async def _finish_request_error(
        self, request: VectorDBRequest, err: Exception, where: str
    ) -> None:
        """Propagate engine errors instead of letting requests hang."""
        message = (
            f"VectorDBEngine {where} failed for {request.request_id} "
            f"(query={request.query_id}): {err}"
        )
        logger.error(message)
        await self._finish_request(request, RuntimeError(message))

    async def ingest(
        self,
        request_id: str,
        query_id: str,
        embeddings: List[np.ndarray],
        texts: List[str],
    ) -> ray.ObjectRef:
        """Ingest vectors and texts into database

        Args:
            request_id: Unique identifier for this request
            query_id: Query identifier (not used for table management in FAISS)
            embeddings: List of vector embeddings
            texts: List of corresponding texts

        Returns:
            Ray ObjectRef for tracking completion
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        data = list(zip(embeddings, texts))
        return await self.submit_request(
            request_id=request_id,
            query_id=query_id,
            request_type=RequestType.INGESTION,
            data=data,
        )

    async def search(
        self,
        request_id: str,
        query_id: str,
        query_vectors: List[Union[np.ndarray, list]],
        top_k: int = 5,
    ) -> ray.ObjectRef:
        """Search for similar vectors in database

        Args:
            request_id: Unique identifier for this request
            query_id: Query identifier
            query_vectors: List of vectors to search for (can be single vector in list)
            top_k: Number of results to return

        Returns:
            Ray ObjectRef for results
        """
        # Ensure query_vectors is a list
        if not isinstance(query_vectors, list):
            query_vectors = [query_vectors]

        return await self.submit_request(
            request_id=request_id,
            query_id=query_id,
            request_type=RequestType.SEARCHING,
            data=(query_vectors, top_k),
        )

    async def submit_request(
        self,
        request_id: str,
        query_id: str,
        request_type: RequestType,
        data: Union[
            List[Tuple[Union[np.ndarray, list], str]],
            Tuple[List[Union[np.ndarray, list]], int],
        ],
    ) -> None:
        """Submit a new vector database request"""

        request = VectorDBRequest(
            request_id=request_id,
            query_id=query_id,
            request_type=request_type,
            data=data,
            callback_ref=None,
        )

        logger.info(
            f"submit request in vector_db: {request.request_type} {request.request_id} {request.query_id}"
        )

        # Route request to appropriate queue
        if request_type == RequestType.INGESTION:
            if self.insert_queue.qsize() >= self.max_queue_size:
                raise RuntimeError("Ingestion queue is full")
            logger.debug(f"put request {request.request_id} in vector_db insert queue")
            await self.insert_queue.put(request)
        else:  # RequestType.SEARCHING
            if self.search_queue.qsize() >= self.max_queue_size:
                raise RuntimeError("Search queue is full")
            await self.search_queue.put(request)

        # Track request
        if query_id not in self.query_requests:
            self.query_requests[query_id] = []
        self.query_requests[query_id].append(request)

    async def _process_inserts(self):
        """Process insertion requests"""
        while self.running:
            batch_requests: List[VectorDBRequest] = []
            unresolved_by_id: Dict[str, VectorDBRequest] = {}
            try:
                # Collect batch of insertion requests
                while len(batch_requests) == 0:
                    try:
                        request = await asyncio.wait_for(
                            self.insert_queue.get(), timeout=0.1
                        )
                        batch_requests.append(request)
                        unresolved_by_id[request.request_id] = request
                        logger.debug(
                            f"vector_db insert batch_requests len: {len(batch_requests)}"
                        )
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"vector_db insert error: {e}")
                        break

                if not batch_requests:
                    continue

                # Collect all texts and embeddings from batch
                all_texts = []
                all_embeddings = []

                for request in batch_requests:
                    for embedding, text in request.data:
                        # Convert numpy array to list if needed
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()

                        all_texts.append(text)
                        all_embeddings.append(embedding)

                # Batch insert into FAISS
                begin = time.time()

                text_embedding_pairs = list(zip(all_texts, all_embeddings))
                self.vector_store.add_embeddings(
                    text_embeddings=text_embedding_pairs,
                )

                end = time.time()
                logger.debug(
                    f"FAISS insert time for {len(all_texts)} vectors: {end - begin}"
                )

                # Update results and clean up
                for request in batch_requests:
                    await self._finish_request(request, True)
                    unresolved_by_id.pop(request.request_id, None)

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error in insert processing: {e}")
                for request in list(unresolved_by_id.values()):
                    try:
                        await self._finish_request_error(request, e, "ingestion")
                    except Exception as callback_err:
                        logger.error(
                            f"Failed to propagate ingestion error for "
                            f"{request.request_id}: {callback_err}"
                        )
                continue

    async def _process_searches(self):
        """Process search requests"""
        while self.running:
            request: Optional[VectorDBRequest] = None
            try:
                try:
                    request = await asyncio.wait_for(
                        self.search_queue.get(), timeout=0.02
                    )
                except asyncio.TimeoutError:
                    continue

                query_vectors, top_k = request.data

                logger.debug(
                    f"search in vector_db: {request.request_id} {request.query_id}"
                )

                # Execute vector search for each query vector
                begin = time.time()
                all_results = []

                for query_vector in query_vectors:
                    # Ensure query vector is in correct format
                    if isinstance(query_vector, np.ndarray):
                        query_vector = query_vector.tolist()

                    # Perform similarity search with scores
                    docs_and_scores = self.vector_store.similarity_search_by_vector(
                        embedding=query_vector, k=top_k
                    )

                    # Format results
                    search_results = [doc.page_content for doc in docs_and_scores]
                    all_results.append(search_results)

                end = time.time()
                logger.debug(
                    f"search time for {len(query_vectors)} vectors: {end - begin}"
                )

                # If there is only one query vector, return a single result list
                if len(all_results) == 1:
                    search_results = all_results[0]
                else:
                    search_results = all_results

                logger.debug(f"search results count: {len(search_results)}")

                # Update results and clean up
                await self._finish_request(request, search_results)

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error in search processing: {e}")
                if request is not None:
                    try:
                        await self._finish_request_error(request, e, "search")
                    except Exception as callback_err:
                        logger.error(
                            f"Failed to propagate search error for "
                            f"{request.request_id}: {callback_err}"
                        )
                continue

    async def cleanup_query(self, query_id: str):
        """Clean up resources for a query - with FAISS, we don't need to drop tables"""
        # In FAISS, we can optionally remove documents by query_id
        if query_id in self.query_requests:
            del self.query_requests[query_id]

        # Note: FAISS doesn't support easy deletion by metadata filter
        # If needed, we could implement a cleanup that removes all documents with this query_id
        logger.debug(f"Cleaned up query_id: {query_id}")

    async def save_index(self, save_path: str):
        """Save the FAISS index to disk"""
        if self.vector_store:
            self.vector_store.save_local(save_path)
            logger.info(f"Saved FAISS index to {save_path}")
        else:
            logger.error("No vector store to save")

    async def shutdown(self):
        """Shutdown the service"""
        self.running = False

        # Cancel processing tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("VectorDBEngine shutdown complete")
