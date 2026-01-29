import asyncio
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import ray

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
        List[Tuple[Union[np.ndarray, list], str]], Tuple[Union[np.ndarray, list], int]
    ]  # (embeddings, texts) for ingestion or (query_vector, top_k) for searching
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = time.time()


@ray.remote
class VectorDBEngine:
    """Ray Actor for serving vector database operations

    Features:
    - Async request handling
    - Batched insertions
    - Per-query in-memory FAISS indices
    - Concurrent read/write operations
    """

    def __init__(
        self,
        max_batch_size: int = 1000,
        max_queue_size: int = 2000,
        vector_dim: int = 768,
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.vector_dim = vector_dim

        self.name = kwargs.get("name", None)

        # Async queues for different operations
        self.insert_queue = asyncio.Queue(maxsize=max_queue_size)
        self.search_queue = asyncio.Queue(maxsize=max_queue_size)

        # Track active indices and requests
        self.active_indices: Dict[str, str] = {}  # query_id -> index_name
        self.index_store: Dict[str, Dict[str, Any]] = {}  # query_id -> {index, texts}
        self.query_requests: Dict[str, List[VectorDBRequest]] = {}

        # Start processing tasks
        self.running = True
        self.tasks = []
        self.tasks = [
            asyncio.create_task(self._process_inserts()),
            asyncio.create_task(self._process_searches()),
        ]

        self.scheduler_ref = scheduler_ref

    def is_ready(self):
        """Check if the engine is ready"""
        return True

    def _get_index_name(self, query_id: str) -> str:
        """Generate unique index name for a query_id"""
        hash_obj = hashlib.md5(query_id.encode())
        return f"vectors_{hash_obj.hexdigest()}"

    def _get_or_create_index(
        self, query_id: str
    ) -> Tuple[faiss.IndexFlatIP, List[str]]:
        """Create an in-memory FAISS index for the query if absent"""
        if query_id not in self.index_store:
            logger.debug(f"ensure index for query_id: {query_id}")
            index = faiss.IndexFlatIP(self.vector_dim)
            self.index_store[query_id] = {"index": index, "texts": []}
            self.active_indices[query_id] = self._get_index_name(query_id)

        store = self.index_store[query_id]
        return store["index"], store["texts"]

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
            query_id: Query identifier for table management
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
        self, request_id: str, query_id: str, query_vector: np.ndarray, top_k: int = 5
    ) -> ray.ObjectRef:
        """Search for similar vectors in database

        Args:
            request_id: Unique identifier for this request
            query_id: Query identifier for table lookup
            query_vector: Vector to search for
            top_k: Number of results to return

        Returns:
            Ray ObjectRef for results
        """
        return await self.submit_request(
            request_id=request_id,
            query_id=query_id,
            request_type=RequestType.SEARCHING,
            data=(query_vector, top_k),
        )

    async def submit_request(
        self,
        request_id: str,
        query_id: str,
        request_type: RequestType,
        data: Union[List[Tuple[np.ndarray, str]], Tuple[np.ndarray, int]],
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
            try:
                # Collect batch of insertion requests
                batch_requests = []
                batch_data = []

                # while len(batch_requests) < self.max_batch_size:
                while len(batch_requests) == 0:
                    try:
                        request = await asyncio.wait_for(
                            self.insert_queue.get(), timeout=0.1
                        )
                        batch_requests.append(request)
                        batch_data.append(request.data)

                        logger.debug(
                            f"vector_db insert batch_requests len: {len(batch_requests)}"
                        )

                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        logger.error(f"vector_db insert error: {e}")
                        break

                if not batch_requests:
                    continue

                # Group requests by query_id
                query_groups: Dict[str, List[int]] = {}
                for i, request in enumerate(batch_requests):
                    if request.query_id not in query_groups:
                        query_groups[request.query_id] = []
                    query_groups[request.query_id].append(i)

                # Process each query group
                for query_id, indices in query_groups.items():
                    index, texts = self._get_or_create_index(query_id)
                    index_name = self.active_indices[query_id]
                    logger.debug(f"ingest in vector_db: {query_id} {index_name}")

                    group_data = []
                    for i in indices:
                        request_data = batch_data[i]
                        group_data.extend(request_data)

                    logger.debug(f"len of group_data: {len(group_data)}")

                    embeddings_to_add: List[np.ndarray] = []
                    texts_to_add: List[str] = []

                    for embedding, text in group_data:
                        arr = np.asarray(embedding, dtype=np.float32)
                        if arr.ndim > 1:
                            arr = arr.reshape(-1)
                        if arr.shape[0] != self.vector_dim:
                            raise ValueError(
                                f"Embedding dimension {arr.shape[0]} does not match expected {self.vector_dim}"
                            )
                        embeddings_to_add.append(arr)
                        texts_to_add.append(text)

                    if not embeddings_to_add:
                        continue

                    vectors = np.stack(embeddings_to_add)
                    faiss.normalize_L2(vectors)

                    begin = time.time()
                    index.add(vectors)
                    texts.extend(texts_to_add)
                    end = time.time()

                    logger.debug(f"insert time: {end - begin}")
                    logger.debug(
                        f"Index {index_name} contains {index.ntotal} records after ingest"
                    )

                    # Update results and clean up
                    for i in indices:
                        request = batch_requests[i]
                        result_ref = ray.put(True)

                        if self.scheduler_ref is not None:
                            await self.scheduler_ref.on_result.remote(
                                request.request_id, request.query_id, result_ref
                            )

                        if request.query_id in self.query_requests:
                            self.query_requests[request.query_id].remove(request)

            except Exception as e:
                # traceback
                import traceback

                traceback.print_exc()
                logger.error(f"Error in insert processing: {e}")
                continue

    async def _process_searches(self):
        """Process search requests"""
        while self.running:
            try:
                try:
                    request = await asyncio.wait_for(
                        self.search_queue.get(), timeout=0.02
                    )
                except asyncio.TimeoutError:
                    continue

                query_vectors, top_k = request.data
                index_entry = self.index_store.get(request.query_id)

                if index_entry is None or index_entry["index"].ntotal == 0:
                    logger.debug(
                        f"search in vector_db: {request.request_id} {request.query_id} has no index"
                    )
                    result_ref = ray.put([])
                    if self.scheduler_ref is not None:
                        await self.scheduler_ref.on_result.remote(
                            request.request_id, request.query_id, result_ref
                        )
                    if request.query_id in self.query_requests:
                        self.query_requests[request.query_id].remove(request)
                    continue

                index = index_entry["index"]
                texts = index_entry["texts"]
                index_name = self.active_indices.get(request.query_id, "")

                logger.debug(
                    f"search in vector_db: {request.request_id} {request.query_id} {index_name}"
                )

                # Normalize and batch query vectors
                if isinstance(query_vectors, np.ndarray):
                    if query_vectors.ndim == 1:
                        query_vectors = query_vectors[None, :]
                    elif query_vectors.ndim > 2:
                        query_vectors = query_vectors.reshape(
                            query_vectors.shape[0], -1
                        )
                elif not isinstance(query_vectors, list):
                    query_vectors = [query_vectors]

                query_matrix = np.asarray(query_vectors, dtype=np.float32)
                if query_matrix.ndim == 1:
                    query_matrix = query_matrix[None, :]

                if query_matrix.shape[1] != self.vector_dim:
                    raise ValueError(
                        f"Query vector dimension {query_matrix.shape[1]} does not match expected {self.vector_dim}"
                    )

                faiss.normalize_L2(query_matrix)

                begin = time.time()
                distances, indices = index.search(
                    query_matrix, min(top_k, index.ntotal)
                )
                end = time.time()
                logger.debug(
                    f"search time for {len(query_matrix)} vectors: {end - begin}"
                )

                all_results: List[List[Dict[str, Union[str, float]]]] = []
                for dist_row, idx_row in zip(distances, indices):
                    search_results: List[Dict[str, Union[str, float]]] = []
                    for score, idx in zip(dist_row, idx_row):
                        if idx == -1:
                            continue
                        search_results.append(
                            {"text": texts[idx], "score": float(score)}
                        )
                    all_results.append(search_results)

                # If there is only one query vector, return a single result list instead of a nested list
                if len(all_results) == 1:
                    search_results = all_results[0][:top_k]
                else:
                    search_results = [results[:top_k] for results in all_results]

                logger.debug(f"search results: {search_results}")

                # Update results and clean up
                result_ref = ray.put(search_results)

                if self.scheduler_ref is not None:
                    await self.scheduler_ref.on_result.remote(
                        request.request_id, request.query_id, result_ref
                    )

                if request.query_id in self.query_requests:
                    self.query_requests[request.query_id].remove(request)

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in search processing: {e}")
                continue

    async def cleanup_query(self, query_id: str):
        """Clean up resources for a query"""
        if query_id in self.index_store:
            del self.index_store[query_id]
        if query_id in self.active_indices:
            del self.active_indices[query_id]

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

        # Clean up all tables created by queries
        try:
            for query_id in list(self.active_indices.keys()):
                await self.cleanup_query(query_id)
            print("Successfully cleaned up all indices")
        except Exception as e:
            print(f"Error cleaning up indices: {e}")
