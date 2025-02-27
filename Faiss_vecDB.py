from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
import uuid

@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Optional[Dict] = None

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: Union[str, List[str]], metadata: Optional[Dict] = None) -> List[Dict]:
        if isinstance(text, str):
            texts = [text]
            metadatas = [metadata] if metadata else [None]
        else:
            texts = text
            metadatas = [metadata] * len(texts) if metadata else [None] * len(texts)

        all_chunks = []
        for text, meta in zip(texts, metadatas):
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) <= self.chunk_size:
                all_chunks.append({
                    "content": text,
                    "metadata": meta
                })
                continue

            sentences = self.split_into_sentences(text)
            current_chunk = []
            current_length = 0
            for sentence in sentences:
                sentence_len = len(sentence)
                if current_length + sentence_len > self.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    all_chunks.append({
                        "content": chunk_text,
                        "metadata": meta
                    })
                    if self.chunk_overlap > 0:
                        last_sentence = current_chunk[-1]
                        if len(last_sentence) < self.chunk_overlap:
                            current_chunk = [last_sentence]
                            current_length = len(last_sentence)
                        else:
                            current_chunk = []
                            current_length = 0
                    else:
                        current_chunk = []
                        current_length = 0
                current_chunk.append(sentence)
                current_length += sentence_len
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": meta
                })
        return all_chunks

class VectorStore:
    def __init__(self, embedding_dim: int = 384, embedding_fn=None):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.id_to_index = {}  # Map memory IDs to FAISS indices
        self.index_to_id = {}
        self.embedding_fn = embedding_fn  # Function to recompute embeddings

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], ids: List[str]):
        n_vectors = len(embeddings)
        start_idx = len(self.chunks)
        for i, id_ in enumerate(ids):
            idx = start_idx + i
            self.id_to_index[id_] = idx
            self.index_to_id[idx] = id_
        # For simplicity, always use IndexFlatIP
        if self.index is None or not isinstance(self.index, faiss.IndexFlatIP):
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def delete_by_ids(self, ids: List[str]):
        # Identify indices to remove based on current mappings.
        indices_to_remove = [self.id_to_index[id_] for id_ in ids if id_ in self.id_to_index]
        if not indices_to_remove:
            return
        # Determine remaining indices.
        remaining_indices = sorted(set(range(len(self.chunks))) - set(indices_to_remove))
        remaining_chunks = [self.chunks[i] for i in remaining_indices]
        # Recompute embeddings for remaining chunks using the provided embedding function.
        if self.embedding_fn is None:
            raise ValueError("Embedding function not provided; cannot recompute embeddings.")
        texts = [chunk["content"] for chunk in remaining_chunks]
        new_embeddings = self.embedding_fn(texts, normalize_embeddings=True)
        new_embeddings = np.array(new_embeddings).astype('float32')
        # Rebuild the index as a new IndexFlatIP.
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_index.add(new_embeddings)
        self.index = new_index
        self.chunks = remaining_chunks
        # Rebuild ID mappings.
        self.id_to_index = {}
        self.index_to_id = {}
        for new_idx, chunk in enumerate(remaining_chunks):
            mem_id = chunk["metadata"].get("id")
            if mem_id is None:
                mem_id = str(uuid.uuid4())
            self.id_to_index[mem_id] = new_idx
            self.index_to_id[new_idx] = mem_id

    def update_metadata(self, memory_id: str, new_metadata: Dict):
        if memory_id not in self.id_to_index:
            print(f"Warning: No chunk found for memory_id={memory_id}")
            return
        idx = self.id_to_index[memory_id]
        self.chunks[idx]["metadata"] = new_metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 5, nprobe: int = 10) -> List[SearchResult]:
        if self.index is None:
            return []
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k, len(self.chunks)))
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                content=chunk["content"],
                score=float(score),
                metadata=chunk["metadata"]
            ))
        return results

    def save(self, path: str):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str):
        load_path = Path(path)
        self.index = faiss.read_index(str(load_path / "faiss.index"))
        with open(load_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

class FaissSearchEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", chunk_size: int = 500, chunk_overlap: int = 100):
        self.model = SentenceTransformer(model_name)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        # Pass the model's encode function to allow recomputation during deletion.
        self.vector_store = VectorStore(self.model.get_sentence_embedding_dimension(), embedding_fn=self.model.encode)

    def add_texts(self, texts: Union[str, List[str]], metadata: Optional[Union[Dict, List[Dict]]] = None, ids: Optional[Union[str, List[str]]] = None):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(metadata, dict) or metadata is None:
            metadata_list = [metadata] * len(texts)
        else:
            if len(metadata) != len(texts):
                raise ValueError("If you pass a list of metadata, it must be same length as texts.")
            metadata_list = metadata
        if isinstance(ids, str):
            ids = [ids]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        else:
            if len(ids) != len(texts):
                raise ValueError("If you pass a list of ids, it must be same length as texts.")
        all_chunks = []
        all_ids = []
        for text_, meta_, id_ in zip(texts, metadata_list, ids):
            # Ensure the memory id is in metadata.
            if meta_ is None:
                meta_ = {"id": id_}
            else:
                meta_.setdefault("id", id_)
            new_chunks = self.chunker.create_chunks(text_, meta_)
            for _ in new_chunks:
                all_ids.append(id_)
            all_chunks.extend(new_chunks)
        embeddings = []
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts_batch = [chunk["content"] for chunk in batch]
            batch_embeddings = self.model.encode(texts_batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        embeddings_array = np.array(embeddings).astype('float32')
        self.vector_store.add_embeddings(embeddings_array, all_chunks, ids=all_ids)

    def delete_by_ids(self, ids: List[str]):
        return self.vector_store.delete_by_ids(ids)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.model.encode(query, normalize_embeddings=True).astype('float32')
        return self.vector_store.search(query_embedding, top_k)

    def save(self, path: str):
        self.vector_store.save(path)

    def load(self, path: str):
        self.vector_store.load(path)
