import os
from dataclasses import dataclass
import logging
import google.generativeai as genai

from datetime import datetime, timedelta, timezone
import uuid
from typing import List, Dict, Optional, Any

from Faiss_vecDB import FaissSearchEngine, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration settings for the memory system"""
    similarity_threshold: float = 0.7
    consolidation_ratio: int = 3
    consolidation_intervals: List[timedelta] = None
    access_count_threshold: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    model_name: str = "BAAI/bge-small-en-v1.5"

    def __post_init__(self):
        if self.consolidation_intervals is None:
            self.consolidation_intervals = [
                timedelta(days=1),    # First check
                timedelta(days=7),    # Second check
                timedelta(days=30),   # Third check
                timedelta(days=90),   # Final check
            ]

class MemorySummarizer:
    """Handles memory consolidation using Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini model for summarization"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",  # flash is preferred 
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192
            }
        )
        
    async def summarize_memories(
        self,
        memories: List[Dict[str, Any]],
        target_number: int
    ) -> List[Dict[str, str]]:
        """
        Summarize a group of memories into a smaller set.
        
        Args:
            memories: List of memory dictionaries.
            target_number: Number of desired summary memories.
            
        Returns:
            List of consolidated memory dictionaries.
        """
        try:
            # Prepare the memories for summarization.
            memory_texts = [m["payload"]["content"] for m in memories]
            memory_ids = [m["id"] for m in memories]
            
            # Create the prompt.
            prompt = f"""Summarize these memories and create {target_number} distinct memories 
while preserving key information and temporal context:

{'\\n'.join(memory_texts)}"""
            
            # Generate summaries.
            response = await self.model.generate_content_async(prompt)
            summaries = response.text.split('\n\n')  # Assuming summaries are separated by blank lines
            
            # Create consolidated memory records.
            consolidated = []
            for summary in summaries[:target_number]:
                consolidated.append({
                    "content": summary.strip(),
                    "original_memories": memory_ids
                })
                
            return consolidated
            
        except Exception as e:
            logger.error(f"Error in memory summarization: {str(e)}")
            raise
        
class MemoryStore:
    """Main memory management system using FAISS vector storage"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory store with active and permanent memory stores"""
        self.config = config or MemoryConfig()
        
        # Initialize active and permanent memory stores.
        self.active_memories = FaissSearchEngine(
            model_name=self.config.model_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.permanent_memories = FaissSearchEngine(
            model_name=self.config.model_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

    async def add_active_memory(self, content: str, metadata: dict = None) -> str:
        """
        Add a new memory to both active and permanent storage.
        The permanent store retains the original, untouched memory.
        """
        try:
            memory_id = str(uuid.uuid4())
            default_metadata = {
                "id": memory_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": 0
            }
            if metadata:
                default_metadata.update(metadata)
            
            # Add to active memory store.
            self.active_memories.add_texts(
                texts=content,
                metadata=default_metadata,
                ids=[memory_id]
            )
            # Also add to permanent memory store (untouched copy).
            self.permanent_memories.add_texts(
                texts=content,
                metadata=default_metadata,
                ids=[memory_id]
            )
            return memory_id

        except Exception as e:
            logger.error(f"Error adding active memory: {str(e)}")
            raise

    async def consolidate_in_active(
        self,
        consolidated_memories: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Consolidate low-access memories in the active store by replacing them 
        with a summarized version while leaving the permanent store untouched.
        
        Args:
            consolidated_memories: List of consolidated memory dictionaries.
            
        Returns:
            List of new active memory IDs for the consolidated summaries.
        """
        try:
            new_active_ids = []
            
            for memory in consolidated_memories:
                new_memory_id = str(uuid.uuid4())
                new_active_ids.append(new_memory_id)
                consolidated_metadata = {
                    "id": new_memory_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "last_accessed": datetime.now(timezone.utc).isoformat(),
                    "access_count": 0,
                    "original_memories": memory["original_memories"],
                    "consolidation_date": datetime.now(timezone.utc).isoformat(),
                    "consolidated": True  # Flag indicating this is a summary memory.
                }
                self.active_memories.add_texts(
                    texts=memory["content"],
                    metadata=consolidated_metadata,
                    ids=[new_memory_id]
                )
            
            # Remove the original low-access memories from active storage.
            # They remain untouched in the permanent store.
            original_ids = []
            for memory in consolidated_memories:
                original_ids.extend(memory["original_memories"])
                
            self.active_memories.delete_by_ids(original_ids)
            
            return new_active_ids
            
        except Exception as e:
            logger.error(f"Error consolidating memories in active store: {str(e)}")
            raise
            
    async def retrieve_memories(self, query: str, limit: int = 10):
        active_results = self.active_memories.search(query, top_k=limit)
        permanent_results = self.permanent_memories.search(query, top_k=limit)

        # Update the retrieved metadata in-place and build the return data.
        active_formatted = []
        for r in active_results:
            if isinstance(r.metadata, dict):
                metadata = r.metadata
                # Increment access_count.
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()

                rid = metadata.get("id", None)
                if rid:
                    self.active_memories.vector_store.update_metadata(rid, metadata)

            active_formatted.append({
                "id": r.metadata.get("id", "NO_ID_IN_METADATA"),
                "payload": {
                    "content": r.content,
                    **r.metadata
                },
                "score": r.score
            })

        permanent_formatted = []
        for r in permanent_results:
            if isinstance(r.metadata, dict):
                metadata = r.metadata
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                rid = metadata.get("id", None)
                if rid:
                    self.permanent_memories.vector_store.update_metadata(rid, metadata)

            permanent_formatted.append({
                "id": r.metadata.get("id", "NO_ID_IN_METADATA"),
                "payload": {
                    "content": r.content,
                    **r.metadata
                },
                "score": r.score
            })

        return active_formatted, permanent_formatted

    def get_all_active_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all memories from active storage.
        """
        try:
            all_chunks = self.active_memories.vector_store.chunks
            formatted_memories = []

            for chunk in all_chunks:
                content = chunk["content"]
                metadata = chunk["metadata"] or {}

                memory_id = metadata.get("id", "NO_ID_FOUND")

                memory_dict = {
                    "id": memory_id,
                    "payload": {
                        "content": content,
                        "timestamp": metadata.get("timestamp", ""),
                        "access_count": metadata.get("access_count", 0),
                        "last_accessed": metadata.get("last_accessed", "")
                    }
                }
                formatted_memories.append(memory_dict)

            return formatted_memories

        except Exception as e:
            logger.error(f"Error retrieving active memories: {str(e)}")
            raise

    async def check_forgetting_curve(self):
        """
        Check memories against forgetting curve criteria and trigger consolidation 
        in the active store. Low-access memories are summarized and replaced in active.
        The permanent store remains a complete archive.
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get all active memories.
            active_memories = self.get_all_active_memories()
            to_consolidate = []
            
            # Identify memories eligible for consolidation based on each interval.
            for interval in self.config.consolidation_intervals:
                threshold_time = current_time - interval
                interval_memories = [
                    m for m in active_memories
                    if (
                        datetime.fromisoformat(m["payload"].get("last_accessed", m["payload"]["timestamp"])) < threshold_time
                        and m["payload"].get("access_count", 0) < self.config.access_count_threshold
                        and m not in to_consolidate
                    )
                ]
                to_consolidate.extend(interval_memories)
            
            if to_consolidate:
                target_number = max(1, len(to_consolidate) // self.config.consolidation_ratio)
                
                if not hasattr(self, 'summarizer'):
                    api_key = os.getenv('GEMINI_API_KEY')
                    if not api_key:
                        raise ValueError("GEMINI_API_KEY environment variable not set")
                    self.summarizer = MemorySummarizer(api_key)
                
                # Generate consolidated summaries.
                consolidated = await self.summarizer.summarize_memories(
                    to_consolidate,
                    target_number
                )
                
                # Replace low-access memories in active store with the consolidated summaries.
                await self.consolidate_in_active(consolidated)
                
        except Exception as e:
            logger.error(f"Error checking forgetting curve: {str(e)}")
            raise
        
    def save(self, directory: str):
        """Save both memory stores to disk."""
        try:
            os.makedirs(directory, exist_ok=True)
            
            active_dir = os.path.join(directory, "active_memories")
            self.active_memories.vector_store.save(active_dir)
            
            permanent_dir = os.path.join(directory, "permanent_memories")
            self.permanent_memories.vector_store.save(permanent_dir)
            
        except Exception as e:
            logger.error(f"Error saving memory store: {str(e)}")
            raise
            
    def load(self, directory: str):
        """Load both memory stores from disk."""
        try:
            active_dir = os.path.join(directory, "active_memories")
            if os.path.exists(active_dir):
                self.active_memories.vector_store.load(active_dir)
                
            permanent_dir = os.path.join(directory, "permanent_memories")
            if os.path.exists(permanent_dir):
                self.permanent_memories.vector_store.load(permanent_dir)
                
        except Exception as e:
            logger.error(f"Error loading memory store: {str(e)}")
            raise
            
async def test_memory_system():
    """Test the memory system functionality."""
    try:
        config = MemoryConfig(
            similarity_threshold=0.7,
            consolidation_ratio=3,
            access_count_threshold=5,
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        store = MemoryStore(config=config)
        
        # Add test memories.
        memory_id1 = await store.add_active_memory(
            "The project meeting on Monday discussed the new feature implementation timeline."
        )
        memory_id2 = await store.add_active_memory(
            "Team feedback suggests we need to improve the user interface design."
        )
        memory_id3 = await store.add_active_memory(
            "Testing revealed some performance issues in the database queries."
        )
        
        # Test memory retrieval.
        print("\nTesting memory retrieval...")
        active, permanent = await store.retrieve_memories("project meeting")
        print(f"Found {len(active)} active and {len(permanent)} permanent memories")
        
        # Test memory consolidation.
        print("\nTesting memory consolidation...")
        await store.check_forgetting_curve()
        
        # Save the memory store.
        print("\nSaving memory store...")
        store.save("./memory_store")
        
        # Load the memory store.
        print("\nLoading memory store...")
        new_store = MemoryStore(config=config)
        new_store.load("./memory_store")
        
        active, permanent = await new_store.retrieve_memories("project meeting")
        print(f"After loading: Found {len(active)} active and {len(permanent)} permanent memories")
        
        return "Memory system test completed successfully"
        
    except Exception as e:
        logger.error(f"Error in test_memory_system: {str(e)}")
        raise

async def main():
    """Main function to run the memory system."""
    try:
        result = await test_memory_system()
        print(f"\nResult: {result}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
