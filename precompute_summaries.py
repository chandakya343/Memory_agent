import os
import json
import random
from datetime import datetime, timedelta, timezone
import asyncio
import uuid

from memory_bioinspired import MemoryStore, MemoryConfig

# Set your Gemini API key (or however you normally do it)
os.environ["GEMINI_API_KEY"] = 'Gemini_API_Key'
NUM_MEMORIES = 30
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
MAX_DAYS = 60

async def precompute_summaries():
    # For simulation, lower the access count threshold so consolidation is more likely.
    config = MemoryConfig(
        similarity_threshold=0.7,
        consolidation_ratio=3,
        access_count_threshold=2,  # Lowered threshold for simulation visibility.
        model_name="BAAI/bge-small-en-v1.5"
    )
    store = MemoryStore(config=config)

    # Generate random memory contents.
    possible_contents = [
        "Went to the grocery store for milk and eggs",
        "Project meeting discussing feature X and timeline",
        "Had a doctor appointment about knee pain",
        "Vacation planning: looking at flights to Hawaii",
        "Learned about advanced Faiss indexing",
        "Discussed marketing strategy for product launch",
        "Had lunch with an old friend",
        "Attended a local meetup on AI and ML",
        "Debugged performance issues in the database",
        "Brainstormed new UI design ideas with the team",
        "Collected user feedback from alpha testers",
        "Read an interesting paper on quantum computing",
        "Cooked a new recipe with interesting spices",
        "Watched a tutorial on advanced concurrency patterns",
        "Organized a surprise birthday party",
        "Refactored code for better readability",
        "Investigated memory leaks in the program",
        "Call with partner to discuss personal matters",
        "Studied some mathematics for data analysis",
        "Brainstormed new features for next quarter",
        "Met with the design team to finalize specs",
        "Went for a 5-mile run around the park",
        "Tested an experimental feature in the app",
        "Team offsite planning for next month",
        "Took the cat to the vet for a checkup",
        "Explored advanced NLP pipeline improvements",
        "Reviewed a PR with significant refactoring",
        "Helped a colleague debug complex merge conflicts",
        "Learned more about knowledge graphs",
        "Analyzed user engagement metrics from last release"
    ]
    random.shuffle(possible_contents)

    # Prepare pending memories with a scheduled creation time.
    pending_memories = []
    for i in range(NUM_MEMORIES):
        created_day_offset = random.randint(0, MAX_DAYS)
        created_time = START_DATE + timedelta(days=created_day_offset)
        content = possible_contents[i % len(possible_contents)]
        memory_id = str(uuid.uuid4())
        pending_memories.append({
            "id": memory_id,
            "created_time": created_time,
            "content": content,
            "added": False  # flag to track whether this memory has been added
        })

    # Sort pending memories by creation time.
    pending_memories.sort(key=lambda m: m["created_time"])

    simulation_snapshots = {}

    # Simulate day-by-day.
    for current_day in range(MAX_DAYS + 1):
        current_datetime = START_DATE + timedelta(days=current_day)

        # Add new memories that are due on this day.
        for memory in pending_memories:
            if not memory["added"] and memory["created_time"] <= current_datetime:
                await store.add_active_memory(
                    memory["content"],
                    metadata={
                        "timestamp": memory["created_time"].isoformat(),
                        "last_accessed": memory["created_time"].isoformat(),
                        "created_time": memory["created_time"].isoformat()
                    }
                )
                memory["added"] = True

        # For memories already added, simulate random retrievals to increase access counts.
        # Here, we deliberately use a lower increment to keep many memories under threshold.
        for memory in pending_memories:
            if memory["added"] and memory["created_time"] <= current_datetime:
                increment_count = random.randint(0, 1)
                for _ in range(increment_count):
                    _ = await store.retrieve_memories(memory["content"], limit=1)

        # Trigger consolidation on specific days.
        if current_day in (7, 30, 60):
            await store.check_forgetting_curve()

        # Record snapshot of active and permanent memory states.
        active_memories_list = store.get_all_active_memories()
        permanent_chunks = store.permanent_memories.vector_store.chunks
        permanent_list = []
        for ch in permanent_chunks:
            mid = ch.get("metadata", {}).get("id", str(uuid.uuid4()))
            permanent_list.append({
                "id": mid,
                "payload": {
                    "content": ch.get("content", ""),
                    "access_count": ch.get("metadata", {}).get("access_count", 0),
                    "timestamp": ch.get("metadata", {}).get("consolidation_date", ch.get("metadata", {}).get("timestamp", ""))
                }
            })

        simulation_snapshots[current_day] = {
            "day": current_day,
            "active": active_memories_list,
            "permanent": permanent_list
        }

    # Save the day-by-day simulation to JSON.
    with open("simulation_data.json", "w", encoding="utf-8") as f:
        json.dump(simulation_snapshots, f, indent=2)

    print("Precomputation complete! Saved day-by-day simulation to simulation_data.json")


if __name__ == "__main__":
    asyncio.run(precompute_summaries())
