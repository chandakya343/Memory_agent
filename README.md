# Memory Agent

## 1. The Problem: Information Overload in Chatbot Memory
As chatbot contexts grow, the system is overwhelmed by an ever-increasing volume of information. In traditional memory systems, this leads to inefficiencies because too much historical data is retained—even when much of it becomes irrelevant. My project addresses this challenge by introducing a mechanism to “forget” less-accessed memories while preserving the essential context, much like the human brain.

## 2. Biological Inspiration: Emulating Human Memory
Drawing parallels from the human cognitive system, I divided memory into two key components:

#### Short-Term (Active) Memory:
- Like our working memory, this stores recent or frequently accessed information, enabling quick retrieval.
#### Long-Term (Permanent) Memory: 
- Analogous to our long-term storage, this keeps a full archive of experiences, even as some details in active memory are consolidated or forgotten.
My system mimics the natural forgetting curve—where less-relevant information is gradually pruned—while still maintaining a complete record in the permanent store. This agentic aspect, where the system actively decides which memories to retain in full and which to summarize, is a standout feature of my design.

## 3. Major Components and Architecture
 - a. Custom FAISS-Based Vector Database
From Scratch Implementation: I built my own vector database using FAISS, deliberately avoiding any reliance on higher-level abstractions like Langchain. This not only demonstrates a deep understanding of vector search mechanics but also highlights my ability to work with low-level tools.
Text Chunking and Embedding: The system splits texts into manageable chunks (using custom sentence-splitting logic) and creates overlapping segments. These chunks are then embedded using a SentenceTransformer model. This ensures that even long documents are efficiently indexed for similarity searches.
Indexing and Metadata Management: Each chunk is added to a FAISS index (using the IndexFlatIP for inner product search), and custom mappings are maintained to allow for metadata updates and deletion. This meticulous handling ensures that when memories are consolidated or removed, the vector store remains consistent and efficient.
(See details in Faiss_vecDB.py Faiss_vecDB)

- b. Memory Storage and Consolidation
Active vs. Permanent Memory Stores: my system maintains two separate FAISS-backed stores. The active memory store holds the recent, frequently accessed memories, while the permanent memory store retains the original, unmodified copies.
Forgetting Curve and Consolidation: A novel consolidation mechanism identifies memories that have not been accessed frequently (using configurable thresholds and time intervals). These low-access memories are summarized—using a Gemini API–powered summarizer—and replaced in the active store with a consolidated version. This not only reduces memory clutter but also keeps the context relevant without losing historical data.
Agentic Behavior: The system autonomously monitors access patterns and triggers consolidation at predetermined intervals (e.g., after 7, 30, and 60 days in my simulation). This agentic decision-making is a key innovation, enabling the chatbot to dynamically manage its memory in a human-like fashion.
(The logic for memory consolidation and forgetting is implemented in memory_bioinspired.py memory_bioinspired, with simulation support shown in precompute_summaries.py precompute_summaries)

## 4. Implementation Details
- Vector Database Construction:
I built the vector store entirely from scratch. Using FAISS’s IndexFlatIP, I manage embeddings for text chunks and implement deletion by recomputing embeddings from scratch—an approach that is both efficient and elegant.

- Text Chunking:
The TextChunker class intelligently splits text into sentences and groups them into chunks with a fixed size and overlap. This ensures no loss of context between adjacent chunks, vital for high-quality semantic retrieval.

### Memory Consolidation Mechanism:

- Access Counting & Temporal Analysis: Each memory is tagged with metadata such as timestamps and access counts. Periodic checks compare these against the forgetting curve criteria.
- Summarization via Gemini API: When low-access memories are identified, they are summarized to produce a concise version that still encapsulates the essential details. These summaries replace the originals in the active memory store while the full versions remain in permanent storage.
- Simulation of Memory Dynamics: The simulation script (precompute_summaries.py) demonstrates how memories are added over time, accessed randomly to simulate usage, and then consolidated based on the forgetting curve. This not only validates the design but also offers a clear window into how the system dynamically manages its memory over days.

## 5. Novelty and Agentic Aspects
- Ground-Up Development:
Every component—from the vector database to the forgetting mechanism—was implemented without using pre-built boilerplate libraries. This showcases my deep technical expertise and innovative thinking.

### Agentic Memory Management:
The system doesn’t just passively store data; it actively decides which memories are essential for immediate retrieval and which can be summarized. This mimics human cognitive processes, making my project stand out as both novel and biologically inspired.

### Efficient and Dynamic Memory Handling:
By differentiating between active and permanent memories and using a consolidation mechanism, my system is designed to handle increasing amounts of data gracefully, ensuring that only the most relevant information occupies the short-term memory.

## Conclusion
my memory agent project tackles a critical challenge in the design of long-context chatbots by drawing inspiration from human memory systems. With a custom-built FAISS vector database, sophisticated text chunking, and an innovative forgetting mechanism based on access patterns and temporal decay, I have created a system that dynamically manages and consolidates information. This not only reduces cognitive overload but also maintains a complete historical archive for reference, making my work both practically useful and conceptually novel.

This project stands as a testament to my ability to develop advanced, agentic memory systems from the ground up, showcasing originality, technical depth, and a keen insight into human-inspired design.
