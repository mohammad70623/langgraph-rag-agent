# LangGraph RAG Agent

A conversational chatbot that can answer questions from your own PDF documents. Ask it anything - if the answer is in your files, it'll find it.

---

## How it works

The bot uses two main components working together:

**Retrieval (RAG):** Your PDFs are chunked, embedded, and stored in a local FAISS vector database. When you ask a question, the most relevant chunks are pulled out and handed to the LLM as context.

**LangGraph agent:** The LLM (Llama 3.3 70B via Groq) decides on its own whether a question needs to look something up from the documents or can be answered directly. This decision happens through a tool-calling loop — the graph routes between the chat node and the retrieval tool as needed.

```
You ask a question
       ↓
   LLM thinks
       ↓
   Needs docs?
   ├── Yes → retrieve from FAISS → back to LLM → answer
   └── No  → answer directly
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/mohammad70623/langgraph-rag-agent.git
cd langgraph-rag-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
You can get a free key at [console.groq.com](https://console.groq.com).

**4. Add your PDFs**

Drop any PDF files into the `data/` folder.

**5. Build the vector database**
```bash
python create_vectorstore.py
```
This only needs to run once (or again whenever you add new PDFs).

**6. Start chatting**
```bash
python main.py
```

Type `exit` to quit.

---

## Project structure

```
├── data/                    # Put your PDF files here
├── vectorstore/             # Auto-generated after running create_vectorstore.py
├── create_vectorstore.py    # Loads PDFs, chunks them, builds FAISS index
├── graph.py                 # LangGraph graph + RAG tool definition
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) - agent graph
- [LangChain](https://github.com/langchain-ai/langchain) - document loading, splitting, QA chain
- [Groq](https://groq.com) - LLM inference (Llama 3.3 70B)
- [FAISS](https://github.com/facebookresearch/faiss) - local vector store
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - embeddings
