# 🤖 AgentFlow — AI Chatbot with LangGraph, RAG & Web Search

**AgentFlow** is a conversational AI chatbot built with **LangChain, LangGraph, Streamlit, SQLite, RAG, Web Search and LangSmith**.

The application supports normal conversations, persistent chat history, conversation resume, PDF-based question answering, web search for current information, streaming responses, and LangGraph-based tool orchestration.

The project is designed to demonstrate how modern **LLM applications and agentic workflows** can be built using LangGraph with persistence, tools, RAG and observability.

---

## 🚀 Features

* 💬 **Conversational AI Chatbot**
* 🧠 **LangChain + LangGraph**
* 🔄 **Streaming responses**
* 💾 **Persistent conversations using SQLite**
* 🔁 **Resume previous conversations**
* 🗂️ **Multiple conversation threads**
* 🏷️ **Automatic conversation naming**
* 📄 **PDF RAG**
* 🔎 **FAISS vector search**
* 🌐 **Web search using Serper**
* 🛠️ **Tools integrated with LangGraph**
* 📊 **LangSmith integration for tracing and observability**
* 🖥️ **Streamlit user interface**
* 🔐 **Environment variable based API key configuration**

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │        User          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Streamlit Frontend  │
                         │      frontend.py     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LangGraph       │
                         │    StateGraph Flow   │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
              Normal Query     PDF Available   Current Query
                     │              │              │
                     ▼              ▼              ▼
                Simple LLM      FAISS RAG      Tool-enabled LLM
                                    │              │
                                    │              ▼
                                    │        ┌─────────────┐
                                    │        │ Web Search  │
                                    │        │   (Serper)  │
                                    │        └──────┬──────┘
                                    │               │
                                    └───────┬───────┘
                                            ▼
                                  ┌──────────────────┐
                                  │   Final Answer   │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │ SQLite Checkpoint│
                                  │    Persistence   │
                                  └──────────────────┘
```

---

# 🔄 Complete Workflow

AgentFlow uses **LangGraph** to control the conversation workflow.

When a user sends a message, the application determines what type of processing is required.

### 1. User sends a message

The Streamlit frontend sends the user's message to the LangGraph backend.

```text
User
 ↓
Streamlit
 ↓
LangGraph
```

---

### 2. Normal Query

For a normal question such as:

> What is machine learning?

The request goes directly to the LLM.

```text
User Query
    ↓
LangGraph
    ↓
Simple LLM
    ↓
Response
```

No web search or RAG is required.

---

### 3. PDF Question Answering

When a PDF is uploaded, AgentFlow creates a RAG pipeline.

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Document Text
    ↓
Text Splitting
    ↓
RecursiveCharacterTextSplitter
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
```

When the user asks a question:

```text
User Question
      ↓
FAISS Retriever
      ↓
Relevant PDF Chunks
      ↓
LLM + Retrieved Context
      ↓
Final Answer
```

This allows the chatbot to answer questions based on the uploaded document rather than relying only on the LLM's internal knowledge.

---

# 🌐 Web Search Workflow

For queries requiring current information, AgentFlow can use a web-search tool.

Example:

> What is the latest news about India?

The workflow becomes:

```text
User Query
    ↓
LangGraph
    ↓
Tool-enabled LLM
    ↓
Web Search Tool
    ↓
Serper API
    ↓
Search Results
    ↓
Tool Result
    ↓
LangGraph
    ↓
Simple LLM
    ↓
Final Answer
```

The important part is that the search tool is not manually called from the frontend.

**LangGraph manages the tool execution and conversation flow.**

---

# 🧩 LangGraph

LangGraph is used as the main orchestration layer of AgentFlow.

The graph contains:

```text
START
  ↓
chat_node
  ↓
Tool Decision
  ↓
 ┌───────────────┐
 │               │
 ▼               ▼
No Tool        Tool Call
 │               │
 ▼               ▼
Final Answer   ToolNode
                 │
                 ▼
              chat_node
                 │
                 ▼
            Final Answer
```

This makes the application stateful and allows different steps such as:

* LLM calls
* Tool execution
* RAG processing
* Persistence
* Conversation continuation

to be managed through a graph-based workflow.

---

# 🛠️ Tools in LangGraph

AgentFlow demonstrates how external tools can be connected to LangGraph.

Currently, the primary external tool is:

### Web Search Tool

```text
LangGraph
    ↓
Tool-enabled LLM
    ↓
web_search
    ↓
Serper API
    ↓
Search Results
```

The tool is connected to the model using LangChain's tool binding mechanism.

LangGraph's `ToolNode` then handles the tool execution inside the graph.

---

# 📄 RAG using LangGraph

AgentFlow also demonstrates document-based Retrieval-Augmented Generation.

### Document ingestion

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Chunks
 ↓
HuggingFace Embeddings
 ↓
FAISS
```

### Query processing

```text
Question
 ↓
Retriever
 ↓
Relevant Chunks
 ↓
Prompt + Context
 ↓
LLM
 ↓
Answer
```

The retrieved document context is injected into the LLM prompt before generating the final response.

---

# 💾 Persistence with SQLite

AgentFlow uses LangGraph's SQLite checkpointing mechanism to persist conversation state.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
```

The graph uses SQLite as its checkpoint database.

```text
Conversation
     ↓
LangGraph State
     ↓
SQLite Checkpoint
     ↓
Stored Thread
```

This allows the application to maintain conversation state even after the current interaction.

---

# 🔁 Resume Chat

Every conversation is associated with a unique **thread ID**.

```text
Thread ID
   ↓
SQLite
   ↓
Conversation State
   ↓
Resume Previous Chat
```

When a previous conversation is selected, the corresponding thread is loaded from SQLite.

This allows users to:

* Create multiple conversations
* Switch between conversations
* Continue previous conversations
* Preserve message history

---

# 🗂️ Conversation Management

AgentFlow maintains separate conversation threads.

Example:

```text
Conversations
│
├── Machine Learning Discussion
├── Python Interview Preparation
├── PDF Research
└── Current Affairs
```

Each conversation has its own thread ID and persisted state.

The application also generates a conversation title after the first user interaction.

---

# ⚡ Streaming

AgentFlow uses LangGraph/LangChain streaming to provide a better interactive experience.

Instead of waiting for the complete response:

```text
Request
   ↓
Wait
   ↓
Complete Response
```

the application streams the response progressively:

```text
Request
   ↓
First Token
   ↓
More Tokens
   ↓
More Tokens
   ↓
Complete Response
```

This makes the chatbot feel more responsive during longer LLM generations.

---

# 📊 LangSmith Integration

AgentFlow integrates **LangSmith** for observability and debugging.

LangSmith can be used to inspect:

* LLM calls
* LangGraph execution
* Tool calls
* Inputs and outputs
* Latency
* Errors
* Execution traces

Conceptually:

```text
User Request
     ↓
LangGraph
     ├── LLM
     ├── Retriever
     └── Web Search Tool
             ↓
        LangSmith Trace
```

This is useful for understanding how the complete agentic workflow is executing.

---

# 🧠 LangChain + LangGraph

The project demonstrates the different responsibilities of LangChain and LangGraph.

### LangChain

Used for components such as:

* Chat models
* Prompt/messages
* Embeddings
* Retrievers
* Document loaders
* Text splitters
* Tools

### LangGraph

Used for:

* Workflow orchestration
* Stateful execution
* Tool routing
* Persistence
* Conversation state
* Multi-step agentic workflows

In simple terms:

```text
LangChain → Building Blocks

LangGraph → Workflow / Orchestration
```

---

# 🛠️ Tech Stack

| Technology  | Purpose                      |
| ----------- | ---------------------------- |
| Python      | Core programming language    |
| Streamlit   | Frontend / UI                |
| LangChain   | LLM application framework    |
| LangGraph   | Agent workflow orchestration |
| SQLite      | Conversation persistence     |
| FAISS       | Vector similarity search     |
| HuggingFace | Embeddings / NLP models      |
| Serper      | Web search                   |
| LangSmith   | Observability & tracing      |
| Atria API   | LLM provider                 |
| PyPDF       | PDF document loading         |

---

# 📁 Project Structure

```text
AgentFlow/
│
├── App/
│   ├── frontend.py
│   ├── backend.py
│   └── style.css
│
├── conversation_naming.py
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── assets/
    ├── architecture.png
    └── agentflow-demo.gif
```

> `chatbot.db`, `.env`, virtual environments and Python cache files should not be committed to GitHub.

---

# ⚙️ Local Setup

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd AgentFlow
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 API Key Configuration

Create a `.env` file in the project root.

```env
ATRIA_API_KEY=your_atria_api_key
SERPER_API_KEY=your_serper_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=AgentFlow
```

### Required services

| Variable                   | Purpose            |
| -------------------------- | ------------------ |
| `ATRIA_API_KEY`            | LLM access         |
| `SERPER_API_KEY`           | Web search         |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace models |
| `LANGSMITH_API_KEY`        | LangSmith tracing  |

**Never commit your `.env` file or API keys to GitHub.**

---

# ▶️ Run the Application

From the project root:

```bash
streamlit run App/frontend.py
```

Then open:

```text
http://localhost:8501
```

---

# 💬 How to Use AgentFlow

## Normal Conversation

Enter a normal question:

```text
What is deep learning?
```

The chatbot processes it using the normal LLM path.

---

## 🌐 Current Information

Ask a question that requires current information:

```text
What is the latest news about India?
```

AgentFlow can route the request to the web-search tool.

---

## 📄 Ask Questions About a PDF

1. Upload a PDF from the sidebar.
2. Wait for the document to be indexed.
3. Ask questions about the document.

Example:

```text
Summarize this document.
```

or:

```text
What are the main findings mentioned in the document?
```

The application retrieves relevant chunks from FAISS and provides them to the LLM as context.

---

## 🔁 Resume Previous Conversations

Select an existing conversation from the sidebar.

The corresponding thread state is loaded from SQLite, allowing the conversation to continue from where it was left.

---

# 🔄 End-to-End Architecture

The complete AgentFlow architecture can be summarized as:

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ Streamlit UI    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   LangGraph     │
                  │  StateGraph     │
                  └────────┬────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
     Normal Query      PDF Query        Web Query
          │                │                 │
          ▼                ▼                 ▼
      Simple LLM        FAISS          Tool-enabled
                         Retriever          LLM
          │                │                 │
          │                ▼                 ▼
          │            PDF Context      Web Search
          │                │                 │
          └────────────────┼─────────────────┘
                           │
                           ▼
                    Final LLM Answer
                           │
                           ▼
                  Streaming Response
                           │
                           ▼
                 SQLite Checkpoint
                           │
                           ▼
                  Resume Conversation

                    ─────────────
                    LangSmith
                    Observability
                    ─────────────
```

---

# 🎯 Project Goals

The main goal of AgentFlow is to understand and implement the core components required for building modern LLM applications:

* LLM integration
* LangChain fundamentals
* LangGraph workflows
* Stateful conversations
* Persistence
* Streaming
* Tool calling
* Web search
* RAG
* Vector databases
* Conversation management
* Observability with LangSmith

---

# ⚠️ Current Limitations

* The application currently runs locally through Streamlit.
* Web search depends on the configured Serper API.
* LLM response speed depends on the selected model/API provider.
* PDF retrieval quality depends on document structure and chunking.
* API usage may be subject to provider quotas and rate limits.

---

# 🔮 Future Improvements

Possible future improvements include:

* More external tools
* MCP-based tool integration
* Better query routing
* Improved RAG retrieval
* Hybrid search
* Conversation metadata
* User authentication
* More advanced agent workflows
* Deployment to a cloud platform
* Improved evaluation and RAG benchmarking

---

# 🎥 Demo

AgentFlow is currently demonstrated locally using Streamlit.

A demo video showcasing:

* Normal conversation
* PDF RAG
* Web search
* Streaming
* Conversation history
* Resume chat
* LangGraph workflow

will be shared on LinkedIn.

---

# 👨‍💻 Author

**Rajesh Kumar**

B.Tech — Artificial Intelligence & Data Science

### Connect with me

**LinkedIn:** [Rajesh Kumar](https://www.linkedin.com/in/rajeshxdatascience/)

**GitHub:** [rajeshxdatascience](https://github.com/rajeshxdatascience)

---

# ⭐ If you find this project useful

Feel free to explore the repository, experiment with the workflow, and provide feedback.
