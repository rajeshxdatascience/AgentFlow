# 🤖 AgentFlow — AI Chatbot with LangGraph

AgentFlow is a conversational AI chatbot built with **LangChain, LangGraph and Streamlit**.

It supports normal conversations, web search, PDF RAG, streaming responses, persistent chat history, resume chat and LangSmith observability.

## ✨ Features

* 💬 Conversational AI with **Qwen3.8 Flash Free**
* 🧠 **LangChain + LangGraph** workflow
* ⚡ Streaming responses
* 🌐 Web search using **Serper**
* 📄 PDF RAG using **FAISS + HuggingFace Embeddings**
* 💾 Conversation persistence with **SQLite**
* 🔁 Resume previous conversations
* 🗂️ Multiple conversation threads
* 🛠️ Tool calling with LangGraph
* 📊 **LangSmith** tracing and observability
* 🖥️ Streamlit UI

---

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                      LangGraph
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          Normal Query   PDF RAG    Web Search
              │            │            │
              ▼            ▼            ▼
             LLM         FAISS      Serper Tool
              │            │            │
              └────────────┼────────────┘
                           ▼
                     Final Response
                           │
                           ▼
                    SQLite Checkpoint
                           │
                           ▼
                    Resume Conversation
```

**LangSmith** is integrated across the workflow for tracing and debugging.

---

## 🔄 How It Works

### Normal Chat

```text
User → LangGraph → Qwen3.8 Flash → Response
```

### Web Search

```text
User
 ↓
LangGraph
 ↓
Tool-enabled LLM
 ↓
Web Search Tool
 ↓
Serper
 ↓
Search Results
 ↓
Final Answer
```

### PDF RAG

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitting
 ↓
HuggingFace Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Relevant Context + LLM
 ↓
Answer
```

### Persistence

Each conversation has a unique thread ID and its state is stored using **SQLite checkpointing**, allowing previous conversations to be resumed.

---

## 🧩 Tech Stack

| Technology         | Purpose                       |
| ------------------ | ----------------------------- |
| Python             | Core development              |
| Streamlit          | Frontend                      |
| LangChain          | LLM & RAG components          |
| LangGraph          | Workflow & tool orchestration |
| Qwen3.8 Flash Free | LLM                           |
| TokenHarbor        | LLM API                       |
| FAISS              | Vector search                 |
| HuggingFace        | Embeddings                    |
| Serper             | Web search                    |
| SQLite             | Conversation persistence      |
| LangSmith          | Tracing & observability       |

---

## 📁 Project Structure

```text
AgentFlow/
│
├── App/
│   ├── frontend.py
│   ├── backend.py
│   └── style.css
│
├── conversation_naming.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AgentFlow
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file:

```env
TOKENHARBOR_API_KEY=your_tokenharbor_api_key
SERPER_API_KEY=your_serper_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=AgentFlow
```

### 5. Run

```bash
streamlit run App/frontend.py
```

Open:

```text
http://localhost:8501
```

> Never commit your `.env` file or API keys to GitHub.

---

## 🧠 LLM Configuration

AgentFlow uses **Qwen3.8 Flash Free** through the TokenHarbor OpenAI-compatible API.

```python
simple_model = ChatOpenAI(
    model="qwen3.8-flash:free",
    api_key=os.getenv("TOKENHARBOR_API_KEY"),
    base_url="https://tokenharbor.ai/v1",
    temperature=0,
)
```

---

## 🎥 Demo

The project is demonstrated locally through Streamlit.

The demo covers:

* Normal conversation
* Web search
* PDF RAG
* Streaming
* Persistent conversations
* Resume chat
* LangSmith tracing

---

## 👨‍💻 Author

**Rajesh Kumar**
B.Tech — Artificial Intelligence & Data Science

**LinkedIn:** [Rajesh Kumar](https://www.linkedin.com/in/rajeshxdatascience/)

**GitHub:** [rajeshxdatascience](https://github.com/rajeshxdatascience)
