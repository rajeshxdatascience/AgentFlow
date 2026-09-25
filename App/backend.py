# ============================================================
# IMPORTS
# ============================================================

from langgraph.graph import StateGraph, START
from typing import Annotated, Any, Dict, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from conversation_naming import (
    generate_conversation_name,
    create_conversation_names_table,
    retrieve_conversation_names,
)

from dotenv import load_dotenv

import sqlite3
import http.client
import json
import urllib.parse
import os
import tempfile


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 1. LLM + EMBEDDINGS
# ============================================================

# ------------------------------------------------------------
# Simple model
# ------------------------------------------------------------

simple_model = ChatOpenAI(
    model="qwen3.8-flash:free",
    api_key=os.getenv("TOKENHARBOR_API_KEY"),
    base_url="https://tokenharbor.ai/v1",
    temperature=0,
)


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={
        "batch_size": 32,
        "normalize_embeddings": True,
    },
)


# ============================================================
# 2. PDF RETRIEVER STORE
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):

    if thread_id:
        return _THREAD_RETRIEVERS.get(str(thread_id))

    return None


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None,) -> dict:

    """Index a PDF once and keep its FAISS retriever in memory."""

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        docs = PyPDFLoader(temp_path).load()

        # ----------------------------------------------------
        # Split documents
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, separators=["\n\n", "\n"," ","",],)

        chunks = splitter.split_documents(docs)

        # ----------------------------------------------------
        # Create FAISS
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(chunks,embeddings)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3},)

        # ----------------------------------------------------
        # Store retriever
        # ----------------------------------------------------

        thread_id = str(thread_id)

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": (
                filename
                or os.path.basename(temp_path)
            ),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": (
                filename
                or os.path.basename(temp_path)
            ),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:

        try:
            os.remove(temp_path)

        except OSError:
            pass


# ============================================================
# 3. WEB SEARCH TOOL
# ============================================================

@tool
def web_search(query: str) -> str:

    """Search the web for current or time-sensitive information."""

    conn = http.client.HTTPSConnection(
        "google.serper.dev",
        timeout=8,
    )

    encoded_query = urllib.parse.quote_plus(
        query
    )

    headers = {
        "X-API-KEY": os.getenv(
            "SERPER_API_KEY"
        ),
        "Content-Type": "application/json",
    }

    try:

        conn.request(
            "GET",
            f"/search?q={encoded_query}&num=3",
            headers=headers,
        )

        response = conn.getresponse()

        data = json.loads(
            response.read().decode("utf-8")
        )

    finally:

        conn.close()

    results = data.get(
        "organic",
        []
    )

    if not results:
        return "No search results found."

    return "\n\n".join(
        f"{r.get('title', '')}: "
        f"{r.get('snippet', '')[:300]}"
        for r in results[:3]
    )


# ============================================================
# 4. TOOL ENABLED MODEL
# ============================================================

tools = [web_search]

model_with_tools = simple_model.bind_tools(tools)


# ============================================================
# 5. QUERY ROUTING
# ============================================================

WEB_KEYWORDS = [
    "latest",
    "recent",
    "today",
    "current",
    "currently",
    "news",
    "live",
    "price",
    "stock",
    "weather",
    "score",
    "ranking",
    "rankings",
    "now",
    "this week",
    "this month",
]


def needs_web_search(query: str) -> bool:

    query_lower = query.lower()

    return any(
        keyword in query_lower
        for keyword in WEB_KEYWORDS
    )


# ============================================================
# 6. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are AgentFlow, a helpful and concise AI assistant.

Rules:

- Give clear and useful answers.
- Keep answers concise unless the user asks for detail.
- If PDF context is provided and relevant, use it.
- Do not invent facts from the PDF.
- If the PDF does not contain the answer, say so clearly.
- Web search results may be used when provided.
- Do not claim that you searched the web unless web results
  are actually present in the conversation.
"""


# ============================================================
# 7. STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 8. CHAT NODE
# ============================================================

def chat_node(state: ChatState, config: RunnableConfig,):

    thread_id = (config.get("configurable") or {}).get("thread_id")

    # --------------------------------------------------------
    # Latest human message
    # --------------------------------------------------------

    latest_user_message = ""

    for message in reversed(state["messages"]):

        if getattr(message,"type", None) == "human":

            latest_user_message = str(message.content)

            break

    # --------------------------------------------------------
    # Check whether a tool already executed
    # --------------------------------------------------------

    has_tool_result = any(getattr(message, "type", None) == "tool"
        for message in state["messages"])

    # --------------------------------------------------------
    # PDF retrieval
    #
    # Do NOT retrieve again after web tool execution.
    # --------------------------------------------------------

    pdf_context = ""

    retriever = _get_retriever(thread_id)

    if (
        retriever is not None
        and latest_user_message
        and not has_tool_result
    ):

        print("📄 Backend: Retrieving PDF context...")

        documents = retriever.invoke(latest_user_message)

        if documents:

            source_file = (
                _THREAD_METADATA
                .get(
                    str(thread_id),
                    {}
                )
                .get(
                    "filename",
                    "uploaded PDF"
                )
            )

            context_parts = []

            for i, doc in enumerate(documents, start=1):

                context_parts.append(f"[Source {i} | {source_file}]\n"
                    f"{doc.page_content[:1200]}")

            pdf_context = "\n\n".join(context_parts)

            print("📄 Backend: PDF context ready")

    # --------------------------------------------------------
    # PDF instruction
    # --------------------------------------------------------

    if pdf_context:

        pdf_instruction = f"""

PDF CONTEXT:

{pdf_context}

Use the PDF context when it directly
answers the user's question.

Do not invent information that is not
supported by the PDF.
"""

    else:

        pdf_instruction = ""

    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    messages = [SystemMessage(content=(SYSTEM_PROMPT + pdf_instruction)), *state["messages"]]

    # ========================================================
    # ROUTING
    # ========================================================

    # --------------------------------------------------------
    # CASE 1:
    # Tool already executed
    #
    # Tool result is now in the conversation.
    # Use SIMPLE MODEL to generate final answer.
    # --------------------------------------------------------

    if has_tool_result:

        print(
            "🤖 Backend: Generating final answer "
            "from web results..."
        )

        response = simple_model.invoke(messages)

    # --------------------------------------------------------
    # CASE 2:
    # PDF available
    #
    # Direct RAG + simple model.
    # --------------------------------------------------------

    elif pdf_context:

        print(
            "📄 Backend: Generating answer "
            "from PDF..."
        )

        response = simple_model.invoke(messages)

    # --------------------------------------------------------
    # CASE 3:
    # Web search required
    #
    # Only now use tool-enabled model.
    # --------------------------------------------------------

    elif needs_web_search(
        latest_user_message
    ):

        print(
            "🌐 Backend: Using tool-enabled model..."
        )

        response = model_with_tools.invoke(messages)

    # --------------------------------------------------------
    # CASE 4:
    # Normal question
    #
    # No tool schema.
    # --------------------------------------------------------

    else:

        print(
            "🤖 Backend: Using simple model..."
        )

        response = simple_model.invoke(messages)

    return {"messages": [response]}


# ============================================================
# 9. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 10. DATABASE
# ============================================================

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False,)

checkpointer = SqliteSaver(conn=conn)

create_conversation_names_table(conn)


# ============================================================
# 11. CONVERSATION FUNCTIONS
# ============================================================

def generate_thread_name(thread_id):

    return generate_conversation_name(
        thread_id=thread_id,
        checkpointer=checkpointer,
        conn=conn)


def retrieve_all_thread_names():

    return retrieve_conversation_names(conn)


# ============================================================
# 12. LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# 13. RETRIEVE ALL THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = []

    for checkpoint in checkpointer.list(None):

        thread_id = (checkpoint.config["configurable"].get("thread_id"))

        if (thread_id and thread_id not in all_threads):

            all_threads.append(thread_id)

    return all_threads[::-1]


# ============================================================
# 14. DOCUMENT HELPERS
# ============================================================

def thread_has_document(thread_id: str) -> bool:

    return (str(thread_id) in _THREAD_RETRIEVERS)


def thread_document_metadata(thread_id: str) -> dict:

    return _THREAD_METADATA.get(str(thread_id), {})