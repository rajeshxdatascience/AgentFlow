import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage)

import uuid

from backend import (
    chatbot,
    retrieve_all_threads,
    generate_thread_name,
    retrieve_all_thread_names,
    ingest_pdf)


# ============================================================
# CSS
# ============================================================

def load_css():

    with open("APP/style.css") as f:

        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="AgentFlow",
    page_icon="🤖",
    layout="centered",
)

load_css()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id():

    return str(uuid.uuid4())


def reset_chat():

    # Don't create another thread
    # if current chat is empty.

    if not st.session_state["message_history"]:

        return

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def add_thread(thread_id):

    if (thread_id not in st.session_state["chat_threads"]):

        st.session_state["chat_threads"].insert(0, thread_id)


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================
# WEB SEARCH ROUTING
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
# SESSION SETUP
# ============================================================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = retrieve_all_threads()


if "thread_names" not in st.session_state:

    st.session_state["thread_names"] = retrieve_all_thread_names()


if "ingested_docs" not in st.session_state:

    st.session_state["ingested_docs"] = {}


# ============================================================
# ADD CURRENT THREAD
# ============================================================

add_thread(st.session_state["thread_id"])


thread_key = str(st.session_state["thread_id"])


# ============================================================
# CURRENT THREAD DOCUMENTS
# ============================================================

thread_docs = (st.session_state["ingested_docs"].setdefault(thread_key, {}))


# ============================================================
# SIDEBAR THREADS
# ============================================================

threads = (st.session_state["chat_threads"][::-1])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("AgentFlow")


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True):

    reset_chat()

    st.rerun()


# ============================================================
# PDF / RAG
# ============================================================

st.sidebar.subheader("Knowledge Base")


if thread_docs:

    latest_doc = list(thread_docs.values())[-1]

    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks)"
    )

else:

    st.sidebar.info("No PDF indexed yet.")


uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"],)


if uploaded_pdf:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"`{uploaded_pdf.name}` already "
            f"processed for this chat.")

    else:

        with st.sidebar.status(
            "📄 Indexing PDF...",
            expanded=True,
        ) as status_box:

            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=st.session_state[
                    "thread_id"
                ],
                filename=uploaded_pdf.name,
            )

            thread_docs[uploaded_pdf.name] = summary

            status_box.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )


# ============================================================
# CONVERSATIONS
# ============================================================

st.sidebar.header("My Conversations")


for thread_id in st.session_state["chat_threads"]:

    thread_name = (
        st.session_state[
            "thread_names"
        ].get(
            thread_id,
            "New Conversation",
        )
    )

    if st.sidebar.button(
        thread_name,
        key=f"thread_{thread_id}",
        use_container_width=True,
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                temp_messages.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )

            elif isinstance(msg, AIMessage):

                if msg.content:

                    temp_messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content,
                        }
                    )

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="chat-header">
        <h1>🤖 AgentFlow</h1>
        <p>Your intelligent AI assistant.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "configurable": {
        "thread_id": st.session_state[
            "thread_id"
        ]
    }
}


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"],
        avatar=(
            "👤"
            if message["role"] == "user"
            else "🤖"
        ),
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input("Message your AI assistant...")


if user_input:

    # --------------------------------------------------------
    # Add current thread
    # --------------------------------------------------------

    add_thread(st.session_state["thread_id"])


    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )


    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        # ----------------------------------------------------
        # Status box
        # ----------------------------------------------------

        status_box = st.status(
            "🤔 Thinking...",
            expanded=False,
        )


        # ----------------------------------------------------
        # Determine initial backend route
        # ----------------------------------------------------

        is_pdf_chat = bool(
            thread_docs
        )

        is_web_chat = needs_web_search(
            user_input
        )


        if is_pdf_chat:

            status_box.update(
                label="📄 Reading uploaded PDF...",
                state="running",
                expanded=True,
            )

        elif is_web_chat:

            status_box.update(
                label="🌐 Preparing web search...",
                state="running",
                expanded=True,
            )

        else:

            status_box.update(
                label="🤖 Generating response...",
                state="running",
                expanded=True,
            )


        # ----------------------------------------------------
        # Stream assistant response
        # ----------------------------------------------------

        status_holder = {
            "tool_name": None,
            "tool_started": False,
        }


        def ai_only_stream():

            for (
                message_chunk,
                metadata
            ) in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # ============================================
                # AI MESSAGE CHUNK
                # ============================================

                if isinstance(
                    message_chunk,
                    AIMessageChunk,
                ):

                    # ----------------------------------------
                    # Detect tool call
                    # ----------------------------------------

                    tool_name = None

                    tool_call_chunks = getattr(
                        message_chunk,
                        "tool_call_chunks",
                        [],
                    )


                    if tool_call_chunks:

                        for tool_call in (
                            tool_call_chunks
                        ):

                            name = tool_call.get(
                                "name"
                            )

                            if name:

                                tool_name = name

                                break


                    # ----------------------------------------
                    # Completed tool calls
                    # ----------------------------------------

                    if not tool_name:

                        tool_calls = getattr(
                            message_chunk,
                            "tool_calls",
                            [],
                        )

                        if tool_calls:

                            tool_name = (
                                tool_calls[
                                    0
                                ].get(
                                    "name"
                                )
                            )


                    # ----------------------------------------
                    # Tool started
                    # ----------------------------------------

                    if tool_name:

                        status_holder[
                            "tool_name"
                        ] = tool_name

                        status_holder[
                            "tool_started"
                        ] = True

                        status_box.update(
                            label=(
                                f"🔎 Searching the web "
                                f"using `{tool_name}`..."
                            ),
                            state="running",
                            expanded=True,
                        )


                    # ----------------------------------------
                    # Normal content
                    # ----------------------------------------

                    content = (message_chunk.content)


                    if content:

                        if isinstance(content, str):

                            yield content


                        elif isinstance(content, list):

                            for block in content:

                                if isinstance(block, dict):

                                    text = block.get("text")

                                    if text:

                                        yield text


                # ============================================
                # TOOL MESSAGE
                # ============================================

                elif isinstance(message_chunk, ToolMessage):

                    tool_name = (
                        status_holder[
                            "tool_name"
                        ]
                        or getattr(
                            message_chunk,
                            "name",
                            "web_search",
                        )
                    )


                    status_box.update(
                        label=(
                            f"✅ `{tool_name}` "
                            "completed. Generating answer..."
                        ),
                        state="running",
                        expanded=True,
                    )


        # ----------------------------------------------------
        # Write stream
        # ----------------------------------------------------

        ai_message = st.write_stream(ai_only_stream())


        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        status_box.update(
            label="✅ Response complete",
            state="complete",
            expanded=False,
        )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )


    # ========================================================
    # GENERATE CONVERSATION NAME
    # ========================================================

    if len(st.session_state["message_history"]) == 2:

        thread_id = (st.session_state["thread_id"])

        thread_name = generate_thread_name(thread_id)

        st.session_state["thread_names"][thread_id] = thread_name

        st.rerun()