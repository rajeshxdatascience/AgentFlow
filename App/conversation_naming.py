from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
import os
import sqlite3


# ============================================================
# HF MODEL FOR CONVERSATION NAMING
# ============================================================

hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=20,
    temperature=0.2)

title_model = ChatHuggingFace(llm=hf_llm)


# ============================================================
# CREATE TABLE
# ============================================================

def create_conversation_names_table(conn):

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_names (
            thread_id TEXT PRIMARY KEY,
            conversation_name TEXT
        )
    """)

    conn.commit()


# ============================================================
# GENERATE NAME
# ============================================================

def generate_conversation_name(thread_id, checkpointer, conn):

    config = {"configurable": {"thread_id": thread_id}}

    # Get latest checkpoint
    checkpoint_tuple = checkpointer.get_tuple(config)

    if checkpoint_tuple is None:
        return None

    messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])

    if not messages:
        return None

    # First human message
    user_message = None

    for message in messages:

        if isinstance(message, HumanMessage):
            user_message = message.content
            break

    if not user_message:
        return None


    # Generate title
    prompt = f"""
Generate a short and meaningful title for this conversation.

User message:
{user_message}

Rules:
- Maximum 5 words
- Describe the main topic
- Keep it natural
- No quotation marks
- No explanation
- Return ONLY the title

Title:
"""

    response = title_model.invoke(prompt)

    title = response.content.strip()

    # Remove accidental quotes
    title = title.strip('"').strip("'")

    # Save title
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO conversation_names
        (thread_id, conversation_name)
        VALUES (?, ?)
    """, (thread_id, title))

    conn.commit()

    return title


# ============================================================
# RETRIEVE ALL NAMES
# ============================================================

def retrieve_conversation_names(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT thread_id, conversation_name
        FROM conversation_names
    """)

    rows = cursor.fetchall()

    return {
        thread_id: conversation_name
        for thread_id, conversation_name in rows
    }