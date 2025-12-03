import streamlit as st
from src.search import RAGSearch

# Initialize RAGSearch once
rag_search = RAGSearch()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Input box
if query := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    # Get RAG answer
    summary = rag_search.search_and_summarize(query, top_k=3)

    # Add assistant message
    st.session_state["messages"].append({"role": "assistant", "content": summary})
    st.chat_message("assistant").markdown(summary)