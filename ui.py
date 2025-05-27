

# ui.py

import streamlit as st
import time
import os
import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage

from main import graph  # Import the LangGraph pipeline
from logger import log_tool_usage, tool_usage_stats  # ✅ Import from logger.py

csv_path = "tool_usage_log.csv"

# Streamlit UI setup
st.set_page_config(page_title="LangGraph Assistant", layout="wide")
st.title("🧠 LangGraph AI Assistant")

# Tabs for Chat and Analytics
tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Tool Usage Analytics"])

with tab1:
    st.subheader("Ask something and see how tools are triggered!")
    query = st.text_input("Enter your query", placeholder="Try: Tell me a programming joke")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Submit"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Thinking..."):
                start = time.time()
                result = graph.invoke({"messages": [HumanMessage(content=query)]})
                end = time.time()

                chat_response = []
                for m in result["messages"]:
                    if isinstance(m, ToolMessage):
                        # Minimal logging here (only tool name, query, response_time)
                        log_tool_usage(tool_name=m.name, query=query, response_time=end - start)
                        chat_response.append(f"🔧 **Tool used: {m.name}**\n\n🧠 {m.content}")
                    elif hasattr(m, "content") and m.content:
                        chat_response.append(f"🗣️ {m.content}")

                # Save response in session state
                st.session_state.chat_history.append((query, chat_response))

    # Display chat history
    for q, responses in reversed(st.session_state.chat_history):
        st.markdown(f"#### ❓ You asked: {q}")
        for r in responses:
            st.markdown(r)
        st.markdown("---")

with tab2:
    st.subheader("Tool Usage Analytics")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        usage_summary = df["tool_name"].value_counts()
        st.bar_chart(usage_summary)

        with st.expander("📄 Raw Log Data"):
            st.dataframe(df)
    else:
        st.info("No tool usage recorded yet.")
