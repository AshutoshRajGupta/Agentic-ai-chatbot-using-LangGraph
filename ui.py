# ui.py

import streamlit as st
import time
import os
import csv
from datetime import datetime
import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage

from main import graph  # Import the LangGraph pipeline

csv_path = "tool_usage_log.csv"
tool_usage_stats = {}

def log_tool_usage(tool_name, query, response_time):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "tool_name", "query", "response_time_sec"])
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            tool_name,
            query,
            round(response_time, 3)
        ])
    tool_usage_stats[tool_name] = tool_usage_stats.get(tool_name, 0) + 1



# Streamlit UI setup
st.set_page_config(page_title="LangGraph Assistant", layout="wide")
st.title("🧠 LangGraph AI Assistant")

# Tabs for Chat and Analytics
tab1, tab2,tab3, tab4, tab5 = st.tabs(["💬 Chatbot","🌦️ Weather Info", "📘 Ask DBMS PDF", "📊 Tool Usage Analytics","📊 Data Analysis"])

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
                        log_tool_usage(m.name, query, end - start)
                        chat_response.append(f"🔧 *Tool used: {m.name}*\n\n🧠 {m.content}")
                    elif hasattr(m, "content") and m.content:
                        chat_response.append(f"🗣 {m.content}")

                # Save response in session state
                st.session_state.chat_history.append((query, chat_response))

    # Display chat history
    for q, responses in reversed(st.session_state.chat_history):
        st.markdown(f"#### ❓ You asked: {q}")
        for r in responses:
            st.markdown(r)
        st.markdown("---")

# Weather Info Tab
with tab2:
    st.subheader("🌦️ Get Current Weather Info")

    city = st.text_input("Enter city name", placeholder="e.g., Delhi, New York")

    if st.button("Get Weather"):
        if not city.strip():
            st.warning("Please enter a valid city name.")
        else:
            with st.spinner("Fetching weather..."):
                result = graph.invoke({"messages": [HumanMessage(content=f"What is the weather in {city}?")]})
                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        st.markdown(f"🔧 *Tool used: {message.name}*\n\n🧠 {message.content}")
                        log_tool_usage(message.name, city, 0)  # Optional: log weather tool usage
                    elif hasattr(message, "content") and message.content:
                        st.markdown(f"🗣 {message.content}")

# PDF QA Tab
with tab3:
    st.subheader("📘 Ask Questions from DBMS PDF")

    pdf_question = st.text_input("Ask a question based on the DBMS PDF", placeholder="e.g., What is normalization?")

    if st.button("Ask PDF"):
        if not pdf_question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Looking through the PDF..."):
                start = time.time()
                result = graph.invoke({"messages": [HumanMessage(content=pdf_question)]})
                end = time.time()

                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        st.markdown(f"🔧 *Tool used: {message.name}*\n\n🧠 {message.content}")
                        log_tool_usage(message.name, pdf_question, end - start)
                    elif hasattr(message, "content") and message.content:
                        st.markdown(f"🗣 {message.content}")




with tab4:
    st.subheader("Tool Usage Analytics")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        usage_summary = df["tool_name"].value_counts()
        st.bar_chart(usage_summary)

        with st.expander("📄 Raw Log Data"):
            st.dataframe(df)
    else:
        st.info("No tool usage recorded yet.")

# New Data Analysis tab
with tab5:
    st.subheader("CSV Data Analysis")

    # Requesting the agent for data analysis
    if st.button("Analyze CSV Data"):
        with st.spinner("Analyzing..."):
            # Trigger the LangGraph agent to perform CSV analysis
            result = graph.invoke({"messages": [HumanMessage(content="Analyze the CSV data for tool usage stats.")]})

            # Display the analysis result
            for message in result["messages"]:
                if isinstance(message, ToolMessage):
                    st.markdown(f"🔧 *Tool used: {message.name}*\n\n🧠 {message.content}")
                elif hasattr(message, "content") and message.content:
                    st.markdown(f"🗣 {message.content}")



