import os
import requests
import asyncio
from langchain.tools import tool
from langchain_community.tools import  WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Predefined Tools

def get_wikipedia_tool():
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))

def get_tavily_tool():
    return TavilySearchResults()

@tool
def tell_joke(input: str) -> str:
    """Fetches a random programming joke from the JokeAPI."""
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Programming?type=single")
        data = response.json()
        if data.get("joke"):
            return data["joke"]
        else:
            return "Couldn't find a joke at the moment. Try again later!"
    except Exception as e:
        return f"Error fetching joke: {e}"
    


@tool(description="Get current weather for a given city using WeatherAPI with helpful tips.")
def get_weather(city: str) -> str:
    try:
        api_key = os.getenv("WEATHERAPI_KEY")
        if not api_key:
            return "❌ WeatherAPI key is missing."

        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        response = requests.get(url).json()

        if "error" in response:
            return f"❌ Weather not found for '{city}'."

        location = response["location"]["name"]
        temp = response["current"]["temp_c"]
        condition = response["current"]["condition"]["text"].lower()
        humidity = response["current"]["humidity"]
        wind = response["current"]["wind_kph"]

        # Weather tip logic
        tips = []
        if "rain" in condition:
            tips.append("☔ It's rainy – don't forget your umbrella!")
        if "sunny" in condition or "clear" in condition:
            tips.append("🧴 It's sunny – stay hydrated and use sunscreen.")
        if temp < 10:
            tips.append("🧣 It's quite cold – wear warm clothes.")
        elif temp > 30:
            tips.append("🔥 It's hot – avoid going out in the afternoon.")
        if wind > 20:
            tips.append("💨 It's windy – avoid loose clothing.")

        tip_text = "\n".join(tips) if tips else "✅ No special tips. Enjoy the weather!"

        return (
            f"🌤️ **Weather in {location}**\n"
            f"- Temperature: {temp}°C\n"
            f"- Condition: {condition.title()}\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind} km/h\n\n"
            f"💡 **Tips:**\n{tip_text}"
        )

    except Exception as e:
        return f"❌ Error fetching weather: {e}"






@tool(description="Use this tool to answer questions from the uploaded DBMS textbook PDF.")
def ask_pdf(input: str) -> str:
    question = input
    try:
        pdf_path = '../ai-bot/dbms.pdf'
        vectorstore_path = "faiss_dbms_index"

        if not os.path.exists(pdf_path):
            return "❌ PDF file not found at specified path."

        print("[INFO] Loading DBMS PDF...")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        print("[INFO] Initializing text splitter...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        print("[INFO] Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load or create vectorstore
        if os.path.exists(vectorstore_path):
            print("[INFO] Loading existing FAISS vectorstore...")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings,allow_dangerous_deserialization=True)
        else:
            print("[INFO] Splitting documents and creating FAISS vectorstore...")
            docs = splitter.split_documents(documents)
            if not docs:
                return "❌ No content extracted from PDF."

            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(vectorstore_path)
            print("[INFO] FAISS vectorstore saved locally.")

        # Initialize retriever
        retriever = vectorstore.as_retriever()

        # Initialize LLM (Groq)
        print("[INFO] Loading Groq LLM...")
        llm = ChatGroq(
            temperature=0,
            model_name="gemma2-9b-it"
        )

        print("[INFO] Setting up RetrievalQA chain...")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

        print(f"[INFO] Running QA for question: {question}")
        answer = qa.run(question)

        print("[INFO] Answer generated successfully.")
        return answer

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        return f"❌ Error in PDF tool: {str(e)}"
    

import pandas as pd

# Tool to read and analyze the CSV data
@tool(description="Reads and analyzes tool usage data from CSV.")
def analyze_tool_usage_csv(input: str) -> str:
    try:
        # Load the CSV file
        df = pd.read_csv('tool_usage_log.csv')

        if df.empty:
            return "⚠️ No data available to analyze."

        # Tool usage analysis
        tool_usage_counts = df['tool_name'].value_counts().sort_values(ascending=False)
        total_calls = tool_usage_counts.sum()
        most_used_tool = tool_usage_counts.idxmax()
        least_used_tool = tool_usage_counts.idxmin()

        # Response time analysis
        avg_response_time = df['response_time_sec'].mean()
        max_response_time = df['response_time_sec'].max()
        min_response_time = df['response_time_sec'].min()
        response_time_std = df['response_time_sec'].std()

        # ASCII bar chart
        max_bar_width = 40
        max_count = tool_usage_counts.max()
        bar_chart = "\n".join(
            f"{tool_name.ljust(25)} | {'█' * int(count / max_count * max_bar_width)} {count}"
            for tool_name, count in tool_usage_counts.items()
        )

        # Compile analysis result
        analysis_result = f"📊 TOOL USAGE SUMMARY\n"
        analysis_result += f"{'-'*60}\n"
        analysis_result += f"🔹 Total Tool Calls        : {total_calls}\n"
        analysis_result += f"🔹 Most Used Tool          : {most_used_tool} ({tool_usage_counts[most_used_tool]} times)\n"
        analysis_result += f"🔹 Least Used Tool         : {least_used_tool} ({tool_usage_counts[least_used_tool]} times)\n\n"

        analysis_result += f"⏱️ RESPONSE TIME ANALYSIS (in seconds)\n"
        analysis_result += f"{'-'*60}\n"
        analysis_result += f"🔹 Average Response Time   : {avg_response_time:.2f} sec\n"
        analysis_result += f"🔹 Maximum Response Time   : {max_response_time:.2f} sec\n"
        analysis_result += f"🔹 Minimum Response Time   : {min_response_time:.2f} sec\n"
        analysis_result += f"🔹 Std Dev of Response Time: {response_time_std:.2f} sec\n\n"

        analysis_result += f"📈 TOOL USAGE BREAKDOWN (Bar Chart)\n"
        analysis_result += f"{'-'*60}\n"
        analysis_result += bar_chart

        return analysis_result

    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"




def get_all_tools():
    return [
       
        get_wikipedia_tool(),
        get_tavily_tool(),
        get_weather,
        tell_joke,
        ask_pdf,
        analyze_tool_usage_csv,
    ]

