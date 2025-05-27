# tool.py

import os
import requests
from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time
from logger import log_tool_usage  # ✅ now imported from logger.py

# def get_arxiv_tool():
    # return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
# 
# def get_wikipedia_tool():
    # return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
# 
def get_tavily_tool():
    return TavilySearchResults()
# 

def get_arxiv_tool():
    wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)

    class LoggedArxivQueryRun(ArxivQueryRun):
        def invoke(self, query):
            start = time.time()
            result = super().invoke(query)
            end = time.time()

            # Extract metadata if available
            doc = wrapper.run(query)[0] if wrapper.run(query) else {}
            log_tool_usage(
                tool_name="arxiv",
                query=query,
                response_time=end - start,
                response=result,
                query_type="paper",
                paper_title=doc.metadata.get("title", "") if hasattr(doc, "metadata") else "",
                paper_authors=doc.metadata.get("authors", "") if hasattr(doc, "metadata") else "",
                paper_year=doc.metadata.get("year", "") if hasattr(doc, "metadata") else ""
            )
            return result

    return LoggedArxivQueryRun(api_wrapper=wrapper)


def get_wikipedia_tool():
    wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

    class LoggedWikipediaQueryRun(WikipediaQueryRun):
        def invoke(self, query):
            start = time.time()
            result = super().invoke(query)
            end = time.time()

            log_tool_usage(
                tool_name="wikipedia",
                query=query,
                response_time=end - start,
                response=result,
                query_type="encyclopedia"
            )
            return result

    return LoggedWikipediaQueryRun(api_wrapper=wrapper)




@tool
def tell_joke(input: str) -> str:
    """Fetches a random programming joke from the JokeAPI."""
    import time
    start_time = time.time()
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Programming?type=single")
        data = response.json()
        joke = data.get("joke", "Couldn't find a joke at the moment.")
        success = "joke" in data
    except Exception as e:
        joke = f"Error fetching joke: {e}"
        success = False

    log_tool_usage(
        tool_name="tell_joke",
        query=input,
        response_time=time.time() - start_time,
        response=joke,
        joke_category="Programming",
        success=success
    )
    return joke

@tool(description="Use this tool to answer questions from the uploaded DBMS textbook PDF.")
def ask_pdf(input: str) -> str:
    import time
    start_time = time.time()
    question = input
    try:
        pdf_path = '../ai-bot/dbms.pdf'
        vectorstore_path = "faiss_dbms_index"

        if not os.path.exists(pdf_path):
            return "❌ PDF file not found at specified path."

        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(vectorstore_path):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        else:
            docs = splitter.split_documents(documents)
            if not docs:
                return "❌ No content extracted from PDF."
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(vectorstore_path)

        retriever = vectorstore.as_retriever()
        llm = ChatGroq(temperature=0, model_name="qwen-qwq-32b")

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
        answer = qa.run(question)

        log_tool_usage(
            tool_name="ask_pdf",
            query=question,
            response_time=time.time() - start_time,
            response=answer,
            query_type="dbms_question",
            success=True
        )
        return answer

    except Exception as e:
        error_msg = f"❌ Error in PDF tool: {str(e)}"
        log_tool_usage(
            tool_name="ask_pdf",
            query=question,
            response_time=time.time() - start_time,
            response=error_msg,
            query_type="dbms_question",
            success=False
        )
        return error_msg

def get_all_tools():
    return [
        get_arxiv_tool(),
        get_wikipedia_tool(),
        get_tavily_tool(),
        tell_joke,
        ask_pdf
    ]
