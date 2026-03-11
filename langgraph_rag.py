from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0.2)

DATA_PATH = "data/"
def load_pdf_files(data): 
    loader = DirectoryLoader(data, 
                            glob = '*.pdf', 
                            loader_cls = PyPDFLoader)

    documents = loader.load() 
    return documents

