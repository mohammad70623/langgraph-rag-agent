from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

#load pdf
DATA_PATH = "data/"
def load_pdf_files(data): 
    loader = DirectoryLoader(data, 
                            glob = '*.pdf', 
                            loader_cls = PyMuPDFLoader)

    documents = loader.load() 
    return documents

documents = load_pdf_files(data=DATA_PATH)

#Creat Chunks
def creat_chunks(extracted_data): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, 
                                                  chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 
text_chunks = creat_chunks(extracted_data = documents)

#create embedding
def get_embedding_model(): 
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model = get_embedding_model()

#Store Embedding on FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
