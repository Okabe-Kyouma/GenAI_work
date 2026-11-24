import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

working_directory = os.path.dirname(os.path.abspath(__file__))

embedding = HuggingFaceEmbeddings()

llm = ChatGroq(
    model="openai/gpt-oss-20b",    
)

def process_document_to_chromadb(file_name):
    loader = UnstructuredPDFLoader(f"{working_directory}/{file_name}")
    document = loader.load()

    text_slitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_slitter.split_documents(document)
    vectorDb = Chroma.from_documents(texts, embedding, persist_directory=f"{working_directory}/chroma_db")

    return 0



def answer_question(user_question):
    vectorDb = Chroma(persist_directory=f"{working_directory}/chroma_db", embedding_function=embedding)
    retriever = vectorDb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa.invoke({"query": user_question})
    answer = response['result']

    return answer

