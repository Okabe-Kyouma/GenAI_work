import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

vector_db_path = "D:\GenAI_work\myvector_db"
collection_name="document_collection"

embedding = HuggingFaceEmbeddings()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

vector_store = Chroma(persist_directory=vector_db_path,
                      embedding_function=embedding, 
                      collection_name=collection_name
                      )

retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,   
)

query = "What is the main topic of the document?"
response = qa_chain.invoke({"query": query})
print(response["result"])
print("-"*80)
for source in response["source_documents"]:
    print(source.metadata)
