from langchain_community.document_loaders import UnstructuredFileLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import nltk

nltk.download('punkt_tab')

docs_dir_path = "D:\GenAI_work\docs_dir"
vector_db_path = "D:\GenAI_work\myvector_db"
collection_name="document_collection"

embedding = HuggingFaceEmbeddings()

loader = DirectoryLoader(docs_dir_path, glob="./*.pdf", loader_cls=UnstructuredFileLoader)

documents = loader.load()

print(type(documents))

print(len(documents))

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

splits = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(splits, 
                                     embedding, 
                                     persist_directory=vector_db_path, 
                                     collection_name=collection_name)






