from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import streamlit as st

@st.cache_resource
def indexing(file):
    # file_name = '소나기.pdf'

    # 2. 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        file_path = tmp_file.name
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings

    #embeddings = HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings()

    from langchain.indexes import VectorstoreIndexCreator
    from langchain.vectorstores import FAISS

    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        ).from_loaders([loader])

    # 파일로 저장
    index.vectorstore.save_local(file.name)
