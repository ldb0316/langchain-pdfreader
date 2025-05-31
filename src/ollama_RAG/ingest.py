from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
import tempfile
import streamlit as st
import faiss

@st.cache_resource
def indexing(file):
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        file_path = tmp_file.name
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="phi4:latest")

    # GPU 사용 가능 여부 확인
    if faiss.get_num_gpus() > 0:
        # GPU 리소스 설정
        res = faiss.StandardGpuResources()
        
        # GPU 메모리 제한 설정 (예: 4GB)
        res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB in bytes
        
        # 특정 GPU 선택 (예: GPU 0)
        faiss.cuda.get_device(0)
        
        # FAISS 인덱스 생성 시 GPU 사용 설정
        index = VectorstoreIndexCreator(
            vectorstore_cls=FAISS,
            embedding=embeddings,
            vectorstore_kwargs={
                "index": faiss.IndexFlatL2(embeddings.embedding_dimensions),
                "gpu_resources": res
            }
        ).from_loaders([loader])
    else:
        print("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        index = VectorstoreIndexCreator(
            vectorstore_cls=FAISS,
            embedding=embeddings
        ).from_loaders([loader])

    # 로컬 파일로 저장
    index.vectorstore.save_local(file.name)
