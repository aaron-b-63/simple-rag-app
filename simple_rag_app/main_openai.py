import os
import uuid
import tempfile

import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#NOTE(aaron_b) initializer
#session id
if "session_id" not in st.session_state:
    full_uuid = str(uuid.uuid4())
    st.session_state.session_id = full_uuid[:5]  # 最初の5文字を使用

#chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#temporary files (RAG)
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []


#RAG settings
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = CHUNK_SIZE

#RAG overlap
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = CHUNK_OVERLAP


st.write(f"Session ID: `{st.session_state.session_id}`")

st.title("LangChain RAG")

#NOTE(aaron_b) configure OpenAI
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# RAG PART
uploaded_files = st.file_uploader(
    "Upload your files (pdf/md/txt/docx/pdf)",
    type=["pdf", "md", "txt", "docx", "pdf"],
    accept_multiple_files=True
    )

tmp_files = list()
loaders = list()
documents = []

if uploaded_files and st.button("Process RAG"):
    for file in uploaded_files:
        suffix = file.name.split(".")[-1]
        if suffix == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file = tmp_file.name
                tmp_files.append(tmp_file)
                loaders.append(PyPDFLoader(tmp_file)) 
        elif suffix == "md" or suffix == "txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
                tmp_file.write(file.read())
                tmp_file = tmp_file.name
                tmp_files.append(tmp_file)
                loaders.append(TextLoader(tmp_file.name)) 
        elif suffix == "pptx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
                tmp_file.write(file.read())
                tmp_file = tmp_file.name
                tmp_files.append(tmp_file)
                loaders.append(UnstructuredPowerPointLoader(tmp_file.name)) 
        elif suffix == "docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file.read())
                tmp_file = tmp_file.name
                tmp_files.append(tmp_file)
                loaders.append(UnstructuredWordDocumentLoader(tmp_file.name)) 
        
        st.session_state.temp_files += tmp_files

    assert len(tmp_files)==len(loaders)

    for i_loader, loader in enumerate(loaders):
        docs = loader.load()
        for idoc, doc in enumerate(docs):
            doc.metadata["source"] = tmp_files[i_loader]
        documents.extend(docs)

    # Vectorizer
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap
    )
    splits = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    vectorstore.save_local(os.path.join(tempfile.gettempdir(), st.session_state.session_id))
    st.success("successfully processed uploaded files!")

# USER INTERACTIVE PART
st.text_input("input your question",key="user_question")

if st.button("Send"):
    question = st.session_state.user_question
    st.session_state.chat_history.append(question)
    st.markdown("### question")
    st.write(question)

    rag_cache = os.path.join(tempfile.gettempdir(), st.session_state.session_id)
    if os.path.isdir(rag_cache):
        vectorstore = FAISS.load_local(
            os.path.join(tempfile.gettempdir(), st.session_state.session_id),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True  # ← これを追加！
        )  
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
        answer = qa.run(question)
    else:
        st.write("**No RAG resources were inputted. llm answers with general information**")
        answer = ChatOpenAI(temperature=0).predict(question)
    
    st.markdown("### answer")
    st.write(answer)


if st.button("Delete RAG Sources"):
    for tmp in st.session_state.temp_files:
        os.remove(tmp)
