# Phase 1 libraries
from dotenv import load_dotenv
load_dotenv()
import os
import warnings
import logging

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Phase 3 libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tempfile import NamedTemporaryFile

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ----- Streamlit UI Setup -----
st.set_page_config(page_title="📚 PDF AI Assistant", layout="wide")

# Sidebar Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.markdown("## 🤖 PDF Q&A Assistant")
    st.markdown("Upload any PDF and ask questions about its content using **Groq LLM + LangChain**.")
    add_vertical_space(2)
    st.markdown("---")
    st.markdown("👨‍💼 Developed by: *Muhammad Umer Azam*")

# Main App Title
st.title("📄 Smart PDF Chatbot")
st.markdown("""
<style>
    .big-font { font-size: 18px !important; font-weight: 400;}
    .chat-message {
        padding: 0.8em 1em;
        border-radius: 10px;
        margin-bottom: 0.8em;
    }
    .user-msg { background-color: #dfe6e9; }
    .bot-msg { background-color: #a29bfe; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">Upload your PDF and ask questions directly from it using natural language.</div>', unsafe_allow_html=True)
st.markdown("---")

# File Upload
uploaded_file = st.file_uploader("📎 Upload your PDF file", type="pdf")

# Chat memory
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display old messages in style
for message in st.session_state.messages:
    css_class = 'user-msg' if message['role'] == 'user' else 'bot-msg'
    st.markdown(f'<div class="chat-message {css_class}">{message["content"]}</div>', unsafe_allow_html=True)

# PDF Processing
if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    @st.cache_resource
    def load_vectorstore_from_pdf(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
        vectordb = Chroma.from_documents(texts, embeddings)
        return vectordb

    vectorstore = load_vectorstore_from_pdf(tmp_path)

    # Chat Input
    prompt = st.chat_input("💬 Type your question here...")

    if prompt:
        st.markdown(f'<div class="chat-message user-msg">{prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Model and Chain Setup
        try:
            groq_chat = ChatGroq(
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                model_name="llama3-8b-8192"
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            result = qa_chain({"query": prompt})
            answer = result["result"]

            st.markdown(f'<div class="chat-message bot-msg">{answer}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Please upload a PDF to get started.")

