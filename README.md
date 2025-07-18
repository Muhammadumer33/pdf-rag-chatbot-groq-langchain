# 📄 PDF Chatbot with Groq + LangChain

An AI-powered chatbot that allows users to upload a PDF and ask questions based on its content using a natural language interface — built with **Streamlit**, **LangChain**, and **Groq LLM (LLaMA3)**.

---

## 🎥 Demo

*👉 [Insert Video Link Here or Upload to LinkedIn]*  
This chatbot reads, understands, and responds to any uploaded PDF in real time.

---

## 🚀 Features

- 📎 Upload any PDF document
- 🔍 Extracts, chunks, and embeds content using `HuggingFace` sentence transformers
- ⚡️ Powered by **Groq’s blazing-fast LLaMA3-8B** language model
- 💬 Real-time question answering based on PDF content
- 🧠 Uses `Chroma` vector store for semantic search
- ✅ Simple, beautiful interface with **Streamlit**
- 🧾 Chat history preserved with session state

---

## 🧠 Tech Stack

| Component          | Library / Tool                            |
|--------------------|-------------------------------------------|
| Frontend UI        | Streamlit                                 |
| Language Model     | [Groq API - LLaMA3 8B](https://console.groq.com/) |
| RAG Framework      | LangChain                                 |
| Embeddings         | HuggingFace `all-MiniLM-L12-v2`           |
| Vector Store       | ChromaDB                                  |
| PDF Parsing        | PyPDFLoader (LangChain)                   |
| Environment Config | python-dotenv                             |

---

## 🧑‍💻 Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/pdf-chatbot-groq-langchain.git
cd pdf-chatbot-groq-langchain
Create .env file and add your API key

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
python -m streamlit run app.py
