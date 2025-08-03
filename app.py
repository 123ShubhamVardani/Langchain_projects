# Phase 2: LangChain RAG + Groq Chatbot (with model fallback and manual selection)
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Load API Keys ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Page Setup ---
st.set_page_config(page_title="LangChain Chatbot + RAG", layout="wide")
st.markdown("<h1 style='text-align: center;'>💬 LangChain Chatbot + RAG</h1>", unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("LangChain\nChatbot")
    st.markdown("---")

    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=False)

    st.markdown("### 📄 Upload a PDF for RAG (optional)")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        st.success("✅ PDF uploaded successfully!")
    else:
        st.warning("⚠️ No PDF uploaded. Using normal AI chat.")

    model_mode = st.radio("Model Selection Mode", ["Automatic", "Manual"])
    model_list = ["llama3-70b-8192", "llama3-8b-8192"]
    selected_model = st.selectbox("Choose Model", model_list) if model_mode == "Manual" else None

    st.markdown("---")
    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button("📁 Download Chat Log", chat_log, "chat_log.txt", mime="text/plain")

# --- Model Loading ---
llm = None
if model_mode == "Manual":
    try:
        llm = ChatGroq(temperature=temperature, model_name=selected_model, groq_api_key=groq_api_key)
    except Exception as e:
        st.error(f"❌ Failed to load selected model: {e}")
        st.stop()
else:
    for model in model_list:
        try:
            llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=groq_api_key)
            selected_model = model
            print(f"✅ Auto-selected model: {model}")
            break
        except Exception as e:
            print(f"❌ Fallback failed: {model} → {e}")
    if not llm:
        st.error("❌ All fallback models failed. Please check your GROQ_API_KEY or model access.")
        st.stop()

st.sidebar.markdown(f"<span style='color: pink;'>🧠 Using model:</span> <code>{selected_model}</code>", unsafe_allow_html=True)

# --- PDF Upload & Vector DB ---
vector_db = None
if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

# --- Tool Setup ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(Tool(name="Web Search", func=search.run, description="Use this for answering current web-based questions."))

# --- LangChain Agent or RAG Chain ---
if vector_db:
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
else:
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, verbose=False)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Avatars ---
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

# --- Chat Input UI ---
user_input = st.chat_input("Type your message here...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

if user_input:
    st.chat_message("user", avatar=USER_AVATAR).markdown(f"<div class='chat-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        if vector_db:
            response = rag_chain.run(user_input)
        else:
            response = agent.run(user_input)
    except Exception as e:
        response = f"❌ Error: {str(e)}"

    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n[{datetime.now()}] Bot: {response}\n\n")

# --- Footer ---
st.markdown("""
<style>
    .chat-bubble {
        background-color: #1e1e1e;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 5px 0;
        animation: fadeIn 0.3s ease-in-out;
        max-width: 80%;
    }
    .chat-bubble.bot {
        background-color: #2f2f2f;
    }
    .custom-footer {
        position: fixed;
        bottom: 0;
        right: 20px;
        font-size: 12px;
        color: #ccc;
    }
    .custom-footer img {
        height: 24px;
        border-radius: 50%;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
<div class="custom-footer">
    Created by Shubham Vardani <img src="https://avatars.githubusercontent.com/u/104264016?v=4">
</div>
""", unsafe_allow_html=True)
