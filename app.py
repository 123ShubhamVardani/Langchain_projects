# Phase 1: LangChain Groq Chatbot (Updated - Model Fix)

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper

# --- Load API Keys ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# --- Page Setup ---
st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.markdown("💬 LangChain Chatbot", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("LangChain\nChatbot")
    st.markdown("---")

    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)

    model_mode = st.radio("Model Selection Mode", ["Automatic", "Manual"])
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ]

    if model_mode == "Manual":
        selected_model = st.selectbox("Choose Model", available_models)
    else:
        selected_model = None

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button("📁 Download Chat Log", chat_log, "chat_log.txt", mime="text/plain")

# --- Load LLM ---
llm = None
if model_mode == "Automatic":
    for model in available_models:
        try:
            llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=groq_api_key)
            selected_model = model
            break
        except Exception as e:
            print(f"❌ Model failed: {model} → {e}")
    if not selected_model:
        st.error("❌ No available models. Check API key or model status.")
        st.stop()
else:
    try:
        llm = ChatGroq(temperature=temperature, model_name=selected_model, groq_api_key=groq_api_key)
    except Exception as e:
        st.error(f"❌ Failed to load model: {selected_model} → {e}")
        st.stop()

st.sidebar.markdown(f"🧠 Using model: {selected_model}", unsafe_allow_html=True)

# --- Tools ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(Tool(name="Web Search", func=search.run, description="Use this for answering current web-based questions."))

# --- Agent Setup ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    verbose=False
)

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
        st.markdown(f"{msg['content']}", unsafe_allow_html=True)

if user_input:
    st.chat_message("user", avatar=USER_AVATAR).markdown(f"{user_input}", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = agent.run(user_input)
    except Exception as e:
        response = f"❌ Error: {str(e)}"

    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n[{datetime.now()}] Bot: {response}\n\n")

# --- CSS Styling ---
st.markdown("""<style>
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
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(5px);}
    to {opacity: 1; transform: translateY(0);}
}
footer {visibility: hidden;}
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
</div>""", unsafe_allow_html=True)
