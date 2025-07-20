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

# --- Custom CSS Styling ---
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
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(5px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        text-align: right;
        z-index: 9999;
    }
    .footer img {
        border-radius: 50%;
        height: 40px;
        width: 40px;
        object-fit: cover;
    }
    .info-box {
        background-color: #111;
        padding: 15px;
        border-radius: 15px;
        margin-top: 20px;
        color: white;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Header ---
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4712/4712039.png' width='40' style='margin-right: 10px;'>
        <h1 style='margin: 0;'>💬 LangChain Chatbot</h1>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button("📁 Download Chat Log", data=chat_log, file_name="chat_log.txt", mime="text/plain")

# --- Model Fallback Chain ---
model_fallbacks = ["llama3-70b-8192", "llama3-8b-8192", "mistral-saba-24b"]
llm, selected_model = None, None

for model in model_fallbacks:
    try:
        llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=groq_api_key)
        selected_model = model
        break
    except Exception as e:
        print(f"❌ Model failed: {model} → {e}")
        continue

if not llm:
    st.error("❌ Failed to initialize any supported Groq models.")
    st.stop()

st.sidebar.markdown(f"🧠 Using model: `{selected_model}`")

# --- Tool Integration ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(Tool(name="Web Search", func=search.run, description="Use this for answering questions using web results."))

# --- LangChain Agent ---
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, verbose=False)

# --- Session State Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Avatar Icons ---
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

# --- Chat UI ---
user_input = st.chat_input("Type your message here...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

if user_input:
    st.chat_message("user", avatar=USER_AVATAR).markdown(f"<div class='chat-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = agent.run(user_input)
    except Exception as e:
        response = f"❌ Error: {str(e)}"

    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n")
        f.write(f"[{datetime.now()}] Bot: {response}\n\n")

# --- Bot Info Section ---
st.sidebar.markdown("""
---
### 🤖 Bot Info
<div class="info-box">
<b>LangChain Groq Chatbot</b><br>
Ask anything using powerful Groq LLMs + Web Search.<br><br>
🔗 <a href="https://www.linkedin.com/in/shubham-vardani-325428174/" target="_blank">LinkedIn</a><br>
💻 <a href="https://github.com/123ShubhamVardani" target="_blank">GitHub</a><br>
📧 <a href="mailto:shub.vardani@gmail.com">shub.vardani@gmail.com</a>
</div>
""", unsafe_allow_html=True)

# --- Footer Avatar ---
st.markdown("""
<div class="footer">
    <img src="https://avatars.githubusercontent.com/u/120295633?v=4" alt="Shubham Vardani">
</div>
""", unsafe_allow_html=True)
