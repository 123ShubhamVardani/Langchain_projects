import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# --- UI Setup ---
st.set_page_config(page_title="LangChain Jarvis Chatbot", layout="wide")
st.markdown("<h1 style='text-align: center;'>💬 LangChain Chatbot</h1>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
model_options = ["mixtral-8x7b-32768", "mistral-saba-24b", "llama-3-70b"]
selected_model = st.sidebar.selectbox("Choose Model", model_options)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
use_web_search = st.sidebar.checkbox("🔍 Enable Web Search", value=True)

# --- LLM ---
try:
    llm = ChatGroq(
        temperature=temperature,
        model_name=selected_model,
        groq_api_key=groq_api_key
    )
except Exception as e:
    fallback_model = "mistral-saba-24b"
    st.sidebar.warning(f"⚠️ Model '{selected_model}' failed. Using fallback: '{fallback_model}'")
    selected_model = fallback_model
    llm = ChatGroq(
        temperature=temperature,
        model_name=selected_model,
        groq_api_key=groq_api_key
    )
st.sidebar.markdown(f"🧠 Using model: `{selected_model}`")

# --- Tools ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for answering current events or external internet info."
        )
    )

# --- Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # ✅ More robust for LLMs
    handle_parsing_errors=True,
    verbose=False
)

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Avatars ---
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

    # --- Log Chat to File ---
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n")
        f.write(f"[{datetime.now()}] Bot: {response}\n\n")

# --- Download Button for Chat Log ---
if os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "r", encoding="utf-8") as f:
        chat_log = f.read()
    st.download_button(
        label="📁 Download Chat Log",
        data=chat_log,
        file_name="chat_log.txt",
        mime="text/plain"
    )

# --- Custom CSS ---
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
</style>
""", unsafe_allow_html=True)
