import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper

# --- Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# --- Page Setup ---
st.set_page_config(page_title="LangChain Chatbot", layout="wide")

# --- Styling and Custom CSS ---
st.markdown("""
<style>
    body {
        background-color: #121212;
    }
    .main-panel {
        display: flex;
        flex-direction: row;
    }
    .sidebar-left {
        background-color: #1c1c1e;
        padding: 2rem 1rem;
        min-width: 250px;
        height: 100vh;
        border-right: 1px solid #333;
    }
    .sidebar-left h1 {
        font-size: 1.5rem;
        color: #fff;
        margin-bottom: 1.5rem;
    }
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
    .avatar-img {
        height: 36px;
        width: 36px;
        border-radius: 50%;
        object-fit: cover;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(5px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("⚙️ Settings")

    mode = st.radio("Select Mode:", ["Chat Mode", "Latest Updates"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button(
            label="📁 Download Chat Log",
            data=chat_log,
            file_name="chat_log.txt",
            mime="text/plain"
        )

# --- Model Selection ---
model_fallbacks = ["llama3-70b-8192", "llama3-8b-8192", "mistral-saba-24b"]
llm, selected_model = None, None

for model in model_fallbacks:
    try:
        llm = ChatGroq(
            temperature=temperature,
            model_name=model,
            groq_api_key=groq_api_key
        )
        selected_model = model
        break
    except Exception as e:
        print(f"❌ Model failed: {model} → {e}")
        continue

if not llm:
    st.error("❌ Failed to initialize any supported Groq models.")
    st.stop()

st.sidebar.markdown(f"🧠 Using model: `{selected_model}`")

# --- Tool Setup ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(
        Tool(
            name="Web Search",
            func=search.run,
            description="Use this for answering questions using web results."
        )
    )

# --- Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    verbose=False
)

# --- Session ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Avatars ---
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

# --- Chat UI ---
user_input = st.chat_input("Type your message here...")

if mode == "Chat Mode":
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

elif mode == "Latest Updates":
    st.markdown("### 📰 Latest Updates will appear here soon... (Coming in Phase 2 🚧)")
