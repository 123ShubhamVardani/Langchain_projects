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

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
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
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(5px);}
        to {opacity: 1; transform: translateY(0);}
    }

    footer {visibility: hidden;}

    .footer-container {
        position: fixed;
        bottom: 10px;
        right: 10px;
        z-index: 9999;
    }
    .footer-container img {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        border: 2px solid #fff;
        box-shadow: 0px 0px 6px rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("LangChain\nChatbot")
    st.markdown("---")

    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)
    manual_mode = st.toggle("🧩 Manual Model Selection", value=False)

    model_options = ["llama3-70b-8192", "llama3-8b-8192", "mistral-saba-24b"]
    selected_model = None

    if manual_mode:
        selected_model = st.selectbox("Select Groq Model", model_options)

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button("📁 Download Chat Log", data=chat_log, file_name="chat_log.txt", mime="text/plain")

    with st.expander("ℹ️ About this Chatbot"):
        st.markdown("""
        **LangChain Chatbot** is powered by [LangChain](https://www.langchain.com/), 
        integrated with Groq LLMs and optional live search using SerpAPI.

        **Usage:**
        - Chat with natural language
        - Auto fallback to the fastest working model
        - Enable web search for real-time answers

        **Model Priority:**
        1. `llama3-70b-8192`
        2. `llama3-8b-8192`
        3. `mistral-saba-24b`

        **Contact:**
        - GitHub: [123ShubhamVardani](https://github.com/123ShubhamVardani)
        - LinkedIn: [Shubham Vardani](https://www.linkedin.com/in/shubham-vardani-325428174/)
        - Email: [shub.vardani@gmail.com](mailto:shub.vardani@gmail.com)
        """)

# --- Model Setup ---
llm = None
if manual_mode and selected_model:
    try:
        llm = ChatGroq(temperature=temperature, model_name=selected_model, groq_api_key=groq_api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize model `{selected_model}`: {str(e)}")
        st.stop()
else:
    for model in model_options:
        try:
            llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=groq_api_key)
            selected_model = model
            break
        except Exception:
            continue

if not llm:
    st.error("❌ Could not initialize any model. Check API key or quota.")
    st.stop()

st.sidebar.markdown(f"<span style='color: pink;'>🧠 Using model:</span> <code>{selected_model}</code>", unsafe_allow_html=True)

# --- Tools ---
tools = []
if use_web_search and serpapi_api_key:
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools.append(
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for real-time questions."
        )
    )

# --- LangChain Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    verbose=False
)

# --- Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Avatars ---
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

# --- Chat Interface ---
user_input = st.chat_input("Type your message here...")

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(f"<div class='chat-bubble {'bot' if msg['role']=='assistant' else ''}'>{msg['content']}</div>", unsafe_allow_html=True)

# Handle input
if user_input:
    st.chat_message("user", avatar=USER_AVATAR).markdown(f"<div class='chat-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = agent.run(user_input)
    except Exception as e:
        response = f"❌ Error: {str(e)}"

    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Save chat log
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n")
        f.write(f"[{datetime.now()}] Bot: {response}\n\n")

# --- Footer Avatar ---
st.markdown("""
<div class="footer-container">
    <img src="https://raw.githubusercontent.com/123ShubhamVardani/langchain-chatbot/main/shubham_dp.png" alt="Creator DP">
</div>
""", unsafe_allow_html=True)
