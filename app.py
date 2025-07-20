import streamlit as st
import os
from dotenv import load_dotenv
from modules.sidebar import render_sidebar
from modules.avatar_ui import message_avatar_ui
from modules.chat_handler import handle_chat
from modules.utils import init_session_state

# Load environment variables
load_dotenv()

st.set_page_config(page_title="LangChain Chatbot", layout="wide")

# Initialize session state
init_session_state()

# Inject custom CSS for fonts, avatars, and layout
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }

        /* Avatar style for user/bot chat */
        .avatar-img {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-right: 10px;
        }

        .chat-message {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .message-text {
            padding: 0.6rem 1rem;
            border-radius: 0.8rem;
            max-width: 85%;
            background-color: #262730;
        }

        .user-message .message-text {
            background-color: #005bbb;
            color: white;
        }

        .bot-message .message-text {
            background-color: #202123;
        }

        /* DP Footer */
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

# Sidebar
render_sidebar()

# Header
st.markdown("""
    <h1 style='text-align: center;'>💬 LangChain Chatbot</h1>
    <hr style='margin-top: -10px;'>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()
with chat_container:
    for i in range(len(st.session_state.messages)):
        role = st.session_state.messages[i]["role"]
        content = st.session_state.messages[i]["content"]
        message_avatar_ui(role, content)

    user_query = st.chat_input("Type your message here...")
    if user_query:
        handle_chat(user_query)

# Sticky footer with DP
st.markdown("""
    <div class="footer-container">
        <img src="https://raw.githubusercontent.com/123ShubhamVardani/langchain-chatbot/main/shubham_dp.png" alt="Profile DP">
    </div>
""", unsafe_allow_html=True)
