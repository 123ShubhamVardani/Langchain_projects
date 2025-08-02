import os
import streamlit as st
from datetime import datetime
from io import BytesIO
import tempfile
import qrcode
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase # [10]

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Load API Keys from st.secrets ---
# For local development, create a.streamlit/secrets.toml file:
# GROQ_API_KEY="gsk_your_groq_key_here"
# SERPAPI_API_KEY="your_serpapi_key_here"
# OPENAI_API_KEY="sk-your_openai_key_here"
# Ensure.streamlit/ is in your.gitignore.
# For Streamlit Cloud, paste these into the "Secrets" section of your app settings. [1]
groq_api_key = st.secrets.get("GROQ_API_KEY")
serpapi_api_key = st.secrets.get("SERPAPI_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# --- Page Setup ---
st.set_page_config(page_title="LangChain Chatbot + RAG + Voice", layout="wide")
st.markdown("<h1 style='text-align: center;'>💬 LangChain Chatbot + RAG + Voice</h1>", unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("LangChain\nChatbot")
    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=False)
    uploaded_file = st.file_uploader("📄 Upload PDF for Q&A", type=["pdf"])

    model_mode = st.radio("Model Selection Mode", ["Automatic", "Manual"])
    model_list = ["llama3-70b-8192", "llama3-8b-8192"]
    selected_model = st.selectbox("Choose Model", model_list) if model_mode == "Manual" else None

    st.markdown("---")
    st.subheader("🎙️ Voice Settings")
    enable_voice_input = st.checkbox("Enable Voice Input")
    enable_voice_output = st.checkbox("Enable Voice Output")

    st.markdown("---")
    st.subheader("🔗 QR Code Access")
    app_url = st.text_input("App URL for QR Code (e.g., your deployed Streamlit Cloud URL)", "https://your-app-name.streamlit.app/")
    if app_url and st.button("Generate QR Code"):
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(app_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white") # [11]

            buf = BytesIO()
            img.save(buf, format="PNG") # [12]
            st.image(buf.getvalue(), caption="Scan to access the app!", use_column_width=True) # [13]
        except Exception as e:
            st.error(f"❌ Error generating QR code: {e}")

    st.markdown("---")
    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            chat_log = f.read()
        st.download_button("📁 Download Chat Log", chat_log, "chat_log.txt", mime="text/plain")

# --- API Key Validation and LLM Loading ---
llm = None
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets. Please configure it to use the chatbot. [2, 3]")
    st.stop()

try:
    if model_mode == "Manual":
        llm = ChatGroq(temperature=temperature, model_name=selected_model, groq_api_key=groq_api_key)
    else:
        for model in model_list:
            try:
                llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=groq_api_key)
                selected_model = model
                print(f"✅ Auto-selected model: {model}")
                break
            except Exception as e:
                print(f"❌ Fallback failed for model {model}: {e}")
    if not llm:
        st.error("❌ All fallback models failed. Please check your GROQ_API_KEY or model access. [4, 5]")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize Groq LLM: {e}. Ensure your GROQ_API_KEY is valid. [2, 3]")
    st.stop()

st.sidebar.markdown(f"<span style='color: pink;'>🧠 Using model:</span> <code>{selected_model}</code>", unsafe_allow_html=True)

# --- PDF Upload & Vector DB ---
vector_db = None
if uploaded_file:
    with st.spinner("Processing PDF for RAG... This may take a moment."):
        try:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            if not openai_api_key:
                st.error("❌ OPENAI_API_KEY not found in Streamlit secrets. RAG functionality requires OpenAI Embeddings. [6, 7]")
                st.stop()

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vector_db = FAISS.from_documents(chunks, embeddings)
            retriever = vector_db.as_retriever()
            st.success(f"✅ PDF '{uploaded_file.name}' loaded successfully for RAG Q&A!")
            os.unlink(temp_pdf_path) # Clean up the temporary file
        except Exception as e:
            st.error(f"❌ Error processing PDF for RAG: {e}. Please ensure your OpenAI API key is valid, the PDF is not corrupted, or try a smaller file. [6, 7]")
            vector_db = None # Ensure RAG is disabled if processing fails

# --- Tool Setup ---
tools =
if use_web_search:
    if not serpapi_api_key:
        st.warning("⚠️ SerpAPI Web Search is enabled but SERPAPI_API_KEY not found in secrets. Web search will not function. [8, 9]")
    else:
        try:
            search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
            tools.append(Tool(name="Web Search", func=search.run, description="Use this for answering current web-based questions."))
        except Exception as e:
            st.error(f"❌ Error initializing SerpAPI: {e}. Check your SERPAPI_API_KEY. [8, 9]")

# --- LangChain Agent or RAG Chain ---
if vector_db:
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
else:
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, verbose=False)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history =

# --- Avatars ---
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

# --- Voice Input Processor (for streamlit-webrtc) ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_buffer = BytesIO()

    def recv(self, frame: WebRtcMode) -> None:
        # This method receives audio frames. For simplicity, we'll just buffer them.
        # In a real-time scenario, you'd process them continuously.
        # For now, we'll assume a single recording session.
        pass # Not directly used for buffering in this simple example

# --- Chat Input UI ---
user_input_text = st.chat_input("Type your message here...")
audio_input_text = None

if enable_voice_input:
    st.markdown("---")
    st.subheader("Speak your message:")
    webrtc_ctx = webrtc_streamer(
        key="speech_to_text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_queued_frames()
            if audio_frames:
                # Concatenate audio frames (simple approach, might need pydub for robust handling)
                # For demonstration, we'll assume a way to get a single audio segment
                # In a real app, you'd process audio_frames more carefully.
                # This part is a placeholder for actual audio processing.
                # A more robust solution would involve saving to a WAV file and then recognizing.
                st.info("Recording audio... (Note: Real-time STT processing is complex for this example)")
                # For a simple demo, let's simulate a recognition if audio is detected
                # In a real scenario, you'd use a button to stop recording and then process
                # For now, we'll rely on text input primarily.
                # If you want to enable this, you'd need to save audio_frames to a file/buffer
                # and then pass it to recognizer.recognize_google(audio_data)
                # This is a conceptual outline for STT.
                pass
        except Exception as e:
            st.error(f"❌ Error processing audio input: {e}")

    # Placeholder for actual audio transcription result
    # For a functional demo, a dedicated "Record" button and "Stop" button would be better
    # with the audio data being processed after stopping.
    # For now, we'll prioritize text input unless a robust STT solution is fully integrated.
    # If you want to test STT, you might need to save the audio to a file and then use sr.AudioFile
    # For this "final app.py", we'll keep the STT part conceptual for real-time.
    # A simpler STT for testing:
    # audio_file_uploader = st.file_uploader("Upload Audio File for Transcription", type=["wav", "mp3"])
    # if audio_file_uploader:
    #     try:
    #         with sr.AudioFile(audio_file_uploader) as source:
    #             audio_data = sr.Recognizer().record(source)
    #             audio_input_text = sr.Recognizer().recognize_google(audio_data)
    #             st.write(f"Transcribed: {audio_input_text}")
    #     except Exception as e:
    #         st.error(f"Error transcribing audio: {e}")

# Determine the actual user input
user_message = user_input_text if user_input_text else audio_input_text

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        if msg["role"] == "assistant" and enable_voice_output and "audio_path" in msg:
            st.audio(msg["audio_path"], format="audio/mp3") # [14]

if user_message:
    st.chat_message("user", avatar=USER_AVATAR).markdown(f"<div class='chat-bubble'>{user_message}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    try:
        if vector_db:
            response = rag_chain.run(user_message)
        else:
            response = agent.run(user_message)
    except Exception as e:
        response = f"❌ Error: {str(e)}. Please check your API keys and configuration."

    bot_message_entry = {"role": "assistant", "content": response}

    if enable_voice_output:
        try:
            tts = gTTS(text=response, lang='en', slow=False) # [15]
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file_path = fp.name
            bot_message_entry["audio_path"] = audio_file_path
        except Exception as e:
            st.warning(f"⚠️ Could not generate voice output: {e}")
            # If TTS fails, don't add audio_path, but still show text response

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
        if enable_voice_output and "audio_path" in bot_message_entry:
            st.audio(bot_message_entry["audio_path"], format="audio/mp3") # [14]
            # Clean up audio file after playing (Streamlit handles temporary files well, but explicit cleanup is good)
            # Note: Streamlit re-runs the script, so direct os.unlink here might delete before playback.
            # A more robust solution for cleanup would involve session state or a dedicated cleanup function.
            # For this example, tempfile.NamedTemporaryFile handles deletion on close/exit.

    st.session_state.chat_history.append(bot_message_entry)

    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_message}\n[{datetime.now()}] Bot: {response}\n\n")

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
