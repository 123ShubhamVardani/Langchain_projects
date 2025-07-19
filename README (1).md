
# 💬 LangChain + Groq Chatbot

An intelligent Streamlit-based chatbot using **LangChain**, **Groq LLMs**, and optional **SerpAPI** web search.

### 🚀 Features

- 🔁 **Automatic Model Selection** with fallback:
  - `llama3-70b-8192`
  - `llama3-8b-8192`
  - `mistral-saba-24b`
- 🧠 Conversational memory with avatar-based chat UI
- 🌐 Web search powered by SerpAPI (optional toggle)
- ⚙️ Adjustable temperature slider for creativity
- 📁 Chat history log with download option
- 🖥️ Clean Streamlit UI with custom styling

---

### 🛠️ Setup

#### 1. Clone the repository
```bash
git clone https://github.com/123ShubhamVardani/Langchain_projects.git
cd Langchain_projects
```

#### 2. Create a virtual environment and install dependencies
```bash
uv venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

#### 3. Add a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_key   # Optional
```

#### 4. Run the app
```bash
streamlit run app.py
```

---

### 🌍 Live Demo

> 📡 [Click here to try the chatbot live](https://share.streamlit.io/your-deployment-url)

---

### 🧠 Built With

- [LangChain](https://python.langchain.com/)
- [Groq LLMs](https://console.groq.com/)
- [Streamlit](https://streamlit.io/)
- [SerpAPI](https://serpapi.com/) *(optional)*

---

### 📸 UI Preview

| Chat Interface | Settings Sidebar |
|----------------|------------------|
| ![chat](assets/chat.png) | ![sidebar](assets/sidebar.png) |

---

### 🙏 Acknowledgements

Built by **Shubham Vardani**  
Follow me on [LinkedIn](https://linkedin.com/in/shubhamvardani) | [GitHub](https://github.com/123ShubhamVardani)
