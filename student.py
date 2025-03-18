import streamlit as st
import ollama
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import speech_recognition as sr
import pyttsx3
import os
import hashlib
from datetime import datetime
import platform
import wmi
import pytesseract
from PIL import Image
import pdfplumber
from io import BytesIO
import re

# Initialize speech recognizer
recognizer = sr.Recognizer()

# File paths for JSON data and user history
JSON_FILE_PATH = "C:/Users/hp/Downloads/student_helper_finetune.json"
USERS_FILE_PATH = "C:/Users/hp/Downloads/users.json"

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Get CPU model (for system-related queries)
def get_cpu_model():
    try:
        if os.name == "nt":
            c = wmi.WMI()
            for cpu in c.Win32_Processor():
                return cpu.Name
        else:
            return platform.processor() or "Unknown CPU"
    except Exception as e:
        return f"Error detecting CPU: {str(e)}"

# Normalize query for consistent matching
def normalize_query(query):
    return re.sub(r'\s+', ' ', query.strip().lower()).strip()

# Load JSON data (QA pairs)
def load_json_data(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return []

# Save updated data to JSON
def save_json_data(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Load users data
def load_users():
    try:
        if os.path.exists(USERS_FILE_PATH):
            with open(USERS_FILE_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return {}

# Save users data
def save_users(users):
    try:
        with open(USERS_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(users, file, indent=4)
    except Exception as e:
        st.error(f"Error saving users: {str(e)}")

# Load user-specific conversation history
def load_user_conversation_history(username):
    history_file_path = f"C:/Users/hp/Downloads/user_{username}_history.json"
    try:
        if os.path.exists(history_file_path):
            with open(history_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("messages", []), data.get("user_info", {})
        return [], {}
    except Exception as e:
        st.error(f"Error loading user history: {str(e)}")
        return [], {}

# Save user-specific conversation history
def save_user_conversation_history(username, messages, user_info):
    history_file_path = f"C:/Users/hp/Downloads/user_{username}_history.json"
    try:
        data = {"messages": messages, "user_info": user_info}
        with open(history_file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving user history: {str(e)}")

# Cache the TF-IDF vectorizer and vectors
@st.cache_resource
def get_tfidf_vectors(qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_pairs.keys())
    return vectorizer, question_vectors

# Load initial datasets
json_data = load_json_data(JSON_FILE_PATH)
qa_pairs = {normalize_query(item["input"]): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

# Authentication logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.user_info = {}

# Load users
users = load_users()

# Sidebar with authentication
with st.sidebar:
    st.title("StudentHelper.AI 📚")
    st.markdown("---")
    if not st.session_state.logged_in:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_username in users and users[login_username]["password"] == hash_password(login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.session_state.messages, st.session_state.user_info = load_user_conversation_history(login_username)
                st.success(f"Welcome back, {login_username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        
        st.subheader("Register")
        reg_username = st.text_input("New Username", key="reg_username")
        reg_password = st.text_input("New Password", type="password", key="reg_password")
        if st.button("Register"):
            if reg_username in users:
                st.error("Username already exists.")
            elif reg_username and reg_password:
                users[reg_username] = {"password": hash_password(reg_password)}
                save_users(users)
                st.success("Registration successful! Please login.")
            else:
                st.error("Please fill in all fields.")
    else:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.user_info = {}
            st.success("Logged out successfully!")
            st.rerun()

        st.markdown("### Chat History")
        for idx, message in enumerate(st.session_state.messages):
            role = "You" if message["role"] == "user" else "Helper"
            content = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
            source_tag = "[Explanation]" if "[Explanation]" in message["content"] else "[File]" if "[File]" in message["content"] else ""
            st.markdown(f"<div class='history-item'><span class='history-role'>{role}:</span> {content.replace('[Explanation]', '').replace('[File]', '')} <span class='source-tag'>{source_tag}</span></div>", unsafe_allow_html=True)

# Extract text from image
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
        if text.strip():
            return text.strip()
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=300)
                text += pytesseract.image_to_string(img.original) + "\n"
        return text.strip() if text.strip() else "No text could be extracted from the PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Get PDF thumbnail
def get_pdf_thumbnail(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            first_page = pdf.pages[0]
            img = first_page.to_image(resolution=150)
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            return img_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF thumbnail: {str(e)}")
        return None

# Update user info based on query
def update_user_info(query):
    grade_match = re.search(r"my grade is (\d+)", query.lower())
    subject_match = re.search(r"i like (\w+)", query.lower())
    if grade_match:
        st.session_state.user_info["grade"] = int(grade_match.group(1))
    if subject_match:
        st.session_state.user_info["favorite_subject"] = subject_match.group(1).capitalize()

# Auto-train new QA pair
def auto_train_query(query, response, similarity_score=None):
    global qa_pairs, vectorizer, question_vectors, json_data
    query_normalized = normalize_query(query)
    
    if query_normalized in qa_pairs:
        return
    
    confidence_threshold = 0.6
    should_train = (similarity_score is None or similarity_score < confidence_threshold)
    
    if should_train:
        new_entry = {
            "input": query_normalized,
            "output": response,
            "timestamp": datetime.now().isoformat(),
            "source": "Auto-Train"
        }
        json_data.append(new_entry)
        qa_pairs[query_normalized] = response
        save_json_data(JSON_FILE_PATH, json_data)
        get_tfidf_vectors.clear()
        vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)

# Retrieve and blend answer
def retrieve_and_blend_answer(query, use_reasoning=False, image_file=None, pdf_file=None, threshold=0.6):
    query_normalized = normalize_query(query)
    context = ""
    
    update_user_info(query)
    
    if query_normalized == "cpu model" or query_normalized == "what is my cpu":
        return f"Your CPU model is: {get_cpu_model()}"
    
    if query_normalized in qa_pairs:
        return qa_pairs[query_normalized]
    
    if image_file:
        image_text = extract_text_from_image(image_file)
        context += f"\nHomework image content: {image_text}"
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        context += f"\nHomework PDF content: {pdf_text}"
    
    similarity_score = None
    best_match = None
    if question_vectors is not None:
        query_vector = vectorizer.transform([query_normalized])
        similarities = cosine_similarity(query_vector, question_vectors)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0][most_similar_idx]
        
        words = query_normalized.split()
        adjusted_threshold = 0.3 if len(words) == 1 else threshold
        
        if similarity_score >= adjusted_threshold:
            best_match = list(qa_pairs.values())[most_similar_idx]
            best_query = list(qa_pairs.keys())[most_similar_idx]
            if similarity_score > 0.8 or query_normalized == best_query:
                return best_match
            else:
                context += f"\nSimilar answer: {best_match}"
    
    response = generate_llm_response(query, context, use_reasoning)
    
    if not (use_reasoning or image_file or pdf_file):
        auto_train_query(query, response, similarity_score)
    
    return response

# Generate response from LLM
def generate_llm_response(prompt, context=None, use_reasoning=False):
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]) if st.session_state.messages else ""
    user_info_str = ""
    if st.session_state.user_info:
        if "grade" in st.session_state.user_info:
            user_info_str += f"User's grade: {st.session_state.user_info['grade']}\n"
        if "favorite_subject" in st.session_state.user_info:
            user_info_str += f"User's favorite subject: {st.session_state.user_info['favorite_subject']}\n"
    
    full_context = f"{user_info_str}Conversation History:\n{conversation_history}\n\nCurrent Context:\n{context or ''}" if (context or conversation_history or user_info_str) else ""

    if use_reasoning:
        prompt = (
            f"For the query '{prompt}', explain it step-by-step like I’m a student. Use simple words and number the steps to show your thinking. End with the answer."
        )
    elif "summarize" in prompt.lower():
        prompt = f"Summarize this in simple words for a student:\n\n{full_context}\n\nKeep it short and clear."
    else:
        prompt = (
            f"Answer the query '{prompt}' like a friendly teacher. Use simple words for a student. "
            "If I gave my grade or favorite subject, make it fit me. Use any homework info or past chats to help."
        )

    messages = [{'role': 'user', 'content': prompt}]
    if full_context:
        messages.insert(0, {'role': 'system', 'content': full_context})
    try:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Generate a quiz
def generate_quiz(query, subject):
    if "math" in subject.lower():
        return "Here’s a quick Math quiz:\n1. What is 5 + 3?\n2. What is 10 - 4?\n3. What is 2 × 3?\nWrite your answers and check with me!"
    elif "science" in subject.lower():
        return "Science quiz time:\n1. What gas do plants need to grow?\n2. What is H₂O?\n3. What planet do we live on?\nTry answering!"
    elif "english" in subject.lower():
        return "English quiz:\n1. What’s a noun?\n2. Give me a verb.\n3. Make a sentence.\nGo for it!"
    elif "history" in subject.lower():
        return "History quiz:\n1. Who was the first US president?\n2. When did World War II end?\n3. What’s the Declaration of Independence?\nTest yourself!"
    else:
        return f"Ask me a {subject} question like 'What is a fraction?' or 'Who was Cleopatra?' and I’ll make a quiz!"

# Convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn’t hear you clearly.")
            return None
        except sr.RequestError:
            st.error("Oops, there’s a problem with the speech service.")
            return None

# Speak the response
def speak_response(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error speaking: {str(e)}")

# Update JSON and retrain
def update_json_and_retrain(query, response):
    global qa_pairs, vectorizer, question_vectors, json_data
    query_normalized = normalize_query(query)
    source = "User" if "[Explanation]" not in response and "[File]" not in response else \
             ("Explanation" if "[Explanation]" in response else "File")
    
    if query_normalized not in qa_pairs:
        new_entry = {
            "input": query_normalized,
            "output": response,
            "timestamp": datetime.now().isoformat(),
            "source": source
        }
        json_data.append(new_entry)
        save_json_data(JSON_FILE_PATH, json_data)
        
        qa_pairs[query_normalized] = response
        get_tfidf_vectors.clear()
        vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)

# Save conversation history
def save_history_after_interaction(query, response):
    if st.session_state.logged_in:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_user_conversation_history(st.session_state.username, st.session_state.messages, st.session_state.user_info)

# Dark Theme CSS
st.markdown(
    """
    <style>
        body { 
            background-color: #1e1e1e; 
            color: #d0d0d0; 
            font-family: 'Comic Sans MS', 'Arial', sans-serif; 
        }
        .stApp { 
            background-color: #1e1e1e; 
            padding: 20px; 
            height: 100vh; 
            display: flex; 
            flex-direction: column; 
        }
        .chat-container { 
            max-width: 800px; 
            margin: 0 auto; 
            flex-grow: 1; 
            display: flex; 
            flex-direction: column; 
            justify-content: flex-end; 
        }
        .message-bubble { 
            padding: 12px 16px; 
            border-radius: 12px; 
            margin-bottom: 10px; 
            max-width: 70%; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); 
        }
        .user-message { 
            background: #ff6347; 
            color: white; 
            align-self: flex-end; 
        }
        .assistant-message { 
            background: #333333; 
            color: #d0d0d0; 
            align-self: flex-start; 
        }
        .stTextInput input { 
            background-color: #2e2e2e; 
            border: 2px solid #ff6347; 
            border-radius: 10px; 
            padding: 10px; 
            font-size: 14px; 
            color: #d0d0d0; 
        }
        .stTextInput input:focus { 
            border-color: #ff4500; 
            box-shadow: 0 0 4px rgba(255, 69, 0, 0.5); 
        }
        .stButton button { 
            background-color: #ff6347; 
            color: white; 
            border: none; 
            border-radius: 10px; 
            padding: 8px 12px; 
            font-size: 12px; 
            margin-left: 5px; 
        }
        .stButton button:hover { 
            background-color: #ff4500; 
        }
        .header { 
            text-align: center; 
            padding: 10px 0; 
            border-bottom: 2px solid #ff6347; 
        }
        .header h1 { 
            font-size: 28px; 
            color: #ff6347; 
            margin: 0; 
        }
        .header p { 
            font-size: 14px; 
            color: #a0a0a0; 
            margin: 5px 0 0; 
        }
        .input-bar-container { 
            display: flex; 
            align-items: center; 
            gap: 10px; 
            background: #2e2e2e; 
            padding: 10px; 
            border-top: 2px solid #ff6347; 
            position: fixed; 
            bottom: 0; 
            left: 0; 
            right: 0; 
            max-width: 800px; 
            margin: 0 auto; 
        }
        .output-container { 
            margin-bottom: 100px; 
        }
        .source-tag { 
            font-size: 10px; 
            color: #ff6347; 
            margin-left: 8px; 
        }
        .sidebar .sidebar-content { 
            background: #2e2e2e; 
            padding: 15px; 
            border-right: 2px solid #ff6347; 
        }
        .history-item { 
            padding: 10px; 
            border-bottom: 1px solid #ff6347; 
            font-size: 13px; 
            color: #d0d0d0; 
        }
        .history-item:hover { 
            background: #3e3e3e; 
        }
        .history-role { 
            font-weight: bold; 
            color: #ff6347; 
        }
        .upload-container { 
            background: #2e2e2e; 
            padding: 15px; 
            border-radius: 10px; 
            border: 2px solid #ff6347; 
            margin-bottom: 10px; 
        }
        .stFileUploader > div > div > div > div { 
            background-color: #2e2e2e; 
            border: 2px solid #ff6347; 
            border-radius: 10px; 
            padding: 10px; 
            color: #d0d0d0; 
        }
        .preview-image { 
            max-width: 200px; 
            border-radius: 10px; 
            border: 2px solid #ff6347; 
        }
        .typing-indicator { 
            display: flex; 
            align-items: center; 
            padding: 12px 16px; 
            border-radius: 12px; 
            margin-bottom: 10px; 
            max-width: 70%; 
            background: #333333; 
            color: #d0d0d0; 
            align-self: flex-start; 
        }
        .typing-indicator span { 
            display: inline-block; 
            width: 8px; 
            height: 8px; 
            background-color: #ff6347; 
            border-radius: 50%; 
            margin-right: 4px; 
            animation: typing 1s infinite; 
        }
        .typing-indicator span:nth-child(2) { 
            animation-delay: 0.2s; 
        }
        .typing-indicator span:nth-child(3) { 
            animation-delay: 0.4s; 
        }
        @keyframes typing { 
            0% { opacity: 0.3; transform: translateY(0); } 
            50% { opacity: 1; transform: translateY(-5px); } 
            100% { opacity: 0.3; transform: translateY(0); } 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main UI
if st.session_state.logged_in:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Header
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown(f"<h1>StudentHelper.AI - Hi, {st.session_state.username}!</h1>", unsafe_allow_html=True)
    st.markdown("<p>Your School Study Buddy</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Subject selection
    subject = st.selectbox("Pick a Subject", ["Math", "Science", "English", "History", "General"])

    # File upload for homework
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Homework (Image/PDF)", type=["jpg", "png", "pdf"], key="file_uploader")
    if uploaded_file:
        st.image(uploaded_file if uploaded_file.type in ["image/jpeg", "image/png"] else get_pdf_thumbnail(uploaded_file), 
                 caption="Homework Preview", width=200)
        if st.button("Clear File"):
            st.session_state.pop("file_uploader", None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat output
    st.markdown("<div class='output-container'>", unsafe_allow_html=True)
    typing_container = st.empty()
    if st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        role_class = "user-message" if latest_message["role"] == "user" else "assistant-message"
        content = latest_message["content"]
        source_tag = ""
        if "[Explanation]" in content:
            source_tag = "<span class='source-tag'>[Explanation]</span>"
            content = content.replace("[Explanation]", "")
        elif "[File]" in content:
            source_tag = "<span class='source-tag'>[File]</span>"
            content = content.replace("[File]", "")
        st.markdown(f"<div class='message-bubble {role_class}'>{content}{source_tag}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input bar
    st.markdown("<div class='input-bar-container'>", unsafe_allow_html=True)
    query = st.text_input("Ask me anything about school!", key="query_input", placeholder="E.g., 'Explain fractions' or 'Help with history'")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Send"):
            if query:
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
                    response = retrieve_and_blend_answer(f"{subject}: {query}")
                    typing_container.empty()
                    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
    with col2:
        if st.button("🎤 Voice"):
            query = speech_to_text()
            if query:
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
                    response = retrieve_and_blend_answer(f"{subject}: {query}")
                    typing_container.empty()
                    speak_response(response.replace("[Explanation]", "").replace("[File]", ""))
                    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
    with col3:
        if st.button("🤓 Explain"):
            if query:
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
                    response = retrieve_and_blend_answer(f"{subject}: {query}", use_reasoning=True)
                    response = f"[Explanation] {response}"
                    typing_container.empty()
                    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                    save_history_after_interaction(query, response)
                    update_json_and_retrain(query, response)
                    st.rerun()
    with col4:
        if st.button("📝 Quiz"):
            if query:
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
                    response = generate_quiz(query, subject)
                    typing_container.empty()
                    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                    save_history_after_interaction(query, response)
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-container'><h2>Please log in to use StudentHelper.AI</h2></div>", unsafe_allow_html=True)