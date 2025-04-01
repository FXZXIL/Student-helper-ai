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
from manim import *
import re

# Set Streamlit page config
st.set_page_config(page_title="Isha.AI", page_icon="🤖", layout="wide")

# Initialize speech recognizer and engine
recognizer = sr.Recognizer()
speech_engine = pyttsx3.init()
speech_engine.setProperty("rate", 150)
speech_engine.setProperty("volume", 1.0)

# File path for JSON data
JSON_FILE_PATH = "C:/Users/hp/Downloads/wincept_finetune.json"

# Load JSON data once
@st.cache_data
def load_json_data(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return []

# Save JSON data (buffered)
def save_json_data(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Cache TF-IDF vectors
@st.cache_resource
def get_tfidf_vectors(_qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(_qa_pairs.keys())
    return vectorizer, question_vectors

# Initial data load
json_data = load_json_data(JSON_FILE_PATH)
qa_pairs = {item["input"].strip().lower(): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.latest_response = None
    st.session_state.latest_animation_path = None
    st.session_state.json_buffer = json_data
    st.session_state.animation_cache = {}

# Sidebar with chat history
with st.sidebar:
    st.title("Isha.AI 🤖")
    st.markdown("---")
    st.markdown("### Chat History")
    for idx, message in enumerate(st.session_state.messages):
        role = "You" if message["role"] == "user" else "Isha"
        content = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
        source_tag = "[Reasoning]" if "[Reasoning]" in message["content"] else ""
        st.markdown(f"<div class='history-item'><span class='history-role'>{role}:</span> {content.replace('[Reasoning]', '')} <span class='source-tag'>{source_tag}</span></div>", unsafe_allow_html=True)

# Cache LLM responses
@lru_cache(maxsize=100)
def cached_llm_response(prompt, use_reasoning, history_hash):
    messages = [{'role': 'user', 'content': prompt}]
    if history_hash:
        messages.insert(0, {'role': 'system', 'content': f"Conversation History:\n{history_hash}"})
    try:
        response = ollama.chat(model="gemma3:1b", messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Auto-train new QA pair (buffered)
def auto_train_query(query, response, similarity_score=None):
    global qa_pairs, vectorizer, question_vectors
    query_normalized = query.strip().lower()
    
    if query_normalized in qa_pairs:
        return
    
    confidence_threshold = 0.7
    if similarity_score is None or similarity_score < confidence_threshold:
        new_entry = {"input": query_normalized, "output": response, "timestamp": datetime.now().isoformat(), "source": "Auto-Train"}
        st.session_state.json_buffer.append(new_entry)
        qa_pairs[query_normalized] = response
        if len(st.session_state.json_buffer) % 10 == 0:
            get_tfidf_vectors.clear()
            vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)
            save_json_data(JSON_FILE_PATH, st.session_state.json_buffer)

# Generate enhanced Manim animation
def create_manim_animation(response_text):
    response_hash = hashlib.md5(response_text.encode()).hexdigest()
    output_path = f"media/llm_animation_{response_hash}.mp4"
    
    # Check cache
    if response_hash in st.session_state.animation_cache:
        return st.session_state.animation_cache[response_hash]
    
    try:
        class EnhancedResponseScene(Scene):
            def construct(self):
                self.camera.background_color = "#1e1e1e"
                response_length = len(response_text)
                
                # Fixed 3-second timing: 0.8s fade-in, 1.4s hold with effect, 0.8s fade-out
                fade_in_time = 0.8
                hold_time = 1.4
                fade_out_time = 0.8
                
                # Detect content type
                math_match = re.match(r"(\d+)\s*[\+\-\*/]\s*(\d+)\s*equals\s*(\d+)", response_text.lower())
                list_match = "\n- " in response_text or response_text.startswith("- ")
                
                if math_match:  # Enhanced math animation
                    num1, num2, result = math_match.groups()
                    operator = response_text[math_match.start(1) + len(num1):math_match.start(2)].strip()
                    font_size = min(50, 60 - len(f"{num1} {operator} {num2} = {result}"))
                    eq = MathTex(f"{num1} {operator} {num2}", font_size=font_size, color=YELLOW).move_to(LEFT * 2)
                    equals = MathTex("=", font_size=font_size, color=WHITE).next_to(eq, RIGHT)
                    result_tex = MathTex(result, font_size=font_size, color=GREEN).next_to(equals, RIGHT)
                    group = VGroup(eq, equals, result_tex)
                    self.play(Write(eq), run_time=fade_in_time)
                    self.play(FadeIn(equals), FadeIn(result_tex), run_time=fade_in_time / 2)
                    self.play(group.animate.scale(1.1).set_color(ORANGE), run_time=hold_time / 2)  # Bounce effect
                    self.play(group.animate.scale(0.9), run_time=hold_time / 2)
                    self.play(FadeOut(group), run_time=fade_out_time)
                
                elif list_match:  # Enhanced list animation
                    items = [line.strip("- ").strip() for line in response_text.split("\n") if line.strip()]
                    num_items = min(len(items), 3)
                    font_size = max(22, 36 - num_items * 4)
                    texts = [Text(f"• {item[:25]}...", font_size=font_size, color=GREEN) for item in items[:num_items]]
                    group = VGroup(*texts).arrange(DOWN, center=True, buff=0.5)
                    for text in texts:
                        self.play(Write(text), run_time=fade_in_time / num_items)
                    self.play(group.animate.shift(UP * 0.2).set_color(YELLOW), run_time=hold_time / 2)  # Subtle lift
                    self.play(group.animate.shift(DOWN * 0.2), run_time=hold_time / 2)
                    self.play(FadeOut(group), run_time=fade_out_time)
                
                else:  # Enhanced text animation
                    lines = response_text.split(". ")[:2]
                    num_lines = len(lines)
                    font_size = max(26, 42 - response_length // 10)
                    texts = [Text(line[:30] + "..." if len(line) > 30 else line, font_size=font_size, color=YELLOW) 
                             for line in lines]
                    group = VGroup(*texts).arrange(DOWN, center=True, buff=0.5)
                    self.play(LaggedStart(*[FadeIn(text) for text in texts], lag_ratio=0.2), run_time=fade_in_time)
                    self.play(group.animate.set_color(GREEN).scale(1.05), run_time=hold_time / 2)  # Pulse effect
                    self.play(group.animate.scale(0.95), run_time=hold_time / 2)
                    self.play(FadeOut(group), run_time=fade_out_time)
        
        config.quality = "medium_quality"  # Improved quality for better visuals
        scene = EnhancedResponseScene()
        scene.render()
        import shutil
        default_output = scene.renderer.file_writer.movie_file_path
        shutil.move(default_output, output_path)
        st.session_state.animation_cache[response_hash] = output_path
        return output_path
    except Exception as e:
        st.error(f"Animation failed: {str(e)}")
        return None

# Retrieve and blend answer
def retrieve_and_blend_answer(query, use_reasoning=False):
    query_normalized = query.strip().lower()
    
    if query_normalized in qa_pairs:
        return qa_pairs[query_normalized]
    
    similarity_score = None
    if vectorizer and question_vectors is not None:
        query_vector = vectorizer.transform([query_normalized])
        similarities = cosine_similarity(query_vector, question_vectors)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0][most_similar_idx]
        if similarity_score >= 0.7:
            return list(qa_pairs.values())[most_similar_idx]
    
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]) if st.session_state.messages else ""
    history_hash = str(hash(history))
    prompt = f"For the query '{query}', provide a detailed, step-by-step reasoning process." if use_reasoning else query
    response = cached_llm_response(prompt, use_reasoning, history_hash)
    auto_train_query(query, response, similarity_score)
    return response

# Speech to text with timeout
def speech_to_text():
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return recognizer.recognize_google(audio)
    except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError) as e:
        st.error(f"Speech error: {str(e)}")
        return None

# Speak response
def speak_response(text):
    try:
        speech_engine.say(text)
        speech_engine.runAndWait()
    except Exception as e:
        st.error(f"Error speaking: {str(e)}")

# Save conversation history
def save_history_after_interaction(query, response):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.latest_response = response
    st.session_state.latest_animation_path = None

# Custom CSS
st.markdown(
    """
    <style>
        .stApp { background: #1e1e1e; color: #d0d0d0; font-family: Arial; padding: 20px; }
        .chat-container { max-width: 1200px; margin: 0 auto; flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-end; }
        .message-bubble { padding: 16px; border-radius: 8px; margin-bottom: 10px; max-width: 85%; background: #2e2e2e; font-size: 16px; }
        .user-message { background: #1e90ff; color: white; align-self: flex-end; }
        .assistant-message { background: #2e2e2e; color: #d0d0d0; align-self: flex-start; }
        .stTextInput input { background: #2e2e2e; border: 1px solid #404040; border-radius: 8px; padding: 12px; color: #d0d0d0; }
        .stButton button { background: #1e90ff; color: white; border: none; border-radius: 8px; padding: 10px; }
        .input-bar-container { display: flex; gap: 10px; background: #252525; padding: 15px; border-top: 1px solid #404040; position: fixed; bottom: 0; left: 0; right: 0; max-width: 1200px; margin: 0 auto; }
        .output-container { margin-bottom: 150px; overflow-y: auto; }
        .typing-indicator { display: flex; padding: 16px; border-radius: 8px; margin-bottom: 10px; max-width: 85%; background: #2e2e2e; }
        .typing-indicator span { width: 10px; height: 10px; background: #1e90ff; border-radius: 50%; margin-right: 6px; animation: typing 1s infinite; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }
        .video-container { margin-top: 10px; }
        video { max-height: 400px; border-radius: 8px; border: 1px solid #404040; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("<div class='header'><h1 style='color: #1e90ff;'>Isha.AI</h1></div>", unsafe_allow_html=True)

output_container = st.markdown("<div class='output-container'>", unsafe_allow_html=True)
chat_display = st.empty()
typing_container = st.empty()

def update_chat_display():
    with chat_display.container():
        for message in st.session_state.messages[-2:]:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            content = message["content"].replace("[Reasoning]", "")
            source_tag = "<span style='font-size: 12px; color: #1e90ff;'>[Reasoning]</span>" if "[Reasoning]" in message["content"] else ""
            st.markdown(f"<div class='message-bubble {role_class}'>{content}{source_tag}</div>", unsafe_allow_html=True)
        if st.session_state.latest_animation_path:
            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
            st.video(st.session_state.latest_animation_path)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='input-bar-container'>", unsafe_allow_html=True)
query = st.text_input("Ask me anything...", key="query_input")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Send", key="send_button") and query:
        with chat_display.container():
            st.markdown(f"<div class='message-bubble user-message'>{query}</div>", unsafe_allow_html=True)
        typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
        response = retrieve_and_blend_answer(query)
        typing_container.empty()
        save_history_after_interaction(query, response)
        update_chat_display()
with col2:
    if st.button("🎤 Voice", key="voice_button"):
        query = speech_to_text()
        if query:
            with chat_display.container():
                st.markdown(f"<div class='message-bubble user-message'>{query}</div>", unsafe_allow_html=True)
            typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
            response = retrieve_and_blend_answer(query)
            typing_container.empty()
            speak_response(response.replace("[Reasoning]", ""))
            save_history_after_interaction(query, response)
            update_chat_display()
with col3:
    if st.button("🤓 Reason", key="reasoning_button") and query:
        with chat_display.container():
            st.markdown(f"<div class='message-bubble user-message'>{query}</div>", unsafe_allow_html=True)
        typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
        response = f"[Reasoning] {retrieve_and_blend_answer(query, use_reasoning=True)}"
        typing_container.empty()
        save_history_after_interaction(query, response)
        update_chat_display()
with col4:
    if st.button("🎬 Animate", key="animate_button") and st.session_state.latest_response:
        typing_container.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
        animation_path = create_manim_animation(st.session_state.latest_response)
        if animation_path:
            st.session_state.latest_animation_path = animation_path
            typing_container.empty()
            update_chat_display()
        else:
            typing_container.empty()
            with chat_display.container():
                st.markdown(f"<div class='assistant-message'>{st.session_state.latest_response} (Animation unavailable)</div>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)
