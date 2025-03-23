import streamlit as st
import ollama
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import os
from datetime import datetime, timedelta
import platform
import wmi
import pandas as pd
import plotly.express as px
from io import StringIO
import time
from deep_translator import GoogleTranslator

# File paths
JSON_FILE_PATH = "C:/Users/hp/Downloads/student_finetune.json"
HISTORY_FILE_PATH = "C:/Users/hp/Downloads/student_history.json"

# Language dictionaries (unchanged)
translations = {
    "en": {
        "title": "Student.AI - Learn Easy",
        "subtitle": "Your ultimate study companion with quizzes, mock tests, analytics, and detailed planning!",
        "grade_label": "Grade Level",
        "subjects_label": "Favorite Subjects",
        "study_tab": "Study",
        "quiz_tab": "Quiz/Test",
        "results_tab": "Results",
        "plan_tab": "Study Plan",
        "ask_placeholder": "e.g., 'Solve 2x + 3 = 7' or 'What is photosynthesis?'",
        "answer_button": "Answer",
        "tip_button": "📚 Tip",
        "single_quiz_button": "❓ Single Quiz",
        "mock_test_button": "📝 Mock Test",
        "num_questions_label": "Number of Questions",
        "difficulty_label": "Difficulty",
        "submit_answer": "Submit Answer",
        "progress": "Progress",
        "daily_schedule": "Daily Schedule",
        "mark_complete": "Mark Tasks Complete",
        "generate_plan": "Generate Study Plan",
        "no_plan": "Add deadlines and tasks, then click 'Generate Study Plan' to see your schedule!",
        "invalid_plan": "Study plan is empty or missing required data (e.g., start_time). Please generate a new plan.",
        "next_step": "Next Step",
        "all_complete": "All tasks completed! Great job!"
    },
    "ml": {
        "title": "സ്റ്റുഡന്റ്.എഐ - എളുപ്പത്തിൽ പഠിക്കുക",
        "subtitle": "നിന്റെ പഠന സഹായിയായി ക്വിസുകൾ, മോക്ക് ടെസ്റ്റുകൾ, വിശകലനം, വിശദമായ ആസൂത്രണം എന്നിവയോടെ!",
        "grade_label": "ഗ്രേഡ് ലെവൽ",
        "subjects_label": "പ്രിയപ്പെട്ട വിഷയങ്ങൾ",
        "study_tab": "പഠനം",
        "quiz_tab": "ക്വിസ്/ടെസ്റ്റ്",
        "results_tab": "ഫലങ്ങൾ",
        "plan_tab": "പഠന പദ്ധതി",
        "ask_placeholder": "ഉദാ: '2x + 3 = 7 എങ്ങനെ പരിഹരിക്കാം?' അല്ലെങ്കിൽ 'പ്രകാശസംശ്ലേഷണം എന്താണ്?'",
        "answer_button": "ഉത്തരം",
        "tip_button": "📚 നുറുങ്ങ്",
        "single_quiz_button": "❓ ഒറ്റ ക്വിസ്",
        "mock_test_button": "📝 മോക്ക് ടെസ്റ്റ്",
        "num_questions_label": "പ്രശ്നങ്ങളുടെ എണ്ണം",
        "difficulty_label": "കാഠിന്യം",
        "submit_answer": "ഉത്തരം സമർപ്പിക്കുക",
        "progress": "പുരോഗതി",
        "daily_schedule": "ദൈനംദിന ഷെഡ്യൂൾ",
        "mark_complete": "പൂർത്തിയാക്കിയവ അടയാളപ്പെടുത്തുക",
        "generate_plan": "പഠന പദ്ധതി തയ്യാറാക്കുക",
        "no_plan": "അവസാന തീയതികളും ടാസ്കുകളും ചേർത്ത ശേഷം 'പഠന പദ്ധതി തയ്യാറാക്കുക' ക്ലിക്ക് ചെയ്യുക!",
        "invalid_plan": "പഠന പദ്ധതി ശൂന്യമാണ് അല്ലെങ്കിൽ ആവശ്യമായ ഡാറ്റ (ഉദാ: start_time) ഇല്ല. ദയവായി ഒരു പുതിയ പദ്ധതി തയ്യാറാക്കുക。",
        "next_step": "അടുത്ത ഘട്ടം",
        "all_complete": "എല്ലാ ടാസ്കുകളും പൂർത്തിയായി! മികച്ച പ്രവർത്തനം!"
    }
}

# Normalize query
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

# Save JSON data
def save_json_data(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Load conversation history with data migration
def load_conversation_history():
    default_info = {
        "grade": "10", "subjects": ["Math", "Science"], "quiz_score": 0, "quiz_total": 0, 
        "quiz_history": [], "study_plan": []
    }
    try:
        if os.path.exists(HISTORY_FILE_PATH):
            with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                student_info = {**default_info, **data.get("student_info", {})}
                # Migrate old quiz_history entries to include score
                for entry in student_info["quiz_history"]:
                    if "score" not in entry:
                        entry["score"] = 1.0 if entry.get("correct", False) else 0.0
                for task in student_info["study_plan"]:
                    task["subject"] = task.get("subject", "Unknown")
                    task["date"] = task.get("date", datetime.now().strftime("%Y-%m-%d"))
                    task["task"] = task.get("task", "Review Notes")
                    task["details"] = task.get("details", "Complete this task")
                    task["start_time"] = task.get("start_time", "17:00")
                    task["duration"] = task.get("duration", 30)
                    task["completed"] = task.get("completed", False)
                return data.get("messages", []), student_info
        return [], default_info
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        return [], default_info

# Save conversation history
def save_conversation_history(messages, student_info):
    try:
        data = {"messages": messages, "student_info": student_info}
        with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving JSON: {str(e)}")

# Cache TF-IDF vectors
@st.cache_resource
def get_tfidf_vectors(qa_pairs):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_pairs.keys())
    return vectorizer, question_vectors

# Load initial datasets
json_data = load_json_data(JSON_FILE_PATH)
qa_pairs = {normalize_query(item["input"]): item["output"] for item in json_data} if json_data else {}
vectorizer, question_vectors = get_tfidf_vectors(qa_pairs) if qa_pairs else (None, None)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages, st.session_state.student_info = load_conversation_history()
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {"questions": [], "answers": [], "subjects": [], "difficulties": [], "current": 0, "mock_test": False, "timer_start": None, "time_limit": 0}
if "language" not in st.session_state:
    st.session_state.language = "en"

# Language selector in sidebar
with st.sidebar:
    st.header("Settings" if st.session_state.language == "en" else "ക്രമീകരണങ്ങൾ")
    language_options = {"English": "en", "മലയാളം": "ml"}
    st.session_state.language = st.selectbox("Select Language / ഭാഷ തിരഞ്ഞെടുക്കുക", list(language_options.keys()), index=0 if st.session_state.language == "en" else 1)
    lang = language_options[st.session_state.language]

# Translator setup
translator = GoogleTranslator(source='auto', target='ml')

# Get CPU model
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

# Solve math problems
def solve_math_problem(query):
    match = re.search(r"(\d+)x\s*([+-])\s*(\d+)\s*=\s*(\d+)", query.lower())
    if match:
        a, sign, b, c = int(match.group(1)), match.group(2), int(match.group(3)), int(match.group(4))
        b = -b if sign == "-" else b
        x = (c - b) / a
        steps = [
            f"Step 1: Start with the equation: {a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}",
            f"Step 2: Subtract {b} from both sides: {a}x = {c - b}",
            f"Step 3: Divide both sides by {a}: x = {x}"
        ]
        steps_str = "\n".join(steps)
        if lang == "ml":
            steps_ml = [
                f"ഘട്ടം 1: സമവാക്യത്തിൽ നിന്ന് ആരംഭിക്കുക: {a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}",
                f"ഘട്ടം 2: ഇരുവശത്തുനിന്നും {b} കുറയ്ക്കുക: {a}x = {c - b}",
                f"ഘട്ടം 3: ഇരുവശവും {a} കൊണ്ട് ഹരിക്കുക: x = {x}"
            ]
            return "\n".join(steps_ml)
        return steps_str
    return "I couldn’t understand the equation. Try something like '2x + 3 = 7'." if lang == "en" else "സമവാക്യം മനസ്സിലായില്ല. '2x + 3 = 7' പോലെ ശ്രമിക്കുക."

# Auto-train new QA pair
def auto_train_query(query, response):
    global qa_pairs, vectorizer, question_vectors, json_data
    query_normalized = normalize_query(query)
    if query_normalized not in qa_pairs:
        new_entry = {"input": query_normalized, "output": response, "timestamp": datetime.now().isoformat(), "source": "Auto-Train"}
        json_data.append(new_entry)
        qa_pairs[query_normalized] = response
        save_json_data(JSON_FILE_PATH, json_data)
        get_tfidf_vectors.clear()
        vectorizer, question_vectors = get_tfidf_vectors(qa_pairs)

# Generate LLM response with steps
def generate_llm_response(prompt, context=None):
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]]) if st.session_state.messages else ""
    full_context = f"{context or ''}\n\nConversation History:\n{conversation_history}"
    
    if "summarize" in prompt.lower():
        prompt = f"Summarize this content for a grade {st.session_state.student_info['grade']} student in numbered steps:\n\n{full_context}"
    else:
        prompt = f"Answer '{prompt}' for a grade {st.session_state.student_info['grade']} student interested in {', '.join(st.session_state.student_info['subjects'])}. Use numbered steps to explain simply."

    messages = [{'role': 'user', 'content': prompt}]
    if full_context:
        messages.insert(0, {'role': 'system', 'content': full_context})
    try:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        english_response = response['message']['content'] + "\nStep (Final): Hope that helps you learn!"
        if lang == "ml":
            malayalam_response = translator.translate(english_response)
            return malayalam_response.replace("Step", "ഘട്ടം")
        return english_response
    except Exception as e:
        return f"Error: {str(e)}" if lang == "en" else f"പിശക്: {str(e)}"

# Generate single quiz question and answer with difficulty
def generate_quiz(subject=None, difficulty="Medium"):
    subject = subject or (st.session_state.student_info["subjects"][0] if st.session_state.student_info["subjects"] else "Math")
    prompt = f"Create a {difficulty.lower()} quiz question for a grade {st.session_state.student_info['grade']} student in {subject}. Provide the question and the correct answer in this format:\nQuestion: [Your question]\nAnswer: [Correct answer]"
    response = ollama.chat(model="llama3.2:3b", messages=[{'role': 'user', 'content': prompt}])['message']['content']
    try:
        question = response.split("Question: ")[1].split("Answer: ")[0].strip()
        answer = response.split("Answer: ")[1].strip()
        if lang == "ml":
            question = translator.translate(question)
            answer = translator.translate(answer)
        return question, answer, subject, difficulty
    except:
        default_q = "What is 2 + 2?" if lang == "en" else "2 + 2 എന്താണ്?"
        default_a = "4" if lang == "en" else "നാല്"
        return default_q, default_a, "Math", "Easy"

# Generate multiple quiz questions for mock test with adaptive difficulty
def generate_mock_test(num_questions, difficulty="Medium"):
    subjects = st.session_state.student_info["subjects"] or ["Math"]
    history_df = pd.DataFrame(st.session_state.student_info["quiz_history"])
    difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    difficulties = []
    
    if not history_df.empty:
        recent_performance = history_df.tail(10)["correct"].mean()  # Use last 10 for better accuracy
        if recent_performance > 0.75 and difficulty_map[difficulty] < 2:
            difficulty = "Hard"
        elif recent_performance < 0.4 and difficulty_map[difficulty] > 0:
            difficulty = "Easy"
    
    questions, answers, quiz_subjects = [], [], []
    for i in range(num_questions):
        subject = subjects[i % len(subjects)]
        q, a, s, d = generate_quiz(subject, difficulty)
        questions.append(q)
        answers.append(a)
        quiz_subjects.append(s)
        difficulties.append(d)
    return questions, answers, quiz_subjects, difficulties

# Retrieve and blend answer
def retrieve_and_blend_answer(query):
    query_normalized = normalize_query(query)
    context = f"Student Grade: {st.session_state.student_info['grade']}\nSubjects: {', '.join(st.session_state.student_info['subjects'])}"

    if query_normalized == "cpu model" or query_normalized == "what is my cpu":
        cpu = get_cpu_model()
        if lang == "ml":
            return "ഘട്ടം 1: നിന്റെ സിസ്റ്റം പരിശോധിക്കുന്നു...\nഘട്ടം 2: നിന്റെ സിപിയു മോഡൽ: " + translator.translate(cpu)
        return f"Step 1: Checking your system...\nStep 2: Your CPU model is: {cpu}"

    if "math" in query_normalized or "solve" in query_normalized:
        solution = solve_math_problem(query_normalized)
        if lang == "ml":
            return f"**[ഗണിത പരിഹാരം]**\n{solution}"
        return f"**[Math Solution]**\n{solution}"

    if query_normalized in qa_pairs:
        response = qa_pairs[query_normalized]
        if lang == "ml":
            return f"**[അറിയാവുന്ന ഉത്തരം]**\nഘട്ടം 1: ഇത് എനിക്ക് ഇതിനകം അറിയാം!\nഘട്ടം 2: ഇതാ ഉത്തരം: {translator.translate(response)}"
        return f"**[Known Answer]**\nStep 1: I already know this!\nStep 2: Here’s the answer: {response}"

    if question_vectors is not None:
        query_vector = vectorizer.transform([query_normalized])
        similarities = cosine_similarity(query_vector, question_vectors)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0][most_similar_idx]
        if similarity_score >= 0.6:
            best_match = list(qa_pairs.values())[most_similar_idx]
            if lang == "ml":
                return f"**[സമാനമായ ഉത്തരം]**\nഘട്ടം 1: എനിക്ക് സമാനമായ ഒന്ന് കണ്ടെത്തി.\nഘട്ടം 2: ഇതാ അത്: {translator.translate(best_match)}"
            return f"**[Similar Answer]**\nStep 1: I found something close.\nStep 2: Here it is: {best_match}"

    response = generate_llm_response(query, context)
    auto_train_query(query, response)
    if lang == "ml":
        return f"**[ഉത്തരം]**\n{response}"
    return f"**[Answer]**\n{response}"

# Save history
def save_history_after_interaction(query, response):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_conversation_history(st.session_state.messages, st.session_state.student_info)

# Generate enhanced study plan with performance weighting
def generate_study_plan(deadlines, tasks_per_subject):
    study_plan = []
    today = datetime.now()
    task_durations = {"Review Notes": 30, "Practice Problems": 60, "Watch Video": 45, "Quiz Prep": 40}
    task_details = {
        "Review Notes": "Focus on key concepts and summaries" if lang == "en" else "പ്രധാന ആശയങ്ങളും സംഗ്രഹങ്ങളും ശ്രദ്ധിക്കുക",
        "Practice Problems": "Solve exercises from textbook or past quizzes" if lang == "en" else "പാഠപുസ്തകത്തിൽ നിന്നോ മുൻ ക്വിസുകളിൽ നിന്നോ വ്യായാമങ്ങൾ പരിഹരിക്കുക",
        "Watch Video": "Watch educational videos or tutorials" if lang == "en" else "വിദ്യാഭ്യാസ വീഡിയോകൾ അല്ലെങ്കിൽ ട്യൂട്ടോറിയലുകൾ കാണുക",
        "Quiz Prep": "Test yourself with sample questions" if lang == "en" else "സാമ്പിൾ ചോദ്യങ്ങളുമായി സ്വയം പരീക്ഷിക്കുക"
    }
    
    history_df = pd.DataFrame(st.session_state.student_info["quiz_history"])
    subject_weights = {subject: 1.0 for subject in st.session_state.student_info["subjects"]}
    if not history_df.empty:
        performance = history_df.groupby("subject")["correct"].mean()
        for subject in subject_weights:
            subject_weights[subject] = 1.5 - (performance.get(subject, 0.5) * 0.5)  # Lower performance = higher weight
    
    all_tasks = []
    for subject, deadline_str in deadlines.items():
        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
            days_left = (deadline - today).days
            if days_left <= 0:
                continue
            task_list = tasks_per_subject.get(subject, ["Review Notes", "Practice Problems"])
            weighted_tasks = task_list * int(subject_weights.get(subject, 1.0) * 2)  # More tasks for weaker subjects
            for task in weighted_tasks:
                all_tasks.append((subject, task, deadline, days_left))
        except ValueError:
            st.error(f"Invalid deadline format for {subject}. Use YYYY-MM-DD." if lang == "en" else f"{subject}-ന് തെറ്റായ അവസാന തീയതി ഫോർമാറ്റ്. YYYY-MM-DD ഉപയോഗിക്കുക.")
            continue
    
    if not all_tasks:
        return []
    
    days_used = {}
    start_time = 17  # 5:00 PM default start
    for subject, task, deadline, days_left in sorted(all_tasks, key=lambda x: x[2]):
        days_to_use = min(days_left - 1, max(1, len(all_tasks) // 2))
        for day_offset in range(days_to_use):
            date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            if days_used.get(date, 0) >= 120:
                continue
            duration = task_durations.get(task, 30)
            if days_used.get(date, 0) + duration <= 120:
                hour = start_time + (days_used.get(date, 0) // 60)
                minute = days_used.get(date, 0) % 60
                days_used[date] = days_used.get(date, 0) + duration
                study_plan.append({
                    "subject": subject,
                    "date": date,
                    "task": task,
                    "details": task_details.get(task, "Complete this task" if lang == "en" else "ഈ ടാസ്ക് പൂർത്തിയാക്കുക"),
                    "start_time": f"{hour:02d}:{minute:02d}",
                    "duration": duration,
                    "completed": False
                })
                break
    
    study_plan.sort(key=lambda x: (x["date"], x["start_time"]))
    return study_plan

# Enhanced UI with custom CSS
st.markdown("""
    <style>
        .correct { color: green; font-weight: bold; }
        .wrong { color: red; font-weight: bold; }
        .calendar-table { border-collapse: collapse; width: 100%; }
        .calendar-table th, .calendar-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .calendar-table th { background-color: #f2f2f2; font-size: 16px; }
        .stButton>button { width: 100%; }
        .stTextInput>div>input { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Main layout with header
st.title(translations[lang]["title"])
st.caption(translations[lang]["subtitle"])

# Student profile in a container
with st.container():
    st.subheader("Profile" if lang == "en" else "പ്രൊഫൈൽ")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.student_info["grade"] = st.selectbox(translations[lang]["grade_label"], ["8", "9", "10", "11", "12", "College"], index=["8", "9", "10", "11", "12", "College"].index(st.session_state.student_info.get("grade", "10")))
    with col2:
        subjects_en = ["Math", "Science", "History", "English"]
        subjects_ml = ["ഗണിതം", "ശാസ്ത്രം", "ചരിത്രം", "ഇംഗ്ലീഷ്"]
        subjects_display = subjects_ml if lang == "ml" else subjects_en
        selected_subjects = st.multiselect(translations[lang]["subjects_label"], subjects_display, default=[subjects_display[subjects_en.index(s)] for s in st.session_state.student_info.get("subjects", ["Math", "Science"])])
        st.session_state.student_info["subjects"] = [subjects_en[subjects_display.index(s)] for s in selected_subjects]

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs([translations[lang]["study_tab"], translations[lang]["quiz_tab"], translations[lang]["results_tab"], translations[lang]["plan_tab"]])

# Study Tab
with tab1:
    with st.container():
        st.subheader(translations[lang]["study_tab"])
        query = st.text_input(translations[lang]["ask_placeholder"], placeholder=translations[lang]["ask_placeholder"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button(translations[lang]["answer_button"]):
                if query:
                    response = retrieve_and_blend_answer(query)
                    st.markdown(f"**{'You Asked' if lang == 'en' else 'നീ ചോദിച്ചത്'}:** {query}")
                    st.markdown(response)
                    save_history_after_interaction(query, response)
        with col2:
            if st.button(translations[lang]["tip_button"]):
                response = generate_llm_response("Give me a simple study tip in numbered steps.")
                st.markdown(f"**{'Study Tip' if lang == 'en' else 'പഠന നുറുങ്ങ്'}:**")
                st.markdown(response)
                save_history_after_interaction("study tip", response)

# Quiz/Test Tab
with tab2:
    with st.container():
        st.subheader(translations[lang]["quiz_tab"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button(translations[lang]["single_quiz_button"]):
                q, a, s, d = generate_quiz(difficulty="Medium")
                st.session_state.quiz_state = {"questions": [q], "answers": [a], "subjects": [s], "difficulties": [d], "current": 0, "mock_test": False, "timer_start": time.time(), "time_limit": 30}
                st.markdown(f"**{'Quiz Question (Medium)' if lang == 'en' else 'ക്വിസ് ചോദ്യം (മീഡിയം)'}:** {st.session_state.quiz_state['questions'][0]}")
        with col2:
            num_questions = st.slider(translations[lang]["num_questions_label"], 1, 10, 3)
            difficulty_options = ["Easy", "Medium", "Hard"] if lang == "en" else ["എളുപ്പം", "മീഡിയം", "കഠിനം"]
            difficulty = st.selectbox(translations[lang]["difficulty_label"], difficulty_options, index=1)
            difficulty_en = ["Easy", "Medium", "Hard"][difficulty_options.index(difficulty)]
            if st.button(translations[lang]["mock_test_button"]):
                questions, answers, subjects, difficulties = generate_mock_test(num_questions, difficulty_en)
                st.session_state.quiz_state = {"questions": questions, "answers": answers, "subjects": subjects, "difficulties": difficulties, "current": 0, "mock_test": True, "timer_start": time.time(), "time_limit": num_questions * 30}
                st.markdown(f"**{'Mock Test Question 1' if lang == 'en' else 'മോക്ക് ടെസ്റ്റ് ചോദ്യം 1'} ({difficulties[0]}):** {st.session_state.quiz_state['questions'][0]}")

        # Quiz answer input with timer
        if st.session_state.quiz_state["questions"] and st.session_state.quiz_state["current"] < len(st.session_state.quiz_state["questions"]):
            current = st.session_state.quiz_state["current"]
            elapsed_time = time.time() - st.session_state.quiz_state["timer_start"]
            time_left = max(0, st.session_state.quiz_state["time_limit"] - int(elapsed_time))
            with st.container():
                st.write(f"**{'Question' if lang == 'en' else 'ചോദ്യം'} {current + 1}/{len(st.session_state.quiz_state['questions'])} ({st.session_state.quiz_state['difficulties'][current]})**: {st.session_state.quiz_state['questions'][current]}")
                timer_placeholder = st.empty()
                timer_placeholder.write(f"{'Time Left' if lang == 'en' else 'സമയം ബാക്കി'}: {time_left} {'seconds' if lang == 'en' else 'സെക്കന്റുകൾ'}")

                if time_left <= 0:
                    st.error("Time’s up! Moving to next question." if lang == "en" else "സമയം കഴിഞ്ഞു! അടുത്ത ചോദ്യത്തിലേക്ക് നീങ്ങുന്നു.")
                    st.session_state.quiz_state["current"] += 1
                    if st.session_state.quiz_state["current"] >= len(st.session_state.quiz_state["questions"]):
                        st.session_state.quiz_state = {"questions": [], "answers": [], "subjects": [], "difficulties": [], "current": 0, "mock_test": False, "timer_start": None, "time_limit": 0}
                        st.success("Mock Test Complete!" if lang == "en" else "മോക്ക് ടെസ്റ്റ് പൂർത്തിയായി!")
                    else:
                        st.session_state.quiz_state["timer_start"] = time.time()
                    st.rerun()

                student_answer = st.text_input(f"{'Your Answer' if lang == 'en' else 'നിന്റെ ഉത്തരം'}", key=f"quiz_answer_input_{current}")
                if st.button(translations[lang]["submit_answer"]):
                    correct_answer = st.session_state.quiz_state["answers"][current].lower()
                    student_answer_normalized = student_answer.lower().strip()
                    similarity = cosine_similarity(vectorizer.transform([student_answer_normalized]), vectorizer.transform([correct_answer]))[0][0] if student_answer else 0.0
                    score = 1.0 if student_answer_normalized == correct_answer else (0.5 if similarity > 0.8 else 0.0)  # Partial credit for close answers
                    is_correct = score == 1.0
                    
                    if is_correct:
                        st.success("Correct! Great job!" if lang == "en" else "ശരി! മികച്ച പ്രവർത്തനം!")
                    elif score > 0:
                        st.warning(f"Partially correct! Correct answer: {st.session_state.quiz_state['answers'][current]}" if lang == "en" else f"ഭാഗികമായി ശരി! ശരിയായ ഉത്തരം: {st.session_state.quiz_state['answers'][current]}")
                    else:
                        st.error(f"Wrong. Correct answer: {st.session_state.quiz_state['answers'][current]}" if lang == "en" else f"തെറ്റ്. ശരിയായ ഉത്തരം: {st.session_state.quiz_state['answers'][current]}")
                    
                    st.session_state.student_info["quiz_score"] += score
                    st.session_state.student_info["quiz_total"] += 1
                    st.session_state.student_info["quiz_history"].append({
                        "question": st.session_state.quiz_state["questions"][current],
                        "student_answer": student_answer,
                        "correct_answer": st.session_state.quiz_state["answers"][current],
                        "correct": is_correct,
                        "score": score,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": st.session_state.quiz_state["subjects"][current],
                        "difficulty": st.session_state.quiz_state["difficulties"][current]
                    })
                    save_conversation_history(st.session_state.messages, st.session_state.student_info)
                    st.session_state.quiz_state["current"] += 1
                    if st.session_state.quiz_state["current"] >= len(st.session_state.quiz_state["questions"]):
                        st.session_state.quiz_state = {"questions": [], "answers": [], "subjects": [], "difficulties": [], "current": 0, "mock_test": False, "timer_start": None, "time_limit": 0}
                        st.success("Mock Test Complete!" if lang == "en" else "മോക്ക് ടെസ്റ്റ് പൂർത്തിയായി!" if st.session_state.quiz_state["mock_test"] else "ക്വിസ് പൂർത്തിയായി!")
                    else:
                        st.session_state.quiz_state["timer_start"] = time.time()
                    st.rerun()

# Results Tab
with tab3:
    with st.container():
        st.subheader(translations[lang]["results_tab"])
        score = st.session_state.student_info["quiz_score"]
        total = st.session_state.student_info["quiz_total"]
        percentage = score / total * 100 if total > 0 else 0.0
        st.metric(f"{'Overall Score' if lang == 'en' else 'മൊത്തം സ്കോർ'}", f"{score:.1f}/{total} ({percentage:.1f}%)")
        st.progress(percentage / 100)
        
        if percentage >= 80:
            st.success("Excellent work! You’re a quiz master!" if lang == "en" else "മികച്ച പ്രവർത്തനം! നീ ഒരു ക്വിസ് മാസ്റ്റർ ആണ്!")
        elif percentage >= 50:
            st.info("Good effort! Keep practicing to improve!" if lang == "en" else "നല്ല ശ്രമം! മെച്ചപ്പെടുത്താൻ പരിശീലനം തുടരുക!")
        else:
            st.warning("Don’t give up! Let’s try those tough ones again!" if lang == "en" else "നിരാശപ്പെടരുത്! ബുദ്ധിമുട്ടുള്ളവ വീണ്ടും ശ്രമിക്കാം!")

        if st.session_state.student_info["quiz_history"]:
            history_df = pd.DataFrame(st.session_state.student_info["quiz_history"])
            # Ensure 'score' exists, default to 1.0 or 0.0 based on 'correct' if missing
            if "score" not in history_df.columns:
                history_df["score"] = history_df["correct"].apply(lambda x: 1.0 if x else 0.0)
            subject_summary = history_df.groupby("subject").agg({"score": "sum", "correct": "count"}).reset_index()
            subject_summary.columns = ["Subject", "Score", "Total"]
            subject_summary["Percentage"] = (subject_summary["Score"] / subject_summary["Total"] * 100).round(1)
            st.write(f"**{'Performance by Subject' if lang == 'en' else 'വിഷയം അനുസരിച്ചുള്ള പ്രകടനം'}:**")
            st.dataframe(subject_summary)

            fig = px.bar(subject_summary, x="Subject", y="Percentage", title="Quiz Performance by Subject" if lang == "en" else "വിഷയം അനുസരിച്ച് ക്വിസ് പ്രകടനം", range_y=[0, 100])
            st.plotly_chart(fig)

            with st.expander("Detailed Quiz History" if lang == "en" else "വിശദമായ ക്വിസ് ചരിത്രം"):
                for i, entry in enumerate(st.session_state.student_info["quiz_history"], 1):
                    status_class = "correct" if entry["correct"] else "wrong"
                    subject = entry.get("subject", "Unknown")
                    timestamp = entry.get("timestamp", "Not recorded")
                    difficulty = entry.get("difficulty", "Unknown")
                    score = entry.get("score", 1.0 if entry["correct"] else 0.0)  # Fallback for old entries
                    st.markdown(
                        f"<span class='{status_class}'>{i}. **{'Q' if lang == 'en' else 'ചോ'}:** {entry['question']} | **{'Your Answer' if lang == 'en' else 'നിന്റെ ഉത്തരം'}:** {entry['student_answer']} | "
                        f"**{'Correct Answer' if lang == 'en' else 'ശരിയായ ഉത്തരം'}:** {entry['correct_answer']} | **{'Score' if lang == 'en' else 'സ്കോർ'}:** {score:.1f} | "
                        f"**{'Subject' if lang == 'en' else 'വിഷയം'}:** {subject} | **{'Difficulty' if lang == 'en' else 'കാഠിന്യം'}:** {difficulty} | **{'Time' if lang == 'en' else 'സമയം'}:** {timestamp}</span>",
                        unsafe_allow_html=True
                    )

            incorrect = [entry for entry in st.session_state.student_info["quiz_history"] if entry.get("score", 1.0 if entry["correct"] else 0.0) < 1.0]
            if incorrect and st.button("Retry Incorrect Questions" if lang == "en" else "തെറ്റായ ചോദ്യങ്ങൾ വീണ്ടും ശ്രമിക്കുക"):
                questions = [entry["question"] for entry in incorrect]
                answers = [entry["correct_answer"] for entry in incorrect]
                subjects = [entry.get("subject", "Unknown") for entry in incorrect]
                difficulties = [entry.get("difficulty", "Unknown") for entry in incorrect]
                st.session_state.quiz_state = {"questions": questions, "answers": answers, "subjects": subjects, "difficulties": difficulties, "current": 0, "mock_test": True, "timer_start": time.time(), "time_limit": len(incorrect) * 30}
                st.markdown(f"**{'Retry Question 1' if lang == 'en' else 'വീണ്ടും ശ്രമിക്കുക ചോദ്യം 1'}:** {st.session_state.quiz_state['questions'][0]}")
                st.rerun()

            if st.button("Export Quiz History as CSV" if lang == "en" else "ക്വിസ് ചരിത്രം CSV ആയി എക്സ്പോർട്ട് ചെയ്യുക"):
                csv = history_df.to_csv(index=False)
                st.download_button("Download CSV" if lang == "en" else "CSV ഡൗൺലോഡ് ചെയ്യുക", csv, "quiz_history.csv", "text/csv")

        if st.button("Reset Quiz Results" if lang == "en" else "ക്വിസ് ഫലങ്ങൾ പുനഃസജ്ജമാക്കുക"):
            st.session_state.student_info["quiz_score"] = 0
            st.session_state.student_info["quiz_total"] = 0
            st.session_state.student_info["quiz_history"] = []
            save_conversation_history(st.session_state.messages, st.session_state.student_info)
            st.rerun()

# Study Plan Tab
with tab4:
    with st.container():
        st.subheader(translations[lang]["plan_tab"])
        
        deadlines = {}
        tasks_per_subject = {}
        task_options_en = ["Review Notes", "Practice Problems", "Watch Video", "Quiz Prep"]
        task_options_ml = ["നോട്ടുകൾ പരിശോധിക്കുക", "പ്രശ്നങ്ങൾ പരിശീലിക്കുക", "വീഡിയോ കാണുക", "ക്വിസ് തയ്യാറാക്കുക"]
        task_options = task_options_ml if lang == "ml" else task_options_en
        
        with st.expander("Set Deadlines and Tasks" if lang == "en" else "അവസാന തീയതികളും ടാസ്കുകളും സജ്ജമാക്കുക"):
            for subject in st.session_state.student_info["subjects"]:
                subject_display = subjects_ml[subjects_en.index(subject)] if lang == "ml" else subject
                st.write(f"**{subject_display}**")
                col1, col2 = st.columns(2)
                with col1:
                    deadline = st.date_input(f"{'Deadline for' if lang == 'en' else 'അവസാന തീയതി'} {subject_display}", min_value=datetime.now(), key=f"deadline_{subject}")
                    deadlines[subject] = deadline.strftime("%Y-%m-%d")
                with col2:
                    tasks = st.multiselect(f"{'Tasks for' if lang == 'en' else 'ടാസ്കുകൾ'} {subject_display}", task_options, default=[task_options[0], task_options[1]], key=f"tasks_{subject}")
                    tasks_per_subject[subject] = [task_options_en[task_options.index(t)] for t in tasks]
        
        if st.button(translations[lang]["generate_plan"]):
            st.session_state.student_info["study_plan"] = generate_study_plan(deadlines, tasks_per_subject)
            save_conversation_history(st.session_state.messages, st.session_state.student_info)
            st.rerun()

        if st.session_state.student_info["study_plan"]:
            plan_df = pd.DataFrame(st.session_state.student_info["study_plan"])
            required_columns = {"subject", "date", "task", "details", "start_time", "duration", "completed"}
            if not plan_df.empty and all(col in plan_df.columns for col in required_columns):
                total_duration = plan_df["duration"].sum()
                completed_duration = plan_df[plan_df["completed"]]["duration"].sum()
                completed_tasks = len(plan_df[plan_df["completed"]])
                st.metric(f"{translations[lang]['progress']}", f"{completed_tasks}/{len(plan_df)} {'tasks' if lang == 'en' else 'ടാസ്കുകൾ'} | {completed_duration}/{total_duration} {'mins' if lang == 'en' else 'മിനിറ്റ്'}")
                st.progress(completed_duration / total_duration if total_duration > 0 else 0)

                st.write(f"**{translations[lang]['daily_schedule']}:**")
                dates = sorted(plan_df["date"].unique())
                html_table = f"<table class='calendar-table'><tr><th>{'Date' if lang == 'en' else 'തീയതി'}</th><th>{'Tasks' if lang == 'en' else 'ടാസ്കുകൾ'}</th></tr>"
                for date in dates:
                    tasks_for_date = plan_df[plan_df["date"] == date]
                    tasks_html = "<br>".join([ 
                        f"{'☑' if row['completed'] else '☐'} {row['start_time']} - {task_options_ml[task_options_en.index(row['task'])] if lang == 'ml' else row['task']} ({subjects_ml[subjects_en.index(row['subject'])] if lang == 'ml' else row['subject']}) - {row['duration']} {'mins' if lang == 'en' else 'മിനിറ്റ്'}: {row['details']}"
                        for _, row in tasks_for_date.iterrows()
                    ])
                    html_table += f"<tr><td>{date}</td><td>{tasks_html if tasks_html else 'No tasks scheduled' if lang == 'en' else 'ഷെഡ്യൂൾ ചെയ്ത ടാസ്കുകൾ ഇല്ല'}</td></tr>"
                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)

                st.write(f"**{translations[lang]['mark_complete']}:**")
                for i, task in plan_df.iterrows():
                    task_key = f"task_{task['date']}_{i}"
                    completed = st.checkbox(
                        f"{task['date']} {task['start_time']} - {task_options_ml[task_options_en.index(task['task'])] if lang == 'ml' else task['task']} ({subjects_ml[subjects_en.index(task['subject'])] if lang == 'ml' else task['subject']}) - {task['duration']} {'mins' if lang == 'en' else 'മിനിറ്റ്'}: {task['details']}", 
                        value=task["completed"], 
                        key=task_key
                    )
                    if completed != task["completed"]:
                        st.session_state.student_info["study_plan"][i]["completed"] = completed
                        save_conversation_history(st.session_state.messages, st.session_state.student_info)
                        st.rerun()

                progress_df = pd.DataFrame({
                    "Status": ["Completed", "Remaining"] if lang == "en" else ["പൂർത്തിയായി", "ബാക്കി"],
                    "Minutes": [completed_duration, total_duration - completed_duration]
                })
                fig = px.pie(progress_df, values="Minutes", names="Status", title="Study Plan Progress" if lang == "en" else "പഠന പദ്ധതി പുരോഗതി")
                st.plotly_chart(fig)

                next_task = plan_df[~plan_df["completed"]].iloc[0] if not plan_df["completed"].all() else None
                if next_task is not None:
                    st.info(f"**{translations[lang]['next_step']}:** {'On' if lang == 'en' else 'നിന്ന്'} {next_task['date']} {'at' if lang == 'en' else 'ന്'} {next_task['start_time']}, {task_options_ml[task_options_en.index(next_task['task'])] if lang == 'ml' else next_task['task']} {'for' if lang == 'en' else 'ന്'} {subjects_ml[subjects_en.index(next_task['subject'])] if lang == 'ml' else next_task['subject']} ({next_task['details']})")
                else:
                    st.success(translations[lang]["all_complete"])
            else:
                st.write(translations[lang]["invalid_plan"])
        else:
            st.write(translations[lang]["no_plan"])

# Display last response
if st.session_state.messages and not st.session_state.quiz_state["questions"]:
    with st.container():
        st.markdown("---")
        st.subheader(f"{'Last Interaction' if lang == 'en' else 'അവസാന ഇടപെടൽ'}")
        last_query = st.session_state.messages[-2]["content"]
        last_response = st.session_state.messages[-1]["content"]
        st.markdown(f"**{'You Asked' if lang == 'en' else 'നീ ചോദിച്ചത്'}:** {last_query}")
        st.markdown(last_response)