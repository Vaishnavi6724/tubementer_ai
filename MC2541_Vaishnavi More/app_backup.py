from pathlib import Path

app_code = r'''import streamlit as st

from gemini_ai import ask_gemini, generate_summary
from youtube_api import get_video_info
from transcript import get_transcript
from rag import build_knowledge_base, ask_rag
from quiz_generator import create_quiz
from code_runner import run_code
from translator import translate_text
from pdf_export import create_pdf
from database import initialize_database, save_question, create_user, login_user


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TubeMentor AI",
    page_icon="🎓",
    layout="wide",
)

initialize_database()


# ==========================================
# LOGIN SYSTEM
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# ==========================================
# LOGIN / CREATE ACCOUNT PAGE
# ==========================================

if not st.session_state.logged_in:

    st.title("🎓 TubeMentor AI")
    st.markdown("### Your AI-Powered Learning Assistant")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])

    # --------------------------------------
    # LOGIN
    # --------------------------------------

    with tab1:
        st.subheader("🔐 Login")

        email = st.text_input("Email", key="login_email")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button("🚀 Login", use_container_width=True):
            if not email or not password:
                st.warning("Please enter email and password.")
            else:
                user = login_user(email, password)

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = {
                        "id": user[0],
                        "name": user[1],
                        "email": user[2],
                    }
                    st.success(f"Welcome {user[1]}! 🎉")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password.")

    # --------------------------------------
    # CREATE ACCOUNT
    # --------------------------------------

    with tab2:
        st.subheader("📝 Create Account")

        name = st.text_input("Full Name", key="register_name")
        email = st.text_input("Email Address", key="register_email")
        password = st.text_input(
            "Password",
            type="password",
            key="register_password",
        )
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password",
        )

        if st.button("✨ Create Account", use_container_width=True):
            if not name or not email or not password:
                st.warning("Please fill all fields.")
            elif password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must contain at least 6 characters.")
            else:
                success = create_user(name, email, password)

                if success:
                    st.success("✅ Account created successfully!")
                    st.info("Please go to the Login tab.")
                else:
                    st.error("❌ An account with this email already exists.")

    st.stop()


# ==========================================
# LOGGED-IN USER
# ==========================================

user = st.session_state.user

st.sidebar.success(f"👤 {user['name']}")
st.sidebar.caption(user["email"])

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


# ==========================================
# HEADER
# ==========================================

st.title("🎓 TubeMentor AI")
st.write("Your AI-powered DSA Learning Assistant")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📚 TubeMentor AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📺 YouTube Learning",
        "🤖 AI Tutor",
        "📝 AI Notes",
        "🧠 DSA Quiz",
        "💻 Coding Lab",
        "🌐 Translator",
    ],
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    st.success(f"Welcome back, {user['name']}! 👋")
    st.write("Welcome to your TubeMentor AI learning dashboard.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎥 Videos", "0")

    with col2:
        st.metric("❓ Questions", "0")

    with col3:
        st.metric("🧠 Quiz Score", "0%")

    st.info(
        "Welcome to TubeMentor AI! "
        "Start by processing a YouTube DSA video "
        "or asking the AI Tutor a question."
    )


# ==========================================
# YOUTUBE LEARNING
# ==========================================

elif page == "📺 YouTube Learning":

    st.header("📺 YouTube Learning")

    url = st.text_input("Enter YouTube Video URL")

    if st.button("🚀 Process Video"):

        if not url:
            st.warning("Please enter a YouTube URL.")

        else:
            try:
                with st.spinner("Getting YouTube video information..."):
                    info = get_video_info(url)

                st.image(info["thumbnail"], width=500)
                st.subheader(info["title"])
                st.write(f"**Channel:** {info['channel']}")
                st.write(f"**Views:** {info['views']}")

                with st.spinner("Extracting transcript..."):
                    transcript = get_transcript(info["video_id"])

                st.success("Transcript extracted successfully!")

                st.session_state["transcript"] = transcript
                st.session_state["video_info"] = info

                with st.spinner("Building knowledge base..."):
                    count = build_knowledge_base(transcript)

                st.success(
                    f"RAG knowledge base created with {count} chunks."
                )

                st.subheader("📄 Transcript Preview")
                st.write(transcript[:3000])

            except Exception as e:
                st.error(f"Error: {e}")


# ==========================================
# AI TUTOR
# ==========================================

elif page == "🤖 AI Tutor":

    st.header("🤖 AI Tutor")

    question = st.text_area(
        "Ask your DSA question",
        placeholder="Example: Explain binary search with Python code.",
    )

    use_rag = st.checkbox("Use my learning material", value=True)

    if st.button("🧠 Ask AI"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("🤖 TubeMentor AI is thinking..."):

                    if use_rag:
                        answer = ask_rag(question)
                    else:
                        answer = ask_gemini(question)

                st.success("Answer generated!")
                st.markdown(answer)

                save_question(question, answer)

            except Exception as e:
                st.error(f"AI Error: {e}")


# ==========================================
# AI NOTES
# ==========================================

elif page == "📝 AI Notes":

    st.header("📝 AI Notes")

    transcript = st.session_state.get("transcript", "")

    if not transcript:
        st.info("First process a YouTube video.")

    else:
        if st.button("📝 Generate Notes"):

            with st.spinner("Generating notes..."):
                notes = generate_summary(transcript)

            st.session_state["notes"] = notes

        notes = st.session_state.get("notes", "")

        if notes:
            st.markdown(notes)

            filename = "TubeMentor_Notes.pdf"

            if st.button("📥 Create PDF"):
                try:
                    create_pdf(notes, filename)

                    with open(filename, "rb") as file:
                        st.download_button(
                            "Download PDF",
                            file,
                            filename=filename,
                            mime="application/pdf",
                        )

                except Exception as e:
                    st.error(f"PDF Error: {e}")


# ==========================================
# DSA QUIZ
# ==========================================

elif page == "🧠 DSA Quiz":

    st.header("🧠 DSA Quiz")

    topic = st.selectbox(
        "Topic",
        [
            "Arrays",
            "Searching",
            "Sorting",
            "Linked List",
            "Stack",
            "Queue",
            "Trees",
            "Graphs",
        ],
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"],
    )

    number = st.slider(
        "Number of Questions",
        1,
        10,
        5,
    )

    if st.button("🎯 Generate Quiz"):

        material = st.session_state.get(
            "transcript",
            f"Generate questions about {topic}.",
        )

        with st.spinner("Generating quiz..."):
            quiz = create_quiz(material, number, difficulty)

        st.markdown(quiz)


# ==========================================
# CODING LAB
# ==========================================

elif page == "💻 Coding Lab":

    st.header("💻 DSA Coding Lab")

    language = st.selectbox(
        "Programming Language",
        ["Python", "C++", "Java", "C"],
    )

    if language == "Python":

        default_code = """def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


arr = [2, 5, 8, 12, 16, 23, 38]
target = 23

print(binary_search(arr, target))
"""

    elif language == "C++":

        default_code = """#include <iostream>
using namespace std;

int binarySearch(int arr[], int n, int target) {

    int low = 0;
    int high = n - 1;

    while (low <= high) {

        int mid = low + (high - low) / 2;

        if (arr[mid] == target)
            return mid;

        if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}

int main() {

    int arr[] = {2, 5, 8, 12, 16, 23, 38};

    cout << binarySearch(arr, 7, 23);

    return 0;
}
"""

    elif language == "Java":

        default_code = """public class Main {

    static int binarySearch(int[] arr, int target) {

        int low = 0;
        int high = arr.length - 1;

        while (low <= high) {

            int mid = low + (high - low) / 2;

            if (arr[mid] == target)
                return mid;

            if (arr[mid] < target)
                low = mid + 1;
            else
                high = mid - 1;
        }

        return -1;
    }

    public static void main(String[] args) {

        int[] arr = {2, 5, 8, 12, 16, 23, 38};

        System.out.println(binarySearch(arr, 23));
    }
}
"""

    else:

        default_code = """#include <stdio.h>

int binarySearch(int arr[], int n, int target) {

    int low = 0;
    int high = n - 1;

    while (low <= high) {

        int mid = low + (high - low) / 2;

        if (arr[mid] == target)
            return mid;

        if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}

int main() {

    int arr[] = {2, 5, 8, 12, 16, 23, 38};

    printf("%d", binarySearch(arr, 7, 23));

    return 0;
}
"""

    code = st.text_area(
        "Write your code",
        height=350,
        value=default_code,
    )

    if st.button("▶️ Run Code"):

        try:
            with st.spinner("Running code..."):
                result = run_code(code, language)

            st.subheader("💻 Result")

            if result.get("stdout"):
                st.code(result["stdout"])

            if result.get("stderr"):
                st.error(result["stderr"])

            if result.get("compile_output"):
                st.error(result["compile_output"])

            if (
                not result.get("stdout")
                and not result.get("stderr")
                and not result.get("compile_output")
            ):
                st.info("Code executed successfully.")

        except Exception as e:
            st.error(f"Code execution error: {e}")


# ==========================================
# TRANSLATOR
# ==========================================

elif page == "🌐 Translator":

    st.header("🌐 Translator")

    text = st.text_area("Enter text")

    language = st.selectbox(
        "Translate to",
        ["Hindi", "Marathi", "English"],
    )

    language_codes = {
        "Hindi": "hi",
        "Marathi": "mr",
        "English": "en",
    }

    if st.button("🌐 Translate"):

        if not text.strip():
            st.warning("Enter text first.")

        else:
            try:
                with st.spinner("Translating..."):
                    translated = translate_text(
                        text,
                        language_codes[language],
                    )

                st.subheader("Translation")
                st.write(translated)

            except Exception as e:
                st.error(f"Translation error: {e}")
'''

db_code = r'''import sqlite3
import hashlib

DATABASE = "tube_mentor.db"


def get_connection():
    return sqlite3.connect(DATABASE)


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            score INTEGER,
            total INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def create_user(name, email, password):

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (name, email, password)
            VALUES (?, ?, ?)
            """,
            (name.strip(), email.strip().lower(), hash_password(password)),
        )

        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def login_user(email, password):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, name, email
        FROM users
        WHERE email = ? AND password = ?
        """,
        (email.strip().lower(), hash_password(password)),
    )

    user = cursor.fetchone()
    connection.close()

    return user


def save_question(question, answer):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO history (question, answer)
        VALUES (?, ?)
        """,
        (question, answer),
    )

    connection.commit()
    connection.close()
'''

out1 = Path("/mnt/data/app_fixed.py")
out2 = Path("/mnt/data/database_fixed.py")
out1.write_text(app_code, encoding="utf-8")
out2.write_text(db_code, encoding="utf-8")

print(f"Created: {out1}")
print(f"Created: {out2}")
print("Both files were written as UTF-8 Python files.")
