from google import genai
from config import GEMINI_API_KEY

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(question, context=""):
    prompt = f"""
You are TubeMentor AI, an expert DSA tutor.

Use the learning material below when it is provided.

Learning material:
{context}

Student question:
{question}

Give a clear and educational answer.

For DSA questions, use this structure when appropriate:

1. Definition
2. Simple Explanation
3. Example
4. Algorithm / Steps
5. Python Code
6. Time Complexity
7. Space Complexity
8. Interview Tip
"""

    response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

    return response.text


def generate_summary(text):
    prompt = f"""
Summarize the following educational content.

Give:
1. Short Summary
2. Key Points
3. Important Concepts
4. Exam Points

Content:
{text}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


def generate_mcqs(text, number=5, difficulty="Medium"):
    prompt = f"""
Create {number} multiple-choice questions from the following DSA
learning material.

Difficulty: {difficulty}

For every question provide:
- Question
- Four options
- Correct answer
- Short explanation

Learning material:
{text}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text