from gemini_ai import generate_mcqs


def create_quiz(
    content,
    number=5,
    difficulty="Medium"
):

    return generate_mcqs(
        content,
        number,
        difficulty
    )