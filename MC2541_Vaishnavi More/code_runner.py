import base64
import time
import requests

from config import JUDGE0_API_URL


LANGUAGE_IDS = {
    "Python": 71,
    "C++": 54,
    "Java": 62,
    "C": 50
}


def run_code(code, language):

    if language not in LANGUAGE_IDS:
        raise ValueError("Unsupported programming language")

    if not code or not code.strip():
        raise ValueError("No code was provided")

    language_id = LANGUAGE_IDS[language]

    submission_url = f"{JUDGE0_API_URL}/submissions"

    # Encode the ORIGINAL source code exactly once
    encoded_code = base64.b64encode(
        code.encode("utf-8")
    ).decode("utf-8")

    payload = {
        "language_id": language_id,
        "source_code": encoded_code,
        "base64_encoded": True
    }

    response = requests.post(
        submission_url,
        json=payload,
        timeout=20
    )

    response.raise_for_status()

    token = response.json()["token"]

    result_url = (
        f"{JUDGE0_API_URL}/submissions/{token}"
        "?base64_encoded=true"
    )

    for _ in range(20):

        response = requests.get(
            result_url,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

        status_id = result["status"]["id"]

        if status_id in [1, 2]:
            time.sleep(1)
            continue

        # Decode Judge0 output before returning it
        for field in ["stdout", "stderr", "compile_output", "message"]:
            if result.get(field):
                try:
                    result[field] = base64.b64decode(
                        result[field]
                    ).decode("utf-8", errors="replace")
                except Exception:
                    pass

        return result

    raise TimeoutError("Code execution timed out")