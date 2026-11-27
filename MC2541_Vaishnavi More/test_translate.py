import requests

url = "http://localhost:5000/translate"

data = {
    "q": "Welcome to TubeMentor AI",
    "source": "en",
    "target": "hi",
    "format": "text"
}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Response:", response.json())
