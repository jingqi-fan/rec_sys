import requests

res = requests.post(
    "http://127.0.0.1:8000/recommend/",
    json={"user_id": 1}
)
print(res.status_code)
print(res.text)