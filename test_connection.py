import requests

try:
    r = requests.get("https://huggingface.co")
    print("Status:", r.status_code)
except Exception as e:
    print(e)

try:
    r = requests.get("https://api-inference.huggingface.co")
    print("Inference Status:", r.status_code)
except Exception as e:
    print(e)