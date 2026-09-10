import urllib.request
import urllib.parse
import json

def get(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

print("\n--- Drift Check ---")
status, data = post("http://localhost:8000/api/drift/check", {"simulate_drift": False, "n_reference_days": 30, "n_current_days": 7})
print(f"Status: {status}")
print(json.dumps(data, indent=2))
