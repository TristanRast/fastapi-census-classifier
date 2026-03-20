"""
Script to test the FastAPI application locally.
"""
import requests
import json


# Base URL - make sure the server is running
BASE_URL = "http://127.0.0.1:8000"

print("Testing ML Model API...")
print("=" * 80)

# Test GET request
print("\n1. Testing GET request to root endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test POST request - example with income <=50K
print("\n2. Testing POST request with income <=50K example...")
data_low = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=data_low)
    print(f"Status Code: {response.status_code}")
    print(f"Input Data: {json.dumps(data_low, indent=2)}")
    print(f"Prediction: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test POST request - example with income >50K
print("\n3. Testing POST request with income >50K example...")
data_high = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=data_high)
    print(f"Status Code: {response.status_code}")
    print(f"Input Data: {json.dumps(data_high, indent=2)}")
    print(f"Prediction: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Testing complete!")
print("\nMake sure to take a screenshot showing:")
print("  - Successful status codes (200)")
print("  - GET response message")
print("  - POST predictions")
print("Save as: screenshots/local_api.png")
