import json

import requests



# TODO: send a GET using the URL http://127.0.0.1:8000
API_URL = "http://127.0.0.1:8000" # Your code here

response_get = requests.get(API_URL)
response_get.status_code == 200
# TODO: print the status code
# print()
print(f"GET request successful. Status code: {response_get.status_code}")

# TODO: print the welcome message
# print()
message = response_get.json().get("message")
print(message)

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
response_post = requests.post(f"{API_URL}/inference", json=data) # Your code here

response_post.status_code == 200

# TODO: print the status code
# print()
print(f"POST request successful. Status code: {response_get.status_code}")

# TODO: print the result
# print()
result = response_post.json().get("result")
print(result)