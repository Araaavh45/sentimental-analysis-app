import requests

# Test with a clearly positive review
positive_text = "I love this product! It's amazing and works perfectly."
response = requests.post('http://127.0.0.1:5000/predict', json={'text': positive_text})
print(f"Positive test: {positive_text}")
print(f"Result: {response.json()}")

# Test with a clearly negative review
negative_text = "This is terrible. I hate it and it doesn't work at all."
response = requests.post('http://127.0.0.1:5000/predict', json={'text': negative_text})
print(f"Negative test: {negative_text}")
print(f"Result: {response.json()}")

# Test with a neutral review
neutral_text = "It's okay. Not great, not terrible."
response = requests.post('http://127.0.0.1:5000/predict', json={'text': neutral_text})
print(f"Neutral test: {neutral_text}")
print(f"Result: {response.json()}")