import requests
url = "http://localhost:5000/predict"
test_files = {
    "test_file_1": open("./test.jpg", "rb")
}

response = requests.post(url, files=test_files)
print(response.json())