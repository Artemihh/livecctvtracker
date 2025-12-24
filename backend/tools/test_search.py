# backend/tools/test_search.py
import requests, pprint
url = "http://127.0.0.1:8000/api/search/image"
files = {"file": open("test_person.jpeg","rb")}
params = {"top_k":5, "min_sim":0.25}
r = requests.post(url, files=files, params=params, timeout=10)
pprint.pprint((r.status_code, r.json()))
