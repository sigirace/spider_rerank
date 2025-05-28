# %%
import requests

# %%
RERANK_ENDPOINT = "http://localhost:8000/rerank"

# %%
pairs = [
    ['what is panda?', 'hi'],
    ['hi', 'what is panda?'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
]

headers = {"Content-Type": "application/json", 'Accept':'application/json'}

data = {
    "query": "asdf",
    "passage_list": ["가나다라"]
}

# %%
r = requests.post(RERANK_ENDPOINT, headers=headers, json=data)
r

# %%
r.content
# %%
