# %%
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('./bge-reranker-v2-m3')
model.eval()

# %%
pairs = [
    ['what is panda?', 'hi'],
    ['hi', 'what is panda?'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    model_output = model(**inputs, return_dict=True)
    scores = model_output.logits.view(-1, ).float()
    print(scores)

# %%
from typing import List

def get_score(query:str, passage:str):
    """
    쿼리에 대해 한개의 비교 대상. 안쓰일 것으로 예상됨
    """
    
    with torch.no_grad():
        qp_pair = tokenizer([[query, passage]], padding=True, truncation=True, return_tensors='pt', max_length=512)
        result = model(**qp_pair, return_dict=True)
        return result.logits.view(-1, ).float()


def get_scores(query:str, passage_list:List[str]):
    """
    쿼리에 대해 여러 개의 비교 대상
    """

    # 스코어 결과들을 종합하여, 그들을 확률값으로 만들기 위한 정규화 함수
    def exp_normalize(x:list):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()
    
    pairs = [[query, passage] for passage in passage_list]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    return exp_normalize(scores)


# %%
