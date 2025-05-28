from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:

    def __init__(self, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map='auto',trust_remote_code=True,
    use_safetensors=True,).eval()

    def get_score(self, query:str, passage:str):
        """
        쿼리문에 대해 한개의 비교 대상 질의. 안쓰일 것으로 예상됨
        """
        
        with torch.no_grad():
            qp_pair = self.tokenizer(
                [[query, passage]], 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(self.model.device)
            result = self.model(**qp_pair, return_dict=True)
            return result.logits.view(-1, ).float().to('cpu')

    def get_scores(self, query:str, passage_list:List[str]):
        """
        쿼리문에 대해 여러 개의 비교 대상 질의
        """

        # 스코어 결과들을 종합하여, 그들을 확률값으로 만들기 위한 정규화 함수
        def exp_normalize(x:list):
            b = x.max()
            y = np.exp(x - b)
            return y / y.sum()
        
        pairs = [[query, passage] for passage in passage_list]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(self.model.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().to('cpu')
        
        return exp_normalize(scores)
        