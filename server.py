from typing import List, Annotated

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

import uvicorn

from model.bge_reranker import BGEReranker

reranker = BGEReranker("./model/bge-reranker-v2-m3")
app = FastAPI()


@app.get("/greet")
def read_root():
    return {"Hello": "World"}


class ReRankRequestBody(BaseModel):
    query: Annotated[str, Query(max_length=512)]
    passage_list: Annotated[
        List[Annotated[str, Query(max_length=512)]],
        Query(min_length=1, max_length=1000),
    ]


class ReRankResponse(BaseModel):
    scores: List[float]

    def check_length_with_request(self, request: ReRankRequestBody):
        """
        ReRank 결과 응답을 생성했다면, 당연히 요청 내 passage_list 개수 만큼의 스코어 응답이 발생해야 함.
        응답을 전달하기 전에 이를 검증하기 위한 함수.
        """
        if len(request.passage_list) != len(self.scores):
            raise HTTPException(
                status_code=500, detail="Response length does not match request length."
            )


@app.post("/rerank", response_model=ReRankResponse)
def rerank(request: ReRankRequestBody):
    # 응답 생성
    scores = reranker.get_scores(query=request.query, passage_list=request.passage_list)
    response = ReRankResponse(scores=scores)

    # 생성된 응답의 유효성 검증
    response.check_length_with_request(request)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
