# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional
from src.preprocessing import preprocess_input
from src.loadModel import load_trained_model
import math

app = FastAPI(title="더치페이 인식 AI API")

# 모델 로드 (서버 시작 시 한 번만 로드)
MODEL_PATH = './dutchpay_detection_model.json'  # 모델 파일 경로
try:
    model = load_trained_model()
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None

# Pydantic 모델 정의
class Participant(BaseModel):
    user_dutch_chance: Optional[float] = 0
    deposit_amount: Optional[float] = 0
    time_after_payment: Optional[float] = 0
    is_name_present: Optional[int] = 0

class InputData(BaseModel):
    total_dutchpay_amount: float
    participants_count: int
    participants_data: List[Participant]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_proba: float

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    # 입력 데이터 전처리
    try:
        processed_data = {
            'participants_data': [p.dict() for p in data.participants_data],
            'total_dutchpay_amount': data.total_dutchpay_amount,
            'participants_count': data.participants_count
        }
        df = preprocess_input(processed_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"데이터 전처리 오류: {e}")

    print(df)

    # 피처 추출
    X = df  # 전처리 함수에서 이미 피처만 반환

    # 예측
    try:
        prediction = model.predict(X)[0]  # 단일 예측
        prediction_proba = model.predict_proba(X)[0][1]  # 클래스 1의 확률
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {e}")

    response = {
        "prediction": int(prediction),
        "prediction_proba": int(math.floor(prediction_proba * 100))
    }

    return response
