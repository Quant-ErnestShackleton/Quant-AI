# main.py
from src.model import train_model
from src.createData import createDutchData, createNonDutchData

if __name__ == "__main__":
    createDutchData(500) # 학습 데이터 생성
    # train_model() # 모델 학습
    # createNonDutchData(500)