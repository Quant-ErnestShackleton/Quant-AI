# main.py
from src.model import train_model
from src.createData import createDutchData, createNonDutchData
from src.dataToJson import data_to_json

if __name__ == "__main__":
    # createDutchData(500) # 학습 데이터 생성
    # createNonDutchData(500)
    # train_model() # 모델 학습
    data_to_json()