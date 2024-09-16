# main.py
from src.model import train_model
from src.createData import createData
from src.dataToJson import data_to_json

if __name__ == "__main__":
    createData(500, 500)
    # train_model() # 모델 학습
    # data_to_json()