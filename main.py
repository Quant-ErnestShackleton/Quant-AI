# main.py
from src.train import process_and_train_xgboost
# from src.createData import createData
from src.dataToJson import data_to_json

# 함수 호출 예시
dutchpay_csv = './data/generated_dutch_pay_data.csv' # dutchpay csv 파일 경로
non_dutchpay_csv = './data/generated_non_dutch_pay_data.csv' # non-dutchpay csv 파일 경로

if __name__ == "__main__":
    # createData(100000, 100000)

    # 모델 학습
    model, X_test, y_test, y_pred, accuracy, report = process_and_train_xgboost(
        dutchpay_csv_path=dutchpay_csv,
        non_dutchpay_csv_path=non_dutchpay_csv,
        max_participants=50,
        test_size=0.2,
        random_state=42,
        apply_smote=True  # 클래스 불균형 처리 여부
    )
