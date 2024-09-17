import xgboost as xgb

def load_trained_model(model_save_path='dutchpay_detection_model.json'):
    try:
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(model_save_path)
        print(f"모델이 '{model_save_path}'에서 성공적으로 로드되었습니다.")
        return loaded_model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

