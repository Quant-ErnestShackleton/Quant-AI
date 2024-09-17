import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")


def process_and_train_xgboost(
        dutchpay_csv_path,
        non_dutchpay_csv_path,
        max_participants=50,
        test_size=0.2,
        random_state=42,
        apply_smote=True
):
    """
    두 개의 CSV 파일(더치페이 데이터와 non-dutchpay 데이터)을 읽어와서 participants_data를 평탄화 및 패딩한 후,
    XGBoost 모델을 학습하고 평가하는 함수.

    Parameters:
    - dutchpay_csv_path (str): 더치페이 데이터가 저장된 CSV 파일의 경로.
    - non_dutchpay_csv_path (str): non-dutchpay 데이터가 저장된 CSV 파일의 경로.
    - max_participants (int): 평탄화 시 최대 참가자 수. 기본값은 50.
    - test_size (float): 테스트 세트의 비율. 기본값은 0.2 (20%).
    - random_state (int): 재현성을 위한 랜덤 시드. 기본값은 42.
    - apply_smote (bool): 클래스 불균형 처리를 위해 SMOTE를 적용할지 여부. 기본값은 True.

    Returns:
    - model (XGBClassifier): 학습된 XGBoost 모델.
    - X_test (DataFrame): 테스트 피처.
    - y_test (Series): 테스트 레이블.
    - y_pred (ndarray): 예측된 레이블.
    - accuracy (float): 모델 정확도.
    - report (str): 분류 보고서.
    """

    # 1. CSV 파일 읽어오기
    try:
        dutchpay_df = pd.read_csv(dutchpay_csv_path)
        print(f"Dutchpay CSV 파일 '{dutchpay_csv_path}'을 성공적으로 읽어왔습니다.")
    except Exception as e:
        print(f"Dutchpay CSV 파일을 읽어오는 데 실패했습니다: {e}")
        return None

    try:
        non_dutchpay_df = pd.read_csv(non_dutchpay_csv_path)
        print(f"Non-Dutchpay CSV 파일 '{non_dutchpay_csv_path}'을 성공적으로 읽어왔습니다.")
    except Exception as e:
        print(f"Non-Dutchpay CSV 파일을 읽어오는 데 실패했습니다: {e}")
        return None

    # 2. participants_data 컬럼을 리스트 형태로 변환
    def parse_participants_data(data_str):
        try:
            return ast.literal_eval(data_str)
        except:
            return []

    dutchpay_df['participants_data'] = dutchpay_df['participants_data'].apply(parse_participants_data)
    non_dutchpay_df['participants_data'] = non_dutchpay_df['participants_data'].apply(parse_participants_data)
    print("participants_data 컬럼을 성공적으로 파싱했습니다.")

    # 3. 평탄화 및 패딩 함수 정의
    def flatten_and_pad(participants_data, max_participants=50):
        flattened = []
        for participants in participants_data:
            flat = []
            for p in participants:
                flat.extend([
                    p.get('user_dutch_chance', 0),
                    p.get('deposit_amount', 0),
                    p.get('time_after_payment', 0),
                    p.get('is_name_present', 0)
                ])
            # 패딩: 참가자 수가 부족하면 0으로 채움
            while len(flat) < max_participants * 4:
                flat.extend([0, 0, 0, 0])
            # 참가자 수가 많으면 잘라냄
            flat = flat[:max_participants * 4]
            flattened.append(flat)
        # 컬럼 이름 생성
        columns = []
        for i in range(1, max_participants + 1):
            columns.extend([
                f'p{i}_user_dutch_chance',
                f'p{i}_deposit_amount',
                f'p{i}_time_after_payment',
                f'p{i}_is_name_present'
            ])
        return pd.DataFrame(flattened, columns=columns)

    # 4. 평탄화 및 패딩 적용
    print("Dutchpay 데이터 평탄화 및 패딩 중...")
    dutchpay_flat = flatten_and_pad(dutchpay_df['participants_data'], max_participants=max_participants)
    dutchpay_flat['participants_count'] = dutchpay_df['participants_count']
    dutchpay_flat['total_dutchpay_amount'] = dutchpay_df['total_dutchpay_amount']
    dutchpay_flat['label'] = dutchpay_df['label']
    print(f"Dutchpay 데이터 평탄화 및 패딩 완료. 데이터 형태: {dutchpay_flat.shape}")

    print("Non-Dutchpay 데이터 평탄화 및 패딩 중...")
    non_dutchpay_flat = flatten_and_pad(non_dutchpay_df['participants_data'], max_participants=max_participants)
    non_dutchpay_flat['participants_count'] = non_dutchpay_df['participants_count']
    non_dutchpay_flat['total_dutchpay_amount'] = non_dutchpay_df['total_dutchpay_amount']
    non_dutchpay_flat['label'] = non_dutchpay_df['label']
    print(f"Non-Dutchpay 데이터 평탄화 및 패딩 완료. 데이터 형태: {non_dutchpay_flat.shape}")

    # 5. 두 데이터셋 합치기
    combined_df = pd.concat([dutchpay_flat, non_dutchpay_flat], ignore_index=True)
    print(f"두 데이터셋을 합쳤습니다. 총 데이터 수: {combined_df.shape[0]}")

    # 6. 피처와 레이블 분리
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']
    print("피처와 레이블을 성공적으로 분리했습니다.")

    # 7. 클래스 불균형 처리 (필요 시)
    class_counts = y.value_counts()
    print("레이블 분포:")
    print(class_counts)

    if apply_smote and len(class_counts) > 1:
        print("SMOTE를 사용하여 클래스 불균형을 처리 중...")
        smote = SMOTE(random_state=random_state)
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("SMOTE 적용 완료.")
            print("SMOTE 후 레이블 분포:")
            print(pd.Series(y_resampled).value_counts())
        except Exception as e:
            print(f"SMOTE 적용 중 오류 발생: {e}")
            return None
    else:
        if apply_smote:
            print("SMOTE를 적용할 수 없습니다. 레이블이 하나뿐입니다.")
        X_resampled, y_resampled = X, y
        print("클래스 불균형 처리를 건너뜁니다.")

    # 8. 데이터 분할
    print("데이터를 훈련 세트와 테스트 세트로 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=test_size,
        stratify=y_resampled,
        random_state=random_state
    )
    print(f"데이터 분할 완료. 훈련 세트: {X_train.shape}, 테스트 세트: {X_test.shape}")

    # 9. XGBoost 모델 초기화 및 학습
    print("XGBoost 모델을 초기화하고 학습 중...")
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )

    try:
        model.fit(X_train, y_train)
        print("모델 학습 완료.")
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")
        return None

    # 10. 예측 및 평가
    print("테스트 세트에 대한 예측 및 평가 중...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"모델 정확도: {accuracy:.2f}")
    print("분류 보고서:")
    print(report)

    # 모델 저장
    model.save_model('dutchpay_detection_model.json')
    print("모델이 dutchpay_detection_model.json에 성공적으로 저장되었습니다.")

    return model, X_test, y_test, y_pred, accuracy, report
