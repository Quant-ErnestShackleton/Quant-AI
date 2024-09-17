# 데이터 전처리 과정 (평탄화 및 패딩)
import pandas as pd

def flatten_and_pad(participants_data, max_participants=50):
    """
    participants_data를 평탄화하고 패딩하여 고정된 크기의 피처로 변환

    Parameters:
    - participants_data (List[dict]): 참가자 데이터 리스트
    - max_participants (int): 최대 참가자 수

    Returns:
    - DataFrame: 전처리된 피처 데이터프레임
    """
    flattened = []
    for participant in participants_data:
        flat = [
            participant.get('user_dutch_chance', 0),
            participant.get('deposit_amount', 0),
            participant.get('time_after_payment', 0),
            participant.get('is_name_present', 0)
        ]
        flattened.append(flat)

    # 패딩: 참가자 수가 부족하면 0으로 채움
    while len(flattened) < max_participants:
        flattened.append([0, 0, 0, 0])

    # 참가자 수가 많으면 잘라냄
    flattened = flattened[:max_participants]

    # 피처 평탄화
    flat_features = []
    for flat in flattened:
        flat_features.extend(flat)

    # 패딩이 필요한 경우 추가 패딩
    total_features = max_participants * 4
    if len(flat_features) < total_features:
        flat_features.extend([0] * (total_features - len(flat_features)))
    else:
        flat_features = flat_features[:total_features]

    # 컬럼 이름 생성
    columns = []
    for i in range(1, max_participants + 1):
        columns.extend([
            f'p{i}_user_dutch_chance',
            f'p{i}_deposit_amount',
            f'p{i}_time_after_payment',
            f'p{i}_is_name_present'
        ])

    return pd.DataFrame([flat_features], columns=columns)

def preprocess_input(data, max_participants=50):
    """
    입력 데이터를 전처리하여 모델이 예측할 수 있는 형태로 변환

    Parameters:
    - data (dict): API를 통해 받은 JSON 데이터
    - max_participants (int): 최대 참가자 수

    Returns:
    - DataFrame: 전처리된 피처 데이터프레임
    """
    participants_data = data.get('participants_data', [])
    total_dutchpay_amount = data.get('total_dutchpay_amount', 0)
    participants_count = data.get('participants_count', 0)

    # 평탄화 및 패딩
    flat_df = flatten_and_pad(participants_data, max_participants)
    flat_df['participants_count'] = participants_count
    flat_df['total_dutchpay_amount'] = total_dutchpay_amount


    return flat_df
