import pandas as pd
import numpy as np

# 파라미터 재설정
total_samples = 1000  # 총 데이터 샘플 수
dutchpay_ratio = 0.7  # 1/N 더치페이 비율
max_total_dutchpay = 1000000  # 더치페이 총 금액의 최대값
min_total_dutchpay = 15000  # 더치페이 총 금액의 최소값
name_included_ratio = 0.95  # 더치페이에서 이름 포함 비율
average_time_after_payment = 8 * 60  # 입금까지의 평균 시간
max_total_error_percentages = 10  # 최대 총 금액 오차 범위
random_error_percentages = 60  # 참여자 별 최대 오차 범위
max_participants = 10  # 더치페이에 참여할 최대 인원 수
all_random_percentages = 70  # 완전 랜덤 오차 비율

time_beta_data_a = 1 # 더치페이 시간 데이터 베타 분포 a값
time_beta_data_b = 3 # 더치페이 시간 데이터 베타 분포 b값

# 데이터 저장용 리스트
data = []

def createData():
    # 데이터 생성
    for i in range(total_samples):
        # 랜덤한 더치페이 참가 인원 (최대 30명)
        N = np.random.randint(2, max_participants + 1)

        # 랜덤한 더치페이 총 금액
        total_dutchpay_amount = np.random.randint(min_total_dutchpay, max_total_dutchpay)

        total_dutch_rand = np.random.rand()
        # 45% 확률로 더치페이 총 금액 100원 단위
        if total_dutch_rand < 0.45 and total_dutchpay_amount >= 1000:
            total_dutchpay_amount = (total_dutchpay_amount // 100) * 100
        # 25% 확률로 더치페이 총 금액 1000원 단위
        elif total_dutch_rand < 0.8 and total_dutchpay_amount >= 10000:
            total_dutchpay_amount = (total_dutchpay_amount // 1000) * 1000

        deposit_amounts = [total_dutchpay_amount // N] * N

        # 60%는 1/N 더치페이, 나머지 40%는 랜덤 오차 더치페이
        if np.random.rand() < dutchpay_ratio:
            # 1/N 더치페이
            # 60%는 더치금액 100원 단위로
            if np.random.rand() < 0.6:
                # 그중 60%는 100 단위 내림
                if np.random.rand() < 0.6:
                    cur_amount = int(np.floor(deposit_amounts[0] / 100) * 100)
                    deposit_amounts = [cur_amount] * N


                # 나머지 40%는 100 단위 올림
                else:
                    cur_amount = int(np.floor(deposit_amounts[0] / 100) * 100)
                    deposit_amounts = [cur_amount] * N

        # 오차 더치페이
        else:

            # 완전 랜덤 오차
            if np.random.rand() < all_random_percentages:
                # 더치페이 참여자 별 오차 범위
                for i in range(N):
                    random_error = np.random.uniform(1, random_error_percentages + 1)
                    deposit_differents = [0] * N
                    different = deposit_amounts[0] * random_error // 100

                    unit_different_rand = np.random.rand()
                    if unit_different_rand < 0.6 and different >= 1000:
                        different = different * 100 // 100
                    elif unit_different_rand < 0.8 and different >= 10000:
                        different = different * 1000 // 1000

                    target_idx = np.random.randint(0, N)  # 넘겨줄 인덱스
                    deposit_differents[target_idx] += different
                    deposit_differents[i] -= different

                for k in range(N):
                    deposit_amounts[k] += deposit_differents[k]

            else:
                for i in range(np.random.randint(2, N + 1) // 2):
                    random_error = np.random.uniform(1, random_error_percentages + 1)
                    different = deposit_amounts[0] * random_error // 100

                    unit_different_rand = np.random.rand()
                    if unit_different_rand < 0.6 and deposit_differents >= 1000:
                        different = different * 100 // 100
                    elif unit_different_rand < 0.8 and deposit_differents >= 10000:
                        different = different * 1000 // 1000

                    deposit_amounts[i] += different
                    deposit_amounts[N - i] -= different

        # 더치페이 이름 포함 여부 (95% 확률로 포함)
        is_name_present = [1 if np.random.rand() < name_included_ratio else 0 for _ in range(N)]

        # 입금까지의 시간 (베타 분포 적용)
        time_after_payment = np.random.beta(time_beta_data_a, time_beta_data_b, N)

        label = 1

        # 각 사용자의 데이터 생성
        for pk in range(N):
            data.append({
                'user_pk': pk + 1,
                'deposit_amount': deposit_amounts[pk],
                'total_dutchpay_amount': total_dutchpay_amount,
                'time_after_payment': time_after_payment[pk],
                'is_name_present': is_name_present[pk],
                'label': label
            })

    # DataFrame 생성
    df_generated = pd.DataFrame(data)

    # 참가 인원 정보 추가 (participants_count)
    df_generated['participants_count'] = df_generated.groupby('total_dutchpay_amount')['user_pk'].transform('count')

    # 참가자별 정보를 그룹핑하여 하나의 행으로 표현
    grouped_df = df_generated.groupby('total_dutchpay_amount').apply(lambda x: pd.Series({
        'participants_count': x['participants_count'].iloc[0],
        'participants_data': x[['deposit_amount', 'user_pk', 'time_after_payment', 'is_name_present']].to_dict('records'),
        'label': x['label'].iloc[0]  # 더치페이 여부
    })).reset_index()

    # 수정된 데이터를 CSV로 저장
    csv_file_path = './data/dutch_pay_grouped_data_corrected.csv'
    grouped_df.to_csv(csv_file_path, index=False)

    # 경로 반환
    return csv_file_path
