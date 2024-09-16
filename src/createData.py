import pandas as pd
import numpy as np
from src.dataToJson import data_to_json
import matplotlib.pyplot as plt

# 파라미터 재설정
dutchpay_ratio = 0.8  # 1/N 더치페이 비율
name_included_ratio = 0.97  # 더치페이에서 이름 포함 비율
max_total_error_percentages = 10  # 최대 총 금액 오차 범위
random_error_percentages = 60  # 참여자 별 최대 오차 범위
max_participants = 30  # 더치페이에 참여할 최대 인원 수
all_random_percentages = 10  # 완전 랜덤 오차 비율

time_beta_data_a = 1 # 더치페이 시간 데이터 베타 분포 a값
time_beta_data_b = 5 # 더치페이 시간 데이터 베타 분포 b값

dutch_amount_beta_a = 1.2 # 참여자 입금 더치페이 금액 베타 분포 a
dutch_amount_beta_b = 15 # 참여자 입금 더치페이 금액 베타 분포 b

max_dutchpay_amount = 500000  # 참여자 더치페이 금액의 최대값
min_dutchpay_amount = 1500  # 참여자 더치페이 금액의 최소값

user_dutch_chance_a = 8 # 사용자 더치페이 확률 베타 분포 a
user_dutch_chance_b = 5 # 사용자 더치페이 확률 베타 분포 b

user_non_dutch_chance_a = 2.3 # 사용자 논더치페이 확률 베타 분포 a
user_non_dutch_chance_b = 5 # 사용자 논더치페이 확률 베타 분포 b

total_dutch_participant_beta_a = 1.2 # 총 더치페이 인원 베타 분포 a
total_dutch_participant_beta_b = 3 # 총 더치페이 인원 베타 분포 b
total_dutch_participant_beta_shift = 0.07 # 총 더치페이 인원 베타 분포 shift 값 (최소 인원 설정 용)

non_dutch_min_error_percentages = 50 # non-dutch 최소 오차 범위
non_dutch_max_error_percentages = 90 # non-dutch 최대 오차 범위

max_time_after_payment = 10080 # 최대 입금 시간 차이 (분 단위), 7일

# 분포 그래프 이미지 이름
participant_count_graph_name = 'participant_count'
time_after_payment_graph_name = 'time_after_payment'
deposit_amount_graph_name = 'deposit_amount'
user_dutch_chance_graph_name = 'user_dutch_chance'


def createData(dutch_data_counts, non_dutch_data_counts):
    createDutchData(dutch_data_counts)
    data_to_json(True)

    createNonDutchData(non_dutch_data_counts)
    data_to_json(False)


def createDutchData(sample_count):
    # 데이터 저장용 리스트
    data = []

    # 그래프 분포 용 데이터
    participant_counts_graph_data = []
    time_after_payment_graph_data = []
    deposit_amount_graph_data = []
    user_dutch_chance_graph_data = []

    # 더치페이 데이터 생성 (label = 1)
    for i in range(sample_count):
        # 랜덤한 더치페이 참가 인원 (최대 30명)
        N = int((np.random.beta(total_dutch_participant_beta_a, total_dutch_participant_beta_b) + total_dutch_participant_beta_shift) * 30) // 1;

        # 더치페이 금액
        deposit_amounts = [int(np.random.beta(dutch_amount_beta_a, dutch_amount_beta_b) * max_dutchpay_amount + min_dutchpay_amount)] * N

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
                    different = int(deposit_amounts[0] * random_error) // 100

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
                    different = int(deposit_amounts[0] * random_error) // 100

                    unit_different_rand = np.random.rand()
                    if unit_different_rand < 0.6 and deposit_differents >= 1000:
                        different = different * 100 // 100
                    elif unit_different_rand < 0.8 and deposit_differents >= 10000:
                        different = different * 1000 // 1000

                    deposit_amounts[i] += different
                    deposit_amounts[N - i] -= different

        # 더치페이 이름 포함 여부 (97% 확률로 포함)
        is_name_present = [1 if np.random.rand() < name_included_ratio else 0 for _ in range(N)]

        # 입금까지의 시간 (베타 분포 적용)
        time_after_payment = np.random.beta(time_beta_data_a, time_beta_data_b, N)

        label = 1

        # 각 사용자의 데이터 생성
        participant_data = []
        for pk in range(N):

            user_dutch_chance = np.random.beta(user_dutch_chance_a, user_dutch_chance_b)
            deposit_amount = deposit_amounts[pk]
            time_after_payment_trans = int(time_after_payment[pk] * max_time_after_payment)

            user_dutch_chance_graph_data.append(user_dutch_chance)
            deposit_amount_graph_data.append(deposit_amount)
            time_after_payment_graph_data.append(time_after_payment_trans)

            participant_data.append({
                'user_dutch_chance': user_dutch_chance,
                'deposit_amount': deposit_amount,
                'time_after_payment': time_after_payment_trans,
                'is_name_present': is_name_present[pk]
            })

        data.append({
            'total_dutchpay_amount': sum(deposit_amounts),
            'participants_count': N,
            'participants_data': participant_data,
            'label': label
        })

        participant_counts_graph_data.append(N)


    # DataFrame 생성
    df = pd.DataFrame(data)

    # 수정된 데이터를 CSV로 저장
    csv_file_path = './data/generated_dutch_pay_data.csv'
    print(f"CSV 파일이 ${csv_file_path}에 저장되었습니다.")
    df.to_csv(csv_file_path, index=False)

    save_graph_images(participant_counts_graph_data, time_after_payment_graph_data, deposit_amount_graph_data, user_dutch_chance_graph_data, True)

    # 경로 반환
    return csv_file_path

# 음성(non-dutch) 데이터
def createNonDutchData(sample_count):
    # 데이터 저장용 리스트
    data = []

    # 그래프 분포 용 데이터
    participant_counts_graph_data = []
    time_after_payment_graph_data = []
    deposit_amount_graph_data = []
    user_dutch_chance_graph_data = []

    for i in range(sample_count):
        # 랜덤한 더치페이 참가 인원 (최대 30명)
        N = int((np.random.beta(total_dutch_participant_beta_a,
                             total_dutch_participant_beta_b) + total_dutch_participant_beta_shift) * max_participants);

        random = np.random.beta(dutch_amount_beta_a, dutch_amount_beta_b)

        # 더치페이 금액
        deposit_amounts = [int(random * max_dutchpay_amount + min_dutchpay_amount)] * N

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

        # 완전 랜덤 오차
        # 더치페이 참여자 별 오차 범위
        deposit_differents = [0] * N
        for i in range(N):
            # 오차 범위 최소 ~ 최대 사이 값으로 설정
            random_error = np.random.uniform(non_dutch_min_error_percentages, non_dutch_max_error_percentages + 1)

            different = int(deposit_amounts[0] * random_error) // 100

            unit_different_rand = np.random.rand()
            if unit_different_rand < 0.6 and different >= 1000:
                different = different // 100 * 100
            elif unit_different_rand < 0.8 and different >= 10000:
                different = different // 1000 * 1000

            target_idx = np.random.randint(0, N)  # 넘겨줄 인덱스
            deposit_differents[target_idx] += different
            deposit_differents[i] -= different

        for k in range(N):
            deposit_amounts[k] += deposit_differents[k]

        # 더치페이 이름 포함 여부 (50% 확률로 포함)
        is_name_present = [1 if np.random.rand() < 0.5 else 0 for _ in range(N)]


        label = 0

        # 각 사용자의 데이터 생성
        participant_data = []
        for pk in range(N):
            user_dutch_chance = np.random.beta(user_non_dutch_chance_a, user_non_dutch_chance_b)
            deposit_amount = deposit_amounts[pk]
            time_after_payment_trans = int(np.random.rand() * max_time_after_payment)

            user_dutch_chance_graph_data.append(user_dutch_chance)
            deposit_amount_graph_data.append(deposit_amount)
            time_after_payment_graph_data.append(time_after_payment_trans)

            participant_data.append({
                'user_dutch_chance': user_dutch_chance,
                'deposit_amount': deposit_amount,
                'time_after_payment': time_after_payment_trans,
                'is_name_present': is_name_present[pk]
            })

        data.append({
            'total_dutchpay_amount': sum(deposit_amounts),
            'participants_count': N,
            'participants_data': participant_data,
            'label': label
        })

        participant_counts_graph_data.append(N)

    # DataFrame 생성
    df = pd.DataFrame(data)

    # 수정된 데이터를 CSV로 저장
    csv_file_path = './data/generated_non_dutch_pay_data.csv'
    print(f"CSV 파일이 ${csv_file_path}에 저장되었습니다.")
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    save_graph_images(participant_counts_graph_data, time_after_payment_graph_data, deposit_amount_graph_data, user_dutch_chance_graph_data, False)

    # 경로 반환
    return csv_file_path


def save_graph_image(data, image_name, title, is_dutch, bins=316):
    # 한글 폰트 설정 (Windows의 경우 맑은 고딕 사용)
    plt.rcParams['font.family'] = 'Malgun Gothic'

    # 그래프 그리기
    plt.figure(figsize=(4, 6))
    plt.hist(data, bins=bins, color='green', edgecolor='green')  # 히스토그램 형식으로 데이터 분포 표시
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    if is_dutch:
        file_path = './template/graph/dutch/' + image_name + '.png'
    else:
        file_path = './template/graph/non-dutch/' + image_name + '.png'

    # 그래프 이미지 저장
    plt.savefig(file_path)  # 원하는 경로로 저장 가능
    plt.show()


def save_graph_images(participant_counts_graph_data, time_after_payment_graph_data, deposit_amount_graph_data, user_dutch_chance_graph_data, is_dutch):
    # 데이터 분포 그래프 이미지 저장
    save_graph_image(participant_counts_graph_data, participant_count_graph_name, "더치페이 참여자 수", is_dutch, 23)
    save_graph_image(time_after_payment_graph_data, time_after_payment_graph_name, "더치페이 후 입금 시간", is_dutch)
    save_graph_image(deposit_amount_graph_data, deposit_amount_graph_name, "개인 별 더치페이 금액", is_dutch)
    save_graph_image(user_dutch_chance_graph_data, user_dutch_chance_graph_name, "거래 내역 당 더치페이", is_dutch)