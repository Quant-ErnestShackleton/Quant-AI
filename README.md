# 더치페이 AI 모델

1. 이상치 탐지
	* 학습된 데이터를 기준으로 기존과 다른 이상 현상을 탐색
	* 만약 사용한다면 일반적인 결제 내역(쇼핑, 식당)과 다르게 계좌 이체가 연속으로 입금 되는 경우를 이상 현상으로 구분 ^0e4fee
2. 패턴 탐색
	* 데이터에서 패턴을 학습하고, 특정 패턴이 존재하는 지 구분
	* 계좌 이체가 연속으로 입금되는 경우를 패턴으로 인식하면, 더치페이로 탐색


* 이상치 탐지의 경우보다 패턴 탐색 기법을 이용하는 것이 "더치페이 탐색"에서 더욱 적합하다고 판단


* **정확도를 높이기 위해 이후 사용자의 이체 내역(상대방 정보)를 데이터로 저장하여 정확도 구분 추가(09.15)**


# 패턴 탐색

## 1. 패턴 탐색 모델 선정

### Random Forest

- 여러 개의 결정 트리(decision tree)를 만들어 다수결로 최종 결론을 내는 앙상블 모델.
- **장점**:
    - 사용이 매우 쉽고, 데이터 전처리를 많이 하지 않아도 됨.
    - 다양한 특징(변수)을 잘 처리함.
- **단점**:
    - 복잡한 관계를 학습하는 데는 한계가 있을 수 있음.
    - 학습 속도가 느릴 수 있음.

### Logistic Regression

- 이진 분류 문제를 해결하는 선형 모델로, 데이터를 두 가지 클래스로 나누는 데 사용.
- **장점**:
    - 간단한 데이터일 때 빠르고 성능이 좋음.
    - 결과가 직관적이고 해석하기 쉬움.
- **단점**:
    - 복잡한 비선형 관계를 처리하기 어려움.
    - 더치페이처럼 여러 특징이 있는 경우 성능이 떨어질 수 있음.


###  XGBoost

- Gradient Boosting을 기반으로 한 앙상블 모델로, 성능과 효율성을 극대화한 모델.
- **장점**:
    - 매우 높은 정확도를 제공함.
    - 과적합(overfitting)을 방지하는 정규화 기능을 제공함.
- **단점**:
    - 모델 설정이 복잡하고, 하이퍼파라미터 튜닝이 필요함.
    - 메모리와 연산 자원을 많이 소모할 수 있음.



###  LSTM (Long Short-Term Memory)

- 시계열 데이터를 다루는 순환 신경망(RNN) 모델로, 시간에 따라 변하는 패턴을 학습하는 데 강함.
- **장점**:
    - 거래 내역처럼 시간에 따른 의존성이 있는 데이터를 처리하는 데 유리함.
    - 과거 데이터를 기억하고 이를 바탕으로 예측할 수 있음.
- **단점**:
    - 학습 시간이 길고, 모델 구조가 복잡함.
    - 딥러닝 모델이므로 GPU 등의 자원이 필요할 수 있음.


정확도가 높은 **XGBoost**를 이용하기로 결정
하이퍼파라미터 튜닝은 Auto-ML로

# 학습 데이터

더치페이에 대한 데이터를 직접 구하기 어렵고, 데이터의 양 또한 적기 때문에 직접 데이터를 생성


## 데이터 생성 참고 사항

더치페이 데이터 생성 시 구분해야 할 과정

* 더치페이 참여 인원
	* 소규모 (10명 이내)의 데이터가 더 많아야 함
	  
* 1/N로 더치페이 혹은 구성원 마다 다른 금액 더치페이
	* 현재까지는 1/N 더치페이의 경우가 더 많아보임
	  
* 더치페이 총 금액의 최대, 최소 값 
* 결제 후 구성원의 입금까지 걸린 시간
	* 시간을 기준으로 결제 후 가장 많은 비율로
* 계좌 내역에서 '이체'로 구분된 경우에는 더치페이일 확률 높게

---
## ~~더치페이 총 금액 분포~~

~~최대값 **2,000,000**을 기준(value = 1.0) 으로 작성, 8만원 기준 가장 높은 비율~~

~~a = 1.1, b= 5~~

![image](https://github.com/user-attachments/assets/a2711ae7-4aae-41d0-83c5-afae7d0ccaa0)


```python
import numpy as np  
import matplotlib.pyplot as plt  
  
a, b = 1.1, 5 # a는 커지면서 상승 속도 결정, b는 내려가는 구간 조정  
beta_data = np.random.beta(a, b, size=1000000)  
  
print(beta_data)  
  
# Plot the distribution  
plt.figure(figsize=(10, 8))  
plt.hist(beta_data, bins=1000, density=True, alpha=0.7, label=f'Beta Distribution a={a}, b={b}')  
plt.title('Beta Distribution')  
plt.xlabel('Value')  
plt.ylabel('Density')  
plt.legend()  
plt.show()
```

09.15 수정 

더치페이 총 금액보다는 참여자 별 더치페이 금액으로, 2~3만원 데이터가 많이 나오도록 설정
a = 1.2, b = 15

![image](https://github.com/user-attachments/assets/e5e32f64-a55a-42a0-b237-f4b4e0ba234d)

---

### 더치페이 참여 인원

대부분 10명 이내의 소규모 인원, 대규모 인원에서도 더치페이 사용하는 경우 O

=> 베타 분포로 난수 구현

(09.15 추가)

a =1.2, b = 3으로 설정, 최대 값(value = 1.0)은 30명 기준
**shift 0.07 추가하기** (최소 2명으로 설정하기 위해)

![image](https://github.com/user-attachments/assets/ad9eae72-5d44-431f-a529-fd36c31bbd5e)



### 더치페이 후 입금 시간 별 분류

더치페이 후 짧은 시간 내에 송금이 대부분 (예상)
-> 3가지 분포 예상 가능


![image](https://github.com/user-attachments/assets/92eaf4b0-71e0-49a4-8d64-9ca15ca9fb33)

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for exponential, gamma, and lognormal distributions
exp_data = np.random.exponential(scale=1.0, size=1000)
gamma_data = np.random.gamma(shape=2.0, scale=1.0, size=1000)
lognormal_data = np.random.lognormal(mean=0, sigma=1, size=1000)

# Plot the distributions
plt.figure(figsize=(12, 8))

# Exponential Distribution
plt.subplot(3, 1, 1)
plt.hist(exp_data, bins=50, density=True, alpha=0.7, color='b')
plt.title("Exponential Distribution")
plt.xlabel('Value')
plt.ylabel('Density')

# Gamma Distribution
plt.subplot(3, 1, 2)
plt.hist(gamma_data, bins=50, density=True, alpha=0.7, color='g')
plt.title("Gamma Distribution")
plt.xlabel('Value')
plt.ylabel('Density')

# Lognormal Distribution
plt.subplot(3, 1, 3)
plt.hist(lognormal_data, bins=50, density=True, alpha=0.7, color='r')
plt.title("Lognormal Distribution")
plt.xlabel('Value')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

```


베타 분포로 결정 (결제 후 바로 입금하는 경우 고려)

![image](https://github.com/user-attachments/assets/cb9b0619-ace7-4c2c-b0f6-c16092f94d29)

```python
import numpy as np
import matplotlib.pyplot as plt

# 베타 분포로 0~0.5에서 급격히 상승하는 분포 생성
a, b = 1, 3  # a는 커지면서 상승 속도 결정, b는 내려가는 구간 조정
beta_data = np.random.beta(a, b, size=1000)

print(beta_data)

# Plot the distribution
plt.figure(figsize=(8, 6))
plt.hist(beta_data, bins=50, density=True, alpha=0.7, label=f'Beta Distribution a={a}, b={b}')
plt.title('Beta Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```
---

# 음성 데이터(Non-dutch) 추가 (09.15)

## Non-dutch 데이터 특징

1. 입금 금액의 변동성이 너무 큰 경우에는 Non-dutch로 구분 
	* 현재 dutch로 인식되는 데이터에 변동성이 있는 경우 포함(최대 20%) 
	  
2. 참여자와의 더치페이 확률(더치페이 횟수 / 총 거래 횟수) 여부
	*  베타 분포 사용, a = 2.3, b = 5
		<br>
		\<non-dutch 에서의 참여자와의 더치페이 확률 분포\> 
		![image](https://github.com/user-attachments/assets/70d87706-a208-44ab-9662-3a782c2a8be9)

		\<dutch 에서의 참여자와의 더치페이 확률 분포\> 
		![image](https://github.com/user-attachments/assets/cdae4988-4468-49b9-a318-d272842acde2)



# 데이터 평탄화 및 패딩 

현재 데이터 형식을 나타내면 다음과 같은 3차원 데이터 형식이다.

```json
{  
	"total_dutchpay_amount": 160000,  
	"participants_count": 4,  
	"participants_data": [  
		{  
		"user_dutch_chance": 0.743996153142264,  
		"deposit_amount": 40000,  
		"time_after_payment": 901,  
		"is_name_present": 0  
		},  
		{  
		"user_dutch_chance": 0.7993342355832653,  
		"deposit_amount": 40000,  
		"time_after_payment": 2000,  
		"is_name_present": 1  
		},  
		{  
		"user_dutch_chance": 0.6394824521928971,  
		"deposit_amount": 40000,  
		"time_after_payment": 925,  
		"is_name_present": 1  
		},  
		{  
		"user_dutch_chance": 0.6180864173310338,  
		"deposit_amount": 40000,  
		"time_after_payment": 367,  
		"is_name_present": 1  
		}  
	],  
	"label": 1  
}
```

기존에 구상했던 XGBoost 모델은 **2차원 데이터 학습 모델**이다. 따라서 위와 같은 3차원 데이터를 이용하려면 다음 두 가지 중 선택해야 한다.

* **3차원 데이터를 평탄화**
	* 2차원 데이터로 변경하여 기존의 모델을 사용하는 방법이다. 위의 "participants_data" 부분이 현재 2차원 배열인데, 이를 1차원 배열로 만들어서 학습하면 된다.
	
	* 현재  "participants_data"데이터의 수는 "participants_count"의 값과 같다. 따라서 매번 다른 크기의 "participants_data"가 들어오는데, 이 때 수를 고정해야 2차원 배열을 만들 수 있다. "participants_count"를 고정 값으로 주고, 적거나 많은 값은 패딩과 커팅을 이용한다.


* **3차원 데이터 모델을 이용**
	* 딥러닝 모델들은 3차원 데이터를 이용하여 학습 시킨다. 이러한 모델을 이용하여 학습하면 평탄화와 같은 과정을 거치지 않고 3차원 데이터를 학습하면 된다. 하지만 그만큼 많은 데이터 양과, 자원이 필요하다.


주로 3차원 데이터를 학습하는 모델은 피처간의 관계가 모델 성능에 중요한 경우 사용한다. 하지만 위의 경우에는 피처(참가자) 간의 순서나 관계가 중요한 경우가 아니다. 게다가 딥러닝 모델은 많은 데이터를 필요로 하는데, 실제 테스트 데이터의 양도 적어 **기존의 3차원 데이터를 평탄화 한 후, 기존의 XGBoost 모델을 사용**하기로 하였다.

XGBoost 학습 코드

<train.py>

```python 
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
	- max_participants (int): 평탄화 시 최대 참가자 수. 기본값은 50.- test_size (float): 테스트 세트의 비율. 기본값은 0.2 (20%).- random_state (int): 재현성을 위한 랜덤 시드. 기본값은 42.- apply_smote (bool): 클래스 불균형 처리를 위해 SMOTE를 적용할지 여부. 기본값은 True.  
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
		print(f"더치페이 CSV 파일 '{dutchpay_csv_path}'을 성공적으로 읽어왔습니다.")  
	except Exception as e:  
		print(f"더치페이 CSV 파일을 읽어오는 데 실패했습니다: {e}")  
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
	print("더치페이 데이터 평탄화 및 패딩 중...")  
	dutchpay_flat = flatten_and_pad(dutchpay_df['participants_data'], max_participants=max_participants)  
	dutchpay_flat['total_dutchpay_amount'] = dutchpay_df['total_dutchpay_amount']  
	dutchpay_flat['label'] = dutchpay_df['label']  
	print(f"더치페이 데이터 평탄화 및 패딩 완료. 데이터 형태: {dutchpay_flat.shape}")  
	  
	print("Non-Dutchpay 데이터 평탄화 및 패딩 중...")  
	non_dutchpay_flat = flatten_and_pad(non_dutchpay_df['participants_data'], max_participants=max_participants)  
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
	  
	return model, X_test, y_test, y_pred, accuracy, report
```