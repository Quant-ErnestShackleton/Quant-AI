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

### 더치페이 참여 인원

대부분 10명 이내의 소규모 인원, 대규모 인원에서도 더치페이 사용하는 경우 O

=> 베타 분포로 난수 구현




### 더치페이 시간 별 분류

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

