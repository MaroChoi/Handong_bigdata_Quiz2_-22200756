import pandas as pd

# 1. 불러오기
df = pd.read_excel('서울대기오염_2019.xlsx')

# 2. 컬럼 정리
df.columns = [
    '날짜', '측정소명', '미세먼지', '초미세먼지',
    '오존', '이산화질소', '일산화탄소', '아황산가스'
]

# 3. 요약 행 제거 (날짜가 정상 형식이 아닌 행 제거)
df = df[pd.to_datetime(df['날짜'], errors='coerce').notnull()]
df['날짜'] = pd.to_datetime(df['날짜'])

# 4. 구조 및 결측치 확인
print(df.info())
print(df.isnull().sum())

# 1. 결측치 전체 평균으로 대체
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2. 파생 변수 생성
df['월'] = df['날짜'].dt.month
df['요일'] = df['날짜'].dt.day_name()

# 3. 등급 생성 함수
def get_pm_grade(pm):
    if pm <= 30:
        return '좋음'
    elif pm <= 80:
        return '보통'
    elif pm <= 150:
        return '나쁨'
    else:
        return '매우나쁨'

# 4. 적용
df['미세먼지_등급'] = df['미세먼지'].apply(get_pm_grade)
df['초미세먼지_등급'] = df['초미세먼지'].apply(get_pm_grade)

# 5. 확인
print(df.isnull().sum())  # 전부 0이어야 함
print(df[['미세먼지', '미세먼지_등급', '초미세먼지', '초미세먼지_등급']].head())

#시계열이고, 차분 1의 ARIMA로 행정 예측 분석을 하는게 좋아보인다

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. 전체 평균 미세먼지 시계열
pm_series = df[df['측정소명'] == '평균'].groupby('날짜')['미세먼지'].mean()

# 2. 정상성 확인 (ADF Test)
adf_result = adfuller(pm_series)
print("ADF p-value:", adf_result[1])  # > 0.05 → 비정상성

# 3. ARIMA 모델 학습
model = ARIMA(pm_series, order=(1, 1, 1))
model_fit = model.fit()

# 4. 7일 예측
forecast = model_fit.forecast(steps=7)
print("7일 예측 결과:")
print(forecast)

# 5. 시각화
plt.figure(figsize=(12, 4))
pm_series.plot(label='실측', color='blue')
forecast.plot(label='7일 예측', color='red', linestyle='--')
plt.title('ARIMA(1,1,1) 기반 미세먼지 7일 예측')
plt.xlabel('날짜')
plt.ylabel('μg/m³')
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
pm_series.plot(label='실측', color='blue')
forecast.plot(label='7일 예측', color='red', linestyle='--')
plt.title('ARIMA(1,1,1) 기반 평균 미세먼지 7일 예측')
plt.ylabel('μg/m³')
plt.xlabel('날짜')
plt.legend()
plt.grid()
plt.show()
