import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 1. 데이터 불러오기
data = pd.read_csv("SeoulBikeData_1000.csv")

# 2. 입력값과 출력값 설정
X = data[["Temperature(°C)"]]
y = data["Rented Bike Count"]


# 3. 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. 입력값 정규화
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. 인공신경망 만들기
model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    max_iter=1000,
    random_state=42
)


# 6. 인공신경망 학습
model.fit(X_train, y_train)


# 7. 예측
y_pred = model.predict(X_test)


# 8. 결과 확인
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== 자전거 대여량 예측 결과 =====")
print("MSE:", mse)
print("R²:", r2)


# 9. 실제값과 예측값 그래프
plt.scatter(y_test, y_pred)

plt.xlabel("실제 자전거 대여량")
plt.ylabel("예측 자전거 대여량")
plt.title("실제값과 예측값 비교")

plt.show()
