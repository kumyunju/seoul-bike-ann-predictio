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
y = data[["Rented Bike Count"]]


# 3. 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. 입력값 정규화
x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)


# 5. 출력값도 정규화
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train).ravel()


# 6. 인공신경망 만들기
model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    max_iter=3000,
    learning_rate_init=0.01,
    random_state=42
)


# 7. 인공신경망 학습
model.fit(X_train_scaled, y_train_scaled)


# 8. 예측
y_pred_scaled = model.predict(X_test_scaled)

# 정규화된 예측값을 원래 자전거 대여량으로 변환
y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()


# 9. 결과 계산
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("===== 자전거 대여량 ANN 예측 결과 =====")
print("MSE:", mse)
print("R²:", r2)


# 10. 실제값과 예측값 비교 그래프
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("실제 자전거 대여량")
plt.ylabel("예측 자전거 대여량")
plt.title("실제값과 ANN 예측값 비교")

# 실제값 = 예측값 기준선
min_value = min(y_test["Rented Bike Count"].min(), y_pred.min())
max_value = max(y_test["Rented Bike Count"].max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.tight_layout()
plt.show()
# 11. 사용자가 기온을 입력하여 대여량 예측
temperature = float(input("예측할 기온을 입력하세요(°C): "))

# 입력한 기온을 정규화
new_temperature = x_scaler.transform(
    pd.DataFrame([[temperature]], columns=["Temperature(°C)"])
)

# 자전거 대여량 예측
predicted_scaled = model.predict(new_temperature)

# 원래 대여량 단위로 변환
predicted_count = y_scaler.inverse_transform(
    predicted_scaled.reshape(-1, 1)
)[0][0]

print()
print("===== 새로운 기온의 자전거 대여량 예측 =====")
print("입력한 기온:", temperature, "°C")
print("예상 자전거 대여량:", round(predicted_count), "대")
