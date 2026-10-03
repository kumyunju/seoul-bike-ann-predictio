import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
data = pd.read_csv("SeoulBikeData_1000.csv")

X = data[["Temperature(°C)"]]
y = data[["Rented Bike Count"]]

# -----------------------------
# 2. 데이터 정규화
# -----------------------------
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y).ravel()

# -----------------------------
# 3. ANN 모델 생성
# -----------------------------
model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    max_iter=3000,
    learning_rate_init=0.01,
    random_state=42
)

model.fit(X_scaled, y_scaled)

# -----------------------------
# 4. 웹페이지 화면
# -----------------------------
st.title("🚲 자전거 대여량 예측 프로그램")

st.write("기온을 입력하면 ANN을 이용하여 예상 자전거 대여량을 계산합니다.")

temperature = st.number_input(
    "기온을 입력하세요 (°C)",
    min_value=-30.0,
    max_value=40.0,
    value=20.0,
    step=0.1
)

# -----------------------------
# 5. 예측
# -----------------------------
if st.button("자전거 대여량 예측"):

    new_temperature = x_scaler.transform(
        [[temperature]]
    )

    predicted_scaled = model.predict(new_temperature)

    predicted_count = y_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    )[0][0]

    predicted_count = max(0, round(predicted_count))

    st.success(
        f"예상 자전거 대여량: {predicted_count:,} 대"
    )
