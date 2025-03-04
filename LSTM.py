import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import akshare as ak
import ta

# 获取赛力斯（601127）股票数据
stock_code = "601127"
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")

# 处理日期格式
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)

# 仅保留最近 180 天数据（增加数据点）
df = df.iloc[-180:]

# 计算技术指标
df["MA5"] = df["收盘"].rolling(window=5).mean()
df["MA10"] = df["收盘"].rolling(window=10).mean()
df["RSI"] = ta.momentum.RSIIndicator(df["收盘"], window=14).rsi()
df["MACD"], df["MACD_signal"], df["MACD_hist"] = ta.trend.MACD(df["收盘"]).macd(), ta.trend.MACD(df["收盘"]).macd_signal(), ta.trend.MACD(df["收盘"]).macd_diff()

# 删除缺失值
df.dropna(inplace=True)

# 选择训练特征
features = ["收盘", "MA5", "MA10", "RSI", "MACD", "MACD_signal"]

# 归一化数据
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=features)

# 设置窗口大小
window_size = 30  # 窗口30天

# 构造 LSTM 输入数据
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length]["收盘"])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, window_size)

# 确保 X_test 不会过少
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 训练 LSTM 模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, len(features))),
    Dropout(0.3),
    LSTM(50, return_sequences=False),
    Dropout(0.3),
    Dense(25, activation="relu"),
    Dense(1)
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

# 训练模型
history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test))

# 预测股价
y_pred = model.predict(X_test)

# 反归一化预测值（确保没有拼接错误）
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred.reshape(-1, 1), np.zeros((len(y_pred), len(features) - 1)))))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1)))))[:, 0]


# **绘图（修正未来5天问题）**
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_actual)), y_test_actual, label="Actual Price", color="blue", linestyle="-")
plt.plot(range(len(y_pred_actual)), y_pred_actual, label="Predicted Price", color="red", linestyle="-")

# 未来 5 天预测
last_30_days = df_scaled[-window_size:].values.reshape(1, window_size, len(features))
future_prices = []

for _ in range(5):
    next_price = model.predict(last_30_days)
    future_prices.append(next_price[0][0])
    new_data = np.hstack((next_price.reshape(-1, 1), last_30_days[:, -1, 1:].reshape(1, -1)))
    last_30_days = np.append(last_30_days[:, 1:, :], [new_data], axis=1)

future_prices_actual = scaler.inverse_transform(np.hstack((np.array(future_prices).reshape(-1, 1), np.zeros((len(future_prices), len(features) - 1)))))[:, 0]

# **未来股价单独画图**
future_x = range(len(y_test_actual), len(y_test_actual) + 5)
plt.plot(future_x, future_prices_actual, label="Future Prediction", color="green")

plt.title("赛力斯股价预测 - LSTM + 技术指标")
plt.xlabel("天数")
plt.ylabel("价格")
plt.legend()
plt.show()
