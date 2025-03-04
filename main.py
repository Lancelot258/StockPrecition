import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta  # 技术指标库
import os

# 解决中文显示问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# **从 CSV 文件读取数据**
file_path = "stock_data.csv"
df = pd.read_csv(file_path)

# **处理日期格式**
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)

# 只保留最近 120 天数据
df = df.iloc[-120:]

# **计算技术指标**
df["MA5"] = df["收盘"].rolling(window=5).mean()
df["MA10"] = df["收盘"].rolling(window=10).mean()
df["RSI"] = ta.momentum.RSIIndicator(df["收盘"], window=14).rsi()
df["MACD"], df["MACD_signal"], df["MACD_hist"] = ta.trend.MACD(df["收盘"]).macd(), ta.trend.MACD(df["收盘"]).macd_signal(), ta.trend.MACD(df["收盘"]).macd_diff()
df["KDJ_K"] = ta.momentum.StochasticOscillator(df["最高"], df["最低"], df["收盘"]).stoch()
df["BOLL_high"] = ta.volatility.BollingerBands(df["收盘"]).bollinger_hband()
df["BOLL_low"] = ta.volatility.BollingerBands(df["收盘"]).bollinger_lband()
df["ATR"] = ta.volatility.AverageTrueRange(df["最高"], df["最低"], df["收盘"]).average_true_range()
df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["收盘"], df["成交量"]).on_balance_volume()

# **删除缺失值**
df.dropna(inplace=True)

# **选择训练特征**
features = ["收盘", "MA5", "MA10", "RSI", "MACD", "MACD_signal", "KDJ_K", "BOLL_high", "BOLL_low", "ATR", "OBV"]

# **归一化数据**
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=features)

# **设置时间窗口**
seq_length = 5  # 用最近 10 天的数据预测未来 1 天

# **构造 LSTM 输入数据**
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length]["收盘"])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length)

# **划分训练集 & 测试集**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = X_train.reshape(X_train.shape[0], seq_length, len(features))
X_test = X_test.reshape(X_test.shape[0], seq_length, len(features))

# **定义模型存储路径**
model_path = "/mnt/data/stock_lstm_model.h5"

# **如果模型存在，则加载**
if os.path.exists(model_path):
    print("加载已有模型...")
    model = load_model(model_path)
else:
    # **构建 LSTM 模型**
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    # **编译模型**
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())

    # **训练模型**
    history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test))

    # **保存模型**
    # model.save(model_path)
    print("模型训练完成并已保存！")

# **完整预测所有数据（包括训练集和测试集）**
all_X = X  # 训练集 + 测试集
y_pred = model.predict(all_X)

# **反归一化预测值**
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred.reshape(-1, 1), np.zeros((len(y_pred), len(features) - 1)))))[:, 0]
y_actual = scaler.inverse_transform(np.hstack((y.reshape(-1, 1), np.zeros((len(y), len(features) - 1)))))[:, 0]

# **画图对比实际股价 & 预测股价**
plt.figure(figsize=(12, 6))
plt.plot(df.index[seq_length:], y_actual, label="Actual Price", color="blue", linestyle="dashed")
plt.plot(df.index[seq_length:], y_pred_actual, label="Predicted Price", color="red")
plt.title("赛力斯股价预测 - LSTM + 技术指标")
plt.xlabel("日期")
plt.ylabel("价格")
plt.legend()
plt.show()

# **未来 5 天股价预测**
last_10_days = df_scaled[-seq_length:].values
last_10_days = last_10_days.reshape(1, seq_length, len(features))

future_prices = []
for _ in range(5):
    next_price = model.predict(last_10_days)
    future_prices.append(next_price[0][0])

    # **更新输入数据**
    new_data = np.hstack((next_price.reshape(-1, 1), last_10_days[:, -1, 1:].reshape(1, -1)))
    last_10_days = np.append(last_10_days[:, 1:, :], [new_data], axis=1)

# **反归一化未来股价**
future_prices_actual = scaler.inverse_transform(np.hstack((np.array(future_prices).reshape(-1, 1), np.zeros((len(future_prices), len(features) - 1)))))[:, 0]

# **输出预测值**
print(f"未来 5 天的预测股价: {future_prices_actual.flatten()}")
