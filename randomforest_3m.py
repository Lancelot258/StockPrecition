import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票数据并计算技术指标和滞后特征
    """
    stock_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date,
                                  adjust="qfq")
    stock_df["日期"] = pd.to_datetime(stock_df["日期"])
    stock_df = stock_df.sort_values(by="日期", ascending=True).reset_index(drop=True)

    # 计算技术指标和滞后特征
    stock_df["MA5"] = stock_df["收盘"].rolling(window=5).mean()
    stock_df["MA10"] = stock_df["收盘"].rolling(window=10).mean()
    stock_df["MA20"] = stock_df["收盘"].rolling(window=20).mean()
    stock_df["Lag1"] = stock_df["收盘"].shift(1)
    stock_df["Lag2"] = stock_df["收盘"].shift(2)
    stock_df["Lag3"] = stock_df["收盘"].shift(3)
    stock_df.dropna(inplace=True)

    return stock_df


def train_model(X_train, y_train, n_estimators=500, random_state=42):
    """
    训练随机森林模型
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_future(model, recent_data, future_days=7):
    """
    预测未来数据
    """
    future_data = recent_data.iloc[-1:].copy()
    future_predictions = []

    for i in range(future_days):
        # 生成未来时间点的特征
        future_row = future_data.iloc[-1:].copy()
        future_row["Lag1"] = future_row["收盘"]
        future_row["Lag2"] = future_row["Lag1"]
        future_row["Lag3"] = future_row["Lag2"]

        # 更新技术指标
        future_row["MA5"] = future_data["收盘"].iloc[-5:].mean()
        future_row["MA10"] = future_data["收盘"].iloc[-10:].mean()
        future_row["MA20"] = future_data["收盘"].iloc[-20:].mean()

        # 预测未来时间点的收盘价
        X_future = future_row[["MA5", "MA10", "MA20", "成交量", "Lag1", "Lag2", "Lag3"]]
        y_future = model.predict(X_future)

        # 将预测结果添加到未来数据中
        future_row["收盘"] = y_future[0]
        future_data = pd.concat([future_data, future_row], ignore_index=True)

        # 保存预测结果
        future_predictions.append(y_future[0])

    # 生成未来日期
    future_dates = pd.date_range(recent_data["日期"].iloc[-1] + timedelta(days=1), periods=future_days, freq="D")

    # 创建未来预测的DataFrame
    future_df = pd.DataFrame({
        "日期": future_dates,
        "预测收盘价": future_predictions
    })

    return future_df


def plot_results(recent_data, future_df):
    """
    可视化历史数据、历史预测和未来预测
    """
    plt.figure(figsize=(14, 7))
    plt.plot(recent_data["日期"], recent_data["收盘"], label="历史收盘价", color="blue")
    plt.plot(recent_data["日期"], recent_data["历史预测价格"], label="历史预测价格", color="green", linestyle="dashed")
    plt.plot(future_df["日期"], future_df["预测收盘价"], label="未来预测收盘价", color="red", linestyle="dashed",
             marker="o")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title("赛力斯股价历史、历史预测及未来一周预测（近期数据）")
    plt.legend()
    plt.grid()
    plt.show()


def randomforest_3m(stock_code="601127", start_date=None, end_date=None, n_estimators=500, random_state=42,
                    future_days=7):
    """
    主函数：使用过去三个月的数据训练随机森林模型并预测未来
    """
    # 获取最近一年的数据
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    stock_df = get_stock_data(stock_code, start_date, end_date)

    # 仅使用最近3个月的数据
    recent_data = stock_df[stock_df["日期"] >= (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")]

    # 定义特征和目标变量
    X = recent_data[["MA5", "MA10", "MA20", "成交量", "Lag1", "Lag2", "Lag3"]]
    y = recent_data["收盘"]

    # 划分训练集和测试集
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 训练模型
    model = train_model(X_train, y_train, n_estimators=n_estimators, random_state=random_state)

    # 对历史数据进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 将历史预测结果添加到DataFrame中
    recent_data.loc[X_train.index, "历史预测价格"] = y_train_pred
    recent_data.loc[X_test.index, "历史预测价格"] = y_test_pred

    # 预测未来数据
    future_df = predict_future(model, recent_data, future_days=future_days)

    # 显示未来预测结果
    print("未来一周的预测收盘价：")
    print(future_df)

    # 可视化结果
    plot_results(recent_data, future_df)


# 主程序入口
if __name__ == "__main__":
    randomforest_3m()