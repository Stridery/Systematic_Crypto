import pandas as pd

# 读取 CSV 文件
file_path = "data/raw/BTCUSDT_1d.csv"  # 修改为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 打印表头信息
print("列名(header):")
print(df.columns.tolist())

# 查看前 5 行数据
print("\n前 5 行数据预览：")
print(df.head())

# ---- 输出时间范围 ----
# 把 open_time_ms 转成 datetime（UTC 或不带时区都行）
dt = pd.to_datetime(df["open_time_ms"], unit="ms")  # 默认 naive
# 如果想显式看成 UTC，可以加 utc=True：
# dt = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

start_time = dt.min()
end_time = dt.max()

print("\n数据时间范围：")
print("起始时间:", start_time)
print("结束时间:", end_time)
print("总条数:", len(df))