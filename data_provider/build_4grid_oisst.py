import os
import pandas as pd
import numpy as np

# === 配置部分 ===
data_dir = "E:/CodeSpace/MSPT/dataset/multivariate"
save_path = "E:/CodeSpace/MSPT/dataset/oisst_4grid.npy"

# === 1. 找到所有csv文件 ===
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "oisst" in f]
print(f"找到 {len(csv_files)} 个文件:")
for f in csv_files:
    print("  ", f)

# === 2. 读取并提取 lat/lon 信息 ===
records = []
for f in csv_files:
    df = pd.read_csv(os.path.join(data_dir, f))
    lat = round(float(df["lat"].iloc[0]), 2)
    lon = round(float(df["lon"].iloc[0]), 2)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    records.append({"lat": lat, "lon": lon, "sst": df["sst"].values, "date": df["date"].values})

# === 3. 获取所有日期并检查一致性 ===
dates = records[0]["date"]
for r in records[1:]:
    assert np.all(r["date"] == dates), "❌ 日期不一致，请检查数据文件"

# === 4. 排序坐标（纬度从小到大，经度从小到大） ===
lats = sorted(list(set([r["lat"] for r in records])))
lons = sorted(list(set([r["lon"] for r in records])))

H, W = len(lats), len(lons)
T = len(dates)
print(f"✅ 数据维度: 时间步 {T}, 纬度数 {H}, 经度数 {W}")

# === 5. 构建 SST 网格 [T, H, W, 1] ===
sst_grid = np.zeros((T, H, W, 1), dtype=np.float32)

for r in records:
    h_idx = lats.index(r["lat"])
    w_idx = lons.index(r["lon"])
    sst_grid[:, h_idx, w_idx, 0] = r["sst"]

# === 6. 保存数据 ===
np.save(save_path, sst_grid)
print(f"✅ 已保存到 {save_path}")

# === 7. 测试加载 ===
data = np.load(save_path)
print(f"✅ 加载成功, 数据形状: {data.shape}")
print(f"示例: 第1天4格SST值:\n{data[0, :, :, 0]}")
