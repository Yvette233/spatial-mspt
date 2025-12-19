import os
import xarray as xr
import numpy as np
import pandas as pd

# ================= 配置区域 =================
# 1. 您下载的 .nc 文件放在哪？
source_dir = "data/raw_data"  # 请确保这里是您存放 .nc 文件的路径

# 2. 最终保存的文件名
save_path = "dataset/oisst_16grid.npy"

# 3. 定义我们要提取的 16 个格点坐标 (4x4 网格)
# 根据您之前的 CSV 文件名，经纬度范围如下：
# 纬度 (Lat): 14.0, 16.0, 18.0, 20.0 (从南到北)
# 经度 (Lon): 112.0, 114.0, 116.0, 118.0 (从西到东)
target_lats = [14.0, 16.0, 18.0, 20.0]
target_lons = [112.0, 114.0, 116.0, 118.0]

# ================= 开始处理 =================

def process_data():
    # 1. 找到所有年份的文件
    if not os.path.exists(source_dir):
        print(f"❌ 错误：找不到文件夹 {source_dir}，请修改代码里的 source_dir 路径！")
        return

    nc_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.nc')])
    print(f"📂 发现了 {len(nc_files)} 个 .nc 文件。")
    
    if len(nc_files) == 0:
        print("❌ 没找到文件！请确认您把下载的文件放进去了。")
        return

    all_data_list = []
    
    print("🚀 开始极速提取... (可能需要几秒钟)")
    
    # 2. 逐年读取并提取
    # 使用 xarray 的 open_mfdataset 可以一次性读取所有文件，非常快
    try:
        # 打开所有文件，自动拼接时间轴
        ds = xr.open_mfdataset(nc_files, combine='by_coords', parallel=True)
        
        # OISST 的变量名通常是 'sst'
        sst_data = ds['sst']
        
        # 3. 精准锁定 16 个点 (使用最近邻查找，防止浮点数微小误差)
        # 提取目标经纬度的数据
        # sel 方法会自动帮我们找这 4x4 个点
        subset = sst_data.sel(lat=target_lats, lon=target_lons, method='nearest')
        
        # 4. 载入内存并转换格式
        # 目前形状是 [Time, Lat, Lon] -> [15xxx, 4, 4]
        print("📥 正在读取数据到内存...")
        data_numpy = subset.values
        
        # 检查是否有缺失值 (NaN)
        if np.isnan(data_numpy).any():
            print("⚠️ 警告：数据中包含 NaN (缺失值)，正在进行线性插值修复...")
            # 简单修复：沿时间轴插值
            df_temp = pd.DataFrame(data_numpy.reshape(data_numpy.shape[0], -1))
            df_temp = df_temp.interpolate(method='linear', limit_direction='both')
            data_numpy = df_temp.values.reshape(data_numpy.shape)
            
        # 5. 调整维度以符合模型要求
        # 模型需要 [T, H, W, C]，这里 C=1
        # [T, 4, 4] -> [T, 4, 4, 1]
        final_data = data_numpy[..., np.newaxis]
        
        # 6. 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        
        print(f"✅ 成功！数据已打包保存到: {save_path}")
        print(f"📊 数据形状: {final_data.shape} (Time, Lat, Lon, Channel)")
        print(f"   时间范围: {len(ds['time'])} 天")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("提示：请检查文件名是否正确，或者是否安装了 xarray 和 netCDF4。")

if __name__ == "__main__":
    process_data()