import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import requests
from datetime import datetime
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置参数
ROOT_PATH = "./dataset/multivariate/"
OISST_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year}{month:02d}/oisst-avhrr-v02r01.{year}{month:02d}{day:02d}.nc"
START_YEAR = 1982
END_YEAR = 2023
COORDINATES = [
    (14.0, 112.0),
    (14.0, 118.0),
    (20.0, 112.0),
    (20.0, 118.0)
]
FEATURE_COLUMNS = [
    "sst", "sst_anomaly", "ice", "lat", "lon",
    "doy", "month", "day", "weekday", "year", "sst_rolling_avg"
]
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试延迟时间（秒）


def create_directory():
    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)
        print(f"创建目录: {ROOT_PATH}")


def create_session_with_retries():
    """创建带有重试机制的requests会话"""
    session = requests.Session()
    # 配置重试策略
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,  # 重试间隔时间：{backoff_factor} * (2 **({retry} - 1))
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
        allowed_methods=["GET"]  # 对GET请求进行重试
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_daily_oisst(year, month, day, output_dir):
    """下载单天的OISST数据，带重试机制和详细错误提示"""
    session = create_session_with_retries()
    url = OISST_URL.format(year=year, month=month, day=day)
    filename = f"oisst-avhrr-v02r01.{year}{month:02d}{day:02d}.nc"
    save_path = os.path.join(output_dir, filename)
    
    if os.path.exists(save_path):
        return save_path
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件大小用于进度显示
            file_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(
                total=file_size, 
                unit='B', 
                unit_scale=True,
                desc=f"下载 {year}-{month:02d}-{day:02d} (尝试 {attempt}/{MAX_RETRIES})",
                leave=False
            )
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉保持连接的空块
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()
            return save_path
            
        except Exception as e:
            # 清理不完整的文件
            if os.path.exists(save_path):
                os.remove(save_path)
                
            # 如果是最后一次尝试，记录错误
            if attempt == MAX_RETRIES:
                print(f"\n错误：{year}-{month:02d}-{day:02d} 下载失败（已达最大重试次数）- {str(e)}")
                return None
            
            # 重试前等待
            print(f"\n{year}-{month:02d}-{day:02d} 下载失败（尝试 {attempt}/{MAX_RETRIES}），{RETRY_DELAY}秒后重试 - {str(e)}")
            time.sleep(RETRY_DELAY)
    
    return None


def extract_point_data(nc_file, lat, lon):
    try:
        ds = xr.open_dataset(nc_file)
        
        # 找到最接近的网格点
        lat_idx = np.argmin(np.abs(ds.lat.values - lat))
        lon_idx = np.argmin(np.abs(ds.lon.values - lon))
        
        # 提取数据
        data = ds.isel(lat=lat_idx, lon=lon_idx)
        df = data.to_dataframe().reset_index()
        
        # 从文件名提取日期信息
        date_str = os.path.basename(nc_file).split('.')[1]
        df['date'] = datetime.strptime(date_str, "%Y%m%d")
        
        # 重命名列并提取所需特征
        df = df.rename(columns={'sst': 'sst', 'anom': 'sst_anomaly', 'ice': 'ice'})
        df['lat'] = lat
        df['lon'] = lon
        df['doy'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['year'] = df['date'].dt.year
        
        # 暂不计算滚动平均
        return df[["date"] + FEATURE_COLUMNS[:-1]]
    except Exception as e:
        print(f"数据提取错误: {str(e)}")
        return None


def process_all_years(lat, lon):
    temp_dir = "./temp_oisst"
    os.makedirs(temp_dir, exist_ok=True)
    failed_dates = []  # 记录所有下载失败的日期
    all_dfs = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n===== 开始处理年份: {year} =====")
        for month in range(1, 13):
            # 确定当月天数
            if month in [4, 6, 9, 11]:
                days = 30
            elif month == 2:
                # 判断闰年
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    days = 29
                else:
                    days = 28
            else:
                days = 31
                
            # 处理当月每一天
            for day in tqdm(range(1, days + 1), desc=f"处理月份 {month:02d}"):
                nc_file = download_daily_oisst(year, month, day, temp_dir)
                if nc_file:
                    df = extract_point_data(nc_file, lat, lon)
                    if df is not None:
                        all_dfs.append(df)
                else:
                    # 记录下载失败的日期
                    failed_dates.append(f"{year}-{month:02d}-{day:02d}")
    
    # 显示该经纬度点的下载失败统计
    if failed_dates:
        print(f"\n经纬度 {lat}°N, {lon}°E 共有 {len(failed_dates)} 天数据下载失败:")
        # 每10个失败日期打印一行，避免输出过长
        for i in range(0, len(failed_dates), 10):
            print(", ".join(failed_dates[i:i+10]))
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        combined_df = combined_df.sort_values("date").reset_index(drop=True)
        # 计算滚动平均值
        combined_df["sst_rolling_avg"] = combined_df["sst"].rolling(window=7, min_periods=1).mean()
        # 填充缺失值
        combined_df = combined_df.interpolate(method="linear")
        return combined_df
    return None


def save_to_csv(df, lat, lon):
    filename = f"oisst_lat_{lat}_lon_{lon}.csv"
    save_path = os.path.join(ROOT_PATH, filename)
    df.to_csv(save_path, index=False)
    print(f"\n已保存到: {save_path}")
    print(f"数据范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"有效数据量: {len(df)} 行")


def main():
    create_directory()
    
    for lat, lon in COORDINATES:
        print(f"\n\n===== 开始处理经纬度: {lat}°N, {lon}°E =====")
        df = process_all_years(lat, lon)
        if df is not None:
            save_to_csv(df, lat, lon)
    
    # 清理临时文件
    if os.path.exists("./temp_oisst"):
        import shutil
        shutil.rmtree("./temp_oisst")
        print("\n已清理临时文件")


if __name__ == "__main__":
    main()
    