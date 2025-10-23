import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ================================
# 🧩 NOAA OISST 数据集类
# ================================
class OISSTDataset(Dataset):
    """
    读取 NOAA OISST 数据 (.nc)，生成 (B, T, H, W, 1) 张量。
    支持自定义时间长度、步长、空间范围。
    """
    def __init__(self, data_dir, split='train',
                 seq_len=30, pred_len=7, stride=1,
                 lat_range=None, lon_range=None,
                 normalize=True):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.stride = stride
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.normalize = normalize

        # === 1. 读取全部 nc 文件 ===
        self.files = sorted([os.path.join(data_dir, f)
                             for f in os.listdir(data_dir)
                             if f.endswith('.nc') or f.endswith('.nc4')])

        assert len(self.files) > 0, f"No .nc files found in {data_dir}"

        # === 2. 合并数据集 ===
        ds_all = xr.open_mfdataset(self.files, combine='by_coords')
        sst = ds_all['sst']  # [time, lat, lon]
        sst = sst.sel(lat=slice(*lat_range), lon=slice(*lon_range)) if (lat_range and lon_range) else sst
        self.sst = sst.transpose('time', 'lat', 'lon')  # [T, H, W]
        self.time = ds_all['time'].values

        # === 3. 转为 numpy ===
        self.data = self.sst.values.astype(np.float32)
        self.data[np.isnan(self.data)] = 0.0

        # === 4. 归一化 ===
        if normalize:
            self.scaler = StandardScaler()
            T, H, W = self.data.shape
            self.data = self.data.reshape(T, -1)
            self.data = self.scaler.fit_transform(self.data)
            self.data = self.data.reshape(T, H, W)

        # === 5. 数据集划分 ===
        total_len = len(self.data)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        if split == 'train':
            self.data = self.data[:train_end]
        elif split == 'val':
            self.data = self.data[train_end:val_end]
        else:
            self.data = self.data[val_end:]

        self.indices = np.arange(0, len(self.data) - seq_len - pred_len, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq_x = self.data[i:i + self.seq_len]          # [T, H, W]
        seq_y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len]

        # 增加通道维度
        seq_x = np.expand_dims(seq_x, axis=-1)         # [T, H, W, 1]
        seq_y = np.expand_dims(seq_y, axis=-1)

        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)
    

# ================================
# ⚙️ 数据加载函数
# ================================
def load_oisst_dataloader(data_dir, batch_size=4, seq_len=30, pred_len=7,
                          lat_range=None, lon_range=None):
    train_set = OISSTDataset(data_dir, split='train', seq_len=seq_len, pred_len=pred_len,
                             lat_range=lat_range, lon_range=lon_range)
    val_set = OISSTDataset(data_dir, split='val', seq_len=seq_len, pred_len=pred_len,
                           lat_range=lat_range, lon_range=lon_range)
    test_set = OISSTDataset(data_dir, split='test', seq_len=seq_len, pred_len=pred_len,
                            lat_range=lat_range, lon_range=lon_range)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ================================
# ✅ 模块测试
# ================================
if __name__ == "__main__":
    # 示例路径（修改为你的 OISST 数据文件夹路径）
    data_dir = "E:/CodeSpace/MSPT/dataset/multivariate/oisst_lat_14.0_lon_112.0.csv"
    
    train_loader, val_loader, test_loader = load_oisst_dataloader(
        data_dir=data_dir,
        batch_size=2,
        seq_len=30,
        pred_len=7,
        lat_range=(0, 60),
        lon_range=(120, 260)
    )

    for x, y in train_loader:
        print("Input shape :", x.shape)   # [B, T, H, W, 1]
        print("Target shape:", y.shape)
        break
