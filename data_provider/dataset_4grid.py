import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class _StdScaler:
    """简单的均值方差归一化/反归一化器（逐像素逐通道）。"""
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-5):
        self.mean = mean
        self.std = np.maximum(std, eps)
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class OISST4GridDataset(Dataset):
    """
    四格海温网格数据集
    数据形状: [T, H, W, C]
    通过滑动窗口生成样本:
      - x_enc: [seq_len, H, W, C]
      - y_dec: [label_len + pred_len, H, W, C]
        其中前 label_len 作为 decoder 的已知历史，后 pred_len 作为监督目标
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int = 60,
        label_len: int = 30,
        pred_len: int = 10,
        split: str = "train",
        split_ratio=(0.7, 0.2, 0.1),
        normalize: bool = False,
        scaler: _StdScaler = None,
    ):
        super().__init__()
        assert os.path.exists(data_path), f"❌ 文件不存在: {data_path}"
        self.data = np.load(data_path).astype(np.float32)  # [T, H, W, C]
        assert self.data.ndim == 4, f"期待 [T,H,W,C]，得到 {self.data.shape}"
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.split = split
        self.normalize = bool(normalize)

        # === 划分 ===
        T = self.data.shape[0]
        tr_end = int(T * split_ratio[0])
        va_end = int(T * (split_ratio[0] + split_ratio[1]))
        if split == "train":
            self.slice = slice(0, tr_end)
        elif split == "val":
            self.slice = slice(tr_end, va_end)
        else:
            self.slice = slice(va_end, T)
        self.data = self.data[self.slice]
        self.total_len = self.data.shape[0]
        print(f"✅ {split} 数据长度: {self.total_len}")

        # === 归一化（可选）===
        self.scale = False
        self._scaler = None
        if self.normalize:
            if scaler is None:
                # 只在训练集上拟合
                # 按 [T] 维做均值方差，保持 [1,H,W,C] 便于广播
                mean = self.data.mean(axis=0, keepdims=True)
                std = self.data.std(axis=0, keepdims=True)
                self._scaler = _StdScaler(mean, std)
            else:
                self._scaler = scaler

            self.data = self._scaler.transform(self.data)
            self.scale = True  # 兼容 Exp_Main.test() 的判断分支

    def __len__(self):
        # x:[t : t+seq_len], y:[t+seq_len-label_len : t+seq_len+pred_len]
        need = self.seq_len + self.pred_len
        return max(0, self.total_len - need)

    def __getitem__(self, idx):
        seq_start = idx
        seq_end = idx + self.seq_len
        y_start = seq_end - self.label_len
        y_end = seq_end + self.pred_len

        x_enc = self.data[seq_start:seq_end]         # [seq_len, H, W, C]
        y_dec = self.data[y_start:y_end]             # [label_len+pred_len, H, W, C]

        x_enc = torch.from_numpy(x_enc)              # float32
        y_dec = torch.from_numpy(y_dec)
        return x_enc, y_dec

    # 让 Exp_Main.test() 能找到 inverse_transform
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return x
        return self._scaler.inverse_transform(x)


def load_oisst_4grid_dataloader(
    data_path: str,
    seq_len: int = 60,
    label_len: int = 30,
    pred_len: int = 10,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
    normalize: bool = False,
):
    """
    返回: (train_loader, val_loader, test_loader)
    - 如果 normalize=True，会在训练集上拟合均值方差，并应用到 val/test，且支持 inverse_transform。
    """
    # 先构造一个训练集以便拟合 scaler
    train_raw = OISST4GridDataset(
        data_path=data_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        split="train",
        normalize=False,
    )

    if normalize:
        mean = train_raw.data.mean(axis=0, keepdims=True)
        std = train_raw.data.std(axis=0, keepdims=True)
        scaler = _StdScaler(mean, std)
    else:
        scaler = None

    # 用同一个 scaler 分别构造三个 split
    train_set = OISST4GridDataset(
        data_path=data_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        split="train", normalize=normalize, scaler=scaler
    )
    val_set = OISST4GridDataset(
        data_path=data_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        split="val", normalize=normalize, scaler=scaler
    )
    test_set = OISST4GridDataset(
        data_path=data_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        split="test", normalize=normalize, scaler=scaler
    )

    from sys import platform

    # Windows 建议 0~2，Linux 可放到 4~8
    if platform.startswith("win"):
        num_workers = min(num_workers, 2)

    common_kwargs = dict(
        batch_size=batch_size,
        pin_memory=True,
        # Windows 上 persistent_workers=True 反而容易拖慢/报错
        persistent_workers=False,
        # 只有 num_workers>0 才能用 prefetch_factor
        prefetch_factor=2 if num_workers > 0 else None
    )

    train_loader = DataLoader(train_set, shuffle=True,  num_workers=num_workers, drop_last=True,  **common_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, num_workers=num_workers, drop_last=False, **common_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, num_workers=num_workers, drop_last=False, **common_kwargs)
    
    return train_loader, val_loader, test_loader
