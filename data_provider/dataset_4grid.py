import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class OISST4GridDataset(Dataset):
    """
    四格海温网格数据集
    支持格式：[T, H, W, C]
    通过滑动窗口生成训练样本
    """

    def __init__(self, data_path, seq_len=60, pred_len=10, split="train", split_ratio=(0.7, 0.2, 0.1)):
        """
        参数：
        - data_path: .npy 文件路径
        - seq_len: 输入时间步长度
        - pred_len: 预测时间步长度
        - split: 数据划分 ('train', 'val', 'test')
        - split_ratio: 各部分占比
        """
        super().__init__()
        assert os.path.exists(data_path), f"❌ 文件不存在: {data_path}"
        self.data = np.load(data_path)  # [T, H, W, 1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        # === 计算划分索引 ===
        T = self.data.shape[0]
        train_end = int(T * split_ratio[0])
        val_end = int(T * (split_ratio[0] + split_ratio[1]))

        if split == "train":
            self.data = self.data[:train_end]
        elif split == "val":
            self.data = self.data[train_end:val_end]
        else:
            self.data = self.data[val_end:]

        self.total_len = self.data.shape[0]
        print(f"✅ {split} 数据长度: {self.total_len}")

    def __len__(self):
        return self.total_len - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        """
        输出:
        x_enc: [seq_len, H, W, C]
        y_dec: [pred_len, H, W, C]
        """
        seq_start = idx
        seq_end = idx + self.seq_len
        pred_end = seq_end + self.pred_len

        x_enc = self.data[seq_start:seq_end]
        y_dec = self.data[seq_end:pred_end]

        # 转换为 tensor
        x_enc = torch.tensor(x_enc, dtype=torch.float32)
        y_dec = torch.tensor(y_dec, dtype=torch.float32)

        return x_enc, y_dec


def load_oisst_4grid_dataloader(
    data_path,
    seq_len=60,
    pred_len=10,
    batch_size=16,
    num_workers=0
):
    """
    创建 PyTorch dataloaders
    """
    train_set = OISST4GridDataset(data_path, seq_len, pred_len, split="train")
    val_set = OISST4GridDataset(data_path, seq_len, pred_len, split="val")
    test_set = OISST4GridDataset(data_path, seq_len, pred_len, split="test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# === 测试运行 ===
if __name__ == "__main__":
    data_path = "E:/CodeSpace/MSPT/dataset/oisst_4grid.npy"

    train_loader, val_loader, test_loader = load_oisst_4grid_dataloader(
        data_path, seq_len=60, pred_len=10, batch_size=4
    )

    for batch_idx, (x_enc, y_dec) in enumerate(train_loader):
        print("输入:", x_enc.shape, " 目标:", y_dec.shape)
        break
