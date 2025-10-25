from data_provider.dataset_4grid import load_oisst_4grid_dataloader

def get_data_provider(
    dataset_name: str,
    data_path: str = None,
    seq_len: int = 60,
    pred_len: int = 10,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    根据名称加载不同的数据集 DataLoader
    支持原始 MSPT 和自定义的 oisst_4grid
    """
    name = dataset_name.lower()

    if name == "oisst_4grid":
        if data_path is None:
            data_path = "E:/CodeSpace/MSPT/dataset/oisst_4grid.npy"
        print("🔹 使用四格 OISST 数据集加载器")
        return load_oisst_4grid_dataloader(
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    elif name == "oisst":
        from data_provider.loader_oisst import load_oisst_dataloader
        return load_oisst_dataloader()

    else:
        raise ValueError(f"❌ 未知数据集: {dataset_name}")

# 可选：确保包导出
__all__ = ["get_data_provider"]
