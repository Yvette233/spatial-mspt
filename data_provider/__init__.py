from data_provider.dataset_4grid import load_oisst_4grid_dataloader

def get_data_provider(dataset_name):
    """
    根据名称加载不同的数据集 DataLoader
    支持原始 MSPT 和自定义的 oisst_4grid
    """
    if dataset_name.lower() == "oisst_4grid":
        from data_provider.dataset_4grid import load_oisst_4grid_dataloader
        data_path = "E:/CodeSpace/MSPT/dataset/oisst_4grid.npy"
        print("🔹 使用四格 OISST 数据集加载器")
        return load_oisst_4grid_dataloader(
            data_path=data_path,
            seq_len=60,
            pred_len=10,
            batch_size=4
        )

    elif dataset_name.lower() == "oisst":
        from data_provider.loader_oisst import load_oisst_dataloader
        return load_oisst_dataloader()

    else:
        raise ValueError(f"❌ 未知数据集: {dataset_name}")

