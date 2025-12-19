from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Extreme
from data_provider.dataset_4grid import OISST4GridDataset # ✅ 引入我们的新数据集类
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'extreme': Dataset_Extreme,
    'oisst': OISST4GridDataset, # ✅ 在这里注册名字叫 'oisst'
}

def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq

    # ✅ 专门为 OISST 数据集写的加载逻辑
    if args.data == 'oisst':
        # 1. 先加载训练集，算出均值和方差 (Scaler)
        # 这样能保证训练、验证、测试用的是同一把“尺子”
        train_dataset = OISST4GridDataset(
            data_path=args.data_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            split='train',
            normalize=True # 开启归一化
        )
        scaler = train_dataset._scaler
        
        # 2. 根据 flag 加载对应的数据集，并传入刚才算好的 scaler
        if flag == 'train':
            data_set = train_dataset
        else:
            # 把 flag 转换成 split 名字 (val/test)
            split_name = 'val' if flag == 'val' else 'test'
            data_set = OISST4GridDataset(
                data_path=args.data_path,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                split=split_name,
                normalize=True,
                scaler=scaler # 传入训练集的尺子
            )
    else:
        # 其他数据集的通用逻辑
        Data = data_dict[args.data]
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )

    print(flag, len(data_set))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag,
                             num_workers=args.num_workers,
                             drop_last=drop_last)
    return data_set, data_loader