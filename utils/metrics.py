import numpy as np
from sklearn.metrics import r2_score


def _flatten_pred_true(pred, true):
    """
    pred/true: 支持 (N, T, H, W, C) / (N, T, C) / (N, T)
    返回 pred_f, true_f 形状 (N*T*H*W*C,) 以及辅助形状 (N, T, H, W, C)
    """
    assert pred.shape == true.shape, f"shape mismatch: {pred.shape} vs {true.shape}"
    shape = pred.shape
    if pred.ndim == 5:
        N, T, H, W, C = shape
    elif pred.ndim == 3:
        N, T, C = shape
        H, W = 1, 1
        pred = pred.reshape(N, T, 1, 1, C)
        true = true.reshape(N, T, 1, 1, C)
    elif pred.ndim == 2:
        N, T = shape
        H, W, C = 1, 1, 1
        pred = pred.reshape(N, T, 1, 1, 1)
        true = true.reshape(N, T, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported pred ndim={pred.ndim}")
    return pred.reshape(-1), true.reshape(-1), (N, T, H, W, C)

def metric_spatiotemporal(pred, true):
    """
    计算“全局时空”指标 + 逐网格 r²（返回平均）
    pred/true: (N, T, H, W, C) 或兼容形状
    """
    # 全局
    p_flat, t_flat, (N, T, H, W, C) = _flatten_pred_true(pred, true)
    diff = p_flat - t_flat
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    eps = 1e-8
    mape = float(np.mean(np.abs(diff) / (np.abs(t_flat) + eps)))
    mspe = float(np.mean((diff / (t_flat + eps)) ** 2))
    # rse
    t_mean = float(np.mean(t_flat))
    rse = float(np.sqrt(np.sum(diff ** 2) / (np.sum((t_flat - t_mean) ** 2) + eps)))
    # r2（全局）
    r2_global = float(r2_score(t_flat, p_flat))
    # “相关系数”的定义在你原函数里较特殊，这里用全局皮尔逊
    corr_num = float(np.sum((p_flat - np.mean(p_flat)) * (t_flat - np.mean(t_flat))))
    corr_den = float(np.sqrt(np.sum((p_flat - np.mean(p_flat)) ** 2) * np.sum((t_flat - np.mean(t_flat)) ** 2) + eps))
    corr = corr_num / (corr_den + eps)
    acc = float(np.mean(1 - np.abs(diff) / (np.abs(t_flat) + eps)))

    # 逐网格 r²：把每个 (h,w,c) 当作独立序列计算 r² 再平均
    pred_grid = pred.reshape(N * T, H * W * C)  # (N*T, G)
    true_grid = true.reshape(N * T, H * W * C)
    r2_per_grid = []
    for g in range(H * W * C):
        r2g = r2_score(true_grid[:, g], pred_grid[:, g])
        r2_per_grid.append(r2g)
    r2_per_grid = float(np.mean(r2_per_grid))

    return mae, mse, rmse, mape, mspe, rse, corr, r2_global, r2_per_grid, acc


# 相对平方误差
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

# 相关系数
def CORR(pred, true):
    # pred [B, T, C], true [B, T, C]
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0) #  [T, C]
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) # [T, C]
    return (u / d).mean(-1) # [T]

# 平均绝对误差
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

# 均方误差
def MSE(pred, true):
    return np.mean((pred - true) ** 2)

# 均方根误差
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# 平均绝对百分比误差
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

# 平均平方百分比误差
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def ACC(pred, true):
    return 1 - np.mean(np.abs(pred - true) / true)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rse = RSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    r2score = r2_score(pred.squeeze(-1), true.squeeze(-1))
    acc = ACC(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r2score, acc
