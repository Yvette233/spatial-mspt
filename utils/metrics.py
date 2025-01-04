import numpy as np
from sklearn.metrics import r2_score

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
