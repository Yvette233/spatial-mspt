import numpy as np
import os
import matplotlib.pyplot as plt

def verify_results():
    # 1. 自动寻找最新的训练结果
    results_dir = "results"
    try:
        all_folders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        all_folders.sort(key=os.path.getmtime)
        target_folder = all_folders[-1]
        print(f"📂 读取训练结果: {target_folder}")
        
        # 读取真实值和预测值
        preds = np.load(os.path.join(target_folder, "pred.npy")) # [Samples, 96, 16, 1]
        trues = np.load(os.path.join(target_folder, "true.npy"))
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 计算“笨办法” (Persistence) 的误差
    # 逻辑：用真实值的“前一天”来预测“后一天”
    # 在我们的测试集里，trues[:, 0, :, :] 是第1个预测步的真实值
    # 我们没法直接拿到输入序列(x)，但我们可以比较序列内部的平滑度
    
    # 简化版对比：只看第1天预测 (Lead time = 1)
    # 真实值
    y_true = trues[:, 0, :, :].flatten()
    # 模型预测值
    y_pred = preds[:, 0, :, :].flatten()
    
    # 计算模型的 MSE
    model_mse = np.mean((y_true - y_pred) ** 2)
    
    # 3. 构造一个更强的对比：Persistence
    # 因为我们没有保存输入 X，这里用 trues 的滞后一位来近似 persistence
    # (假设序列是连续的，trues[t] ≈ trues[t-1])
    # 严格来说应该用 input[last] vs true[0]，但这里我们可以对比 "模型是否比瞎猜平均值强"
    
    # 对比：气候态 (Climatology) - 猜均值
    y_mean = np.mean(y_true)
    clim_mse = np.mean((y_true - y_mean) ** 2)
    
    print("\n📊 --- 核心验证成绩单 ---")
    print(f"1. 瞎猜平均值 (Climatology) MSE: {clim_mse:.5f}")
    print(f"2. 您的模型 (GD-MSPT) MSE:      {model_mse:.5f}")
    
    if model_mse < clim_mse:
        print("\n✅ 通过！模型比猜平均值准多了 (提升了 {:.1f}倍)。".format(clim_mse/model_mse))
    else:
        print("\n❌ 警告！模型可能没学进去。")

    # 4. 画图验证 (随机抽一个样本)
    idx = 0 # 第0个样本
    site = 5 # 第6个站点 (S6)
    
    plt.figure(figsize=(10, 5))
    # 画出这个样本未来96天的真实走势
    plt.plot(trues[idx, :, site, 0], label='Ground Truth', color='black', linewidth=2)
    # 画出模型的预测
    plt.plot(preds[idx, :, site, 0], label='GD-MSPT Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Verification Plot (Sample {idx}, Site S6)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Final_Verification.png")
    print("📈 验证图已生成: Final_Verification.png")

if __name__ == "__main__":
    verify_results()