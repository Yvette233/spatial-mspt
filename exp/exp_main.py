from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, TrainTracking, adjust_learning_rate, visual, visual_climatology
from utils.metrics import metric
from utils.loss import ReconstructionLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from models.MSPT import Model as SpatialMSPT
from models.SpatialMSPT import Model as SpatialMSPT

from data_provider import get_data_provider

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.cuda.amp import autocast, GradScaler
try:
    import torch.backends.cuda.sdp_kernel as sdp
    sdp.enable_flash_sdp(False)           # 关
    sdp.enable_mem_efficient_sdp(False)   # 关
    sdp.enable_math_sdp(True)             # 只用稳定 math 内核
except Exception:
    pass

def _nan_safe(t, clip=8.0):
    # 强制 float32 + 将 NaN/Inf 替换为有限数 + 适度剪裁，避免极端值扰动
    t = t.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
    return t.clamp(-clip, clip)

def _flat_pair(pred, true):
    # 将任意形状预测/标签展平成 (B, -1)，并长度对齐
    b = pred.shape[0]
    pred = pred.reshape(b, -1)
    true = true.reshape(b, -1)
    m = min(pred.shape[1], true.shape[1])
    return pred[:, :m], true[:, :m]




warnings.filterwarnings('ignore')

# —— MSPT-like（3D口径）指标：兼容你的 5D 张量，计算与原MSPT可比的 MSE/RMSE/MAE/R²
def mspt_like_metrics(y_true, y_pred):
    """
    y_true, y_pred: numpy arrays with shape [N, L, H, W, C] 或 [N,L,Ps,1]
    逻辑：把 (H,W,C) 折成一维“格点”维度，按原MSPT的 3D 视角整体评估
    """
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N, L = y_true.shape[0], y_true.shape[1]
    y_true_f = y_true.reshape(N, L, -1)   # [N,L,Ps]
    y_pred_f = y_pred.reshape(N, L, -1)
    diff = y_pred_f - y_true_f

    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))

    y_mean = float(np.mean(y_true_f))
    ss_res = float(np.sum((y_true_f - y_pred_f)**2))
    ss_tot = float(np.sum((y_true_f - y_mean)**2)) + 1e-12
    r2    = 1.0 - ss_res / ss_tot

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # 原来： self.scaler = GradScaler(enabled=True)
        self.scaler = GradScaler(enabled=bool(getattr(self.args, "use_amp", False)))



    # def _build_model(self):
    #     model = self.model_dict[self.args.model].Model(self.args).float()

    #     if self.args.use_multi_gpu and self.args.use_gpu:
    #         model = nn.DataParallel(model, device_ids=self.args.device_ids)
    #     return model
    
    def _sanitize_and_norm(self, x, mean=20.0, std=10.0):
            # 先灭 NaN/Inf，再做合理范围裁剪（SST 常见范围）
            x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-5.0)
            x = x.clamp(-5.0, 40.0)
            # 统一到大致 [-3, 2] 的量级，注意: 只是训练/损失内部用
            x = (x - mean) / std
            return x

    def _build_model(self):
        """
        Build model. Support SpatialMSPT (custom) and fallback to model_dict for legacy models.
        """
        # If user selected our SpatialMSPT
        if getattr(self.args, 'model', '').lower() == 'spatialmspt':
            model = SpatialMSPT(self.args).float()
        else:
            # legacy behavior (keep existing registry)
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    class EMA:
        """Exponential Moving Average of model parameters (trainable params only)."""
        def __init__(self, model, decay: float = 0.999):
            self.decay = float(decay)
            self.shadow = {}   # name -> tensor clone
            self.backup = {}   # for apply/restore
            self.register(model)

        


        def register(self, model):
            # 只跟踪可训练参数
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.data.clone()

        @torch.no_grad()
        def update(self, model):
            # 动态容错：如果某些参数是“后注册”的，这里会补齐 shadow
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = p.data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

        @torch.no_grad()
        def apply_shadow(self, model):
            # 将 EMA 权重临时加载到模型上用于评估
            self.backup = {}
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    self.backup[name] = p.data.clone()
                    p.data.copy_(self.shadow[name])

        @torch.no_grad()
        def restore(self, model):
            # 恢复原训练权重
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.backup:
                    p.data.copy_(self.backup[name])
            self.backup = {}


    
    def _get_data(self, flag):
        if getattr(self.args, 'model', '').lower() == 'spatialmspt':
            train_loader, val_loader, test_loader = get_data_provider(
                "oisst_4grid",
                data_path=self.args.data_path,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers
            )


            class WrappedLoader:
                def __init__(self, loader): self.loader = loader
                def __iter__(self):
                    for batch in self.loader:
                        if isinstance(batch, (list, tuple)) and len(batch) == 2:
                            batch_x, batch_y = batch
                        else:
                            yield batch; continue
                        B = batch_x.shape[0]
                        batch_x_mark = torch.zeros((B, batch_x.shape[1], 1), dtype=torch.float32)
                        batch_y_mark = torch.zeros((B, batch_y.shape[1], 1), dtype=torch.float32)
                        yield batch_x, batch_y, batch_x_mark, batch_y_mark
                def __len__(self): return len(self.loader)

            # ✅ 返回一个哑 dataset，避免 test() 里访问 scale 报错
            class DummySet: 
                scale = False

            if flag == 'train':
                return DummySet(), WrappedLoader(train_loader)
            elif flag == 'val':
                return DummySet(), WrappedLoader(val_loader)
            else:
                return DummySet(), WrappedLoader(test_loader)
        else:
            return data_provider(self.args, flag)
    
    def _select_optimizer(self):
        # AdamW 比 Adam 在权重衰减上更“正统”
        wd = getattr(self.args, 'weight_decay', 1e-4)
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd, betas=(0.9, 0.99))


    def _select_scheduler(self, optimizer, train_loader):
        name = str(self.args.lradj)
        if name == 'OneCycleLR':
            steps_per_epoch = len(train_loader)
            # 若没显式给 max_lr，则用 base lr 的 10 倍
            max_lr = getattr(self.args, 'max_lr', None)
            if max_lr is None:
                max_lr = self.args.learning_rate * 10.0
            pct_start = getattr(self.args, 'pct_start', 0.1)
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=self.args.train_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start
            )
        elif name == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=400, T_mult=1, eta_min=self.args.learning_rate * 0.1
            )
        else:
            return None

    def _get_optimizer(self):
        wd = getattr(self.args, 'weight_decay', 1e-4)
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.args.learning_rate,
                                weight_decay=wd)


    def _select_criterion(self):
        loss_name = str(self.args.loss).lower()
        if loss_name in ['mse', 'mse_loss']:
            criterion = torch.nn.MSELoss()
        elif loss_name in ['mae', 'l1', 'mae_loss', 'l1loss']:
            criterion = torch.nn.L1Loss()
        elif loss_name in ['huber', 'smoothl1', 'smoothl1loss']:
            # beta 越小越接近 L1，默认 0.5 比较稳
            criterion = torch.nn.SmoothL1Loss(beta=0.5)
        else:
            print(f"[warn] Unknown loss '{self.args.loss}', fallback to Huber.")
            criterion = torch.nn.SmoothL1Loss(beta=0.5)
        return criterion


    def _makedirs(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)
        if not os.path.exists(self.test_results_save_path):
            os.makedirs(self.test_results_save_path)

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        use_amp = bool(getattr(self.args, "use_amp", False))
        # 原来是 with autocast():
        with autocast(enabled=use_amp):
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 与 train 一致：先归一化，再构造 dec_inp
                batch_x = self._sanitize_and_norm(batch_x)
                batch_y = self._sanitize_and_norm(batch_y)
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, ...],
                     torch.zeros_like(batch_y[:, -self.args.pred_len:, ...])],
                    dim=1
                ).to(self.device)

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, ...]
                batch_y = batch_y[:, -self.args.pred_len:, ...].to(self.device)

                # 先灭 NaN/Inf 再算损失
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model_save_path = os.path.join(self.args.model_save_path, setting)
        self.results_save_path = os.path.join(self.args.results_save_path, setting)
        self.test_results_save_path = os.path.join(self.args.test_results_save_path, setting)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        train_steps = len(train_loader)
        #scheduler = self._select_scheduler(model_optim, self.args.learning_rate, train_steps, self.args.train_epochs)
        scheduler = self._select_scheduler(model_optim, train_loader)

        print(f"Using {self.args.lradj} learning rate adjustment")

        criterion = self._select_criterion()
        test_criterion = self._select_criterion()
        
        self._makedirs()

        self.ema = self.EMA(self.model, decay=0.999) if getattr(self.args, 'sota_mode', 0) else None

        print("Starting training")
        for epoch in range(self.args.train_epochs):
            train_track = TrainTracking(self.args.train_epochs, train_steps)
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
               # 1) 上设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()  # 先留 CPU，等归一化后再搬一部分
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 2) 先归一化（与 encoder 同数域），再构造 decoder 输入
                batch_x = self._sanitize_and_norm(batch_x)
                batch_y = self._sanitize_and_norm(batch_y)
                dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, ...])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, ...], dec_zeros], dim=1).to(self.device).float()

                model_optim.zero_grad(set_to_none=True)

                # --- 前向 ---
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, ...]
                batch_y_t = batch_y[:, -self.args.pred_len:, ...].to(self.device)

                # 先灭 NaN/Inf，再计算损失（关键！）
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y_t = torch.nan_to_num(batch_y_t, nan=0.0, posinf=0.0, neginf=0.0)
                loss = criterion(outputs, batch_y_t)


                train_loss.append(loss.item())
                train_track(i, epoch, loss)

                # --- 非有限值保护（关键）---
                if not torch.isfinite(loss):
                    print(f"[skip step] non-finite loss at epoch {epoch+1} iter {i}: {loss.item()}")
                    try:
                        o_min, o_max = float(outputs.min().item()), float(outputs.max().item())
                        y_min, y_max = float(batch_y_t.min().item()), float(batch_y_t.max().item())
                        print(f"    outputs[{o_min:.3f}, {o_max:.3f}]  targets[{y_min:.3f}, {y_max:.3f}]")
                    except Exception:
                        pass
                    model_optim.zero_grad(set_to_none=True)
                    continue

                # --- AMP 反向 & 梯度裁剪 & 更新 ---
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(model_optim)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                # 可选：观测梯度是否在发散
                # if i % 50 == 0: print(f"[grad-norm] {float(total_norm):.3f}")
                self.scaler.step(model_optim)
                self.scaler.update()


                # EMA（若启用）仍然每步更新
                if getattr(self, "ema", None) is not None:
                    self.ema.update(self.model)

                # OneCycleLR：每个 step 后调用（你原来的写法）
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                # 循环内不再额外调用 zero_grad（已在开头设置 set_to_none=True）
            
            # ✅ epoch 结束：只对“非 OneCycle”的调度器 step 一次
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # ✅ 打印学习率：只打印一次（epoch 末）
            print("Adjusting learning rate to: {:.7f}".format(
                scheduler.get_last_lr()[0] if scheduler is not None else self.args.learning_rate
            ))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # 评估前
            if self.ema is not None:
                self.ema.apply_shadow(self.model)
            vali_loss = self.vali(vali_data, vali_loader, test_criterion)
            test_loss = self.vali(test_data, test_loader, test_criterion)
            if self.ema is not None:
                self.ema.restore(self.model)
            
            



            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.model_save_path)
            print("Adjusting learning rate to: {:.7f}".format(scheduler.get_last_lr()[0]))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self, setting, load_weight=True):
        test_data, test_loader = self._get_data(flag='test')
        if load_weight:
            print('loading supervised model weight')
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, 'checkpoint.pth'), map_location=self.device))
            test_results_save_path = self.args.test_results_save_path + setting + '/'
            results_save_path = self.args.results_save_path + setting + '/'

        if not os.path.exists(test_results_save_path):
            os.makedirs(test_results_save_path)

        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
    
        preds = []
        trues = []

        weights = []
        hooks = []

        # register forward hooks to get gates of Multi-Scale Periodic Patch Embedding
        if 'MSPT' in self.args.model:
            import importlib
            def import_class(module_name, class_name):
                # 导入模块
                module = importlib.import_module(module_name)
                # 从模块中获取类
                clazz = getattr(module, class_name)
                return clazz
            myclass = import_class('models.'+self.args.model, 'MultiScalePeriodicPatchEmbedding')

            def get_weight(name):
                # hook
                def hook(model, input, output):
                    weights.append(output[1].detach().cpu().numpy())
                return hook
            
            for name, module in self.model.named_modules():
                if isinstance(module, myclass):
                    hooks.append(module.register_forward_hook(get_weight(name)))
            

    
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 与 train 一致：先归一化，再构造 dec_inp
                batch_x = self._sanitize_and_norm(batch_x.float().to(self.device))
                batch_y = self._sanitize_and_norm(batch_y.float())
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, ...],
                     torch.zeros_like(batch_y[:, -self.args.pred_len:, ...])],
                    dim=1
                ).to(self.device)


                # encoder - decoder
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # 预测/真值的切片（保留空间与通道，不做额外 f_dim 截断）
                outputs = outputs[:, -self.args.pred_len:, ...]
                batch_y  = batch_y[:,  -self.args.pred_len:, ...]
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()  # (B, L, H, W, 1)
                    # 取空间平均做示意曲线（也可以固定 (h0,w0)）
                    input_1d = input.mean(axis=(2,3,4))      # (B, L)
                    true_1d  = true.mean(axis=(2,3,4))       # (B, pred_len)
                    pred_1d  = pred.mean(axis=(2,3,4))       # (B, pred_len)

                    gt = np.concatenate((input_1d[0, -min(365, input_1d.shape[1]):], true_1d[0]), axis=0)
                    pd = np.concatenate((input_1d[0, -min(365, input_1d.shape[1]):], pred_1d[0]), axis=0)
                    visual(gt, pd, os.path.join(test_results_save_path, str(i) + '.pdf'))

        # 每个元素形状都是 [B_i, pred_len, C]，直接按样本维拼接
        preds = np.concatenate(preds, axis=0)  # [N, pred_len, C]
        trues = np.concatenate(trues, axis=0)  # [N, pred_len, C]
        print('test shape:', preds.shape, trues.shape)


        from utils.metrics import metric_spatiotemporal

        # 不再 reshape 丢空间信息，直接用 5D
        mae, mse, rmse, mape, mspe, rse, corr, r2_global, r2_grid, acc = metric_spatiotemporal(preds, trues)
        print('GLOBAL - mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_global:{}, r2_grid:{}, acc:{}'.format(
            mse, mae, rmse, mape, mspe, rse, r2_global, r2_grid, acc))
        print('corr (global Pearson):', corr)
        # === 新增：MSPT 可对比口径（把空间展平到 batch 轴） ===
        # 将 (N,T,H,W,C) → (N*H*W, T, C)，再用原 3D metric 计算
        if preds.ndim == 5:      # (N,T,H,W,C)
            N, T, H, W, C = preds.shape
            mspt_pred = preds.reshape(N * H * W, T, C)
            mspt_true = trues.reshape(N * H * W, T, C)
        elif preds.ndim == 4:    # (N,T,HW,C)
            N, T, HW, C = preds.shape
            mspt_pred = preds.reshape(N * HW, T, C)
            mspt_true = trues.reshape(N * HW, T, C)
        else:                    # 已经是 (N,T,C)
            mspt_pred, mspt_true = preds, trues

       # 与 MSPT 一样的 3D 指标（返回 dict，避免解包长度不匹配）
        mspt = mspt_like_metrics(mspt_true, mspt_pred)  # {"mse","rmse","mae","r2"}
        print(f"[Eval] ours-5D MSE: {mse:.6f} | MSPT-3D MSE: {mspt['mse']:.6f}")



        # ====== Real-world metrics (反归一化后的真实指标) ======
        try:
            # 如果你的 Dataset 有保存均值与标准差
            mean, std = getattr(self.args, 'data_mean', 0.0), getattr(self.args, 'data_std', 1.0)
            # 若未定义，可改为 dataset.mean / dataset.std
            if hasattr(self, 'train_data'):
                if hasattr(self.train_data, 'mean'): mean = self.train_data.mean
                if hasattr(self.train_data, 'std'): std = self.train_data.std

            preds_real = preds * std + mean
            trues_real = trues * std + mean

            mse_real = np.mean((preds_real - trues_real)**2)
            mae_real = np.mean(np.abs(preds_real - trues_real))
            rmse_real = np.sqrt(mse_real)

            # 决定系数
            ss_res = np.sum((trues_real - preds_real)**2)
            ss_tot = np.sum((trues_real - np.mean(trues_real))**2)
            r2_real = 1 - ss_res / ss_tot

            # 皮尔逊相关系数
            corr_real = np.corrcoef(trues_real.flatten(), preds_real.flatten())[0, 1]

            print(f"🔹 Real-world (°C) Metrics: MSE={mse_real:.4f}, MAE={mae_real:.4f}, RMSE={rmse_real:.4f}, R²={r2_real:.4f}, Corr={corr_real:.4f}")
            # === 逐格点（四个点）RMSE/MAE/R²，按论文风格在真实温度(°C)下计算 ===
            try:
                # preds_real, trues_real 已在上面得到，形状 [N, T, H, W, C]
                pr = preds_real
                tr = trues_real
                if pr.ndim == 5 and pr.shape[-1] == 1:
                    pr = pr[..., 0]
                    tr = tr[..., 0]  # 现在是 [N, T, H, W]

                N, T, H, W = pr.shape
                site_stats = []  # [(rmse, mae, r2)]
                print("—— Per-site metrics (°C, 与论文一致：RMSE/MAE/R²) ——")
                idx = 0
                for i in range(H):
                    for j in range(W):
                        y_hat = pr[:, :, i, j].reshape(-1)   # 该格点在所有样本×步长上的预测
                        y_true = tr[:, :, i, j].reshape(-1)

                        mse_ij = ((y_hat - y_true) ** 2).mean()
                        rmse_ij = mse_ij ** 0.5
                        mae_ij = (abs(y_hat - y_true)).mean()

                        ss_res = ((y_true - y_hat) ** 2).sum()
                        ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
                        r2_ij = 1.0 - ss_res / ss_tot

                        site_stats.append((rmse_ij, mae_ij, r2_ij))
                        print(f"S{idx+1}  RMSE:{rmse_ij:.3f}  MAE:{mae_ij:.3f}  R²:{r2_ij:.3f}")
                        idx += 1

                # 可选：四点平均（方便与表格的 Avg 行对应）
                avg_rmse = sum(s[0] for s in site_stats) / len(site_stats)
                avg_mae  = sum(s[1] for s in site_stats) / len(site_stats)
                avg_r2   = sum(s[2] for s in site_stats) / len(site_stats)
                print(f"Avg  RMSE:{avg_rmse:.3f}  MAE:{avg_mae:.3f}  R²:{avg_r2:.3f}")
                # —— 把四点的指标和 Avg 一并写入结果文件
                try:
                    with open("result_sstp_forecast.txt", "a", encoding="utf-8") as f_avg:
                        per_site_str = " | ".join(
                            [f"S{k+1}: RMSE={s[0]:.3f}, MAE={s[1]:.3f}, R²={s[2]:.3f}" for k, s in enumerate(site_stats)]
                        )
                        f_avg.write(
                            f"Per-site (°C): {per_site_str} | Avg: RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}, R²={avg_r2:.3f}\n"
                        )
                    # （可选）再打印一行醒目的 Avg 行
                    print(f"Per-site Avg (°C)  RMSE:{avg_rmse:.3f}  MAE:{avg_mae:.3f}  R²:{avg_r2:.3f}")
                except Exception as e:
                    print(f"[⚠️ Write Avg failed: {e}]")

            except Exception as e:
                print(f"[⚠️ Per-site metrics skipped: {e}]")

        except Exception as e:
            print(f"[⚠️ Real metrics skipped: {e}]")


        f = open("result_sstp_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_global:{}, acc:{} | mspt_mse:{}'.format(
            mse, mae, rmse, mape, mspe, rse, r2_global, acc, mspt["mse"]))
        f.write('\n')
        f.write('corr:{}'.format(corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(results_save_path + 'metrics.npy',np.array([mae, mse, rmse, mape, mspe, rse, r2_global, acc]))
        np.save(results_save_path + 'pred.npy', preds)
        np.save(results_save_path + 'true.npy', trues)

        if 'MSPT' in self.args.model:
            for hook in hooks: hook.remove()
            weights = np.array(weights)
            print('weights:', weights.shape)
            weights = weights.reshape(-1, weights.shape[-1])
            print('weights:', weights.shape)
            np.save(results_save_path + 'weights.npy', weights)
    

    def get_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # def get_layer_output(self, inp, layers=None, unwrap=False):
    #     """
    #     Args:
    #         inp: can be numpy array, torch tensor or dataloader
    #     """
    #     self.model.eval()
    #     device = next(self.model.parameters()).device
    #     if isinstance(inp, np.ndarray): inp = torch.Tensor(inp).to(device)
    #     if isinstance(inp, torch.Tensor): inp = inp.to(device)
        
    #     return get_layer_output(inp, model=self.model, layers=layers, unwrap=unwrap)

    # def get_layer_output(self, inp, model, layers=None, unwrap=False):
    #     """
    #     layers is a list of module names
    #     """
    #     orig_model = model
        
    #     if unwrap: model = unwrap_model(model)
    #     if not layers: layers = list(dict(model.named_children()).keys())
    #     if not isinstance(layers, list): layers = [layers]

    #     activation = {}
    #     def getActivation(name):
    #         # the hook signature
    #         def hook(model, input, output):
    #             activation[name] = output.detach().cpu().numpy()
    #         return hook

    #     # register forward hooks on the layers of choice    
    #     h_list = [getattr(model, layer).register_forward_hook(getActivation(layer)) for layer in layers]
        
    #     model.eval()
    #     out = orig_model(inp)    
    #     for h in h_list: h.remove()
    #     return activation