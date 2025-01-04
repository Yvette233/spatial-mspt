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

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.learning_rate)
        return model_optim

    def _select_scheduler(self, optimizer, lr, train_steps, train_epochs):
        if self.args.lradj == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          gamma=0.95)
        elif self.args.lradj == 'Constant':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1**epoch)
        elif self.args.lradj == 'HalfLR':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.5**epoch)
        elif self.args.lradj == 'HalfLR2':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.5**(epoch // 2))
        elif self.args.lradj == 'HalfLR4':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.5**(epoch // 4))
        elif self.args.lradj == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=0.000001,
                max_lr=lr,
                step_size_up=train_steps,
                step_size_down=train_steps,
                mode='triangular2',
                gamma=1.0,
                scale_fn=None,
                scale_mode='cycle',
                cycle_momentum=True,
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1,
                verbose=False)
        elif self.args.lradj == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_steps, eta_min=0)
        elif self.args.lradj == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=1, eta_min=0)
        elif self.args.lradj == 'ReduceLR':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=0.0001,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-08)
        elif self.args.lradj == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=train_steps,
                epochs=train_epochs,
                pct_start=0.3)
        elif self.args.lradj == 'None':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1**epoch)

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        return criterion

    def _makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # if not os.path.exists(self.results_save_path):
        #     os.makedirs(self.results_save_path)
        # if not os.path.exists(self.test_results_save_path):
        #     os.makedirs(self.test_results_save_path)

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                         batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                         batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                     batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                     batch_y_mark)

        return outputs

<<<<<<< HEAD
=======
    def vali_pretrain(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                # encoder - decoder
                outputs, mask = self._model_forward(batch_x, batch_x_mark, batch_x, batch_x_mark)

                loss = criterion(outputs, batch_x, mask)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp,
                                              batch_y_mark)
                # outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
<<<<<<< HEAD

        self.model_save_path = os.path.join(self.args.model_save_path, setting)
        self.results_save_path = os.path.join(self.args.results_save_path,
                                              setting)
        self.test_results_save_path = os.path.join(
            self.args.test_results_save_path, setting)

        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
=======
        
        self.model_save_path = os.path.join(self.args.model_save_path, setting)
        self.results_save_path = os.path.join(self.args.results_save_path, setting)
        self.test_results_save_path = os.path.join(self.args.test_results_save_path, setting)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

        model_optim = self._select_optimizer()

        train_steps = len(train_loader)
        scheduler = self._select_scheduler(model_optim,
                                           self.args.learning_rate,
                                           train_steps, self.args.train_epochs)
        print(f"Using {self.args.lradj} learning rate adjustment")

        criterion = self._select_criterion()
        test_criterion = self._select_criterion()

<<<<<<< HEAD
        self._makedirs()
=======
        if self.args.pretrain:
            patch_size = self.args.patch_size_ssl
            criterion = ReconstructionLoss(patch_size)
            test_criterion = ReconstructionLoss(patch_size)
            pretrain_setting = re.sub(r'_ll\d+_pl\d+', '', setting)
            self.pretrain_model_save_path = os.path.join(self.args.model_save_path, pretrain_setting, 'PS'+str(patch_size))
            self.pretrain_results_save_path = os.path.join(self.args.results_save_path, pretrain_setting, 'PS'+str(patch_size))
            self.pretrain_test_results_save_path = os.path.join(self.args.test_results_save_path, pretrain_setting, 'PS'+str(patch_size))
            self._makedir(self.pretrain_model_save_path)
            self._makedir(self.pretrain_results_save_path)
            self._makedir(self.pretrain_test_results_save_path)
   
        else:
            if self.args.finetune:
                assert not self.args.pretrain, "Cannot pretrain and finetune at the same time"
                self.transfer_weights(self.pretrain_model_save_path)

                early_stopping = EarlyStopping(patience=self.args.patience, watch_epoch=self.args.freeze_epochs, verbose=True)

                scheduler = self._select_scheduler(model_optim, self.args.learning_rate, train_steps, self.args.train_epochs) if self.args.freeze_epochs == 0 else [self._select_scheduler(model_optim, self.args.learning_rate, train_steps, self.args.freeze_epochs), self._select_scheduler(model_optim, self.args.learning_rate//2, train_steps, self.args.train_epochs-self.args.freeze_epochs)]
            
            self._makedir(self.model_save_path)
            self._makedir(self.results_save_path)
            self._makedir(self.test_results_save_path)
        
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        print("Starting training")
        for epoch in range(self.args.train_epochs):
            train_track = TrainTracking(self.args.train_epochs, train_steps)
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device)

<<<<<<< HEAD
                # encoder - decoder
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp,
                                              batch_y_mark)
=======
                    loss = criterion(outputs, batch_x, mask)
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                train_track(i, epoch, loss)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == "OneCycleLR":
                    scheduler.step()

            if self.args.lradj != "OneCycleLR":
                scheduler.step()

<<<<<<< HEAD
            print("Epoch: {} cost time: {}".format(epoch + 1,
                                                   time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, test_criterion)
            test_loss = self.vali(test_data, test_loader, test_criterion)
=======
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(epoch + 1, vali_loss, self.model, self.pretrain_model_save_path, self.args.model_save_filename)
                print("Adjusting learning rate to: {:.7f}".format(scheduler.get_last_lr()[0]))
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))
            early_stopping(vali_loss, self.model, self.model_save_path)
            print("Adjusting learning rate to: {:.7f}".format(
                scheduler.get_last_lr()[0]))
            if early_stopping.early_stop:
                print("Early stopping")
                break

<<<<<<< HEAD
    def test(self, setting, load_weight=True):
        test_data, test_loader = self._get_data(flag='test')
        if load_weight:
            print('loading supervised model weight')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.model_save_path + setting,
                                        'checkpoint.pth'),
                           map_location=self.device))
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

            myclass = import_class('models.' + self.args.model,
                                   'MultiScalePeriodicPatchEmbedding')

            def get_weight(name):
                # hook
                def hook(model, input, output):
                    weights.append(output[1].detach().cpu().numpy())

                return hook

            for name, module in self.model.named_modules():
                if isinstance(module, myclass):
                    hooks.append(module.register_forward_hook(
                        get_weight(name)))

=======
            if isinstance(scheduler, list):
                print("Stage 1 of finetuning")
                self.freeze()
                for epoch in range(self.args.freeze_epochs):
                    train_track = TrainTracking(self.args.freeze_epochs, train_steps)
                    train_loss = []

                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        model_optim.zero_grad()

                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float()

                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())

                        train_track(i, epoch, loss)

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()
                        
                        if self.args.lradj == "OneCycleLR":
                            scheduler[0].step()
                
                    if self.args.lradj != "OneCycleLR":
                        scheduler[0].step()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, test_criterion)
                    test_loss = self.vali(test_data, test_loader, test_criterion)

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(epoch + 1, vali_loss, self.model, self.model_save_path, self.args.model_save_filename)
                    print("Adjusting learning rate to: {:.7f}".format(scheduler[0].get_last_lr()[0]))

                print("Stage 2 of finetuning")
                self.unfreeze()
                for epoch in range(self.args.freeze_epochs, self.args.train_epochs):
                    train_track = TrainTracking(self.args.train_epochs-self.args.freeze_epochs, train_steps)
                    train_loss = []

                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        model_optim.zero_grad()

                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float()
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())

                        train_track(i, epoch, loss)

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()
                        
                        if self.args.lradj == "OneCycleLR":
                            scheduler[1].step()
                
                    if self.args.lradj != "OneCycleLR":
                        scheduler[1].step()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, test_criterion)
                    test_loss = self.vali(test_data, test_loader, test_criterion)

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(epoch + 1, vali_loss, self.model, self.model_save_path, self.args.model_save_filename)
                    print("Adjusting learning rate to: {:.7f}".format(scheduler[1].get_last_lr()[0]))
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

        
            else:
                print("Starting training")
                for epoch in range(self.args.train_epochs):
                    train_track = TrainTracking(self.args.train_epochs, train_steps)
                    train_loss = []

                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        model_optim.zero_grad()

                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float()
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())

                        train_track(i, epoch, loss)

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()
                        
                        if self.args.lradj == "OneCycleLR":
                            scheduler.step()
                
                    if self.args.lradj != "OneCycleLR":
                        scheduler.step()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, test_criterion)
                    test_loss = self.vali(test_data, test_loader, test_criterion)

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(epoch + 1, vali_loss, self.model, self.model_save_path, self.args.model_save_filename)
                    print("Adjusting learning rate to: {:.7f}".format(scheduler.get_last_lr()[0]))
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break


    
    def test(self, setting, load_weight=True):
        test_data, test_loader = self._get_data(flag='test')
        if load_weight:
            
            if self.args.pretrain:
                print('loading pretrain model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.pretrain_model_save_path, self.args.model_save_filename), map_location=self.device))
                
            elif self.args.finetune:
                print('loading finetune model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, self.args.model_save_filename), map_location=self.device))

            else:
                print('loading supervised model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, self.args.model_save_filename), map_location=self.device))
                self.model_save_path = os.path.join(self.args.model_save_path, setting)
                self.results_save_path = os.path.join(self.args.results_save_path, setting)
                self.test_results_save_path = os.path.join(self.args.test_results_save_path, setting)
    
        preds = []
        trues = []

        # hooks: get the gates of AdaptiveFourierTransformGateLayer Class of Model
        # layers = list(dict(self.model.named_children()).keys())
        # print(layers)
        gates_list = []
        def getGates(name):
            # the hook signature
            def hook(model, input, output):
                # print(1)
                # print(output.shape)
                gates_list.append(output.detach().cpu().numpy())
            return hook
        
        hooks = []
        # register forward hooks on the layers of choice 
        # from models.MSPT_CI import AdaptiveFourierTransformGateLayer   
        from models.PGMST import PeriodGuidedMultiScaleRouter
        for name, module in self.model.named_modules():
        #     print(name)
            if isinstance(module, PeriodGuidedMultiScaleRouter):
                hooks.append(module.register_forward_hook(getGates(name)))
        # for layer in layers:
            # if isinstance(getattr(self.model, layer), PeriodGuidedMultiScaleRouter):
            #     # getattr(self.model, layer).register_forward_hook(getGates(layer))
            #     hooks.append(getattr(self.model, layer).register_forward_hook(getGates(layer)))
    
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

<<<<<<< HEAD
                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp,
                                              batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(
                        outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, -365:, -1], true[0, :, -1]),
                                        axis=0)
                    pd = np.concatenate((input[0, -365:, -1], pred[0, :, -1]),
                                        axis=0)
                    visual(
                        gt, pd,
                        os.path.join(test_results_save_path,
                                     str(i) + '.pdf'))
=======
                    outputs = outputs.detach().cpu().numpy()
                    batch_x = batch_x.detach().cpu().numpy()
                    
                    # print(output.shape, batch_x.shape)
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(shape)
                    
                    pred = outputs[:, :, -1:]
                    true = batch_x[:, :, -1:]
                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        # input = batch_x.detach().cpu().numpy()
                        # if test_data.scale and self.args.inverse:
                        #     shape = input.shape
                        #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = true[0, :, -1]
                        pd = pred[0, :, -1]
                        visual(gt, pd, os.path.join(self.pretrain_test_results_save_path, str(i) +  '.pdf'))
        else:
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
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
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, -365:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -365:, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(self.test_results_save_path, str(i) + '.pdf'))
        
        # for h in h_list: h.remove()
        for hook in hooks: hook.remove()
        # for layer in layers:
        #     if isinstance(getattr(self.model, layer), AdaptiveFourierTransformGateLayer):
        #         getattr(self.model, layer).remove_hook()
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

<<<<<<< HEAD
        mae, mse, rmse, mape, mspe, rse, corr, r2_score, acc = metric(
            preds, trues)
        print(
            'mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
            .format(mse, mae, rmse, mape, mspe, rse, r2_score, acc))
=======
        gates_list = np.array(gates_list) 
        print('gates_list shape:', gates_list.shape)
        gates_list = gates_list.reshape(-1, gates_list.shape[-1])
        print('gates_list shape:', gates_list.shape)
        # plt.close()
        # sns.heatmap(gates_list, cmap='viridis')
        # plt.savefig('./gates1.pdf')

        mae, mse, rmse, mape, mspe, rse, corr, r2_score, acc = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'.format(mse, mae, rmse, mape, mspe, rse, r2_score, acc))
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab
        print('corr:', corr)
        f = open("result_sstp_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write(
            'mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
            .format(mse, mae, rmse, mape, mspe, rse, r2_score, acc))
        f.write('\n')
        f.write('corr:{}'.format(corr))
        f.write('\n')
        f.write('\n')
        f.close()
<<<<<<< HEAD

        np.save(results_save_path + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe, rse, r2_score, acc]))
        np.save(results_save_path + 'pred.npy', preds)
        np.save(results_save_path + 'true.npy', trues)
=======
        
        if self.args.pretrain:
            np.save(self.pretrain_results_save_path + '/metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, r2_score, acc]))
            np.save(self.pretrain_results_save_path + '/pred.npy', preds) 
            np.save(self.pretrain_results_save_path + '/true.npy', trues)
        else:
            np.save(self.results_save_path + '/metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, r2_score, acc]))
            np.save(self.results_save_path + '/pred.npy', preds) 
            np.save(self.results_save_path + '/true.npy', trues)
            np.save(self.results_save_path + '/gates.npy', gates_list)
    
    def test_climatology(self, setting, load_weight=True):
        test_data, test_loader = self._get_data(flag='test')
        climatology_data, climatology_loader = self._get_data(flag='test_climatology')
        if load_weight:
            if self.args.pretrain:
                print('loading pretrain model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, 'pretrain', 'checkpoint.pth'), map_location=self.device))
                test_results_save_path = self.args.test_results_save_path + setting + '/pretrain/'
                results_save_path = self.args.results_save_path + setting + '/pretrain/'
            elif self.args.finetune:
                print('loading finetune model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, 'finetune', 'checkpoint.pth'), map_location=self.device))
                test_results_save_path = self.args.test_results_save_path + setting + '/finetune/'
                results_save_path = self.args.results_save_path + setting + '/finetune/'
            else:
                print('loading supervised model weight')
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, 'default', 'checkpoint.pth'), map_location=self.device))
                test_results_save_path = self.args.test_results_save_path + setting + '/default/'
                results_save_path = self.args.results_save_path + setting + '/default/'
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

        if 'MSPT' in self.args.model:
            for hook in hooks:
                hook.remove()
            weights = np.array(weights)
            print('weights:', weights.shape)
            weights = weights.reshape(-1, weights.shape[-1])
            print('weights:', weights.shape)
            np.save(results_save_path + 'weights.npy', weights)

    # def test_climatology(self, setting, load_weight=True):
    #     test_data, test_loader = self._get_data(flag='test')
    #     climatology_data, climatology_loader = self._get_data(
    #         flag='test_climatology')
    #     if load_weight:
    #         if self.args.pretrain:
    #             print('loading pretrain model weight')
    #             self.model.load_state_dict(
    #                 torch.load(os.path.join(
    #                     self.args.model_save_path + setting, 'pretrain',
    #                     'checkpoint.pth'),
    #                            map_location=self.device))
    #             test_results_save_path = self.args.test_results_save_path + setting + '/pretrain/'
    #             results_save_path = self.args.results_save_path + setting + '/pretrain/'
    #         elif self.args.finetune:
    #             print('loading finetune model weight')
    #             self.model.load_state_dict(
    #                 torch.load(os.path.join(
    #                     self.args.model_save_path + setting, 'finetune',
    #                     'checkpoint.pth'),
    #                            map_location=self.device))
    #             test_results_save_path = self.args.test_results_save_path + setting + '/finetune/'
    #             results_save_path = self.args.results_save_path + setting + '/finetune/'
    #         else:
    #             print('loading supervised model weight')
    #             self.model.load_state_dict(
    #                 torch.load(os.path.join(
    #                     self.args.model_save_path + setting, 'default',
    #                     'checkpoint.pth'),
    #                            map_location=self.device))
    #             test_results_save_path = self.args.test_results_save_path + setting + '/default/'
    #             results_save_path = self.args.results_save_path + setting + '/default/'

    #     if not os.path.exists(test_results_save_path):
    #         os.makedirs(test_results_save_path)

    #     if not os.path.exists(results_save_path):
    #         os.makedirs(results_save_path)

    #     preds = []
    #     trues = []
    #     climatologys = []

    #     self.model.eval()

    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark,
    #                 batch_y_mark), (climatology_batch_x, climatology_batch_y,
    #                                 climatology_batch_x_mark,
    #                                 climatology_batch_y_mark) in zip(
    #                                     enumerate(test_loader),
    #                                     enumerate(climatology_loader)):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(
    #                 batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat(
    #                 [batch_y[:, :self.args.label_len, :], dec_inp],
    #                 dim=1).float().to(self.device)
    #             # encoder - decoder
    #             outputs = self._model_forward(batch_x, batch_x_mark, dec_inp,
    #                                           batch_y_mark)

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #             climatology_batch_y = climatology_batch_y[:, -self.args.
    #                                                       pred_len:, :].to(
    #                                                           self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #             climatology_batch_y = climatology_batch_y.detach().cpu().numpy(
    #             )
    #             if test_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = test_data.inverse_transform(
    #                     outputs.squeeze(0)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(
    #                     batch_y.squeeze(0)).reshape(shape)

    #             if climatology_data.scale and self.args.inverse:
    #                 shape = climatology_batch_y.shape
    #                 climatology_batch_y = climatology_data.inverse_transform(
    #                     climatology_batch_y.squeeze(0)).reshape(shape)

    #             outputs = outputs[:, :, f_dim:]
    #             batch_y = batch_y[:, :, f_dim:]
    #             climatology_batch_y = climatology_batch_y[:, :, f_dim:]

    #             pred = outputs
    #             true = batch_y
    #             climatology = climatology_batch_y
    #             preds.append(pred)
    #             trues.append(true)
    #             climatologys.append(climatology)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(
    #                         input.squeeze(0)).reshape(shape)
    #                 gt = np.concatenate((input[0, -365:, -1], true[0, :, -1]),
    #                                     axis=0)
    #                 pd = np.concatenate((input[0, -365:, -1], pred[0, :, -1]),
    #                                     axis=0)
    #                 cg = np.concatenate(
    #                     (input[0, -365:, -1], climatology[0, :, -1]), axis=0)
    #                 visual_climatology(
    #                     gt, pd, cg,
    #                     os.path.join(test_results_save_path,
    #                                  'climatology_' + str(i) + '.pdf'))

    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     climatologys = np.array(climatologys)
    #     print('test shape:', preds.shape, trues.shape, climatologys.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     climatologys = climatologys.reshape(-1, climatologys.shape[-2],
    #                                         climatologys.shape[-1])
    #     print('test shape:', preds.shape, trues.shape, climatologys.shape)

    #     mae1, mse1, rmse1, mape1, mspe1, rse1, corr1, r2_score1, acc1 = metric(
    #         preds, trues)
    #     mae2, mse2, rmse2, mape2, mspe2, rse2, corr2, r2_score2, acc2 = metric(
    #         climatologys, trues)
    #     print(
    #         'pred and true: mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
    #         .format(mse1, mae1, rmse1, mape1, mspe1, rse1, r2_score1, acc1))
    #     print('corr:', corr1)
    #     print(
    #         'climatology and true: mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
    #         .format(mse2, mae2, rmse2, mape2, mspe2, rse2, r2_score2, acc2))
    #     print('corr:', corr2)
    #     f = open("result_forecast_climatology.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write(
    #         'pred and true: mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
    #         .format(mse1, mae1, rmse1, mape1, mspe1, rse1, r2_score1, acc1))
    #     f.write('\n')
    #     f.write('corr:{}'.format(corr1))
    #     f.write('\n')
    #     f.write(
    #         'climatology and true: mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, r2_score:{}, acc:{}'
    #         .format(mse2, mae2, rmse2, mape2, mspe2, rse2, r2_score2, acc2))
    #     f.write('\n')
    #     f.write('corr:{}'.format(corr2))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(
    #         results_save_path + 'metrics_climatology.npy',
    #         np.array([
    #             mae1, mse1, rmse1, mape1, mspe1, rse1, r2_score1, acc1, mae2,
    #             mse2, rmse2, mape2, mspe2, rse2, r2_score2, acc2
    #         ]))
    #     np.save(results_save_path + 'preds.npy', preds)
    #     np.save(results_save_path + 'trues.npy', trues)
    #     np.save(results_save_path + 'climatologys.npy', climatologys)

    def get_model(self):
<<<<<<< HEAD
        return self.model.module if isinstance(self.model,
                                               nn.DataParallel) else self.model
=======
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def freeze(self):
        if hasattr(self.get_model(), 'head'): 
            # print('model head is available')
            for param in self.get_model().parameters(): param.requires_grad = False        
            for param in self.get_model().head.parameters(): param.requires_grad = True
            # print('model is frozen except the head')
        
    def unfreeze(self):
        for param in self.get_model().parameters(): param.requires_grad = True
    
    def transfer_weights(self, weights_path, exclude_head=True):
        # state_dict = model.state_dict()
        new_state_dict = torch.load(weights_path, map_location=self.device)
        #print('new_state_dict',new_state_dict)
        matched_layers = 0
        unmatched_layers = []
        for name, param in self.model.state_dict().items():        
            if exclude_head and 'head' in name: continue
            if name in new_state_dict:            
                matched_layers += 1
                input_param = new_state_dict[name]
                if input_param.shape == param.shape: param.copy_(input_param)
                else: unmatched_layers.append(name)
            else:
                unmatched_layers.append(name)
                pass # these are weights that weren't in the original model, such as a new head
        if matched_layers == 0: raise Exception("No shared weight names were found between the models")
        else:
            if len(unmatched_layers) > 0:
                print(f'check unmatched_layers: {unmatched_layers}')
            else:
                print(f"weights from {weights_path} successfully transferred!")
        self.model = self.model.to(self.device)
    
    # get gates from AdaptiveFourierTransformGateLayer Class
    # def model_with_gates_hook(self, model):
    #     gates_list = []

    #     def get_gates(name):
    #         def hook(model, input, output):
    #             gates_list.append(output.detach().cpu().numpy())
    #         return hook
    
    #     from models.MSPT_CI import AdaptiveFourierTransformGateLayer
    #     for layer in model.modules():
    #         if isinstance(layer, AdaptiveFourierTransformGateLayer):
    #             layer.register_forward_hook(get_gates(layer))

    #     return model
    
    # def clear_gates_hook(self, model):
    #     for layer in model.modules():
    #         if hasattr(layer, 'hooks'):
    #             for hook in layer.hooks:
    #                 hook.remove()
    #     return model

    # def get_gate_layer_output(self, ):
    #     gates_list = []

    #     def get_gates(name):
    #         def hook(model, input, output):
    #             gates_list.append(output.detach().cpu().numpy())
    #         return hook
        
    #     from models.MSPT_CI import AdaptiveFourierTransformGateLayer
    #     for layer in self.model.modules():
    #         if isinstance(layer, AdaptiveFourierTransformGateLayer):
    #             layer.register_forward_hook(get_gates(layer))

    #     return gates_list
>>>>>>> ac6402236b184772ba5ba3e4825d67f257ef3eab

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
