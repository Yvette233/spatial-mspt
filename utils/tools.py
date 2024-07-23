import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {
            1: 1e-6, 2: 5e-6, 3: 1e-5, 4: 5e-5, 5: 1e-5, 6: 5e-5, 7:1e-4, 8: 5e-5, 9: 1e-5, 10: 5e-6,
            11: 1e-6, 12: 5e-7, 13: 1e-7, 14: 5e-8, 15: 1e-8, 16: 5e-9, 17: 1e-9, 18: 5e-10, 19: 1e-10, 20: 5e-11
        }
    elif args.lradj == 'type4':
        lr_adjust = {
            1: 1e-6, 2: 3e-6, 3: 8e-6, 4: 2e-5, 5: 5e-5, 6: 9e-5, 7:1e-4, 8: 9e-5, 9: 5e-5, 10: 2e-5,
            11: 8e-6, 12: 3e-6, 13: 1e-6, 14: 7e-7, 15: 3e-7, 16: 8e-8, 17: 3e-8, 18: 8e-9, 19: 4e-9, 20: 1e-9
        }
    elif args.lradj == 'type5':
        lr_adjust = {
            1: 1e-5, 2: 3e-5, 3: 8e-5, 4: 2e-4, 5: 5e-4, 6: 9e-4, 7:1e-3, 8: 9e-4, 9: 5e-4, 10: 2e-4,
            11: 8e-5, 12: 3e-5, 13: 1e-5, 14: 7e-6, 15: 3e-6, 16: 8e-7, 17: 3e-7, 18: 8e-8, 19: 4e-8, 20: 1e-8
        }
    elif args.lradj == 'type6':
        rate_list = [3, 5, 10, 30, 70, 100, 80, 40, 15, 8, 4, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        lr_adjust = {
            epoch: args.learning_rate * rate_list[epoch - 1]
        }
    elif args.lradj == 'type7':
        rate_list = [3, 5, 10, 30, 30, 30, 30, 30, 15, 8, 4, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        lr_adjust = {
            epoch: args.learning_rate * rate_list[epoch - 1]
        }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, watch_epoch=0, verbose=False, delta=0):
        self.patience = patience
        self.watch_epoch = watch_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        print(f'Patience count starts from {self.watch_epoch + 1} epoch')

    def __call__(self, current_epoch, val_loss, model, path, filename):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, filename)
        elif score < self.best_score + self.delta:
            if current_epoch > self.watch_epoch:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.save_last_checkpoint(model, path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, filename)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, filename):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + filename)
        self.val_loss_min = val_loss
    
    def save_last_checkpoint(self, model, path):
        torch.save(model.state_dict(), path + '/' + 'last.pth')

import time
class TrainTracking:
    def __init__(self, num_epochs, num_steps):
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.time_now = time.time()
        self.iter_count = 0

    def __call__(self, cur_step, cur_epoch, loss):
        self.iter_count += 1
        if (cur_step + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(cur_step + 1, cur_epoch + 1, loss.item()))
            speed = (time.time() - self.time_now) / self.iter_count
            left_time = speed * ((self.num_epochs - cur_epoch) * self.num_steps - cur_step)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            self.iter_count = 0
            self.time_now = time.time()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_climatology(true, preds=None, climatology=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    if climatology is not None:
        plt.plot(climatology, label='Climatology', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)