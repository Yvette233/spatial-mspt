import os
import torch
from models import FC_LSTM, MSPT, MSPT_S, MSPT_L, MSPT_E, MAE, MAEncoder, DLinear, NLinear, TransDtSt_Part, iTransformer, MICN, TimesNet, Climatology, Informer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MSPT': MSPT,
            'MSPT_S': MSPT_S,
            'MSPT_L': MSPT_L,
            'MSPT_E': MSPT_E,
            'MAE': MAE,
            'MAEncoder': MAEncoder,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FC_LSTM': FC_LSTM,
            'TransDtSt_Part': TransDtSt_Part,
            'iTransformer': iTransformer,
            'MICN': MICN,
            'TimesNet': TimesNet, 
            'Climatology': Climatology,
            'Informer': Informer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass