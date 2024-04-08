import os
import torch
from models import AFTN, Autoformer, Transformer, TimesNet, Nonstationary_Transformer, Linear, NLinear, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, ours, MTST, MSPEncoder, MSPT, MTSEncoder, FITS, MPT, AlignMTST, AFTNet, AFNO, AFTNet_M, CNN, LSTM, MSPTEmbed
# from ours_old import AFTN, AFTNet, MSPNet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'Linear': Linear,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'ours': ours,
            'AFTN': AFTN,
            'MTST': MTST,
            'MSPEncoder': MSPEncoder,
            'MSPT': MSPT,
            'MTSEncoder': MTSEncoder,
            'FITS': FITS,
            'MPT': MPT,
            'AlignMTST': AlignMTST,
            'AFTNet': AFTNet,
            'AFNO': AFNO,
            'AFTNet_M': AFTNet_M,
            'CNN': CNN,
            'LSTM': LSTM,
            'MSPTEmbed': MSPTEmbed
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