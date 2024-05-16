import torch
from torch import nn
import torch.nn.functional as F

# LSTM combined with Fully Connected Layer
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.lstm = nn.LSTM(input_size=configs.enc_in,
                            hidden_size=configs.d_model,
                            num_layers=configs.e_layers,
                            dropout=configs.dropout,
                            batch_first=True)
        
        self.fc = nn.Linear(configs.seq_len*configs.d_model, configs.pred_len)
        self.activation = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        h_t = torch.zeros(self.configs.e_layers, x_enc.size(0), self.configs.d_model).to(x_enc.device)
        c_t = torch.zeros(self.configs.e_layers, x_enc.size(0), self.configs.d_model).to(x_enc.device)
        output, (h_t, c_t) = self.lstm(x_enc, (h_t, c_t))
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        output = self.activation(output)
        return output