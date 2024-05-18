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
        
        self.fc = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        h_t = torch.zeros(self.configs.e_layers, x_enc.size(0), self.configs.d_model).to(x_enc.device)
        c_t = torch.zeros(self.configs.e_layers, x_enc.size(0), self.configs.d_model).to(x_enc.device)
        output, (h_t, c_t) = self.lstm(x_enc, (h_t, c_t)) # [B, T, N]
        output = self.fc(output[:, -1:, :].flatten(start_dim=1)) # [B, 1, N] -> [B, N]
        output = output.unsqueeze(-1).repeat(1, 1, x_enc.size(-1))
        return output