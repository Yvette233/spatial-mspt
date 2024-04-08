import torch
import torch.nn as nn

import re
import os

import matplotlib.pyplot as plt
import seaborn as sns


class TimesNetFeatureMapVisual(nn.Module):
    def __init__(self, setting, model, top_k=5, e_layers=2, d_layer=1, num_test_samples=2000, re_pattern=r"^model\.\d\.conv$"):
        super(TimesNetFeatureMapVisual, self).__init__()
        self.setting = setting
        self.model = model
        self.top_k = top_k
        if "TimesNet" in self.setting:
            self.layers = e_layers
        # if "ours" in self.setting:
        #     self.layers = e_layers
        if "AFTNet" in self.setting:
            self.layers = d_layer
        if "MSPN" in self.setting:
            self.layers = e_layers
        
        self.current_index = 0
        self.random_index = torch.randint(0, num_test_samples, (1,)).item() # 0~1999

        self.re_pattern = re_pattern
        # print(self.re_pattern)

        self.hook_handles = []

        self.inputs = []
        self.outputs = []
        
        self.layer_inputs = []
        self.layer_outputs = []

    def get_feature_map(self, module, input, output):
        self.inputs.append(input[0].cpu())
        self.outputs.append(output.cpu())
            
    def get_layer_feature_map(self, module, input, output):
        self.layer_inputs.append(input[0].cpu())
        self.layer_outputs.append(output.cpu())

    def remove_hooks(self):
        # 移除已注册的钩子
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    def __call__(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        self.inputs = []
        self.outputs = []
        self.layer_inputs = []
        self.layer_outputs = []
        if self.current_index == self.random_index:
            if "TimesNet" in self.setting or "AFTN" in self.setting or "MSPN" in self.setting:
                for name, module in self.model.named_modules():
                    # print(name)
                    # print(re.match(r"^decoder\.model\.\d\.conv$", name))
                    # if re.match(r"^decoder\.model\.\d\.conv$", name):
                    if re.match(self.re_pattern, name):
                        print(name)
                        module.register_forward_hook(self.get_feature_map)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                self.visual()
                self.remove_hooks()
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        self.current_index += 1
        return outputs
    
    def visual(self):
        # result save
        folder_path = '/root/autodl-tmp/visual_result/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # visual 
        for layer in range(self.layers):
            internal = len(self.inputs)//self.layers//self.top_k
            print(len(self.inputs))
            if self.top_k == 1:
                _, N, F, P = self.inputs[layer].shape # B=1
                rate = max(F, P) // min(F, P)
                height = 5 * rate if F > P else 5
                width = 5 if F > P else 5 * rate
                height = 50 if height > 50 else height
                width = 50 if width > 50 else width
                for li in range(layer*internal, layer*internal+internal):
                    for n in range(N):
                        plt.figure(figsize=(width, height))
                        sns.heatmap(self.inputs[li][0, n, :, :])
                        plt.savefig(folder_path + 'test_{}_layer_{}_part_{}_channel_{}_input.png'.format(self.current_index, layer, li%internal, n), bbox_inches='tight')
                        plt.close()
            else:
                _, N, _, _ = self.inputs[layer].shape # B=1
                for i in range(internal):
                    for n in range(N):
                        fig, axes = plt.subplots(nrows=1, ncols=self.top_k, figsize=(self.top_k * 10, 10))
                        for li in range(i+layer*self.top_k*internal, layer*self.top_k*internal+self.top_k*internal, internal):
                            sns.heatmap(self.inputs[li][0, n, :, :], ax=axes[(li-layer*self.top_k*internal)//internal]) # layer=0, top_k=5, li=0,1,2,3,4, layer=1, top_k=5, li=5,6,7,8,9 %5=0,1,2,3,4
                        plt.savefig(folder_path + 'test_{}_layer_{}_part_{}_channel_{}_input.png'.format(self.current_index, layer, i, n), bbox_inches='tight')
                        plt.close()
        
        
        for layer in range(self.layers):
            internal = len(self.outputs)//self.layers//self.top_k
            if self.top_k == 1:
                _, N, F, P = self.outputs[layer].shape # B=1
                rate = max(F, P) // min(F, P)
                height = 5 * rate if F > P else 5
                width = 5 if F > P else 5 * rate
                height = 50 if height > 50 else height
                width = 50 if width > 50 else width
                for li in range(layer*internal, layer*internal+internal):
                    for n in range(N):
                        plt.figure(figsize=(width, height))
                        sns.heatmap(self.outputs[layer][0, n, :, :])
                        plt.savefig(folder_path + 'test_{}_layer_{}_part_{}_channel_{}_output.png'.format(self.random_index, layer, li%internal, n), bbox_inches='tight')
                        plt.close()
            else:
                _, N, _, _ = self.outputs[layer].shape # B=1
                for i in range(internal):
                    for n in range(N):
                        fig, axes = plt.subplots(nrows=1, ncols=self.top_k, figsize=(self.top_k * 10, 10))
                        for li in range(i+layer*self.top_k*internal, layer*self.top_k*internal+self.top_k*internal, internal):
                            sns.heatmap(self.outputs[li][0, n, :, :], ax=axes[(li-layer*self.top_k*internal)//internal])
                        plt.savefig(folder_path + 'test_{}_layer_{}_part_{}_channel_{}_output.png'.format(self.random_index, layer, i, n), bbox_inches='tight')
                        plt.close()
            
                    
                      
            
    