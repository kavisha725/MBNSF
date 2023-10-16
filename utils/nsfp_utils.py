# Functions in this file are copied from: https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior

import torch.nn as nn

class Neural_Prior(nn.Module):
    def __init__(self, input_size=1000, dim_x=3, output_dim=3, filter_size=128, act_fn='relu', layer_size=8, output_feat=False):
        super().__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_feat = output_feat
        
        self.nn_layers = nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(nn.Sigmoid())
            for _ in range(layer_size-1):
                self.nn_layers.append(nn.Sequential(nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(nn.Sigmoid())
            self.nn_layers.append(nn.Linear(filter_size, output_dim))
        else:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, output_dim)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        if self.output_feat:
            feat = []
        for layer in self.nn_layers:
            x = layer(x)
            if self.output_feat and layer == nn.Linear:
                feat.append(x)

        if self.output_feat:
            return x, feat
        else:
            return x


