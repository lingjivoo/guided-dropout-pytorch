import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from guided_dropout import GuidedDropout


def dropout_selection(drop_type,dim,drop_rate):
    if 'GuidedDropout' in drop_type:
        drop = GuidedDropout(dim,drop_rate)
    else:
        drop = nn.Dropout(drop_rate)
    return drop


class HiddenLayer(nn.Module):
    def __init__(self, in_dim,out_dim, act_layer=nn.ReLU):
        super(HiddenLayer, self).__init__()
        if in_dim is None:
            raise ValueError("{} must be an int".format(in_dim))
        if out_dim is None:
            out_dim = in_dim
        if act_layer is None:
            self.act = nn.Identity()
        else:
            self.act = act_layer()
        self.hidden_layer = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        x = self.hidden_layer(x)
        x = self.act(x)
        return x


class NN(nn.Module):
    def __init__(self, image_size, hidden_layers, layer=HiddenLayer, num_classes=10,RGB_Image=True,drop=None,drop_rate=0.2):
        super(NN, self).__init__()
        self.input_dim = image_size ** 2
        if RGB_Image:
            self.input_dim = self.input_dim * 3
        in_dim = self.input_dim
        layers = []
        for dim in hidden_layers:
            out_dim = dim
            layers.append(layer(in_dim,out_dim))
            if drop is not None:
                layers.append(dropout_selection(drop,dim,drop_rate))
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)
        self.fc = nn.Linear(in_dim, num_classes)
        self.step = 0

    def forward(self, x):
        b,_,_,_ = x.shape
        x = x.view(b,-1)
        x = self.hidden(x)
        x = self.fc(x)
        return x

    # #top-k scheduler
    # def dropout_scheduler(self, epoch, step_epochs=[40, 50,90,100,140,150]):
    #     if epoch in step_epochs:
    #         drop_rate_steps = [0.2, 0.0, 0.15, 0.0, 0.1, 0.0]
    #         drop_rate = drop_rate_steps[self.step]
    #         self.step +=1
    #         for m in self.modules():
    #             if isinstance(m, self.drop):
    #                 if m.begin_flag:
    #                     m.drop_rate = drop_rate
    #                 else:
    #                     m.begin_flag = True

    #DR scheduler
    def dropout_scheduler(self,epoch,step_epochs=[0,2,150]):
        if epoch in step_epochs:
            for m in self.modules():
                if isinstance(m, GuidedDropout):
                    if m.begin_flag:
                        m.drop_rate = m.drop_rate - 0.05
                    else:
                        m.begin_flag = True


def mlp(image_size=32,hidden_dim=1024,depth=3,layer=HiddenLayer,num_classes=10,RGB_Image=True,drop=None,drop_rate=0.2):
    hidden_layers = [hidden_dim for i in range(depth)]
    return NN(image_size,hidden_layers,layer,num_classes,RGB_Image,drop = drop,drop_rate=drop_rate)


if __name__=="__main__":
    x = torch.randn(1,3,10,10).cuda()
    nn = mlp(10,drop='GuidedDropout').cuda()
    # nn = mlp(10,drop='Dropout',drop_rate=0.1).cuda()
    out = nn(x)
    print("out:",out.shape)
    print(out.sum())