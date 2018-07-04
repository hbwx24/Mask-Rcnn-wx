# _*_coding:utf-8_*_
# Author:WeiXin
# Email:weixin_ict@163.com
# 2018/7/4

import torch.nn as nn
import torch
#resnet
import torchvision.models.resnet as resnet

#path='/home/weixin/.torch/models/resnet18-5c106cde.pth'

class Conv_Body():
    def __init__(self,net='resnet18',pretrained=False,path_pretrain=None):
        if pretrained:
            self.pretrained_model=torch.load(path_pretrain)
            
        if net=='resnet18':
            resnet18=resnet.resnet18()
            resnet18.load_state_dict({k:v for k,v in self.pretrained_model.items() if k in resnet18.state_dict()})
            self.conv_body=nn.Sequential(*[resnet18.layer1,
                                           resnet18.layer2,
                                           resnet18.layer3,
                                           resnet18.layer4])
            
            