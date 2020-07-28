import numpy as np
from chainer.links.caffe import CaffeFunction
import torch
import torch.nn as nn

import model


class Converter(object):

    @classmethod
    def convert_feature_params(self, caffe_parameter_path):
        caffe_model = CaffeFunction(caffe_parameter_path)
        pytorch_model = model.FeatureI2V()

        conv = []
        linear = []
        for child in caffe_model.children():
            if 'conv' in child.name:
                for params in child.namedparams():
                    conv.append(params[1].array)
            elif 'encode' in child.name:
                for params in child.namedparams():
                    linear.append(params[1].array)

        c = 0
        for param in pytorch_model.features:
            if 'conv' in str(type(param)):
                param.weight = torch.nn.Parameter(torch.tensor(conv[2*c]))
                param.bias = torch.nn.Parameter(torch.tensor(conv[2*c+1]))
                c += 1

        pytorch_model.encode1[0].weight = torch.nn.Parameter(torch.tensor(linear[0]))
        pytorch_model.encode1[0].bias = torch.nn.Parameter(torch.tensor(linear[1]))
        pytorch_model.encode2[0].weight = torch.nn.Parameter(torch.tensor(linear[2]))
        pytorch_model.encode2[0].bias = torch.nn.Parameter(torch.tensor(linear[3]))

        torch.save(pytorch_model.state_dict(), './feature_parameter.pth')
        print('Converted!!')


    @classmethod
    def convert_tag_params(self, caffe_parameter_path):
        caffe_model = CaffeFunction(caffe_parameter_path)
        pytorch_model = model.TagI2V()

        conv = []
        for child in caffe_model.children():
            for params in child.namedparams():
                conv.append(params[1].array)
        
        c = 0
        for param in pytorch_model.net:
            if 'conv' in str(type(param)):
                param.weight = torch.nn.Parameter(torch.tensor(conv[2*c]))
                param.bias = torch.nn.Parameter(torch.tensor(conv[2*c+1]))
                c += 1
        
        torch.save(pytorch_model.state_dict(), './tag_parameter.pth')
        print('Converted!!')

