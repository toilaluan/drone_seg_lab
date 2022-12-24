import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

def resnet50(dilation=[7]):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    new_model = nn.Sequential(*list(model.children())[:-2])
    for j in dilation:
        for i in range(len(new_model[7])):
            if i == 0:
                new_model[j][i].conv2.stride = (1, 1)
                new_model[j][i].downsample[0].stride = (1, 1)
            new_model[j][i].conv2.dilation = (2, 2)
            new_model[j][i].conv2.padding = (2, 2)
    backbone = IntermediateLayerGetter(new_model, {'4':'feat', '7':'out'})
    return backbone