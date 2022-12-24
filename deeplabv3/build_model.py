import torch
import torch.nn as nn
import torch.nn.functional as F 

from .backbone import resnet50 
from .aspp import ASPP, BasicBlock

class DeeplabV3plus(nn.Module):

    def __init__(self, atrous=[6, 12, 18], num_classes=21):
        super(DeeplabV3plus, self).__init__()
        self.backbone = resnet50([7])
        self.aspp = ASPP(2048, 256, atrous)
        self.lowconv = BasicBlock(in_channels=256, out_channels=48, kernel_size=1,
                                 stride=1, padding=0, dilation=1, bias=True)
        self.middle1 = BasicBlock(in_channels=304, out_channels=256, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bias=True)
        self.middle2 = BasicBlock(in_channels=256, out_channels=256, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.cls = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bias=True)
    
    def forward(self, x):
        feature_dict = self.backbone(x)
        last_fea = feature_dict['out']
        logit_map = self.aspp(last_fea)
        low_fea = feature_dict['feat']
        low_fea = self.lowconv(low_fea)
        logit_map = F.interpolate(logit_map, scale_factor=4, mode="bilinear", align_corners=True)
        mid_fea = torch.concat([low_fea, logit_map], axis=1)
        mid_fea = self.dropout1(mid_fea)
        mid_fea = self.middle1(mid_fea)
        mid_fea = self.dropout2(mid_fea)
        mid_fea = self.middle2(mid_fea)
        final_fea = F.interpolate(mid_fea, scale_factor=4, mode="bilinear", align_corners=True)
        final_logit = self.cls(final_fea)
        return final_logit


if __name__ == "__main__":
    model = DeeplabV3plus()
    print(model(torch.randn((2, 3, 512, 512))).size())