"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch.nn as nn
import torchvision.models as models




class SimSiam(nn.Module):

    def __init__(self, embedding_dim=1000, pred_dim=512):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 2048, bias=False),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(2048, 2048, bias=False),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(2048, embedding_dim, bias=False),
                                        nn.BatchNorm1d(embedding_dim, affine=False)) # output layer
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(embedding_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, embedding_dim)) # output layer


    def forward(self, x, y=None):
        if self.training:

            z_left = self.resnet(x)
            z_right = self.resnet(y)

            p_left = self.predictor(z_left)
            p_right = self.predictor(z_right)

            return p_left, z_right.detach(), p_right, z_left.detach()
        else:
            return self.resnet(x)
