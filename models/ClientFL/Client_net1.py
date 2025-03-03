
import torch
from torch import nn
from math import pi
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import avg_pool1d

class Client_net1(nn.Module):
    def __init__(self,n_class=10):
        super(Client_net1, self).__init__()
        self.n_class=n_class
        self.layer1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5,stride=2),
                                    nn.BatchNorm1d(8),
                                    nn.GELU())
        self.layer2 = nn.Sequential(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5,stride=2),
                                    nn.BatchNorm1d(16),
                                    nn.GELU())
        self.layer3 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
                                    nn.BatchNorm1d(32),
                                    nn.GELU())

        self.Clasifiier = nn.Sequential(nn.Flatten(),nn.Linear(32*125,self.n_class))

        self._features=nn.Sequential(self.layer1,self.layer2,self.layer3)
        self.instance_projector = nn.Sequential(
            # nn.Linear(32,32),
            # nn.ReLU(),
            nn.Linear(32, 64),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = F.adaptive_avg_pool1d(out, 1)
        feat = out.view(out.size(0), -1)
        return feat
    def features_attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x=self.instance_projector(x)
        return x

    def forward(self,x,mode='test'):
        x=self.layer1(x)
        x=self.layer2(x)
        out=self.layer3(x)
        out = self.Clasifiier(out)

        return out


if __name__ == '__main__':
    model_1=Client_net1().cuda()
    input=torch.ones((128,1,512)).cuda()
    print(input.shape)
    output=model_1(input)
    print(output.shape)
    # import torch
    #
    # print(torch.cuda.is_available())

