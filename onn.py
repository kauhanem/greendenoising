import torch.nn as nn
from fastonn import SelfONN2d

class ONN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=12) -> None:
        super(ONN, self).__init__()

        bias = True

        model_head = nn.Sequential(
            SelfONN2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        model_tail = nn.Sequential(
            SelfONN2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.model = nn.Sequential(
            model_head,
            model_tail
        )
        
    def forward(self, x):
        n = self.model(x)
        return x-n