import torch.nn as nn

class sCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=12) -> None:
        super(sCNN, self).__init__()

        bias = True

        head = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        tail = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.model = nn.Sequential(
            head,
            tail
        )
        
    def forward(self, x):
        n = self.model(x)
        return x-n