import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17) -> None:
        super(DnCNN, self).__init__()

        bias = True

        head = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        
        bodyLayer = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(nc, momentum=0.9, eps=1e-04, affine=True),
            nn.ReLU(inplace=True)
        )

        body = [bodyLayer for _ in range(nb-2)]

        tail = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.model = nn.Sequential(
            head,
            *body,
            tail
        )
        
    def forward(self, x):
        n = self.model(x)
        return x-n