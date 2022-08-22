import megengine.hub as hub
import megengine.module as M


class DnCNN(M.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super().__init__()

        layers = []
        layers.append(M.Conv2d(image_channels, n_channels,
                      kernel_size, stride=1, padding=1, bias=True))
        layers.append(M.ReLU())
        for _ in range(depth - 2):
            layers.append(M.Conv2d(n_channels, n_channels,
                          kernel_size, stride=1, padding=1, bias=False))
            layers.append(M.BatchNorm2d(n_channels, eps=0.0001, momentum=0.05))
            layers.append(M.ReLU())
        layers.append(M.Conv2d(n_channels, image_channels,
                      kernel_size, stride=1, padding=1, bias=False))
        self.dncnn = M.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/96/files/f5b97483-70f8-43a7-9526-4851122d2aaa"
)
def dncnn_25():
    """DnCNN for gaussian denoising (sigma = 25)"""
    return DnCNN()
