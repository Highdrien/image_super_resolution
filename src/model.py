import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, upscale_factor, hidden_channels_1, hidden_channels_2):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, hidden_channels_1, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(hidden_channels_2, 3 * (upscale_factor ** 2), kernel_size=3, stride=1, padding='same')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


def get_model(config):
    model = Net(upscale_factor=config.upscale_factor,
                hidden_channels_1=config.model.hidden_channels_1,
                hidden_channels_2=config.model.hidden_channels_2)
    return model


if __name__ == "__main__":
    model = Net(upscale_factor=3, hidden_channels_1=64, hidden_channels_2=32)