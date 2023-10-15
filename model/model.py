from collections import OrderedDict

import torch


class VAE(torch.nn.Module):

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = encoder(in_channels)
        self.mean = torch.nn.Linear(64 * 16 * 16, latent_dim)
        self.var = torch.nn.Linear(64 * 16 * 16, latent_dim)
        self.decoder = decoder(latent_dim, in_channels)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        enc_outputs = self.encoder(inputs)
        mean = self.mean(enc_outputs)
        var = self.var(enc_outputs)
        latent = sample_gaussian(mean, var)
        outputs = self.decoder(latent)
        return outputs


def encoder(in_channels: int) -> torch.nn.Sequential:
    conv1_1 = conv3x3(in_channels, out_channels=32)
    conv1_2 = conv3x3(in_channels=32, out_channels=32)
    pool1 = torch.nn.MaxPool2d(kernel_size=2)
    conv2_1 = conv3x3(in_channels=32, out_channels=64)
    conv2_2 = conv3x3(in_channels=64, out_channels=64)
    pool2 = torch.nn.MaxPool2d(kernel_size=2)
    flatten = torch.nn.Flatten()
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("conv1_1", conv1_1),
                ("conv1_2", conv1_2),
                ("pool1", pool1),
                ("conv2_1", conv2_1),
                ("conv2_2", conv2_2),
                ("pool2", pool2),
                ("flatten", flatten),
            ]
        )
    )


def decoder(latent_dim: int, out_channels: int) -> torch.nn.Sequential:
    fc = torch.nn.Linear(latent_dim, 64 * 16 * 16)
    unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(64, 16, 16))
    upsample1 = torch.nn.Upsample(scale_factor=2)
    conv1_1 = conv3x3(in_channels=64, out_channels=32)
    conv1_2 = conv3x3(in_channels=32, out_channels=32)
    upsample2 = torch.nn.Upsample(scale_factor=2)
    conv2_1 = conv3x3(in_channels=32, out_channels=3)
    conv2_2 = conv3x3(in_channels=3, out_channels=out_channels)
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("fc", fc),
                ("unflatten", unflatten),
                ("upsample1", upsample1),
                ("conv1_1", conv1_1),
                ("conv1_2", conv1_2),
                ("upsample2", upsample2),
                ("conv2_1", conv2_1),
                ("conv2_2", conv2_2),
            ]
        )
    )


@torch.no_grad()
def sample_gaussian(
    mean: torch.FloatTensor, var: torch.FloatTensor
) -> torch.FloatTensor:
    eps = torch.randn(mean.shape)
    std = var.sqrt()
    return mean + eps * std


def conv3x3(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    conv = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1
    )
    relu = torch.nn.ReLU()
    return torch.nn.Sequential(OrderedDict([("conv", conv), ("relu", relu)]))
