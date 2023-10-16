import torch


class VAE(torch.nn.Module):

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(
        self, inputs: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        mean, var = self.encoder(inputs)
        latent = random_sample(mean, var)
        outputs = self.decoder(latent)
        return outputs, mean, var


class Encoder(torch.nn.Module):

    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        self.conv1_1 = ConvReLU(in_channels, out_channels=32)
        self.conv1_2 = ConvReLU(in_channels=32, out_channels=32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = ConvReLU(in_channels=32, out_channels=64)
        self.conv2_2 = ConvReLU(in_channels=64, out_channels=64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 8 * 8, out_features)
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(64 * 8 * 8, out_features),
            torch.nn.Softplus(),
        )

    def forward(
        self, inputs: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        conv1_1 = self.conv1_1(inputs)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        flatten = self.flatten(pool2)
        mean = self.fc1(flatten)
        var = self.fc2(flatten)
        return mean, var


class Decoder(torch.nn.Module):

    def __init__(self, in_features: int, out_channels: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_features, 64 * 8 * 8)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(64, 8, 8))
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.deconv1_1 = ConvReLU(in_channels=64, out_channels=32)
        self.deconv1_2 = ConvReLU(in_channels=32, out_channels=32)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.deconv2_1 = ConvReLU(in_channels=32, out_channels=3)
        self.deconv2_2 = ConvSigmoid(in_channels=3, out_channels=out_channels)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        fc = self.fc(inputs)
        unflatten = self.unflatten(fc)
        upsample1 = self.upsample1(unflatten)
        deconv1_1 = self.deconv1_1(upsample1)
        deconv1_2 = self.deconv1_2(deconv1_1)
        upsample2 = self.upsample2(deconv1_2)
        deconv2_1 = self.deconv2_1(upsample2)
        deconv2_2 = self.deconv2_2(deconv2_1)
        return deconv2_2


class ConvReLU(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.relu(self.conv(inputs))


class ConvSigmoid(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.sigmoid(self.conv(inputs))


@torch.no_grad()
def random_sample(
    mean: torch.FloatTensor, var: torch.FloatTensor
) -> torch.FloatTensor:
    std = var.sqrt()
    latent = torch.normal(mean, std)
    return latent  # type: ignore
