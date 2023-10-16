import torch

from argparse import ArgumentParser

from model import VAE


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(in_channels=1, latent_dim=10).to(device)
    print(model)
    inputs = torch.zeros(1, 1, 32, 32, device=device)
    outputs, mean, var = model(inputs)
    print(outputs.shape)


if __name__ == "__main__":
    main()
