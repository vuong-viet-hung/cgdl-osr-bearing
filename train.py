from pathlib import Path
from argparse import ArgumentParser

import torch

from model import VAE


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = VAE(in_channels=1, latent_dim=10).to(args.device)
    print(model)
    with torch.no_grad():
        inputs = torch.zeros(1, 1, 32, 32, device=args.device)
        outputs, mean, logvar = model(inputs)
    print(outputs.shape)


if __name__ == "__main__":
    main()
