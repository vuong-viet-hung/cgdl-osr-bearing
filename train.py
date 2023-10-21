from pathlib import Path
from argparse import ArgumentParser

import torch

from dataloader import cwru
from model import VAE


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_dl, test_dl, val_dl = cwru.make_dataloaders(
        args.data_dir, "12k_DE", args.batch_size
    )
    train_iter = iter(train_dl)
    spectrogram_batch, target_batch = next(train_iter)
    print(spectrogram_batch.shape)

    model = VAE(in_channels=1, latent_dim=10).to(args.device)
    print(model)
    with torch.no_grad():
        inputs = torch.zeros(1, 1, 32, 32, device=args.device)
        outputs, mean, logvar = model(inputs)
    print(outputs.shape)


if __name__ == "__main__":
    main()
