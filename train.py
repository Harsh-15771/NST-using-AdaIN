import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_image", type=str, default=r"content_data", help="Location of stored content image")
    parser.add_argument("--style_image", type=str, default=r"style_data", help="Location of stored style image")
    parser.add_argument("--vgg", type=str, default=r"vgg_normalised.pth", help="Location of stored vgg model")
    parser.add_argument("--experiment", type=str, default="experiment1", help="Name of the experiment")

    parser.add_argument("--final_size", type=int, default=256, help="Final size of the image")
    parser.add_argument("--content_size", type=int, default=512, help="Size of the content image before crop/resize")
    parser.add_argument("--style_size", type=int, default=512, help="Size of the style image before crop/resize")
    parser.add_argument("--crop", action="store_true", default=True, help="Whether to crop the image")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--lr_decay", type=float, default=5e-5, help="Learning rate decay for training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train in THIS run")

    parser.add_argument("--content_weight", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("--style_weight", type=float, default=5, help="Weight for style loss")

    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging training progress")
    parser.add_argument("--save_interval", type=int, default=2, help="Interval for saving model checkpoints")

    parser.add_argument("--resume", action="store_true", default=False, help="Whether to resume training from a checkpoint")
    parser.add_argument("--decoder_path", type=str, default=None, help="Location of stored decoder model for resuming training")
    parser.add_argument("--optimizer_path", type=str, default=None, help="Location of stored optimizer state for resuming training")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch number already completed before this run. Example: if resuming from epoch_20, set start_epoch=20")

    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(save_dir / "args.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Datasets / Dataloaders
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_image, transform=content_transform)
    style_dataset = ImageFolderDataset(args.style_image, transform=style_transform)

    print(f"Number of content images: {len(content_dataset)}")
    print(f"Number of style images: {len(style_dataset)}")

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    # Models
    encoder = VGGEncoder(args.vgg)
    encoder.eval()

    decoder = Decoder()

    use_dp = torch.cuda.device_count() > 1

    # Multi-GPU support
    if use_dp:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    # Resume
    if args.resume:
        if args.decoder_path is None or args.optimizer_path is None:
            raise ValueError("For resume=True, both decoder_path and optimizer_path must be provided.")

        print(f"Resuming from decoder: {args.decoder_path}")
        print(f"Resuming from optimizer: {args.optimizer_path}")

        decoder_ckpt = torch.load(args.decoder_path, map_location=device)
        optimizer_ckpt = torch.load(args.optimizer_path, map_location=device)

        if isinstance(decoder, torch.nn.DataParallel):
            decoder.module.load_state_dict(decoder_ckpt)
        else:
            decoder.load_state_dict(decoder_ckpt)

        optimizer.load_state_dict(optimizer_ckpt)

    mse_loss = torch.nn.MSELoss()
    total_epochs = args.start_epoch + args.epochs

    for epoch in range(args.start_epoch, total_epochs):
        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=min(len(content_dataloader), len(style_dataloader))
        )

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        for content_batch, style_batch in progress_bar:
            content_images = content_batch.to(device, non_blocking=True)
            style_images = style_batch.to(device, non_blocking=True)

            c_feats = encoder(content_images)
            s_feats = encoder(style_images)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)
            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for gf, sf in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(gf)
                s_mean, s_std = calc_mean_std(sf)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            loss_s *= args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, Content Loss: {loss_c.item():.4f}, Style Loss: {loss_s.item():.4f}"
            )

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        num_batches = min(len(content_dataloader), len(style_dataloader))
        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        current_epoch = epoch + 1

        if current_epoch % args.log_interval == 0:
            tqdm.write(
                f"Epoch [{current_epoch}/{total_epochs}], "
                f"Loss: {running_loss:.4f}, "
                f"Content Loss: {running_closs:.4f}, "
                f"Style Loss: {running_sloss:.4f}"
            )

        if current_epoch % args.save_interval == 0:
            decoder_state = decoder.module.state_dict() if isinstance(decoder, torch.nn.DataParallel) else decoder.state_dict()

            torch.save(decoder_state, save_dir / f"decoder_epoch_{current_epoch}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_epoch_{current_epoch}.pth")

            with torch.no_grad():
                output = torch.cat([content_images, style_images, g], dim=0)
                save_image(output.cpu(), save_dir / f"output_epoch_{current_epoch}.png", nrow=args.batch_size)

            tqdm.write(f"Saved checkpoint at epoch {current_epoch}")

    # Save final model at the end of this run
    final_decoder_state = decoder.module.state_dict() if isinstance(decoder, torch.nn.DataParallel) else decoder.state_dict()
    torch.save(final_decoder_state, save_dir / "decoder_final.pth")
    torch.save(optimizer.state_dict(), save_dir / "optimizer_final.pth")
    tqdm.write("Saved final model for this run.")


if __name__ == "__main__":
    main()