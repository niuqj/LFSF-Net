import torch
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LFSFNet
from losses import CombinedLoss
from dataloader import create_dataloaders
from tqdm.auto import tqdm  # progress bar
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

# --------------------------- Command-line arguments --------------------------- #
parser.add_argument('--seed', type=int, default=923,
                    help='Random seed for full reproducibility.')
parser.add_argument('--dataset', type=str, default='LOLv1',
                    help='Dataset name, used mainly for logging and checkpoint naming.')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of training epochs.')
parser.add_argument('--train_low', type=str, default='/home/niu/code/LYT-Net/data/LOLv1/Train/input',
                    help='Directory with low-light training images.')
parser.add_argument('--train_high', type=str, default='/home/niu/code/LYT-Net/data/LOLv1/Train/target',
                    help='Directory with ground-truth training images.')
parser.add_argument('--test_low', type=str, default='/home/niu/code/LYT-Net/data/LOLv1/Test/input',
                    help='Directory with low-light validation images.')
parser.add_argument('--test_high', type=str, default='/home/niu/code/LYT-Net/data/LOLv1/Test/target',
                    help='Directory with ground-truth validation images.')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='Initial learning rate for the Adam optimizer.')
parser.add_argument('--save_dir', type=str, default='/home/niu/code/LYT-Net/PyTorch/pth',
                    help='Directory where checkpoints will be saved.')
opt = parser.parse_args()

# --------------------------- Utility functions --------------------------- #

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)
            total_psnr += calculate_psnr(output, high)
            total_ssim += calculate_ssim(output, high)

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim


def set_seed_torch(seed=622):
    """Completely fix random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------- Main training loop --------------------------- #

def main():
    set_seed_torch(opt.seed)

    # Paths & hyper-parameters taken from CLI
    train_low_path = opt.train_low
    train_high_path = opt.train_high
    test_low_path = opt.test_low
    test_high_path = opt.test_high
    learning_rate = opt.learning_rate
    num_epochs = opt.num_epochs
    save_dir = opt.save_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(
        train_low_path, train_high_path, test_low_path, test_high_path,
        crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    model = LFSFNet().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20, factor=0.9, verbose=True)

    scaler = torch.cuda.amp.GradScaler()
    best_psnr = 0.0
    best_ssim = 0.0

    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss/batch_idx:.4f}")
        progress_bar.close()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs} | PSNR: {avg_psnr:.6f} | SSIM: {avg_ssim:.6f}')

        scheduler.step(avg_psnr)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            os.makedirs(save_dir, exist_ok=True)
            ckpt_name = f"{opt.dataset}_{opt.seed}_PSNR_{best_psnr:.4f}_SSIM_{best_ssim:.4f}.pth"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f'Saved best model to: {ckpt_path}')


if __name__ == '__main__':
    main()
