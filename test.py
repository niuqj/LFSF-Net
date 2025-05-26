import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LFSFNet
from dataloader import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
import lpips


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

def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    # 实例化 LPIPS 模型
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            # 保存输出图像
            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

            # 计算 PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # 计算 SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

            # 计算 LPIPS：先从 [0,1] 映射到 [-1,1]
            lpips_val = lpips_fn(output * 2 - 1, high * 2 - 1)
            total_lpips += lpips_val.item()

    n = len(dataloader)
    avg_psnr  = total_psnr  / n
    avg_ssim  = total_ssim  / n
    avg_lpips = total_lpips / n
    return avg_psnr, avg_ssim, avg_lpips

def main():
    # 路径和设备
    test_low  = '/home/niu/code/LYT-Net/data/LOLv1/Test/input'
    test_high = '/home/niu/code/LYT-Net/data/LOLv1/Test/target'
    weights_path = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = os.path.basename(os.path.dirname(test_low))
    result_dir = os.path.join('results', dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(
        None, None,
        test_low, test_high,
        crop_size=None,
        batch_size=1
    )
    print(f'Test loader size: {len(test_loader)}')

    model = LFSFNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, result_dir)
    print(f'Validation PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, LPIPS: {avg_lpips:.6f}')

if __name__ == '__main__':
    main()
