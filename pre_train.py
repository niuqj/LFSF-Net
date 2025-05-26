import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from model import FFT
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

def load_weights(model, weight_path, freeze_layers=False):
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        model_state_dict = model.state_dict()

        # 只加载匹配的层权重
        checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)  # 如果格式不同，做适应

        # 检查匹配的键并加载
        matching_keys = [k for k in checkpoint_state_dict if k in model_state_dict]
        model_state_dict.update({k: checkpoint_state_dict[k] for k in matching_keys})

        model.load_state_dict(model_state_dict)

        # 不冻结任何层，确保所有参数可训练
        for name, param in model.named_parameters():
            param.requires_grad = True  # 确保所有参数都可以训练
            print(f"解冻参数: {name}")

    return model

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
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim


    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def main():
    # 超参数设置
    #train_low = '/home/niu/code/LYT-Net/data/LOLv1/Train/input'
    #train_high = '/home/niu/code/LYT-Net/data/LOLv1/Train/target'
    #test_low = '/home/niu/code/LYT-Net/data/LOLv1/Test/input'
    #test_high = '/home/niu/code/LYT-Net/data/LOLv1/Test/target'

    train_low = '/home/niu/code/LYT-Net/data/LOLv2/Real_captured/Train/Low'
    train_high = '/home/niu/code/LYT-Net/data/LOLv2/Real_captured/Train/Normal'
    test_low = '/home/niu/code/LYT-Net/data/LOLv2/Real_captured/Test/Low'
    test_high = '/home/niu/code/LYT-Net/data/LOLv2/Real_captured/Test/Normal'

    #train_low = '/home/niu/code/LYT-Net/data/LOLv2/Synthetic/Train/Low'
    #train_high = '/home/niu/code/LYT-Net/data/LOLv2/Synthetic/Train/Normal'
    #test_low = '/home/niu/code/LYT-Net/data/LOLv2/Synthetic/Test/Low'
    #test_high = '/home/niu/code/LYT-Net/data/LOLv2/Synthetic/Test/Normal'
    # 数据加载器
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'训练数据加载器大小: {len(train_loader)}; 测试数据加载器大小: {len(test_loader)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFT().to(device)
    # 加载预训练的权重
    weight_path = '/home/niu/code/LYT-Net/PyTorch/pth/LOLv1/LOLv1_923_PSNR_26.119586_SSIM_0.000000_150.pth'
    model = load_weights(model, weight_path)
    learning_rate = 1e-7
    min_lr = 1e-9
    num_epochs = 500
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    first_decay_steps = 150*len(train_loader)
    T_0 = first_decay_steps
    T_mult = 2
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=T_0,T_mult=T_mult,eta_min=min_lr,last_epoch=-1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    print('训练开始。')
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs} - Current Learning Rate: {current_lr:.10f}')
        model.train()
        train_loss = 0.0
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'第 {epoch + 1}/{num_epochs} 轮, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), '/home/niu/code/LYT-Net/PyTorch/pth/LOLv2real/LOLv2.pth')
            print(f'保存模型，当前最佳 PSNR: {best_psnr:.6f}')

if __name__ == '__main__':
    main()
