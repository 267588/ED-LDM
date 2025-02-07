
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
# 计算 PSNR 的函数
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # 完全相同的图像
    max_pixel = 255.
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# 计算 SSIM 的函数
def compute_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=255, multichannel=True, win_size=5, channel_axis=0)

def torch_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 255.0) -> torch.Tensor:
    """
    计算批量图像的 PSNR（支持 GPU）
    Args:
        img1: 输入图像1 (Tensor, shape=[B, C, H, W], 范围 [0, max_val])
        img2: 输入图像2 (Tensor, shape=[B, C, H, W], 范围 [0, max_val])
        max_val: 像素最大值（默认255）
    Returns:
        psnr: 标量值（平均PSNR）
    """
    mse = torch.mean((img1.float() - img2.float()) ** 2, dim=[1, 2, 3])  # [B]
    psnr = 10 * torch.log10(max_val**2 / (mse + 1e-10))  # 避免除以零
    return torch.mean(psnr)  # 返回批次平均PSNR

def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    生成高斯滤波器核
    Args:
        size: 核大小（奇数）
        sigma: 高斯标准差
        device: 设备类型 (e.g., 'cuda')
    Returns:
        kernel: 高斯核 (Tensor, shape=[1, 1, size, size])
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.outer(g)  # 外积生成二维核
    return kernel.view(1, 1, size, size)  # 扩展为4D张量


def torch_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    计算批量图像的 SSIM（支持 GPU）
    Args:
        img1: 输入图像1 (Tensor, shape=[B, C, H, W], 范围 [0, max_val])
        img2: 输入图像2 (Tensor, shape=[B, C, H, W], 范围 [0, max_val])
        max_val: 像素最大值（默认255）
        window_size: 滑动窗口大小（默认11）
        sigma: 高斯核标准差（默认1.5）
    Returns:
        ssim: 标量值（平均SSIM）
    """
    # 标准化到 [0,1]
    img1 = img1.float() / max_val
    img2 = img2.float() / max_val

    # 生成高斯核
    kernel = gaussian_kernel(window_size, sigma, img1.device)  # [1,1,11,11]
    padding = window_size // 2

    # 计算均值
    mu1 = torch.nn.functional.conv2d(img1, kernel, padding=padding, groups=1)
    mu2 = torch.nn.functional.conv2d(img2, kernel, padding=padding, groups=1)

    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, kernel, padding=padding, groups=1) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, kernel, padding=padding, groups=1) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, kernel, padding=padding, groups=1) - mu1_mu2

    # SSIM 参数
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )

    # 取空间维度均值（H,W），再取批次和通道均值
    return torch.mean(ssim_map, dim=[1, 2, 3]).mean()