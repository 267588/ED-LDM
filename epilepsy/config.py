# configs.py
from dataclasses import dataclass
from torchvision import transforms
import torch
@dataclass
class TrainConfig:
    dataSize =256
    # 数据配置
    batch_size: int = 4
    num_workers: int = 1

    # 模型架构配置
    latent_dim: int = 4  # VQVAE潜空间维度
    text_embed_dim: int = 512  # CLIP文本编码维度
    cond_channels: int = 8  # 条件通道数（图像4 + 文本4）
    timesteps: int = 1000  # 扩散时间步数
    latent_shape = (latent_dim,64,64)
    # 训练参数
    scheduler_type: str = "squaredcos_cap_v2"  # 可选 ["linear", "cosine", "sigmoid", "sqrt"]
    beta_start: float = 1e-5
    beta_end: float = 0.02
    cosine_s: float = 0.008  # 余弦调度偏移量
    lr: float = 3e-4
    milestones = []
    epochs: int = 1000
    save_interval: int = 800  # 模型保存间隔
    device : str= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 路径配置
    train_data_dir: str = "/mntcephfs/med_dataset/ZH/Data/train"
    test_data_dir: str = "/mntcephfs/med_dataset/ZH/Data/test"
    vqvae_dir: str = "/mntcephfs/lab_data/wangcm/zh/Epilepsy/vqvae_model.pth"
    Diffusion_dir: str = "/mntcephfs/lab_data/wangcm/zh/Epilepsy/best_model.pt"
    save_best: bool = True  # 是否保存最佳模型
    save_dir: str = "/mntcephfs/lab_data/wangcm/zh/Epilepsy/Diffusion_result"
    eval_save: str = "/mntcephfs/lab_data/wangcm/zh/Epilepsy/Diffusion_result/eval_results"
    maskPath: str = "/mntcephfs/med_dataset/ZH/Data/mask_test"
    #classifier free guidance
    cond_drop_prob_img: float = 0.1  # 图像条件丢弃概率
    cond_drop_prob_text: float = 0.1  # 文本条件丢弃概率
    guidance_scale_img: float = 7.5  # 图像引导比例
    guidance_scale_text: float = 7.5  # 文本引导比例
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((dataSize, dataSize)),
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomRotation(10),   # 随机旋转 ±10 度
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])