# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
import os

class MultiConditionDataset(Dataset):
    """加载图像-文本配对数据集（支持指定数据根目录）"""

    def __init__(self, root_dir: str, config, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录路径
            config (TrainConfig): 训练配置对象
            transform (callable, optional): 图像预处理变换
        """
        super().__init__()
        self.root_dir = root_dir
        self.config = config
        self.transform = transform

        # 初始化数据路径列表
        self.img_dir = os.path.join(root_dir, "images")
        self.cond_img_dir = os.path.join(root_dir, "cond_images")
        self.text_path = os.path.join(root_dir, "texts.txt")

        # 验证目录存在性
        self._validate_paths()

        # 加载数据路径和文本 需要修改
        self.img_paths, self.cond_img_paths, self.texts, self.filenames = self._load_data_paths()
    def _validate_paths(self):
        """验证必要目录和文件是否存在"""
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        if not os.path.exists(self.cond_img_dir):
            raise FileNotFoundError(f"条件图像目录不存在: {self.cond_img_dir}")
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"文本文件不存在: {self.text_path}")

    def _load_data_paths(self):
        """加载数据路径和对应文本描述"""
        # 读取文本描述文件（假设格式：filename,description）
        with open(self.text_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",", 1) for line in f.readlines()]

        img_paths = []
        cond_img_paths = []
        texts = []

        for filename, text in lines:
            img_path = os.path.join(self.img_dir, filename)
            cond_img_path = os.path.join(self.cond_img_dir, filename)

            if os.path.exists(img_path) and os.path.exists(cond_img_path):
                img_paths.append(img_path)
                cond_img_paths.append(cond_img_path)
                texts.append(text)
            else:
                print(f"警告：缺失文件 {filename}，已跳过")

        return img_paths, cond_img_paths, texts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.img_paths[idx]).convert("L")
        cond_img = Image.open(self.cond_img_paths[idx]).convert("L")

        # 应用预处理（需适配单通道）
        if self.transform:
            img = self.transform(img)
            cond_img = self.transform(cond_img)

        # 添加通道维度（如果转换未自动处理）
        if img.dim() == 2:  # [H, W]
            img = img.unsqueeze(0)  # [1, H, W]
            cond_img = cond_img.unsqueeze(0)

        # 获取文本
        text = self.texts[idx]

        return {
            "image": img,  # [C, H, W]
            "cond_image": cond_img,  # [C, H, W]
            "text": text  # str
        }


def get_dataloaders(config):
    # 初始化数据集
    train_dataset = MultiConditionDataset(
        root_dir=config.train_data_dir,
        config=config,
        transform=config.transform
    )

    test_dataset = MultiConditionDataset(
        root_dir=config.test_data_dir,
        config=config,
        transform=config.transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True  # 保持worker进程活跃
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader