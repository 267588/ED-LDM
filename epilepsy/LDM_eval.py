import torch
import numpy as np
from datasetEval import get_dataloaders, MultiConditionDataset
from config import TrainConfig
from model import LatentDiffusion, myVQVAE, TextConditionEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from diffusers import DDPMScheduler
import metrics as Metrics
from torchvision.io import read_image
from torchvision.utils import save_image
config = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_results_to_txt(results, file_path):
    with open(file_path, 'w') as f:
        # 写入 PSNR 和 SSIM
        f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f}\n")
        f.write(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n\n")

        # 写入阈值对应指标
        f.write("Threshold Metrics:\n")
        f.write("Threshold\tDice\tIoU\tPrecision\tRecall\tAccuracy\tF1\n")
        for t in sorted(results['threshold_metrics'].keys()):
            metrics = results['threshold_metrics'][t]
            f.write(
                f"{t:.2f}\t"
                f"{metrics['dice']['mean']:.4f}±{metrics['dice']['std']:.4f}\t"
                f"{metrics['iou']['mean']:.4f}±{metrics['iou']['std']:.4f}\t"
                f"{metrics['precision']['mean']:.4f}±{metrics['precision']['std']:.4f}\t"
                f"{metrics['recall']['mean']:.4f}±{metrics['recall']['std']:.4f}\t"
                f"{metrics['acc']['mean']:.4f}±{metrics['acc']['std']:.4f}\t"
                f"{metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f}\n"
            )

# 修改数据集类以保留文件名信息
class EnhancedDataset(MultiConditionDataset):
    def _load_data_paths(self):
        with open(self.text_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",", 1) for line in f.readlines()]

        img_paths = []
        filenames = []  # 新增文件名存储
        cond_img_paths = []
        texts = []

        for filename, text in lines:
            img_path = os.path.join(self.img_dir, filename)
            cond_img_path = os.path.join(self.cond_img_dir, filename)

            if os.path.exists(img_path) and os.path.exists(cond_img_path):
                img_paths.append(img_path)
                filenames.append(filename)  # 记录文件名
                cond_img_paths.append(cond_img_path)
                texts.append(text)

        return img_paths, cond_img_paths, texts, filenames  # 返回新增的filenames

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data["filename"] = os.path.basename(self.img_paths[idx])  # 添加文件名
        return data


# 加载模型（保持原样）
def load_model():
    vqvae = myVQVAE.to(device)
    vqvae.load_state_dict(torch.load(config.vqvae_dir, map_location=device))
    vqvae.eval()

    text_encoder = TextConditionEncoder(config.latent_shape).to(device)
    text_encoder.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.scheduler_type
    )
    model = LatentDiffusion(config, noise_scheduler).to(device)
    model.load_state_dict(torch.load(config.Diffusion_dir,map_location=device))
    return vqvae, text_encoder, model


def match_histograms_gpu(source, target, channel_axis=1):
    """
    GPU版本的直方图匹配
    Args:
        source: 待匹配图像 (Tensor, shape=[B,C,H,W])
        target: 目标图像 (Tensor, shape=[B,C,H,W])
        channel_axis: 通道轴 (默认为1)
    Returns:
        matched: 匹配后的图像 (Tensor)
    """
    matched = []
    for s, t in zip(source, target):
        s_flat = s.reshape(s.shape[channel_axis], -1)  # [C, H*W]
        t_flat = t.reshape(t.shape[channel_axis], -1)

        s_sorted = torch.sort(s_flat, dim=1)[0]
        t_sorted = torch.sort(t_flat, dim=1)[0]

        # 计算映射关系
        quantiles = torch.linspace(0, 1, steps=s_flat.shape[1], device=s.device)
        s_quantiles = torch.quantile(s_sorted, quantiles, dim=1)
        t_quantiles = torch.quantile(t_sorted, quantiles, dim=1)

        # 插值匹配
        matched_flat = torch.zeros_like(s_flat)
        for c in range(s_flat.shape[0]):
            interp = torch_interp(s_flat[c], s_quantiles[c], t_quantiles[c])
            matched_flat[c] = interp

        matched.append(matched_flat.reshape(s.shape))

    return torch.stack(matched)
# 指标计算函数（保持原样）
def calculate_metrics(pred_mask, gt_mask):
    # 输入应为GPU张量
    pred = pred_mask.flatten().bool()  # GPU布尔张量
    gt = gt_mask.flatten().bool()

    tp = torch.sum(pred & gt).float()
    fp = torch.sum(pred & ~gt).float()
    fn = torch.sum(~pred & gt).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (pred == gt).float().mean()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return dice, iou, precision, recall, acc, f1


def torch_interp(x, xp, fp):
    """
    自定义PyTorch一维线性插值函数，模拟numpy.interp
    参数：
        x: 待插值的点 (Tensor)
        xp: 已知的x坐标（必须单调递增）(Tensor)
        fp: 已知的y坐标 (Tensor)
    返回：
        插值后的值 (Tensor)
    """
    # 处理边界条件，确保xp是单调递增的
    xp, indices_sort = torch.sort(xp)
    fp = fp[indices_sort]

    # 查找插入位置
    indices = torch.searchsorted(xp, x, right=True)

    # 限制索引在有效范围内
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # 获取相邻的xp和fp值
    x_low = xp[indices - 1]
    x_high = xp[indices]
    f_low = fp[indices - 1]
    f_high = fp[indices]

    # 计算插值权重
    delta = x_high - x_low
    delta = torch.where(delta == 0, 1e-8, delta)  # 避免除以零
    weight = (x - x_low) / delta

    # 线性插值
    interp_val = f_low + weight * (f_high - f_low)

    # 处理x超出xp范围的情况
    interp_val = torch.where(x < xp[0], fp[0], interp_val)
    interp_val = torch.where(x > xp[-1], fp[-1], interp_val)

    return interp_val


def match_histograms_gpu(source, target, channel_axis=1):
    matched = []
    for s, t in zip(source, target):
        s_flat = s.reshape(s.shape[channel_axis], -1)  # [C, H*W]
        t_flat = t.reshape(t.shape[channel_axis], -1)

        s_sorted = torch.sort(s_flat, dim=1)[0]
        t_sorted = torch.sort(t_flat, dim=1)[0]

        quantiles = torch.linspace(0, 1, steps=s_flat.shape[1], device=s.device)
        s_quantiles = torch.quantile(s_sorted, quantiles, dim=1)
        t_quantiles = torch.quantile(t_sorted, quantiles, dim=1)

        matched_flat = torch.zeros_like(s_flat)
        for c in range(s_flat.shape[0]):
            # 确保分位数单调递增
            s_q, _ = torch.sort(s_quantiles[c])
            t_q, _ = torch.sort(t_quantiles[c])
            # 使用自定义插值
            interp = torch_interp(s_flat[c], s_q, t_q)
            matched_flat[c] = interp

        matched.append(matched_flat.reshape(s.shape))
    return torch.stack(matched)

def evaluate():
    vqvae, text_encoder, model = load_model()

    # 使用增强版数据集
    test_dataset = EnhancedDataset(
        root_dir=config.test_data_dir,
        config=config,
        transform=config.transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    thresholds = np.linspace(0, 1, 11)
    metrics = {t: {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'acc': [], 'f1': []}
               for t in thresholds}
    all_psnr = []
    all_ssim = []

    # 定义mask根目录（根据实际路径修改！）
    mask_root = config.maskPath  # 需要用户修改！！！

    for batch in tqdm(test_loader):
        # 获取文件名并构建mask路径
        filenames = batch['filename']
        mask_paths = [os.path.join(mask_root, fname) for fname in filenames]

        # 加载mask（假设为二值图像）
        gt_masks = []
        for mp in mask_paths:
            mask = read_image(mp).to(device).float() / 255.0  # 直接加载到GPU

            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=(256, 256),
                mode='nearest'
            ).squeeze(0) #特定数据集图像resize


            gt_masks.append(mask)

        gt_masks = torch.stack(gt_masks, dim=0)  # 使用PyTorch堆叠，保持张量在GPU
        gt_masks = gt_masks.squeeze(1)  # 移除通道维度（假设原mask为单通道）
        # 模型推理
        img = batch['image'].to(device)
        cond_img = batch['cond_image'].to(device)
        texts = batch['text']

        with torch.no_grad():
            cond_z = vqvae.encode(cond_img)
            text_inputs = text_encoder.tokenizer(texts, padding=True, return_tensors="pt").to(device)
            text_emb = text_encoder.clip_model(**text_inputs).last_hidden_state.mean(dim=1)
            text_cond = text_encoder.projection(text_emb)
            reconstructed = model.sample(cond_z,
                                         text_cond,
                                         guidance_scale_img=3.0,
                                         guidance_scale_text=5.0
                                         )
            # 保存特征图（归一化到 [0, 255]）
            feature_maps_path = os.path.join(config.eval_save, "feature_maps")  # 特征图保存路径
            os.makedirs(feature_maps_path, exist_ok=True)  # 创建目录

            for i, filename in enumerate(filenames):
                feature_map = reconstructed[i]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                feature_map = (feature_map * 255).byte()
                feature_map_path = os.path.join(feature_maps_path, f"feature_{filename}")
                save_image(feature_map, feature_map_path)
            reconstructed = vqvae.decode(reconstructed)
            reconstructed = torch.clamp(reconstructed, -1.0, 1.0)

        # 直方图匹配
        original = img.to(device)
        matched = match_histograms_gpu(reconstructed, original, channel_axis=1)

        original_uint8 = ((original + 1) * 127.5).clamp(0, 255).byte()
        recon_uint8 = ((matched + 1) * 127.5).clamp(0, 255).byte()
        #重建图像保存
        # 保存重建图像
        reconstructed_path = config.eval_save  # 假设 config.eval_save 是保存路径
        for i, filename in enumerate(filenames):
            save_name = f"reconstructed_{filename}"
            save_image(recon_uint8[i], os.path.join(reconstructed_path, save_name))
        # 计算指标
        all_psnr.append(Metrics.torch_psnr(original_uint8, recon_uint8).item())
        all_ssim.append(Metrics.torch_ssim(original_uint8, recon_uint8).item())

        # 生成差异图
        diff_map = (recon_uint8.float() - original_uint8.float()).abs() / 255.0  # 归一化到[0,1]
        diff_map = diff_map.mean(dim=1)
        print(diff_map.shape)
        # 遍历阈值计算指标
        for t in thresholds:
            pred_masks = (diff_map > t).float()

            # 创建当前阈值对应的文件夹
            threshold_dir = os.path.join(reconstructed_path, f"threshold_{t:.2f}")
            os.makedirs(threshold_dir, exist_ok=True)

            for i, (filename, pred, gt) in enumerate(zip(filenames, pred_masks, gt_masks)):
                save_name = f"pred_mask_{filename}"

                dice, iou, p, r, acc, f1 = calculate_metrics(pred, gt)
                metrics[t]['dice'].append(dice)
                metrics[t]['iou'].append(iou)
                metrics[t]['precision'].append(p)
                metrics[t]['recall'].append(r)
                metrics[t]['acc'].append(acc)
                metrics[t]['f1'].append(f1)
                # pred_masks图像保存
                pred = (pred * 255).byte()
                save_image(pred, os.path.join(threshold_dir, save_name))

        # 结果输出与保存（新增计算部分）
        print(f"PSNR: {torch.mean(torch.tensor(all_psnr)):.2f} ± {torch.std(torch.tensor(all_psnr)):.2f}")
        print(f"SSIM: {torch.mean(torch.tensor(all_ssim)):.4f} ± {torch.std(torch.tensor(all_ssim)):.4f}")

        # 绘制指标曲线
        plt.figure(figsize=(10, 6))
        for metric in ['dice', 'iou', 'f1', 'precision', 'recall', 'acc']:
            metric_values = [torch.mean(torch.tensor(metrics[t][metric])).item() for t in thresholds]
            plt.plot(thresholds, metric_values, label=metric)
        plt.legend()
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.savefig(f"{config.eval_save}/metrics_plot.png", dpi=300, bbox_inches='tight')

        # 计算各指标的平均值和标准差
        threshold_metrics_summary = {}
        for t in thresholds:
            threshold_metrics_summary[t] = {
                'dice': {
                    'mean': torch.mean(torch.tensor(metrics[t]['dice'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['dice'])).item()
                },
                'iou': {
                    'mean': torch.mean(torch.tensor(metrics[t]['iou'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['iou'])).item()
                },
                'precision': {
                    'mean': torch.mean(torch.tensor(metrics[t]['precision'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['precision'])).item()
                },
                'recall': {
                    'mean': torch.mean(torch.tensor(metrics[t]['recall'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['recall'])).item()
                },
                'acc': {
                    'mean': torch.mean(torch.tensor(metrics[t]['acc'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['acc'])).item()
                },
                'f1': {
                    'mean': torch.mean(torch.tensor(metrics[t]['f1'])).item(),
                    'std': torch.std(torch.tensor(metrics[t]['f1'])).item()
                }
            }

        # 保存结果到文本文件
        results = {
            'psnr_mean': torch.mean(torch.tensor(all_psnr)).item(),
            'psnr_std': torch.std(torch.tensor(all_psnr)).item(),
            'ssim_mean': torch.mean(torch.tensor(all_ssim)).item(),
            'ssim_std': torch.std(torch.tensor(all_ssim)).item(),
            'threshold_metrics': threshold_metrics_summary
        }

        save_results_to_txt(results, f"{config.eval_save}/results.txt")

if __name__ == "__main__":
    evaluate()