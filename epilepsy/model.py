
from generative.networks.nets import VQVAE
import torch
import torch.nn as nn
from config import TrainConfig
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from modules.unet import UNetModel
from generative.networks.nets import DiffusionModelUNet
class TextConditionEncoder(nn.Module):
    def __init__(self, latent_shape):  # 假设 H//64=4, W//64=4
        super().__init__()
        # 初始化CLIP模型
        # textmodelPath = "D:\Desktop\epilepsy\code\epilepsy\epilepsy\\textmodel"
        textmodelPath = "/mntcephfs/lab_data/wangcm/zh/Epilepsy/textmodel"
        self.clip_model = CLIPTextModel.from_pretrained(textmodelPath,local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(textmodelPath,local_files_only=True)
        # 计算目标维度
        self.latent_channels = latent_shape[0]
        self.spatial_size = (latent_shape[1], latent_shape[2])

        # 文本特征投影层
        self.projection = nn.Sequential(
            nn.Linear(512, 512),  # 保持输入维度为768
            nn.GELU(),
            nn.Linear(512, self.latent_channels * self.spatial_size[0] * self.spatial_size[1]),
            nn.Unflatten(1, (self.latent_channels, self.spatial_size[0], self.spatial_size[1]))
        )

    def forward(self, texts):
        # 文本编码
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.clip_model.device)
        text_emb = self.clip_model(**inputs).last_hidden_state.mean(dim=1)  # [B, 768]

        # 投影到目标空间
        spatial_cond = self.projection(text_emb)  # [B, 4, H//64, W//64]
        return spatial_cond

textConditionEncoder =  TextConditionEncoder(
            latent_shape=TrainConfig.latent_shape
        )



# myVQVAE = VQVAE(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     num_channels=(128, 256, 512),
#     num_res_channels=512,
#     num_res_layers=2,
#     downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
#     upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
#     num_embeddings=256,
#     embedding_dim=4,
#     output_act="tanh",
# )#1xHxW ->[B, 4, H/64, W/64]
myVQVAE = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256),  # 确保与下采样层数匹配
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=(
        (2, 4, 1, 1),  # 第一次下采样：256x256 → 128
        (2, 4, 1, 1)   # 第二次下采样：128 → 64
    ),
    upsample_parameters=(
        (2, 4, 1, 1, 0),  # 第一次上采样：64 → 64x64
        (2, 4, 1, 1, 0)   # 第二次上采样：64x64 → 256x256
    ),
    num_embeddings=256,
    embedding_dim=4,
    output_act="tanh",
)#[B, 4, H/16, W/16]
# diffusion.py

class CrossAttention(nn.Module):
    """交叉注意力模块，用于融合文本嵌入和图像特征"""

    def __init__(self, dim=128, max_seq_length=4):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1)  # 使用卷积生成查询
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1)  # 使用卷积生成键
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1)  # 使用卷积生成值
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1)  # 输出卷积

    def forward(self, x, text_cond):
        """
        参数说明：
        x: 图像特征 [B, C, H, W]
        text_cond: 文本嵌入 [B, 4, H, W]
        """
        # 计算查询、键、值
        q = self.to_q(x)  # [B, dim, H, W]
        k = self.to_k(text_cond)  # [B, dim, H, W]
        v = self.to_v(text_cond)  # [B, dim, H, W]

        # 计算注意力分数
        q = q.view(q.shape[0], q.shape[1], -1)  # [B, dim, H*W]
        k = k.view(k.shape[0], k.shape[1], -1)  # [B, dim, H*W]
        v = v.view(v.shape[0], v.shape[1], -1)  # [B, dim, H*W]

        # 计算注意力分数
        dots = torch.einsum('bcd,bce->bde', q * self.scale, k)  # [B, H*W, H*W]
        attn = dots.softmax(dim=-1)

        # 应用注意力权重
        out = torch.einsum('bde,bce->bcd', attn, v)  # [B, dim, H*W]
        out = out.view(x.shape[0], -1, x.shape[2], x.shape[3])  # [B, dim, H, W]

        # 通过输出卷积
        out = self.to_out(out)

        return out
class ConditionalUNet(nn.Module):
    """基于第三方库的条件UNet适配器"""

    def __init__(self,TrainConfig):
        super().__init__()
        # 计算总输入通道数（潜空间+条件图像+条件文本）
        in_channels = TrainConfig.latent_dim *2
        # 初始化第三方UNet
        self.unet = UNetModel(
            image_size=TrainConfig.dataSize/4,  # 根据潜空间实际尺寸调整
            in_channels=in_channels,
            out_channels=TrainConfig.latent_dim,  # 预测噪声维度需匹配潜空间
            model_channels=128,  # 基础通道数
            num_res_blocks=8,  # 残差块数量
            attention_resolutions=(16, 8),# 使用注意力的分辨率层级
            num_heads_upsample=-1,
            num_head_channels=-1,
            channel_mult=(1,2,4),  # 通道数倍增系数
            num_heads=8,  # 注意力头数
            use_scale_shift_norm=True,  # 使用scale-shift归一化
            resblock_updown=True,  # 在上下采样中使用残差块
        )
        self.cross_attention = CrossAttention(
            dim=TrainConfig.latent_dim,  # 与潜空间维度相同
            max_seq_length=4  # 文本条件的通道数
        )


    def forward(self, x, t, text_cond):
        """
        参数说明：
        x: 噪声潜变量 [B, C, H, W]
        t: 时间步 [B]
        cond: 拼接后的条件特征 [B, Cond_C, H, W]
        """
        # 拼接潜变量和条件特征
        pre_noise_imgCond = self.unet(x, t)
        pre_noise_textCond = self.cross_attention(pre_noise_imgCond, text_cond)
        return pre_noise_textCond


class LatentDiffusion(nn.Module):
    """潜空间扩散模型"""

    def __init__(self, TrainConfig, noise_scheduler, cond_drop_prob_img=0.1, cond_drop_prob_text=0.1):
        super().__init__()
        self.config = TrainConfig
        self.unet = ConditionalUNet(TrainConfig)
        self.betas = torch.linspace(1e-4, 0.02, TrainConfig.timesteps)
        self.alphas = 1 - self.betas
        self.noise_scheduler = noise_scheduler
        self.cond_drop_prob_img = cond_drop_prob_img
        self.cond_drop_prob_text = cond_drop_prob_text
    def encode_image(self, x):
        """编码图像到潜空间"""
        quantized, _ = self.vqvae.encode(x)
        return quantized

    def forward(self, z, cond_img, text_cond):
        """
        前向传播流程
        :param x: 原始输入图像 [B, 1, H, W]
        :param cond_img: 条件图像 [B, 1, H, W]
        :param texts: 文本描述列表 [B]
        """
        # 独立生成图像和文本的掩码
        mask_img = torch.rand(z.size(0), device=z.device) > self.cond_drop_prob_img
        mask_text = torch.rand(z.size(0), device=z.device) > self.cond_drop_prob_text

        cond_img = cond_img * mask_img[:, None, None, None]
        text_cond = text_cond * mask_text[:, None, None, None]

        device = z.device
        b = z.shape[0]
        t = torch.randint(0, self.config.timesteps, (b,), device=device).long()

        # 添加噪声
        noise = torch.randn_like(z).to(device)
        # alpha_t = self.alphas[t].view(-1, 1, 1, 1).to(device)
        # 使用 noise_scheduler 添加噪声（关键修改）
        z_noisy = self.noise_scheduler.add_noise(z, noise, t).to(device)

        # 3. 拼接图像
        combined_imgs = torch.cat([z_noisy, cond_img], dim=1)  # [B, 8, 64, 64]

        # 预测噪声
        pred_noise = self.unet(combined_imgs, t, text_cond)

        return F.mse_loss(pred_noise, noise)

    def sample(self, cond_img, text_cond, num_steps=None, guidance_scale_img=3, guidance_scale_text=5):
        self.eval()
        device = next(self.parameters()).device
        num_steps = num_steps or self.config.timesteps

        # 初始化潜变量噪声，与条件图像尺寸相同
        z_t = torch.randn_like(cond_img, device=device)

        # 准备四种条件组合
        batch_size = cond_img.size(0)
        cond_mask = [
            (True, True),  # 全条件
            (True, False),  # 仅图像条件
            (False, True),  # 仅文本条件
            (False, False)  # 无条件
        ]

        # 配置噪声调度器
        self.noise_scheduler.set_timesteps(num_steps, device=device)

        # 渐进式去噪循环
        for t in self.noise_scheduler.timesteps:
            # 为每个条件组合构建输入
            all_inputs = []
            all_texts = []

            for img_mask, text_mask in cond_mask:
                # 应用条件掩码
                masked_img = cond_img * img_mask
                masked_text = text_cond * text_mask

                # 拼接当前潜变量与条件图像
                combined_input = torch.cat([z_t, masked_img], dim=1)
                all_inputs.append(combined_input)
                all_texts.append(masked_text)

            # 合并所有条件输入
            model_input = torch.cat(all_inputs, dim=0)
            text_input = torch.cat(all_texts, dim=0)
            timesteps = torch.cat([t.repeat(batch_size)] * 4, dim=0)

            # 批量前向预测
            with torch.no_grad():
                noise_pred = self.unet(model_input, timesteps, text_input)

            # 分解预测结果
            pred_full, pred_img, pred_text, pred_uncond = noise_pred.chunk(4)

            # 应用双条件引导
            noise_pred = pred_uncond + \
                         guidance_scale_img * (pred_img - pred_uncond) + \
                         guidance_scale_text * (pred_text - pred_uncond) + \
                         (guidance_scale_img * guidance_scale_text) * (pred_full - pred_img - pred_text + pred_uncond)

            # 更新潜变量
            z_t = self.noise_scheduler.step(
                noise_pred[:batch_size],  # 取原始batch结果
                t,
                z_t
            ).prev_sample

        return z_t
if __name__ == "__main__":
    print("Number of model parameters:", sum([p.numel() for p in myVQVAE.parameters()]))

