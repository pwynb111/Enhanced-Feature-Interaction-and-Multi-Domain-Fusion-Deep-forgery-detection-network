import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class CFIE(nn.Module):
    """Cross Feature Interaction Enhancement"""
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        
        # 特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel, 1),
            nn.Sigmoid()
        )
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        
        # 1. 特征相似度计算
        cos_sim = F.cosine_similarity(fa, fb, dim=1)  # B,H,W
        cos_sim = cos_sim.unsqueeze(1)  # B,1,H,W
        
        # 2. 特征交互
        fa_trans = self.transform(fa)
        fb_trans = self.transform(fb)
        
        # 3. 相似度引导的特征增强
        fa_enhanced = fa + fb_trans * cos_sim
        fb_enhanced = fb + fa_trans * cos_sim
        
        # 4. 通道注意力
        channel_weight = self.channel_att(fa_enhanced)
        
        # 5. 最终输出
        out_a = fa_enhanced * channel_weight
        out_b = fb_enhanced * channel_weight
        
        # 6. 残差连接
        out_a = self.gamma * out_a + fa
        out_b = self.gamma * out_b + fb
        
        return out_a, out_b
    


class EMFF(nn.Module):
    """Enhanced Multi-scale Feature Fusion Module"""
    def __init__(self, in_channels, size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, fa, fb):
        b1, c1, h1, w1 = fa.size()
        b2, c2, h2, w2 = fb.size()
        
        assert b1 == b2 and c1 == c2 and self.size == h2
        
        # 1. 特征对齐
        padding = abs(h1 % self.size - self.size) % self.size
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, 
                                         padding // 2, (padding + 1) // 2)).to(fa.device)
        fa = pad(fa)
        _, _, h1, w1 = fa.size()
        window = h1 // self.size
        
        # 2. 特征变换
        unfold = nn.Unfold(kernel_size=window, stride=window)
        fa_unfold = unfold(fa)
        
        # 3. 特征重组
        L = (h1 // window) ** 2
        fa_patches = fa_unfold.view(b1, c1, window*window, L)
        
        fa_patches = fa_patches.permute(0, 3, 1, 2)
        
        # 4. 特征增强
        enhanced_patches = []
        for i in range(L):
            patch = fa_patches[:, i].view(b1, c1, window, window)
            enhanced = self.enhance(patch)
            enhanced_patches.append(enhanced.view(b1, c1, -1))
            
        enhanced = torch.stack(enhanced_patches, dim=1)
        
        enhanced = enhanced.permute(0, 2, 3, 1)
        
        # 5. 特征重组为原始尺寸
        fold = nn.Fold(output_size=(h1, w1), kernel_size=window, stride=window)
        output = fold(enhanced.view(b1, c1*window*window, L))

        # 6. 调整到目标尺寸
        if output.size(-1) != self.size:
            output = F.adaptive_avg_pool2d(output, (self.size, self.size))

        
        # 7. 残差连接
        output = self.gamma * output + self.beta * fb
        return output

class EFG(nn.Module):
    """ Efficient Feature Guide Module"""
    def __init__(self, in_channel, num_heads=2, reduction_ratio=8):
        super().__init__()
        self.in_channel = in_channel
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads
        
        # 多头注意力
        self.query = nn.Conv2d(in_channel, in_channel, 1)
        self.key = nn.Conv2d(in_channel, in_channel, 1)
        self.value = nn.Conv2d(in_channel, in_channel, 1)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction_ratio, 1),
            nn.LayerNorm([in_channel // reduction_ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction_ratio, in_channel, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 1)
        )
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        
        # 1. 多头自注意力
        q = self.query(fb).view(B, self.num_heads, self.head_dim, -1)
        k = self.key(fb).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(fa).view(B, self.num_heads, self.head_dim, -1)
        
        # 计算注意力
        scale = float(self.head_dim) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 特征重建
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        # 2. 通道注意力
        channel_weight = self.channel_att(out)
        out = out * channel_weight
        
        # 3. 空间注意力
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        out = out * spatial
        
        # 4. 特征增强
        enhanced = self.feature_enhance(out)
        
        # 5. 残差连接
        output = self.gamma * enhanced + self.beta * fa
        
        return output





