# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BSRN_arch import BSConvU, ESDB
from .torch_wavelets import DWT_2D, IDWT_2D
from ptflops import get_model_complexity_info


class CloudRobustNonLocal(nn.Module):
    """云鲁棒非局部模块：专门处理大面积缺失区域（支持任意T的mask融合）"""

    def __init__(self, in_channels, region_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.region_size = region_size  # 局部区块大小

        # 区域特征提取器
        self.region_pool = nn.AdaptiveAvgPool2d(region_size)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU()
        )

        # 特征融合
        self.fuse = nn.Conv2d(in_channels, in_channels, 1)

        # 通道数恢复层
        self.channel_adjust = nn.Conv2d(in_channels // 8, in_channels, 1)

        # 自适应池化层，确保云掩码与特征图尺寸匹配
        self.mask_pool = nn.AdaptiveAvgPool2d(region_size)

    def forward(self, x, cloud_mask_list=None):
        """
        输入：
            x: [B, C, H, W]
            cloud_mask_list:
                - [B, T, H, W]（推荐）
                - 或 [B, 1, H, W]
                - 或 [B, T, 1, H, W]
        输出：
            out: [B, C, H, W]
            cloud_mask: [B, 1, H, W] 或 None
        """
        B, C, H, W = x.shape

        # 将多时相注意力图融合为单通道（不再写死 T=3）
        cloud_mask = None
        if cloud_mask_list is not None:
            if cloud_mask_list.dim() == 4:
                # [B,T,H,W] 或 [B,1,H,W]
                if cloud_mask_list.shape[1] == 1:
                    cloud_mask = cloud_mask_list
                else:
                    cloud_mask = cloud_mask_list.mean(dim=1, keepdim=True)  # -> [B,1,H,W]
            elif cloud_mask_list.dim() == 5:
                # [B,T,1,H,W]
                cloud_mask = cloud_mask_list.mean(dim=1)  # -> [B,1,H,W]
            else:
                raise ValueError(f"cloud_mask_list shape not supported: {cloud_mask_list.shape}")

        # 第一步：提取区域特征
        region_features = self.region_pool(x)                   # [B, C, S, S]
        region_features = self.feature_extract(region_features) # [B, C', S, S]

        # 第二步：区域间非局部交互
        B2, C2, S, _ = region_features.shape
        region_vec = region_features.view(B2, C2, S * S).permute(0, 2, 1)  # [B, S*S, C']

        # 区域相似度
        sim = torch.bmm(region_vec, region_vec.transpose(1, 2))  # [B, S*S, S*S]

        # 如果有云掩码，抑制云区域权重
        if cloud_mask is not None:
            region_mask = self.mask_pool(cloud_mask)   # [B,1,S,S]
            region_mask = region_mask.view(B2, S * S)  # [B,S*S]
            sim_mask = torch.bmm(region_mask.unsqueeze(-1), region_mask.unsqueeze(1))  # [B,S*S,S*S]
            sim = sim * (1 - sim_mask)

        attn = F.softmax(sim, dim=-1)

        # 特征传播
        y_region = torch.bmm(attn, region_vec)  # [B, S*S, C']
        y_region = y_region.permute(0, 2, 1).view(B2, C2, S, S)  # [B, C', S, S]

        # 上采样到原始分辨率
        y_map = F.interpolate(y_region, size=(H, W), mode='bilinear', align_corners=True)
        y_map = self.channel_adjust(y_map)

        return self.fuse(x + y_map), cloud_mask


class WaveDownampler(nn.Module):
    """小波下采样模块：使用Haar小波分解，处理高低频信息"""

    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT_2D(wave='haar')  # 小波分解
        # 卷积处理水平/垂直方向的高频分量
        self.conv_lh = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_hl = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        # 注意力机制生成
        self.to_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0),
            nn.Sigmoid()
        )
        # 特征融合
        self.pw = nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0)

    def forward(self, x):
        """
        输入:
            x: [B, C, H, W] 特征图
        输出:
            o: [B, C*2, H/2, W/2] 压缩后的低频特征
            hi_bands: [B, C*3, H/2, W/2] 三组高频分量
        """
        x = self.dwt(x)  # [B, C*4, H/2, W/2]
        x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)

        lh = self.conv_lh(x_ll + x_lh)
        hl = self.conv_hl(x_ll + x_hl)

        att_map = self.to_att(lh + hl)

        x_s = self.pw(x)
        o = torch.mul(x_s, att_map) + x_s

        hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        return o, hi_bands


class FrequencyAwareCloudAttention(nn.Module):
    """增强型云感知注意力：自适应融合多源信息并抑制地物变化"""

    def __init__(self, in_channels, color_dim=3, reduction_ratio=8, dilations=[1, 2, 4]):
        super().__init__()
        self.in_channels = in_channels

        self.color_adapter = nn.Sequential(
            nn.Conv2d(color_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1)
        )

        self.multi_scale_conv = nn.ModuleList()
        for d in dilations:
            self.multi_scale_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 4, 3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True)
                )
            )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 4 * len(dilations) + 8, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.cloud_learner = nn.Sequential(
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.color_guided_att = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.attention_fusion_gate = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.attention_suppression = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.res_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, hi_feature, color_feature):
        identity = hi_feature

        color_embed = self.color_adapter(color_feature)  # [B, 8, H, W]

        scale_features = []
        for conv in self.multi_scale_conv:
            scale_features.append(conv(hi_feature))
        freq_features = torch.cat(scale_features, dim=1)

        fused = self.fusion(torch.cat([freq_features, color_embed], dim=1))

        cloud_prob = self.cloud_learner(fused)
        color_att = self.color_guided_att(color_embed)
        att_maps = torch.cat([cloud_prob, color_att], dim=1)
        fusion_gate = self.attention_fusion_gate(att_maps)

        att_map = (1 - fusion_gate) * color_att + fusion_gate * cloud_prob

        suppressed_att = self.attention_suppression(att_map)

        weighted_hi = hi_feature * (1 - suppressed_att)
        weighted_x = weighted_hi + self.res_conv(identity)
        return weighted_x, suppressed_att


class EnhancedFMBlock(nn.Module):
    def __init__(self, dim, kernel_size, conv_ratio=1.25, cg=16):
        super().__init__()
        hidden_dim = int(dim * conv_ratio)
        group = max(1, int(hidden_dim / cg))

        self.spatial_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )

        self.channel_path = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GroupNorm(group, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=group),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        x = self.spatial_path(x) + x
        x = self.channel_path(x) + x
        return x


class GatedFusion(nn.Module):
    """使用门控机制控制特征融合比例"""

    def __init__(self, in_ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x_pix, x_idwt):
        x = torch.cat([x_pix, x_idwt], dim=1)
        gate = self.gate_conv(x)
        gate_pix, gate_idwt = gate.chunk(2, dim=1)
        return gate_pix * x_pix + gate_idwt * x_idwt


class FMBConv(nn.Module):
    """轻量卷积块：ShuffleMixer风格的卷积模块"""

    def __init__(self, dim, conv_ratio=1.5, cg=16):
        super().__init__()
        hidden_dim = int(dim * conv_ratio)
        group = int(hidden_dim / cg)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=group),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.conv(x) + x


class WaveUpsampler(nn.Module):
    """小波上采样模块：结合像素洗牌与小波重建的上采样"""

    def __init__(self, pix_ch):
        super().__init__()
        self.idwt = IDWT_2D(wave='haar')
        self.upsampling = nn.Sequential(
            nn.Conv2d(pix_ch, pix_ch * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.interact = GatedFusion(pix_ch)
        self.fuse_conv = FMBConv(dim=pix_ch)

    def forward(self, x, hi_bands):
        x_1, x_2 = x.chunk(2, dim=1)

        pix_up = self.upsampling(x_1)
        o_idwt = self.idwt(torch.cat([x_2, hi_bands], dim=1))

        o = self.interact(pix_up, o_idwt)
        return self.fuse_conv(o)


class WaveBottleNeck(nn.Module):
    """小波瓶颈块：处理低频分量的高效结构"""

    def __init__(self, in_ch=64, n_lo_block=2):
        super().__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.lo_blocks = nn.Sequential(
            *[EnhancedFMBlock(dim=in_ch, kernel_size=7, conv_ratio=1.5) for _ in range(n_lo_block)]
        )
        self.esdb = ESDB(in_channels=in_ch, out_channels=in_ch, conv=BSConvU)

    def forward(self, x):
        x_w = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x_w.chunk(4, dim=1)
        x_ll = self.lo_blocks(x_ll)
        x_out = self.idwt(torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1))
        return self.esdb(x_out) + x


class Mix(nn.Module):
    """自适应特征融合：可学习的加权融合模块"""

    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        return fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))


class FourierConv(nn.Module):
    """傅里叶卷积模块：处理薄云干扰的频域操作"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(channels // reduction, 4)

        self.att = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        att_weights = self.att(self.avg_pool(mag).view(B, C))
        mag_att = mag * att_weights.view(B, C, 1, 1)

        real = mag_att * torch.cos(phase)
        imag = mag_att * torch.sin(phase)
        x_fft = torch.complex(real, imag)

        x_inv = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')

        return x + self.conv(x_inv)


class PerTimeEncoder(nn.Module):
    """单时相编码器：独立处理每个时间点的输入"""

    def __init__(self, input_nc, ngf):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=7, padding=0),
            nn.SiLU(inplace=True)
        )
        self.fourier1 = FourierConv(ngf)

        self.down2 = nn.Sequential(
            WaveBottleNeck(in_ch=ngf),
            WaveDownampler(ngf)
        )
        self.fourier2 = FourierConv(ngf * 2)

        self.down3 = nn.Sequential(
            WaveBottleNeck(in_ch=ngf * 2),
            WaveBottleNeck(in_ch=ngf * 2),
            WaveDownampler(ngf * 2)
        )
        self.fourier3 = FourierConv(ngf * 4)

    def forward(self, x):
        x = F.relu(self.conv(x))
        d1 = self.down1(x)
        d1 = self.fourier1(d1)

        d2, hi_bands_layer1 = self.down2(d1)   # [B,ngf*2,H/2,W/2], [B,ngf*3,H/2,W/2]
        d2 = self.fourier2(d2)

        d3, hi_bands_layer2 = self.down3(d2)   # [B,ngf*4,H/4,W/4], [B,ngf*6,H/4,W/4]
        d3 = self.fourier3(d3)

        return d3, hi_bands_layer1, hi_bands_layer2


class MaskGuidedTemporalFusion(nn.Module):
    """
    统一的掩码引导时序融合模块：
    1) 逐时相 mask 抑制
    2) 时间维聚合（sum）
    3) 局部增强
    4) 通道注意力重标定

    可同时用于：
    - 深层特征融合
    - H/2 高频融合
    - H/4 高频融合
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)

        self.enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_features, cloud_mask=None):
        """
        temporal_features: [B, T, C, H, W]
        cloud_mask:
            [B, T, H, W] 或 [B, T, 1, H, W] 或 None

        return:
            fused: [B, C, H, W]
        """
        if temporal_features.dim() != 5:
            raise ValueError(f"temporal_features must be [B,T,C,H,W], got {temporal_features.shape}")

        if cloud_mask is not None:
            if cloud_mask.dim() == 4:
                cloud_mask = cloud_mask.unsqueeze(2)  # [B,T,1,H,W]
            elif cloud_mask.dim() == 5:
                pass
            else:
                raise ValueError(f"cloud_mask shape not supported: {cloud_mask.shape}")

            temporal_features = temporal_features * (1.0 - cloud_mask)

        fused_sum = temporal_features.sum(dim=1)  # [B,C,H,W]

        enhanced = self.enhancer(fused_sum)
        att = self.channel_att(enhanced)
        fused = enhanced * att

        return fused


class WaveDH(nn.Module):
    """多时相小波融合网络（方案B：三条融合分支统一为 mask-guided temporal fusion）"""

    def __init__(self, input_nc=3, output_nc=3, ngf=32, n_lo_b=2, n_bottles=3):
        super(WaveDH, self).__init__()
        self.ngf = ngf

        self.per_time_encoder = PerTimeEncoder(input_nc, ngf)

        # ==============================
        # 三条分支统一使用相同风格的融合模块
        # ==============================
        self.deep_fusion = MaskGuidedTemporalFusion(
            in_channels=ngf * 4,
            reduction=16
        )

        self.hi_hr_fusion = MaskGuidedTemporalFusion(
            in_channels=ngf * 3,
            reduction=8
        )

        self.hi_lr_fusion = MaskGuidedTemporalFusion(
            in_channels=ngf * 6,
            reduction=8
        )

        self.bottleneck = nn.ModuleList([
            nn.Sequential(
                WaveBottleNeck(in_ch=ngf * 4, n_lo_block=n_lo_b),
                WaveBottleNeck(in_ch=ngf * 4, n_lo_block=n_lo_b)
            ) for _ in range(n_bottles)
        ])

        self.non_local_pyramid = nn.ModuleList([
            CloudRobustNonLocal(ngf * 4, region_size=16),  # H/4
            CloudRobustNonLocal(ngf * 2, region_size=8),   # H/2
            CloudRobustNonLocal(ngf, region_size=4)        # H（当前未使用，保留）
        ])

        self.up1 = WaveUpsampler(pix_ch=ngf * 2)
        self.up2 = WaveUpsampler(pix_ch=ngf)
        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        )

        self.tanh = nn.Tanh()
        self.mix1 = Mix(m=-1)

        # 只在 H/2（hi_layer1）预测云注意力
        self.hi_layer1_att = FrequencyAwareCloudAttention(ngf * 3)

    def forward(self, input):
        """
        输入:
            input: [B, T, C, H, W]
        输出:
            out: [B, output_nc, H, W]
            cloud_att_layer1_list: list of masks (每时相H/2) + nonlocal mask
            cloud_att_layer2_list: list of masks (每时相H/4) + nonlocal mask
        """
        B, T, C, H, W = input.shape

        all_d3 = []
        all_hi_layer1 = []
        all_hi_layer2 = []
        color_features_layer1 = []  # rgb@H/2

        for t in range(T):
            time_slice = input[:, t]  # [B, C, H, W]
            rgb = time_slice[:, :3]

            # RGB 使用 H/2（一次降采样）
            color_layer1 = F.interpolate(
                rgb, size=(H // 2, W // 2),
                mode='bilinear', align_corners=False
            )

            d3, hi_layer1, hi_layer2 = self.per_time_encoder(time_slice)

            all_d3.append(d3)
            all_hi_layer1.append(hi_layer1)
            all_hi_layer2.append(hi_layer2)
            color_features_layer1.append(color_layer1)

        # =========================================================
        # 先在 H/2 的高频上预测每时相云注意力，再下采样到 H/4
        # 这里按方案B：高频融合使用 raw high-frequency + mask
        # =========================================================
        cloud_att_layer1_list = []  # 每时相 [B,1,H/2,W/2]
        cloud_att_layer2_list = []  # 每时相 [B,1,H/4,W/4]

        for hi1, rgb1 in zip(all_hi_layer1, color_features_layer1):
            _, cloud_att1 = self.hi_layer1_att(hi1, rgb1)   # 这里只取 mask
            cloud_att_layer1_list.append(cloud_att1)

            # H/4 mask：由 H/2 mask 平均池化得到
            cloud_att2 = F.avg_pool2d(cloud_att1, kernel_size=2, stride=2)  # [B,1,H/4,W/4]
            cloud_att_layer2_list.append(cloud_att2)

        # 堆叠成张量供 temporal fusion / non-local 使用
        cloud_att_layer1 = torch.stack(cloud_att_layer1_list, dim=1).squeeze(2)  # [B,T,H/2,W/2]
        cloud_att_layer2 = torch.stack(cloud_att_layer2_list, dim=1).squeeze(2)  # [B,T,H/4,W/4]

        # =========================================================
        # 统一的 mask-guided temporal fusion
        # =========================================================

        # H/2 高频融合：raw hi_layer1 + cloud_att_layer1
        fused_hi_hr = self.hi_hr_fusion(
            torch.stack(all_hi_layer1, dim=1),   # [B,T,ngf*3,H/2,W/2]
            cloud_att_layer1                     # [B,T,H/2,W/2]
        )

        # H/4 高频融合：raw hi_layer2 + cloud_att_layer2
        fused_hi_lr = self.hi_lr_fusion(
            torch.stack(all_hi_layer2, dim=1),   # [B,T,ngf*6,H/4,W/4]
            cloud_att_layer2                     # [B,T,H/4,W/4]
        )

        # H/4 深层特征融合：raw d3 + cloud_att_layer2
        fused_d3 = self.deep_fusion(
            torch.stack(all_d3, dim=1),          # [B,T,ngf*4,H/4,W/4]
            cloud_att_layer2                     # [B,T,H/4,W/4]
        )

        # =========================================================
        # 瓶颈处理
        # =========================================================
        x = fused_d3
        for layer in self.bottleneck:
            x = layer(x)

        x_out_mix = self.mix1(fused_d3, x)

        # H/4 non-local（用 cloud_att_layer2）
        x_out_mix, cloud_attn_low = self.non_local_pyramid[0](x_out_mix, cloud_mask_list=cloud_att_layer2)

        # =========================================================
        # 上采样重建：H/2 non-local（用 cloud_att_layer1）
        # =========================================================
        x_up1 = self.up1(x_out_mix, fused_hi_lr)  # [B,ngf*2,H/2,W/2]
        x_up1, cloud_attn_mid = self.non_local_pyramid[1](x_up1, cloud_mask_list=cloud_att_layer1)

        x_up2 = self.up2(x_up1, fused_hi_hr)      # [B,ngf,H,W]
        out = self.up3(x_up2)

        # 追加 non-local 输出的融合后 mask（保持原来的可视化习惯）
        cloud_att_layer1_list.append(cloud_attn_mid)   # [B,1,H/2,W/2]
        cloud_att_layer2_list.append(cloud_attn_low)   # [B,1,H/4,W/4]

        return self.tanh(out), cloud_att_layer1_list, cloud_att_layer2_list


def main():
    """测试网络结构和计算复杂度"""
    model = WaveDH(input_nc=3, output_nc=3, ngf=32, n_lo_b=2, n_bottles=3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 输入尺寸: [批次, 时相数, 通道数, 高, 宽]
    input_shape = (3, 3, 256, 256)  # (T, C, H, W)
    dummy_input = torch.randn(1, *input_shape).to(device)

    # 前向传播测试
    with torch.no_grad():
        output, _, _ = model(dummy_input)
        print(f"输出尺寸: {output.shape}")  # [1, 3, 256, 256]

    # 计算模型复杂度（ptflops 对 5D 输入可能不兼容，失败会被捕获）
    print("\n模型复杂度分析:")
    try:
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True
        )
        print("计算量 (MACs): ", macs)
        print("参数量 (Parameters): ", params)
    except Exception as e:
        print(f"复杂度计算失败: {e}")


if __name__ == "__main__":
    main()