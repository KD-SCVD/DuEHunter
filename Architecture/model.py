import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F

class FirstStageNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gnn = GCNConv(input_dim, input_dim * 2)  # GNN 层（默认处理 出度→入度）
        self.relu = nn.ReLU()
        # 单图聚合（全局平均池化）
        self.readout = lambda x: torch.sum(x, dim=0, keepdim=True)

    def forward(self, x, edge_index):
        # edge_index 格式为 [[出度], [入度]]，直接传入 GNN
        edge_index = edge_index.t()  # 关键转换
        x = self.gnn(x, edge_index)
        x = self.relu(x)
        # x = self.readout(x)  # 输出形状 [1, input_dim * 2]
        return x


# 第二阶段网络（不变）
class SecondStageNetwork(nn.Module):
    def __init__(self,input_dim,  input_channels = 1, hidden_channels = 1):
        super().__init__()
        self.readout = lambda x: torch.sum(x, dim=0, keepdim=True)
        self.high_confidence_model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.confront_model = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2,input_dim * 2),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size, _,= x.shape

        #====== 高置信特征挖掘==========
        high_confidence = x.view(batch_size, 1, 32, 16)
        high_confidence = self.high_confidence_model(high_confidence)
        high_confidence = high_confidence.view(batch_size, -1)  # 展平
        high_confidence= self.softmax(high_confidence)

        # =====前置特征==============
        front_feature = self.confront_model(x)

        # ========特征补全 ===========
        complete_feature = torch.cat((high_confidence, front_feature), dim=1)

        return complete_feature, high_confidence, front_feature

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim = 2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim*4, input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim*2, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim,input_dim//2 ),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim//2, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        pre = self.predictor(x)
        return pre

class Student_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.first_stage = FirstStageNetwork(input_dim)
        self.second_stage = SecondStageNetwork(input_dim)
        self.predictor = Predictor(input_dim)
        self.readout = lambda x: torch.sum(x, dim=0, keepdim=True)

    def forward(self, x, edge):
        first_feature = self.first_stage(x, edge)
        first_feature = self.readout(first_feature)
        complete_feature, _, _ = self.second_stage(first_feature)
        pre = self.predictor(complete_feature)
        return pre


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, concatenated_features):
        """
        输入: concatenated_features [batch_size, num_features, feature_dim]
        输出: 加权融合后的特征 [batch_size, feature_dim//4]
        """
        num_features, feature_dim = concatenated_features.shape

        # 使用自注意力机制为每个特征分配权重
        # query, key, value都来自拼接的特征
        attn_output, attn_weights = self.multihead_attn(
            concatenated_features,
            concatenated_features,
            concatenated_features
        )

        # 对注意力输出进行池化（可以根据需要选择mean/max/sum）
        fused = torch.mean(attn_output, dim=0).unsqueeze(0)  # [batch_size, feature_dim]

        # 输出变换
        out = self.fc_out(fused)
        return self.tanh(out), attn_weights  # 返回注意力权重用于可视化


class Denoise_teacher(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, input_dim, heads=num_heads)
        self.conv2 = GATConv(input_dim * num_heads, input_dim *2, heads=1)

    def forward(self, x, edge_index):
        # 输入: x - 节点特征矩阵 [num_nodes, num_features]
        #      edge_index - 边索引 [2, num_edges]

        x = F.elu(self.conv1(x, edge_index.t()))  # 输出: [num_nodes, hidden_dim * num_heads]
        x = self.conv2(x, edge_index.t())  # 输出: [num_nodes, num_classes]

        return x  # 每个节点的类别预测概率

class SpatialAnomalyDetector(nn.Module):
    """空间层面异常特征挖掘（使用局部池化）"""

    def __init__(self, input_dim, pool_size=4):
        super().__init__()
        # 将1D特征转换为2D特征图 (1x32x32)
        self.unflatten = nn.Unflatten(1, (1, 32, 16))

        # 局部池化参数
        self.pool_size = pool_size
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        # 全连接神经网络
        self.fc = nn.Sequential(
            nn.Linear(input_dim//4, input_dim // 2 ),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2 , input_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # 输入x形状: [batch, 1024]
        batch, _ = x.shape
        x = self.unflatten(x)  # [batch, 1, 32, 32]

        # 局部池化（输出尺寸计算：32/pool_size）
        avg_out = self.avg_pool(x)  # [batch, 1, 8, 8] (当pool_size=4时)
        max_out = self.max_pool(x)  # [batch, 1, 8, 8]

        # 拼接并融合
        combined = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, 4, 4]
        combined = combined.view(batch, 1,-1)
        spatial_feat = self.fc(combined)

        return spatial_feat


class ChannelAnomalyDetector(nn.Module):
    """通道层面异常特征挖掘（使用局部平均）"""

    def __init__(self, input_dim, window_size=4):
        super().__init__()
        # 局部通道平均（将512维分成若干窗口）
        self.window_size = window_size
        self.num_windows = input_dim // window_size
        self.input_dim = input_dim

        # 窗口平均池化
        self.local_avg = nn.Sequential(
            nn.Unflatten(1, (self.num_windows *2 , self.window_size)),
            nn.AvgPool1d(kernel_size=window_size, stride=window_size)  # [batch, num_windows, 1]
        )

        # 特征融合卷积
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 4, kernel_size=1)
        )

        # 输出调整层，确保输出维度与输入维度匹配
        self.output_adjust = nn.Linear(input_dim *2, input_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # 局部通道平均
        local_avg = self.local_avg(x)  # [batch, num_windows, 1]
        local_avg = local_avg.transpose(1, 2)  # [batch, 1, num_windows]

        # 卷积处理
        conv_output = self.conv(local_avg)  # [batch, 4, num_windows]

        # 展平并调整输出维度
        flattened = conv_output.view(batch_size, -1)  # [batch, 4 * num_windows]
        output = self.output_adjust(flattened)  # [batch, input_dim]

        return output
class Complementary_teacher(nn.Module):
    def __init__(self, input_dim, perturbation, beta=0.7):
        super().__init__()
        self.perturbation = perturbation
        self.spatial_detector = SpatialAnomalyDetector(input_dim)
        self.channel_detector = ChannelAnomalyDetector(input_dim)
        self.generator = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 2, input_dim *2),
            nn.Softmax(dim=1)
        )
        self.beta = nn.Parameter(torch.tensor(beta))  # 可学习的权重参数

    def forward(self, x):
        # 空间特征挖掘
        spatial_feat = self.spatial_detector(x)

        # 通道特征挖掘
        channel_feat = self.channel_detector(x)
        batch, feature_dim = channel_feat.size()
        # 加权融合 (β∈[0,1])
        spatial_feat = spatial_feat.view(batch, feature_dim)
        fused_feat = self.beta * spatial_feat + (1 - self.beta) * channel_feat
        if self.perturbation == "Conditional":
            complement_feat = self.generator(1 - fused_feat)
        elif self.perturbation == "Gauss":
            device = fused_feat.device
            gauss_noise = torch.randn_like(fused_feat, device=device)
            complement_feat = self.generator(fused_feat + gauss_noise)
        elif self.perturbation == "Uniform":
            device = fused_feat.device
            uniform_noise = torch.rand_like(fused_feat, device=device) * 2 - 1  # [-1, 1]
            complement_feat = self.generator(fused_feat + uniform_noise)
        elif self.perturbation == "SaltPepper":
            device = fused_feat.device
            salt_pepper_noise = torch.rand_like(fused_feat, device=device)
            # 添加椒盐噪声：10%的概率为0或1
            salt_mask = (salt_pepper_noise > 0.95).float()
            pepper_mask = (salt_pepper_noise < 0.05).float()
            noisy_feat = fused_feat * (1 - salt_mask - pepper_mask) + salt_mask
            complement_feat = self.generator(noisy_feat)
        elif self.perturbation == "Dropout":
            # 使用dropout作为噪声
            dropout_mask = torch.bernoulli(torch.ones_like(fused_feat) * 0.8)  # 保留80%的特征
            noisy_feat = fused_feat * dropout_mask
            complement_feat = self.generator(noisy_feat)
        elif self.perturbation == "Multiplicative":
            device = fused_feat.device
            mult_noise = torch.randn_like(fused_feat, device=device) * 0.3 + 1.0  # 均值1.0，标准差0.3
            complement_feat = self.generator(fused_feat * mult_noise)
        elif self.perturbation == "Masking":
            # 随机屏蔽部分特征通道
            batch_size, channels = fused_feat.shape[0], fused_feat.shape[1]
            device = fused_feat.device
            mask_ratio = 0.3  # 屏蔽30%的通道
            mask_channels = int(channels * mask_ratio)
            channel_mask = torch.ones(batch_size, channels, device=device)
            for i in range(batch_size):
                masked_indices = torch.randperm(channels, device=device)[:mask_channels]
                channel_mask[i, masked_indices] = 0
            # 扩展mask到所有空间维度
            spatial_dims = len(fused_feat.shape) - 2
            for _ in range(spatial_dims):
                channel_mask = channel_mask.unsqueeze(-1)
            noisy_feat = fused_feat * channel_mask
            complement_feat = self.generator(noisy_feat)
        elif self.perturbation == "Spectral":
            # 在频域添加噪声
            device = fused_feat.device
            # 对每个样本进行傅里叶变换
            feat_fft = torch.fft.fft2(fused_feat, dim=(-2, -1))
            # 添加频域噪声
            freq_noise = torch.randn_like(feat_fft, device=device) * 0.1
            noisy_fft = feat_fft + freq_noise
            # 逆变换回空间域
            noisy_feat = torch.fft.ifft2(noisy_fft, dim=(-2, -1)).real
            complement_feat = self.generator(noisy_feat)
        elif self.perturbation == "Adversarial":
            # 添加小的对抗性扰动
            device = fused_feat.device
            adv_noise = torch.sign(torch.randn_like(fused_feat, device=device)) * 0.01
            complement_feat = self.generator(fused_feat + adv_noise)
        else:
            complement_feat = self.generator(fused_feat)

        return complement_feat

class Model(nn.Module):
    def __init__(self, input_dim, perturbation):
        super().__init__()
        self.student = Student_Model(input_dim)
        self.denoise_model = Denoise_teacher(input_dim)
        self.complement_model = Complementary_teacher(input_dim, perturbation)