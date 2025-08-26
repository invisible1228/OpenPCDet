import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    """金字塔池化模块"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, in_channels//len(pool_sizes), 1),
                    nn.BatchNorm2d(in_channels//len(pool_sizes)),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        h, w = x.shape[2:]
        pyramid_feats = [x]
        
        for conv in self.convs:
            feat = conv(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_feats.append(feat)
            
        output = torch.cat(pyramid_feats, dim=1)
        output = self.fusion_conv(output)
        
        return output

class PPFE(nn.Module):
    """点金字塔特征增强模块"""
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        
        # 多尺度特征提取
        self.pyramid_pooling = PyramidPoolingModule(input_channels)
        
        # 特征增强模块
        self.enhancement = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*2, 3, padding=1),
            nn.BatchNorm2d(input_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels*2, input_channels, 1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, data_dict):
        """
        Args:
            data_dict: containing 'spatial_features_2d'
        Returns:
            data_dict: with enhanced features
        """
        spatial_features = data_dict['spatial_features_2d']
        
        # 金字塔池化增强
        enhanced_feat = self.pyramid_pooling(spatial_features)
        enhanced_feat = self.enhancement(enhanced_feat)
        
        data_dict['spatial_features_2d'] = enhanced_feat
        
        return data_dict