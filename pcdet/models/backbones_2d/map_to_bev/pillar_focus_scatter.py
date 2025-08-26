import torch
import torch.nn as nn


class PillarFocusScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = grid_size
        
        # 从配置中获取BEV特征数量
        self.num_bev_features = self.model_cfg.get('NUM_BEV_FEATURES', 64)
        
        # Focus增强参数
        self.focus_enhancement = self.model_cfg.get('FOCUS_ENHANCEMENT', True)
        self.spatial_attention = self.model_cfg.get('SPATIAL_ATTENTION', True)
        
        if self.spatial_attention:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(self.num_bev_features, self.num_bev_features // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_bev_features // 4, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        
        for batch_idx in range(batch_size):
            # 创建BEV特征图 - 确保在相同设备上
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)  # 使用pillar_features的设备

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            
            # 获取当前batch的pillar特征
            if pillar_features.dim() == 2:
                pillars = pillar_features[batch_mask, :]
            else:
                pillars = pillar_features[batch_mask, :].squeeze()
                
            if len(pillars.shape) == 1:
                pillars = pillars.unsqueeze(0)
            
            # 确保索引在有效范围内
            valid_indices = indices < spatial_feature.shape[1]
            if valid_indices.any():
                spatial_feature[:, indices[valid_indices]] = pillars[valid_indices].t()
            
            # 重塑为BEV格式
            batch_spatial_features.append(spatial_feature.view(self.num_bev_features, self.ny, self.nx))

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        
        # 应用Focus增强
        if self.focus_enhancement and self.spatial_attention:
            attention_weights = self.spatial_conv(batch_spatial_features)
            batch_spatial_features = batch_spatial_features * attention_weights
        
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict