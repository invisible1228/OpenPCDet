import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                             for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        
        if self.use_norm:
            torch.backends.cudnn.enabled = False
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            torch.backends.cudnn.enabled = True
        x = F.relu(x)
        
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFocusVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, 
                 grid_size=None, depth_downsample_factor=None, **kwargs):
        super().__init__(model_cfg)
        
        # 保存必需的参数
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size if grid_size is not None else [1, 1, 1]
        self.depth_downsample_factor = depth_downsample_factor if depth_downsample_factor is not None else 1
        
        # VFE配置参数
        self.use_norm = self.model_cfg.get('USE_NORM', True)
        self.with_distance = self.model_cfg.get('WITH_DISTANCE', False)
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSLUTE_XYZ', True)
        
        # PillarFocus特定参数
        self.dynamic_pillar = self.model_cfg.get('DYNAMIC_PILLAR', True)
        self.min_points_threshold = self.model_cfg.get('MIN_POINTS_THRESHOLD', 10)
        self.expansion_ratio = self.model_cfg.get('EXPANSION_RATIO', 0.25)
        
        # 计算输入特征维度
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 保存voxel相关参数
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1] 
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # Dynamic pillar focus权重
        if self.dynamic_pillar:
            self.focus_weight = nn.Parameter(torch.ones(1))
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def dynamic_pillar_focus(self, voxel_features, voxel_num_points):
        """应用动态pillar focus机制"""
        if not self.dynamic_pillar:
            return voxel_features
            
        batch_size, max_points, feature_dim = voxel_features.shape
        
        # 计算每个pillar的重要性分数
        point_counts = voxel_num_points.float()
        importance_score = torch.where(
            point_counts >= self.min_points_threshold,
            torch.ones_like(point_counts),
            point_counts / self.min_points_threshold * self.expansion_ratio
        )
        
        # 应用focus权重
        focus_weights = importance_score.unsqueeze(-1).unsqueeze(-1) * self.focus_weight
        voxel_features = voxel_features * focus_weights
        
        return voxel_features

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        
        # 应用动态pillar focus
        voxel_features = self.dynamic_pillar_focus(voxel_features, voxel_num_points)
        
        # 计算点的均值作为pillar中心
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)  
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        batch_dict['pillar_features'] = features.squeeze()
        return batch_dict