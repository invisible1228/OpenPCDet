import torch
import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# 创建一个完整的mock dataset类
class MockPointFeatureEncoder:
    def __init__(self, num_point_features=4):
        self.num_point_features = num_point_features

class MockDataset:
    def __init__(self, class_names, num_point_features=4):
        self.class_names = class_names
        self.point_feature_encoder = MockPointFeatureEncoder(num_point_features)
        
        # 从配置文件中获取参数，或使用默认值
        self.point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.voxel_size = [0.16, 0.16, 4]
        
        # 计算grid_size - 使用numpy array
        self.grid_size = np.array([
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        ])
        
        # 其他必需的属性
        self.depth_downsample_factor = 1
        self.training = True
        self.logger = None

# 加载配置
cfg_from_yaml_file('/det/OpenPCDet/tools/cfgs/kitti_models/pillarfocus_pop_rcnn.yaml', cfg)

# 创建模拟数据 - 确保在正确的设备上
def create_mock_batch():
    batch_size = 1
    max_voxels = 100
    max_points = 32
    num_features = 4
    
    # 创建更合理的体素坐标
    grid_x, grid_y, grid_z = 432, 496, 1  # 基于我们的point cloud range和voxel size
    
    # 模拟voxel数据 - 确保在正确设备上
    voxels = torch.randn(max_voxels, max_points, num_features, device=device)
    voxel_coords = torch.zeros(max_voxels, 4, dtype=torch.long, device=device)
    voxel_coords[:, 0] = 0  # batch_idx
    voxel_coords[:, 1] = torch.randint(0, grid_z, (max_voxels,), device=device)  # z
    voxel_coords[:, 2] = torch.randint(0, grid_y, (max_voxels,), device=device)  # y  
    voxel_coords[:, 3] = torch.randint(0, grid_x, (max_voxels,), device=device)  # x
    voxel_num_points = torch.randint(1, max_points, (max_voxels,), device=device)
    
    batch_dict = {
        'voxels': voxels,
        'voxel_coords': voxel_coords, 
        'voxel_num_points': voxel_num_points,
        'batch_size': batch_size
    }
    
    return batch_dict

# 创建mock dataset
point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[-1].VOXEL_SIZE

mock_dataset = MockDataset(cfg.CLASS_NAMES, num_point_features=4)
mock_dataset.point_cloud_range = point_cloud_range
mock_dataset.voxel_size = voxel_size

mock_dataset.grid_size = np.array([
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
])

print(f"📊 Mock Dataset Info:")
print(f"   Class names: {mock_dataset.class_names}")
print(f"   Point cloud range: {mock_dataset.point_cloud_range}")
print(f"   Voxel size: {mock_dataset.voxel_size}")
print(f"   Grid size: {mock_dataset.grid_size}")

# 构建模型并移到正确设备
print("\n🏗️  Building model...")
try:
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=mock_dataset)
    model = model.to(device)  # 确保模型在正确设备上
    print("✅ Model built successfully!")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 打印模型结构
    print(f"📋 Model structure:")
    for name, module in model.named_children():
        if module is not None:
            print(f"   - {name}: {type(module).__name__}")
        else:
            print(f"   - {name}: None")
            
except Exception as e:
    print(f"❌ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试前向传播
print("\n🚀 Testing forward pass...")
model.eval()
with torch.no_grad():
    batch_dict = create_mock_batch()
    print(f"📊 Input batch info:")
    print(f"   Voxels shape: {batch_dict['voxels'].shape}, device: {batch_dict['voxels'].device}")
    print(f"   Voxel coords shape: {batch_dict['voxel_coords'].shape}, device: {batch_dict['voxel_coords'].device}")
    print(f"   Voxel num points shape: {batch_dict['voxel_num_points'].shape}, device: {batch_dict['voxel_num_points'].device}")
    
    try:
        result = model(batch_dict)
        print("✅ Forward pass successful!")
        
        if isinstance(result, tuple):
            pred_dicts, recall_dicts = result
            print(f"📈 Prediction type: {type(pred_dicts)}")
            if isinstance(pred_dicts, list) and len(pred_dicts) > 0:
                print(f"📈 Prediction keys: {list(pred_dicts[0].keys())}")
                for key, value in pred_dicts[0].items():
                    if isinstance(value, torch.Tensor):
                        print(f"   - {key}: {value.shape}")
                    else:
                        print(f"   - {key}: {type(value)}")
        else:
            print(f"📈 Output type: {type(result)}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        
print("\n🎉 Test completed!")