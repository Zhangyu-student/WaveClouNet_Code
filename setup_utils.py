import os
import torch
from dataset import Sen2_MTC_New_Multi, MultipleDataset, Landsat, S2PatchDataset
from models.WaveCloudNet import WaveDH
from torch.utils.data import DataLoader, random_split

def create_dataloaders(config):
    """根据配置创建训练和测试数据加载器"""
    if config['dataset_type'] == 'new_multi':
        dataset_class = Sen2_MTC_New_Multi
        # 创建数据集实例
        train_dataset = dataset_class(data_root=config['data_root'], mode="train")
        test_dataset = dataset_class(data_root=config['data_root'], mode="test")
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )

    elif config['dataset_type'] == 'old_multi':
        dataset_class = MultipleDataset
        total_data = dataset_class(
            data_root=os.path.join(config['data_root'], "multipleImage"),
            band=3,
        )
        print(len(total_data))
        train_data, val_data, test_data = random_split(
            dataset=total_data,
            lengths=(2504, 313, 313),
            generator=torch.Generator().manual_seed(2022),
        )

        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                                  num_workers= config.get('num_workers', 4), drop_last=False, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config.get('num_workers', 2), drop_last=False, pin_memory=True, persistent_workers=True)

    elif config['dataset_type'] == 'landsat':
        # 使用 Landsat 类，通过 mode 参数加载相应的 split
        dataset_class = Landsat

        # 1. 创建训练数据集实例 (mode='train')
        train_dataset = dataset_class(data_root=config['data_root'], mode="train")
        # 2. 创建测试数据集实例 (mode='test' 或 'val'，这里假设使用 'test')
        # 如果需要验证集，可以改为 mode="val"
        test_dataset = dataset_class(data_root=config['data_root'], mode="test")

        # 3. 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        # 4. 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,  # 测试/验证集不打乱
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )

    elif config['dataset_type'] == 's2asiawest':
        dataset_class = S2PatchDataset

        train_dataset = dataset_class(
            data_root=config['data_root'],
            mode="train",
            use_split_subdir=True,  # ✅ 关键
            scale=10000.0,  # 若你tif还是0-10000
            mean=None, std=None,  # 先不做标准化也能跑
            augment=True,
        )

        test_dataset = dataset_class(
            data_root=config['data_root'],
            mode="test",
            use_split_subdir=True,  # ✅ 关键
            scale=10000.0,
            mean=None, std=None,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )
    else:
        raise ValueError(f"无效的dataset_type: {config['dataset_type']}")


    return train_loader, test_loader


def create_model(config):
    """根据配置创建模型实例"""
    # 根据数据集类型确定输入通道数
    if config['dataset_type'] == 'new_multi':
        input_channels = 3
    elif config['dataset_type'] == 'old_multi':
        input_channels = 3
    elif config['dataset_type'] == 'landsat':
        input_channels = 7
    elif config['dataset_type'] == 's2asiawest':
        input_channels = 13
    else:
        raise ValueError(f"无效的dataset_type: {config['dataset_type']}")

    # 从配置获取模型参数或使用默认值
    model_params = config.get('model_params', {})

    # 创建模型实例
    model = WaveDH(
        input_nc=input_channels,
        output_nc=input_channels, # 默认输出通道与输入保持一致
        ngf=model_params.get('ngf', 32),
        n_lo_b=model_params.get('n_lo_b', 2),
        n_bottles=model_params.get('n_bottles', 3)
    ).to(config['device'])

    # 加载预训练权重（如果存在）
    pretrained_path = config.get('pretrained_path')
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location=config['device'])
            model.load_state_dict(state_dict)
            print(f"✅ 成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            print("⚠️ 将从头开始训练")
    else:
        print("ℹ️ 没有提供预训练权重路径或文件不存在，将从头开始训练")

    return model
