"""
数据处理模块 - 负责数据预处理和数据集加载
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from config import (
    DATASET_VAL_DIR, BATCH_SIZE, RESIZE_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD
)

logger = logging.getLogger(__name__)

def get_data_transforms():
    """
    获取数据预处理转换
    
    返回:
        transforms.Compose: 数据预处理转换组合
    """
    preprocess = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return preprocess

def load_dataset():
    """
    加载imagenet-mini数据集的验证部分
    
    返回:
        dataset: 加载的数据集
        dataloader: 数据加载器
    """
    # 获取数据预处理转换
    preprocess = get_data_transforms()
    
    # 加载数据集
    try:
        dataset = datasets.ImageFolder(root=DATASET_VAL_DIR, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        logger.info(f"数据集加载成功: {len(dataset)} 张图片, {len(dataloader)} 个batch")
        return dataset, dataloader
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        exit(1)
