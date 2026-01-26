"""
模型加载模块 - 负责加载和初始化模型
"""

import torch
import torch_npu
import logging
from config import MODEL_PATH, MODEL_FP16_PATH, DEVICE_TYPE, USE_FP16

logger = logging.getLogger(__name__)

def load_model():
    """
    加载TorchScript模型并迁移到指定设备
    
    返回:
        model: 加载并迁移到设备的模型
        device: 使用的设备
        is_fp16: 是否使用FP16精度
    """
    # 配置NPU设备
    device = torch.device(DEVICE_TYPE)
    logger.info(f"使用设备: {device}")
    
    # 根据USE_FP16选择模型路径
    model_path = MODEL_FP16_PATH if USE_FP16 else MODEL_PATH
    precision_str = "FP16" if USE_FP16 else "FP32"
    logger.info(f"使用精度: {precision_str}")
    
    # 加载 TorchScript 模型到CPU
    try:
        traced_script_module = torch.jit.load(model_path, map_location='cpu')
        traced_script_module.eval()
        logger.info(f"模型加载成功: {model_path}")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        exit(1)
    
    # 将模型迁移到NPU设备
    try:
        traced_script_module = traced_script_module.to(device)
        
        # 如果使用FP16，转换模型精度
        if USE_FP16:
            traced_script_module = traced_script_module.half()
            logger.info("模型已转换为FP16精度")
        
        logger.info("模型已迁移至NPU设备")
    except Exception as e:
        logger.error(f"模型迁移失败: {e}")
        exit(1)
    
    # 冻结并禁用优化
    # traced_script_module = torch.jit.freeze(traced_script_module)

    return traced_script_module, device