"""
ResNet ONNX 推理主程序 - 使用 ONNX 模型在 NPU 上进行推理
"""

import logging
import numpy as np
import onnxruntime as ort
import time
from tqdm import tqdm
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, ONNX_MODEL_PATH,
    DATASET_VAL_DIR, BATCH_SIZE, RESIZE_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, WARMUP_BATCH_COUNT
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    # 获取数据预处理转换
    preprocess = get_data_transforms()
    
    # 加载数据集
    try:
        dataset = datasets.ImageFolder(root=DATASET_VAL_DIR, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        return dataset, dataloader
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        exit(1)

def load_onnx_model():
    """
    加载 ONNX 模型并配置 NPU 执行提供者
    
    返回:
        session: ONNX Runtime 会话
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    # 配置 ONNX Runtime 选项
    options = ort.SessionOptions()
    
    # 创建 ONNX Runtime 会话，使用 CANN 执行提供者进行 NPU 推理
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=options,
        providers=[
            (
                "CANNExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                    "op_select_impl_mode": "high_performance",
                    "optypelist_for_implmode": "Gelu",
                    "enable_cann_graph": True
                },
            ),
            "CPUExecutionProvider",
        ]
    )
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    logger.info(f"ONNX 模型加载成功: {ONNX_MODEL_PATH}")
    logger.info(f"输入名称: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
    logger.info(f"输出名称: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
    
    return session

def run_onnx_inference(session, dataloader):
    """
    执行 ONNX 模型推理并统计性能指标
    
    参数:
        session: ONNX Runtime 会话
        dataloader: 数据加载器
        
    返回:
        dict: 包含推理结果和性能指标的字典
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    logger.info("开始 ONNX 推理...")
    start_time = time.time()
    
    # 初始化统计变量
    correct = 0
    total = 0
    batch_count = 0
    
    # 用于统计吞吐量的变量
    warmup_time = 0
    formal_start_time = None
    formal_batch_times = []  # 记录每个正式batch的处理时间
    formal_batch_sizes = []  # 记录每个正式batch的大小
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 预热阶段进度条
    warmup_pbar = tqdm(
        range(WARMUP_BATCH_COUNT), 
        desc="预热阶段", 
        unit="batch",
        position=0,
        leave=True,
    )
    
    # 预热阶段
    for batch_idx in warmup_pbar:
        # 获取数据
        inputs, labels = next(iter(dataloader))
        batch_count += 1
        batch_start_time = time.time()
        
        # 将 PyTorch 张量转换为 NumPy 数组
        inputs_np = inputs.numpy()
        
        # 创建 IO 绑定
        io_binding = session.io_binding()
        
        # 将输入数据绑定到 NPU 设备
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
        io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
        io_binding.bind_output(output_name, device_type="cann", device_id=0)
        
        # 执行推理
        session.run_with_iobinding(io_binding)
        
        # 获取输出结果
        outputs = io_binding.get_outputs()[0].numpy()
        
        # 计算预测结果
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_size = labels.size(0)
        
        warmup_time += batch_time
    
    # 关闭预热进度条
    warmup_pbar.close()
    
    # 正式推理阶段进度条
    formal_pbar = tqdm(
        range(len(dataloader) - WARMUP_BATCH_COUNT), 
        desc="正式推理", 
        unit="batch",
        position=0,
    )
    
    # 跳过已处理的预热批次
    dataloader_iter = iter(dataloader)
    for i in range(WARMUP_BATCH_COUNT):
        next(dataloader_iter)  # 跳过预热批次
    
    # 正式推理阶段
    for batch_idx in formal_pbar:
        inputs, labels = next(dataloader_iter)
        batch_start_time = time.time()
        
        if formal_start_time is None:
            formal_start_time = batch_start_time
        
        # 将 PyTorch 张量转换为 NumPy 数组
        inputs_np = inputs.numpy()
        
        # 创建 IO 绑定
        io_binding = session.io_binding()
        
        # 将输入数据绑定到 NPU 设备
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
        io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
        io_binding.bind_output(output_name, device_type="cann", device_id=0)
        
        # 执行推理
        session.run_with_iobinding(io_binding)
        
        # 获取输出结果
        outputs = io_binding.get_outputs()[0].numpy()
        
        # 计算预测结果
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_size = labels.size(0)
        
        formal_batch_times.append(batch_time)
        formal_batch_sizes.append(batch_size)
    
        end_time = time.time()

    # 关闭正式推理进度条
    formal_pbar.close()

    # 计算总体指标
    total_time = end_time - start_time
    formal_time = end_time - formal_start_time if formal_start_time else 0
    accuracy = 100 * correct / total

    # 计算吞吐量（仅基于正式batch）
    if formal_batch_times:
        avg_batch_time = sum(formal_batch_times) / len(formal_batch_times)
        total_formal_images = sum(formal_batch_sizes)
        throughput = total_formal_images / formal_time  # 图片/秒
        avg_batch_size = total_formal_images / len(formal_batch_sizes)
    else:
        throughput = 0
        avg_batch_time = 0
        avg_batch_size = 0

    # 返回结果
    results = {
        "model_type": "ONNX",
        "device_used": "NPU (CANN)",
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "total_inference_time": total_time,
        "warmup_batch_count": WARMUP_BATCH_COUNT,
        "warmup_time": warmup_time,
        "formal_inference_time": formal_time,
        "formal_batch_count": len(formal_batch_times),
        "formal_images_processed": sum(formal_batch_sizes),
        "throughput_images_per_sec": throughput,
        "avg_batch_time": avg_batch_time,
        "avg_batch_size": avg_batch_size,
        "batch_count": batch_count
    }
    
    # 输出最终结果
    logger.info(f"ONNX 推理完成 - 总耗时: {total_time:.2f}s, 预热: {warmup_time:.2f}s, 正式: {formal_time:.2f}s")
    logger.info(f"准确率: {accuracy:.2f}%, 吞吐量: {throughput:.2f} 图片/秒")
    
    return results

def save_results(results):
    """保存推理结果到 JSON 文件"""
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    import json
    with open("onnx_inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("结果已保存到 onnx_inference_results.json")

def print_results(results):
    """打印推理结果"""
    print("\n===== ONNX 推理结果 =====")
    print(f"模型类型: {results['model_type']}")
    print(f"使用设备: {results['device_used']}")
    print(f"总图片数: {results['total_images']}")
    print(f"正确预测数: {results['correct_predictions']}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"总推理时间: {results['total_inference_time']:.2f}s")
    print(f"预热批次: {results['warmup_batch_count']}")
    print(f"预热时间: {results['warmup_time']:.2f}s")
    print(f"正式推理时间: {results['formal_inference_time']:.2f}s")
    print(f"正式批次: {results['formal_batch_count']}")
    print(f"正式处理图片数: {results['formal_images_processed']}")
    print(f"吞吐量: {results['throughput_images_per_sec']:.2f} 图片/秒")
    print(f"平均批次时间: {results['avg_batch_time']:.4f}s")
    print(f"平均批次大小: {results['avg_batch_size']:.1f}")
    print("=========================\n")

def main():
    """主函数 - 协调各模块完成 ONNX 推理任务"""
    # 设置日志
    logger = setup_logging()
    
    # 加载数据集
    dataset, dataloader = load_dataset()
    logger.info(f"数据集加载成功: {len(dataset)} 张图片, {len(dataloader)} 个batch")
    
    # 加载 ONNX 模型
    session = load_onnx_model()
    
    # 执行推理
    results = run_onnx_inference(session, dataloader)
    
    # 保存结果
    save_results(results)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    main()
