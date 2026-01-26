


"""
ResNet OM 推理主程序 - 使用 OM 模型在 NPU 上进行推理
"""

import logging
import numpy as np
import time
from tqdm import tqdm
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, DATASET_VAL_DIR, BATCH_SIZE, RESIZE_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ais_bench.infer.interface import InferSession

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
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        return dataset, dataloader
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        exit(1)

def load_om_model():
    """
    加载 OM 模型
    
    返回:
        session: OM 推理会话
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    # 初始化推理会话
    model_path = "/ascend-models/bench_01_resnet/resnet50.om"
    # model_path = "/home/xubowen/bench_01_resnet/CANNExecutionProvider_main_graph_14988549985323266822_0_0_584614948278820172.om"
    session = InferSession(device_id=0, model_path=model_path)
    
    logger.info(f"OM 模型加载成功: {model_path}")
    
    return session

def run_om_inference(session, dataloader):
    """
    执行 OM 模型推理并统计性能指标
    
    参数:
        session: OM 推理会话
        dataloader: 数据加载器
        
    返回:
        dict: 包含推理结果和性能指标的字典
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    logger.info("开始 OM 推理...")
    start_time = time.time()
    
    # 初始化统计变量
    correct = 0
    total = 0
    batch_count = 0
    
    # 用于统计吞吐量的变量
    batch_times = []  # 记录每个batch的处理时间
    batch_sizes = []  # 记录每个batch的大小
    
    # 推理进度条
    pbar = tqdm(
        dataloader, 
        desc="OM 推理", 
        unit="batch",
        position=0,
    )
    
    # 推理循环
    for inputs, labels in pbar:
        batch_start_time = time.time()
        
        # 将 PyTorch 张量转换为 NumPy 数组
        inputs_np = inputs.numpy()
        
        # 执行推理
        # inputs_np 的形状是 [batch_size,3,224,224]
        # outputs 的形状是 [batch_size,1000]
        outputs = session.infer([inputs_np])
        
        # 计算预测结果
        predicted = np.argmax(outputs[0], axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_size = labels.size(0)
        
        batch_times.append(batch_time)
        batch_sizes.append(batch_size)
        batch_count += 1
        
        # 更新进度条显示当前信息
        pbar.set_postfix({
            'Acc': f'{100 * correct / total:.2f}%',
            'Batch_Time': f'{batch_time:.3f}s'
        })
    
    end_time = time.time()
    
    # 关闭进度条
    pbar.close()
    
    # 计算总体指标
    total_time = end_time - start_time
    accuracy = 100 * correct / total
    
    # 计算吞吐量
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_images = sum(batch_sizes)
        throughput = total_images / total_time  # 图片/秒
        avg_batch_size = total_images / len(batch_sizes)
    else:
        throughput = 0
        avg_batch_time = 0
        avg_batch_size = 0
    
    # 返回结果
    results = {
        "model_type": "OM",
        "device_used": "NPU",
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "total_inference_time": total_time,
        "batch_count": batch_count,
        "throughput_images_per_sec": throughput,
        "avg_batch_time": avg_batch_time,
        "avg_batch_size": avg_batch_size
    }
    
    # 输出最终结果
    logger.info(f"OM 推理完成 - 总耗时: {total_time:.2f}s")
    logger.info(f"准确率: {accuracy:.2f}%, 吞吐量: {throughput:.2f} 图片/秒")
    
    return results

def save_results(results):
    """保存推理结果到 JSON 文件"""
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    import json
    with open("om_inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("结果已保存到 om_inference_results.json")

def print_results(results):
    """打印推理结果"""
    print("\n===== OM 推理结果 =====")
    print(f"模型类型: {results['model_type']}")
    print(f"使用设备: {results['device_used']}")
    print(f"总图片数: {results['total_images']}")
    print(f"正确预测数: {results['correct_predictions']}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"总推理时间: {results['total_inference_time']:.2f}s")
    print(f"批次数量: {results['batch_count']}")
    print(f"吞吐量: {results['throughput_images_per_sec']:.2f} 图片/秒")
    print(f"平均批次时间: {results['avg_batch_time']:.4f}s")
    print(f"平均批次大小: {results['avg_batch_size']:.1f}")
    print("=========================\n")

def main():
    """主函数 - 协调各模块完成 OM 推理任务"""
    # 设置日志
    logger = setup_logging()
    
    # 加载数据集
    dataset, dataloader = load_dataset()
    logger.info(f"数据集加载成功: {len(dataset)} 张图片, {len(dataloader)} 个batch")
    
    # 加载 OM 模型
    session = load_om_model()
    
    # 执行推理
    results = run_om_inference(session, dataloader)
    
    # 保存结果
    save_results(results)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    main()
