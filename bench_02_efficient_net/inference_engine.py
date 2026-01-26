"""
推理引擎模块 - 负责执行推理和性能统计
"""

import torch
import torch_npu
import time
import logging
import numpy as np
from tqdm import tqdm
from config import WARMUP_BATCH_COUNT, USE_FP16

logger = logging.getLogger(__name__)

def run_inference(model, device, dataloader):
    """
    执行模型推理并统计性能指标
    
    参数:
        model: 加载的模型
        device: 使用的设备
        dataloader: 数据加载器
        
    返回:
        dict: 包含推理结果和性能指标的字典
    """
    torch.npu.set_compile_mode(jit_compile=False)

    logger.info("开始推理...")
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
    
    # 用于统计延迟的变量
    e2e_latencies = []  # 端到端延迟列表 (预处理开始到后处理结束)
    inference_latencies = []  # 纯推理延迟列表 (model()执行时间)
    
    with torch.no_grad():
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
            batch_start_time = time.time()
            batch_count += 1
            inputs, labels = next(iter(dataloader))
            
            # 将输入数据迁移到NPU设备
            inputs = inputs.to(device).contiguous()
            labels = labels.to(device)
            
            # 如果使用FP16，转换输入数据精度
            if USE_FP16:
                inputs = inputs.half()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
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
        
        formal_start_time = time.time()
        # 正式推理阶段
        for batch_idx in formal_pbar:
            # 端到端延迟开始时间戳 (预处理开始前)
            t_start_e2e = time.time()

            inputs, labels = next(dataloader_iter)
            
            # 将输入数据迁移到NPU设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 如果使用FP16，转换输入数据精度
            if USE_FP16:
                inputs = inputs.half()

            t_start_infer = time.time()
            outputs = model(inputs)
            t_end_infer = time.time()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            # print(f"predicted: {predicted}")
            # print(f"labels: {labels}")
            # print("==============================")

            correct += (predicted == labels).sum().item()
            
            # 端到端延迟结束时间戳 (后处理结束后)
            t_end_e2e = time.time()
            
            # 计算并记录延迟
            e2e_latency = t_end_e2e - t_start_e2e
            inference_latency = t_end_infer - t_start_infer
            
            e2e_latencies.append(e2e_latency)
            inference_latencies.append(inference_latency)
            
            batch_time = e2e_latency
            batch_size = labels.size(0)
            
            formal_batch_times.append(batch_time)
            formal_batch_sizes.append(batch_size)
        
            end_time = time.time()

        # 关闭正式推理进度条
        formal_pbar.close()

    # 计算总体指标
    total_time = end_time - start_time
    formal_time = end_time - formal_start_time
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
    
    # 计算延迟统计
    def calculate_percentiles(latencies):
        """计算延迟的P95和P99"""
        if not latencies:
            return {"p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
        
        latencies_array = np.array(latencies)
        return {
            "p95": np.percentile(latencies_array, 95),
            "p99": np.percentile(latencies_array, 99),
            "mean": np.mean(latencies_array),
            "min": np.min(latencies_array),
            "max": np.max(latencies_array)
        }
    
    # 只计算纯推理延迟的统计信息
    inference_latency_stats = calculate_percentiles(inference_latencies)
    
    # 计算端到端延迟的统计信息
    e2e_latency_stats = calculate_percentiles(e2e_latencies)

    # 返回结果
    results = {
        "device_used": str(device),
        "use_fp16": USE_FP16,
        "precision": "FP16" if USE_FP16 else "FP32",
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
        "batch_count": batch_count,
        # 添加纯推理延迟统计数据
        "inference_latency_stats": inference_latency_stats,
        # 添加端到端延迟统计数据
        "e2e_latency_stats": e2e_latency_stats,
        # 保留原始延迟数据
        "e2e_latencies": e2e_latencies,
        "inference_latencies": inference_latencies
    }
    
    # 输出最终结果
    logger.info(f"推理完成 - 总耗时: {total_time:.2f}s, 预热: {warmup_time:.2f}s, 正式: {formal_time:.2f}s")
    logger.info(f"准确率: {accuracy:.2f}%, 吞吐量: {throughput:.2f} 图片/秒")
    
    return results