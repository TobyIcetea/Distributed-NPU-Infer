"""
结果保存模块 - 负责保存推理结果和日志
"""

import json
import logging
from config import RESULTS_FILE

logger = logging.getLogger(__name__)

def save_results(results):
    """
    保存推理结果到JSON文件
    
    参数:
        results (dict): 包含推理结果和性能指标的字典
    """
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"推理结果已保存至 {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def print_results(results):
    """
    打印推理结果到控制台
    
    参数:
        results (dict): 包含推理结果和性能指标的字典
    """
    print(f"\n推理结果:")
    print(f"- 使用精度: {results['precision']}")
    print(f"- 总耗时: {results['total_inference_time']:.2f} 秒")
    print(f"- 预热时间: {results['warmup_time']:.2f} 秒")
    print(f"- 正式推理时间: {results['formal_inference_time']:.2f} 秒")
    # 这个模型用 torch_npu 推理的结果没眼看，直接跳过
    # print(f"- 准确率: {results['accuracy']:.2f}%")
    print(f"- 吞吐量: {results['throughput_images_per_sec']:.2f} FPS")
    print(f"- 使用设备: {results['device_used']}")
    
    # 只打印纯推理延迟统计信息
    print("\n纯推理延迟统计:")
    print(f"- 平均延迟: {results['inference_latency_stats']['mean']*1000:.2f} ms")
    print(f"- P95延迟: {results['inference_latency_stats']['p95']*1000:.2f} ms")
    print(f"- P99延迟: {results['inference_latency_stats']['p99']*1000:.2f} ms")
    print(f"- 最小延迟: {results['inference_latency_stats']['min']*1000:.2f} ms")
    print(f"- 最大延迟: {results['inference_latency_stats']['max']*1000:.2f} ms")
    
    # 打印端到端延迟统计信息
    # print("\n端到端延迟统计:")
    # print(f"- 平均延迟: {results['e2e_latency_stats']['mean']*1000:.2f} ms")
    # print(f"- P95延迟: {results['e2e_latency_stats']['p95']*1000:.2f} ms")
    # print(f"- P99延迟: {results['e2e_latency_stats']['p99']*1000:.2f} ms")
    # print(f"- 最小延迟: {results['e2e_latency_stats']['min']*1000:.2f} ms")
    # print(f"- 最大延迟: {results['e2e_latency_stats']['max']*1000:.2f} ms")
    
    # 计算并显示处理效率（端到端）
    total_images = results['total_images']
    total_e2e_time = sum(results['e2e_latencies'])
    e2e_efficiency = total_images / total_e2e_time if total_e2e_time > 0 else 0
    
    print("\n端到端处理效率:")
    print(f"- 总处理图片数: {total_images}")
    print(f"- 总端到端时间: {total_e2e_time:.2f} 秒")
    print(f"- 平均端到端时间: {total_e2e_time/len(results['e2e_latencies']):.4f} 秒/批次")
    print(f"- 处理效率: {e2e_efficiency:.2f} 图片/秒")