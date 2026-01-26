import subprocess
import re
import time
import threading
import datetime
import matplotlib.pyplot as plt

class NPUPowerMonitor:
    def __init__(self, device_id, interval=1.0):
        """
        初始化 NPU 功耗监视器
        :param device_id: NPU 设备 ID (例如 21504)
        :param interval: 采样间隔 (秒)
        """
        self.device_id = str(device_id)
        self.interval = interval
        self.records = []  # 存储数据: [(time_offset, power), ...]
        self.start_time = None
        self._running = False
        self._thread = None
    
    def _get_power(self):
        """单次调用 npu-smi 获取功耗"""
        cmd = ["npu-smi", "info", "-t", "power", "-i", self.device_id]
        try:
            # 执行命令
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            output = result.decode('utf-8')
            
            # 匹配格式: NPU Real-time Power(W) : 47.9
            match = re.search(r"NPU Real-time Power\(W\)\s*:\s*([\d\.]+)", output)
            if match:
                return float(match.group(1))
        except Exception:
            pass  # 忽略单次获取失败，避免打断线程
        return None

    def _monitor_loop(self):
        """后台线程循环"""
        self.start_time = time.time()
        while self._running:
            current_power = self._get_power()
            if current_power is not None:
                # 记录相对时间（从开始经过的秒数）和当前功耗
                elapsed = time.time() - self.start_time
                self.records.append((elapsed, current_power))
            
            time.sleep(self.interval)

    def start(self):
        """开始后台监视"""
        if self._running:
            print("Monitor is already running.")
            return
        
        print(f"Start monitoring NPU {self.device_id} power...")
        self.records = []  # 清空旧数据
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监视"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join()  # 等待线程结束
        print(f"Monitoring stopped. Captured {len(self.records)} samples.")

    def analyze_and_plot(self, save_path="power_curve.png"):
        """分析数据并画图"""
        if not self.records:
            print("No data collected.")
            return

        # 解包数据
        times, powers = zip(*self.records)
        
        # 统计指标
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        total_time = times[-1] - times[0] if len(times) > 1 else 0
        # 简单估算能耗 (焦耳) = 平均功率(W) * 时间(s)
        total_energy = avg_power * total_time 

        print(f"\n=== NPU Power Analysis ===")
        print(f"Duration    : {total_time:.2f} s")
        print(f"Avg Power   : {avg_power:.2f} W")
        print(f"Max Power   : {max_power:.2f} W")
        print(f"Total Energy: {total_energy:.2f} Joules")
        print(f"==========================")

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(times, powers, label=f"NPU {self.device_id}", color='tab:blue')
        
        # 绘制平均线
        plt.axhline(y=avg_power, color='r', linestyle='--', label=f'Avg: {avg_power:.1f}W')
        
        plt.title(f"NPU Power Consumption (Device {self.device_id})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        print(f"Power curve saved to {save_path}")
        plt.close()