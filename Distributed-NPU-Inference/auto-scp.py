import json
import subprocess
import os
import argparse
from pathlib import Path

def auto_scp(json_file, source_dir, dest_dir):
    # 加载JSON文件
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: {json_file} 文件未找到")
        return
    except json.JSONDecodeError:
        print(f"错误: {json_file} 格式不正确")
        return

    # 处理每个节点
    for node_name, node_info in data.items():
        filename = node_info.get('filename')
        if not filename:
            print(f"警告: {node_name} 缺少 filename 字段，跳过")
            continue

        source_file = os.path.join(source_dir, filename)
        if not os.path.exists(source_file):
            print(f"错误: 源文件 {source_file} 不存在，跳过")
            continue

        # 构建远程命令来检查并创建目标目录
        check_dir_cmd = [
            'ssh',
            node_name,
            f'mkdir -p {dest_dir} && echo "目录已存在或创建成功" || echo "目录创建失败"'
        ]

        # 执行SCP传输
        print(f"\n处理节点 {node_name}:")
        print(f"1. 确保目标目录存在...")
        try:
            # 检查并创建远程目录
            result = subprocess.run(check_dir_cmd, capture_output=True, text=True)
            if "失败" in result.stdout:
                print(f"错误: 无法在 {node_name} 上创建目录 {dest_dir}")
                continue
            print("  目录验证成功")

            # 执行文件传输
            print(f"2. 传输 {filename} 到 {node_name}:{dest_dir}...")
            scp_command = ['scp', source_file, f"{node_name}:{dest_dir}"]
            subprocess.run(scp_command, check=True)
            print(f"  传输成功!")

        except subprocess.CalledProcessError as e:
            print(f"失败: 处理 {node_name} 时出错 - {e}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='自动将视频文件传输到NPU节点')
    parser.add_argument('--input', required=True, help='JSON配置文件路径')
    parser.add_argument('--source-dir', required=True, help='源文件目录')
    parser.add_argument('--dest-dir', required=True, help='目标目录路径')

    args = parser.parse_args()

    # 规范化路径
    source_dir = os.path.abspath(args.source_dir)
    dest_dir = args.dest_dir.rstrip('/')  # 去除末尾的斜杠

    # 检查源目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误: 源目录 {source_dir} 不存在")
        exit(1)

    # 执行主函数
    auto_scp(args.input, source_dir, dest_dir)
