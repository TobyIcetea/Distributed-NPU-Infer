import argparse
import os
import glob
import cv2
import sys
import json
import subprocess

def collect_results(json_file_path):
    # 创建segments目录（如果不存在）
    os.makedirs("segments", exist_ok=True)

    # 读取JSON文件
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 遍历每个节点
    for node in data:
        filename = data[node]["filename"]
        remote_file = f"{node}:/root/workdir/data/output-file/output_{filename}"
        local_dir = "segments/"

        # 构造SCP命令
        command = ["scp", remote_file, local_dir]

        print(f"正在复制 {remote_file} 到 segments...")
        try:
            # 执行SCP命令
            subprocess.run(command, check=True)
            print(f"成功复制 {remote_file}")
        except subprocess.CalledProcessError as e:
            print(f"错误：复制 {remote_file} 失败，错误信息：{str(e)}")

def merge_videos(input_files, output_dir):
    """
    合并多个视频片段为一个完整视频

    参数:
        input_files: 输入视频文件列表
        output_dir: 合并后的输出目录
    """
    if not input_files:
        raise ValueError("没有找到任何视频文件")

    print(f"找到 {len(input_files)} 个视频片段:")
    for f in input_files:
        print(f"  {f}")

    # 生成输出文件名
    first_file = input_files[0]
    base_name = os.path.basename(first_file).replace('output_', '', 1)
    if '_' in base_name:
        base_part = base_name.split('_')[0]
        output_filename = f"{base_part}_output.mp4"
    else:
        output_filename = 'output_' + base_name

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 读取第一个视频获取参数
    sample = cv2.VideoCapture(first_file)
    fps = sample.get(cv2.CAP_PROP_FPS)
    width = int(sample.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    sample.release()

    # 创建输出视频写入器
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # 逐个读取并写入视频片段
    for input_file in input_files:
        print(f"正在处理: {input_file}")
        cap = cv2.VideoCapture(input_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()

    out.release()
    print(f"\n视频合并完成: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个视频片段为一个完整视频")
    parser.add_argument("--input-dir", required=True, help="输入目录路径")
    parser.add_argument("--json", required=True, help="JSON配置文件路径")
    parser.add_argument("--output-dir", required=True, help="合并后的输出目录")

    args = parser.parse_args()

    collect_results(args.json)

    # 读取JSON配置
    with open(args.json, "r") as f:
        data = json.load(f)

    # 收集输入文件列表
    input_files = []
    for node in data:
        filename = data[node]["filename"]
        file_path = os.path.join(args.input_dir, f"output_{filename}")
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，跳过")
            continue
        input_files.append(file_path)

    if not input_files:
        raise ValueError("没有找到任何有效输入文件")

    try:
        merge_videos(input_files, args.output_dir)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
