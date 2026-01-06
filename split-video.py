import argparse
import os
import cv2
import json

def split_video(input_path, output_dir, segments):
    # 验证比例总和为1
    total_ratio = sum(seg['ratio'] for seg in segments)
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError("比例之和必须等于1")

    # 准备输出路径并创建目录
    outputs = []
    for seg in segments:
        output_path = os.path.join(output_dir, seg['filename'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        outputs.append(output_path)

    # 检查文件名唯一性
    if len(outputs) != len(set(outputs)):
        raise ValueError("输出文件名存在重复")

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # 计算分割点
    split_points = []
    accumulated = 0.0
    for seg in segments[:-1]:  # 最后一个分割点不需要计算
        accumulated += seg['ratio']
        split_points.append(int(total_frames * accumulated))

    # 创建视频写入器
    writers = [cv2.VideoWriter(output, codec, fps, (width, height))
              for output in outputs]

    # 读取并写入帧
    current_writer_idx = 0
    next_split = split_points[current_writer_idx] if split_points else total_frames

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        writers[current_writer_idx].write(frame)

        # 检查是否需要切换到下一个写入器
        if frame_count + 1 == next_split and current_writer_idx < len(writers) - 1:
            current_writer_idx += 1
            next_split = split_points[current_writer_idx] if current_writer_idx < len(split_points) else total_frames

    # 释放资源
    cap.release()
    for writer in writers:
        writer.release()

    print("视频已分割为:")
    for output in outputs:
        print(f"  {output}")


def parse_npu_json(json_path):
    try:
        with open(json_path, 'r') as f:
            npu_data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"JSON文件未找到: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON文件格式错误: {json_path}")

    segments = []
    total_power = 0
    filenames = set()

    # 获取命令行输入的input参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args, _ = parser.parse_known_args()  # 允许部分解析参数
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    devices = list(npu_data.items())  # 保持原始顺序

    for idx, (device, config_power) in enumerate(devices, 1):
        power = float(config_power)
        filename = f"{base_name}_{idx}.mp4"

        # 数据校验
        if power <= 0:
            raise ValueError(f"设备 {device} 的算力值必须大于0")
        if not filename.strip():
            raise ValueError(f"设备 {device} 的文件名不能为空")
        if filename in filenames:
            raise ValueError(f"文件名重复: {filename}")
        filenames.add(filename)
        total_power += power
        segments.append({
            'device': device,
            'power': power,
            'filename': filename.strip()
        })

    if total_power <= 0:
        raise ValueError("总算力值必须大于0")

    # 计算比例
    for seg in segments:
        seg['ratio'] = seg['power'] / total_power

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据NPU算力配置分割视频")
    parser.add_argument("--input", required=True, help="输入视频文件路径")
    parser.add_argument("--output", required=True, help="输出目录路径")
    parser.add_argument("--npu-json", required=True,
                      help="包含NPU算力配置的JSON文件路径")

    args = parser.parse_args()

    try:
        segments = parse_npu_json(args.npu_json)
        # 提取比例列表（保持原有分割逻辑兼容）
        ratios = [seg['ratio'] for seg in segments]
        # 调用修改后的分割函数
        split_video(args.input, args.output, segments)

        # 将处理后的数据写回JSON文件
        modified_data = {}
        for seg in segments:
            device = seg['device']
            modified_data[device] = {
                'power': seg['power'],
                'filename': seg['filename']
            }

        with open(args.npu_json, 'w') as f:
            json.dump(modified_data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"错误: {e}")