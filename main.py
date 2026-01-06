import cv2
import numpy as np
import torch
import argparse
import os
from glob import glob
from skvideo.io import vreader, FFmpegWriter
from ais_bench.infer.interface import InferSession

from det_utils import letterbox, scale_coords, nms


def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)  # HWC2CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img, scale_ratio, pad_size


def draw_bbox(bbox, img0, color, wt, names):
    """在图片上画预测框"""
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    return img0

def get_labels_from_txt(path):
    """从txt文件获取图片标签"""
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict

def infer_image(img_path, model, class_names, cfg):
    """图片推理"""
    # 图片载入
    image = cv2.imread(img_path)
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    # 图片预测结果可视化
    img_vis = draw_bbox(pred_all, image, (0, 255, 0), 2, class_names)
    cv2.imwrite("output.jpg", img_vis)  # 保存结果图片
    return img_vis  # 可选返回

def infer_frame_with_vis(image, model, labels_dict, cfg, bgr2rgb=True):
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    # 图片预测结果可视化
    img_vis = draw_bbox(pred_all, image, (0, 255, 0), 2, labels_dict)
    return img_vis

def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode('.jpg', image)[1])


def infer_video(video_path, model, labels_dict, cfg, output_file_path):
    """视频推理并保存为 output_file_path"""
    # 读入视频
    cap = cv2.VideoCapture(video_path)
    # 获取原视频参数（帧率、分辨率）
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象（输出到output.mp4）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 确保你的OpenCV支持该编码
    writer = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    while True:
        ret, img_frame = cap.read()
        if not ret:
            break
        # 对视频帧进行推理
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg, bgr2rgb=True)
        # 写入处理后的帧
        writer.write(image_pred)

    # 释放资源
    cap.release()
    writer.release()
    print("视频已保存为 ", output_file_path)

def infer_camera(model, labels_dict, cfg):
    """外设摄像头实时推理（移除Jupyter依赖，仅保留基础逻辑）"""
    # 查找可用摄像头
    def find_camera_index():
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index
        raise ValueError("未检测到摄像头")

    # 初始化摄像头
    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index)

    # 创建窗口用于显示（可选）
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    while True:
        _, img_frame = cap.read()
        # 推理处理
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg)
        # 显示处理结果（按Q键退出）
        cv2.imshow("Camera Feed", image_pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

cfg = {
    'conf_thres': 0.4,  # 模型置信度阈值，阈值越低，得到的预测框越多
    'iou_thres': 0.5,  # IOU阈值，高于这个阈值的重叠预测框会被过滤掉
    'input_shape': [640, 640],  # 模型输入尺寸
}


def main():
    # 参数设置
    parser = argparse.ArgumentParser(description="Process video files in directory")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing video files")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 模型存放位置
    model_path = 'yolo.om'
    label_path = './coco_names.txt'

    # 初始化推理模型
    model = InferSession(0, model_path)
    labels_dict = get_labels_from_txt(label_path)

    # 获取目录中所有视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']  # 支持的视频格式
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(input_dir, ext)))

    # 处理每个视频文件
    for input_file_path in video_files:
        input_filename = os.path.basename(input_file_path)
        output_filename = f"output_{input_filename}"
        output_file_path = os.path.join(output_dir, output_filename)

        print(f"Processing: {input_file_path}")
        infer_video(input_file_path, model, labels_dict, cfg, output_file_path)
        print(f"Saved to: {output_file_path}")


if __name__ == "__main__":
    main()
