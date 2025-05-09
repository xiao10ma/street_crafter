import os
import imageio
import numpy as np
import cv2
from tqdm import tqdm

def generate_video_from_images(input_dir, output_path, fps=10):
    """
    从指定目录中读取所有非mask图片，按顺序排列并生成视频
    
    参数:
        input_dir: 输入图片目录
        output_path: 输出视频路径
        fps: 视频帧率
    """
    # 获取所有不以mask.png结尾的图片
    image_files = [f for f in os.listdir(input_dir) if f.endswith('_0.png')]
    
    # 按文件名排序
    image_files = sorted(image_files)

    image_files = image_files[:40]
    
    if not image_files:
        print("未找到符合条件的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 读取图片
    images = []
    for img_file in tqdm(image_files, desc="读取图片"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV读取的是BGR，转为RGB
            images.append(img)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成视频
    print(f"正在生成视频: {output_path}")
    imageio.mimwrite(output_path, images, fps=fps)
    print(f"视频生成完成: {output_path}")

if __name__ == "__main__":
    # 指定输入和输出路径
    input_directory = "data/waymo/121/images"
    output_video = "./video.mp4"
    
    # 生成视频
    generate_video_from_images(input_directory, output_video, fps=10)
