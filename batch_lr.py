import os
import torch
import cv2
import numpy as np
from src.loftr import LoFTR, default_cfg
import time
import argparse
import logging
from copy import deepcopy
from matplotlib import cm
from src.utils.plotting import make_matching_figure
import matplotlib.pyplot as plt

def setup_args():
    parser = argparse.ArgumentParser(description="Process image pairs for feature matching.")
    parser.add_argument('--image_dir', default='/home/surgicalai/Data/images_loFTR', help='Directory containing input images')
    parser.add_argument('--weights', default='weights/indoor_ds_new.ckpt', help='Path to model weights')
    parser.add_argument('--save_dir', default='/home/surgicalai/Data/output/LoFTR', help='Directory to save matching images')
    return parser.parse_args()

def main():
    args = setup_args()
    
    # 检查保存目录并创建
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.isdir(args.save_dir) or not os.access(args.save_dir, os.W_OK):
        print(f"Error: Cannot write to {args.save_dir}. Check directory permissions.")
        return

    # 设置日志文件路径
    log_file = os.path.join(args.save_dir, 'output.log')
    
    # 检查图像目录写入权限
    if not os.path.isdir(args.image_dir) or not os.access(args.image_dir, os.W_OK):
        print(f"Error: Cannot write to {args.image_dir}. Check directory permissions.")
        return

    # 配置日志
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='a',
        force=True
    )
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("Logging initialized successfully.")
    print(f"Logging to {log_file}")

    start_time = time.time()

    # 模型初始化
    try:
        config = deepcopy(default_cfg)
        config['coarse']['temp_bug_fix'] = True
        matcher = LoFTR(config=config)
        matcher.load_state_dict(torch.load(args.weights)['state_dict'])
        matcher = matcher.eval().cuda() if torch.cuda.is_available() else matcher.eval()
        logging.info("Model initialized successfully.")
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        print(f"Error: Model initialization failed: {e}")
        return

    # 获取图像文件
    image_files = sorted([f for f in os.listdir(args.image_dir) 
                         if f.endswith(('.jpg', '.png'))])
    if not image_files:
        logging.error("No valid images found in the directory.")
        print("Error: No valid images found.")
        return

    logging.info(f"Found {len(image_files)} images in {args.image_dir}")

    # 按前缀分组左右目图像
    image_pairs = []
    prefix_dict = {}
    for img in image_files:
        # 假设文件名格式为 prefix_0.png 或 prefix_1.png
        if '_' in img and img.rsplit('.', 1)[0].endswith(('_0', '_1')):
            prefix = img.rsplit('_', 1)[0]  # 提取前缀，如 'aaa', 'bbb'
            suffix = img.rsplit('_', 1)[1].rsplit('.', 1)[0]  # 提取 '_0' 或 '_1'
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {}
            prefix_dict[prefix][suffix] = img

    # 配对左右目图像
    for prefix in prefix_dict:
        if '0' in prefix_dict[prefix] and '1' in prefix_dict[prefix]:
            image_pairs.append((prefix_dict[prefix]['0'], prefix_dict[prefix]['1']))
        else:
            logging.warning(f"Missing pair for prefix {prefix}: {prefix_dict[prefix]}")

    if not image_pairs:
        logging.error("No valid left-right image pairs found.")
        print("Error: No valid image pairs found.")
        return

    logging.info(f"Found {len(image_pairs)} left-right image pairs")

    # 处理每对图像
    for pair_idx, (img0_name, img1_name) in enumerate(image_pairs):
        logging.info(f"\nProcessing Pair {pair_idx + 1} (Images: {img0_name}, {img1_name})")
        pair_start_time = time.time()

        try:
            # 加载图像对
            img0_path = os.path.join(args.image_dir, img0_name)
            img1_path = os.path.join(args.image_dir, img1_name)
            img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

            if img0_raw is None or img1_raw is None:
                raise ValueError(f"Failed to load {img0_path} or {img1_path}")

            # 预处理：调整大小并转换为张量
            img0_raw = cv2.resize(img0_raw, (640, 480))
            img1_raw = cv2.resize(img1_raw, (640, 480))
            img0 = torch.from_numpy(img0_raw)[None][None].float().cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].float().cuda() / 255.
            batch = {'image0': img0, 'image1': img1}

            # 模型推理
            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

            # 生成并保存匹配图像
            color = cm.jet(mconf)
            text = [
                'LoFTR',
                'Matches: {}'.format(len(mkpts0)),
            ]
            fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
            save_path = os.path.join(args.save_dir, f'match_pair{pair_idx + 1}_{img0_name}_{img1_name}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

            # 记录匹配点数量
            num_matches = len(mkpts0)
            pair_time = time.time() - pair_start_time
            logging.info(f"Pair {pair_idx + 1}: "
                         f"{img0_name} - {img1_name}, "
                         f"Matches: {num_matches}, Time: {pair_time:.2f}s, "
                         f"Saved to: {save_path}")

            # 释放内存
            del batch, img0, img1
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Pair {pair_idx + 1} failed: {e}")
            continue

    # 总统计
    total_time = time.time() - start_time
    logging.info(f"\nAll pairs processed!")
    logging.info(f"Total runtime: {total_time:.2f}s")

    # 确保日志写入磁盘
    for handler in logging.getLogger().handlers:
        handler.flush()

if __name__ == "__main__":
    main()