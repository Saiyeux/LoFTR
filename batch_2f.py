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
    parser = argparse.ArgumentParser(description="Process image pairs from left and right directories for feature matching.")
    parser.add_argument('--left_dir', default='/home/surgicalai/Data/misc/images/L', help='Directory containing left images')
    parser.add_argument('--right_dir', default='/home/surgicalai/Data/misc/images/R', help='Directory containing right images')
    parser.add_argument('--weights', default='weights/indoor_ds_new.ckpt', help='Path to model weights')
    parser.add_argument('--save_dir', default='/home/surgicalai/Data/misc/images_loFTR', help='Directory to save matching images')
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
    for dir_path in [args.left_dir, args.right_dir]:
        if not os.path.isdir(dir_path) or not os.access(dir_path, os.R_OK):
            print(f"Error: Cannot read from {dir_path}. Check directory permissions.")
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

    # 获取左右目图像文件
    left_images = sorted([f for f in os.listdir(args.left_dir) 
                         if f.endswith(('.jpg', '.png'))])
    right_images = sorted([f for f in os.listdir(args.right_dir) 
                          if f.endswith(('.jpg', '.png'))])

    if not left_images or not right_images:
        logging.error("No valid images found in one or both directories.")
        print("Error: No valid images found in one or both directories.")
        return

    logging.info(f"Found {len(left_images)} left images and {len(right_images)} right images")

    # 提取文件名（不含扩展名）并找出共同文件名
    left_names = {os.path.splitext(f)[0] for f in left_images}
    right_names = {os.path.splitext(f)[0] for f in right_images}
    common_names = sorted(left_names.intersection(right_names))

    if not common_names:
        logging.error("No matching image pairs found between left and right directories.")
        print("Error: No matching image pairs found.")
        return

    logging.info(f"Found {len(common_names)} matching image pairs")

    # 处理每对图像
    for pair_idx, base_name in enumerate(common_names):
        # 查找对应的左目和右目图像文件（支持不同扩展名）
        left_file = next((f for f in left_images if os.path.splitext(f)[0] == base_name), None)
        right_file = next((f for f in right_images if os.path.splitext(f)[0] == base_name), None)

        if left_file is None or right_file is None:
            logging.warning(f"Pair {pair_idx + 1} skipped: Missing file for {base_name}")
            continue

        logging.info(f"\nProcessing Pair {pair_idx + 1} (Images: L/{left_file}, R/{right_file})")
        pair_start_time = time.time()

        try:
            # 加载图像对
            img0_path = os.path.join(args.left_dir, left_file)
            img1_path = os.path.join(args.right_dir, right_file)
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
            save_path = os.path.join(args.save_dir, f'match_pair{pair_idx + 1}_{base_name}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

            # 记录匹配点数量
            num_matches = len(mkpts0)
            pair_time = time.time() - pair_start_time
            logging.info(f"Pair {pair_idx + 1}: "
                         f"L/{left_file} - R/{right_file}, "
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