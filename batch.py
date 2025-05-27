import os
import torch
import cv2
import numpy as np
from src.loftr import LoFTR, default_cfg
import time
import argparse
import logging
from copy import deepcopy

def setup_args():
    parser = argparse.ArgumentParser(description="Process image pairs for feature matching.")
    parser.add_argument('--image_dir', required=True, help='Directory containing input images')
    parser.add_argument('--weights', default='weights/indoor_ds_new.ckpt', 
                        help='Path to model weights')
    return parser.parse_args()

def main():
    args = setup_args()
    
    # 设置日志文件路径
    log_file = os.path.join(args.image_dir, 'output.log')
    
    # 检查目录写入权限
    if not os.path.isdir(args.image_dir) or not os.access(args.image_dir, os.W_OK):
        print(f"Error: Cannot write to {args.image_dir}. Check directory permissions.")
        return

    # 配置日志，追加模式，强制刷新
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='a',  # 追加模式（可改为 'w' 覆盖）
        force=True  # 确保日志配置生效
    )
    
    # 添加文件处理器，确保日志写入
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # 记录日志配置完成
    logging.info("Logging initialized successfully.")
    print(f"Logging to {log_file}")  # 控制台提示日志路径

    # 记录总开始时间
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

    # 分组处理：每10张为一组
    group_size = 10
    for group_idx, start_idx in enumerate(range(0, len(image_files), group_size)):
        # 获取当前组的图像（最多10张）
        group_files = image_files[start_idx:start_idx + group_size]
        if len(group_files) < 2:
            logging.warning(f"Group {group_idx + 1} has {len(group_files)} images, skipping.")
            continue

        logging.info(f"\nProcessing Group {group_idx + 1} ({len(group_files)} images)")

        # 记录组内匹配点
        group_matches = []
        group_start_time = time.time()

        # 相邻图像两两配对（1-2, 2-3, ..., n-1-n）
        for i in range(len(group_files) - 1):
            pair_start_time = time.time()

            try:
                # 加载图像对
                img0_path = os.path.join(args.image_dir, group_files[i])
                img1_path = os.path.join(args.image_dir, group_files[i + 1])
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

                # 记录匹配点数量
                num_matches = len(mkpts0)
                group_matches.append(num_matches)

                # 日志记录
                pair_time = time.time() - pair_start_time
                logging.info(f"Group {group_idx + 1}, Pair {i + 1}: "
                            f"{group_files[i]} - {group_files[i + 1]}, "
                            f"Matches: {num_matches}, Time: {pair_time:.2f}s")

                # 释放内存
                del batch, img0, img1
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Group {group_idx + 1}, Pair {i + 1} failed: {e}")
                continue

        # 组内统计
        if group_matches:
            total_matches = sum(group_matches)
            avg_matches = total_matches / len(group_matches)
            group_time = time.time() - group_start_time
            logging.info(f"Group {group_idx + 1} Summary: "
                        f"Total Matches: {total_matches}, "
                        f"Average Matches per Pair: {avg_matches:.2f}, "
                        f"Group Time: {group_time:.2f}s")
        else:
            logging.warning(f"Group {group_idx + 1} has no valid matches.")

    # 总统计
    total_time = time.time() - start_time
    logging.info(f"\nAll groups processed!")
    logging.info(f"Total runtime: {total_time:.2f}s")

    # 确保日志写入磁盘
    for handler in logging.getLogger().handlers:
        handler.flush()

if __name__ == "__main__":
    main()