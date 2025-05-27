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

def setup_args():
    parser = argparse.ArgumentParser(description="Process image pairs for feature matching.")
    parser.add_argument('--image_dir', default='/home/surgicalai/Data/images_2', help='Directory containing input images')
    parser.add_argument('--weights', default='weights/indoor_ds_new.ckpt', help='Path to model weights')
    parser.add_argument('--group_size', type=int, default=100, help='Number of images per group for processing')
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

    # 配置日志，追加模式，强制刷新
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

    # 确保至少有2张图像
    if len(image_files) < 2:
        logging.error("Need at least 2 images for matching.")
        print("Error: Need at least 2 images.")
        return

    # 分组处理：按 group_size 分组，保证 n-1 组输出
    group_size = min(args.group_size, len(image_files))
    num_images = len(image_files)
    num_groups = num_images - 1

    logging.info(f"Processing {num_groups} groups with group_size={group_size}")

    for group_idx in range(num_groups):
        # 确定当前组的图像对
        group_files = [image_files[group_idx], image_files[group_idx + 1]]
        logging.info(f"\nProcessing Group {group_idx + 1} (Images: {group_files[0]}, {group_files[1]})")

        # 记录组内匹配点
        group_matches = []
        group_start_time = time.time()

        # 处理当前图像对
        pair_start_time = time.time()
        try:
            # 加载图像对
            img0_path = os.path.join(args.image_dir, group_files[0])
            img1_path = os.path.join(args.image_dir, group_files[1])
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
            save_path = os.path.join(args.save_dir, f'match_group{group_idx + 1}_{group_files[0]}_{group_files[1]}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)  # 关闭图像，防止内存泄漏

            # 记录匹配点数量
            num_matches = len(mkpts0)
            group_matches.append(num_matches)

            # 日志记录
            pair_time = time.time() - pair_start_time
            logging.info(f"Group {group_idx + 1}, Pair: "
                        f"{group_files[0]} - {group_files[1]}, "
                        f"Matches: {num_matches}, Time: {pair_time:.2f}s, "
                        f"Saved to: {save_path}")

            # 释放内存
            del batch, img0, img1
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Group {group_idx + 1}, Pair failed: {e}")
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
    import matplotlib.pyplot as plt  # 导入 matplotlib 用于保存图像
    main()