import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split, get_object_center
from detector import name2estimator
from utils.base_utils import load_cfg, project_points, transformation_crop
from utils.database_utils import compute_normalized_view_correlation
from utils.pose_utils import scale_rotation_difference_from_cameras


def get_gt_bbox(que_pose, que_K, render_poses, render_Ks, object_center, size=128):
    """
    计算真实框信息
    @param que_pose: 查询图像的pose [3,4]
    @param que_K: 查询图像的相机内参 [3,3]
    @param render_poses: 参考视图的poses [rfn, 3, 4]
    @param render_Ks: 参考视图的Ks [rfn, 3, 3]
    @param object_center: 物体中心 [3]
    @param size: 裁剪尺寸，默认128
    @return: gt_position, gt_scale_r2q, gt_angle_r2q, gt_bbox
    """
    # 计算与参考视图的相关性
    gt_corr = compute_normalized_view_correlation(que_pose[None], render_poses, object_center, False)
    gt_ref_idx = np.argmax(gt_corr[0])
    
    # 计算尺度和角度差异
    gt_scale_r2q, gt_angle_r2q = scale_rotation_difference_from_cameras(
        render_poses[gt_ref_idx][None], que_pose[None], 
        render_Ks[gt_ref_idx][None], que_K[None], object_center)
    gt_scale_r2q, gt_angle_r2q = gt_scale_r2q[0], gt_angle_r2q[0]
    
    # 计算物体中心在图像中的投影位置
    gt_position = project_points(object_center[None], que_pose, que_K)[0][0]
    
    # 计算边界框 [x, y, w, h]
    gt_bbox = np.concatenate([gt_position - size / 2 * gt_scale_r2q, np.full(2, size) * gt_scale_r2q])
    
    return gt_position, gt_scale_r2q, gt_angle_r2q, gt_bbox


def crop_image_by_bbox(img, bbox, size=128):
    """
    根据边界框裁剪图片
    @param img: 输入图片 [h, w, 3]
    @param bbox: 边界框 [x, y, w, h]
    @param size: 输出尺寸
    @return: 裁剪后的图片 [size, size, 3]
    """
    x, y, w, h = bbox
    position = np.array([x + w/2, y + h/2], dtype=np.float32)
    
    # transformation_crop的scale参数：如果scale=1/scale_r2q，其中scale_r2q是参考到查询的尺度比
    # 如果bbox的尺寸是w*h，输出尺寸是size*size，我们希望bbox区域被缩放到size
    # 所以scale_r2q = w/size，scale = size/w
    bbox_size = max(w, h)
    scale = size / bbox_size  # 直接计算缩放因子，使bbox区域缩放到size
    
    # 使用transformation_crop进行裁剪（不旋转，angle=0）
    cropped_img, _ = transformation_crop(img, position, scale, 0, size)
    
    return cropped_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/cas6d_train.yaml')
    parser.add_argument('--object_name', type=str, default='linemod/cat')
    parser.add_argument('--split_type', type=str, default='linemod_test')
    parser.add_argument('--ref_num', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='../data/exp/cropped_images/linemod_cat')
    parser.add_argument('--crop_size', type=int, default=128)
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_cfg(args.cfg)
    cfg["det_ref_view_num"] = args.ref_num
    
    # 解析数据库
    ref_database_name = que_database_name = args.object_name
    ref_database = parse_database_name(ref_database_name)
    que_database = parse_database_name(que_database_name)
    
    # 初始化estimator以获取参考视图信息
    estimator = name2estimator[cfg['type']](cfg)
    ref_split = args.split_type
    estimator.build(ref_database, split_type=ref_split)
    
    # 获取物体中心
    object_center = get_object_center(ref_database)
    
    # 获取查询图像ID列表
    _, que_ids = get_database_split(que_database, args.split_type)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始处理 {len(que_ids)} 张图片...")
    print(f"输出目录: {output_dir}")
    
    # 处理每张图片
    for que_id in tqdm(que_ids):
        try:
            # 获取查询图像信息
            img = que_database.get_image(que_id)
            K = que_database.get_K(que_id)
            pose_gt = que_database.get_pose(que_id)
            
            # 获取参考视图信息
            ref_poses = estimator.ref_info['poses']  # [rfn, 3, 4]
            ref_Ks = estimator.ref_info['Ks']  # [rfn, 3, 3]
            
            # 计算真实框
            gt_position, gt_scale_r2q, gt_angle_r2q, gt_bbox = get_gt_bbox(
                pose_gt, K, ref_poses, ref_Ks, object_center, size=args.crop_size
            )
            
            # 裁剪图片
            cropped_img = crop_image_by_bbox(img, gt_bbox, size=args.crop_size)
            
            # 保存裁剪后的图片
            output_path = output_dir / f"{que_id}.jpg"
            imsave(str(output_path), cropped_img)
            
        except Exception as e:
            print(f"处理图片 {que_id} 时出错: {e}")
            continue
    
    print(f"完成！共处理 {len(que_ids)} 张图片，保存到 {output_dir}")


if __name__ == "__main__":
    main()

