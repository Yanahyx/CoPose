import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
from copy import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
import numpy as np
from skimage.io import imread, imsave
import pickle
import random
import cv2
import os
from tqdm import tqdm, trange
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset.database import parse_database_name, get_database_split, get_ref_point_cloud, get_diameter, get_object_center
from detector import name2estimator
from utils.base_utils import load_cfg, save_pickle, read_pickle, project_points, transformation_crop
from utils.database_utils import compute_normalized_view_correlation
from utils.draw_utils import draw_bbox, concat_images_list, draw_bbox_3d, pts_range_to_bbox_pts
from utils.pose_utils import compute_metrics_impl, scale_rotation_difference_from_cameras
from utils.bbox_utils import bboxes_iou

name2ID = {'benchvise': 2, 'cam': 4, 'cat': 6, 'driller': 8, 'duck': 9}

def que_rescale_experiments(que_imgs, scale_ratios, que_id):
    scale_factor = scale_ratios[int(que_id)]
    h, w, _ = que_imgs.shape
    que_imgs = cv2.resize(que_imgs, (int(w*scale_factor), int(h*scale_factor)))
    return que_imgs, scale_factor

def get_gt_info(que_pose, que_K, render_poses, render_Ks, object_center):
    gt_corr = compute_normalized_view_correlation(que_pose[None], render_poses, object_center, False)
    gt_ref_idx = np.argmax(gt_corr[0])
    gt_scale_r2q, gt_angle_r2q = scale_rotation_difference_from_cameras(
        render_poses[gt_ref_idx][None], que_pose[None], render_Ks[gt_ref_idx][None], que_K[None], object_center)
    gt_scale_r2q, gt_angle_r2q = gt_scale_r2q[0], gt_angle_r2q[0]
    gt_position = project_points(object_center[None], que_pose, que_K)[0][0]
    size = 128
    gt_bbox = np.concatenate([gt_position - size / 2 * gt_scale_r2q, np.full(2, size) * gt_scale_r2q])
    return gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_corr[0]

def visualize_intermediate_results(img, K, inter_results, ref_info, object_bbox_3d, object_center=None, pose_gt=None):
    ref_imgs = ref_info['ref_imgs']  # an,rfn,h,w,3

    if pose_gt is not None:
        gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_scores = \
            get_gt_info(pose_gt, K, ref_info['poses'], ref_info['Ks'], object_center)

    output_imgs = []
    # visualize detection
    det_scale_r2q = inter_results['det_scale_r2q']
    det_position = inter_results['det_position']
    det_que_img = inter_results['det_que_img']
    size = det_que_img.shape[0]

    ## visualize
    pr_bbox = np.concatenate([det_position - size / 2 * det_scale_r2q, np.full(2, size) * det_scale_r2q])
    bbox_img = img
    if pose_gt is not None: bbox_img = draw_bbox(bbox_img, gt_bbox, color=(0, 255, 0), thickness=4)
    bbox_img = draw_bbox(bbox_img, pr_bbox, color=(0, 0, 255), thickness=4)
    output_imgs.append(bbox_img)

    pr_bbox = torch.tensor([pr_bbox[0], pr_bbox[1], pr_bbox[0]+pr_bbox[2], pr_bbox[1]+pr_bbox[3]]).float()
    gt_bbox = torch.tensor([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]]).float()

    iou = bboxes_iou(gt_bbox[None], pr_bbox[None], False)

    return concat_images_list(*output_imgs), iou, pr_bbox, gt_bbox, gt_scale_r2q

def main(args):
    cfg = load_cfg(args.cfg)
    cfg["det_ref_view_num"] = args.ref_num
    object_name = args.object_name
    if object_name.startswith('linemod'):
        ref_database_name = que_database_name = object_name
        que_split = 'linemod_test'
    elif object_name.startswith('genmop'):
        ref_database_name = object_name+'-ref'
        que_database_name = object_name+'-test'
        que_split = 'all'
    else:
        raise NotImplementedError

    ref_database = parse_database_name(ref_database_name)
    estimator = name2estimator[cfg['type']](cfg)
    ref_split = que_split if args.split_type is None else args.split_type
    estimator.build(ref_database, split_type=ref_split)

    que_database = parse_database_name(que_database_name)
    _, que_ids = get_database_split(que_database, que_split)
    object_pts = get_ref_point_cloud(ref_database)
    object_center = get_object_center(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    est_name = estimator.cfg["name"] # + f'-{args.render_pose_name}'
    est_name = est_name + args.split_type if args.split_type is not None else est_name
    Path(f'../data/exp/eval/poses/{object_name}').mkdir(exist_ok=True,parents=True)
    Path(f'../data/exp/vis_inter/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
    Path(f'../data/exp/vis_final/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
    Path(f'../data/exp/vis_heatmap/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)

    iou_list = []
    gt_scale_r2q_list = []
    metric = MeanAveragePrecision()

    for idx, que_id in enumerate(tqdm(que_ids)):
        # estimate pose
        img = que_database.get_image(que_id)
        K = que_database.get_K(que_id)

        pose_pr, inter_results = estimator.predict(img, K, save_heatmap_path=f'../data/exp/vis_heatmap/{est_name}/{object_name}/{que_id}-heatmap.jpg')

        pose_gt = que_database.get_pose(que_id)

        inter_img, iou, pr_bbox, gt_bbox, gt_scale_r2q = visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d, object_center, pose_gt)
        imsave(f'../data/exp/vis_inter/{est_name}/{object_name}/{que_id}-inter.jpg', inter_img)
        preds = [dict(boxes=pr_bbox[None], scores=pr_bbox.new_ones([1]), labels=pr_bbox.new_zeros([1]).long())]
        target = [dict(boxes=gt_bbox[None], labels=gt_bbox.new_zeros([1]).long())]
        metric.update(preds, target)

        iou_list.append(iou.item())
        gt_scale_r2q_list.append(gt_scale_r2q)

    mAP = metric.compute()["map"]
    metric.reset()

    iou_list = np.asarray(iou_list).mean()
    gt_scale_r2q_list = np.asarray(gt_scale_r2q_list).mean()
    print("%s --- mean GT scale: %.4f --- Mean IoU: %.4f --- mAP: %.4f \n" % (object_name, gt_scale_r2q_list, iou_list, mAP))
    with open(f'../data/exp/performance.txt', 'a') as f:
        f.write('%s --- mean GT scale: %.4f --- Mean IoU: %.4f --- mAP: %.4f \n' % (object_name, gt_scale_r2q_list, iou_list, mAP))
    f.close()
    return iou_list, mAP

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=False)
    parser.add_argument('--object_name', type=str, default='warrior')
    parser.add_argument('--scale_ratio', type=float, default=1)
    parser.add_argument('--ref_num', type=int, default=32)
    parser.add_argument('--eval_only', action='store_true', dest='eval_only', default=False)
    parser.add_argument('--symmetric', action='store_true', dest='symmetric', default=False)
    parser.add_argument('--split_type', type=str, default=None)
    args = parser.parse_args()

    args.cfg = 'configs/cas6d_train.yaml'
    ## LINEMOD
    iou_list, mAP_list = [], []
    for obj in ['cat', 'benchvise', 'cam', 'driller', 'duck']:
        args.object_name = 'linemod/' + obj
        iou, mAP = main(args)
        iou_list.append(iou)
        mAP_list.append(mAP)

    iou_list = np.asarray(iou_list).mean()
    mAP_list = np.asarray(mAP_list).mean()
    print("Average --- Mean IoU: %.4f --- mAP: %.4f \n" % (iou_list, mAP_list))
    with open(f'../data/exp/performance.txt', 'a') as f:
        f.write('Average --- Mean IoU: %.4f --- mAP: %.4f \n' % (iou_list, mAP_list))
    f.close()
    
    # GenMOP
    iou_list, mAP_list = [], []
    for obj in ['chair', 'plug_en', 'piggy', 'scissors', 'tformer']:
        args.object_name = 'genmop/' + obj
        iou, mAP = main(args)
        iou_list.append(iou)
        mAP_list.append(mAP)

    iou_list = np.asarray(iou_list).mean()
    mAP_list = np.asarray(mAP_list).mean()

    print("Average --- Mean IoU: %.4f --- mAP: %.4f \n" % (iou_list, mAP_list))
    with open(f'../data/exp/performance.txt', 'a') as f:
        f.write('Average --- Mean IoU: %.4f --- mAP: %.4f \n' % (iou_list, mAP_list))
    f.close()


