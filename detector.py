import cv2
import numpy as np
import torch
import os
import logging
from pathlib import Path
from skimage.io import imsave

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataset.database import BaseDatabase, get_database_split, get_object_vert, get_object_center

from network import name2network
from utils.base_utils import load_cfg, transformation_offset_2d, transformation_scale_2d, \
    transformation_compose_2d, transformation_crop, transformation_rotation_2d
from utils.database_utils import select_reference_img_ids_fps, normalize_reference_views
from utils.pose_utils import estimate_pose_from_similarity_transform_compose

def compute_similarity_transform(pts0, pts1):
    """
    @param pts0:
    @param pts1:
    @return: sR @ p0 + t = p1
    """
    ref_c = np.mean(pts0, 0)
    que_c = np.mean(pts1, 0)
    ref_d = pts0 - ref_c[None, :]
    que_d = pts1 - que_c[None, :]
    scale = np.mean(np.linalg.norm(que_d,2,1)) / np.mean(np.linalg.norm(ref_d,2,1))
    ref_d_ = ref_d * scale
    U, S, VT = np.linalg.svd(ref_d_.T @ que_d)
    rotation = VT.T @ U.T
    offset = - scale * (rotation @ ref_c) + que_c
    return scale, rotation, offset

def compute_similarity_transform_batch(pts0, pts1):
    """
    @param pts0:
    @param pts1:
    @return: sR @ p0 + t = p1
    """
    c0 = np.mean(pts0, 1) # n, 2
    c1 = np.mean(pts1, 1) # n, 2
    d0 = pts0 - c0[:, None, :]
    d1 = pts1 - c1[:, None, :]
    scale = np.mean(np.linalg.norm(d1,2,2,keepdims=True),1,keepdims=True) / \
            np.mean(np.linalg.norm(d0,2,2,keepdims=True),1,keepdims=True) # n,1,1
    d0_ = d0 * scale # n,k,2
    U, S, VT = np.linalg.svd(d0_.transpose([0,2,1]) @ d1) # n,2,2
    rotation = VT.transpose([0,2,1]) @ U.transpose([0,2,1]) # n,2,2
    offset = - scale * (rotation @ c0[:,:,None]) + c1[:,:,None]
    return scale, rotation, offset # [n,1,1] [n,2,2] [n,2,1]

def compute_inlier_mask(scale, rotation, offset, corr, thresh):
    x0=corr[None, :, :2] # [1,k,2]
    x1=corr[None, :, 2:] # [1,k,2]
    x1_ = scale * (x0 @ rotation.transpose([0,2,1])) + offset.transpose([0,2,1])
    mask = np.linalg.norm(x1-x1_,2,2) < thresh
    return mask

def ransac_similarity_transform(corr):
    n, _ = corr.shape
    batch_size=4096
    bad_seed_thresh=4
    inlier_thresh=5
    best_inlier, best_mask = 0, None
    iter_num = 0
    confidence = 0.99
    while True:
        idx = np.random.randint(0,n,[batch_size,2])
        seed0 = corr[idx[:,0]] # b,4
        seed1 = corr[idx[:,1]] # b,4
        bad_mask = np.linalg.norm(seed0 - seed1, 2, 1) < bad_seed_thresh
        seed0 = seed0[~bad_mask]
        seed1 = seed1[~bad_mask]
        seed = np.stack([seed0,seed1],1)
        scale, rotation, offset = compute_similarity_transform_batch(seed[:,:,:2],seed[:,:,2:]) #
        mask = compute_inlier_mask(scale,rotation,offset,corr,inlier_thresh) # b,n
        inlier_num = np.sum(mask,1)
        if np.max(inlier_num) >= best_inlier:
            best_mask = mask[np.argmax(inlier_num)]
        iter_num += seed.shape[0]
        inlier_ratio = np.mean(best_mask)
        if 1-(1-inlier_ratio**2)**iter_num > confidence:
            break

    inlier_corr=corr[best_mask]
    scale, rotation, offset = compute_similarity_transform_batch(inlier_corr[None,:,:2],inlier_corr[None,:,2:])
    scale, rotation, offset = scale[0,0,0], rotation[0], offset[0,:,0]
    return scale, rotation, offset, best_mask

def compose_similarity_transform(scale, rotation, offset):
    M = transformation_scale_2d(scale)
    M = transformation_compose_2d(M, np.concatenate([rotation, np.zeros([2, 1])], 1).astype(np.float32))
    M = transformation_compose_2d(M, transformation_offset_2d(offset[0], offset[1]))
    return M


class Gen6DEstimator:
    default_cfg={
        'ref_resolution': 128,
        "ref_view_num": 64,
        "det_ref_view_num": 32,
        'detector': None,

    }
    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.ref_info={}

        self.detector = self._load_module(self.cfg['detector'])

    @staticmethod
    def _load_module(cfg):
        refiner_cfg = load_cfg(cfg)
        refiner = name2network[refiner_cfg['network']](refiner_cfg)

        state_dict = torch.load(f'../data/checkpoints/{refiner_cfg["name"]}/model_best.pth')

        refiner.load_state_dict(state_dict['network_state_dict'])
        refiner.cuda().eval()
        return refiner

    # def _check(self, ref_point_cloud, ref_imgs, ref_poses, ref_Ks, ref_ids, database):
    #     rfn = ref_imgs.shape[0]
    #     output_imgs = []
    #     for rfi in range(rfn):
    #         pts2d, _ = project_points(ref_point_cloud, ref_poses[rfi], ref_Ks[rfi])
    #         kps_img = draw_keypoints(ref_imgs[rfi],pts2d)//2+ref_imgs[rfi]//2
    #         img_raw = database.get_image(ref_ids[rfi])
    #         output_imgs.append(concat_images_list(img_raw,kps_img,vert=True))
    #
    #     imsave(f'data/vis_val/check.jpg',concat_images_list(*output_imgs))
    #     import ipdb; ipdb.set_trace()

    def build(self, database: BaseDatabase, split_type: str):
        object_center = get_object_center(database)
        object_vert = get_object_vert(database)
        ref_ids_all, _ = get_database_split(database, split_type)

        # use fps to select reference images for detection and selection
        ref_ids = select_reference_img_ids_fps(database, ref_ids_all, self.cfg['ref_view_num'])
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = \
            normalize_reference_views(database, ref_ids, self.cfg['ref_resolution'], 0.05)

        # in-plane rotation for viewpoint selection
        rfn, h, w, _ = ref_imgs.shape
        ref_imgs_rots = []
        angles = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
        for angle in angles:
            M = transformation_offset_2d(-w/2,-h/2)
            M = transformation_compose_2d(M, transformation_rotation_2d(angle))
            M = transformation_compose_2d(M, transformation_offset_2d(w/2,h/2))
            H_ = np.identity(3).astype(np.float32)
            H_[:2,:3] = M
            ref_imgs_rot = []
            for rfi in range(rfn):
                H_new = H_ @ ref_Hs[rfi]
                ref_imgs_rot.append(cv2.warpPerspective(database.get_image(ref_ids[rfi]), H_new, (w,h), flags=cv2.INTER_LINEAR))
            ref_imgs_rots.append(np.stack(ref_imgs_rot, 0))
        ref_imgs_rots = np.stack(ref_imgs_rots, 0) # an,rfn,h,w,3

        self.detector.load_ref_imgs(ref_imgs[:self.cfg['det_ref_view_num']])
        self.ref_info={'imgs': ref_imgs, 'ref_imgs': ref_imgs_rots, 'masks': ref_masks, 'Ks': ref_Ks, 'poses': ref_poses, 'center': object_center}

        self.ref_geo_dis = self.detector.ref_dis_estimation(torch.from_numpy(ref_poses[:self.cfg['det_ref_view_num']]).float().cuda())
        self.detector.ref_knn(self.ref_geo_dis, k=self.detector.knn)

    def plot_score_heatmap(self, score, save_path=None, scale_factor=4):
        """
        将相关性分数绘制成热力图并保存
        @param score: 分数数组，shape为[1,60,80]或其他维度
        @param save_path: 保存路径，如果为None则不保存
        @param scale_factor: 放大倍数，默认4倍（60x80 -> 240x320）
        @return: 热力图图像数组 (H, W, 3)
        """
        # 将score转换为numpy数组（如果是tensor）
        if torch.is_tensor(score):
            score = score.cpu().numpy()
        
        # 根据score的维度进行处理
        score_2d = None
        if score.ndim == 2:
            # 2D数组，直接使用
            score_2d = score
        elif score.ndim == 3:
            # 3D数组，如[1,60,80]，取第一个通道
            if score.shape[0] == 1:
                score_2d = score[0]  # [1,60,80] -> [60,80]
            else:
                score_2d = np.mean(score, axis=0)
        # 归一化到0-255范围
        score_min = score_2d.min()
        score_max = score_2d.max()
        if score_max > score_min:
            score_norm = ((score_2d - score_min) / (score_max - score_min) * 255).astype(np.uint8)
        else:
            score_norm = np.zeros_like(score_2d, dtype=np.uint8)
        
        # 应用热力图颜色映射 (使用蓝色系COLORMAP_COOL，从青到蓝)
        heatmap = cv2.applyColorMap(score_norm, cv2.COLORMAP_COOL)
        # 将BGR转换为RGB（cv2返回BGR，imsave需要RGB）
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 放大显示（60x80 -> 240x320，默认4倍）
        h, w = heatmap.shape[:2]
        if scale_factor > 1:
            heatmap = cv2.resize(heatmap, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST)
        
        if save_path is not None:
            # 确保目录存在（参考eval_detector.py的方式）
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            # 使用imsave保存，与eval_detector.py一致
            imsave(save_path, heatmap)   
        return heatmap

    def predict(self, que_img, que_K, pose_init=None, save_heatmap_path=None):
        inter_results={}

        if pose_init is None:
            # stage 1: detection
            with torch.no_grad():
                detection_outputs = self.detector.detect_que_imgs(que_img[None])
                position = detection_outputs['positions'][0]
                scale_r2q = detection_outputs['scales'][0]
                score = detection_outputs['scores'][0]
                que_img_, _ = transformation_crop(que_img, position, 1/scale_r2q, 0, self.cfg['ref_resolution'])  # h,w,3
                inter_results['det_position'] = position
                inter_results['det_scale_r2q'] = scale_r2q
                inter_results['det_que_img'] = que_img_
                inter_results['det_score'] = score
                scales_all = detection_outputs['scales_all'][0]
                inter_results['det_scales_all'] = scales_all
                
                # 绘制score热力图（如果提供了保存路径则保存）
                if save_heatmap_path is not None:
                    # 为score和scales_all生成不同的文件名
                    save_dir = Path(save_heatmap_path).parent
                    save_name = Path(save_heatmap_path).stem  # 不含扩展名的文件名
                    save_ext = Path(save_heatmap_path).suffix  # 扩展名
                    score_heatmap_path = str(save_dir / f'{save_name}_score{save_ext}')
                    scale_heatmap_path = str(save_dir / f'{save_name}_scale{save_ext}')
                else:
                    score_heatmap_path = None
                    scale_heatmap_path = None
                
                score_heatmap = self.plot_score_heatmap(score, save_path=score_heatmap_path)
                inter_results['det_heatmap'] = score_heatmap
                
                # 绘制scales_all热力图
                scale_heatmap = self.plot_score_heatmap(scales_all, save_path=scale_heatmap_path)
                inter_results['det_scale_heatmap'] = scale_heatmap
        return None, inter_results

name2estimator={
    'cas6d': Gen6DEstimator,
}