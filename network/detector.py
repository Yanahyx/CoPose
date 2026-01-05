import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from network.pretrain_models import VGGBNPretrain
from utils.base_utils import color_map_forward, transformation_crop, to_cpu_numpy
from utils.bbox_utils import parse_bbox_from_scale_offset

def matrix_distance(x1, x2):
    N, C = x1.shape
    M, C = x2.shape

    inner = 2*torch.matmul(x1, x2.transpose(1, 0))
    xx1 = torch.sum(x1**2, dim=-1, keepdim=True)
    xx2 = torch.sum(x2**2, dim=-1, keepdim=True)
    pairwise_distance = xx1 - inner + xx2.transpose(1, 0)

    return pairwise_distance

class BaseDetector(nn.Module):
    def load_impl(self, ref_imgs):
        raise NotImplementedError

    def detect_impl(self, que_imgs):
        raise NotImplementedError

    def load(self, ref_imgs):
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0, 3, 1, 2).cuda()
        self.load_impl(ref_imgs)

    def detect(self, que_imgs):
        que_imgs = torch.from_numpy(color_map_forward(que_imgs)).permute(0, 3, 1, 2).cuda()
        return self.detect_impl(que_imgs) # 'scores' 'select_pr_offset' 'select_pr_scale'

    @staticmethod
    def parse_detect_results(results):
        """

        @param results: dict
            pool_ratio: int -- pn
            scores: qn,1,h/pn,w/pn
            select_pr_offset: qn,2,h/pn,w/pn
            select_pr_scale:  qn,1,h/pn,w/pn
            select_pr_angle:  qn,2,h/pn,w/pn # optional
        @return: all numpy ndarray
        """
        qn = results['scores'].shape[0]
        pool_ratio = results['pool_ratio']

        # max scores
        _, score_x, score_y = BaseDetector.get_select_index(results['scores']) # qn
        position = torch.stack([score_x, score_y], -1)  # qn,2

        # offset
        offset = results['select_pr_offset'][torch.arange(qn),:,score_y,score_x] # qn,2
        position = position + offset

        # to original coordinate
        position = (position + 0.5) * pool_ratio - 0.5 # qn,2

        # scale
        scale_r2q = results['select_pr_scale'][torch.arange(qn),0,score_y,score_x] # qn
        scale_r2q = 2**scale_r2q
        outputs = {'position': position.detach().cpu().numpy(), 'scale_r2q': scale_r2q.detach().cpu().numpy()}
        # rotation
        if 'select_pr_angle' in results:
            angle_r2q = results['select_pr_angle'][torch.arange(qn),:,score_y,score_x] # qn,2
            angle = torch.atan2(angle_r2q[:,1],angle_r2q[:,0])
            outputs['angle_r2q'] = angle.cpu().numpy() # qn
        return outputs

    @staticmethod
    def detect_results_to_bbox(dets, length):
        pos = dets['position'] # qn,2
        length = dets['scale_r2q'] * length # qn,
        length = length[:,None]
        begin = pos - length/2
        return np.concatenate([begin,length,length],1)

    @staticmethod
    def detect_results_to_image_region(imgs, dets, region_len):
        qn = len(imgs)
        img_regions = []
        for qi in range(qn):
            pos = dets['position'][qi]; scl_r2q = dets['scale_r2q'][qi]
            ang_r2q = dets['angle_r2q'][qi] if 'anlge_r2q' in dets else 0
            img = imgs[qi]
            img_region, _ = transformation_crop(img, pos, 1/scl_r2q, -ang_r2q, region_len)
            img_regions.append(img_region)
        return img_regions

    @staticmethod
    def get_select_index(scores):
        """
        @param scores: qn,rfn or 1,hq,wq
        @return: qn
        """
        qn, rfn, hq, wq = scores.shape

        select_id = torch.argmax(scores.flatten(1), 1)
        select_ref_id = select_id // (hq * wq)
        select_h_id = (select_id - select_ref_id * hq * wq) // wq
        select_w_id = select_id - select_ref_id * hq * wq - select_h_id * wq
        return select_ref_id, select_w_id, select_h_id

    #    select_indices = torch.topk(scores.flatten(1), k=16, dim=-1)[1]
    #    return select_indices

    @staticmethod
    def parse_detection(scores, scales, offsets, pool_ratio):
        """

        @param scores:    qn,1,h/8,w/8
        @param scales:    qn,1,h/8,w/8
        @param offsets:   qn,2,h/8,w/8
        @param pool_ratio:int
        @return: position in x_cur
        """
        qn, _, _, _ = offsets.shape

        _, score_x, score_y = BaseDetector.get_select_index(scores) # qn
        positions = torch.stack([score_x, score_y], -1)  # qn,2

        offset = offsets[torch.arange(qn),:,score_y,score_x] # qn,2
        positions = positions + offset

        # to original coordinate
        positions = (positions + 0.5) * pool_ratio - 0.5 # qn,2

        # scale
        scales = scales[torch.arange(qn),0,score_y,score_x] # qn
        scales = 2**scales
        return positions, scales # [qn,2] [qn]

def disable_bn_grad(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

def disable_bn_track(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

class Detector(BaseDetector):
    default_cfg={
        "vgg_score_stats": [[36.264317,13.151907],[13910.291,5345.965],[829.70807,387.98788]],
        "vgg_score_max": 10,

        "detection_scales": [-1.0,-0.5,0.0,0.5],
        "train_feats": False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__()
        self.backbone = VGGBNPretrain()
        if self.cfg["train_feats"]:
            # disable BN training only
            disable_bn_grad(self.backbone)
        else:
            for para in self.backbone.parameters():
                para.requires_grad = False

        self.pool_ratio = 8
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.embed_0 = nn.Sequential(nn.InstanceNorm2d(512, affine=False), nn.Conv2d(512, 128, kernel_size=1))
        self.embed_1 = nn.Sequential(nn.InstanceNorm2d(512, affine=False), nn.Conv2d(512, 128, kernel_size=1))
        self.embed_2 = nn.Sequential(nn.InstanceNorm2d(512, affine=False), nn.Conv2d(512, 128, kernel_size=1))

        self.conv_num = 6
        self.ref_num = self.cfg["train_dataset_cfg"]["reference_num"]
        self.knn = 4
        self.d = 64

        self.score_norm = nn.InstanceNorm2d(3*self.conv_num, affine=False)

        self.scale_conv = nn.Sequential(
            nn.Conv3d(3*self.conv_num*self.knn, self.d, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(self.d,self.d,1,1),
        )
        self.score_conv = nn.Sequential(
            nn.Conv3d(3*self.knn, self.d, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(self.d,self.d,1,1),
        )
        self.scale_predict=nn.Sequential(
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,1,3,1,1),
        )
        self.offset_predict=nn.Sequential(
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,2,3,1,1),
        )
        self.score_predict=nn.Sequential(
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,self.d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d,1,3,1,1),
        )

    def extract_feats(self, imgs):
        imgs = self.img_norm(imgs)

        if self.cfg['train_feats']:
            disable_bn_track(self.backbone)
            x0, x1, x2 = self.backbone(imgs)
        else:
            self.backbone.eval()
            with torch.no_grad():
                x0, x1, x2 = self.backbone(imgs)
        return x0, x1, x2

    def load_impl(self, ref_imgs):
        # resize to 120,120
        ref_imgs = F.interpolate(ref_imgs,size=(120,120))

        ref_x0, ref_x1, ref_x2 = self.extract_feats(ref_imgs)

        self.ref_center_feats = (ref_x0, ref_x1, ref_x2)

        rfn, _, h, w = ref_imgs.shape
        self.ref_shape = [h, w]

    def feat_resize(self, x, H_new, W_new):
        H, W = x.shape[-2:]
        if H != H_new or W != W_new:
            x = F.pad(x, pad=(0, W_new-W, 0, H_new-H))
        return x

    def correlation_dilated(self, que_x, ref_x, size_offset, dilation):
        Bq, C, Hq, Wq = que_x.shape
        Br, C, Hr, Wr = ref_x.shape

        Hr_new, Wr_new = Hr+size_offset, Wr+size_offset
        if size_offset != 0:
            ref_x_resize = F.interpolate(ref_x, size=(Hr_new, Wr_new), mode='bilinear', align_corners=False)
        else:
            ref_x_resize = ref_x
        ref_normalized = F.normalize(ref_x_resize, p=2, dim=1)

        ks = Hr_new + (Hr_new-1) * (dilation-1)
        score = F.conv2d(que_x, ref_normalized, stride=1, padding=int((ks-1) / 2), dilation=dilation) / Hr_new
        score = self.feat_resize(score, Hq, Wq)

        return score

    def correlation_pooling(self, que_x, ref_x, size_offset):
        Bq, C, Hq, Wq = que_x.shape
        Br, C, Hr, Wr = ref_x.shape

        Hr_new, Wr_new = Hr+size_offset, Wr+size_offset
        if size_offset != 0:
            ref_x_resize = F.adaptive_avg_pool2d(ref_x, (Hr_new, Wr_new))
        else:
            ref_x_resize = ref_x

        ref_normalized = F.normalize(ref_x_resize, p=2, dim=1)

        score = F.conv2d(que_x, ref_normalized, stride=1, padding=int((Hr_new-1) / 2)) / Hr_new
        score = self.feat_resize(score, Hq, Wq)

        return score

    def finer_grained_correlaion_2(self, que_x, ref_x, conv_func, scale_factor):
        que_x = conv_func(que_x)
        ref_x = conv_func(ref_x)

        que_x = F.normalize(que_x, p=2, dim=1)

        score0 = self.correlation_pooling(que_x, ref_x, size_offset=-2) #1
        score1 = self.correlation_pooling(que_x, ref_x, size_offset=-1) #2
        score2 = self.correlation_pooling(que_x, ref_x, size_offset=0) #3
        score3 = self.correlation_dilated(que_x, ref_x, size_offset=-1, dilation=2) #3
        score4 = self.correlation_dilated(que_x, ref_x, size_offset=0, dilation=2) #5
        score5 = self.correlation_dilated(que_x, ref_x, size_offset=1, dilation=2) #7

        scores = torch.stack([score0, score1, score2, score3, score4, score5], dim=2)

        qn, rfn, cn, h, w = scores.shape
        if scale_factor is not None:
            scores = scores.reshape(qn*rfn, cn, h, w)
            scores = F.interpolate(scores, scale_factor=scale_factor)
            h, w = scores.shape[-2:]
            scores = scores.reshape(qn, rfn, cn, h, w)
        scores = scores.transpose(1, 2)

        return scores

    def finer_grained_correlaion_1(self, que_x, ref_x, conv_func, scale_factor):
        que_x = conv_func(que_x)
        ref_x = conv_func(ref_x)

        que_x = F.normalize(que_x, p=2, dim=1)

        score0 = self.correlation_pooling(que_x, ref_x, size_offset=-4) #3
        score1 = self.correlation_pooling(que_x, ref_x, size_offset=-2) #5
        score2 = self.correlation_pooling(que_x, ref_x, size_offset=0) #7
        score3 = self.correlation_dilated(que_x, ref_x, size_offset=-2, dilation=2) # 9
        score4 = self.correlation_dilated(que_x, ref_x, size_offset=-1, dilation=2) # 11
        score5 = self.correlation_dilated(que_x, ref_x, size_offset=0, dilation=2) # 13

        scores = torch.stack([score0, score1, score2, score3, score4, score5], dim=2)

        qn, rfn, cn, h, w = scores.shape
        if scale_factor is not None:
            scores = scores.reshape(qn*rfn, cn, h, w)
            scores = F.interpolate(scores, scale_factor=scale_factor)
            h, w = scores.shape[-2:]
            scores = scores.reshape(qn, rfn, cn, h, w)
        scores = scores.transpose(1, 2)

        return scores

    def finer_grained_correlaion_0(self, que_x, ref_x, conv_func, scale_factor):
        que_x = conv_func(que_x)
        ref_x = conv_func(ref_x)

        que_x = F.normalize(que_x, p=2, dim=1)

        score0 = self.correlation_pooling(que_x, ref_x, size_offset=-8) #7
        score1 = self.correlation_pooling(que_x, ref_x, size_offset=-4) #11
        score2 = self.correlation_pooling(que_x, ref_x, size_offset=0) #15
        score3 = self.correlation_dilated(que_x, ref_x, size_offset=-4, dilation=2) #21
        score4 = self.correlation_dilated(que_x, ref_x, size_offset=-2, dilation=2) #25
        score5 = self.correlation_dilated(que_x, ref_x, size_offset=0, dilation=2) #29

        scores = torch.stack([score0, score1, score2, score3, score4, score5], dim=2)

        qn, rfn, cn, h, w = scores.shape
        if scale_factor is not None:
            scores = scores.reshape(qn*rfn, cn, h, w)
            scores = F.interpolate(scores, scale_factor=scale_factor)
            h, w = scores.shape[-2:]
            scores = scores.reshape(qn, rfn, cn, h, w)
        scores = scores.transpose(1, 2)

        return scores

    def get_scores(self, que_imgs):
        que_x0, que_x1, que_x2 = self.extract_feats(que_imgs)
        ref_x0, ref_x1, ref_x2 = self.ref_center_feats # rfn,f,hr,wr

        score0 = self.finer_grained_correlaion_0(que_x0, ref_x0, self.embed_0, scale_factor=None)
        score1 = self.finer_grained_correlaion_1(que_x1, ref_x1, self.embed_1, scale_factor=2)
        score2 = self.finer_grained_correlaion_2(que_x2, ref_x2, self.embed_2, scale_factor=4)

        scores = torch.stack([score0, score1, score2], dim=1)

        return scores

    def detect_impl(self, que_imgs):
        qn, _, hq, wq = que_imgs.shape
        hs, ws = hq // 8, wq // 8

        ht=(hq//32+1)*32 if hq%32!=0 else hq
        wt=(wq//32+1)*32 if wq%32!=0 else wq
        if ht != hq or wt != wq:
            que_imgs = F.interpolate(que_imgs, size=(ht,wt), mode='bilinear', align_corners=False)

        scores = self.get_scores(que_imgs)
        qn, _, cn, rfn, hcs, wcs = scores.shape
        if ht != hq or wt != wq:
            scores = F.interpolate(scores.reshape(qn, -1, hcs, wcs), size=(hs, ws), mode='bilinear', align_corners=False).reshape(qn, _, cn, rfn, hs, ws)

        qn, fn, cn, rfn, hs, ws = scores.shape

        scores = scores.transpose(1, 3).reshape(qn*rfn, cn*fn, hs, ws)
        scores = self.score_norm(scores).reshape(qn, rfn, cn, fn, hs, ws)
        scores = scores.transpose(1, 3)
        scores = scores.flatten(4)

        ### scale-aware
        scores_feats_ss = scores.reshape(qn, -1, rfn, hs*ws) #qn, d, cn, rfn, hs*ws
        scores_feats_ss = scores_feats_ss[:, :, self.knn_indices.reshape(-1)]
        scores_feats_ss = scores_feats_ss.reshape(qn, -1, rfn, self.knn, hs*ws)
        scores_feats_ss = scores_feats_ss.transpose(2, 3).reshape(qn, -1, rfn, hs, ws)

        scores_feats_ss = self.scale_conv(scores_feats_ss)
        scores_feats_ss = scores_feats_ss.max(dim=2)[0]
        select_scale = self.scale_predict(scores_feats_ss)

        ### scale-robust
        max_score = scores.max(dim=-1, keepdim=True)[0]
        weights = (max_score - scores).mean(dim=-1)
        weights = torch.softmax(weights, dim=2)
        scores_feats_rb = (scores * weights[..., None]).sum(dim=2) #qn, fn, rfn, hs*ws
        scores_feats_rb = scores_feats_rb[:, :, self.knn_indices.reshape(-1)]
        scores_feats_rb = scores_feats_rb.reshape(qn, fn, rfn, self.knn, hs*ws)
        scores_feats_rb = scores_feats_rb.transpose(2, 3).reshape(qn, -1, rfn, hs, ws)

        scores_feats_rb = self.score_conv(scores_feats_rb)
        scores_feats_rb = scores_feats_rb.max(dim=2)[0]

        select_offset = self.offset_predict(scores_feats_rb)
        scores = self.score_predict(scores_feats_rb)

        _, select_w_id, select_h_id = self.get_select_index(scores)
        que_select_id = torch.stack([select_w_id, select_h_id],1) # qn, 2

        outputs = {'scores': scores, 'que_select_id': que_select_id, 'pool_ratio': self.pool_ratio, 'select_pr_offset': select_offset, 'select_pr_scale': select_scale,}

        return outputs

    def forward(self, data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()

        self.que_cen = que_imgs_info["cens"]
        ref_imgs = ref_imgs_info['imgs']

        self.ref_geo_dis = self.ref_dis_estimation(ref_imgs_info['poses'])
        self.ref_knn(self.ref_geo_dis, k=self.knn)

        self.load_impl(ref_imgs)
        outputs = self.detect_impl(que_imgs_info['imgs'])
        return outputs

    def ref_dis_estimation(self, ref_poses):
        rfn = ref_poses.shape[0]
        ref_Rs = ref_poses[:, :, :3].reshape(rfn, 9)
        sim = (torch.sum(ref_Rs[:, None] * ref_Rs[None], dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi
        return geo_dis

    def ref_feat_dis_estimation(self, ref_feat):
        rfn, C, Hr, Wr = ref_feat.shape

        x = ref_feat.transpose(1, 0).reshape(C, rfn*Hr*Wr)

        inner = 2*torch.matmul(x.transpose(1, 0), x)
        xx = torch.sum(x**2, dim=0, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(1, 0)

        pairwise_distance = pairwise_distance.reshape(rfn, Hr, Wr, rfn, Hr*Wr)
        ref_nn_indices = pairwise_distance.min(dim=-1)[1] ## rfn, Hr, Wr, rfn

        return ref_nn_indices

    def ref_knn(self, geo_dis, k):
        self.knn_dis, self.knn_indices = torch.topk(geo_dis, k=k, dim=-1, largest=False)

    def load_ref_imgs(self, ref_imgs):
        """
        @param ref_imgs: [an,rfn,h,w,3] in numpy
        @return:
        """
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0,3,1,2) # rfn,3,h,w
        ref_imgs = ref_imgs.cuda()

        rfn, _, h, w = ref_imgs.shape
        self.load_impl(ref_imgs)

    def detect_que_imgs(self, que_imgs):
        """
        @param que_imgs: [qn,h,w,3]
        @return:
        """
        que_imgs = torch.from_numpy(color_map_forward(que_imgs)).permute(0,3,1,2).cuda()
        qn, _, h, w = que_imgs.shape

        outputs = self.detect_impl(que_imgs)
        scores = outputs['scores'].detach()
        scales_all=outputs['select_pr_scale'].detach()
        positions, scales = self.parse_detection(
            outputs['scores'].detach(), outputs['select_pr_scale'].detach(),
            outputs['select_pr_offset'].detach(), self.pool_ratio)
        detection_results = {'positions': positions, 'scales': scales, 'scores': scores, 'scales_all': scales_all}
        detection_results = to_cpu_numpy(detection_results)
        return detection_results