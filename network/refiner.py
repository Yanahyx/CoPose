import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from dataset.database import NormalizedDatabase, normalize_pose, get_object_center, get_diameter, denormalize_pose
from network.operator import pose_apply_th, normalize_coords
from network.pretrain_models import VGGBNPretrainV3
from utils.base_utils import pose_inverse, project_points, color_map_forward, to_cuda, pose_compose
from utils.database_utils import look_at_crop, select_reference_img_ids_refinement, normalize_reference_views
from utils.pose_utils import let_me_look_at, compose_sim_pose, pose_sim_to_pose_rigid
from utils.imgs_info import imgs_info_to_torch
from network.vis_dino_encoder import VitExtractor


# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.sigmoid(self.conv1(x_cat))
        return x * x_att

# 特征融合模块
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        # 用于特征对齐
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 注意力增强
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, feat1, feat2):
        # 对齐特征维度
        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        
        # 确保特征图大小一致
        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        
        # 门控融合
        gate = self.gate(torch.cat([feat1, feat2], dim=1))
        fused = gate * feat1 + (1 - gate) * feat2
        
        # 注意力增强
        fused = self.channel_att(fused)
        fused = self.spatial_att(fused)
        
        return fused

class RefineFeatureNet(nn.Module):
    def __init__(self, \
                 norm_layer='instance',\
                 use_dino=True,  # 默认启用DINO
                 dino_model='dino_vits8',
                 upsample=False):

        super().__init__()
        if norm_layer == 'instance':
            norm = nn.InstanceNorm2d
        else:
            norm = nn.BatchNorm2d  # 增加BatchNorm作为选项
        
        self.upsample = upsample
        self.use_dino = use_dino
        self.dino_model = dino_model
        
        # 多尺度特征处理
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            norm(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            norm(128),
            ChannelAttention(128)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            norm(128),
            ChannelAttention(128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            norm(128),
            ChannelAttention(128)
        )
        
        # 骨干特征融合
        self.backbone_fusion = nn.Sequential(
            nn.Conv2d(128*3, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            SpatialAttention()
        )
        
        if self.upsample:
            self.down_sample = nn.Conv2d(in_channels=256, \
                                        out_channels=128, \
                                        kernel_size=1,\
                                        stride=1,\
                                        padding=0, \
                                        bias=True)
        
        # DINO特征提取和融合
        if self.use_dino:
            # 根据不同DINO模型选择合适的特征维度
            self.dino_dim = 384 if 's8' in dino_model or 's16' in dino_model else 768
            
            # 特征融合模块
            self.dino_fusion = FeatureFusionModule(256, self.dino_dim, 256)
            
            # 最终特征处理
            self.final_conv = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                norm(128),
                nn.ReLU(True),
                ChannelAttention(128)
            )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 初始化DINO特征提取器
        if self.use_dino:
            self.dino_extractor = VitExtractor(model_name=dino_model).eval()
            for param in self.dino_extractor.parameters():
                param.requires_grad = False
            self.dino_extractor.requires_grad_(False)
        
        # 初始化骨干网络
        self.backbone = VGGBNPretrainV3().eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.img_norm = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
    def forward(self, imgs):
        batch_size, _, h, w = imgs.shape
        
        # 可选的上采样
        if self.upsample:
            imgs = F.interpolate(imgs, size=(int(1.5*h), int(1.5*h)))
        
        # 保存原始图像用于DINO（如果启用）
        if self.use_dino:
            dino_imgs = imgs.clone()
        
        # 骨干网络特征提取
        imgs_normalized = self.img_norm(imgs)
        with torch.no_grad():
            x0, x1, x2 = self.backbone(imgs_normalized)
            # 特征归一化
            x0 = F.normalize(x0, dim=1)
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1)
        
        # 多尺度特征处理
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x1), scale_factor=2, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.conv2(x2), scale_factor=4, mode='bilinear', align_corners=False)
        
        # 融合骨干特征
        backbone_feats = torch.cat([x0, x1, x2], dim=1)
        backbone_feats = self.backbone_fusion(backbone_feats)
        
        # 如果启用DINO，提取并融合DINO特征
        if self.use_dino:
            # 准备DINO输入
            dino_input_size = 256  # DINO模型的标准输入大小
            dino_input = F.interpolate(dino_imgs, size=(dino_input_size, dino_input_size), mode='bilinear')
            
            # 提取DINO特征
            with torch.no_grad():
                dino_output = self.dino_extractor.get_vit_attn_feat(dino_input)
                dino_feat = dino_output['feat']  # [batch, seq_len, hidden_dim]
            
            # 重塑并插值到目标尺寸
            h_dino, w_dino = dino_input_size // 8, dino_input_size // 8  # 对于s8模型
            dino_feat = dino_feat[:, 1:, :]  # 移除CLS token
            dino_feat = dino_feat.permute(0, 2, 1).reshape(batch_size, self.dino_dim, h_dino, w_dino)
            
            # 调整到与骨干特征相同的空间尺寸
            dino_feat = F.interpolate(
                dino_feat, 
                size=backbone_feats.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 使用融合模块融合特征
            fused_features = self.dino_fusion(backbone_feats, dino_feat)
            
            # 最终特征处理
            features = self.final_conv(fused_features)
        else:
            # 如果不使用DINO，直接使用骨干特征
            features = backbone_feats
        
        # 如果启用上采样，进行下采样调整
        if self.upsample:
            features = self.down_sample(features)
        
        return features

class RefineVolumeEncodingNet(nn.Module):
    def __init__(self,norm_layer='no_norm'):
        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm3d
        else:
            raise NotImplementedError

        self.mean_embed = nn.Sequential(
            nn.Conv3d(128 * 2, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, 1, 1)
        )
        self.var_embed = nn.Sequential(
            nn.Conv3d(128, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, 1, 1)
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(64*2, 64, 3, 1, 1), # 32
            norm(64),
            nn.ReLU(True),
        ) # 32

        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1),
            norm(128),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),
            norm(128),
            nn.ReLU(True),
        ) # 16

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1),
            norm(256),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
        )  #8

        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 2, 1),
            norm(512),
            nn.ReLU(True),
            nn.Conv3d(512, 512, 3, 1, 1)
        )

    def forward(self, mean, var):
        x = torch.cat([self.mean_embed(mean),self.var_embed(var)],1)
        x = self.conv0(x)
        x = self.conv2(self.conv1(x))
        x = self.conv4(self.conv3(x))
        x = self.conv5(x)
        
        return x

def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)

class RefineRegressor(nn.Module):
    def __init__(self, upsample=False):
        super().__init__()
        if upsample:
            self.fc = nn.Sequential(   fc( int((1.5)**3*512 * 4**3) , 512), nn.Dropout(p=0.15), fc(512, 512))
        else:
            self.fc = nn.Sequential(fc(512 * 4**3, 512), fc(512, 512))
        self.fcr = nn.Linear(512,4)
        self.fct = nn.Linear(512,2)
        self.fcs = nn.Linear(512,1)

    def forward(self, x):
   
        x = self.fc(x)
        r = F.normalize(self.fcr(x),dim=1)
        t = self.fct(x)
        s = self.fcs(x)
        return r, t, s


from loguru import logger

class VolumeRefiner(nn.Module):
    default_cfg = {
        "refiner_sample_num": 32,
    }
    def __init__(self, cfg, upsample=False):
        self.cfg={**self.default_cfg, **cfg}
        super().__init__()
        
        self.use_dino = self.cfg.get("use_dino", False)  
        logger.debug( f"VolumeRefiner use_dino:{self.use_dino}" )
        self.upsample = upsample
        self.feature_net = RefineFeatureNet('instance', self.use_dino, upsample)
        self.volume_net = RefineVolumeEncodingNet('instance')
        self.regressor = RefineRegressor(upsample)
        
        # used in inference
        self.ref_database = None
        self.ref_ids = None

    @staticmethod
    def interpolate_volume_feats(feats, verts, projs, h_in, w_in):
        """
        @param feats: b,f,h,w 
        @param verts: b,sx,sy,sz,3
        @param projs: b,3,4 : project matric
        @param h_in:  int
        @param w_in:  int
        @return:
        """
        b, sx, sy, sz, _ = verts.shape
        b, f, h, w = feats.shape
        R, t = projs[:,:3,:3], projs[:,:3,3:] # b,3,3  b,3,1
        verts = verts.reshape(b,sx*sy*sz,3)
        verts = verts @ R.permute(0, 2, 1) + t.permute(0, 2, 1) #

        depth = verts[:, :, -1:]
        depth[depth < 1e-4] = 1e-4
        verts = verts[:, :, :2] / depth  # [b,sx*sy*sz,2]
        verts = normalize_coords(verts, h_in, w_in) # b,sx*sy*sz,2]
        verts = verts.reshape([b, sx, sy*sz, 2])
        volume_feats = F.grid_sample(feats, verts, mode='bilinear', align_corners=False) # b,f,sx,sy*sz
        return volume_feats.reshape(b, f, sx, sy, sz)


    def construct_feature_volume(self, que_imgs_info, ref_imgs_info, feature_extractor, sample_num):
        """_summary_

        Args:
            que_imgs_info (_type_): _description_
            ref_imgs_info (_type_): _description_
            feature_extractor (_type_): 特征提取器
            sample_num (_type_): 采样图片的个数

        Returns:
            _type_: _description_
        """
        # build a volume on the unit cube
        sn = sample_num
        device = que_imgs_info['imgs'].device
        vol_coords = torch.linspace(-1, 1, sample_num, dtype=torch.float32, device=device)
        vol_coords = torch.stack(torch.meshgrid(vol_coords,vol_coords,vol_coords),-1) # sn,sn,sn,3
        vol_coords = vol_coords.reshape(1,sn**3,3)

        # rotate volume to align with the input pose, but still in the object coordinate
        poses_in = que_imgs_info['poses_in'] # qn,3,4
    
        rotation = poses_in[:,:3,:3] # qn,3,3
        vol_coords = vol_coords @ rotation # qn,sn**3,3
        qn = poses_in.shape[0]
        vol_coords = vol_coords.reshape(qn, sn, sn, sn, 3)

        # project onto every reference view
        ref_poses = ref_imgs_info['poses'] # qn,rfn,3,4
        ref_Ks = ref_imgs_info['Ks'] # qn,rfn,3,3
        ref_proj = ref_Ks @ ref_poses # qn,rfn,3,4

        vol_feats_mean, vol_feats_std = [], []
        h_in, w_in = ref_imgs_info['imgs'].shape[-2:]

        for qi in range(qn):
            ref_feats = feature_extractor(ref_imgs_info['imgs'][qi]) # rfn,f,h,w
            rfn = ref_feats.shape[0]
            vol_coords_cur = vol_coords[qi:qi+1].repeat(rfn,1,1,1,1) # rfn,sx,sy,sz,3
            vol_feats = VolumeRefiner.interpolate_volume_feats(ref_feats, vol_coords_cur, ref_proj[qi], h_in, w_in)

            vol_feats_mean.append(torch.mean(vol_feats, 0))
            vol_feats_std.append(torch.std(vol_feats, 0))

        vol_feats_mean = torch.stack(vol_feats_mean, 0)
        vol_feats_std = torch.stack(vol_feats_std, 0)

        # project onto query view
        h_in, w_in = que_imgs_info['imgs'].shape[-2:]
        que_feats = feature_extractor(que_imgs_info['imgs']) # qn,f,h,w
        que_proj = que_imgs_info['Ks_in'] @ que_imgs_info['poses_in']
        vol_feats_in = VolumeRefiner.interpolate_volume_feats(que_feats, vol_coords, que_proj, h_in, w_in) # qn,f,sx,sy,sz

        return vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords

    def forward(self, data):
        is_inference = data['inference'] if 'inference' in data else False
        que_imgs_info = data['que_imgs_info'].copy()
        ref_imgs_info = data['ref_imgs_info'].copy()

        if self.upsample:
            refiner_sample_num = int(self.cfg['refiner_sample_num']*1.5) 
        else:
            refiner_sample_num = self.cfg['refiner_sample_num']

        vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = self.construct_feature_volume(
            que_imgs_info, ref_imgs_info, self.feature_net, refiner_sample_num) # qn,f,dn,h,w   qn,dn

        vol_feats = torch.cat([vol_feats_mean, vol_feats_in], 1)
        vol_feats = self.volume_net(vol_feats, vol_feats_std)
        vol_feats = vol_feats.flatten(1) # qn, f* 4**3
        rotation, offset, scale = self.regressor(vol_feats)
        outputs={'rotation': rotation, 'offset': offset, 'scale': scale}

        if not is_inference:
            # used in training not inference
            qn, sx, sy, sz, _ = vol_coords.shape
            grids = pose_apply_th(que_imgs_info['poses_in'], vol_coords.reshape(qn, sx * sy * sz, 3))
            outputs['grids'] = grids

        return outputs

    def load_ref_imgs(self,ref_database,ref_ids):
        self.ref_database = ref_database
        self.ref_ids = ref_ids

    def refine_que_imgs(self, que_img, que_K, in_pose, size=128, ref_num=6, ref_even=False):
        """
        @param que_img:  [h,w,3]
        @param que_K:    [3,3]
        @param in_pose:  [3,4]
        @param size:     int
        @param ref_num:  int
        @param ref_even: bool
        @return:
        """
        margin = 0.05
        ref_even_num = min(128,len(self.ref_ids))

        # normalize database and input pose
        ref_database = NormalizedDatabase(self.ref_database) # wrapper: object is in the unit sphere at origin
        in_pose = normalize_pose(in_pose, ref_database.scale, ref_database.offset)
        object_center = get_object_center(ref_database)
        object_diameter = get_diameter(ref_database)

        # warp the query image to look at the object w.r.t input pose
        _, new_f = let_me_look_at(in_pose, que_K, object_center)
        in_dist = np.linalg.norm(pose_inverse(in_pose)[:,3] - object_center)
        in_f = size * (1 - margin) / object_diameter * in_dist
        scale = in_f / new_f
        position = project_points(object_center[None], in_pose, que_K)[0][0]
        que_img_warp, que_K_warp, in_pose_warp, que_pose_rect, H = look_at_crop(
            que_img, que_K, in_pose, position, 0, scale, size, size)

        que_imgs_info = {
            'imgs': color_map_forward(que_img_warp).transpose([2,0,1]),  # 3,h,w
            'Ks_in': que_K_warp.astype(np.float32), # 3,3
            'poses_in': in_pose_warp.astype(np.float32), # 3,4
        }

        # print( que_imgs_info['imgs'].shape ,  que_imgs_info['Ks_in'].shape  , que_imgs_info['poses_in'].shape )

        # select reference views for refinement
        ref_ids = select_reference_img_ids_refinement(ref_database, object_center, self.ref_ids, \
                                                      in_pose_warp, ref_num, ref_even, ref_even_num)

        # normalize the reference images and align the in-plane orientation w.r.t input pose.
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = normalize_reference_views(
            ref_database, ref_ids, size, margin, True, in_pose_warp, que_K_warp)

        ref_imgs_info = {
            'imgs': color_map_forward(np.stack(ref_imgs, 0)).transpose([0, 3, 1, 2]),  # rfn,3,h,w
            'poses': np.stack(ref_poses, 0).astype(np.float32),
            'Ks': np.stack(ref_Ks, 0).astype(np.float32),
        }

        # print( ref_imgs_info['imgs'].shape ,  ref_imgs_info['poses'].shape  , ref_imgs_info['Ks'].shape )


        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))

        for k,v in que_imgs_info.items(): que_imgs_info[k] = v.unsqueeze(0)
        for k,v in ref_imgs_info.items(): ref_imgs_info[k] = v.unsqueeze(0)

        with torch.no_grad():
            outputs = self.forward({'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info, 'inference': True})
            quat = outputs['rotation'].detach().cpu().numpy()[0] # 4
            scale = 2**outputs['scale'].detach().cpu().numpy()[0] # 1
            offset = outputs['offset'].detach().cpu().numpy()[0] # 2

            # print("scale:", scale , "quat:", quat, "offset:", offset )

        # compose rotation/scale/offset into a similarity transformation matrix
        pose_sim = compose_sim_pose(scale, quat, offset, in_pose_warp, object_center)
        # convert the similarity transformation to the rigid transformation
        pose_pr = pose_sim_to_pose_rigid(pose_sim, in_pose_warp, que_K_warp, que_K_warp, object_center)
        # apply the pose residual
        pose_pr = pose_compose(pose_pr, pose_inverse(que_pose_rect))
        # convert back to original coordinate system (because we use NormalizedDatabase to wrap the input)
        pose_pr = denormalize_pose(pose_pr, ref_database.scale, ref_database.offset)
        return pose_pr
    

if __name__ == "__main__":
    from utils.base_utils import load_cfg
    cfg = "configs/refiner/refiner_pretrain.yaml"
    refiner_cfg = load_cfg(cfg)
    refiner = VolumeRefiner(refiner_cfg)
    refiner_sample_num = 32

    ref_imgs_info = {
        'imgs': torch.randn(6,3,128,128) , # rfn,3,h,w
        'poses': torch.randn(6, 3, 4),
        'Ks': torch.randn(6,3,3),
    }

    que_imgs_info = {
        'imgs': torch.randn(3,128,128),  # 3,h,w
        'Ks_in': torch.randn(3, 3), # 3,3
        'poses_in':  torch.randn(3, 4), # 3,4
    }

    for k,v in que_imgs_info.items(): que_imgs_info[k] = v.unsqueeze(0)
    for k,v in ref_imgs_info.items(): ref_imgs_info[k] = v.unsqueeze(0)

    # pose_pr = refiner.refine_que_imgs(que_img, que_K, pose_pr, size=128, ref_num=6, ref_even=True)
    vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = refiner.construct_feature_volume(
            que_imgs_info, ref_imgs_info, refiner.feature_net, refiner_sample_num)

    mock_data = torch.randn(6,3,128,128)
    net = RefineFeatureNet()
    out =  net(mock_data)
    print(out.shape)