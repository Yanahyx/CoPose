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
from transformers import AutoImageProcessor, AutoModel


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

class RefineFeatureNet(nn.Module):
    def __init__(self, \
                 norm_layer='instance',\
                 use_dino=True,\
                 upsample=False):

        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            norm(64),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64*3, 128, 3, 1, 1),
            norm(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            norm(128),
        )
        self.upsample = upsample
        self.use_dino = use_dino

        if self.upsample:
            self.down_sample = nn.Conv2d(in_channels=128, \
                                        out_channels=64, \
                                        kernel_size=1,\
                                        stride=1,\
                                        padding=0, \
                                        bias=True)  

        if self.use_dino:
            # 改进方案：渐进式融合网络 + 特征对齐 + 门控机制
            # 方案1：渐进式降维融合 (896->512->256->128)，保留更多信息
            self.fuse_conv = nn.Sequential(
                # 第一层：896 -> 512，使用3x3卷积提取空间特征
                nn.Conv2d(in_channels=768+128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                norm(512),
                nn.ReLU(True),
                # 第二层：512 -> 256
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                norm(256),
                nn.ReLU(True),
                # 第三层：256 -> 128
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            )
            
            # 方案2：特征对齐 - 对DINO特征进行归一化，使其与VGG特征分布对齐
            # self.dino_proj = nn.Sequential(
            #     nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=True),
            #     norm(256),
            #     nn.ReLU(True),
            # )
            
            # 方案4：残差连接路径
            self.residual_proj = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        
        if self.use_dino:
            self.dino_processor = AutoImageProcessor.from_pretrained("../dinov3-vitb16-pretrain-lvd1689m")
            self.dino_model = AutoModel.from_pretrained(
                "../dinov3-vitb16-pretrain-lvd1689m",
                dtype=torch.float16,  # 改为float32提高数值稳定性
                device_map="auto",
                attn_implementation="sdpa"
            ).eval()
            for para in self.dino_model.parameters():
                para.requires_grad = False
            self.dino_model.requires_grad_(False)
            self.dino_patch_size = self.dino_model.config.patch_size
            self.dino_hidden_size = self.dino_model.config.hidden_size
            self.dino_num_register_tokens = self.dino_model.config.num_register_tokens 

        self.backbone = VGGBNPretrainV3().eval()
        for para in self.backbone.parameters():
            para.requires_grad = False
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
      
    def forward(self, imgs):
        _,_, h,w = imgs.shape
        if self.upsample:
            imgs = F.interpolate(imgs, size=(int(1.5*h), int(1.5*h)))

        if self.use_dino:
            dino_imgs = imgs.clone()
            
        imgs = self.img_norm(imgs) #[6, 3, 128, 128]
        self.backbone.eval()
        with torch.no_grad():
            x0, x1, x2 = self.backbone(imgs)
            x0 = F.normalize(x0, dim=1)
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1)
        x0 = self.conv0(x0) #[6, 64, 32, 32]
        x1 = F.interpolate(self.conv1(x1),scale_factor=2,mode='bilinear') #[6, 64, 32, 32]
        x2 = F.interpolate(self.conv2(x2),scale_factor=4,mode='bilinear') #[6, 64, 32, 32]
        x = torch.cat([x0,x1,x2],1) #[6, 192, 32, 32]
        x = self.conv_out(x)  #[6, 128, 32, 32]
        if self.use_dino:
            # -------------------------------------------------------- 
            # 使用dinov3处理，类似test.py中的方式
            inputs = self.dino_processor(images=dino_imgs, return_tensors="pt")
            # 计算patch数量
            img_height, img_width = inputs.pixel_values.shape[-2:]  # 6,3,224,24
            num_patches_height = img_height // self.dino_patch_size # 224/16=14
            num_patches_width = img_width // self.dino_patch_size # 224/16=14
            # 获取模型输出
            with torch.inference_mode():
                outputs = self.dino_model(**inputs)
            
            last_hidden_states = outputs.last_hidden_state  # [6, 1+4+196, 768]
            # 提取patch features（跳过cls token和register tokens）
            patch_features_flat = last_hidden_states[:, 1 + self.dino_num_register_tokens:, :]  # [6,196, 768]
            
            # 转换为空间布局并上采样到32x32
            patch_features = patch_features_flat.permute(0, 2, 1).reshape(
                -1, self.dino_hidden_size, num_patches_height, num_patches_width
            )  # [6, 768, 14, 14]
            dino_fea = F.interpolate(
                patch_features, size=(32, 32), mode='bilinear', align_corners=False
            )  # [6, 768, 32, 32]
            dino_fea = dino_fea.float()  # 转换为float32
            
            # 改进的融合流程（简化版）：
            # 保持DINO特征的完整信息，使用渐进式网络降维
            fused_fea = torch.cat((x, dino_fea), dim=1)   # [6, 128+768=896, 32, 32]
            x_fused = self.fuse_conv(fused_fea)  # [6, 128, 32, 32]
            
            # 步骤5：残差连接 - 保留部分原始VGG特征，增强梯度流
            x_residual = self.residual_proj(x)  # [6, 128, 32, 32]
            x = x_fused + 0.2 * x_residual  # 残差连接，权重可调（建议范围0.1-0.3）
            # import pdb
            # pdb.set_trace()
            # --------------------------------------------------------
        return x   #[6, 128, 32, 32]

class VolumeFeatureFusion(nn.Module):
    """使用交叉注意力机制融合参考特征和查询特征"""
    def __init__(self, feat_dim=128, norm_layer='instance'):
        super().__init__()
        if norm_layer == 'instance':
            norm = nn.InstanceNorm3d
        else:
            raise NotImplementedError
        
        # 特征投影层
        self.query_proj = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, 1, 1, 0),
            norm(feat_dim),
            nn.ReLU(True)
        )
        self.key_proj = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, 1, 1, 0),
            norm(feat_dim),
            nn.ReLU(True)
        )
        self.value_proj = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, 1, 1, 0),
            norm(feat_dim),
            nn.ReLU(True)
        )
        
        # 输出投影层
        self.out_proj = nn.Sequential(
            nn.Conv3d(feat_dim * 2, feat_dim, 1, 1, 0),
            norm(feat_dim),
            nn.ReLU(True)
        )
        
        # 门控机制：控制参考特征和查询特征的融合权重
        self.gate = nn.Sequential(
            nn.Conv3d(feat_dim * 2, feat_dim, 3, 1, 1),
            norm(feat_dim),
            nn.ReLU(True),
            nn.Conv3d(feat_dim, 2, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        
    def forward(self, ref_feat, que_feat):
        """
        @param ref_feat: [b, f, sx, sy, sz] 参考特征（均值）
        @param que_feat: [b, f, sx, sy, sz] 查询特征
        @return: [b, f, sx, sy, sz] 融合后的特征
        """
        # 改进的融合方式：使用通道注意力和空间特征交互
        # 1. 特征投影
        q = self.query_proj(que_feat)  # [b, f, sx, sy, sz]
        k = self.key_proj(ref_feat)    # [b, f, sx, sy, sz]
        v = self.value_proj(ref_feat)  # [b, f, sx, sy, sz]
        
        # 2. 计算通道级别的相似度（更高效）
        # 对每个空间位置，计算查询和参考特征在通道维度上的相似度
        b, f, sx, sy, sz = q.shape
        # 计算逐元素的相似度（点积）
        similarity = torch.sum(q * k, dim=1, keepdim=True)  # [b, 1, sx, sy, sz]
        similarity = similarity / (f ** 0.5)  # 缩放
        attn_map = torch.sigmoid(similarity)  # [b, 1, sx, sy, sz]
        
        # 3. 使用注意力图调制参考特征
        attn_feat = attn_map * v  # [b, f, sx, sy, sz]
        
        # 4. 门控融合：自适应权重
        concat_feat = torch.cat([attn_feat, que_feat], dim=1)  # [b, 2f, sx, sy, sz]
        gate_weights = self.gate(concat_feat)  # [b, 2, sx, sy, sz]
        
        # 5. 加权融合
        fused = gate_weights[:, 0:1] * attn_feat + gate_weights[:, 1:2] * que_feat
        
        # 6. 残差连接和输出投影
        concat_out = torch.cat([fused, que_feat], dim=1)  # [b, 2f, sx, sy, sz]
        out = self.out_proj(concat_out) + que_feat  # 残差连接
        
        return out


class RefineVolumeEncodingNet(nn.Module):
    def __init__(self,norm_layer='no_norm', use_attention_fusion=True):
        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm3d
        else:
            raise NotImplementedError

        self.use_attention_fusion = use_attention_fusion
        
        if self.use_attention_fusion:
            # 使用注意力融合模块
            self.fusion_module = VolumeFeatureFusion(feat_dim=128, norm_layer=norm_layer)
            # 融合后的特征维度仍然是128，所以mean_embed输入改为128*2（融合特征+查询特征）
            self.mean_embed = nn.Sequential(
                nn.Conv3d(128 * 2, 64, 3, 1, 1),
                norm(64),
                nn.ReLU(True),
                nn.Conv3d(64, 64, 3, 1, 1)
            )
        else:
            # 原始方式
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

    def forward(self, mean, var, que_feat=None):
        if self.use_attention_fusion and que_feat is not None:
            # 使用注意力融合
            fused_feat = self.fusion_module(mean, que_feat)  # [b, 128, sx, sy, sz]
            # 将融合特征和查询特征拼接
            mean_input = torch.cat([fused_feat, que_feat], dim=1)  # [b, 256, sx, sy, sz]
        else:
            # 原始方式
            mean_input = mean
        
        x = torch.cat([self.mean_embed(mean_input), self.var_embed(var)], 1)
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
        # 启用注意力融合机制
        self.volume_net = RefineVolumeEncodingNet('instance', use_attention_fusion=True)
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

            # 改进：使用加权平均，权重基于特征的一致性
            # 先计算简单平均作为初始参考
            vol_feats_mean_init = torch.mean(vol_feats, 0, keepdim=True)  # [1, f, sx, sy, sz]
            # 计算每个参考视图与初始平均值的余弦相似度作为权重
            vol_feats_norm = F.normalize(vol_feats, p=2, dim=1)  # [rfn, f, sx, sy, sz]
            mean_norm = F.normalize(vol_feats_mean_init, p=2, dim=1)  # [1, f, sx, sy, sz]
            similarity = torch.sum(vol_feats_norm * mean_norm, dim=1, keepdim=True)  # [rfn, 1, sx, sy, sz]
            # 使用温度缩放使权重分布更合理
            temperature = 2.0
            weights = F.softmax(similarity / temperature, dim=0)  # [rfn, 1, sx, sy, sz]
            
            # 加权平均
            vol_feats_mean_weighted = torch.sum(vol_feats * weights, dim=0)  # [f, sx, sy, sz]
            vol_feats_mean.append(vol_feats_mean_weighted)
            
            # 计算加权标准差（使用加权平均作为中心）
            vol_feats_mean_expanded = vol_feats_mean_weighted.unsqueeze(0).expand_as(vol_feats)  # [rfn, f, sx, sy, sz]
            diff = vol_feats - vol_feats_mean_expanded
            vol_feats_std.append(torch.sqrt(torch.sum(weights * diff ** 2, dim=0) + 1e-6))

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

        # 改进的融合方式：使用注意力机制融合参考特征和查询特征
        vol_feats = self.volume_net(vol_feats_mean, vol_feats_std, vol_feats_in)
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