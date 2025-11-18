import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import torch.nn.functional as F
import os

# 获取绝对路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(base_dir, "data/LINEMOD/ape/JPEGImages/000000.jpg")
image = load_image(image_path)
# 将图片调整至128x128
image = image.resize((128, 128))
print("Image size:", image.height, image.width)

processor = AutoImageProcessor.from_pretrained("dinov3-vitb16-pretrain-lvd1689m")
model = AutoModel.from_pretrained(
    "dinov3-vitb16-pretrain-lvd1689m",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

patch_size = model.config.patch_size
dino_hidden_size = model.config.hidden_size
print("Dino hidden size:", dino_hidden_size)
print("Patch size:", patch_size) # 16
print("Num register tokens:", model.config.num_register_tokens) # 4

inputs = processor(images=image, return_tensors="pt")
device = next(model.parameters()).device
inputs = inputs.to(device)
print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

batch_size, _, img_height, img_width = inputs.pixel_values.shape #1,3,224,224
num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size # 224/16=14, 224/16=14
num_patches_flat = num_patches_height * num_patches_width #196

with torch.inference_mode():
  outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # [1, 1 + 4 + 196, 768]
assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

cls_token = last_hidden_states[:, 0, :]
patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]  # [1, 196, 768]
# 转换为类似refiner.py中dino_fea的维度 [1, 384, 32, 32]
# 先reshape成空间布局 [1, 768, 14, 14]
patch_features = patch_features_flat.permute(0, 2, 1).reshape(batch_size, model.config.hidden_size, num_patches_height, num_patches_width)  # [1, 768, 14, 14]
print("patch_features shape before:", patch_features.shape)  # [1, 768, 14, 14]

# 降维：768 -> 384，使用1x1卷积（对2D特征图更直观）
reduce_dim = torch.nn.Conv2d(model.config.hidden_size, 384, kernel_size=1, bias=False).to(patch_features.device)
patch_features_384 = reduce_dim(patch_features)  # [1, 384, 14, 14]

# 上采样到32x32
dino_fea = F.interpolate(patch_features_384, size=(32, 32), mode='bilinear', align_corners=False)  # [1, 384, 32, 32]
print("dino_fea shape:", dino_fea.shape)  # [1, 384, 32, 32]
import pdb
pdb.set_trace()