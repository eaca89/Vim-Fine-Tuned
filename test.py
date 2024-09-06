from pathlib import PurePath
import torch
import sys
import os
# Get the absolute path of the vim directory
vim_path = os.path.abspath('vim')
# Add the vim directory to the system path
sys.path.append(vim_path)
from vim.models_mamba import VisionMamba

from huggingface_hub import snapshot_download

VIM_REPO = "hustvl/Vim-small-midclstok"

pretrained_model_dir = snapshot_download(
    repo_id=VIM_REPO,
    # Comment the next line the first time to have the files be
    # downloaded. 
    local_files_only=True
)

MODEL_FILE = PurePath(pretrained_model_dir, "vim_s_midclstok_ft_81p6acc.pth")
print(MODEL_FILE)

model = VisionMamba(
    patch_size=16,
    stride=8,
    embed_dim=384,
    depth=24,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type='mean',
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="v2",
    if_cls_token=True,
    if_devide_out=True,
    use_middle_cls_token=True,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224,
)

checkpoint = torch.load(str(MODEL_FILE), map_location='cpu')
# Important: make sure the values of this match what's used to instantiate the VisionMamba class.
# If not, loading the checkpoint will fail.
checkpoint["args"]

model.load_state_dict(checkpoint["model"])

model.eval()
model.to("cuda")

from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

test_image = Image.open("test.jpg")
test_image = test_image.resize((224, 224))
image_as_tensor = transforms.ToTensor()(test_image)
normalized_tensor = transforms.Normalize(
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(image_as_tensor)
# test_image

x = normalized_tensor.unsqueeze(0).cuda()
pred = model(x)
# Note: the returned label can be verified with https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
print(pred.argmax())