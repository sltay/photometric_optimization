# from torch._C import device
from torchvision import transforms
# from torchvision.transforms.transforms import CenterCrop
import time
import torch
from os import path, mkdir

from renderer import Renderer

import util

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardGouraudShader, HardPhongShader, PointLights, TexturesVertex,
)

# from model.dino import DinoVits8_Pretrained_Orig
from models.facecamera import FaceCameraModel


DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)



outdir = '/mnt/hdd/data1/modality/shuffled/processed_data/train_synthetic'
if path.exists(outdir) is False:
    mkdir(outdir)


vis_trans = transforms.Compose([
    transforms.ToPILImage()
])

config = {
    # FLAME
    'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
    'flame_lmk_embedding_path': './data/landmark_embedding.npy',
    'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
    'camera_params': 3,
    'shape_params': 100,
    'expression_params': 50,
    'pose_params': 6,
    'tex_params': 50,
    'use_face_contour': True,

    'cropped_size': 256,
    'batch_size': 1,
    'image_size': 420,
    'e_lr': 0.005,
    'e_wd': 0.0001,
    'savefolder': './test_results/',
    # weights of losses and reg terms
    'w_pho': 8,
    'w_lmks': 1,
    'w_shape_reg': 1e-4,
    'w_expr_reg': 1e-4,
    'w_pose_reg': 0,
}

# %% rendering
image_size = 420
device = 'cuda'

config = util.dict2obj(config)
util.check_mkdir(config.savefolder)

mesh_file = './data/head_template_mesh.obj'
render = Renderer(image_size, obj_filename=mesh_file).to(device)

from models.facecamera import FaceCameraModel

fcm = FaceCameraModel(config=config, render=render)

# %% sampling
# pose in range -0.2 to 0.2
# betas in range -3 to 3

facemodel = Pose(device=DEVICE).eval()
model = FaceCameraModel(mesh=flame_mesh, renderer=renderer, face_model=facemodel).to(DEVICE)

for sample in range(2000):


    pose = torch.rand(1,15).cuda()
    pose = pose.mul(0.3).sub(0.15)
    betas = torch.rand(1,400).cuda()
    betas = betas.mul(3.0).sub(1.5)

    # trans = torch.rand(1,3).cuda()
    # trans[0][2] = 0.0


    img = model(pose, betas).permute(0,3,1,2)[:,0:3,:,90:330]
    img_vis = vis_trans(img[0].detach()).convert("RGB")
    img_vis.save("{:s}/{:04d}.jpg".format(outdir, sample))

