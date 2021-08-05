# from torch._C import device
from torchvision import transforms
# from torchvision.transforms.transforms import CenterCrop
from dataset.video_shuffle_dataset import VideoShuffleDataset
import time
import torch
from os import path, mkdir


from renderer import Renderer
import util

# from model.dino import DinoVits8_Pretrained_Orig
from models.facecamera import FaceCameraModel
from models.dino import DinoVits8_Pretrained_Orig
from models.barlowtwins import BarlowTwins
from models.resnet import ResNet18_Pretrained_Orig
from models.simsiam import SimSiam

import imageio

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

train_dataset = VideoShuffleDataset('/mnt/hdd/data1/modality/shuffled/processed_data/train/')

net_type = 'simsiam'

if net_type == 'barlow_trained':
    args = type('', (), {})()
    args.projector = '8192-8192-8192'
    args.batch_size = 75
    args.lambd = 0.0051
    args.weight_decay = 1e-6
    net = BarlowTwins(args).cuda()
    net.load_state_dict(torch.load('/home/w0457094/git/modality/checkpoints/barlow-29-best.pth'))
elif net_type == 'dino':
    net = DinoVits8_Pretrained_Orig().cuda()
elif net_type == 'barlow_pretrained':
    net = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50').to(DEVICE)
elif net_type == 'resnet':
    net = ResNet18_Pretrained_Orig().to(DEVICE)
elif net_type == 'simsiam':
    net = SimSiam().to(DEVICE)
    net.load_state_dict(torch.load('/home/w0457094/git/photometric_optimization/checkpoints/resnet50-52-best.pth'))


net.eval()

outdir = '/home/w0457094/git/photometric_optimization/output/'
if path.exists(outdir) is False:
    mkdir(outdir)
if path.exists(path.join(outdir, net_type)) is False:
    mkdir(path.join(outdir, net_type))

normalise_trans = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
#     transforms.ToTensor(),

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

config = util.dict2obj(config)
util.check_mkdir(config.savefolder)

mesh_file = './data/head_template_mesh.obj'
render = Renderer(config.image_size, obj_filename=mesh_file).to(DEVICE)



# %% fitting


# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
lrn_rate = 0.005

num_epochs = 500

# loss_func = torch.nn.L1Loss()
loss_function = torch.nn.CosineSimilarity()

test_frame = 2
test_sequences = [5, 25, 100, 150, 205, 250]
for test_seq in test_sequences:
    test_item_idx = [test_seq,test_frame]

    test_images, _ = train_dataset[test_item_idx[0]]
    test_image = test_images[(test_item_idx[1]*3):(test_item_idx[1]*3)+3,:,:]
    test_encoding = net(test_image[None,:,:,:].cuda()).detach()

    test_image_vis = vis_trans(test_image)
    test_image_vis.save(f"/home/w0457094/git/photometric_optimization/output/test_img.jpg")


    # # We will save images periodically and compose them into a GIF.
    # filename_output = "./flame_optimization_demo.gif"
    # writer = imageio.get_writer(filename_output, mode='I', duration=0.3)



    # Initialize a model using the renderer, mesh and reference image
    model = FaceCameraModel(config, render, DEVICE)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lrn_rate)


    results = []

    t0 = time.time()


    for epoch in range(num_epochs):

        optimiser.zero_grad()

        img_render = model()[:,:,90:330]
        # print(img_render.shape)
        # img_render = model().permute(0,3,1,2)[:,0:3,:,90:330]
        img = normalise_trans(img_render)
        encoding = net(img[None,...])

        # loss = loss_func(encoding, test_encoding)
        loss = -loss_function(encoding, test_encoding).mean() * 0.5
        loss.backward()
        optimiser.step()

        img_vis = vis_trans(img.detach().cpu()).convert("RGB")
        img_vis.save(f"/home/w0457094/git/photometric_optimization/output/fit{epoch:03d}.jpg")
        print(f"Epoch {epoch+1:4d}, "
            f"Loss: {loss:5.8f}, pose[0]: {model.pose[0][0].item():f}")
                
        results.append(loss.item())
    t1 = time.time()


    print(f"Time to train: {t1-t0:0.2f}")

    
    test_image.mul_(torch.tensor([0.229, 0.224, 0.225])[:,None,None].repeat(1,420,240)).add_(torch.tensor([0.485, 0.456, 0.406])[:,None,None].repeat(1,420,240))
    test_image_vis = vis_trans(test_image)
    test_image_vis.save("{:s}/{:03d}_{:d}.jpg".format(path.join(outdir, net_type),test_seq,test_frame))
    img_vis = vis_trans(img_render.detach().cpu()).convert("RGB")
    img_vis.save("{:s}/{:03d}_{:d}_fit.jpg".format(path.join(outdir, net_type),test_seq,test_frame))



# %% plot

# losses= zip(*results)
# fig, ax = plt.subplots(2, 1, sharex=True, figsize=[10, 10])
# ax[0].plot(losses)
# ax[0].set_title("L1 Loss")
# # ax[1].plot(rx, "c", label="rx")
# # ax[1].plot(ry, "m", label="ry")
# # ax[1].plot(rz, "y", label="rz")
# # ax[1].plot(np.ones(num_epochs) * vx, "c--", label="x true")
# # ax[1].plot(np.ones(num_epochs) * vy, "m--", label="y true")
# # ax[1].plot(np.ones(num_epochs) * vz, "y--", label="z true")
# # ax[1].set_title("Rotation Values")
# plt.legend()
# fig.savefig("face_fit.jpg")