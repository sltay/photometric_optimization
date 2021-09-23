from pkg_resources import require
from torch._C import device
import torch.nn as nn
import torch

from dcamera.camera import Lookat

import util


from .FLAME import FLAME, FLAMETex


class FaceCameraModel(nn.Module):
    def __init__(self, config, render, device='cuda'):
        super().__init__()


        self.config = config
        self.device = device
        self.render = render
        self.flame = FLAME(config).to(device)
        self.flametex = FLAMETex(config).to(device)

        self.reset()

        # # Create an optimizable parameter for the x, y, z position of the camera and light. 
        # self.shape = nn.Parameter(torch.zeros(1, self.config.shape_params).float().to(self.device))
        # self.tex = nn.Parameter(torch.zeros(1, self.config.tex_params).float().to(self.device))
        # self.exp = nn.Parameter(torch.zeros(1, self.config.expression_params).float().to(self.device))
        # self.pose = nn.Parameter(torch.zeros(1, self.config.pose_params).float().to(self.device))
        # # cam = torch.zeros(1, self.config.camera_params); cam[:, 0] = 5.
        # # self.cam = nn.Parameter(cam.float().to(self.device))
        # self.sh_coef = nn.Parameter(torch.tensor([2.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,0.0]).float().to(self.device))
        # # self.lights = torch.rand(1, 9, 3).float().to(self.device)

        # self.cam = torch.zeros(1, self.config.camera_params).float().to(self.device) 
        # self.cam[:,0] = 5
        # # self.lights = torch.rand(1, 9, 3).float().to(self.device)

        # self.lights = torch.ones((1,9,3), device=self.device)

    def reset(self):

        # Create an optimizable parameter for the x, y, z position of the camera and light. 
        self.shape = nn.Parameter(torch.zeros(1, self.config.shape_params).float().to(self.device))
        self.tex = nn.Parameter(torch.zeros(1, self.config.tex_params).float().to(self.device))
        self.exp = nn.Parameter(torch.zeros(1, self.config.expression_params).float().to(self.device))
        self.pose = nn.Parameter(torch.zeros(1, self.config.pose_params).float().to(self.device))
        # cam = torch.zeros(1, self.config.camera_params); cam[:, 0] = 5.
        # self.cam = nn.Parameter(cam.float().to(self.device))
        # self.sh_coef = nn.Parameter(torch.tensor([2.0, 0.1, 0.1, 0.1]).float().to(self.device))
        self.sh_coef = torch.tensor([2.0, 0.1, 0.1, 0.1]).float().to(self.device)
        # self.lights = torch.rand(1, 9, 3).float().to(self.device)

        # self.cam = torch.zeros(1, self.config.camera_params).float().to(self.device) 
        # self.cam[:,0] = 5

        self.cam = Lookat(aov=15, eye=[0.0, 0.0, -1.5], target=[0.0, 0.0, 0.0], width=420, height=420, device=self.device)
        # self.lights = torch.rand(1, 9, 3).float().to(self.device)

        self.lights = torch.ones((1,9,3), device=self.device)


    def set_cam(self, eye, aov=None):
        self.cam.eye = eye
        if aov is not None:
            self.cam.aov = aov

    def set_random(self, shape_on=True, tex_on=True, exp_on=True, pose_on=True, cam_on=True, lights_on=True):

        if shape_on:
            self.shape = nn.Parameter(torch.randn((1, self.config.shape_params), device=self.device).float()*.6)
        if tex_on:
            self.tex = nn.Parameter(torch.randn((1, self.config.tex_params), device=self.device).float()*.5)
        if exp_on:
            self.exp = nn.Parameter(torch.randn((1, self.config.expression_params), device=self.device).float())
        if pose_on:
            self.pose = nn.Parameter(torch.cat((
                torch.randn((1, 3), device=self.device).float()*.1,
                (torch.rand((1, 1), device=self.device)* .7-.2).clamp(min=0.0),
                torch.randn((1, 2), device=self.device).float()*.02
            ),dim=1))
        if cam_on:
            # only randomise camera position - keep target and aov the same
            eye = torch.rand(3, device=self.device)
            eye[0] = eye[0] * 2.0 - 1.0
            eye[1] = eye[1] * 1.4 - 0.7
            eye[2] = eye[2] * -1 -.5

            self.cam.eye = nn.Parameter(eye)

            # x,y,z = torch.randn(3)
            # z = z *.08 + 5.0
            # x = x *.03
            # y = y * .03
            # self.cam[:,0] = z
            # self.cam[:,1] = x
            # self.cam[:,2] = y
        if lights_on:
            self.lights = torch.ones((1,9,3), device=self.device)
            sh_coefs = torch.randn(9, device=self.device)
            sh_coefs[0] = (torch.abs(sh_coefs[0] * 1.5) + 1).clamp_max(2.7)
            sh_coefs[1] = sh_coefs[1] * 0.2 
            sh_coefs[2] = sh_coefs[2] * 0.2
            sh_coefs[3] = sh_coefs[3] * 0.2
            sh_coefs[4] = sh_coefs[4] * 0.
            sh_coefs[5] = sh_coefs[5] * 0.
            sh_coefs[6] = sh_coefs[6] * 0.
            sh_coefs[7:] = 0.0

            sh_coefs = sh_coefs[None,:,None].repeat(1,1,3)


            # self.lights = nn.Parameter(torch.cat((torch.ones((1,1,3), device=self.device)*2,torch.ones((1,1,3), device=self.device)*-0.5,torch.zeros((1, 7, 3), device=self.device)),dim=1).float())


    def forward(self, shape=None, tex=None, exp=None, pose=None, sh_coef=None, return_alpha=True):

        if pose != None:
            self.pose = nn.Parameter(pose)
        if shape != None:
            self.shape = nn.Parameter(shape)
        if tex != None:
            self.tex = nn.Parameter(tex)
        if exp  != None:
            self.exp = nn.Parameter(exp)
        if sh_coef  != None:
            self.sh_coef = nn.Parameter(sh_coef)


        pose = torch.cat((
            self.pose[[0],:3].clamp(min=-0.1, max=0.1),
            self.pose[[0],3:4].clamp(min=0.0, max = 0.5),
            self.pose[[0],4:].clamp(min=-0.02, max=0.02)
            ),dim=1)


        vertices, _, _ = self.flame(shape_params=self.shape, expression_params=self.exp, pose_params=pose)
        # trans_vertices = util.batch_orth_proj(vertices, self.cam)


                
        trans_vertices = self.cam.project(vertices[0])
        trans_vertices = trans_vertices[None, :, :3]
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        albedos = self.flametex(self.tex) / 255.

        sh_coef = torch.cat((self.sh_coef, torch.zeros(5, device=self.device)))
        ops = self.render(vertices, trans_vertices, albedos, self.lights.mul(sh_coef[None,:, None].repeat(1,1,3)))
        # ops = self.render(vertices, trans_vertices, albedos, lights=None)
        image = ops['images'][0].float()
        alpha = ops['alpha_images'][0].bool()

        if return_alpha:
            return image, alpha
        else:
            return image

