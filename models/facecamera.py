from pkg_resources import require
import torch.nn as nn
import torch


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

        # Create an optimizable parameter for the x, y, z position of the camera and light. 
        self.shape = nn.Parameter(torch.zeros(1, self.config.shape_params).float().to(self.device))
        self.tex = nn.Parameter(torch.zeros(1, self.config.tex_params).float().to(self.device))
        self.exp = nn.Parameter(torch.zeros(1, self.config.expression_params).float().to(self.device))
        self.pose = nn.Parameter(torch.zeros(1, self.config.pose_params).float().to(self.device))
        # cam = torch.zeros(1, self.config.camera_params); cam[:, 0] = 5.
        # self.cam = nn.Parameter(cam.float().to(self.device))
        self.lights = nn.Parameter(torch.rand(1, 9, 3).float().to(self.device))

        self.cam = torch.zeros(1, self.config.camera_params).float().to(self.device) 
        self.cam[:,0] = 5
        # self.lights = torch.rand(1, 9, 3).float().to(self.device)


    def set_random(self, shape_on=True, tex_on=True, exp_on=True, pose_on=True):

        if shape_on:
            self.shape = nn.Parameter(torch.rand((1, self.config.shape_params), device=self.device).float()*.3-.15)
        if tex_on:
            self.tex = nn.Parameter(torch.rand((1, self.config.tex_params), device=self.device).float()*2-1)
        if exp_on:
            self.exp = nn.Parameter(torch.rand((1, self.config.expression_params), device=self.device).float()*1.0-.5)
        if pose_on:
            self.pose = nn.Parameter(torch.rand((1, self.config.pose_params), device=self.device).float()*.2-.1)


    def forward(self, shape=None, tex=None, exp=None, pose=None, lights=None):

        if pose != None:
            self.pose = nn.Parameter(pose)
        if shape != None:
            self.shape = nn.Parameter(shape)
        if tex != None:
            self.tex = nn.Parameter(tex)
        if exp  != None:
            self.exp = nn.Parameter(exp)
        if lights  != None:
            self.lights = nn.Parameter(lights)

        vertices, _, _ = self.flame(shape_params=self.shape, expression_params=self.exp, pose_params=self.pose)
        trans_vertices = util.batch_orth_proj(vertices, self.cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        albedos = self.flametex(self.tex) / 255.
        # ops = self.render(vertices, trans_vertices, albedos, self.lights)
        ops = self.render(vertices, trans_vertices, albedos, lights=None)
        image = ops['images'][0].float()
        alpha = ops['alpha_images'][0]

        return image, alpha