import torch
from renderer import Renderer
import util
from models.facecamera import FaceCameraModel
from .image_transformations import Transform


class SampledFlameDataset(torch.utils.data.Dataset):
    r"""
    A different flame sample at each getitem call
    """
    def __init__(self,img_size = 384, dataset_len=10000, device = 'cpu', frontal=True, random_tex=False):
        super(SampledFlameDataset, self).__init__()

        self.dataset_len = dataset_len
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
            'image_size': img_size,
            'e_lr': 0.005,
            'e_wd': 0.0001,
        }

        config = util.dict2obj(config)

        mesh_file = './data/head_template_mesh.obj'
        render = Renderer(config.image_size, obj_filename=mesh_file).to(device)
        self.face_camera_model = FaceCameraModel(config, render, device)
        self.device = device
        self.transform = Transform()
        self.frontal = frontal 
        self.random_tex = random_tex



    def __getitem__(self, index):
        """
        Args:
            index: Image sample index.
        Returns:
            an image and a label
        """
        # sample flame
        self.face_camera_model.set_random(tex_on=self.random_tex)
        if self.frontal:
            self.face_camera_model.set_cam(eye=torch.nn.Parameter(torch.tensor([0.0, 0.0, -1.5], device=self.device)))

        # render image and normalised image
        image_norm, image = self.transform(self.face_camera_model, norm_only=True)

        # transform once for augmented example
        image_aug_norm, image_aug = self.transform(self.face_camera_model)
        
        return image, image_norm, image_aug, image_aug_norm


    def __len__(self):
        return self.dataset_len




class ContrastiveFlameDataset(torch.utils.data.Dataset):
    r"""
    A different flame sample at each getitem call
    """
    def __init__(self,img_size = 420, dataset_len=10000, device = 'cpu'):
        super(ContrastiveFlameDataset, self).__init__()

        self.dataset_len = dataset_len
        self.config = {
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
            'image_size': img_size,
            'e_lr': 0.005,
            'e_wd': 0.0001,
        }

        self.config = util.dict2obj(self.config)

        mesh_file = './data/head_template_mesh.obj'
        self.render = Renderer(self.config.image_size, obj_filename=mesh_file).to(device)
        self.face_camera_model = FaceCameraModel(self.config, self.render, device)
        self.device = device
        self.transform = Transform()

    #     self.fcms = self.initialise_queries()



    # def initialise_queries(self):

    #     self.fcms = []
    #     for _ in range(self.dataset_len):
    #         self.face_camera_model.set_random(cam_on=False)
    #         self.fcms.append(self.face_camera_model.clone())


    def __getitem__(self, index):
        """
        Args:
            index: Image sample index.
        Returns:
            an image and a label
        """
        # sample flame - frontal fixed camera
        self.face_camera_model.set_random()
        self.face_camera_model.set_cam(eye=torch.nn.Parameter(torch.tensor([0.0, 0.0, -1.5], device=self.device)))
        image, _ = self.face_camera_model()
        image_norm = self.transform(self.face_camera_model, norm_only=True)

        # transform once for positive augmented example
        image_aug_pos_norm = self.transform(self.face_camera_model)

        # sample flame again to create negative example
        self.face_camera_model.set_random()

        # transform once for negative augmented example
        image_aug_neg_norm = self.transform(self.face_camera_model)
        
        return image, image_norm, image_aug_pos_norm, image_aug_neg_norm 


    def __len__(self):
        return self.dataset_len



