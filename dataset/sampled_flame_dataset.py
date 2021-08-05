import torch
from renderer import Renderer
import util
from models.facecamera import FaceCameraModel



class SampledFlameDataset(torch.utils.data.Dataset):
    r"""
    A different flame sample at each getitem call
    """
    def __init__(self,img_size = 420, dataset_len=10000, device = 'cpu'):
        super(SampledFlameDataset, self).__init__()

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



    def __getitem__(self, index):
        """
        Args:
            index: Image sample index.
        Returns:
            an image and a label
        """
        # sample flame
        self.face_camera_model.set_random()
        image = self.face_camera_model()
        
        return image


    def __len__(self):
        return self.dataset_len



