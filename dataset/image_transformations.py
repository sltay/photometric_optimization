import torchvision.transforms as transforms
import torch

torch.manual_seed(17)

class Transform:
    def __init__(self, randomise_camera=True, randomise_texture=True, randomise_lighting=True):


        self.randomise_camera = randomise_camera
        self.randomise_lighting = randomise_lighting
        self.randomise_texture = randomise_texture

        self.transform = transforms.Compose([
            # transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.2)],
                p=0.4
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur([5,5])],
                p=0.2
            ),
            # transforms.RandomInvert(p=0.2),
            transforms.RandomSolarize(threshold= 192.0, p=0.2),
            transforms.RandomApply(
                [transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomPosterize(bits=5),
                transforms.ConvertImageDtype(torch.float32)],
                p=0.2
            )
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.2)],
                p=0.4
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur([5,5])],
                p=0.2
            ),            
            transforms.RandomSolarize(threshold= 192.0,p=0.2),
            # transforms.RandomInvert(p=0.2),
            transforms.RandomApply(
                [transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomPosterize(bits=5),
                transforms.ConvertImageDtype(torch.float32)],
                p=0.2
            )
        ])

        self.norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
 

    def randomise_background(self,x, alpha):
        
        r,g,b = torch.rand(3)

        background = torch.ones_like(x)
        background[0,:,:] = r
        background[1,:,:] = g
        background[2,:,:] = b

        x[~alpha.repeat((3,1,1)).bool()] = background[~alpha.repeat((3,1,1)).bool()]  
        return x


    def randomise_camera_texture_lights(self, face_camera_model):
        face_camera_model.set_random(shape_on=False, tex_on=self.randomise_texture, exp_on=False, pose_on=False, cam_on=self.randomise_camera, lights_on=self.randomise_lighting)


    def __call__(self, face_cam_model, norm_only=False):
        

        if norm_only is False:
            # randomise camera, texure and lighting as required with a defined probability
            if torch.rand(1) > 0.2:
                self.randomise_camera_texture_lights(face_cam_model)
                
        y, alpha = face_cam_model()

        if norm_only is False:
            # randomise background
            if torch.rand(1) > 0.2:
                y = self.randomise_background(y, alpha)
            # other augmentations
            y = self.transform(y)

        return self.norm_transform(y), y


