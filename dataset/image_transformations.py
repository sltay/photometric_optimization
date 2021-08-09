import torchvision.transforms as transforms
import torch

torch.manual_seed(17)

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
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
            transforms.RandomInvert(p=0.2),
            transforms.RandomSolarize(threshold= 192.0, p=0.2),
            transforms.RandomApply(
                [transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomPosterize(bits=5),
                transforms.ConvertImageDtype(torch.float32)],
                p=0.2
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
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
            transforms.RandomInvert(p=0.2),
            transforms.RandomApply(
                [transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomPosterize(bits=5),
                transforms.ConvertImageDtype(torch.float32)],
                p=0.2
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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

    def __call__(self, x, alpha=None):
        
        keep_orig = torch.rand(2)

        if alpha is not None:
            # randomise background
            y1 = self.randomise_background(x.clone(), alpha)
            if keep_orig[0] < 0.2:
                y2 = x.clone()
            else:
                y2 = self.randomise_background(x.clone(), alpha)
        else:
            y1 = x.clone()
            y2 = x.clone()

        y1 = self.transform(y1)

        if keep_orig[1] < 0.2:
            # only normalise - don't modify otherwise
            y2 = self.norm_transform(y2)
        else:
            y2 = self.transform_prime(y2)
        return y1, y2


