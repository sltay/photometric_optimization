import os
import os.path
from PIL import Image
from torchvision import transforms
import torch
import glob



class ImageRecord(object):
    """
    Helper class. This class
    represents an image's metadata.
    """
    def __init__(self, imgpath, label):
        self._imgpath = imgpath
        self._label = label


    @property
    def imgpath(self):
        return self._imgpath


    @property
    def label(self):
        return self._label


class FaceImageDataset(torch.utils.data.Dataset):
    r"""
    A dataset class for faces from directories of images.
    """
    def __init__(self,
                 datadir: str, 
                 img_resize = [420, 240]):
        super(FaceImageDataset, self).__init__()

        self.datadir = datadir
        self.trans = transforms.Compose([
            transforms.Resize(img_resize), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        self._parse_dir()

    def _load_image(self, imgpath):
        return self.trans(Image.open(imgpath).convert('RGB'))

    def _parse_dir(self):
        
        dirlist = glob.glob(os.path.join(self.datadir, '*'))
        self.image_list = list()

        for x in dirlist:
            imglist = sorted(glob.glob(os.path.join(x, '*.jpg')))
            for img in imglist:
                self.image_list.append(ImageRecord(img, 1))



    def __getitem__(self, index):
        """
        Args:
            index: Image sample index.
        Returns:
            an image and a label
        """
        record = self.image_list[index]
        img, label = self._get(record)

        return img, label

    def _get(self, record):
        """
        Loads the image at the corresponding
        indices.

        Args:
            record: ImageRecord denoting a image sample.
        Returns:
            1) 1 tensor image
            2) An integer denoting the label (0 or 1).
        """

        img = self._load_image(record.imgpath)
        return img, record.label

    def __len__(self):
        return len(self.image_list)



