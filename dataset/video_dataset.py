import os
import os.path
from PIL import Image
from torchvision import transforms
import torch
import glob


class FiveFrameRecord(object):
    """
    Helper class. This class
    represents a sequence of 5 frame's metadata.
    """
    def __init__(self, seqdir, imgnames, label):
        self._seqdir = seqdir
        self._imgnames = imgnames
        self._label = label


    @property
    def seqdir(self):
        return self._seqdir

    @property
    def imgnames(self):
        return self._imgnames

    @property
    def label(self):
        return self._label


class VideoDataset(torch.utils.data.Dataset):
    r"""
    A dataset class for video frames.
    """
    def __init__(self,
                 datadir: str, 
                 img_resize = 384, normalise=True):
        super(VideoDataset, self).__init__()

        self.datadir = datadir
        if normalise:
            self.trans = transforms.Compose([
                transforms.Resize(275), 
                transforms.CenterCrop(img_resize), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        else:
            self.trans = transforms.Compose([
                transforms.Resize(275), 
                transforms.CenterCrop(img_resize), 
                transforms.ToTensor()
                ])

        self._parse_dir()

    def _load_image(self, directory, imgname):
        return self.trans(Image.open(os.path.join(directory, imgname)).convert('RGB'))

    def _parse_dir(self):
        
        dirlist = glob.glob(os.path.join(self.datadir, '*'))
        self.frame_list = list()

        for x in dirlist:
            self.frame_list.append(FiveFrameRecord(x, ['0.jpg','1.jpg','2.jpg','3.jpg','3.jpg'], 1))

    
    def getitempath(self, seqindex, imgindex):
        return os.path.join(self.frame_list[seqindex].seqdir, self.frame_list[seqindex].imgnames[imgindex])

    def __getitem__(self, index):
        """
        Args:
            index: Sequence index.
        Returns:
            a sequence of images and a label
        """
        record = self.frame_list[index]
        img0, img1, img2, img3, img4, label = self._get(record)

        return torch.cat((img0, img1, img2, img3, img4)), label

    def _get(self, record):
        """
        Loads the sequence frames at the corresponding
        indices.

        Args:
            record: FrameRecord denoting a 5 frame sample.
        Returns:
            1) 3 tensor images
            2) An integer denoting the sequence label (0 or 1).
        """
        # trans = transforms.ToTensor()
        img0 = self._load_image(record.seqdir,record.imgnames[0])         
        img1 = self._load_image(record.seqdir,record.imgnames[1])         
        img2 = self._load_image(record.seqdir,record.imgnames[2])        
        img3 = self._load_image(record.seqdir,record.imgnames[3])       
        img4 = self._load_image(record.seqdir,record.imgnames[4])       

        return img0, img1, img2, img3, img4, record.label

    def __len__(self):
        return len(self.frame_list)



