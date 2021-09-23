import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import glob

def create_shuffled_data(dataset, outputdir, ssd_threshold=1200, ssd_upper_threshold=3000):
    """
    Samples 3 frame tuples in the correct order and incorrect order

    - first, sample 5 (consecutive) frames, indexed a-e, such that a<b<c<d<e
    - positive examples created using b,c,d and d,c,b
    - negative examples created using b,a,d and b,e,d
    - frame c must not be similar to frames a or e
    - only temporal windows with high motion (large ssd) are considered
    - image tuples written to disk as memory mapped files to speed up dataloading
    - 1 tuple per second of data subject to meeting ssd threshold

    parameters:
        dataset: videodataset object
        outputdir: (string) location to write tuple data
        ssd_threshold: (float) minimum sum of square difference between subsequent frames in sequence
        ssd_upper_threshold: (float) maximum sum of square difference between subsequent frames in sequence
    """

    imgdiridx = 0
    if os.path.exists(outputdir) is False:
        os.mkdir(outputdir)

    for k in range(len(dataset)):
        print('Dataset item {:d}, num samples {:d}'.format(k,imgdiridx))
        # load left and right image sequences (ignore speaker id)
        l,r,_ = dataset[k]

        sample_length = l.shape[1]

        # ssd of consecutive frames
        ssdl = torch.sum(torch.pow(torch.diff(l, dim=1),2), dim=(0,2,3))
        maxssd, idx = torch.max(ssdl, dim=0, keepdim=True)

        # does it satisfy threshold criteria?
        if maxssd[0] > ssd_threshold and maxssd[0] < ssd_upper_threshold:
            
            # ensure that we won't index out of bounds
            idx = max(min(idx, sample_length-3),2)

            #  make a directory for left and right
            imgoutdir_l = os.path.join(outputdir,'{:05d}'.format(imgdiridx))
            if os.path.exists(imgoutdir_l) is False:
                os.mkdir(imgoutdir_l)
            imgoutdir_r = os.path.join(outputdir,'{:05d}'.format(imgdiridx+1))
            if os.path.exists(imgoutdir_r) is False:
                os.mkdir(imgoutdir_r)
            imgdiridx += 2

            # save images to disk. Permutations performed during dataset creation/dataloading
            for seqidx, imgidx in enumerate(range(idx-2,idx+3)):
                img = Image.fromarray((l[:,imgidx,:,:].permute(1,2,0).numpy() * 255).astype(np.uint8))
                img.save(os.path.join(imgoutdir_l, '{:d}.jpg'.format(seqidx)))
                img = Image.fromarray((r[:,imgidx,:,:].permute(1,2,0).numpy() * 255).astype(np.uint8))
                img.save(os.path.join(imgoutdir_r, '{:d}.jpg'.format(seqidx)))



class TupleRecord(object):
    """
    Helper class. This class
    represents a tuple's metadata.
    """
    def __init__(self, tupledir, imgnames, label):
        self._tupledir = tupledir
        self._imgnames = imgnames
        self._label = label


    @property
    def tupledir(self):
        return self._tupledir

    @property
    def imgnames(self):
        return self._imgnames

    @property
    def label(self):
        return self._label


class VideoShuffleDataset(torch.utils.data.Dataset):
    r"""
    A dataset class for shuffled video tuples.
    """
    def __init__(self,
                 datadir: str, 
                 img_resize = [420, 240]):
        super(VideoShuffleDataset, self).__init__()

        self.datadir = datadir
        self.trans = transforms.Compose([
            transforms.Resize(img_resize), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        self._parse_dir()

    def _load_image(self, directory, imgname):
        return self.trans(Image.open(os.path.join(directory, imgname)).convert('RGB'))

    def _parse_dir(self):
        
        dirlist = glob.glob(os.path.join(self.datadir, '*'))
        self.tuple_list = list()

        for x in dirlist:
            self.tuple_list.append(TupleRecord(x, ['1.jpg','2.jpg','3.jpg'], 1))
            # self.tuple_list.append(TupleRecord(x, ['3.jpg','2.jpg','1.jpg'], 1))
            # self.tuple_list.append(TupleRecord(x, ['1.jpg','0.jpg','3.jpg'], 0))
            # self.tuple_list.append(TupleRecord(x, ['1.jpg','4.jpg','3.jpg'], 0))
            # self.tuple_list.append(TupleRecord(x, ['3.jpg','0.jpg','1.jpg'], 0))
            # self.tuple_list.append(TupleRecord(x, ['3.jpg','4.jpg','1.jpg'], 0))

    
    def getitempath(self, tupleindex, imgindex):
        return os.path.join(self.tuple_list[tupleindex].tupledir, self.tuple_list[tupleindex].imgnames[imgindex])

    def __getitem__(self, index):
        """
        Args:
            index: Tuple sample index.
        Returns:
            a tuple of images and a label
        """
        record = self.tuple_list[index]
        img1, img2, img3, label = self._get(record)

        return torch.cat((img1,img2,img3)), label

    def _get(self, record):
        """
        Loads the tuple frames at the corresponding
        indices.

        Args:
            record: TupleRecord denoting a tuple sample.
        Returns:
            1) 3 tensor images
            2) An integer denoting the tuple label (0 or 1).
        """
        # trans = transforms.ToTensor()

        img1 = self._load_image(record.tupledir,record.imgnames[0])         
        img2 = self._load_image(record.tupledir,record.imgnames[1])        
        img3 = self._load_image(record.tupledir,record.imgnames[2])       

        # return trans(img1), trans(img2), trans(img3), record.label
        return img1, img2, img3, record.label

    def __len__(self):
        return len(self.tuple_list)



