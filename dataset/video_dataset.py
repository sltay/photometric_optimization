import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from .transforms import VideoFilePathToTensor
import cv2

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.
    """
    def __init__(self, pathl, pathr, label, frames):
        self._label = label
        self._pathl = pathl
        self._pathr = pathr
        self._frames = frames


    @property
    def pathl(self):
        return self._pathl

    @property
    def pathr(self):
        return self._pathr

    @property
    def frames(self):
        return self._frames

    @property
    def num_frames(self):
        return len(self._frames)  # +1 because end frame is inclusive

    @property
    def label(self):
        return self._label



class VideoDataset(torch.utils.data.Dataset):
    r"""
    A dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.



    Note:
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.

    """
    def __init__(self,
                 annotationfile_path: str,
                 frames_per_segment: int = 25,
                 transform = None,
                 random_shift: bool = True,
                 test_mode: bool = False,
                 fps: int = 25):
        super(VideoDataset, self).__init__()

        self.annotationfile_path = annotationfile_path
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.fps = fps

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

    def _parse_list(self):
        
        self.video_list = list()

        for x in open(self.annotationfile_path):
            pathl, pathr, label = x.strip().split()

             # open video file
            cap = cv2.VideoCapture(pathl)
            assert(cap.isOpened())
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            sample_factor = int(orig_fps / self.fps)

            f = 0
            while f <= num_frames:
                self.video_list.append(VideoRecord(pathl, pathr, label, list(np.arange(f,f+(self.frames_per_segment*sample_factor),sample_factor))))
                f += self.frames_per_segment*sample_factor

    

    def __getitem__(self, index):
        """

        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """
        record = self.video_list[index]

        return self._get(record)

    def _get(self, record):
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from.
        Returns:
            1) A list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) An integer denoting the video label.
        """

        videoframetoimgtransform = VideoFilePathToTensor()

        imagesl = videoframetoimgtransform(record.pathl, record.frames)
        imagesr = videoframetoimgtransform(record.pathr, record.frames)


        if self.transform is not None:
            imagesl = self.transform(imagesl)
            imagesr = self.transform(imagesr)

        return imagesl, imagesr, record.label

    def __len__(self):
        return len(self.video_list)



class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
