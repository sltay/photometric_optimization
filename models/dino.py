"""done in pytorch
@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:2104.14294},
  year={2021}
}
"""
import torch
import torch.nn as nn




class DinoVits8_Pretrained(nn.Module):

    def __init__(self):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(384, 512)
        self.fc2 = nn.Linear(3 * 512, 2)


    def forward_once(self, x):
        output = self.dino(x)
        output = self.fc1(self.relu(output))
        return output.view(output.size(0), -1)

    def forward(self, x):
        output1 = self.forward_once(x[:,0:3,:,:])
        output2 = self.forward_once(x[:,3:6,:,:])
        output3 = self.forward_once(x[:,6:9,:,:])

        output = torch.cat((output1, output2, output3),dim=1)
        # output = self.fc1(self.relu(output))
        output = self.fc2(self.relu(output))
        return output


class DinoVits8_Pretrained_Orig(nn.Module):

    def __init__(self):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

    def forward(self, x):

        return self.dino(x)





