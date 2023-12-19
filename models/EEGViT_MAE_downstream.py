# modified from "https://github.com/ruiqiRichard/EEGViT/blob/master/models/EEGViT_pretrained.py"

import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_MAE_downstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        config = transformers.ViTConfig(
            num_channels=256,
            image_size=(129,14),
            patch_size=(8,1)
        )

        model = transformers.ViTForImageClassification(config=config)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.ViT = model

    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits

        return x
    
    # load pretrained encoder (EEGViT) weights
    def load_pretrained(self, model_path):
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
        downstream_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in downstream_dict and
                           downstream_dict[k].size() == v.size()}
        downstream_dict.update(pretrained_dict)
        self.load_state_dict(downstream_dict)
