import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig


class EEGViTPre_finetune(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )

        self.batchnorm1 = nn.BatchNorm2d(256, False)

        config = ViTConfig(
            num_channels=256,
            image_size=(129, 14),
            patch_size=(8, 1)
        )

        model = ViTForImageClassification(config=config)
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.classifier = nn.Sequential(
            nn.Linear(768, 1000, bias=True),
            nn.Dropout(0.1),
            nn.Linear(1000, 2, bias=True)
        )

        self.ViT = model

    def load_pretrained(self, model_path: str):
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
        finetune_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in finetune_dict and
                           finetune_dict[k].size() == v.size()}
        finetune_dict.update(pretrained_dict)
        self.load_state_dict(finetune_dict)

    def forward(self, eeg):
        x = self.conv1(eeg)
        x = self.batchnorm1(x)
        pred = self.ViT.forward(x).logits
        return pred
