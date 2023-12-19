# modified from "https://github.com/ruiqiRichard/EEGViT/blob/master/models/EEGViT_pretrained.py"

import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers


class EEGViT_MAE_pretrained(nn.Module):
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
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129,14)})
        config.update({'patch_size': (8,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)

        d_model = 768  # This should match the output dimension of the encoder
        nhead = 8  # Number of heads for multi-head attention
        num_layers = 2  # Number of transformer layers
        output_dim = 129 * 500  # Output dimension
        
        model.classifier = TransformerDecoder(d_model, nhead, num_layers, output_dim)

        self.ViT = model

    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, output_dim):
        super(TransformerDecoder, self).__init__()
        
        # EEGViT-MAE Decoder Layer
        decoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_dim)
        
    def forward(self, x):
        x = self.transformer_decoder(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
