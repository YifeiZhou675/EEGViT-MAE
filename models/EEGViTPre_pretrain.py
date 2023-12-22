from torch import nn
from transformers import ViTForImageClassification, ViTConfig


class EEGViTPre_pretrain(nn.Module):
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

        model_name = "google/vit-base-patch16-224"
        config = ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 14)})
        config.update({'patch_size': (8, 1)})

        model = ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)

        d_model = 768  # This should match the output dimension of the encoder
        nhead = 8  # Number of heads for multi-head attention
        num_layers = 2  # Number of transformer layers
        output_dim = 129 * 500  # Output dimension

        if num_layers:
            model.classifier = Decoder(d_model, nhead, num_layers, output_dim)
        else:
            model.classifier = nn.Sequential(
                nn.Linear(d_model, d_model//2, bias=True),
                nn.ReLU(),
                nn.Linear(d_model//2, output_dim, bias=True)
            )

        self.ViT = model

    def forward_loss(self, eeg, pred, mask):
        target = eeg.squeeze()
        pred = pred.squeeze()
        mask = mask.squeeze()

        loss = (pred - target) ** 2
        loss = (loss * (1 - mask)).sum() / (1 - mask).sum()
        return loss

    def forward(self, eeg, mask):
        masked_eeg = eeg * mask
        x = self.conv1(masked_eeg)
        x = self.batchnorm1(x)
        pred = self.ViT.forward(x).logits
        pred = pred.view(eeg.shape)
        loss = self.forward_loss(eeg, pred, mask)
        return loss


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, output_dim):
        super().__init__()

        decoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(d_model, d_model//2, bias=True)
        self.decoder_pred = nn.Linear(d_model//2, output_dim, bias=True)

    def forward(self, x):
        x = self.decoder(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        pred = self.decoder_pred(x)
        return pred

