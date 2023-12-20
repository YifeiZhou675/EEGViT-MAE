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

        model.classifier = Decoder(d_model, nhead, num_layers, output_dim)

        self.ViT = model


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, output_dim):
        super().__init__()

        decoder_embed_dim = 512
        self.decoder_embed = nn.Linear(d_model, decoder_embed_dim, bias=True)
        decoder_layers = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(decoder_embed_dim, d_model, bias=True)
        self.fc2 = nn.Linear(d_model, 2048, bias=True)
        self.decoder_pred = nn.Linear(2048, output_dim, bias=True)

    def forward(self, x):
        x = self.decoder_embed(x)
        x = self.decoder(x)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = nn.Dropout(0.1)(x)
        x = self.fc2(x)
        x = nn.GELU()(x)
        x = nn.Dropout(0.1)(x)
        x = self.decoder_pred(x)
        return x

