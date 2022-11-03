import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from lib.model import resnet
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PIZZA(nn.Module):
    def __init__(self, backbone, img_feature_dim, multi_frame, with_translation=True,
                 dim_feedforward_transformer=512, num_heads_transformer=8,
                 num_encoder_layers_transformer=1, dropout_transformer=0.0):
        super(PIZZA, self).__init__()
        # RGB image encoder
        if backbone == "resnet18":
            self.backbone = resnet.resnet18(pretrained=False, num_classes=img_feature_dim)
        if backbone == "resnet50":
            self.backbone = resnet.resnet50(pretrained=False, num_classes=img_feature_dim)
        # MLP for rotation
        self.MLP_rotation = nn.Sequential(nn.Linear(img_feature_dim + img_feature_dim, 800),
                                          nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                          nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                          nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True),
                                          nn.Linear(200, 3))
        self.with_translation = with_translation
        if self.with_translation:
            # MLP for translation
            self.MLP_translation = nn.Sequential(nn.Linear(img_feature_dim + img_feature_dim, 800),
                                                 nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                                 nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                                 nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
            self.fc_translation2d = nn.Linear(200, 2)
            self.fc_depth = nn.Linear(200, 1)
        else:
            self.MLP_translation = None
            self.fc_translation2d = None
            self.fc_depth = None
        # Transformer
        self.multi_frame = multi_frame
        if self.multi_frame:
            encoder_layer = TransformerEncoderLayer(img_feature_dim, num_heads_transformer,
                                                    dim_feedforward_transformer, dropout_transformer)
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers_transformer)
        else:
            self.transformer_encoder = None
        self.img_feature_dim = img_feature_dim

    def forward(self, images):
        # Image is of dimension BxLx3x224x224
        [batch_size, len_sequences, img_channel, image_height, image_width] = images.shape
        images = images.view(batch_size * len_sequences, img_channel, image_height, image_width)
        img_embedding = self.backbone(images).view(batch_size, len_sequences, self.img_feature_dim)
        if self.multi_frame:
            img_embedding = self.transformer_encoder(img_embedding)
        img_embedding_previous = img_embedding[:, :-1, :]
        img_embedding = torch.cat((img_embedding_previous, img_embedding[:, 1:, :]), axis=2)
        img_embedding = img_embedding.view(batch_size * (len_sequences - 1), 2 * self.img_feature_dim)
        rot = self.MLP_rotation(img_embedding).view(batch_size, len_sequences - 1, 3)
        if self.with_translation:
            translation_embed = self.MLP_translation(img_embedding)
            delta_uv = self.fc_translation2d(translation_embed).view(batch_size, len_sequences - 1, 2)
            delta_depth = self.fc_depth(translation_embed).view(batch_size, len_sequences - 1, 1)
            return rot, delta_uv, delta_depth
        else:
            return rot


if __name__ == '__main__':
    model = PIZZA(backbone="resnet18", backbone_pretrained=True, img_feature_dim=512, multi_frame=False).cuda()
    imgs = Variable(torch.rand(10, 8, 3, 224, 224)).cuda()
    pred_axis_angles = model(imgs)
    print("axis_angles", pred_axis_angles.size())