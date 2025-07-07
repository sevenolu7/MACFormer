import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from Model_GLAM import GLAM
from Model_MSAF import MSAF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, ip_dim, block_channel=8, block_dropout=0.2, reduction_factor=4, split_block=5):
        super(Model, self).__init__()

        # taclBERT 模块
        self.bert = BertModel.from_pretrained("tacl-bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = True

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

        # 全连接层
        self.fc_url = nn.Linear(768, 128)
        self.fc_ip = nn.Linear(128, 128)

        self.model_glam = GLAM(in_channels=12, seq_len=200, feature_dim=768)

        # MSAF 模块，用于多模态融合
        self.msaf = MSAF(in_channels=[128, 128], block_channel=block_channel, block_dropout=block_dropout,
                         reduction_factor=reduction_factor, split_block=split_block)

        # 输出层
        self.fc_output = nn.Linear(256, 2)

    def forward(self, x, ip_embeds):
        # BERT 特征提取
        context, types, mask = x[0], x[1], x[2]
        outputs, _ = self.bert(input_ids=context, token_type_ids=types, attention_mask=mask,
                               output_all_encoded_layers=True)

        # 获取层次特征
        pyramid = torch.stack(outputs, dim=0).permute(1, 0, 2, 3)  # [batch16, 12, 200, 768]
        url_features = self.model_glam(pyramid)  # [batch, 768]
        url_features = self.fc_url(url_features)  # [batch, 128]
        url_features = url_features.unsqueeze(2)  # 添加一个长度维度，形状变为 [batch, 128, 1]

        ip_features = F.relu(self.fc_ip(ip_embeds))  # [batch, 128]
        ip_features = ip_features.unsqueeze(2)  # [batch, 128, 1]

        fused_features = self.msaf([url_features, ip_features])
        # print("MSAF Output Shapes:")
        # for i, feature in enumerate(fused_features):
        #     print(f"Modality {i}: {feature.shape}")

        fused_features = torch.cat(fused_features, dim=1)  # 拼接特征
        # print("Fused Features Shape:", fused_features.shape)

        fused_features = self.dropout(fused_features)
        out = self.fc_output(fused_features)

        return out
