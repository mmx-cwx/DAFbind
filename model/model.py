
import math
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer

from attention import *


# DNA dataset
class m_model_esm_650m_64(nn.Module):
    def __init__(self, num_heads, pretrain_model_path, num_att, dif_att):
        super().__init__()
        self.pretrain_model_path = pretrain_model_path
        self.num_heads = num_heads
        self.num_att = num_att
        self.dif_att = dif_att
        self.relu = nn.ReLU(True)

        self.feature_att = nn.ModuleList([
            feature_axial_MultiHeadSelfAttention(embed_dim=1280, num_heads=self.num_heads, key_size=1280,
                                                 value_size=1280) for i in range(self.num_att)
        ])

        self.esm2 = AutoModel.from_pretrained(self.pretrain_model_path)
        #
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 3], padding=[0, 1], stride=1)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 5], padding=[0, 2], stride=1)
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 7], padding=[0, 3], stride=1)

        self.cnn00 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 3], padding=[0, 1], stride=1)
        self.cnn11 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 5], padding=[0, 2], stride=1)
        self.cnn22 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 7], padding=[0, 3], stride=1)

        # self.bigrucell = nn.GRU(768, 128, bidirectional=True)  # 双向GRU

        self.num_layers = 8
        self.hidden_size = 640
        with torch.no_grad():
            old_position_embeddings = self.esm2.embeddings.position_embeddings.weight.clone().detach()
            new_position_embeddings = self.esm2.embeddings.position_embeddings_test.weight.clone().detach()
            alpha = torch.tensor(0.4)
            for j in range(2052):
                x = j // 1024 + 1
                y = j % 1024
                new_position_embeddings[j] = alpha * old_position_embeddings[x] + (1 - alpha) * \
                                             old_position_embeddings[y]

        self.esm2.embeddings.position_embeddings_test.weight = nn.Parameter(
            new_position_embeddings.clone().detach().requires_grad_(True))

        # 冻结模型的参数
        # for param in self.esm2.parameters():
        #     param.requires_grad = False

        self.fnn3 = nn.Sequential(
            nn.Linear((1280 + 256), 512),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True)
        )

        self.prot = nn.ModuleList([
            feature_cross_fuse_Attention_diffusion_f_1_use_bilstm(embed_dim=256, input_dim_a=1024, input_dim_b=1280, hidden_dim_a=32,
                                                   hidden_dim_b=32) for i in range(0, self.dif_att)
        ]
        )

    def forward(self, inputs, noise=None, seq='', prot_feature='',fold=0):

        model_out = self.esm2(**inputs).last_hidden_state[:, 1:-1]
        prot_feature = prot_feature.unsqueeze(dim=0)

        loss = 0.0

        tmp = model_out

        for i, att in enumerate(self.feature_att):
            model_out = att(model_out)

        for i, att in enumerate(self.prot):
            prot_feature, l = att(prot_feature, model_out)
            loss = loss + l

        model_out = torch.cat((model_out, prot_feature), dim=-1)
        tmp = torch.cat((tmp, prot_feature), dim=-1)

        cnn_out = self.cnn0(model_out) + self.cnn1(model_out) + self.cnn2(model_out)
        cnn_out = cnn_out.mean(dim=0, keepdim=True)

        cnn_out1 = self.cnn00(tmp) + self.cnn11(tmp) + self.cnn22(tmp)
        cnn_out1 = cnn_out1.mean(dim=0, keepdim=True)

        return self.fnn3(cnn_out + cnn_out1), loss

# RNA dataset
class m_model_esm_650m_32(nn.Module):
    def __init__(self, num_heads, pretrain_model_path, num_att, dif_att):
        super().__init__()
        self.pretrain_model_path = pretrain_model_path
        self.num_heads = num_heads
        self.num_att = num_att
        self.hhm_att = dif_att
        self.relu = nn.ReLU(True)

        self.feature_att = nn.ModuleList([
            feature_axial_MultiHeadSelfAttention(embed_dim=1280, num_heads=self.num_heads, key_size=1280,
                                                 value_size=1280) for i in range(self.num_att)
        ])
        self.esm2 = AutoModel.from_pretrained(self.pretrain_model_path)
        #
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 3], padding=[0, 1], stride=1)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 5], padding=[0, 2], stride=1)
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 7], padding=[0, 3], stride=1)

        self.cnn00 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 3], padding=[0, 1], stride=1)
        self.cnn11 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 5], padding=[0, 2], stride=1)
        self.cnn22 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[1, 7], padding=[0, 3], stride=1)

        # self.bigrucell = nn.GRU(768, 128, bidirectional=True)  # 双向GRU

        self.num_layers = 8
        self.hidden_size = 640
        with torch.no_grad():
            old_position_embeddings = self.esm2.embeddings.position_embeddings.weight.clone().detach()
            new_position_embeddings = self.esm2.embeddings.position_embeddings_test.weight.clone().detach()
            alpha = torch.tensor(0.4)
            for j in range(2052):
                x = j // 1024 + 1
                y = j % 1024
                new_position_embeddings[j] = alpha * old_position_embeddings[x] + (1 - alpha) * \
                                             old_position_embeddings[y]

        self.esm2.embeddings.position_embeddings_test.weight = nn.Parameter(
            new_position_embeddings.clone().detach().requires_grad_(True))

        # 冻结模型的参数
        # for param in self.esm2.parameters():
        #     param.requires_grad = False

        self.fnn3 = nn.Sequential(
            nn.Linear((1280 + 256), 512),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True)
        )

        self.prot = nn.ModuleList([
            feature_cross_fuse_Attention_diffusion_f_1_use_bilstm(embed_dim=256, input_dim_a=1024, input_dim_b=1280, hidden_dim_a=16,
                                                   hidden_dim_b=16) for i in range(0, self.dif_att)
        ]
        )

    def forward(self, inputs, noise=None, seq='', prot_feature=''):

        model_out = self.esm2(**inputs).last_hidden_state[:, 1:-1]
        prot_feature = prot_feature.unsqueeze(dim=0)
        loss = 0.0

        tmp = model_out

        for i, att in enumerate(self.feature_att):
            model_out = att(model_out)

        for i, att in enumerate(self.prot):
            prot_feature, l = att(prot_feature, model_out)
            loss = loss + l

        model_out = torch.cat((model_out, prot_feature), dim=-1)
        tmp = torch.cat((tmp, prot_feature), dim=-1)

        cnn_out = self.cnn0(model_out) + self.cnn1(model_out) + self.cnn2(model_out)
        cnn_out = cnn_out.mean(dim=0, keepdim=True)

        cnn_out1 = self.cnn00(tmp) + self.cnn11(tmp) + self.cnn22(tmp)
        cnn_out1 = cnn_out1.mean(dim=0, keepdim=True)

        return self.fnn3(cnn_out + cnn_out1), loss
