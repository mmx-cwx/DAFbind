import math
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from diffusion import *

class feature_axial_MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, key_size, value_size, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_head_dim = key_size // num_heads
        self.k_head_dim = key_size // num_heads
        self.v_head_dim = value_size // num_heads

        self.W_q = nn.Linear(embed_dim, key_size, bias=bias)
        self.W_k = nn.Linear(embed_dim, key_size, bias=bias)
        self.W_v = nn.Linear(embed_dim, value_size, bias=bias)

        self.out_proj = nn.Linear(value_size, embed_dim, bias=bias)


    def forward(self, x):
        q  = self.W_q(x)  # (N, L, key_size)
        k  = self.W_k(x)  # (N, L, key_size)
        v  = self.W_v(x)  # (N, L, value_size)
        att = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(k.size(-1))

        att = F.softmax(att, dim=-1)
        output = torch.matmul(att, v.transpose(-1, -2)).transpose(-1, -2)
        output = self.out_proj(output)
        return output

class feature_cross_fuse_Attention_diffusion_f_1_use_bilstm(nn.Module):
    def __init__(self, embed_dim=512,bias = False, input_dim_a=512,input_dim_b=512,hidden_dim_a=256,hidden_dim_b=512):
        super().__init__()

        # 双向 LSTM 用于模态 A
        self.bilstm_a = nn.LSTM(input_dim_a, hidden_dim_a, batch_first=True, bidirectional=True)
        # 单向 LSTM 用于模态 B
        self.lstm_b = nn.LSTM(input_dim_b, hidden_dim_b, batch_first=True ,bidirectional=True)

        self.out_proj = nn.Linear(hidden_dim_b*2, embed_dim, bias=bias)


        self.attention_matrix = hidden_dim_b

        self.diffusion_model1 = MLPDiffusion(n_steps=n_steps, input_dim=int(self.attention_matrix * self.attention_matrix*4),
                                             hidden_dim=int(self.attention_matrix * self.attention_matrix*2)).to('cuda')
        self.diffusion_model2 = MLPDiffusion(n_steps=n_steps, input_dim=int(self.attention_matrix * self.attention_matrix*4),
                                             hidden_dim=int(self.attention_matrix * self.attention_matrix*2)).to('cuda')
        self.diffusion_model3 = MLPDiffusion(n_steps=n_steps, input_dim=int(self.attention_matrix * self.attention_matrix*4),
                                             hidden_dim=int(self.attention_matrix * self.attention_matrix*2)).to('cuda')
        self.diffusion_model4 = MLPDiffusion(n_steps=n_steps, input_dim=int(self.attention_matrix * self.attention_matrix*4),
                                             hidden_dim=int(self.attention_matrix * self.attention_matrix*2)).to('cuda')

        self.fusion_linear = nn.Sequential(
            nn.Linear(diffusion_N * (int (self.attention_matrix * 2)), 512 ),
            nn.ReLU(inplace=True),
            nn.Linear(512, int(self.attention_matrix * 2)),
        )
    def forward(self, x, y):

        # BiLSTM 处理模态 A
        output_a, (h_a, c_a) = self.bilstm_a(x)  # h_a 和 c_a 的形状是 (2, batch_size, hidden_dim)
        # 将双向 LSTM 的隐藏状态（2 * hidden_dim）映射到单向 LSTM 的维度

        # 将模态 A 的 BiLSTM 输出作为模态 B 的 LSTM 初始隐藏状态
        output_b, _ = self.lstm_b(y, (h_a,c_a))
        k = v = output_a
        q = output_b
        att = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(k.size(-1))
        # 添加扩散模型
        loss = 0.0
        input2, l2 = diffusion_loss_fn_1(self.diffusion_model1, att.squeeze(dim=0), alphas_bar_sqrt,
                                       one_minus_alphas_bar_sqrt, 0,
                                       int(1 * n_steps / diffusion_N))
        loss += l2.to('cpu')
        input3, l3 = diffusion_loss_fn_1(self.diffusion_model2, att.squeeze(dim=0), alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                       int(1 * n_steps / diffusion_N), int(2 * n_steps / diffusion_N))
        loss += l3.to('cpu')
        input4, l4 = diffusion_loss_fn_1(self.diffusion_model3, att.squeeze(dim=0), alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                       int(2 * n_steps / diffusion_N), int(3 * n_steps / diffusion_N))
        loss += l4.to('cpu')
        input5, l5 = diffusion_loss_fn_1(self.diffusion_model4, att.squeeze(dim=0), alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                       int(3 * n_steps / diffusion_N), int(4 * n_steps / diffusion_N))
        loss += l5.to('cpu')
        input2 = torch.cat([input2, input3, input4, input5], dim=-1).unsqueeze(dim=0)
        input2 = self.fusion_linear(input2)
        att = F.softmax(input2, dim=-1)
        output = torch.matmul(att,v.transpose(-1, -2)).transpose(-1, -2)
        output = self.out_proj(output)
        return output ,loss