import torch
from torch import nn as nn

#扩散模型
# 确定超参数的值
diffusion_N = 4
n_steps = 600

## 制定每一步的betas
betas = torch.linspace(-6, 6, n_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
betas = betas.to('cuda')
## 计算 alpha, alpha_prod, alpha_prod_pervious, alpha_bar_sqrt 等变量的值

alphas = (1 - betas).to('cuda').detach()
alphas_prod = torch.cumprod(alphas, 0).to('cuda').detach()
alphas_bar_sqrt = torch.sqrt(alphas_prod).to('cuda').detach()
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to('cuda').detach()

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps,input_dim, hidden_dim):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, hidden_dim),
                nn.Embedding(n_steps, hidden_dim),
                nn.Embedding(n_steps, hidden_dim),
            ]
        )

    def forward(self, x, t):
        #  x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


def diffusion_loss_fn_1(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, start,n_steps):

    # 生成时间步
    t = torch.randint(start, n_steps, (1,)).to('cuda')
    # 对称扩展时间步
    #t = torch.cat([t, n_steps - 1 - t[:batch_size - t_half]], dim=0)
    t = t.unsqueeze(-1)
    # x0的系数
    t =t.to('cpu')
    weidu = x_0.shape[0]
    a = alphas_bar_sqrt[t]   # torch.Size([batchsize, 1])
    x_0 = x_0.flatten()

    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t].to('cuda') # torch.Size([batchsize, 1])

    # 生成随机噪音eps
    e = torch.randn(1,x_0.shape[0]).to('cuda')

    # 构造模型的输入
    x = x_0.to('cuda') * a.to('cuda') + e.to('cuda') * aml.to('cuda')  # torch.Size([batchsize, 2])
    # 送入模型，得到t时刻的随机噪声预测值
    output1 = model(x.to('cuda'), t.squeeze(-1).to('cuda')).to('cuda')  # t.squeeze(-1)为torch.Size([batchsize])
    output = (x-aml*output1)/a
    return output.reshape(weidu,-1).to('cuda'),(e - output1).square().mean()
