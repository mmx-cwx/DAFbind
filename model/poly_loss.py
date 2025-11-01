import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedPolyLoss(nn.Module):
    def __init__(self, epsilon=1e-2, power=2, weight=None):
        """
        :param epsilon: 修正项的系数
        :param power: 修正项的阶数
        :param weight: 类别权重，用于交叉熵部分
        """
        super(WeightedPolyLoss, self).__init__()
        self.epsilon = epsilon
        self.power = power
        self.weight = weight
        
    def forward(self, inputs, targets):
        """
        :param inputs: 模型输出 logits, shape: [batch_size, num_classes]
        :param targets: 标签, shape: [batch_size]
        """
        # 交叉熵损失（带权重）

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none').to('cuda')
        # 预测概率
        probs = F.softmax(inputs, dim=-1)
        pt = probs[torch.arange(inputs.size(0)), targets]
        # Poly Loss
        poly_loss = ce_loss + self.epsilon * (1 - pt) ** self.power
        return poly_loss.mean()

