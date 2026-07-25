"""动态焦点损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 3. 动态焦点损失函数
# ==============================

class DynamicFocalLoss(nn.Module):
    """
    动态焦点损失函数 (Dynamic Focal Loss)

    描述:
    - 根据训练 epoch 调整 γ，早期接近加权交叉熵
    - 后期增大 γ；样本难度由标准 focal 因子 (1-p_t)^γ 自适应处理
    - 对违约类(y=1)增强关注
    """

    def __init__(
        self,
        alpha_pos=0.75,
        alpha_neg=0.25,
        gamma_base=0.0,
        gamma_max=2.0,
        num_epoch=100,
        schedule_power=1.0,
    ):
        super().__init__()
        if not 0.0 < alpha_pos < 1.0 or not 0.0 < alpha_neg < 1.0:
            raise ValueError("alpha_pos and alpha_neg must be in (0, 1).")
        if gamma_base < 0.0 or gamma_max < gamma_base:
            raise ValueError("Require 0 <= gamma_base <= gamma_max.")
        if num_epoch < 1:
            raise ValueError("num_epoch must be at least 1.")
        if schedule_power <= 0.0:
            raise ValueError("schedule_power must be positive.")

        self.alpha_pos = float(alpha_pos)  # 正类（违约）全局权重
        self.alpha_neg = float(alpha_neg)  # 负类（正常）全局权重
        self.gamma_base = float(gamma_base)
        self.gamma_max = float(gamma_max)
        self.num_epoch = int(num_epoch)
        self.schedule_power = float(schedule_power)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前 epoch 以调整 gamma"""
        self.current_epoch = max(int(epoch), 0)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] 模型输出(logits)
            targets: [N] 真实标签
        """
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("DynamicFocalLoss expects binary logits with shape [N, 2].")
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("targets must have shape [N].")

        # 在 float32 中使用 log_softmax，避免 softmax 下溢以及 log(p + eps)
        # 截断极难样本的梯度。
        log_probs = F.log_softmax(inputs.float(), dim=-1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()

        progress = min(
            ((self.current_epoch + 1) / self.num_epoch) ** self.schedule_power,
            1.0,
        )
        # gamma 只按 epoch 调度；(1-p_t)^gamma 本身已经根据预测难度
        # 调权，无需再令 gamma 依赖 p_t。
        gamma = self.gamma_base + (
            self.gamma_max - self.gamma_base
        ) * progress

        # 类别权重在整个训练集上一次性确定，避免随随机 mini-batch 抖动。
        alpha_t = torch.where(
            targets == 1,
            log_p_t.new_tensor(self.alpha_pos),
            log_p_t.new_tensor(self.alpha_neg),
        )
        focal_base = (1.0 - p_t).clamp_min(
            torch.finfo(p_t.dtype).eps
        )
        focal_weight = focal_base.pow(gamma)
        loss = -alpha_t * focal_weight * log_p_t

        return loss.mean()

