"""模型架构定义"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 2. 模型架构定义
# ==============================

class TSCrossAttention(nn.Module):
    """
    时序-静态特征交叉注意力机制 (TS-CrossAttention)

    静态特征作为 Query 查询时序关键节点；时序摘要作为 Query 反向查询
    静态特征，最后拼接两个方向的上下文表示。
    """

    def __init__(self, static_dim, temporal_dim, hidden_dim=64, num_heads=4,
                 dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for multi-head attention.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.static_query_proj = nn.Linear(static_dim, hidden_dim)
        self.temporal_token_proj = nn.Linear(temporal_dim, hidden_dim)
        self.temporal_query_proj = nn.Linear(temporal_dim, hidden_dim)
        self.static_key_proj = nn.Linear(static_dim, hidden_dim)
        self.static_value_proj = nn.Linear(static_dim, hidden_dim)

        self.static_to_temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 单令牌静态 Key/Value 使得 softmax 注意力恒等于 1
        # 使用时序条件门控，确保反向时序交互具有实际意义
        self.reverse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, static_feat, temporal_feat):
        """
        Args:
            static_feat: [batch, static_dim]
            temporal_feat: [batch, time_steps, temporal_dim]
        Returns:
            fused_feat: [batch, hidden_dim]
            attention_weights: [batch, context_steps]
        """
        temporal_tokens = self.temporal_token_proj(temporal_feat)
        q_static = self.static_query_proj(static_feat).unsqueeze(1)
        c_s2x, alpha = self.static_to_temporal_attn(
            q_static,
            temporal_tokens,
            temporal_tokens,
            need_weights=True
        )

        temporal_summary = temporal_feat.mean(dim=1)
        q_temporal = self.temporal_query_proj(temporal_summary).unsqueeze(1)
        static_key = self.static_key_proj(static_feat).unsqueeze(1)
        static_value = self.static_value_proj(static_feat).unsqueeze(1)
        reverse_gate = self.reverse_gate(
            torch.cat([q_temporal.squeeze(1), static_key.squeeze(1)], dim=-1)
        )
        c_x2s = (reverse_gate * static_value.squeeze(1)).unsqueeze(1)

        fused = torch.cat([c_s2x.squeeze(1), c_x2s.squeeze(1)], dim=-1)
        return self.output_proj(fused), alpha.squeeze(1)


class NonlinearGlobalLocalFusion(nn.Module):
    """
    非线性全局-局部门控融合（NGL-GF）。

    全局分支通过多统计量非线性聚合建模长期信用状态；局部分支仅在固定
    因果窗口内计算加性注意力，并显式编码相邻时间步突变。最终以门控
    残差方式注入原始时序，因此不会用单一聚合向量替代或抹平原始序列。

    局部注意力权重形状为 [B, T, W]，W 为固定窗口，避免 T x T 矩阵。
    """

    def __init__(self, hidden_dim, window_size=3, dropout=0.1,
                 use_global_context=True, use_local_attention=True):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if not (use_global_context or use_local_attention):
            raise ValueError("At least one fusion branch must be enabled.")

        self.hidden_dim = hidden_dim
        self.window_size = int(window_size)
        self.use_global_context = use_global_context
        self.use_local_attention = use_local_attention
        # 低秩瓶颈控制参数量；在 d=64/128 时显著少于旧版多头交叉注意力。
        bottleneck = max(hidden_dim // 4, 8)

        if use_global_context:
            # mean/max/last/std/trend + static context + multi-scale context
            self.global_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 7, bottleneck),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else:
            self.global_encoder = None

        if use_local_attention:
            # [current, signed change, absolute change] preserves local shocks.
            self.event_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 3, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.local_query = nn.Linear(hidden_dim * 3, bottleneck)
            self.local_key = nn.Linear(hidden_dim, bottleneck)
            self.local_score = nn.Linear(bottleneck, 1, bias=False)
            self.local_value = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.event_encoder = None
            self.local_query = None
            self.local_key = None
            self.local_score = None
            self.local_value = None

        gate_input_dim = hidden_dim * 5 + 1
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim * 2)
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _temporal_statistics(temporal_feat):
        mean = temporal_feat.mean(dim=1)
        maximum = temporal_feat.max(dim=1).values
        last = temporal_feat[:, -1]
        std = temporal_feat.std(dim=1, unbiased=False)
        trend = temporal_feat[:, -1] - temporal_feat[:, 0]
        return mean, maximum, last, std, trend

    @staticmethod
    def _causal_window_mask(time_steps, window_size, device):
        positions = torch.arange(window_size, device=device).unsqueeze(0)
        first_valid = window_size - 1 - torch.arange(
            time_steps, device=device
        ).unsqueeze(1)
        return positions >= first_valid

    def _local_context(self, temporal_feat, static_feat, global_context):
        batch_size, time_steps, hidden_dim = temporal_feat.shape

        delta = torch.zeros_like(temporal_feat)
        if time_steps > 1:
            delta[:, 1:] = temporal_feat[:, 1:] - temporal_feat[:, :-1]
        # t=0 的 delta 恒为 0；直接 sqrt(0) 的反向导数为无穷，会在第一
        # 个优化步骤后污染全部参数。加小量保证梯度有限。
        shock_score = (delta.square().mean(dim=-1) + 1e-8).sqrt()

        event_tokens = self.event_encoder(
            torch.cat([temporal_feat, delta, delta.abs()], dim=-1)
        )

        window_size = min(self.window_size, time_steps)
        padded = F.pad(event_tokens, (0, 0, window_size - 1, 0))
        # unfold returns [B, T, D, W]; move the local-window axis before D.
        windows = padded.unfold(1, window_size, 1).permute(0, 1, 3, 2)

        static_context = static_feat.unsqueeze(1).expand(
            batch_size, time_steps, hidden_dim
        )
        global_expanded = global_context.unsqueeze(1).expand(
            batch_size, time_steps, hidden_dim
        )
        query = self.local_query(
            torch.cat([temporal_feat, static_context, global_expanded], dim=-1)
        ).unsqueeze(2)

        scores = self.local_score(
            torch.tanh(query + self.local_key(windows))
        ).squeeze(-1)
        valid_mask = self._causal_window_mask(
            time_steps, window_size, temporal_feat.device
        )
        scores = scores.masked_fill(
            ~valid_mask.unsqueeze(0),
            torch.finfo(scores.dtype).min
        )
        attention_weights = torch.softmax(scores, dim=-1)
        local_context = (
            attention_weights.unsqueeze(-1) * self.local_value(windows)
        ).sum(dim=2)
        return local_context, attention_weights, shock_score

    def forward(self, static_feat, temporal_feat, multi_scale_context=None):
        batch_size, time_steps, hidden_dim = temporal_feat.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )

        zeros = temporal_feat.new_zeros(batch_size, hidden_dim)
        if multi_scale_context is None:
            multi_scale_context = zeros

        if self.use_global_context:
            statistics = self._temporal_statistics(temporal_feat)
            global_context = self.global_encoder(
                torch.cat([
                    *statistics,
                    static_feat,
                    multi_scale_context
                ], dim=-1)
            )
        else:
            global_context = zeros

        if self.use_local_attention:
            local_context, local_weights, shock_score = self._local_context(
                temporal_feat, static_feat, global_context
            )
        else:
            local_context = torch.zeros_like(temporal_feat)
            local_weights = None
            delta = torch.zeros_like(temporal_feat)
            if time_steps > 1:
                delta[:, 1:] = temporal_feat[:, 1:] - temporal_feat[:, :-1]
            shock_score = (delta.square().mean(dim=-1) + 1e-8).sqrt()

        global_expanded = global_context.unsqueeze(1).expand_as(temporal_feat)
        gate_shock = (
            shock_score
            if self.use_local_attention
            else torch.zeros_like(shock_score)
        )
        gate_input = torch.cat([
            temporal_feat,
            local_context,
            global_expanded,
            temporal_feat * global_expanded,
            (temporal_feat - global_expanded).abs(),
            gate_shock.unsqueeze(-1)
        ], dim=-1)
        local_gate, global_gate = torch.sigmoid(
            self.gate_network(gate_input)
        ).chunk(2, dim=-1)

        if not self.use_local_attention:
            local_gate = torch.zeros_like(local_gate)
        if not self.use_global_context:
            global_gate = torch.zeros_like(global_gate)

        update = (
            local_gate * local_context
            + global_gate * global_expanded
        )
        fused_sequence = self.output_norm(
            temporal_feat + self.output_dropout(update)
        )
        diagnostics = {
            'local_attention': local_weights,
            'local_gate': (
                local_gate.mean(dim=-1)
                if self.use_local_attention else None
            ),
            'global_gate': (
                global_gate.mean(dim=-1)
                if self.use_global_context else None
            ),
            'shock_score': shock_score,
            'global_context_norm': global_context.norm(dim=-1)
        }
        return fused_sequence, diagnostics


class AdaptiveGatedResUnit(nn.Module):
    """
        自适应门控残差单元 (AG-ResUnit)

        描述:
        - 引入可学习参数λ动态调节遗忘门、输入门、输出门比例
        - 残差连接构建深层网络(8层)
        - 扩展长期记忆能力至36个月
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 四个标准门共用一次矩阵乘法，保持公式不变并显著减少 kernel launch。
        self.gate_projection = nn.Linear(
            input_dim + hidden_dim,
            hidden_dim * 4,
        )

        # 自适应门控系数 λ：由当前输入和历史状态动态生成，而不是全局常数
        self.lambda_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # 初始时接近标准门控，训练后再逐步学习更强的长期记忆偏置。
        nn.init.constant_(self.lambda_gate[0].bias, -1.5)

        # 残差投影 (当输入维度不等于隐藏维度时)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: 当前输入 [batch, input_dim]
            h_prev: 前一隐藏状态 [batch, hidden_dim]
            c_prev: 前一细胞状态 [batch, hidden_dim]
        """
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=-1)  # 在最后一维进行拼接

        # 原始门控计算；一次线性投影后再分块。
        f_raw, i_raw, o_raw, c_raw = self.gate_projection(combined).chunk(
            4,
            dim=-1,
        )
        f_tilde = torch.sigmoid(f_raw)
        i_tilde = torch.sigmoid(i_raw)
        o_tilde = torch.sigmoid(o_raw)

        # 自适应门控: λ 随样本和时间状态变化，λ 越大越偏记忆，越小越偏更新
        lambda_val = self.lambda_gate(combined)

        # 候选细胞状态
        c_tilde = torch.tanh(c_raw)

        # lambda 越大越偏向保留旧记忆；原实现只是交换 f/i 两个门，
        # 并不能保证注释所描述的“偏记忆/偏更新”行为。
        f_t = lambda_val + (1.0 - lambda_val) * f_tilde
        i_t = (1.0 - lambda_val) * i_tilde
        o_t = o_tilde

        # 细胞状态更新
        c_t = f_t * c_prev + i_t * c_tilde

        # 隐藏状态
        h_t = o_t * torch.tanh(c_t)

        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x if x.shape[-1] == self.hidden_dim else 0

        h_t = h_t + residual  # 残差相加
        h_t = self.layer_norm(h_t)

        return h_t, c_t


class AGBiLSTMLayer(nn.Module):
    """
    基于 AG-ResUnit 的 LSTM 层，可切换单向/双向以支持真实消融实验。
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.forward_layers = nn.ModuleList([
            AdaptiveGatedResUnit(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.backward_layers = nn.ModuleList([
            AdaptiveGatedResUnit(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ]) if bidirectional else None

        self.dropout = nn.Dropout(dropout)

    def _run_direction(self, x, layers, reverse=False):
        batch_size, time_steps, _ = x.shape

        def init_states():
            # 保持与输入相同的 dtype，避免自动混合精度下发生隐式类型提升。
            h = x.new_zeros((batch_size, self.hidden_dim))
            c = x.new_zeros((batch_size, self.hidden_dim))
            return h, c

        states = [init_states() for _ in range(self.num_layers)]
        outputs = []
        time_range = range(time_steps - 1, -1, -1) if reverse else range(time_steps)

        for t in time_range:
            layer_input = x[:, t, :]
            for layer_idx, layer in enumerate(layers):
                h, c = states[layer_idx]
                h_new, c_new = layer(layer_input, h, c)
                states[layer_idx] = (h_new, c_new)
                layer_input = h_new
            outputs.append(layer_input)

        if reverse:
            outputs = outputs[::-1]
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        """
        Args:
            x: [batch, time_steps, input_dim]
        Returns:
            output: [batch, time_steps, hidden_dim * directions]
        """
        forward_seq = self._run_direction(x, self.forward_layers, reverse=False)
        if self.bidirectional:
            backward_seq = self._run_direction(x, self.backward_layers, reverse=True)
            output = torch.cat([forward_seq, backward_seq], dim=-1)
        else:
            output = forward_seq

        return self.dropout(output)


class MultiScaleTemporalEncoder(nn.Module):
    """
    多尺度时序编码器
    - 细粒度（月度）：原始月频数据
    - 中粒度（季度）：3个月滑动平均
    - 粗粒度（年度）：12个月滑动窗口；不足12个月时退化为全局趋势摘要
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        coarse_window=12,
        max_steps=None,
        mode='legacy',
    ):
        super().__init__()
        if mode not in {'legacy', 'lightweight'}:
            raise ValueError("multiscale mode must be 'legacy' or 'lightweight'.")
        self.coarse_window = coarse_window
        self.max_steps = max_steps
        self.mode = mode
        self.output_dim = hidden_dim * 2

        if mode == 'legacy':
            self.fine_encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.medium_encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.coarse_encoder = (
                nn.LSTM(
                    input_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
                if max_steps is None or max_steps >= coarse_window
                else None
            )
        # 短序列回退方案，避免单时间步输入BiLSTM
            self.coarse_fallback = nn.Linear(
                input_dim * 2,
                hidden_dim * 2,
            )
        else:
            statistic_dim = input_dim * 3
            self.fine_projection = nn.Sequential(
                nn.Linear(statistic_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
            )
            self.medium_filter = nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size=3,
                groups=input_dim,
                bias=False,
            )
            nn.init.constant_(self.medium_filter.weight, 1.0 / 3.0)
            self.medium_projection = nn.Sequential(
                nn.Linear(statistic_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
            )
            self.coarse_projection = nn.Sequential(
                nn.Linear(statistic_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
            )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
        )

    @staticmethod
    def _moving_average(x, window, padding=0):
        if padding:
            left = padding
            right = window - 1 - left
            x = F.pad(
                x.transpose(1, 2),
                (left, right),
                mode='replicate',
            ).transpose(1, 2)
            padding = 0
        return nn.functional.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=window,
            stride=1,
            padding=padding
        ).transpose(1, 2)

    @staticmethod
    def _bidirectional_state(hidden_state):
        """使用 BiLSTM 的完整前向和后向历史信息"""
        return torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1)

    @staticmethod
    def _statistics(x):
        time_steps = x.shape[1]
        level = x.mean(dim=1)
        recent = x[:, -1, :]
        trend = (x[:, -1, :] - x[:, 0, :]) / max(time_steps - 1, 1)
        return torch.cat([level, recent, trend], dim=-1)

    def forward(self, x):
        """
        Args:
            x: [batch, time_steps, input_dim]
        """
        _, time_steps, _ = x.shape

        if self.mode == 'lightweight':
            fine_repr = self.fine_projection(self._statistics(x))
            if time_steps >= 3:
                padded = F.pad(
                    x.transpose(1, 2),
                    (1, 1),
                    mode='replicate',
                )
                medium_x = self.medium_filter(padded).transpose(1, 2)
            else:
                medium_x = x
            medium_repr = self.medium_projection(
                self._statistics(medium_x)
            )
            coarse_repr = self.coarse_projection(self._statistics(x))
            return self.fusion(torch.cat(
                [fine_repr, medium_repr, coarse_repr],
                dim=-1,
            ))

        _, (fine_hidden, _) = self.fine_encoder(x)
        fine_repr = self._bidirectional_state(fine_hidden)

        if time_steps >= 3:
            medium_x = self._moving_average(x, window=3, padding=1)
            _, (medium_hidden, _) = self.medium_encoder(medium_x)
            medium_repr = self._bidirectional_state(medium_hidden)
        else:
            medium_repr = fine_repr

        if time_steps >= self.coarse_window and self.coarse_encoder is not None:
            coarse_x = self._moving_average(x, window=self.coarse_window, padding=0)
            _, (coarse_hidden, _) = self.coarse_encoder(coarse_x)
            coarse_repr = self._bidirectional_state(coarse_hidden)
        else:
            level = x.mean(dim=1, keepdim=True)
            trend = (x[:, -1:, :] - x[:, :1, :]) / max(time_steps - 1, 1)
            coarse_stats = torch.cat([level, trend], dim=-1)
            coarse_repr = self.coarse_fallback(coarse_stats.squeeze(1))

        multi_scale = torch.cat([fine_repr, medium_repr, coarse_repr], dim=-1)
        return self.fusion(multi_scale)


class AABiLSTM(nn.Module):
    """
    非线性全局-局部自适应双向 LSTM 融合模型。

    完整架构:
    1. 特征编码层（静态 Embedding + 多尺度时序编码）
    2. 非线性全局聚合 + 局部冲击注意力 + 门控残差融合
    3. AG-ResUnit 堆叠层（可扩展至 8 层双向 LSTM）
    4. 预测输出层（Dynamic Focal Loss 在 Trainer 中启用）

    fusion_type='global_local' 为推荐实现；'legacy_cross' 保留旧版
    TS-CrossAttention，便于兼容历史实验和进行同条件对比。
    """

    def __init__(self, static_dim, temporal_dim, temporal_steps, hidden_dim=128, num_layers=8, num_classes=2,
                 dropout=0.3, num_heads=4, use_cross_attention=True, use_ag_resunit=True,
                 bidirectional=True, use_multiscale=True, fusion_type='global_local',
                 local_window=3, use_global_context=True, use_local_attention=True,
                 temporal_categorical_index=None, temporal_category_min=-2,
                 temporal_category_max=8, use_step_embedding=False,
                 multiscale_mode='legacy'):
        super().__init__()
        if fusion_type not in {'global_local', 'legacy_cross'}:
            raise ValueError(
                "fusion_type must be 'global_local' or 'legacy_cross'."
            )
        if (
            use_cross_attention
            and fusion_type == 'legacy_cross'
            and hidden_dim % num_heads != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible by num_heads for legacy attention."
            )
        if use_cross_attention and not (
            use_global_context or use_local_attention
        ) and fusion_type == 'global_local':
            raise ValueError(
                "Global-local fusion needs at least one enabled branch."
            )

        self.hidden_dim = hidden_dim
        self.temporal_steps = temporal_steps
        self.use_cross_attention = use_cross_attention
        self.fusion_type = fusion_type if use_cross_attention else 'none'
        self.use_ag_resunit = use_ag_resunit
        self.bidirectional = bidirectional
        self.use_multiscale = use_multiscale
        self.decision_threshold = 0.5
        self.temperature = 1.0
        self.calibration_scale = 1.0
        self.calibration_bias = 0.0
        self.entropy_threshold = 0.65
        self.temporal_categorical_index = temporal_categorical_index
        self.temporal_category_min = int(temporal_category_min)
        self.temporal_category_max = int(temporal_category_max)
        self.use_step_embedding = bool(use_step_embedding)
        if (
            temporal_categorical_index is not None
            and not 0 <= temporal_categorical_index < temporal_dim
        ):
            raise ValueError("temporal_categorical_index is out of range.")
        self.model_config = {
            'static_dim': int(static_dim),
            'temporal_dim': int(temporal_dim),
            'temporal_steps': int(temporal_steps),
            'hidden_dim': int(hidden_dim),
            'num_layers': int(num_layers),
            'num_classes': int(num_classes),
            'dropout': float(dropout),
            'num_heads': int(num_heads),
            'use_cross_attention': bool(use_cross_attention),
            'use_ag_resunit': bool(use_ag_resunit),
            'bidirectional': bool(bidirectional),
            'use_multiscale': bool(use_multiscale),
            'fusion_type': str(fusion_type),
            'local_window': int(local_window),
            'use_global_context': bool(use_global_context),
            'use_local_attention': bool(use_local_attention),
            'temporal_categorical_index': temporal_categorical_index,
            'temporal_category_min': int(temporal_category_min),
            'temporal_category_max': int(temporal_category_max),
            'use_step_embedding': bool(use_step_embedding),
            'multiscale_mode': str(multiscale_mode),
        }

        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim)
        self.temporal_category_embedding = None
        if temporal_categorical_index is not None:
            category_count = (
                self.temporal_category_max
                - self.temporal_category_min
                + 1
            )
            self.temporal_category_unknown = category_count
            self.temporal_category_embedding = nn.Embedding(
                category_count + 1,
                hidden_dim,
            )
            numeric_mask = torch.ones(temporal_dim, dtype=torch.float32)
            numeric_mask[temporal_categorical_index] = 0.0
            self.register_buffer(
                'temporal_numeric_mask',
                numeric_mask,
                persistent=True,
            )
        else:
            self.temporal_category_unknown = None
            self.register_buffer(
                'temporal_numeric_mask',
                torch.ones(temporal_dim, dtype=torch.float32),
                persistent=True,
            )
        self.step_embedding = (
            nn.Embedding(temporal_steps, hidden_dim)
            if use_step_embedding else None
        )
        self.temporal_input_norm = (
            nn.LayerNorm(hidden_dim)
            if temporal_categorical_index is not None or use_step_embedding
            else nn.Identity()
        )

        if use_multiscale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                temporal_dim,
                hidden_dim // 2,
                coarse_window=12,
                max_steps=temporal_steps,
                mode=multiscale_mode,
            )
            multi_scale_dim = self.temporal_encoder.output_dim
        else:
            self.temporal_encoder = None
            multi_scale_dim = 0
        self.multi_scale_dim = multi_scale_dim

        if use_cross_attention:
            if fusion_type == 'global_local':
                self.feature_fusion = NonlinearGlobalLocalFusion(
                    hidden_dim=hidden_dim,
                    window_size=min(local_window, max(temporal_steps, 1)),
                    dropout=dropout,
                    use_global_context=use_global_context,
                    use_local_attention=use_local_attention
                )
                self.cross_attention = None
            else:
                self.feature_fusion = None
                self.cross_attention = TSCrossAttention(
                    static_dim=hidden_dim,
                    temporal_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
        else:
            self.feature_fusion = None
            self.cross_attention = None

        self.multiscale_context_proj = (
            (
                nn.Identity()
                if multi_scale_dim == hidden_dim
                else nn.Linear(multi_scale_dim, hidden_dim)
            )
            if use_multiscale and use_cross_attention else None
        )
        self.cross_context_norm = (
            nn.LayerNorm(hidden_dim)
            if self.fusion_type == 'legacy_cross' else None
        )

        if use_ag_resunit:
            self.sequence_encoder = AGBiLSTMLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
            sequence_dim = self.sequence_encoder.output_dim
        else:
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.sequence_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=lstm_dropout
            )
            sequence_dim = hidden_dim * (2 if bidirectional else 1)
        self.sequence_dim = sequence_dim

        # 静态和多尺度表示保留直达分类头的跳连，避免它们只能经门控时序
        # 分支间接传播，在小样本 German 数据上尤其容易造成信息瓶颈。
        classifier_input_dim = sequence_dim + hidden_dim
        if use_multiscale:
            classifier_input_dim += multi_scale_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _encode_sequence(self, projected_temporal):
        if self.use_ag_resunit:
            return self.sequence_encoder(projected_temporal)
        sequence_out, _ = self.sequence_encoder(projected_temporal)
        return sequence_out

    def _pool_sequence(self, sequence_out):
        """对 BiLSTM 进行池化，同时保留完整的后向历史信息"""
        if self.bidirectional:
            forward_final = sequence_out[:, -1, :self.hidden_dim]
            backward_final = sequence_out[:, 0, self.hidden_dim:]
            return torch.cat([forward_final, backward_final], dim=-1)
        return sequence_out[:, -1, :]

    def forward(self, static_feat, temporal_feat, return_attention=False):
        """
        Args:
            static_feat: [batch, static_dim]
            temporal_feat: [batch, time_steps, temporal_dim]
        """
        static_emb = self.static_embedding(static_feat)
        temporal_numeric = temporal_feat * self.temporal_numeric_mask.view(
            1,
            1,
            -1,
        )
        projected_temporal = self.temporal_projection(temporal_numeric)
        if self.temporal_category_embedding is not None:
            category_value = temporal_feat[
                :,
                :,
                self.temporal_categorical_index,
            ]
            category_index = torch.round(category_value).long()
            valid_category = (
                (category_index >= self.temporal_category_min)
                & (category_index <= self.temporal_category_max)
            )
            category_index = category_index - self.temporal_category_min
            category_index = torch.where(
                valid_category,
                category_index,
                torch.full_like(
                    category_index,
                    self.temporal_category_unknown,
                ),
            )
            projected_temporal = (
                projected_temporal
                + self.temporal_category_embedding(category_index)
            )
        if self.step_embedding is not None:
            step_index = torch.arange(
                temporal_feat.shape[1],
                device=temporal_feat.device,
            )
            projected_temporal = (
                projected_temporal
                + self.step_embedding(step_index).unsqueeze(0)
            )
        projected_temporal = self.temporal_input_norm(
            projected_temporal
        )
        multi_scale_repr = (
            self.temporal_encoder(temporal_numeric)
            if self.use_multiscale else None
        )

        sequence_input = projected_temporal
        fusion_diagnostics = None

        if self.use_cross_attention:
            multi_scale_context = (
                self.multiscale_context_proj(multi_scale_repr)
                if multi_scale_repr is not None else None
            )
            if self.fusion_type == 'global_local':
                sequence_input, fusion_diagnostics = self.feature_fusion(
                    static_emb,
                    projected_temporal,
                    multi_scale_context
                )
            else:
                attention_context = projected_temporal
                if multi_scale_context is not None:
                    attention_context = torch.cat([
                        projected_temporal,
                        multi_scale_context.unsqueeze(1)
                    ], dim=1)
                cross_fused, attention_weights = self.cross_attention(
                    static_emb, attention_context
                )
                sequence_input = self.cross_context_norm(
                    projected_temporal + cross_fused.unsqueeze(1)
                )
                fusion_diagnostics = {
                    'legacy_attention': attention_weights
                }

        sequence_out = self._encode_sequence(sequence_input)
        temporal_repr = self._pool_sequence(sequence_out)

        fusion_parts = [temporal_repr, static_emb]
        if multi_scale_repr is not None:
            fusion_parts.append(multi_scale_repr)

        logits = self.classifier(torch.cat(fusion_parts, dim=-1))

        if return_attention:
            return logits, fusion_diagnostics
        return logits


