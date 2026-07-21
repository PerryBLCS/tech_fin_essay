import numpy as np
import pandas as pd
import os
import sys
import copy
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, average_precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
TAIWAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")


# ==============================
# 1. 数据加载与预处理
# ==============================

class CreditDataLoader(Dataset):
    """
    处理 German Credit 和 Taiwan Credit 数据集
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.temporal_scaler = StandardScaler()
        self.static_feature_names = None
        self.temporal_feature_names = None
        self.temporal_step_names = None
        self.temporal_flat_feature_names = None

    def load_german_credit(self, filepath=None):
        """
        German Credit Dataset (1000样本, 20特征, 30%违约率)
        将6个类别属性编码为伪时序数据
        """
        # 如果没有提供文件路径, 使用 UCI 在线数据
        source = GERMAN_CREDIT_URL if filepath is None else filepath
        print(f"Loading German Credit data from: {source}")
        df = pd.read_csv(source, sep=r"\s+", header=None)

        if df.shape[1] == 25:
            data = df.astype(float).values
            X = data[:, :-1]
            y = (data[:, -1] == 2).astype(int)
            static_features = X[:, :6].astype(np.float32)
            temporal_features = X[:, 6:].reshape(X.shape[0], 6, 3).astype(np.float32)
            self.static_feature_names = [f'german_static_{i}' for i in range(static_features.shape[1])]
            self.temporal_feature_names = ['category_code', 'risk_prior', 'category_frequency']
            self.temporal_step_names = [f'german_category_step_{i}' for i in range(6)]
            self.temporal_flat_feature_names = [
                f'{step}_{feature}'
                for step in self.temporal_step_names
                for feature in self.temporal_feature_names
            ]
            return static_features, temporal_features, y

        if df.shape[1] != 21:
            raise ValueError(
                f"Unexpected German Credit data shape: {df.shape}. "
                "Expected raw german.data with 20 features + 1 label."
            )

        y = (df.iloc[:, -1].astype(int).values == 2).astype(int)

        # 数值列索引
        numeric_cols = [1, 4, 7, 10, 12, 15, 17]
        # 类别列索引
        categorical_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        pseudo_temporal_cols = [0, 2, 3, 5, 6, 8]
        static_categorical_cols = [col for col in categorical_cols if col not in pseudo_temporal_cols]

        # 1. 数值特征
        numeric_df = df[numeric_cols].astype(np.float32)

        # 2. 类别特征独热编码
        categorical_df = pd.get_dummies(
            df[static_categorical_cols].astype(str),
            prefix=[f"A{col}" for col in static_categorical_cols],
            dtype=np.float32
        )

        # 3. 合并为静态特征
        static_features = pd.concat([numeric_df, categorical_df], axis=1).values.astype(np.float32)
        self.static_feature_names = [str(c) for c in list(numeric_df.columns) + list(categorical_df.columns)]

        # 4. 构建伪时序特征：按顺序使用6个类别属性
        temporal_features = self._build_german_pseudo_temporal(df).astype(np.float32)

        return static_features, temporal_features, y

    def _build_german_pseudo_temporal(self, df):
        """基于六个分类属性构建伪时间特征"""
        selected_cols = [0, 2, 3, 5, 6, 8]
        self.temporal_step_names = [
            'checking_status',
            'credit_history',
            'purpose',
            'savings_account',
            'employment_since',
            'personal_status_sex',
        ]
        self.temporal_feature_names = ['category_code', 'risk_prior', 'category_frequency']
        self.temporal_flat_feature_names = [
            f'{step}_{feature}'
            for step in self.temporal_step_names
            for feature in self.temporal_feature_names
        ]

        risk_priors = {
            0: {'A11': 1.00, 'A12': 0.65, 'A13': 0.35, 'A14': 0.15},
            2: {'A30': 1.00, 'A31': 0.75, 'A32': 0.50, 'A33': 0.30, 'A34': 0.15},
            3: {'A40': 0.55, 'A41': 0.25, 'A42': 0.45, 'A43': 0.35, 'A44': 0.65,
                'A45': 0.60, 'A46': 0.70, 'A48': 0.25, 'A49': 0.50, 'A410': 0.40},
            5: {'A61': 0.90, 'A62': 0.60, 'A63': 0.40, 'A64': 0.20, 'A65': 0.50},
            6: {'A71': 0.85, 'A72': 0.65, 'A73': 0.45, 'A74': 0.30, 'A75': 0.20},
            8: {'A91': 0.50, 'A92': 0.45, 'A93': 0.35, 'A94': 0.30, 'A95': 0.40},
        }

        steps = []
        n_rows = len(df)
        for col in selected_cols:
            values = df[col].astype(str)
            categories = sorted(values.unique())
            denom = max(len(categories) - 1, 1)
            code_map = {cat: idx / denom for idx, cat in enumerate(categories)}
            frequency_map = (values.value_counts() / max(n_rows, 1)).to_dict()
            priors = risk_priors.get(col, {})

            category_code = values.map(code_map).astype(float).values
            risk_prior = values.map(lambda v: priors.get(v, code_map.get(v, 0.5))).astype(float).values
            category_frequency = values.map(frequency_map).astype(float).values
            steps.append(np.stack([category_code, risk_prior, category_frequency], axis=1))

        return np.stack(steps, axis=1)

    def load_taiwan_credit(self, filepath=None):
        """
        Taiwan Credit Card Dataset (30000样本, 24特征, 约22%违约率)
        """

        if filepath is None:

            if len(sys.argv) > 1:
                filepath = sys.argv[1]
            else:
                filepath = TAIWAN_CREDIT_URL

        filepath = str(filepath).strip().strip('"').strip("'")
        is_url = filepath.lower().startswith(('http://', 'https://'))

        if not is_url and not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Taiwan Credit data file not found: {filepath}\n"
                "Pass a valid .xls/.xlsx/.csv path, or leave filepath=None to use the online UCI URL."
            )

        try:
            path_for_ext = filepath.split('?', 1)[0]
            ext = os.path.splitext(path_for_ext)[1].lower()

            if ext in ['.xls', '.xlsx']:
                # UCI 原始 Excel 第一行通常是说明，真正表头在第二行，所以 header=1
                df = pd.read_excel(filepath, header=1)
            else:
                df = pd.read_csv(filepath)

        except ImportError as e:
            raise ImportError(
                "Reading Taiwan .xls files requires xlrd >= 2.0.1. "
                "Install xlrd, or use a .csv version of the Taiwan dataset."
            ) from e
        except Exception as e:
            raise RuntimeError(f"读取数据集失败: {filepath}\n原始错误: {e}") from e

        # 清理列名
        df.columns = [str(c).strip() for c in df.columns]

        # 兼容不同版本的目标列名
        target_candidates = [
            'default.payment.next.month',
            'default payment next month',
            'Y'
        ]
        target_col = None
        for c in target_candidates:
            if c in df.columns:
                target_col = c
                break

        if target_col is None:
            raise KeyError(
                "没有找到目标列。请确认 Excel 表头是否正确。\n"
                f"当前读取到的列名为: {list(df.columns)}\n"
                "目标列通常应为 default payment next month 或 default.payment.next.month。"
            )

        # 兼容 Excel 中列名带空格的情况
        rename_map = {
            'default payment next month': 'default.payment.next.month'
        }
        df = df.rename(columns=rename_map)
        target_col = 'default.payment.next.month' if 'default.payment.next.month' in df.columns else target_col

        # 目标变量
        y = df[target_col].astype(int).values

        # 静态特征: ID 只是样本编号，不能作为预测特征；类别变量用 one-hot 编码
        static_num_cols = ['LIMIT_BAL', 'AGE']
        static_cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

        # 时序特征: 6个月的 PAY, BILL_AMT, PAY_AMT
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        needed_cols = static_num_cols + static_cat_cols + pay_cols + bill_cols + pay_amt_cols
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(
                "数据集中缺少必要列：\n"
                f"{missing_cols}\n"
                f"当前读取到的列名为: {list(df.columns)}"
            )

        base_static_df = pd.get_dummies(
            df[static_num_cols + static_cat_cols],
            columns=static_cat_cols,
            drop_first=False,
            dtype=np.float32
        )

        # 使用目标月份之前可用的六个月历史数据
        limit_balance = np.maximum(df['LIMIT_BAL'].astype(float).values, 1.0)
        bill_values = df[bill_cols].astype(float).values
        payment_values = df[pay_amt_cols].astype(float).values
        status_values = df[pay_cols].astype(float).values
        utilization = np.clip(bill_values / limit_balance[:, None], -10.0, 10.0)
        safe_bill = np.maximum(np.abs(bill_values), 1.0)
        payment_to_bill = np.clip(payment_values / safe_bill, -10.0, 10.0)

        chronological_bill = utilization[:, ::-1]
        trend_axis = np.arange(chronological_bill.shape[1], dtype=np.float32)
        trend_axis -= trend_axis.mean()
        trend_denominator = float(np.square(trend_axis).sum())
        bill_trend_6m = (chronological_bill @ trend_axis) / max(trend_denominator, 1e-7)

        engineered_static = pd.DataFrame({
            'recent_bill_to_limit': utilization[:, 0],
            'avg_bill_to_limit_6m': utilization.mean(axis=1),
            'max_bill_to_limit_6m': utilization.max(axis=1),
            'bill_trend_6m': bill_trend_6m,
            'recent_payment_shock': status_values[:, 0] - status_values[:, 1],
            'delinquency_count_6m': (status_values > 0).sum(axis=1),
            'max_delinquency_6m': status_values.max(axis=1),
            'recent_payment_to_bill': payment_to_bill[:, 0],
            'avg_payment_to_bill_6m': payment_to_bill.mean(axis=1),
        }).clip(-10.0, 10.0)

        static_df = pd.concat(
            [base_static_df.reset_index(drop=True), engineered_static.reset_index(drop=True)],
            axis=1
        )
        static_features = static_df.astype(np.float32).values
        self.static_feature_names = [str(c) for c in static_df.columns]

        temporal_features = np.stack([
            status_values,
            bill_values,
            payment_values,
            utilization,
            payment_to_bill,
        ], axis=-1).astype(np.float32)  # [N, 6, 5]
        self.temporal_feature_names = [
            'payment_status',
            'bill_amount',
            'payment_amount',
            'bill_to_limit',
            'payment_to_bill',
        ]
        self.temporal_step_names = ['recent_month', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6']
        self.temporal_flat_feature_names = [
            name
            for step in range(6)
            for name in (
                pay_cols[step],
                bill_cols[step],
                pay_amt_cols[step],
                f'bill_to_limit_{step + 1}',
                f'payment_to_bill_{step + 1}',
            )
        ]

        print(f"Loaded real Taiwan Credit data from: {filepath}")
        return static_features, temporal_features, y

    @staticmethod
    def extend_temporal_features(temporal_features, target_steps, method='linear_trend'):
        """
        扩展时间序列数据到指定的时间步数

        参数:
            temporal_features: 原始时间序列数据 (n_samples, current_steps, n_features)
            target_steps: 目标时间步数 (如 12, 24, 36)
            method: 扩展方法
                - 'linear_trend': 基于线性趋势外推（默认）
                - 'periodic': 周期性重复
                - 'last_value': 保持最后值不变

        返回:
            扩展后的时间序列数据 (n_samples, target_steps, n_features)
        """
        current_steps = temporal_features.shape[1]

        if target_steps <= current_steps:
            return temporal_features

        n_samples = temporal_features.shape[0]
        n_features = temporal_features.shape[2]
        extended = np.zeros((n_samples, target_steps, n_features), dtype=temporal_features.dtype)
        extended[:, :current_steps, :] = temporal_features

        if method == 'linear_trend':
            for i in range(current_steps, target_steps):
                decay_factor = 0.85 ** (i - current_steps)
                if current_steps >= 2:
                    trend = temporal_features[:, -1, :] - temporal_features[:, -2, :]
                    extended[:, i, :] = extended[:, i - 1, :] + trend * decay_factor
                else:
                    extended[:, i, :] = temporal_features[:, -1, :]

        elif method == 'periodic':
            repeats = int(np.ceil(target_steps / current_steps))
            full_extended = np.tile(temporal_features, (1, repeats, 1))
            extended = full_extended[:, :target_steps, :]

        elif method == 'last_value':
            for i in range(current_steps, target_steps):
                extended[:, i, :] = temporal_features[:, -1, :]

        else:
            raise ValueError(
                f"Unknown extension method: {method}. Choose from 'linear_trend', 'periodic', 'last_value'")

        return extended

    def preprocess(self, static_features, temporal_features, y, test_size=0.15, val_size=0.15):
        """
        数据预处理: 标准化、分割
        """
        # 先分割，再只用训练集拟合 scaler，避免验证/测试集信息泄漏
        X_static_train, X_static_temp, X_temporal_train, X_temporal_temp, y_train, y_temp = train_test_split(
            static_features, temporal_features, y, test_size=test_size + val_size, random_state=42, stratify=y)
        val_ratio = val_size / (test_size + val_size)
        X_static_val, X_static_test, X_temporal_val, X_temporal_test, y_val, y_test = train_test_split(X_static_temp,
                                                                                                       X_temporal_temp,
                                                                                                       y_temp,
                                                                                                       test_size=1 - val_ratio,
                                                                                                       random_state=42,
                                                                                                       stratify=y_temp)

        binary_static_cols = np.all((X_static_train == 0) | (X_static_train == 1), axis=0)
        continuous_static_cols = ~binary_static_cols
        if np.any(continuous_static_cols):
            X_static_train = X_static_train.copy()
            X_static_val = X_static_val.copy()
            X_static_test = X_static_test.copy()
            X_static_train[:, continuous_static_cols] = self.scaler.fit_transform(
                X_static_train[:, continuous_static_cols]
            )
            X_static_val[:, continuous_static_cols] = self.scaler.transform(
                X_static_val[:, continuous_static_cols]
            )
            X_static_test[:, continuous_static_cols] = self.scaler.transform(
                X_static_test[:, continuous_static_cols]
            )

        n_train, time_steps, n_features = X_temporal_train.shape
        self.temporal_scaler = StandardScaler()

        def scale_temporal(x, fit=False):
            reshaped = x.reshape(-1, n_features)
            scaled = self.temporal_scaler.fit_transform(reshaped) if fit else self.temporal_scaler.transform(reshaped)
            return scaled.reshape(x.shape[0], time_steps, n_features)

        X_temporal_train = scale_temporal(X_temporal_train, fit=True)
        X_temporal_val = scale_temporal(X_temporal_val)
        X_temporal_test = scale_temporal(X_temporal_test)

        return (
            X_static_train, X_temporal_train, y_train,
            X_static_val, X_temporal_val, y_val,
            X_static_test, X_temporal_test, y_test,
        )


class CreditDataset(Dataset):
    """用于信用风险数据的 PyTorch 数据集"""

    def __init__(self, static_features, temporal_features, labels):
        self.static_features = torch.FloatTensor(static_features)
        self.temporal_features = torch.FloatTensor(temporal_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'static': self.static_features[idx],
            'temporal': self.temporal_features[idx],
            'label': self.labels[idx]
        }


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

        # 标准 LSTM 门控参数
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # 自适应门控系数 λ：由当前输入和历史状态动态生成，而不是全局常数
        self.lambda_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

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

        # 原始门控计算
        f_tilde = torch.sigmoid(self.W_f(combined))  # 候选遗忘门
        i_tilde = torch.sigmoid(self.W_i(combined))  # 候选输入门
        o_tilde = torch.sigmoid(self.W_o(combined))  # 候选输出门

        # 自适应门控: λ 随样本和时间状态变化，λ 越大越偏记忆，越小越偏更新
        lambda_val = self.lambda_gate(combined)

        # 候选细胞状态
        c_tilde = torch.tanh(self.W_c(combined))

        f_t = lambda_val * f_tilde + (1 - lambda_val) * i_tilde
        i_t = (1 - lambda_val) * f_tilde + lambda_val * i_tilde
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
            h = torch.zeros((batch_size, self.hidden_dim), device=x.device)
            c = torch.zeros((batch_size, self.hidden_dim), device=x.device)
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

    def __init__(self, input_dim, hidden_dim, coarse_window=12):
        super().__init__()
        self.coarse_window = coarse_window
        self.output_dim = hidden_dim * 2

        self.fine_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.medium_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.coarse_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 短序列回退方案，避免单时间步输入BiLSTM
        self.coarse_fallback = nn.Linear(input_dim * 2, hidden_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2)
        )

    @staticmethod
    def _moving_average(x, window, padding=0):
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

    def forward(self, x):
        """
        Args:
            x: [batch, time_steps, input_dim]
        """
        _, time_steps, _ = x.shape

        _, (fine_hidden, _) = self.fine_encoder(x)
        fine_repr = self._bidirectional_state(fine_hidden)

        if time_steps >= 3:
            medium_x = self._moving_average(x, window=3, padding=1)
            _, (medium_hidden, _) = self.medium_encoder(medium_x)
            medium_repr = self._bidirectional_state(medium_hidden)
        else:
            medium_repr = fine_repr

        if time_steps >= self.coarse_window:
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
    自适应注意力双向LSTM融合模型 (AA-BiLSTM)

    完整架构:
    1. 特征编码层（静态 Embedding + 多尺度时序编码）
    2. TS-CrossAttention 层（异构数据深度融合）
    3. AG-ResUnit 堆叠层（可扩展至 8 层双向 LSTM）
    4. 预测输出层（Dynamic Focal Loss 在 Trainer 中启用）
    """

    def __init__(self, static_dim, temporal_dim, temporal_steps, hidden_dim=128, num_layers=8, num_classes=2,
                 dropout=0.3, num_heads=4, use_cross_attention=True, use_ag_resunit=True,
                 bidirectional=True, use_multiscale=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temporal_steps = temporal_steps
        self.use_cross_attention = use_cross_attention
        self.use_ag_resunit = use_ag_resunit
        self.bidirectional = bidirectional
        self.use_multiscale = use_multiscale

        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim)

        if use_multiscale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                temporal_dim,
                hidden_dim // 2,
                coarse_window=12
            )
            multi_scale_dim = self.temporal_encoder.output_dim
        else:
            self.temporal_encoder = None
            multi_scale_dim = 0
        self.multi_scale_dim = multi_scale_dim

        if use_cross_attention:
            self.cross_attention = TSCrossAttention(
                static_dim=hidden_dim,
                temporal_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.cross_attention = None

        self.multiscale_context_proj = (
            nn.Linear(multi_scale_dim, hidden_dim)
            if use_multiscale and use_cross_attention else None
        )
        self.cross_context_norm = nn.LayerNorm(hidden_dim) if use_cross_attention else None

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

        classifier_input_dim = sequence_dim
        if not use_cross_attention:
            classifier_input_dim += hidden_dim
        if use_multiscale and not use_cross_attention:
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
        projected_temporal = self.temporal_projection(temporal_feat)
        multi_scale_repr = self.temporal_encoder(temporal_feat) if self.use_multiscale else None

        sequence_input = projected_temporal
        attention_weights = None

        if self.use_cross_attention:
            attention_context = projected_temporal
            if multi_scale_repr is not None:
                multi_scale_token = self.multiscale_context_proj(multi_scale_repr).unsqueeze(1)
                attention_context = torch.cat([projected_temporal, multi_scale_token], dim=1)
            cross_fused, attention_weights = self.cross_attention(static_emb, attention_context)
            # 先进行多尺度/时序交叉注意力，再进行 AG-BiLSTM 编码
            sequence_input = self.cross_context_norm(projected_temporal + cross_fused.unsqueeze(1))

        sequence_out = self._encode_sequence(sequence_input)
        temporal_repr = self._pool_sequence(sequence_out)

        fusion_parts = [temporal_repr]
        if not self.use_cross_attention:
            fusion_parts.append(static_emb)
            if multi_scale_repr is not None:
                fusion_parts.append(multi_scale_repr)

        logits = self.classifier(torch.cat(fusion_parts, dim=-1))

        if return_attention:
            return logits, attention_weights
        return logits


# ==============================
# 3. 动态焦点损失函数
# ==============================

class DynamicFocalLoss(nn.Module):
    """
    动态焦点损失函数 (Dynamic Focal Loss)

    描述:
    - 根据训练epoch和类别预测难度动态调整γ
    - 早期训练: γ较小，关注易分样本
    - 后期训练: γ增大，聚焦难分样本
    - 对违约类(y=1)增强关注
    """

    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma_base=1.0, gamma_max=3.0, num_epoch=100):
        super().__init__()
        self.alpha_pos = alpha_pos  # 正类 (违约) 权重
        self.alpha_neg = alpha_neg  # 负类 (正常) 权重
        self.gamma_base = gamma_base
        self.gamma_max = gamma_max
        self.num_epoch = num_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前 epoch 以调整 gamma"""
        self.current_epoch = epoch

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] 模型输出(logits)
            targets: [N] 真实标签
        """
        # 计算概率
        probs = torch.softmax(inputs, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # 预测正确类的概率

        # 动态调整 gamma
        # gamma = gamma_base + (gamma_max - gamma_base) * (epoch / max_epochs) * (1 - p_t)
        progress = self.current_epoch / self.num_epoch
        gamma = self.gamma_base + (self.gamma_max - self.gamma_base) * progress * (1 - p_t)

        N_pos_num = (targets == 1).sum().float()
        N_neg_num = (targets == 0).sum().float()

        neg_adjust = 1.0 + N_pos_num / (N_neg_num + 1e-7)
        alpha_t = torch.where(
            targets == 1,
            torch.as_tensor(self.alpha_pos, device=inputs.device, dtype=inputs.dtype),
            torch.as_tensor(self.alpha_neg, device=inputs.device, dtype=inputs.dtype) * neg_adjust
        )
        # Focal Loss计算
        focal_weight = (1 - p_t) ** gamma
        log_p_t = torch.log(p_t + 1e-7)
        loss = -alpha_t * focal_weight * log_p_t

        return loss.mean()


# ==============================
# 4. 训练与评估
# ==============================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.counter = 0


class Trainer:
    """模型训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_epochs=100, lr=1e-3, weight_decay=1e-4,
                 use_dynamic_focal=True, use_class_weight=False,
                 use_early_stopping=False, threshold_min_sensitivity=0.40):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.decision_threshold = 0.5
        self.threshold_min_sensitivity = threshold_min_sensitivity
        self.use_early_stopping = use_early_stopping

        # 优化器: AdamW + 余弦退火
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # 损失函数
        if use_dynamic_focal:
            self.criterion = DynamicFocalLoss(
                num_epoch=num_epochs,
                alpha_pos=0.75,  # 增强对违约类的关注
                alpha_neg=0.25,
                gamma_base=1.0,
                gamma_max=3.0
            )
        else:
            # 标准交叉熵更适合优化整体 Accuracy；如需强调少数类可打开 use_class_weight
            if use_class_weight:
                weights = self._compute_class_weights(train_loader)
                self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.early_stopping = EarlyStopping(patience=10)
        self.history = {'train_loss': [], 'val_auc': [], 'val_auc_pr': [], 'val_f1': [], 'val_accuracy': []}

    @staticmethod
    def _compute_class_weights(loader):
        """计算类别权重"""
        all_labels = []
        for batch in loader:
            all_labels.extend(batch['label'].numpy())
        counts = np.bincount(all_labels)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 2  # 归一化
        return torch.FloatTensor(weights)

    def train_epoch(self, epoch):
        """训练一个 epoch（混合精度 + 梯度裁剪）"""
        self.model.train()
        if isinstance(self.criterion, DynamicFocalLoss):
            self.criterion.set_epoch(epoch)

        total_loss = 0
        scaler = torch.cuda.amp.GradScaler()  # 混合精度缩放器

        for batch in self.train_loader:
            static = batch['static'].to(device, non_blocking=True)
            temporal = batch['temporal'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            self.optimizer.zero_grad()

            # 混合精度前向传播
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                outputs = self.model(static, temporal)
                loss = self.criterion(outputs, labels)

            # 混合精度反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪（防止梯度爆炸）
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.scheduler.step()  # 学习率调度
        return avg_loss

    def evaluate(self, loader, threshold=None):
        """评估模型"""
        self.model.eval()
        if threshold is None:
            threshold = self.decision_threshold
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                static = batch['static'].to(device)
                temporal = batch['temporal'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(static, temporal)
                probs = torch.softmax(outputs, dim=-1)
                preds = (probs[:, 1] >= threshold).long()

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())  # 违约类概率
                all_labels.append(labels.cpu().numpy())

        # 计算指标
        # 展平嵌套列表
        flat_labels = np.concatenate(all_labels)
        flat_preds = np.concatenate(all_preds)
        flat_probs = np.concatenate(all_probs)

        accuracy = accuracy_score(flat_labels, flat_preds)
        try:
            auc = roc_auc_score(flat_labels, flat_probs)
        except ValueError:
            auc = np.nan
        try:
            auc_pr = average_precision_score(flat_labels, flat_probs)
        except ValueError:
            auc_pr = np.nan
        f1 = f1_score(flat_labels, flat_preds, zero_division=0)
        sensitivity = recall_score(flat_labels, flat_preds)  # 敏感性/召回率
        specificity = specificity_score(flat_labels, flat_preds)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'auc_pr': auc_pr,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'threshold': threshold,
            'predictions': flat_preds,  # 返回展平后的数组
            'probabilities': flat_probs,
            'labels': flat_labels,
        }

    def find_best_threshold(self, min_sensitivity=0.40):
        """仅在验证集上选择概率阈值"""
        validation = self.evaluate(self.val_loader, threshold=0.5)
        labels = validation['labels']
        probabilities = validation['probabilities']
        scored = []

        for threshold in np.linspace(0.05, 0.95, 181):
            predictions = (probabilities >= threshold).astype(int)
            sensitivity = recall_score(labels, predictions)
            if sensitivity + 1e-12 < min_sensitivity:
                continue
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, zero_division=0)
            scored.append((accuracy, f1, sensitivity, -abs(threshold - 0.5), threshold))

        if not scored:
            return 0.5, {
                'accuracy': validation['accuracy'],
                'f1': validation['f1'],
                'sensitivity': validation['sensitivity'],
            }

        accuracy, f1, sensitivity, _, threshold = max(scored)
        return float(threshold), {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'sensitivity': float(sensitivity),
        }

    def train(self):
        """完整训练流程"""
        best_auc = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_auc_pr'].append(val_metrics['auc_pr'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # 学习率调整
            self.scheduler.step()

            # 早停检查
            if self.use_early_stopping:
                self.early_stopping(val_metrics['auc'])

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_model_state = copy.deepcopy(self.model.state_dict())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"AUC-PR: {val_metrics['auc_pr']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f} | "
                      f"Sens: {val_metrics['sensitivity']:.4f}")

            if self.use_early_stopping and self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.decision_threshold, validation_threshold_metrics = self.find_best_threshold(
            min_sensitivity=self.threshold_min_sensitivity
        )
        print(
            f"Selected validation-only threshold: {self.decision_threshold:.3f} | "
            f"Val Accuracy: {validation_threshold_metrics['accuracy']:.4f} | "
            f"Val Sensitivity: {validation_threshold_metrics['sensitivity']:.4f}"
        )

        # The untouched test set is evaluated once with the validation cutoff.
        test_metrics = self.evaluate(self.test_loader, threshold=self.decision_threshold)
        test_metrics['validation_threshold_accuracy'] = validation_threshold_metrics['accuracy']
        test_metrics['validation_threshold_sensitivity'] = validation_threshold_metrics['sensitivity']
        return test_metrics, self.history


def specificity_score(y_true, y_pred):
    """计算特异性"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


class SequenceBaselineModel(nn.Module):
    """LSTM/Bi-LSTM/Attention-LSTM/ResNet-LSTM 基线模型"""

    def __init__(self, static_dim, temporal_dim, hidden_dim=64, model_type='lstm', dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.use_attention = model_type == 'attention_lstm'
        self.use_residual = model_type == 'resnet_lstm'
        bidirectional = model_type in {'bilstm', 'resnet_lstm'}

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(
            temporal_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        seq_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Linear(seq_dim, 1) if self.use_attention else None
        self.residual_proj = nn.Linear(temporal_dim, seq_dim) if self.use_residual else None
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, static_feat, temporal_feat):
        static_repr = self.static_encoder(static_feat)
        seq_out, _ = self.lstm(temporal_feat)
        if self.use_residual:
            seq_out = seq_out + self.residual_proj(temporal_feat)
        if self.use_attention:
            attn = torch.softmax(self.attention(seq_out), dim=1)
            temporal_repr = (attn * seq_out).sum(dim=1)
        else:
            temporal_repr = seq_out[:, -1, :]
        return self.classifier(torch.cat([static_repr, temporal_repr], dim=-1))


def train_sequence_baseline(name, model, X_static_train, X_temporal_train, y_train,
                            X_static_test, X_temporal_test, y_test,
                            epochs=20, batch_size=128, lr=1e-3):
    """训练一个轻量级神经网络基线模型，并返回评估指标"""
    train_dataset = CreditDataset(X_static_train, X_temporal_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=torch.cuda.is_available())
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            static = batch['static'].to(device)
            temporal = batch['temporal'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(static, temporal), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        static = torch.FloatTensor(X_static_test).to(device)
        temporal = torch.FloatTensor(X_temporal_test).to(device)
        labels = np.asarray(y_test)
        probs = torch.softmax(model(static, temporal), dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    try:
        auc_pr = average_precision_score(labels, probs)
    except ValueError:
        auc_pr = np.nan

    return {
        'acc': accuracy_score(labels, preds),
        'auc': auc,
        'auc_pr': auc_pr,
        'f1': f1_score(labels, preds, zero_division=0),
        'sensitivity': recall_score(labels, preds),
        'specificity': specificity_score(labels, preds),
    }


# ==============================
# 5. 可解释性分析 (SHAP)
# ==============================

class SHAPModelWrapper(nn.Module):
    """包装模型以适配SHAP的输入格式"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        """SHAP会传入列表形式的输入"""
        if isinstance(inputs, (list, tuple)):
            static_feat, temporal_feat = inputs[0], inputs[1]
        else:
            raise ValueError("Expected list/tuple input [static, temporal]")
        return self.model(static_feat, temporal_feat, return_attention=False)


class Explainer:
    """SHAP 可解释性分析"""

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    def explain(self, static_data, temporal_data, sample_size=100):
        """使用 SHAP 解释模型预测"""
        if shap is None:
            raise ImportError("shap is not installed")

        self.model.eval()

        # 创建背景数据
        background_static = static_data[:sample_size]
        background_temporal = temporal_data[:sample_size]

        # 使用包装后的模型
        wrapped_model = SHAPModelWrapper(self.model).to(device)
        wrapped_model.eval()

        # 转换为tensor
        background_static = torch.tensor(background_static).float().to(device)
        background_temporal = torch.tensor(background_temporal).float().to(device)

        test_sample_static = torch.tensor(static_data[sample_size:sample_size + 50]).float().to(device)
        test_sample_temporal = torch.tensor(temporal_data[sample_size:sample_size + 50]).float().to(device)

        try:
            # 尝试使用 DeepExplainer
            print("Attempting SHAP DeepExplainer...")
            explainer = shap.DeepExplainer(wrapped_model, [background_static, background_temporal])
            shap_values = explainer.shap_values([test_sample_static, test_sample_temporal])
            print("SHAP DeepExplainer succeeded.")
            return shap_values, explainer
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to KernelExplainer (slower but more robust)...")
            # 备用方案：使用 KernelExplainer
            return self._kernel_explain(static_data, temporal_data, sample_size)

    def _kernel_explain(self, static_data, temporal_data, sample_size):
        """使用 KernelExplainer 作为备用方案"""
        wrapped_model = SHAPModelWrapper(self.model).to(device)
        wrapped_model.eval()

        # 展平特征用于 KernelExplainer
        static_flat = static_data[:sample_size]
        temporal_flat = temporal_data[:sample_size].reshape(sample_size, -1)
        background_flat = np.concatenate([static_flat, temporal_flat], axis=1)

        def model_predict(flat_input):
            n = flat_input.shape[0]
            static_dim = static_data.shape[1]
            temporal_dim_2d = temporal_data.shape[1]
            temporal_dim_3d = temporal_data.shape[2]

            static_part = torch.FloatTensor(flat_input[:, :static_dim]).to(device)
            temporal_part = torch.FloatTensor(
                flat_input[:, static_dim:].reshape(n, temporal_dim_2d, temporal_dim_3d)
            ).to(device)

            with torch.no_grad():
                output = wrapped_model([static_part, temporal_part])
                probs = torch.softmax(output, dim=-1)
            return probs[:, 1].cpu().numpy()

        # 测试数据展平
        test_static_flat = static_data[sample_size:sample_size + 50]
        test_temporal_flat = temporal_data[sample_size:sample_size + 50].reshape(50, -1)
        test_flat = np.concatenate([test_static_flat, test_temporal_flat], axis=1)

        explainer = shap.KernelExplainer(model_predict, background_flat[:100])
        shap_values = explainer.shap_values(test_flat, nsamples=100)

        # 重构 SHAP 值格式以匹配双输入
        static_shap = shap_values[:, :static_data.shape[1]]
        temporal_shap = shap_values[:, static_data.shape[1]:].reshape(
            50, temporal_data.shape[1], temporal_data.shape[2]
        )

        print("KernelExplainer succeeded.")
        return [static_shap, temporal_shap], explainer


def build_feature_names(data_loader, static_dim, temporal_steps, temporal_dim):
    static_names = data_loader.static_feature_names or [f'static_{i}' for i in range(static_dim)]
    if data_loader.temporal_flat_feature_names:
        temporal_names = list(data_loader.temporal_flat_feature_names)
    else:
        temporal_feature_names = data_loader.temporal_feature_names or [f'temporal_feature_{i}' for i in range(temporal_dim)]
        temporal_step_names = data_loader.temporal_step_names or [f't{t}' for t in range(temporal_steps)]
        temporal_names = [
            f'{step}_{feature}'
            for step in temporal_step_names
            for feature in temporal_feature_names
        ]
    return list(static_names), temporal_names


def run_shap_summary(model, data_loader, X_static_test, X_temporal_test, dataset_name, sample_size=None):
    """运行 SHAP 并在支持时输出简洁的全局解释摘要"""
    # 根据数据集大小动态调整样本数
    if sample_size is None:
        n_test = len(X_static_test)
        sample_size = min(200, max(50, n_test // 10))

    if len(X_static_test) < sample_size + 5:
        sample_size = max(5, len(X_static_test) // 2)

    static_names, temporal_names = build_feature_names(
        data_loader,
        X_static_test.shape[1],
        X_temporal_test.shape[1],
        X_temporal_test.shape[2]
    )

    try:
        shap_values, _ = Explainer(model).explain(
            X_static_test,
            X_temporal_test,
            sample_size=sample_size
        )
    except Exception as e:
        print(f"SHAP analysis skipped: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 处理 SHAP 返回值
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        static_values = np.asarray(shap_values[0])
        temporal_values = np.asarray(shap_values[1])
    else:
        print("SHAP values computed, but returned layout is not a two-input explanation.")
        return shap_values

    # 如果是二分类，取正类的SHAP值
    if static_values.ndim >= 3 and static_values.shape[-1] == 2:
        static_values = static_values[..., 1]
    if temporal_values.ndim >= 4 and temporal_values.shape[-1] == 2:
        temporal_values = temporal_values[..., 1]

    temporal_values = temporal_values.reshape(temporal_values.shape[0], -1)
    static_importance = np.abs(static_values).mean(axis=0).reshape(-1)
    temporal_importance = np.abs(temporal_values).mean(axis=0).reshape(-1)
    all_names = static_names + temporal_names
    all_scores = np.concatenate([static_importance, temporal_importance])
    top_idx = np.argsort(all_scores)[::-1][:10]

    print(f"\nTop SHAP warning indicators for {dataset_name.upper()}:")
    for rank, idx in enumerate(top_idx, 1):
        name = all_names[idx] if idx < len(all_names) else f'feature_{idx}'
        print(f"{rank:02d}. {name}: {all_scores[idx]:.6f}")

    return shap_values


def summarize_cross_attention(model, data_loader, X_static_test, X_temporal_test, sample_size=256):
    """输出时序交叉注意力在时间步和多尺度上下文上的平均权重"""
    if not hasattr(model, 'forward'):
        return None

    n_samples = min(sample_size, len(X_static_test))
    if n_samples <= 0:
        return None

    model.eval()
    try:
        with torch.no_grad():
            static = torch.FloatTensor(X_static_test[:n_samples]).to(device)
            temporal = torch.FloatTensor(X_temporal_test[:n_samples]).to(device)
            _, attention_weights = model(static, temporal, return_attention=True)
    except Exception as e:
        print(f"Cross-attention summary skipped: {e}")
        return None

    if attention_weights is None:
        print("Cross-attention summary skipped: model did not return attention weights.")
        return None

    weights = attention_weights.detach().cpu().numpy()
    avg_weights = weights.mean(axis=0).reshape(-1)
    names = data_loader.temporal_step_names or [f't{i}' for i in range(X_temporal_test.shape[1])]
    if len(avg_weights) > len(names):
        names = list(names) + ['multi_scale_context']
    elif len(avg_weights) < len(names):
        names = list(names[:len(avg_weights)])

    top_idx = np.argsort(avg_weights)[::-1][:min(10, len(avg_weights))]
    print("\nTop TS-CrossAttention context weights:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:02d}. {names[idx]}: {avg_weights[idx]:.6f}")
    return avg_weights

# ==============================
# 6. 主程序
# ==============================

def run_experiment(dataset_name='german', epoch=None, batch_size=None, data_path=None,
                   run_analysis=True, make_plots=True,
                   threshold_min_sensitivity=0.40):
    """
    运行完整实验
    Args:
        dataset_name: 'german' 或 'taiwan'
        epoch: 训练轮数
        batch_size: 批次大小
        data_path: Taiwan 数据集路径，可传入 .xls/.xlsx/.csv
        run_imbalance_study: 是否运行训练集违约率欠采样鲁棒性实验
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in {'german', 'taiwan'}:
        raise ValueError("dataset_name must be 'german' or 'taiwan'.")

    print(f"\n{'=' * 60}")
    print(f"Running AA-BiLSTM Experiment on {dataset_name.upper()} Dataset")
    print(f"{'=' * 60}")
    set_seed(7 if dataset_name == 'german' else 42)

    # 1. 数据加载
    data_loader = CreditDataLoader()

    if dataset_name == 'german':
        static_feat, temporal_feat, y = data_loader.load_german_credit(data_path)
        batch_size = 32 if batch_size is None else batch_size
        epoch = 100 if epoch is None else epoch
        hidden_dim = 64
        num_layers = 4
        num_heads = 4
        dropout = 0.3
        lr = 1e-3
        weight_decay = 1e-4
    else:
        static_feat, temporal_feat, y = data_loader.load_taiwan_credit(data_path)
        batch_size = 128 if batch_size is None else batch_size
        epoch = 150 if epoch is None else epoch
        hidden_dim = 128
        num_layers = 8
        num_heads = 8
        dropout = 0.4
        lr = 5e-4
        weight_decay = 2e-4

    print(f"Data shape: Static {static_feat.shape}, Temporal {temporal_feat.shape}")
    print(f"Class distribution: {Counter(y)}")
    print(f"Default rate: {y.mean() * 100:.2f}%")

    # 2. 数据预处理
    (X_static_train, X_temporal_train, y_train,
     X_static_val, X_temporal_val, y_val,
     X_static_test, X_temporal_test, y_test) = data_loader.preprocess(static_feat, temporal_feat, y)

    # 3. 创建 DataLoader
    train_dataset = CreditDataset(X_static_train, X_temporal_train, y_train)
    val_dataset = CreditDataset(X_static_val, X_temporal_val, y_val)
    test_dataset = CreditDataset(X_static_test, X_temporal_test, y_test)

    pin_memory = torch.cuda.is_available()
    num_workers = 4 if torch.cuda.is_available() else 0
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=num_workers // 2,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=num_workers // 2,
        persistent_workers=persistent
    )

    # 4. 模型初始化
    static_dim = X_static_train.shape[1]
    temporal_steps = X_temporal_train.shape[1]
    temporal_dim = X_temporal_train.shape[2]

    model = AABiLSTM(
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        temporal_steps=temporal_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        num_heads=num_heads,
        use_cross_attention=True,
        use_ag_resunit=True,
        bidirectional=True,
        use_multiscale=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练
    # 根据数据集大小动态调整早停耐心值
    patience = 20 if dataset_name == 'german' else 30

    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        num_epochs=epoch,
        lr=lr,
        weight_decay=weight_decay,
        use_dynamic_focal=True,
        use_class_weight=False,
        use_early_stopping=True,
        threshold_min_sensitivity=threshold_min_sensitivity
    )
    trainer.early_stopping.patience = patience  # 动态设置耐心值

    test_metrics, history = trainer.train()

    # 6. 结果输出
    print(f"\n{'=' * 60}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"AUC-ROC:   {test_metrics['auc']:.4f}")
    print(f"AUC-PR:    {test_metrics['auc_pr']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Sensitivity (Recall): {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity'] * 100:.2f}%)")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Decision Threshold: {test_metrics['threshold']:.3f}")
    print(f"{'=' * 60}\n")

    # 7. 对比基准模型
    if run_analysis:
        print("Comparison with baseline models...")
        compare_baselines(X_static_train, X_temporal_train, y_train,
                          X_static_test, X_temporal_test, y_test,
                          deep_epochs=min(epoch, 30),
                          batch_size=batch_size)

    # 8. 消融实验
    if run_analysis:
        print("\nRunning ablation study...")
        run_ablation_study(static_dim, temporal_dim, temporal_steps,
                           train_loader, val_loader, test_loader,
                           epoch=min(epoch, 50),
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           dropout=dropout,
                           lr=lr,
                           weight_decay=weight_decay)

    if run_analysis:
        run_history_length_study(
            static_feat,
            temporal_feat,
            y,
            dataset_name=dataset_name,
            epoch=min(epoch, 30),
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            extend_method='linear_trend'  # 启用数据扩展
        )

    if run_analysis:
        run_imbalance_robustness_study(
            X_static_train,
            X_temporal_train,
            y_train,
            X_static_val,
            X_temporal_val,
            y_val,
            X_static_test,
            X_temporal_test,
            y_test,
            epoch=min(epoch, 30),
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay
        )

    if run_analysis:
        run_shap_summary(model, data_loader, X_static_test, X_temporal_test, dataset_name)
        summarize_cross_attention(model, data_loader, X_static_test, X_temporal_test)

    # 9. 可视化训练历史
    if make_plots:
        plot_training_history(history, dataset_name)

    return model, test_metrics, history


def compare_baselines(X_static_train, X_temporal_train, y_train,
                      X_static_test, X_temporal_test, y_test,
                      deep_epochs=20, batch_size=128):
    """与传统ML和深度学习基准模型对比。"""
    from sklearn.ensemble import RandomForestClassifier

    # 展平时序特征用于传统 ML
    X_train_flat = np.concatenate([
        X_static_train,
        X_temporal_train.reshape(X_temporal_train.shape[0], -1)
    ], axis=1)

    X_test_flat = np.concatenate([
        X_static_test,
        X_temporal_test.reshape(X_static_test.shape[0], -1)
    ], axis=1)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced'),
        'DNN': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                             alpha=1e-4, max_iter=300, random_state=42)
    }

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42
        )
    except Exception as e:
        print(f"XGBoost skipped: {e}")

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42
        )
    except Exception as e:
        print(f"LightGBM skipped: {e}")

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train_flat, y_train)
            preds = model.predict(X_test_flat)
            probs = model.predict_proba(X_test_flat)[:, 1]
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = np.nan
            try:
                auc_pr = average_precision_score(y_test, probs)
            except ValueError:
                auc_pr = np.nan

            results[name] = {
                'acc': accuracy_score(y_test, preds),
                'auc': auc,
                'auc_pr': auc_pr,
                'f1': f1_score(y_test, preds, zero_division=0),
                'sensitivity': recall_score(y_test, preds),
                'specificity': specificity_score(y_test, preds),
            }

            print(f"{name:20s} | Acc: {results[name]['acc']:.3f} | "
                  f"AUC: {results[name]['auc']:.3f} | AUC-PR: {results[name]['auc_pr']:.3f} | "
                  f"F1: {results[name]['f1']:.3f} | "
                  f"Sens: {results[name]['sensitivity']:.3f} | "
                  f"Spec: {results[name]['specificity']:.3f}"
                  )

        except Exception as e:
            print(f"{name} skipped: {e}")

    deep_models = {
        'Standard LSTM': 'lstm',
        'Bi-LSTM': 'bilstm',
        'Attention-LSTM': 'attention_lstm',
        'ResNet-LSTM': 'resnet_lstm'
    }
    for name, model_type in deep_models.items():
        try:
            model = SequenceBaselineModel(
                static_dim=X_static_train.shape[1],
                temporal_dim=X_temporal_train.shape[2],
                hidden_dim=64,
                model_type=model_type
            )
            results[name] = train_sequence_baseline(
                name,
                model,
                X_static_train,
                X_temporal_train,
                y_train,
                X_static_test,
                X_temporal_test,
                y_test,
                epochs=deep_epochs,
                batch_size=batch_size
            )
            print(f"{name:20s} | Acc: {results[name]['acc']:.3f} | "
                  f"AUC: {results[name]['auc']:.3f} | AUC-PR: {results[name]['auc_pr']:.3f} | "
                  f"F1: {results[name]['f1']:.3f} | "
                  f"Sens: {results[name]['sensitivity']:.3f} | "
                  f"Spec: {results[name]['specificity']:.3f}")

        except Exception as e:
            print(f"{name} skipped: {e}")

    return results


def run_ablation_study(static_dim, temporal_dim, temporal_steps,
                       train_loader, val_loader, test_loader, epoch=30,
                       hidden_dim=64, num_layers=4, num_heads=4, dropout=0.3,
                       lr=1e-3, weight_decay=1e-4):
    """
       消融实验: 验证各模块贡献

       配置:
       1. 标准LSTM (无CrossAttention, 无AG-ResUnit, 无Dynamic Focal)
       2. + TS-CrossAttention
       3. + AG-ResUnit
       4. + Bi-Directional
       5. + Dynamic Focal Loss (完整模型)
    """

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    configs = [
        ('Standard LSTM', False, False, False, False, False),
        ('+ TS-CrossAttention', True, False, False, True, False),
        ('+ AG-ResUnit', True, True, False, True, False),
        ('+ Bi-Directional', True, True, True, True, False),
        ('+ Dynamic Focal (Full)', True, True, True, True, True)
    ]

    results = []
    for name, use_cross, use_ag, use_bi, use_multiscale, use_focal in configs:
        print(f"\nTesting: {name}")

        model = AABiLSTM(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            temporal_steps=temporal_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            num_heads=num_heads,
            use_cross_attention=use_cross,
            use_ag_resunit=use_ag,
            bidirectional=use_bi,
            use_multiscale=use_multiscale
        )

        trainer = Trainer(
            model, train_loader, val_loader, test_loader,
            num_epochs=epoch,
            lr=lr,
            weight_decay=weight_decay,
            use_dynamic_focal=use_focal,
            use_early_stopping=True
        )


        metrics, _ = trainer.train()
        results.append((name, metrics['auc'], metrics['auc_pr'], metrics['accuracy'], metrics['f1'], metrics['sensitivity'],metrics['specificity']))
        print(f"Result: AUC={metrics['auc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")

        # 打印消融结果表格
        print("\n" + "=" * 110)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 110)
        print(
            f"{'Configuration':<25} {'AUC':<10} {'AUC-PR':<10} {'Acc':<10} {'F1':<10} {'Sensitivity':<12} {'Specificity':<12} {'Improvement'}")
        print("-" * 110)

        base_auc = results[0][1]
        for name, auc, auc_pr, acc, f1, sens, spec in results:
            improvement = f"+{(auc - base_auc) * 100:.1f}%" if auc != base_auc else "baseline"
            print(
                f"{name:<25} {auc:<10.4f} {auc_pr:<10.4f} {acc:<10.4f} {f1:<10.4f} {sens:<12.4f} {spec:<12.4f} {improvement}")
        print("=" * 110)


def subsample_training_rate(X_static_train, X_temporal_train, y_train,
                            target_default_rate, random_state=42):
    """对某一类进行随机欠采样，调整训练集的违约率至目标水平"""
    if not 0 < target_default_rate < 1:
        raise ValueError("target_default_rate must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    y_train = np.asarray(y_train)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes are required for imbalance robustness study.")

    desired_pos_if_all_neg = int(round(target_default_rate / (1 - target_default_rate) * len(neg_idx)))
    if 1 <= desired_pos_if_all_neg <= len(pos_idx):
        keep_pos = rng.choice(pos_idx, size=desired_pos_if_all_neg, replace=False)
        keep_neg = neg_idx
    else:
        desired_neg_if_all_pos = int(round((1 - target_default_rate) / target_default_rate * len(pos_idx)))
        desired_neg_if_all_pos = max(1, min(desired_neg_if_all_pos, len(neg_idx)))
        keep_pos = pos_idx
        keep_neg = rng.choice(neg_idx, size=desired_neg_if_all_pos, replace=False)

    keep_idx = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep_idx)
    return X_static_train[keep_idx], X_temporal_train[keep_idx], y_train[keep_idx]


def run_imbalance_robustness_study(X_static_train, X_temporal_train, y_train,
                                   X_static_val, X_temporal_val, y_val,
                                   X_static_test, X_temporal_test, y_test,
                                   target_rates=(0.02, 0.05, 0.10, 0.22, 0.30),
                                   epoch=20, batch_size=128, hidden_dim=128,
                                   num_layers=8, num_heads=8, dropout=0.4,
                                   lr=5e-4, weight_decay=2e-4):
    """在受控的训练集违约率下评估模型的鲁棒性"""
    print("\n" + "=" * 60)
    print("IMBALANCE ROBUSTNESS STUDY")
    print("=" * 60)
    print(f"{'Rate':<8} {'Actual':<8} {'LSTM AUC':<10} {'AA AUC':<10} {'AA AUC-PR':<10} {'Sens':<10}")
    print("-" * 60)

    val_loader = DataLoader(
        CreditDataset(X_static_val, X_temporal_val, y_val),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        CreditDataset(X_static_test, X_temporal_test, y_test),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    results = []
    for rate in target_rates:
        try:
            xs_train, xt_train, ys_train = subsample_training_rate(
                X_static_train,
                X_temporal_train,
                y_train,
                rate,
                random_state=int(rate * 10000) + 42
            )
        except ValueError as e:
            print(f"{rate:<8.2%} skipped: {e}")
            continue

        actual_rate = float(np.mean(ys_train))
        train_loader = DataLoader(
            CreditDataset(xs_train, xt_train, ys_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        lstm_model = SequenceBaselineModel(
            static_dim=X_static_train.shape[1],
            temporal_dim=X_temporal_train.shape[2],
            hidden_dim=min(hidden_dim, 128),
            model_type='lstm',
            dropout=dropout
        )
        lstm_metrics = train_sequence_baseline(
            'Standard LSTM',
            lstm_model,
            xs_train,
            xt_train,
            ys_train,
            X_static_test,
            X_temporal_test,
            y_test,
            epochs=epoch,
            batch_size=batch_size,
            lr=lr
        )

        aa_model = AABiLSTM(
            static_dim=X_static_train.shape[1],
            temporal_dim=X_temporal_train.shape[2],
            temporal_steps=X_temporal_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            num_heads=num_heads,
            use_cross_attention=True,
            use_ag_resunit=True,
            bidirectional=True,
            use_multiscale=True
        )
        trainer = Trainer(
            aa_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=epoch,
            lr=lr,
            weight_decay=weight_decay,
            use_dynamic_focal=True,
            use_early_stopping=True
        )
        aa_metrics, _ = trainer.train()
        row = {
            'target_rate': rate,
            'actual_rate': actual_rate,
            'lstm_auc': lstm_metrics['auc'],
            'aa_auc': aa_metrics['auc'],
            'aa_auc_pr': aa_metrics['auc_pr'],
            'aa_sensitivity': aa_metrics['sensitivity'],
        }
        results.append(row)
        print(f"{rate:<8.2%} {actual_rate:<8.2%} {row['lstm_auc']:<10.4f} "
              f"{row['aa_auc']:<10.4f} {row['aa_auc_pr']:<10.4f} {row['aa_sensitivity']:<10.4f}")

    print("=" * 60)
    return results


def run_history_length_study(static_features, temporal_features, y, dataset_name,
                             history_lengths=(3, 6, 12, 24, 36), epoch=30,
                             batch_size=128, hidden_dim=128, num_layers=8,
                             num_heads=8, dropout=0.4, extend_method='linear_trend'):
    """
    长期依赖能力验证。支持数据扩展以测试更长时间窗口。

    参数:
        extend_method: 数据扩展方法 ('linear_trend', 'periodic', 'last_value', None)
                      None表示不扩展，仅使用真实数据
    """
    available_steps = temporal_features.shape[1]

    print("\n" + "=" * 60)
    print("HISTORY LENGTH STUDY")
    print("=" * 60)
    print(f"Original dataset has {available_steps} time steps")
    print(f"Testing history lengths: {history_lengths}")
    print(f"Extension method: {extend_method if extend_method else 'None (real data only)'}")
    print("-" * 60)

    # 定义三种模型配置
    model_configs = {
        'Standard LSTM': {
            'model_class': SequenceBaselineModel,
            'kwargs': {'model_type': 'lstm', 'hidden_dim': min(hidden_dim, 128), 'dropout': dropout}
        },
        'Bi-LSTM': {
            'model_class': SequenceBaselineModel,
            'kwargs': {'model_type': 'bilstm', 'hidden_dim': min(hidden_dim, 128), 'dropout': dropout}
        },
        'AA-BiLSTM': {
            'model_class': AABiLSTM,
            'kwargs': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_classes': 2,
                'dropout': dropout,
                'num_heads': num_heads,
                'use_cross_attention': True,
                'use_ag_resunit': True,
                'bidirectional': True,
                'use_multiscale': True
            }
        }
    }

    results = {name: [] for name in model_configs.keys()}

    for length in history_lengths:
        print(f"\n{'=' * 40}")
        print(f"Testing history length: {length} months")
        print(f"{'=' * 40}")

        # 决定是否需要扩展数据
        if length <= available_steps:
            # 使用真实数据（截取最后length个月）
            subset_temporal = temporal_features[:, -length:, :]
            print(f"Using real data (last {length} months from {available_steps})")
        elif extend_method:
            # 先截取到最大可用，然后扩展到目标长度
            subset_temporal = CreditDataLoader.extend_temporal_features(
                temporal_features, length, method=extend_method
            )
            print(f"Extended from {available_steps} to {length} months using '{extend_method}' method")
        else:
            print(f"Skipped: dataset only has {available_steps} steps and extension is disabled")
            for name in model_configs.keys():
                results[name].append({'history_length': length, 'auc': None, 'status': 'skipped'})
            continue

        # 更新temporal_step_names
        data_loader = CreditDataLoader()
        data_loader.temporal_step_names = [f'month_{i + 1}' for i in range(length)]
        data_loader.temporal_feature_names = ['feature_' + str(i) for i in range(temporal_features.shape[2])]

        # 数据预处理
        (X_static_train, X_temporal_train, y_train,
         X_static_val, X_temporal_val, y_val,
         X_static_test, X_temporal_test, y_test) = data_loader.preprocess(
            static_features,
            subset_temporal,
            y
        )

        train_loader = DataLoader(
            CreditDataset(X_static_train, X_temporal_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            CreditDataset(X_static_val, X_temporal_val, y_val),
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            CreditDataset(X_static_test, X_temporal_test, y_test),
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

        # 测试三种模型
        for model_name, config in model_configs.items():
            print(f"\nTraining {model_name}...")

            if config['model_class'] == SequenceBaselineModel:
                model = config['model_class'](
                    static_dim=X_static_train.shape[1],
                    temporal_dim=X_temporal_train.shape[2],
                    **config['kwargs']
                )
            else:  # AABiLSTM
                model = config['model_class'](
                    static_dim=X_static_train.shape[1],
                    temporal_dim=X_temporal_train.shape[2],
                    temporal_steps=X_temporal_train.shape[1],
                    **config['kwargs']
                )

            trainer = Trainer(
                model,
                train_loader,
                val_loader,
                test_loader,
                num_epochs=epoch,
                lr=1e-3 if dataset_name == 'german' else 5e-4,
                weight_decay=1e-4 if dataset_name == 'german' else 2e-4,
                use_dynamic_focal=False,
                use_early_stopping=True
            )
            metrics, _ = trainer.train()

            results[model_name].append({
                'history_length': length,
                'auc': metrics['auc'],
                'status': 'evaluated'
            })
            print(f"{model_name}: AUC={metrics['auc']:.4f}")

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("HISTORY LENGTH STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Length':<10} {'Standard LSTM':<15} {'Bi-LSTM':<15} {'AA-BiLSTM':<15}")
    print("-" * 70)

    for i, length in enumerate(history_lengths):
        lstm_auc = results['Standard LSTM'][i]['auc']
        bilstm_auc = results['Bi-LSTM'][i]['auc']
        aa_auc = results['AA-BiLSTM'][i]['auc']

        lstm_str = f"{lstm_auc:.4f}" if lstm_auc is not None else "N/A"
        bilstm_str = f"{bilstm_auc:.4f}" if bilstm_auc is not None else "N/A"
        aa_str = f"{aa_auc:.4f}" if aa_auc is not None else "N/A"

        print(f"{length:<10} {lstm_str:<15} {bilstm_str:<15} {aa_str:<15}")
    print("=" * 70)

    return results


def plot_training_history(history, dataset_name):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # AUC曲线
    axes[1].plot(history['val_auc'], label='Validation AUC')
    axes[1].set_title('Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].grid(True)

    auc_pr_history = history.get('val_auc_pr', [])
    if auc_pr_history:
        axes[2].plot(auc_pr_history, label='Validation AUC-PR')
    else:
        axes[2].text(0.5, 0.5, 'AUC-PR N/A', ha='center', va='center')
    axes[2].set_title('Validation AUC-PR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC-PR')
    axes[2].grid(True)

    # F1曲线
    axes[3].plot(history['val_f1'], label='Validation F1')
    axes[3].set_title('Validation F1-Score')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('F1')
    axes[3].grid(True)
    plt.suptitle(f'AA-BiLSTM Training History - {dataset_name.upper()}')
    plt.tight_layout()
    plt.savefig(f'training_history_{dataset_name}.png', dpi=150)
    plt.close(fig)
    print(f"Saved training history plot to training_history_{dataset_name}.png")


def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Default'],
                yticklabels=['Normal', 'Default'])
    plt.title(f'Confusion Matrix - {dataset_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png', dpi=150)
    plt.close()


# ==============================
# 7. 运行入口
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Paper-aligned AA-BiLSTM credit-risk experiment.'
    )
    parser.add_argument('--dataset', choices=['german', 'taiwan'], default=None)
    parser.add_argument('--data-path', default=None,
                        help='Local German .data or Taiwan .xls/.xlsx/.csv file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Default: paper setting (100 German / 150 Taiwan).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Default: paper setting (32 German / 128 Taiwan).')
    parser.add_argument('--min-sensitivity', type=float, default=0.40,
                        help='Validation sensitivity floor during threshold selection.')
    parser.add_argument('--result-json', default=None,
                        help='Optional path for scalar test metrics.')
    parser.add_argument('--checkpoint', default=None,
                        help='Optional path for the trained PyTorch state_dict.')
    return parser.parse_args()


def scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if key in {'predictions', 'probabilities', 'labels'}:
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


if __name__ == '__main__':
    args = parse_args()

    dataset_name = args.dataset
    if dataset_name is None:
        print("=" * 40)
        print("可用数据集：german / taiwan")
        while True:
            choice = input("请输入选择的数据集：").strip().lower()
            if choice in {'german', 'taiwan'}:
                dataset_name = choice
                break
            print("输入无效，请输入‘german’或‘taiwan’")

    while True:
        ans = input("是否进行验证环节（包含基准模型对比、消融实验）（Y/N）：").strip().upper()
        if ans in {'Y', 'N'}:
            run_analysis = (ans == 'Y')
            break
        print("输入无效，请输入 Y 或 N。")

    while True:
        ans = input("是否绘图（Y/N）：").strip().upper()
        if ans in {'Y', 'N'}:
            make_plots = (ans == 'Y')
            break
        print("输入无效，请输入 Y 或 N。")

    print("Starting experiments...")

    model, metrics, history = run_experiment(
        dataset_name=dataset_name,
        epoch=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        run_analysis=run_analysis,
        make_plots=make_plots,
        threshold_min_sensitivity=args.min_sensitivity
    )

    if make_plots:
        plot_confusion_matrix(metrics['labels'], metrics['predictions'], dataset_name)

    if args.result_json:
        result_dir = os.path.dirname(os.path.abspath(args.result_json))
        os.makedirs(result_dir, exist_ok=True)
        with open(args.result_json, 'w', encoding='utf-8') as result_file:
            json.dump(scalar_metrics(metrics), result_file, ensure_ascii=False, indent=2)
        print(f"Saved scalar metrics to {args.result_json}")

    if args.checkpoint:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved model checkpoint to {args.checkpoint}")

    print("\nExperiment completed successfully!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f} (Target: >0.8000)")
    print(f"Final AUC: {metrics['auc']:.4f}")
    print(f"Final Sensitivity: {metrics['sensitivity']:.4f}")