import numpy as np
import pandas as pd
import os
import copy
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression  # Platt calibration
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statistics import NormalDist
import warnings

# 抑制已知无害的第三方库警告，保留模型行为相关的真实警告
warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*lbfgs failed to converge.*',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn',
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='seaborn',
)
# matplotlib 非交互环境下 Agg 后端无害警告
warnings.filterwarnings(
    'ignore',
    message='.*Figures are typically created.*',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*Solver terminated early.*',
    category=UserWarning,
)

# ============================================================
# AA-BiLSTM 信用风险预测模型 — 论文代码
# ============================================================
# 本代码主要包括：
#   1. 数据加载与预处理（Taiwan Credit 数据集）
#   2. 模型架构：AA-BiLSTM（自适应双向 LSTM）——
#      包含 AdaptiveFusion 融合模块、AG-ResUnit 门控残差单元、
#      MultiScaleTemporalEncoder 多尺度时序编码器
#   3. DynamicFocalLoss 动态焦点损失函数
#   4. 训练器（含 Platt 概率校准、阈值搜索、EMA、混合精度）
#   5. SHAP 可解释性分析
#   6. 对比实验（传统 ML / 深度学习基线、消融实验、不平衡鲁棒性）
#
# 运行入口：python final_essay.py --dataset taiwan
# ============================================================

TAIWAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


def set_seed(seed, deterministic=True):
    """设置 Python、NumPy 和 PyTorch 随机种子，确保实验可复现。

    Args:
        seed: 随机种子值
        deterministic: 是否启用 cuDNN 确定性模式（启用后速度略降但结果可复现）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


# 设备选择：优先使用 CUDA GPU，否则回退到 CPU
device = torch.device('cpu')   #如果有cuda的话可以搞这一句：'cuda' if torch.cuda.is_available() else 'cpu'
# 启用高精度矩阵乘法（PyTorch 2.0+ 支持），提升 float32 计算精度
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


# ==============================
# 1. 数据加载与预处理
# ==============================

class CreditDataLoader:
    """
    Taiwan Credit 数据集的加载与预处理。

    负责：
    - 加载原始数据（本地文件或 UCI 在线源）
    - 构建静态特征（数值 + one-hot 类别）和时序特征（真实月度数据）
    - 训练集拟合类别编码与 StandardScaler，避免分布信息泄漏
    - 按 train / validation / calibration / test 四层分割数据
    - 导出部署所需的预处理状态（特征名、缩放器参数、类别映射）

    Attributes:
        dataset_name: 数据集名称
        scaler: 静态连续特征的 StandardScaler
        temporal_scaler: 时序连续特征的 StandardScaler
        static_feature_names: 静态特征列名列表
        temporal_feature_names: 时序特征维度名列表
        temporal_step_names: 时序步名列表
    """

    def __init__(self):
        self.dataset_name = None
        self.scaler = StandardScaler()
        self.temporal_scaler = StandardScaler()
        self.static_feature_names = None
        self.temporal_feature_names = None
        self.temporal_step_names = None
        self.temporal_flat_feature_names = None
        self.continuous_static_cols_ = None
        self.continuous_temporal_cols_ = None
        self.temporal_categorical_indices_ = []
        self.static_dummy_columns_ = None
        self.split_indices_ = None
        self.calibration_data_ = None

        # 原始静态字段会保留到数据切分之后；one-hot 词表只能用训练集拟合。
        self._raw_static_numeric = None
        self._raw_static_numeric_names = None
        self._raw_static_categorical = None
        self._raw_static_categorical_names = None
        self._raw_static_prefixes = None


    def load_taiwan_credit(self, filepath=None):
        """
        加载 Taiwan Credit Card 数据集（30000 样本, 24 特征, 约 22% 违约率）。

        处理流程：
        - 读取 .xls/.xlsx/.csv 文件（兼容 UCI 原始格式）
        - 构建静态特征：基础数值特征（LIMIT_BAL, AGE）+ one-hot 类别 + 工程特征
          （账单利用率趋势、还款冲击、逾期统计等 9 个衍生特征）
        - 构建时序特征（6 个月 × 5 通道）：还款状态、有符号 log 账单金额、
          有符号 log 还款金额、账单利用率、还款占账单比
        - 时序方向统一为"最早月 → 最近月"

        Args:
            filepath: 本地文件路径，为 None 时自动从 UCI 在线下载

        Returns:
            static_features: (N, S) 静态特征矩阵
            temporal_features: (N, 6, 5) 时序特征矩阵
            y: (N,) 二分类标签（1=违约）

        Raises:
            FileNotFoundError: 文件路径无效
            ImportError: 缺少读取 Excel 所需的库（xlrd >= 2.0.1）
            KeyError: 数据缺少必要列
            RuntimeError: 读取过程出错
        """
        # 不从 sys.argv 猜测路径；否则 `--dataset taiwan` 会被误认为文件名。
        self.dataset_name = 'taiwan'
        self.temporal_categorical_indices_ = [0]
        if filepath is None:
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
        except (OSError, Exception) as e:
            # UCI 服务器 SSL 证书在某些环境（如 Python 3.12+）下不被信任；
            # 回退到手动下载 + 宽松 SSL 模式
            if is_url and ext in ['.xls', '.xlsx']:
                try:
                    import ssl
                    import urllib.request
                    import io

                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    with urllib.request.urlopen(
                        filepath, context=ctx, timeout=30
                    ) as resp:
                        raw_bytes = resp.read()
                    df = pd.read_excel(io.BytesIO(raw_bytes), header=1)
                except Exception as fallback_e:
                    raise RuntimeError(
                        f"无法读取数据集（直接读取与 SSL 回退均失败）。\n"
                        f"直接读取错误: {e}\n"
                        f"SSL 回退错误: {fallback_e}\n"
                        "请手动下载数据集并通过 --data-path 指定本地路径。\n"
                        f"下载地址: {TAIWAN_CREDIT_URL}"
                    ) from fallback_e
            else:
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
        # limit_balance 至少为 1，避免除以零
        limit_balance = np.maximum(df['LIMIT_BAL'].astype(float).values, 1.0)
        bill_values = df[bill_cols].astype(float).values          # (N, 6) 账单金额
        payment_values = df[pay_amt_cols].astype(float).values    # (N, 6) 还款金额
        status_values = df[pay_cols].astype(float).values         # (N, 6) 还款状态

        # --- 原始比率特征（clamp 到 ±10 防止极端值主导损失） ---
        utilization = np.clip(bill_values / limit_balance[:, None], -10.0, 10.0)
        # 安全的 bill 绝对值，防止 payment_to_bill 被零除
        safe_bill = np.maximum(np.abs(bill_values), 1.0)
        payment_to_bill = np.clip(payment_values / safe_bill, -10.0, 10.0)

        # 账单利用率趋势：将利用率按时间正序排列（最早→最近），
        # 用 de-mean 后的时间轴做线性回归，斜率即 6 个月趋势
        chronological_bill = utilization[:, ::-1]
        trend_axis = np.arange(chronological_bill.shape[1], dtype=np.float32)
        trend_axis -= trend_axis.mean()  # 中心化，使截距不等于第 0 月的值
        trend_denominator = float(np.square(trend_axis).sum())
        bill_trend_6m = (chronological_bill @ trend_axis) / max(trend_denominator, 1e-7)

        # --- 9 个工程特征（从原始 6 个月数据手工推导） ---
        engineered_static = pd.DataFrame({
            'recent_bill_to_limit': utilization[:, 0],              # 最近月账单利用率
            'avg_bill_to_limit_6m': utilization.mean(axis=1),       # 6 月平均利用率
            'max_bill_to_limit_6m': utilization.max(axis=1),        # 6 月最高利用率（捕捉极端透支）
            'bill_trend_6m': bill_trend_6m,                         # 利用率线性趋势（正=恶化）
            'recent_payment_shock': status_values[:, 0] - status_values[:, 1],  # 最近两月还款状态突变
            'delinquency_count_6m': (status_values > 0).sum(axis=1),  # 6 月内逾期月数
            'max_delinquency_6m': status_values.max(axis=1),        # 6 月内最高逾期级别
            'recent_payment_to_bill': payment_to_bill[:, 0],        # 最近月还款占账单比
            'avg_payment_to_bill_6m': payment_to_bill.mean(axis=1), # 6 月平均还款占比
        }).clip(-10.0, 10.0)  # clamp 所有工程特征，防止极端比率破坏训练稳定性

        self._raw_static_numeric = pd.concat(
            [
                df[static_num_cols].astype(np.float32).reset_index(drop=True),
                engineered_static.astype(np.float32).reset_index(drop=True),
            ],
            axis=1,
        ).to_numpy(dtype=np.float32)
        self._raw_static_numeric_names = (
            static_num_cols + [str(c) for c in engineered_static.columns]
        )
        self._raw_static_categorical = (
            df[static_cat_cols].astype(str).to_numpy()
        )
        self._raw_static_categorical_names = list(static_cat_cols)
        self._raw_static_prefixes = list(static_cat_cols)

        static_df = pd.concat(
            [base_static_df.reset_index(drop=True), engineered_static.reset_index(drop=True)],
            axis=1
        )
        static_features = static_df.astype(np.float32).values
        self.static_feature_names = [str(c) for c in static_df.columns]

        # 时序特征构建：
        # 原始字段顺序默认为”最近月 → 最早月”；统一反转为”最早月 → 最近月”，
        # 使局部因果窗口和趋势方向符合真实时间顺序。
        #
        # 金额（bill/payment）是强长尾变量，直接输入会导致梯度被少数极端值主导；
        # signed_log1p: sign(x) * log(1 + |x|) 保留正负方向同时压缩量级。
        bill_signal = np.sign(bill_values) * np.log1p(np.abs(bill_values))
        payment_signal = np.sign(payment_values) * np.log1p(
            np.abs(payment_values)
        )
        # 5 通道 × 6 月，反转时间轴后得到 (N, 6, 5)
        temporal_features = np.stack([
            status_values,      # 还款状态（-2~8，离散序数值）
            bill_signal,        # 有符号 log 账单金额
            payment_signal,     # 有符号 log 还款金额
            utilization,        # 账单利用率
            payment_to_bill,    # 还款占账单比
        ], axis=-1)[:, ::-1, :].copy().astype(np.float32)
        self.temporal_feature_names = [
            'payment_status',
            'signed_log_bill_amount',
            'signed_log_payment_amount',
            'bill_to_limit',
            'payment_to_bill',
        ]
        self.temporal_step_names = [
            'month_6', 'month_5', 'month_4',
            'month_3', 'month_2', 'recent_month'
        ]
        self.temporal_flat_feature_names = [
            name
            for step in reversed(range(6))
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

    def _encode_static_from_training(self, train_idx, *evaluation_indices):
        """使用训练集类别词表对多个数据子集做 one-hot 编码。

        先对训练集拟合 one-hot 列，再对验证/校准/测试集对齐到相同列空间，
        避免测试数据的类别信息泄漏到训练阶段。

        Args:
            train_idx: 训练样本索引
            *evaluation_indices: 待编码的评估集索引（如 val_idx, calibration_idx, test_idx）

        Returns:
            tuple of ndarray: 各子集的编码后静态特征；无原始数据时返回 None
        """
        if self._raw_static_numeric is None:
            return None

        numeric = np.asarray(self._raw_static_numeric, dtype=np.float32)
        categorical = self._raw_static_categorical
        if categorical is None or categorical.shape[1] == 0:
            self.static_feature_names = list(self._raw_static_numeric_names)
            return tuple(
                numeric[idx].copy()
                for idx in (train_idx, *evaluation_indices)
            )

        columns = list(self._raw_static_categorical_names)
        prefixes = list(self._raw_static_prefixes)

        def dummy_frame(indices):
            frame = pd.DataFrame(categorical[indices], columns=columns)
            return pd.get_dummies(
                frame.astype(str),
                prefix=prefixes,
                drop_first=False,
                dtype=np.float32,
            )

        train_dummy = dummy_frame(train_idx)
        self.static_dummy_columns_ = [str(c) for c in train_dummy.columns]

        def combine(indices, dummy):
            # 评估集可能缺少训练集中某些类别 → reindex 补 0，
            # 确保所有子集的 one-hot 列顺序和维度完全一致
            aligned = dummy.reindex(
                columns=train_dummy.columns,
                fill_value=0.0,
            ).to_numpy(dtype=np.float32)
            return np.concatenate([numeric[indices], aligned], axis=1)

        self.static_feature_names = (
            list(self._raw_static_numeric_names) + self.static_dummy_columns_
        )
        return tuple(
            [combine(train_idx, train_dummy)]
            + [
                combine(indices, dummy_frame(indices))
                for indices in evaluation_indices
            ]
        )

    @staticmethod
    def _scaler_state(scaler):
        """导出 StandardScaler 的拟合参数，用于部署时复现完全相同的缩放。

        mean_/scale_ → 标准化公式 x' = (x - mean)/scale；
        n_samples_seen_ → 在线部署时如需增量更新 scaler 可继续累加。
        """
        if not hasattr(scaler, 'mean_'):
            return None
        n_seen = scaler.n_samples_seen_
        return {
            'mean': np.asarray(scaler.mean_).tolist(),
            'scale': np.asarray(scaler.scale_).tolist(),
            'var': np.asarray(scaler.var_).tolist(),
            'n_features_in': int(scaler.n_features_in_),
            'n_samples_seen': (
                int(n_seen) if np.ndim(n_seen) == 0
                else np.asarray(n_seen).tolist()
            ),
        }

    def export_preprocessing_state(self):
        """导出部署所需的特征顺序、缩放器与类别映射。"""
        return {
            'static_feature_names': list(self.static_feature_names or []),
            'temporal_feature_names': list(self.temporal_feature_names or []),
            'temporal_step_names': list(self.temporal_step_names or []),
            'temporal_flat_feature_names': list(
                self.temporal_flat_feature_names or []
            ),
            'continuous_static_cols': (
                self.continuous_static_cols_.astype(bool).tolist()
                if self.continuous_static_cols_ is not None else None
            ),
            'continuous_temporal_cols': (
                self.continuous_temporal_cols_.astype(bool).tolist()
                if self.continuous_temporal_cols_ is not None else None
            ),
            'temporal_categorical_indices': list(
                self.temporal_categorical_indices_
            ),
            'dataset_name': self.dataset_name,
            'static_dummy_columns': list(self.static_dummy_columns_ or []),
            'raw_static_numeric_names': list(
                self._raw_static_numeric_names or []
            ),
            'raw_static_categorical_names': list(
                self._raw_static_categorical_names or []
            ),
            'raw_static_prefixes': list(self._raw_static_prefixes or []),
            'static_scaler': self._scaler_state(self.scaler),
            'temporal_scaler': self._scaler_state(self.temporal_scaler),
            'taiwan_amount_transform': 'signed_log1p_v1',
        }

    def preprocess(
        self,
        static_features,
        temporal_features,
        y,
        test_size=0.15,
        val_size=0.10,
        calibration_size=0.10,
        split_seed=42,
    ):
        """
        数据预处理主流程：分层分割 + 编码 + 标准化。

        流程：
        1. 输入验证（形状、有限性、二分类标签）
        2. 按 train / validation / calibration / test 四层分层分割
        3. 训练集拟合类别 one-hot 词表，其他集对齐
        4. 训练集拟合 StandardScaler，只对连续列做标准化
        5. 训练集拟合时序缩放器，对评估集做 transform

        Args:
            static_features: (N, S) 静态特征
            temporal_features: (N, T, F) 时序特征
            y: (N,) 二分类标签
            test_size: 测试集比例
            val_size: 验证集比例
            calibration_size: Platt 校准集比例
            split_seed: 分层分割随机种子

        Returns:
            9 元组: (X_static_train, X_temporal_train, y_train,
                      X_static_val, X_temporal_val, y_val,
                      X_static_test, X_temporal_test, y_test)
            注：校准集数据存储在 self.calibration_data_ 属性中
        """
        static_features = np.asarray(static_features, dtype=np.float32)
        temporal_features = np.asarray(temporal_features, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if static_features.ndim != 2 or temporal_features.ndim != 3:
            raise ValueError(
                "Expected static [N, S] and temporal [N, T, F] arrays."
            )
        if not (
            len(static_features) == len(temporal_features) == len(y)
        ):
            raise ValueError("Static, temporal and label sample counts must match.")
        if not np.isfinite(static_features).all() or not np.isfinite(
            temporal_features
        ).all():
            raise ValueError("Input features contain NaN or infinite values.")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Labels must contain both binary values 0 and 1.")
        if (
            test_size <= 0.0
            or val_size <= 0.0
            or calibration_size <= 0.0
            or test_size + val_size + calibration_size >= 1.0
        ):
            raise ValueError(
                "test_size, val_size and calibration_size must be positive "
                "and sum to less than 1."
            )

        # 分层分割顺序：先切 test，再从剩余 development 中切 train / (val+calibration)，
        # 最后在 operating 中把 calibration 从 val 中分离出来。
        # 每步 test_size 需按当前剩余比例换算，确保最终四子集比例精确匹配：
        #   test = test_size（全局比例）
        #   val = (1-test_size) * heldout_fraction（development 内的 val 比例）
        #   calibration = (1-test_size) * calibration_fraction（development 内的 cal 比例）
        all_indices = np.arange(len(y))
        # 第一步：切出测试集
        development_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        # 第二步：在 development 中切出训练集和操作集（val+calibration）
        heldout_fraction = val_size + calibration_size
        train_idx, operating_idx = train_test_split(
            development_idx,
            test_size=heldout_fraction / (1.0 - test_size),  # 换算到 development 内部的相对比例
            random_state=split_seed + 1,
            stratify=y[development_idx],
        )
        # 第三步：在 operating 中分离验证集和校准集
        calibration_fraction = calibration_size / heldout_fraction
        val_idx, calibration_idx = train_test_split(
            operating_idx,
            test_size=calibration_fraction,  # operating 内部校准集相对占比
            random_state=split_seed + 2,
            stratify=y[operating_idx],
        )
        self.split_indices_ = {
            'train': train_idx.copy(),
            'validation': val_idx.copy(),
            'calibration': calibration_idx.copy(),
            'test': test_idx.copy(),
            'split_seed': int(split_seed),
        }

        encoded_static = self._encode_static_from_training(
            train_idx,
            val_idx,
            calibration_idx,
            test_idx,
        )
        if encoded_static is None:
            X_static_train = static_features[train_idx].copy()
            X_static_val = static_features[val_idx].copy()
            X_static_calibration = static_features[calibration_idx].copy()
            X_static_test = static_features[test_idx].copy()
        else:
            (
                X_static_train,
                X_static_val,
                X_static_calibration,
                X_static_test,
            ) = encoded_static

        X_temporal_train = temporal_features[train_idx].copy()
        X_temporal_val = temporal_features[val_idx].copy()
        X_temporal_calibration = temporal_features[
            calibration_idx
        ].copy()
        X_temporal_test = temporal_features[test_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_calibration = y[calibration_idx]
        y_test = y[test_idx]

        # 区分 binary/continuous 列：one-hot 列值只有 0 和 1，
        # 不应被 StandardScaler 缩放（会破坏语义）；只对连续列做标准化。
        binary_static_cols = np.all((X_static_train == 0) | (X_static_train == 1), axis=0)
        continuous_static_cols = ~binary_static_cols
        self.continuous_static_cols_ = continuous_static_cols.copy()
        if np.any(continuous_static_cols):
            X_static_train = X_static_train.copy()
            X_static_val = X_static_val.copy()
            X_static_calibration = X_static_calibration.copy()
            X_static_test = X_static_test.copy()
            # train 集上 fit_transform，其余集仅 transform（防止信息泄漏）
            X_static_train[:, continuous_static_cols] = self.scaler.fit_transform(
                X_static_train[:, continuous_static_cols]
            )
            X_static_val[:, continuous_static_cols] = self.scaler.transform(
                X_static_val[:, continuous_static_cols]
            )
            X_static_calibration[:, continuous_static_cols] = (
                self.scaler.transform(
                    X_static_calibration[:, continuous_static_cols]
                )
            )
            X_static_test[:, continuous_static_cols] = self.scaler.transform(
                X_static_test[:, continuous_static_cols]
            )

        n_train, time_steps, n_features = X_temporal_train.shape
        self.temporal_scaler = StandardScaler()
        # 时序特征中可能含类别通道（如 Taiwan 的 PAY 状态），
        # 这些通道不应被标准化。continuous_temporal_cols 标记哪些通道可缩放。
        continuous_temporal_cols = np.ones(n_features, dtype=bool)
        for categorical_index in self.temporal_categorical_indices_:
            if not 0 <= categorical_index < n_features:
                raise ValueError(
                    "Temporal categorical feature index is out of range."
                )
            continuous_temporal_cols[categorical_index] = False
        self.continuous_temporal_cols_ = continuous_temporal_cols.copy()

        def scale_temporal(x, fit=False):
            """对 (N, T, F) 的时序特征按通道缩放。

            将 (N*T, F_continuous) 展平后做 StandardScaler，
            再恢复为 (N, T, F)，保证每个时间步使用相同的缩放参数。
            """
            result = x.astype(np.float32, copy=True)
            if not np.any(continuous_temporal_cols):
                return result
            reshaped = result[:, :, continuous_temporal_cols].reshape(
                -1,
                int(continuous_temporal_cols.sum()),
            )
            # (N*T, F_cont) 展平 → 缩放 → 恢复回 (N, T, F_cont)，
            # 再按 continuous_temporal_cols 掩码写回原始 (N, T, F) 矩阵中对应列。
            # 类别通道（如 PAY 状态）保持原值不变。
            scaled = (
                self.temporal_scaler.fit_transform(reshaped)
                if fit else self.temporal_scaler.transform(reshaped)
            )
            result[:, :, continuous_temporal_cols] = scaled.reshape(
                x.shape[0],
                time_steps,
                int(continuous_temporal_cols.sum()),
            )
            return result

        X_temporal_train = scale_temporal(
            X_temporal_train, fit=True
        ).astype(np.float32, copy=False)
        X_temporal_val = scale_temporal(
            X_temporal_val
        ).astype(np.float32, copy=False)
        X_temporal_calibration = scale_temporal(
            X_temporal_calibration
        ).astype(np.float32, copy=False)
        X_temporal_test = scale_temporal(
            X_temporal_test
        ).astype(np.float32, copy=False)
        X_static_train = X_static_train.astype(np.float32, copy=False)
        X_static_val = X_static_val.astype(np.float32, copy=False)
        X_static_calibration = X_static_calibration.astype(
            np.float32,
            copy=False,
        )
        X_static_test = X_static_test.astype(np.float32, copy=False)
        self.calibration_data_ = (
            X_static_calibration,
            X_temporal_calibration,
            y_calibration,
        )

        # 验证时序类别通道的值接近整数（PAY 状态等），
        # 若偏离超过 1e-3 说明缩放参数可能错误应用到了类别通道。
        for cat_idx in self.temporal_categorical_indices_:
            cat_vals = X_temporal_train[:, :, cat_idx]
            rounded = np.round(cat_vals)
            max_deviation = float(np.abs(cat_vals - rounded).max())
            if max_deviation > 1e-3:
                raise ValueError(
                    f"Temporal categorical channel {cat_idx} values are not "
                    f"close to integers (max deviation={max_deviation:.4f}). "
                    "Ensure categorical channels are excluded from scaling."
                )

        return (
            X_static_train, X_temporal_train, y_train,
            X_static_val, X_temporal_val, y_val,
            X_static_test, X_temporal_test, y_test,
        )


class CreditDataset(Dataset):
    """用于信用风险数据的 PyTorch Dataset。

    每个样本返回一个字典：{'static': (S,), 'temporal': (T, F), 'label': scalar}

    Args:
        static_features: (N, S) 静态特征
        temporal_features: (N, T, F) 时序特征
        labels: (N,) 标签
    """

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

class AdaptiveFusion(nn.Module):

    """
    自适应全局-局部门控融合（Adaptive Global-Local Fusion）。

    全局分支通过多统计量（mean/max/last/std/trend + static + multi-scale context）
    非线性聚合建模长期信用状态。局部分支仅在固定因果窗口内计算冲击注意力，
    显式编码相邻时间步的突变，配合 event_encoder 捕获 [当前值, 一阶差分, 差分绝对值]
    三种局部冲击信号。最终通过可学习门控残差方式注入原始时序，
    不替代或抹平序列信息。

    局部注意力权重形状为 [B, T, W]（W=固定窗口，默认 3）。

    Args:
        hidden_dim: 隐藏维度
        window_size: 局部注意力因果窗口大小
        dropout: Dropout 比率
        use_global_context: 是否启用全局上下文分支
        use_local_attention: 是否启用局部冲击注意力分支
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
        # 低秩瓶颈：将中间表示压缩到 hidden_dim/4，显著减少门控网络参数量。
        # d=64 时 bottleneck=16，d=128 时 bottleneck=32。
        bottleneck = max(hidden_dim // 4, 8)

        if use_global_context:
            # 全局编码输入 = 5 个时序统计量 + 静态特征 + 多尺度上下文
            # = mean/max/last/std/trend (5×hidden_dim) + static (hidden_dim) + multiscale (hidden_dim)
            # = 7 × hidden_dim
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
            # 事件编码器：将 [当前值, 一阶差分, 差分绝对值] 三个视角融合为局部冲击表示
            self.event_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 3, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            # query: 融合时序当前值 + 静态 + 全局上下文 → 决定"关注什么"
            self.local_query = nn.Linear(hidden_dim * 3, bottleneck)
            # key: 窗口内每个 token 的表示 → 决定"被关注什么"
            self.local_key = nn.Linear(hidden_dim, bottleneck)
            # score: 单标量输出（bias=False 避免学习到常数偏向）
            self.local_score = nn.Linear(bottleneck, 1, bias=False)
            # value: 加权聚合前的值投影
            self.local_value = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.event_encoder = None
            self.local_query = None
            self.local_key = None
            self.local_score = None
            self.local_value = None

        # 门控输入维度 = 5×hidden_dim + 1：
        #   temporal_feat, local_context, global_expanded,
        #   temporal×global（元素乘积交互）, |temporal-global|（差异），
        #   shock_score（单通道冲击强度）
        gate_input_dim = hidden_dim * 5 + 1
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim * 2)  # 输出 local_gate 和 global_gate 各 hidden_dim 维
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _compute_shock_score(temporal_feat):
        """计算每个时间步的冲击强度 = sqrt(mean(delta²))。

        使用 delta 的 L2 范数度量相邻步之间的整体变化幅度。
        +1e-8 防止 sqrt(0) 导致 NaN。
        """
        delta = torch.zeros_like(temporal_feat)
        if temporal_feat.shape[1] > 1:
            delta[:, 1:] = temporal_feat[:, 1:] - temporal_feat[:, :-1]
        return (delta.square().mean(dim=-1) + 1e-8).sqrt()

    @staticmethod
    def _temporal_statistics(temporal_feat):
        """提取时序的五种全局统计量。

        Returns:
            (mean, maximum, last, std, trend) 五元组，每个形状为 (B, hidden_dim)
        """
        mean = temporal_feat.mean(dim=1)
        maximum = temporal_feat.max(dim=1).values
        last = temporal_feat[:, -1]
        std = temporal_feat.std(dim=1, unbiased=False)
        trend = temporal_feat[:, -1] - temporal_feat[:, 0]
        return mean, maximum, last, std, trend

    @staticmethod
    def _causal_window_mask(time_steps, window_size, device):
        """构建因果注意力掩码，确保每个时间步只能看到当前及之前的窗口位置。

        返回 (window_size, time_steps) 的 bool 矩阵：
          mask[w, t] = True 当且仅当窗口位置 w 对时间步 t 可见。

        例：window_size=3, time_steps=5 时：
          位置 0(t-2) 位置 1(t-1) 位置 2(t)
        t=0:   F          F          T     (只能看当前)
        t=1:   F          T          T     (看当前和前1步)
        t>=2:  T          T          T     (完整窗口)

        推导：对于时间步 t，可见的最小窗口位置 = window_size - 1 - t，
        即位置 w 可见 ⇔ w >= window_size - 1 - t。
        """
        positions = torch.arange(window_size, device=device).unsqueeze(0)   # (1, W)
        first_valid = window_size - 1 - torch.arange(
            time_steps, device=device
        ).unsqueeze(1)                                                       # (T, 1)
        return positions >= first_valid                                      # (T, W)

    def _local_context(self, temporal_feat, static_feat, global_context):
        """在因果窗口内计算局部冲击注意力。

        步骤：
        1. 计算一阶差分 delta 和冲击强度 shock_score
        2. event_encoder 将 [当前值, delta, |delta|] 编码为 event token
        3. 展开因果窗口，query 融合时序/静态/全局上下文后对窗口内 token 打分
        4. softmax 加权聚合得到局部上下文表示

        Returns:
            (local_context, attention_weights, shock_score):
            - local_context: (B, T, hidden_dim) 局部上下文
            - attention_weights: (B, T, W) 注意力权重
            - shock_score: (B, T) 每步冲击强度
        """
        batch_size, time_steps, hidden_dim = temporal_feat.shape

        # delta: 一阶差分，同时用于构建 event token 和计算 shock_score
        delta = torch.zeros_like(temporal_feat)
        if time_steps > 1:
            delta[:, 1:] = temporal_feat[:, 1:] - temporal_feat[:, :-1]
        # shock_score = sqrt(mean(delta²))，+1e-8 防 sqrt(0) 梯度爆炸
        shock_score = (delta.square().mean(dim=-1) + 1e-8).sqrt()
        event_tokens = self.event_encoder(
            torch.cat([temporal_feat, delta, delta.abs()], dim=-1)
        )

        # 因果窗口展开：先在时间维左端补 window_size-1 个零 token，
        # 使第 0 步也能"看到"一个完整窗口（前面是 padding 的零），
        # 再 unfold 出每个时间步对应的 W 个连续 token。
        # 变换链：
        #   pad:     (B, T, D) → (B, T+W-1, D)  左端补零
        #   unfold:  (B, T+W-1, D) → (B, T, D, W)  沿时间维滑窗
        #   permute: (B, T, D, W) → (B, T, W, D)  W 维移到 D 前，方便后续 batch matmul
        window_size = min(self.window_size, time_steps)
        padded = F.pad(event_tokens, (0, 0, window_size - 1, 0))  # (0,0)=不补D, (W-1,0)=左补W-1右补0
        windows = padded.unfold(1, window_size, 1).permute(0, 1, 3, 2)  # → (B, T, W, D)

        static_context = static_feat.unsqueeze(1).expand(
            batch_size, time_steps, hidden_dim
        )
        global_expanded = global_context.unsqueeze(1).expand(
            batch_size, time_steps, hidden_dim
        )
        # query: (B, T, 1, bottleneck) — unsqueeze(2) 插入窗口维用于与
        #   key(windows): (B, T, W, bottleneck) 做 broadcasting 加法
        # additive attention: score = linear(tanh(query + key))
        query = self.local_query(
            torch.cat([temporal_feat, static_context, global_expanded], dim=-1)
        ).unsqueeze(2)

        scores = self.local_score(
            torch.tanh(query + self.local_key(windows))
        ).squeeze(-1)  # → (B, T, W)
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
        """前向传播：全局聚合 + 局部冲击注意力 + 门控残差融合。

        Args:
            static_feat: (B, hidden_dim) 静态特征嵌入
            temporal_feat: (B, T, hidden_dim) 投影后的时序特征
            multi_scale_context: (B, multi_scale_dim) 多尺度上下文，可选

        Returns:
            (fused_sequence, diagnostics):
            - fused_sequence: (B, T, hidden_dim) 融合后的时序特征
            - diagnostics: dict 含 local_attention/局部门/全局门/shock_score 等诊断信息
        """
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
            shock_score = self._compute_shock_score(temporal_feat)

        global_expanded = global_context.unsqueeze(1).expand_as(temporal_feat)
        # shock_score 始终计算（用于 diagnostics），但仅在启用局部注意力时
        # 才注入门控网络；否则门控输入的 shock 通道置零。
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

        # 四个标准门 f/i/o/c 共用一次矩阵乘法输出 4×hidden_dim，
        # 减少 GPU kernel launch 次数（4 次 → 1 次），提升训练吞吐。
        self.gate_projection = nn.Linear(
            input_dim + hidden_dim,
            hidden_dim * 4,
        )

        # 自适应门控系数 λ：由当前输入和历史状态动态生成，每个样本每步独立。
        # λ ∈ [0,1]，通过 sigmoid 约束；λ 大 → 偏记忆，λ 小 → 偏更新。
        self.lambda_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # bias 初始化为 -1.5 → sigmoid(-1.5) ≈ 0.18，
        # 即训练初期 λ ≈ 0.18，接近标准 LSTM 门控行为，
        # 训练后期 λ 可随梯度上升，逐步引入更强的长期记忆。
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

        # 原始门控一次投影后分 4 块（各 hidden_dim 维）
        f_raw, i_raw, o_raw, c_raw = self.gate_projection(combined).chunk(
            4,
            dim=-1,
        )
        f_tilde = torch.sigmoid(f_raw)   # 遗忘门基础值
        i_tilde = torch.sigmoid(i_raw)   # 输入门基础值
        o_tilde = torch.sigmoid(o_raw)   # 输出门

        # 自适应门控系数：λ 越大 → 遗忘门越偏向 1（保留旧记忆），
        #                输入门越偏向 0（抑制新信息），即更偏”记忆”模式。
        lambda_val = self.lambda_gate(combined)

        # 候选细胞状态
        c_tilde = torch.tanh(c_raw)

        # λ 调制后的有效门控：
        # - f_t: λ=0 → f_tilde（标准遗忘），λ=1 → 1（完全保留）
        # - i_t: λ=0 → i_tilde（标准输入），λ=1 → 0（完全阻断新信息）
        # - o_t 不受 λ 影响，保持标准输出门行为
        f_t = lambda_val + (1.0 - lambda_val) * f_tilde
        i_t = (1.0 - lambda_val) * i_tilde
        o_t = o_tilde

        # 细胞状态更新
        c_t = f_t * c_prev + i_t * c_tilde

        # 隐藏状态
        h_t = o_t * torch.tanh(c_t)

        # 残差连接：维度不匹配时用投影对齐；匹配时直接恒等
        # 由于 __init__ 保证 input_dim != hidden_dim 时才创建 residual_proj，
        # else 分支中 x.shape[-1] == hidden_dim 恒成立。
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x

        h_t = h_t + residual  # 残差相加
        h_t = self.layer_norm(h_t)

        return h_t, c_t


class AGBiLSTMLayer(nn.Module):
    """
    基于 AG-ResUnit 的多层门控 LSTM 层。

    支持切换 单向/双向 以支持消融实验中分别评估 AG-ResUnit 和 Bi-Directional
    各自的贡献。多层之间通过逐层传递隐藏状态堆叠，每层可选择独立的
    AdaptiveGatedResUnit 实例。

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏维度
        num_layers: AG-ResUnit 堆叠层数
        dropout: 层间 dropout 比率
        bidirectional: 是否开启双向时序处理
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
        """按指定方向逐层运行 AG-ResUnit 堆叠。

        每层独立维护隐藏/细胞状态对 (h, c)，按时间步迭代更新。
        多层之间将上一层的隐藏状态作为下一层的输入。

        Args:
            x: (B, T, input_dim) 输入序列
            layers: nn.ModuleList 中的 AG-ResUnit 层列表
            reverse: True 表示从 T-1 向 0 反向迭代

        Returns:
            (B, T, hidden_dim) 该方向最后一层所有时间步的输出
        """
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
    多尺度时序编码器。

    在三个时间尺度上分别提取表示：
    - 细粒度（月度）：原始月频数据的统计量 / BiLSTM 编码
    - 中粒度（季度）：3 个月滑动平均后编码
    - 粗粒度（年度）：12 个月滑动窗口编码；不足 12 个月时退化为全局趋势摘要

    支持两种模式：
    - 'legacy'：每个尺度用独立 BiLSTM 编码
    - 'lightweight'：用统计量 + Conv1d 平滑替代 BiLSTM（参数更少，适合 Taiwan）

    Args:
        input_dim: 时序特征维度
        hidden_dim: 单方向隐藏维度（输出为 2*hidden_dim）
        coarse_window: 粗粒度滑动窗口大小（默认 12）
        max_steps: 最大时间步数（用于判断 coarse_encoder 是否启用）
        mode: 'legacy' 或 'lightweight'
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
        self.coarse_fallback = None  # 仅 legacy 模式使用，lightweight 设为 None

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
            # 分组卷积（groups=input_dim）→ 每个通道独立做 1D 卷积
            # 权重初始化为 1/3 → 等价于 3 个月简单滑动平均（不学习）
            self.medium_filter = nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size=3,
                groups=input_dim,   # 逐通道独立卷积，不混合通道
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
        """时序滑动平均（沿时间维）。

        Args:
            x: (B, T, D) 输入序列
            window: 窗口大小
            padding: 边界填充量（0=valid 无填充；1=replicate 左右各补 1，保持时间对齐）
                     padding=1 时：左补 1、右补 window-2，使滑动窗口中心对齐当前步

        Returns:
            (B, T, D) 平滑后序列
        """
        if padding:
            left = padding
            right = window - 1 - left  # 总 padding = window-1，使输出长度 = 输入长度
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
        """从 BiLSTM 最终隐藏状态中提取前向+后向拼接表示。

        Args:
            hidden_state: BiLSTM 的 h_n，(num_layers*2, B, hidden_dim)

        Returns:
            (B, hidden_dim*2) 前向最终状态与后向最终状态的拼接
        """
        return torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1)

    @staticmethod
    def _statistics(x):
        """提取时序的 lightweight 统计量：level（均值）、recent（末值）、trend（首尾差/步数）。

        Args:
            x: (B, T, D) 输入序列

        Returns:
            (B, D*3) 拼接后的统计量
        """
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
                # Conv1d 需要 (B, D, T) 输入；先 transpose→pad→conv→transpose 回 (B, T, D)
                padded = F.pad(
                    x.transpose(1, 2),       # (B, T, D) → (B, D, T)
                    (1, 1),                   # 时间维左右各补 1（replicate 边界值）
                    mode='replicate',
                )
                medium_x = self.medium_filter(padded).transpose(1, 2)  # 卷积后转回 (B, T, D)
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
            # 短序列回退：时间步不足以做 coarse_window 滑窗时，
            # 用全序列 level（均值）和 trend（首尾差/步数）两个统计量
            # 经 Linear 投影生成粗粒度表示，避免 BiLSTM 在 <2 步输入上崩溃
            level = x.mean(dim=1, keepdim=True)    # (B, 1, D)
            trend = (x[:, -1:, :] - x[:, :1, :]) / max(time_steps - 1, 1)  # 归一化趋势
            coarse_stats = torch.cat([level, trend], dim=-1)  # (B, 1, 2*D)
            coarse_repr = self.coarse_fallback(coarse_stats.squeeze(1))

        multi_scale = torch.cat([fine_repr, medium_repr, coarse_repr], dim=-1)
        return self.fusion(multi_scale)


class AABiLSTM(nn.Module):

    """
    自适应双向 LSTM 信用风险模型（AA-BiLSTM）。

    完整架构（4 层流水线）:
    1. 特征编码层：静态 Embedding（Linear+ReLU+LayerNorm）+
       多尺度时序编码（MultiScaleTemporalEncoder）
    2. AdaptiveFusion：全局统计聚合 + 局部冲击注意力 + 门控残差融合
    3. 序列编码层：AG-ResUnit 堆叠（可扩展至 8 层双向 LSTM）
    4. 分类器：静态跳连 + 多尺度跳连 + 时序池化 → MLP → logits

    关键设计决策：
    - 静态和多尺度表示通过跳连直达分类头，避免仅经门控时序分支间接传播
      造成小样本下的信息瓶颈
    - 支持可选的时序类别 Embedding（Taiwan PAY 状态）和位置 Embedding
    - DynamicFocalLoss 在 Trainer 中启用，不由模型自身管理

    Args:
        static_dim: 静态特征维度
        temporal_dim: 时序特征通道数
        temporal_steps: 时序步数
        hidden_dim: 隐藏维度
        num_layers: AG-ResUnit 堆叠深度
        num_classes: 输出类别数（固定为 2）
        dropout: Dropout 比率
        use_fusion: 是否启用 AdaptiveFusion
        use_ag_resunit: 是否使用 AG-ResUnit（否则为标准 LSTM）
        bidirectional: 是否双向
        use_multiscale: 是否启用多尺度时序编码
        local_window: 局部注意力窗口大小
        use_global_context: 是否启用全局上下文分支
        use_local_attention: 是否启用局部冲击注意力分支
        temporal_categorical_index: 时序特征中类别变量的索引（Taiwan 为 0）
        temporal_category_min: 类别 Embedding 的最小类别值
        temporal_category_max: 类别 Embedding 的最大类别值
        use_step_embedding: 是否添加可学习的时间步 Embedding
        multiscale_mode: MultiScaleTemporalEncoder 模式（'legacy'/'lightweight'）
    """

    def __init__(self, static_dim, temporal_dim, temporal_steps, hidden_dim=128, num_layers=8, num_classes=2,
                 dropout=0.3, use_fusion=True, use_ag_resunit=True,
                 bidirectional=True, use_multiscale=True,
                 local_window=3, use_global_context=True, use_local_attention=True,
                 temporal_categorical_index=None, temporal_category_min=-2,
                 temporal_category_max=8, use_step_embedding=False,
                 multiscale_mode='legacy'):
        super().__init__()
        if use_fusion and not (use_global_context or use_local_attention):
            raise ValueError(
                "AdaptiveFusion requires at least one branch enabled."
            )

        self.hidden_dim = hidden_dim
        self.temporal_steps = temporal_steps
        self.use_fusion = use_fusion
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
            'use_fusion': bool(use_fusion),
            'use_ag_resunit': bool(use_ag_resunit),
            'bidirectional': bool(bidirectional),
            'use_multiscale': bool(use_multiscale),
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

        if use_fusion:
            self.feature_fusion = AdaptiveFusion(
                hidden_dim=hidden_dim,
                window_size=min(local_window, max(temporal_steps, 1)),
                dropout=dropout,
                use_global_context=use_global_context,
                use_local_attention=use_local_attention
            )
        else:
            self.feature_fusion = None
        self.multiscale_context_proj = (
            (
                nn.Identity()
                if multi_scale_dim == hidden_dim
                else nn.Linear(multi_scale_dim, hidden_dim)
            )
            if use_multiscale and use_fusion else None
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

        # 分类头输入 = 时序池化 + 静态嵌入（跳连）+ 多尺度（跳连，可选）
        # 跳连设计确保静态/多尺度信息不依赖门控时序分支间接传播；
        # 这对避免信息瓶颈至关重要。
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
        """将融合后的时序特征通过 AG-ResUnit 或标准 LSTM 编码。

        Args:
            projected_temporal: (B, T, hidden_dim) 融合后时序特征

        Returns:
            (B, T, sequence_dim) 编码后的序列
        """
        if self.use_ag_resunit:
            return self.sequence_encoder(projected_temporal)
        sequence_out, _ = self.sequence_encoder(projected_temporal)
        return sequence_out

    def _pool_sequence(self, sequence_out):
        """序列池化：双向取前向末步 + 后向首步，单向取末步。

        双向模式下，前向末步包含完整的"过去→未来"信息，
        后向首步包含完整的"未来→过去"信息，拼接二者保留全局时序上下文。

        Args:
            sequence_out: (B, T, sequence_dim) 编码后的序列

        Returns:
            (B, sequence_dim) 池化后的时序表示
        """
        if self.bidirectional:
            forward_final = sequence_out[:, -1, :self.hidden_dim]
            backward_final = sequence_out[:, 0, self.hidden_dim:]
            return torch.cat([forward_final, backward_final], dim=-1)
        return sequence_out[:, -1, :]

    def forward(self, static_feat, temporal_feat, return_attention=False):
        """
        前向传播：完整 4 层流水线。

        流程：static_embedding → temporal_projection → multiscale_encoding →
              adaptive_fusion → sequence_encoding → pooling →
              classifier（跳连 static + multiscale）

        Args:
            static_feat: (B, static_dim) 静态特征
            temporal_feat: (B, T, temporal_dim) 时序特征
            return_attention: 是否返回 AdaptiveFusion 的诊断信息

        Returns:
            若 return_attention=False: (B, num_classes) logits
            若 return_attention=True: (logits, fusion_diagnostics)
        """
        static_emb = self.static_embedding(static_feat)
        # 类别通道乘以 0（mask），仅数值通道参与线性投影；
        # 类别通道单独通过 Embedding 层处理，避免被当做连续值。
        temporal_numeric = temporal_feat * self.temporal_numeric_mask.view(
            1,
            1,
            -1,
        )
        projected_temporal = self.temporal_projection(temporal_numeric)
        if self.temporal_category_embedding is not None:
            # 提取类别通道值（如 Taiwan 的 PAY 状态，-2~8 的整数）
            category_value = temporal_feat[
                :,
                :,
                self.temporal_categorical_index,
            ]
            # round 处理 float32 精度导致的微小偏移（如 1.999 → 2）
            category_index = torch.round(category_value).long()
            # 越界类别值（如缺失值）映射到 unknown 索引
            valid_category = (
                (category_index >= self.temporal_category_min)
                & (category_index <= self.temporal_category_max)
            )
            category_index = category_index - self.temporal_category_min  # 偏移到 0 起始
            category_index = torch.where(
                valid_category,
                category_index,
                torch.full_like(
                    category_index,
                    self.temporal_category_unknown,  # 超出 [-2, 8] 的值统一映射到此
                ),
            )
            # 类别 Embedding 与数值投影相加（残差式注入语义信息）
            projected_temporal = (
                projected_temporal
                + self.temporal_category_embedding(category_index)
            )
        if self.step_embedding is not None:
            # 可学习的位置编码：告知模型当前是第几个时间步
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
        if self.use_fusion:
            multi_scale_context = (
                self.multiscale_context_proj(multi_scale_repr)
                if multi_scale_repr is not None else None
            )
            sequence_input, fusion_diagnostics = self.feature_fusion(
                static_emb,
                projected_temporal,
                multi_scale_context
            )

        sequence_out = self._encode_sequence(sequence_input)
        temporal_repr = self._pool_sequence(sequence_out)

        fusion_parts = [temporal_repr, static_emb]
        if multi_scale_repr is not None:
            fusion_parts.append(multi_scale_repr)

        logits = self.classifier(torch.cat(fusion_parts, dim=-1))

        if return_attention:
            return logits, fusion_diagnostics
        return logits


# ==============================
# 3. 动态焦点损失函数
# ==============================

class DynamicFocalLoss(nn.Module):
    """
    动态焦点损失函数（Dynamic Focal Loss）。

    核心机制：
    - gamma 按 epoch 调度从 gamma_base 增长到 gamma_max，
      早期接近加权交叉熵（容易收敛），后期增大难样本权重（聚焦难例）
    - alpha 类别权重在初始化时从训练集全局频率计算，避免 mini-batch 随机抖动
    - 调度函数：gamma = gamma_base + (gamma_max - gamma_base) * (epoch/num_epoch)^schedule_power

    公式：
        FL = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha_pos: 正类（违约）的全局权重
        alpha_neg: 负类（正常）的全局权重
        gamma_base: gamma 起始值
        gamma_max: gamma 终止值
        num_epoch: 总 epoch 数（用于 gamma 调度）
        schedule_power: 调度曲线的幂指数（<1 时 gamma 前期增长快，>1 时后期增长快）
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
        计算当前 epoch 的动态 focal loss。

        使用 log_softmax（而非 softmax + log）避免数值下溢，
        并 clamp p_t > eps 防止 focal_weight 的 pow 操作出现无效梯度。

        Args:
            inputs: (N, 2) 模型输出 logits
            targets: (N,) 真实标签（0 或 1）

        Returns:
            scalar: 该批次的平均 loss
        """
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("DynamicFocalLoss expects binary logits with shape [N, 2].")
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("targets must have shape [N].")

        # 使用 log_softmax（而非 softmax+log）避免 softmax 下溢；
        # log_p_t 直接从 log_softmax gather 得到，数值更稳定。
        log_probs = F.log_softmax(inputs.float(), dim=-1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()  # 正确类别概率，用于计算 focal 因子

        # gamma 调度：power < 1 时前期增长快（早期即关注难样本），
        #          power > 1 时后期增长快（先易后难）。
        progress = min(
            ((self.current_epoch + 1) / self.num_epoch) ** self.schedule_power,
            1.0,
        )
        gamma = self.gamma_base + (
            self.gamma_max - self.gamma_base
        ) * progress

        # alpha 由训练集全局频率确定（初始化为固定值），不随 mini-batch 变化
        alpha_t = torch.where(
            targets == 1,
            log_p_t.new_tensor(self.alpha_pos),
            log_p_t.new_tensor(self.alpha_neg),
        )
        # focal_weight = (1-p_t)^γ：p_t 越大（越容易），权重越小
        # clamp_min(eps) 防止 p_t=1 时 pow 操作的梯度退化
        focal_base = (1.0 - p_t).clamp_min(
            torch.finfo(p_t.dtype).eps
        )
        focal_weight = focal_base.pow(gamma)
        # FL = -α_t * (1-p_t)^γ * log(p_t)
        loss = -alpha_t * focal_weight * log_p_t

        return loss.mean()


# ==============================
# 4. 训练与评估
# ==============================

class EarlyStopping:
    """早停机制：验证指标连续 patience 轮不提升（提升 < min_delta）则停止训练。

    Args:
        patience: 容忍轮数
        min_delta: 最小提升阈值
    """

    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_auc):
        if not np.isfinite(val_auc):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return
        if self.best_score is None:
            self.best_score = val_auc
            self.counter = 0
        elif val_auc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.counter = 0


def specificity_score(y_true, y_pred):
    """计算 Specificity（真负率）= TN / (TN + FP)。

    sklearn 未直接提供此指标，从 confusion matrix 手动计算。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


class Trainer:
    """
    AA-BiLSTM 模型训练器。

    集成了完整的训练流程：
    - 混合精度训练（AMP）+ 梯度裁剪
    - AdamW 优化器 + 余弦退火学习率调度
    - EMA（指数移动平均）权重平滑
    - Platt 概率校准（交叉拟合，含 OOF 评估）
    - 基于 Wilson 下界的阈值搜索（保证敏感性下限的同时最大化 Accuracy）
    - 早停机制
    - 多目标 checkpoint 选择（AUC / AUC-PR / hybrid / accuracy）

    关键设计：
    - 阈值搜索使用原始 margin 排序（不受校准影响），校准映射保持单调性
    - EMA 权重在训练结束后与最佳 checkpoint 竞争，取验证分数更高者
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 calibration_loader=None,
                 num_epochs=100, lr=1e-3, weight_decay=1e-4,
                 use_dynamic_focal=True, use_class_weight=False,
                 use_early_stopping=False, threshold_min_sensitivity=0.40,
                 selection_metric='auc_pr',
                 threshold_objective='hybrid',
                 class_balance_power=0.5,
                 focal_gamma_max=2.0,
                 ema_decay=0.995,
                 calibrate_probabilities=True,
                 threshold_confidence=0.80):
        if num_epochs < 1:
            raise ValueError("num_epochs must be at least 1.")
        if selection_metric not in {'auc', 'auc_pr', 'hybrid', 'accuracy'}:
            raise ValueError(
                "selection_metric must be 'auc', 'auc_pr', 'hybrid' or "
                "'accuracy'."
            )
        if threshold_objective not in {
            'hybrid', 'f1', 'balanced_accuracy', 'accuracy'
        }:
            raise ValueError(
                "threshold_objective must be 'hybrid', 'f1', "
                "'balanced_accuracy' or 'accuracy'."
            )
        if not 0.0 <= class_balance_power <= 1.0:
            raise ValueError("class_balance_power must be in [0, 1].")
        if focal_gamma_max < 0.0:
            raise ValueError("focal_gamma_max must be non-negative.")
        if ema_decay is not None and not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1) or None.")
        if not 0.5 <= threshold_confidence < 1.0:
            raise ValueError(
                "threshold_confidence must be in [0.5, 1)."
            )
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.calibration_loader = calibration_loader or val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.decision_threshold = 0.5
        self.threshold_min_sensitivity = threshold_min_sensitivity
        self.use_early_stopping = use_early_stopping
        self.selection_metric = selection_metric
        self.threshold_objective = threshold_objective
        self.class_balance_power = float(class_balance_power)
        self.focal_gamma_max = float(focal_gamma_max)
        self.ema_decay = ema_decay
        self.calibrate_probabilities = bool(calibrate_probabilities)
        self.temperature = 1.0
        self.calibration_scale = 1.0
        self.calibration_bias = 0.0
        self.threshold_confidence = float(threshold_confidence)
        self.amp_enabled = device.type == 'cuda'

        # 优化器: AdamW + 余弦退火
        optimizer_kwargs = {
            'lr': lr,
            'weight_decay': weight_decay,
        }
        if device.type == 'cuda':
            optimizer_kwargs['fused'] = True
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop('fused', None)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
            )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # 损失函数
        if use_dynamic_focal:
            counts = self._compute_class_counts(train_loader)
            # class_balance_power=0.5：次幂根比完全逆频率温和。
            #   完全逆频率（power=1.0）→ 极端少数类 alpha≈0.9+
            #   平方根（power=0.5）→ 少数类 alpha≈0.7~0.8
            # 温和的 alpha 减少假阳性（False Positive），再由验证集阈值保障敏感性。
            raw_weights = np.power(
                1.0 / np.maximum(counts, 1.0),
                self.class_balance_power,
            )
            raw_weights /= raw_weights.sum()
            alpha_neg = float(raw_weights[0])
            alpha_pos = float(raw_weights[1])
            self.criterion = DynamicFocalLoss(
                num_epoch=num_epochs,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                gamma_base=0.0,
                gamma_max=self.focal_gamma_max,
                schedule_power=0.5,
            )
            print(
                f"Dynamic focal loss: alpha_neg={alpha_neg:.3f}, "
                f"alpha_pos={alpha_pos:.3f}, "
                f"gamma=0.0->{self.focal_gamma_max:.1f}"
            )
        else:
            # 标准交叉熵更适合优化整体 Accuracy；如需强调少数类可打开 use_class_weight
            if use_class_weight:
                weights = self._compute_class_weights(train_loader)
                self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.early_stopping = EarlyStopping(patience=10)
        self.history = {
            'train_loss': [],
            'val_auc': [],
            'val_auc_pr': [],
            'val_f1': [],
            'val_accuracy': [],
            'val_sensitivity': [],
            'val_balanced_accuracy': [],
            'learning_rate': [],
        }
        self.ema_state = (
            {
                name: value.detach().clone()
                for name, value in self.model.state_dict().items()
            }
            if self.ema_decay is not None else None
        )
        try:
            self.scaler = torch.amp.GradScaler(
                'cuda',
                enabled=self.amp_enabled,
            )
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.amp_enabled,
            )

    @staticmethod
    def _compute_class_counts(loader):
        """从训练数据集读取全局类别数。

        CreditDataset 始终暴露 .labels 属性，因此直接读取；
        遍历 DataLoader 的分支为通用数据集兼容路径（CreditDataset 下不可达）。
        """
        labels = loader.dataset.labels
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
        if len(counts) != 2 or np.any(counts == 0):
            raise ValueError(
                f"Training data must contain both binary classes; got {counts.tolist()}."
            )
        return counts.astype(np.float64)

    @classmethod
    def _compute_class_weights(cls, loader):
        """计算全局 inverse-frequency 类别权重。

        公式：w_i = (1/count_i) / sum(1/counts) * 2
        *2 使得二分类任务中权重均值为 1（即 loss 量级与不加权一致），
        避免权重整体偏大导致学习率需要重新调节。
        """
        counts = cls._compute_class_counts(loader)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 2
        return torch.FloatTensor(weights)

    def _update_ema(self):
        """更新模型权重的指数移动平均（EMA）。

        公式：shadow = decay * shadow + (1 - decay) * current_weight
        仅对浮点参数做平滑，整数型参数（如 batch norm 计数）直接复制。
        """
        if self.ema_state is None:
            return
        with torch.no_grad():
            for name, value in self.model.state_dict().items():
                shadow = self.ema_state[name]
                if torch.is_floating_point(value):
                    shadow.mul_(self.ema_decay).add_(
                        value.detach(),
                        alpha=1.0 - self.ema_decay,
                    )
                else:
                    shadow.copy_(value)

    def _selection_score(self, metrics):
        """根据 selection_metric 计算验证集选择分数。

        'hybrid' 模式：0.55*AUC-PR + 0.20*BalancedAcc + 0.15*Acc + 0.10*Sensitivity
        以排序能力为主，同时轻度偏向 Accuracy 与敏感性更均衡的 epoch。
        """
        if self.selection_metric == 'hybrid':
            pass  # fallthrough below
        elif self.selection_metric == 'accuracy':
            return float(metrics['accuracy'])
        else:
            return float(metrics[self.selection_metric])
        # 排序能力为主，同时轻度偏向 Accuracy 与敏感性更均衡的 epoch。
        values = (
            metrics['auc_pr'],
            metrics['balanced_accuracy'],
            metrics['accuracy'],
            metrics['sensitivity'],
        )
        if not all(np.isfinite(value) for value in values):
            return np.nan
        return float(
            0.55 * metrics['auc_pr']
            + 0.20 * metrics['balanced_accuracy']
            + 0.15 * metrics['accuracy']
            + 0.10 * metrics['sensitivity']
        )

    def _collect_logits(self, loader):
        """遍历 DataLoader 收集所有样本的 logits 和标签。

        Args:
            loader: 数据加载器

        Returns:
            (logits, labels): logits 形状 (N, 2)，labels 形状 (N,)
        """
        self.model.eval()
        logits_batches = []
        label_batches = []
        with torch.inference_mode():
            for batch in loader:
                static = batch['static'].to(device, non_blocking=True)
                temporal = batch['temporal'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                logits = self.model(static, temporal)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        "Non-finite model outputs detected during calibration."
                    )
                logits_batches.append(logits.float().cpu())
                label_batches.append(labels.cpu())
        if not logits_batches:
            raise RuntimeError("Evaluation DataLoader produced no samples.")
        return (
            torch.cat(logits_batches, dim=0),
            torch.cat(label_batches, dim=0),
        )

    @staticmethod
    def _fit_platt_parameters(margins, labels):
        """用 LogisticRegression 在 margin 上做 Platt 校准参数估计。

        Platt 公式: P(y=1|x) = 1 / (1 + exp(-(scale * margin + bias)))
        其中 scale = coef_[0,0], bias = intercept_[0]。
        C=10.0 提供适度的 L2 正则，防止 scale 在小样本上发散。
        scale <= 0 说明校准方向与标签相反 → 退回 identity（1.0, 0.0）。
        """
        margins = np.asarray(margins, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64)
        if len(np.unique(labels)) != 2:
            return 1.0, 0.0
        calibrator = LogisticRegression(
            C=10.0,            # 适度的 L2 正则化，防止小数据过拟合
            solver='lbfgs',
            max_iter=2000,
            random_state=42,
        )
        calibrator.fit(margins, labels)
        scale = float(calibrator.coef_[0, 0])
        bias = float(calibrator.intercept_[0])
        if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0.0:
            return 1.0, 0.0  # 校准失败 → 退回恒等变换
        return scale, bias

    @staticmethod
    def _apply_platt(margins, scale, bias):
        """对 margin 应用 Platt 校准：calibrated_prob = sigmoid(scale * margin + bias)。

        clip 到 [-40, 40]：sigmoid(-40) ≈ 4e-18, sigmoid(40) ≈ 1-4e-18，
        足以覆盖 float64 有效精度范围，同时防止 overflow。
        """
        calibrated_logit = np.clip(
            float(scale) * np.asarray(margins, dtype=np.float64)
            + float(bias),
            -40.0,
            40.0,
        )
        return 1.0 / (1.0 + np.exp(-calibrated_logit))

    def fit_platt_scaling(self):
        """
        在专用的校准集（calibration split）上拟合交叉验证 Platt 概率校准。

        流程：
        1. 对校准集计算 margin = logit_pos - logit_neg
        2. 做 StratifiedKFold 交叉拟合（最多 5-fold），得到 OOF（out-of-fold）概率
        3. 再用全量校准集拟合最终的 scale/bias
        4. 比较 NLL（校准前 vs 校准后 vs OOF），若校准后 NLL 反而恶化则退回 identity

        重要：阈值搜索在原始 margin 排序空间进行，然后再通过校准参数映射到
        校准后概率空间，这样确保校准不会扰乱排序质量。
        """
        logits, label_tensor = self._collect_logits(
            self.calibration_loader
        )
        labels = label_tensor.numpy().astype(np.int64)
        # margin = 正类 logit - 负类 logit，作为 Platt 校准的输入
        margins = (logits[:, 1] - logits[:, 0]).numpy()
        raw_probabilities = self._apply_platt(margins, 1.0, 0.0)  # 校准前原始概率
        class_counts = np.bincount(labels, minlength=2)
        # fold 数不超过少数类样本数
        n_splits = int(min(5, class_counts.min()))

        if n_splits >= 2:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )
            oof_probabilities = np.empty(len(labels), dtype=np.float64)
            for fit_idx, holdout_idx in splitter.split(margins, labels):
                fold_scale, fold_bias = self._fit_platt_parameters(
                    margins[fit_idx],
                    labels[fit_idx],
                )
                oof_probabilities[holdout_idx] = self._apply_platt(
                    margins[holdout_idx],
                    fold_scale,
                    fold_bias,
                )
        else:
            oof_probabilities = raw_probabilities.copy()

        scale, bias = self._fit_platt_parameters(margins, labels)
        fitted_probabilities = self._apply_platt(
            margins,
            scale,
            bias,
        )
        nll_before = log_loss(
            labels,
            np.clip(raw_probabilities, 1e-7, 1.0 - 1e-7),
            labels=[0, 1],
        )
        nll_after = log_loss(
            labels,
            np.clip(fitted_probabilities, 1e-7, 1.0 - 1e-7),
            labels=[0, 1],
        )
        oof_nll = log_loss(
            labels,
            np.clip(oof_probabilities, 1e-7, 1.0 - 1e-7),
            labels=[0, 1],
        )
        # 防退化保护：
        # - 全量校准 NLL 恶化 → 退回 identity（scale=1, bias=0）
        # - OOF NLL 恶化超过 0.02 → OOF 退回到原始概率
        # （OOF 容差更大，因为交叉验证本身有额外方差）
        if nll_after > nll_before + 1e-8:
            scale, bias = 1.0, 0.0
            nll_after = nll_before
        if oof_nll > nll_before + 0.02:
            oof_probabilities = raw_probabilities
            oof_nll = nll_before

        return scale, bias, labels, raw_probabilities, {
            'method': 'cross_fitted_platt',
            'folds': n_splits,
            'nll_before': float(nll_before),
            'nll_after': float(nll_after),
            'oof_nll': float(oof_nll),
            'scale': float(scale),
            'bias': float(bias),
        }

    def train_epoch(self, epoch):
        """训练一个 epoch：混合精度前向 → 反向 → 梯度裁剪 → 优化器步进 → EMA 更新。

        Args:
            epoch: 当前 epoch 编号（用于 DynamicFocalLoss 的 gamma 调度）

        Returns:
            float: 该 epoch 的平均训练 loss
        """
        self.model.train()
        if isinstance(self.criterion, DynamicFocalLoss):
            self.criterion.set_epoch(epoch)

        total_loss = 0.0
        total_samples = 0
        nan_loss_streak = 0
        MAX_NAN_STREAK = 5  # 连续 NaN batch 数超过此值才视为真正的数值问题

        for batch in self.train_loader:
            static = batch['static'].to(device, non_blocking=True)
            temporal = batch['temporal'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # set_to_none=True：将梯度设为 None 而非 zeros()，
            # 减少显存占用，且避免 AdamW 对零梯度累加不必要的动量更新。
            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播（AMP）：
            # CUDA 上自动将大部分运算转为 float16，节省显存并加速；
            # 损失计算保持在 float32 避免下溢。
            try:
                with torch.autocast(
                    device_type=device.type,
                    enabled=self.amp_enabled,
                ):
                    outputs = self.model(static, temporal)
                    loss = self.criterion(outputs, labels)
            except RuntimeError:
                # AMP autocast 在某些 op 上可能失败（罕见），回退到 FP32
                outputs = self.model(static, temporal)
                loss = self.criterion(outputs, labels)

            if not torch.isfinite(loss):
                nan_loss_streak += 1
                if nan_loss_streak >= MAX_NAN_STREAK:
                    raise FloatingPointError(
                        f"Non-finite training loss in {MAX_NAN_STREAK} "
                        "consecutive batches. Check input scaling and "
                        "the model's numerical operations."
                    )
                # 跳过此 batch：未做 backward，无需 scaler 处理
                continue
            nan_loss_streak = 0  # 正常 batch 重置计数器

            # 混合精度反向传播：scaler 将 loss 放大防止 float16 梯度下溢
            self.scaler.scale(loss).backward()

            # 梯度裁剪前必须 unscale，否则裁剪的是放大的梯度
            self.scaler.unscale_(self.optimizer)
            # max_norm=1.0：将梯度 L2 范数限制在 1.0 以内，
            # 选择 1.0 而非更小的值是因为 AMP 已提供一定数值稳定性。
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False,  # 非有限梯度不抛异常，由后续检查处理
            )
            if not torch.isfinite(grad_norm):
                # 非有限梯度：跳过参数更新；不清除 scaler 内部状态，
                # 下一正常 batch 的 step() + update() 会自然调整 scale
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()

            batch_samples = labels.shape[0]
            # 加权平均：每个 batch 的 loss 按其样本数加权，
            # 确保最后一个不完整 batch 不会过度影响总 loss
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        if total_samples == 0:
            raise RuntimeError("Training DataLoader produced no samples.")
        avg_loss = total_loss / total_samples
        return avg_loss

    def evaluate(
        self,
        loader,
        threshold=None,
        temperature=None,
        calibration_scale=None,
        calibration_bias=None,
    ):
        """评估模型，返回排序、分类和概率校准指标的完整集合。

        计算指标包括：Accuracy, AUC-ROC, AUC-PR, F1, F2, Precision,
        Sensitivity(Recall), Specificity, Balanced Accuracy, Brier Score。

        Args:
            loader: 评估数据加载器
            threshold: 决策阈值（默认 self.decision_threshold）
            temperature: 温度缩放参数（与 calibration_scale 互斥）
            calibration_scale: Platt 校准的 scale 参数
            calibration_bias: Platt 校准的 bias 参数

        Returns:
            dict: 含所有指标及原始 predictions/probabilities/labels 的字典
        """
        self.model.eval()
        if threshold is None:
            threshold = self.decision_threshold
        if calibration_scale is None:
            calibration_scale = self.calibration_scale
        if calibration_bias is None:
            calibration_bias = self.calibration_bias
        if temperature is not None:
            temperature = float(np.clip(temperature, 0.05, 20.0))
            calibration_scale = 1.0 / temperature
            calibration_bias = 0.0
        calibration_scale = float(calibration_scale)
        calibration_bias = float(calibration_bias)
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.inference_mode():
            for batch in loader:
                static = batch['static'].to(device, non_blocking=True)
                temporal = batch['temporal'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                outputs = self.model(static, temporal)
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        "Non-finite model outputs detected during evaluation."
                    )
                # margin = logit_pos - logit_neg：正类相对于负类的 log-odds。
                # 通过 Platt 校准后再做 sigmoid 得到校准后违约概率。
                margin = outputs[:, 1].float() - outputs[:, 0].float()
                positive_probability = torch.sigmoid(
                    calibration_scale * margin + calibration_bias
                )
                preds = (positive_probability >= threshold).long()

                all_preds.append(preds.cpu().numpy())
                all_probs.append(positive_probability.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_labels:
            raise RuntimeError("Evaluation DataLoader produced no samples.")
        flat_labels = np.concatenate(all_labels)
        flat_preds = np.concatenate(all_preds)
        flat_probs = np.concatenate(all_probs)
        clipped_probs = np.clip(flat_probs, 1e-7, 1.0 - 1e-7)

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
        f2 = fbeta_score(
            flat_labels,
            flat_preds,
            beta=2.0,
            zero_division=0,
        )
        precision = precision_score(
            flat_labels,
            flat_preds,
            zero_division=0,
        )
        sensitivity = recall_score(
            flat_labels,
            flat_preds,
            zero_division=0,
        )
        specificity = specificity_score(flat_labels, flat_preds)
        balanced_accuracy = balanced_accuracy_score(
            flat_labels,
            flat_preds,
        )
        brier = brier_score_loss(flat_labels, clipped_probs)

        return {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'auc_pr': float(auc_pr),
            'f1': float(f1),
            'f2': float(f2),
            'precision': float(precision),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'balanced_accuracy': float(balanced_accuracy),
            'brier_score': float(brier),
            'temperature': float(
                1.0 / max(calibration_scale, 1e-7)
            ),
            'calibration_scale': calibration_scale,
            'calibration_bias': calibration_bias,
            'threshold': float(threshold),
            'predictions': flat_preds,
            'probabilities': flat_probs,
            'labels': flat_labels,
        }

    @staticmethod
    def _wilson_lower_bound(successes, total, confidence=0.80):
        """Wilson 分数区间下界：对二项比例 p 的单侧置信下界。

        公式（Wilson score interval lower bound）：
          lower = (p + z²/(2n) - z*sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)

        相比 Wald 区间（p ± z*sqrt(p(1-p)/n)），Wilson 区间在
        p 接近 0 或 1 时不会越界，小样本下更保守可靠。
        """
        if total <= 0:
            return 0.0
        proportion = successes / total
        z_value = NormalDist().inv_cdf(confidence)  # 单侧 z 值
        denominator = 1.0 + z_value ** 2 / total
        centre = proportion + z_value ** 2 / (2.0 * total)
        radius = z_value * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value ** 2 / (4.0 * total ** 2)
        )
        return float((centre - radius) / denominator)

    def find_best_threshold(
        self,
        min_sensitivity=None,
        labels=None,
        probabilities=None,
    ):
        """在验证/校准集上搜索满足敏感性下限的最优决策阈值。

        搜索策略：
        1. 遍历所有候选阈值（预测概率的相邻中点）
        2. 使用 Wilson 置信下界评估敏感性下限是否满足要求
           （小样本（正类 < 50）回退到点估计）
        3. 在满足约束的候选中，按 threshold_objective 选择最优：
           - 'hybrid': 0.75*Acc + 0.10*BalAcc + 0.10*F1 + 0.025*Sens + 0.025*Spec
           - 'f1': 直接取最高 F1
           - 'balanced_accuracy': 直接取最高 Balanced Accuracy
           - 'accuracy': 直接取最高 Accuracy

        Args:
            min_sensitivity: 敏感性下限；None 时使用 self.threshold_min_sensitivity
            labels: 真实标签，None 时从 calibration_loader 读取
            probabilities: 预测概率，None 时从 calibration_loader 读取

        Returns:
            (threshold, metrics_dict): 最优阈值和对应的指标字典
        """
        if min_sensitivity is None:
            min_sensitivity = self.threshold_min_sensitivity
        if labels is None or probabilities is None:
            operating = self.evaluate(
                self.calibration_loader,
                threshold=0.5,
            )
            labels = operating['labels']
            probabilities = operating['probabilities']
        labels = np.asarray(labels, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        scored = []
        point_constrained = []

        # 候选阈值生成：取相邻唯一概率值的中间点，避免在概率分布的
        # 稀疏区域产生大量无效候选。候选数 > 5000 时用量化降采样。
        unique_probabilities = np.unique(probabilities)
        if len(unique_probabilities) > 5000:
            unique_probabilities = np.quantile(
                unique_probabilities,
                np.linspace(0.0, 1.0, 5000),
            )
        if len(unique_probabilities) == 1:
            candidates = np.asarray([0.5], dtype=float)  # 所有样本概率相同，只能用默认值
        else:
            candidates = (
                unique_probabilities[:-1] + unique_probabilities[1:]
            ) / 2.0
            # 边界扩展：确保候选覆盖 [0, 1] 区间两端
            candidates = np.concatenate([
                [max(0.0, unique_probabilities[0] - 1e-7)],
                candidates,
                [min(1.0, unique_probabilities[-1] + 1e-7)],
            ])

        for threshold in candidates:
            predictions = (probabilities >= threshold).astype(int)
            sensitivity = recall_score(
                labels,
                predictions,
                zero_division=0,
            )
            true_positive = int(
                np.sum((labels == 1) & (predictions == 1))
            )
            positive_count = int(np.sum(labels == 1))
            sensitivity_lower = self._wilson_lower_bound(
                true_positive,
                positive_count,
                confidence=self.threshold_confidence,
            )
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, zero_division=0)
            balanced_accuracy = balanced_accuracy_score(labels, predictions)
            specificity = specificity_score(labels, predictions)
            if self.threshold_objective == 'f1':
                objective = f1
            elif self.threshold_objective == 'balanced_accuracy':
                objective = balanced_accuracy
            elif self.threshold_objective == 'accuracy':
                objective = accuracy
            else:
                # 在敏感性硬约束内以 Accuracy 为主，再奖励类间平衡与 F1。
                objective = (
                        0.75 * accuracy
                        + 0.10 * balanced_accuracy
                        + 0.10 * f1
                        + 0.025 * sensitivity
                        + 0.025 * specificity
                )
            candidate = (
                objective,
                balanced_accuracy,
                accuracy,
                sensitivity,
                sensitivity_lower,
                f1,
                -abs(threshold - 0.5),
                threshold
            )
            if sensitivity + 1e-12 >= min_sensitivity:
                point_constrained.append(candidate)
            if sensitivity_lower + 1e-12 >= min_sensitivity:
                scored.append(candidate)

        if int(np.sum(labels == 1)) < 50:
            scored = point_constrained
            constraint_type = 'point_estimate_small_sample'
        else:
            constraint_type = 'wilson_lower_bound'
            if not scored:
                scored = point_constrained
                constraint_type = 'point_estimate_fallback'

        if not scored:
            fallback_predictions = (probabilities >= 0.5).astype(int)
            return 0.5, {
                'accuracy': float(
                    accuracy_score(labels, fallback_predictions)
                ),
                'f1': float(f1_score(
                    labels,
                    fallback_predictions,
                    zero_division=0,
                )),
                'balanced_accuracy': float(
                    balanced_accuracy_score(
                        labels,
                        fallback_predictions,
                    )
                ),
                'sensitivity': float(recall_score(
                    labels,
                    fallback_predictions,
                    zero_division=0,
                )),
                'specificity': specificity_score(
                    labels,
                    fallback_predictions,
                ),
                'sensitivity_lower_bound': 0.0,
                'constraint_type': 'unconstrained_fallback',
                'objective': np.nan,
            }

        # 按 objective 选最优；相同 objective 时按阈值接近 0.5 程度打破平局
        (
            objective,
            balanced_accuracy,
            accuracy,
            sensitivity,
            sensitivity_lower,
            f1,
            _,
            threshold,
        ) = max(scored, key=lambda x: (x[0], x[6]))  # x[0]=objective, x[6]=-abs(th-0.5)
        return float(threshold), {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'balanced_accuracy': float(balanced_accuracy),
            'sensitivity': float(sensitivity),
            'sensitivity_lower_bound': float(sensitivity_lower),
            'constraint_type': constraint_type,
            'specificity': float(
                specificity_score(
                    labels,
                    (probabilities >= threshold).astype(int),
                )
            ),
            'objective': float(objective),
        }

    def train(self):
        """完整训练流程"""
        best_score = -np.inf
        best_model_state = None
        best_epoch = None

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_auc_pr'].append(val_metrics['auc_pr'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_sensitivity'].append(
                val_metrics['sensitivity']
            )
            self.history['val_balanced_accuracy'].append(
                val_metrics['balanced_accuracy']
            )
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # 学习率调整
            self.scheduler.step()

            # 早停检查
            selection_score = self._selection_score(val_metrics)
            if self.use_early_stopping:
                self.early_stopping(selection_score)

            if np.isfinite(selection_score) and selection_score > best_score:
                best_score = selection_score
                best_epoch = epoch + 1
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

        if best_model_state is None:
            raise RuntimeError(
                f"No finite validation {self.selection_metric} was observed."
            )

        best_source = 'checkpoint'
        # EMA 权重与最佳 checkpoint 竞争：加载 EMA 权重重新评估，
        # 若 EMA 的验证分数更高，则用 EMA 权重替代 checkpoint。
        if self.ema_state is not None:
            self.model.load_state_dict(self.ema_state)
            ema_metrics = self.evaluate(
                self.val_loader,
                threshold=0.5,
                temperature=1.0,  # 用原始概率（无温度缩放）评估
            )
            ema_score = self._selection_score(ema_metrics)
            if np.isfinite(ema_score) and ema_score > best_score:
                best_model_state = copy.deepcopy(self.ema_state)
                best_score = ema_score
                best_source = 'ema'

        self.model.load_state_dict(best_model_state)
        calibration_metrics = {
            'method': 'identity',
            'nll_before': np.nan,
            'nll_after': np.nan,
            'oof_nll': np.nan,
        }
        if self.calibrate_probabilities:
            (
                self.calibration_scale,
                self.calibration_bias,
                operating_labels,
                raw_operating_probabilities,
                calibration_metrics,
            ) = self.fit_platt_scaling()
        else:
            operating_logits, operating_label_tensor = (
                self._collect_logits(self.calibration_loader)
            )
            operating_labels = operating_label_tensor.numpy()
            # 不校准时 margin 已经已知，无需后续从概率反推
            operating_margin = (
                operating_logits[:, 1] - operating_logits[:, 0]
            ).numpy()
            raw_operating_probabilities = self._apply_platt(
                operating_margin,
                1.0,
                0.0,
            )
            self.calibration_scale = 1.0
            self.calibration_bias = 0.0
            operating_margin_from_probs = operating_margin
        self.temperature = 1.0 / max(self.calibration_scale, 1e-7)

        # 阈值搜索在原始 margin 概率空间进行（排序不变），
        # 然后将原始阈值通过 Platt 映射到校准后概率空间。
        # 路径：raw_threshold → logit（log-odds 变换）→ calibrated_prob（Platt sigmoid）
        raw_decision_threshold, validation_threshold_metrics = self.find_best_threshold(
            min_sensitivity=self.threshold_min_sensitivity,
            labels=operating_labels,
            probabilities=raw_operating_probabilities,
        )
        clipped_raw_threshold = float(np.clip(
            raw_decision_threshold,
            1e-7,
            1.0 - 1e-7,
        ))
        # 将概率阈值转为 logit 空间：log(p/(1-p))
        raw_threshold_logit = np.log(
            clipped_raw_threshold / (1.0 - clipped_raw_threshold)
        )
        # 用 Platt 参数映射 logit → 校准后概率
        self.decision_threshold = float(self._apply_platt(
            np.asarray([raw_threshold_logit]),
            self.calibration_scale,
            self.calibration_bias,
        )[0])
        # 校准路径：需要从原始概率反推 margin（logit 空间）用于 Platt 校准
        # 非校准路径：margin 已在上方直接获取，无需重复计算
        if self.calibrate_probabilities:
            operating_margin_from_probs = np.log(
                np.clip(raw_operating_probabilities, 1e-7, 1.0 - 1e-7)
                / np.clip(
                    1.0 - raw_operating_probabilities,
                    1e-7,
                    1.0,
                )
            )
        calibrated_operating_probabilities = self._apply_platt(
            operating_margin_from_probs,
            self.calibration_scale,
            self.calibration_bias,
        )
        # 熵阈值 = 验证集预测熵的 95 分位数
        # 熵高 → 模型不确定 → 可在部署时触发人工复核或保守处理
        val_probability = np.clip(
            calibrated_operating_probabilities,
            1e-12,
            1.0 - 1e-12,
        )
        validation_entropy = -(
            val_probability * np.log(val_probability)
            + (1.0 - val_probability) * np.log(1.0 - val_probability)
        )
        self.entropy_threshold = float(
            np.quantile(validation_entropy, 0.95)
        )
        self.model.decision_threshold = float(self.decision_threshold)
        self.model.temperature = float(self.temperature)
        self.model.calibration_scale = float(self.calibration_scale)
        self.model.calibration_bias = float(self.calibration_bias)
        self.model.entropy_threshold = float(self.entropy_threshold)
        print(
            f"Selected dedicated operating threshold: "
            f"{self.decision_threshold:.3f} | "
            f"Accuracy: {validation_threshold_metrics['accuracy']:.4f} | "
            f"Sensitivity: {validation_threshold_metrics['sensitivity']:.4f} | "
            f"lower bound: "
            f"{validation_threshold_metrics['sensitivity_lower_bound']:.4f}"
        )
        print(
            f"Calibration: {calibration_metrics['method']} | "
            f"scale={self.calibration_scale:.3f}, "
            f"bias={self.calibration_bias:.3f} | "
            f"operating NLL {calibration_metrics['nll_before']:.4f}"
            f"->{calibration_metrics['nll_after']:.4f}"
        )
        print(
            f"Restored {best_source} from epoch {best_epoch} with validation "
            f"{self.selection_metric}={best_score:.4f}"
        )

        # 用验证集确定的阈值在测试集上做一次最终评估
        test_metrics = self.evaluate(self.test_loader, threshold=self.decision_threshold)
        test_metrics['threshold_accuracy'] = validation_threshold_metrics['accuracy']
        test_metrics['threshold_sensitivity'] = validation_threshold_metrics['sensitivity']
        test_metrics['threshold_specificity'] = validation_threshold_metrics['specificity']
        test_metrics['threshold_objective'] = validation_threshold_metrics['objective']
        test_metrics['threshold_sensitivity_lower_bound'] = validation_threshold_metrics[
            'sensitivity_lower_bound'
        ]
        test_metrics['raw_margin_probability_threshold'] = float(
            raw_decision_threshold
        )
        test_metrics['threshold_constraint_type'] = validation_threshold_metrics[
            'constraint_type'
        ]
        test_metrics['calibration_method'] = calibration_metrics['method']
        test_metrics['calibration_nll_before'] = calibration_metrics['nll_before']
        test_metrics['calibration_nll_after'] = calibration_metrics['nll_after']
        test_metrics['calibration_oof_nll'] = calibration_metrics['oof_nll']
        test_metrics['entropy_threshold'] = self.entropy_threshold
        test_metrics['checkpoint_source'] = best_source
        test_metrics['best_epoch'] = best_epoch
        test_metrics['selection_metric'] = self.selection_metric
        test_metrics['best_validation_score'] = float(best_score)
        return test_metrics, self.history


class SequenceBaselineModel(nn.Module):
    """序列基线模型：统一接口支持 LSTM / Bi-LSTM / Attention-LSTM / ResNet-LSTM。

    注：Attention-LSTM 为单向；Bi-LSTM 和 ResNet-LSTM 均为双向。
        ResNet-LSTM 残差连接建立在双向 LSTM 之上，
        因此其表示能力天然包含双向信息。

    Args:
        static_dim: 静态特征维度
        temporal_dim: 时序特征维度
        hidden_dim: 隐藏维度
        model_type: 'lstm' / 'bilstm' / 'attention_lstm' / 'resnet_lstm'
        dropout: Dropout 比率
    """

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
        """前向传播：静态编码 → LSTM 序列编码 → 可选残差/注意力 → 分类。

        Args:
            static_feat: (B, static_dim) 静态特征
            temporal_feat: (B, T, temporal_dim) 时序特征

        Returns:
            (B, 2) 二分类 logits
        """
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
    """训练一个轻量级神经网络基线模型并返回测试集评估指标。

    使用标准 CrossEntropyLoss + AdamW，固定阈值 0.5。

    Args:
        name: 模型名称（用于日志）
        model: nn.Module 实例
        X_static_train/X_temporal_train/y_train: 训练数据
        X_static_test/X_temporal_test/y_test: 测试数据
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        dict: 含 acc / auc / auc_pr / f1 / sensitivity / specificity
    """
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
            # 梯度裁剪：error_if_nonfinite=False 避免 NaN 梯度导致整个实验崩溃
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                          error_if_nonfinite=False)
            optimizer.step()

    # 批量化评估，避免全量数据一次性加载到 GPU（Taiwan 30k 样本可能 OOM）
    model.eval()
    test_dataset = CreditDataset(X_static_test, X_temporal_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             pin_memory=torch.cuda.is_available())
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for batch in test_loader:
            static = batch['static'].to(device)
            temporal = batch['temporal'].to(device)
            labels_batch = batch['label']
            probs_batch = torch.softmax(model(static, temporal), dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs_batch)
            all_labels.append(labels_batch.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
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
    """包装双输入模型以适配 SHAP 的单列表输入或多位置参数输入。

    SHAP explainer 期望模型接受单一参数（列表或元组），此包装器自动解包。

    Args:
        model: 接受 (static, temporal) 双参数的 nn.Module
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        """兼容 SHAP 的列表输入和多位置参数输入。

        同时兼容两种模型接口：
        - AABiLSTM: forward(static, temporal, return_attention=False)
        - SequenceBaselineModel: forward(static, temporal)
        """
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            static_feat, temporal_feat = inputs[0]
        elif len(inputs) == 2:
            static_feat, temporal_feat = inputs
        else:
            raise ValueError("Expected static and temporal model inputs.")
        try:
            return self.model(static_feat, temporal_feat, return_attention=False)
        except TypeError:
            # 模型不接受 return_attention 参数（如 SequenceBaselineModel）
            return self.model(static_feat, temporal_feat)


class Explainer:
    """
    SHAP 可解释性分析封装。

    提供双输入模型（静态 + 时序）的 SHAP 解释：
    - 优先使用 DeepExplainer（快，基于梯度）
    - 失败时自动回退到 KernelExplainer（慢但鲁棒）
    - 自动展平时序特征以兼容 KernelExplainer 的单输入格式
    """

    def __init__(self, model, feature_names=None):
        """初始化 SHAP 解释器。

        Args:
            model: 接受 (static, temporal) 双输入的 nn.Module
            feature_names: 特征名列表（可选，用于 SHAP 绘图标签）
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP analysis requires the optional 'shap' package."
            ) from exc
        self.model = model
        self.feature_names = feature_names
        self.shap = shap

    def explain(self, static_data, temporal_data, sample_size=100):
        """使用 SHAP 解释模型对双输入特征的预测。

        优先尝试 DeepExplainer（梯度法，速度快），失败后自动回退到
        KernelExplainer（扰动法，速度慢但兼容性更好）。

        Args:
            static_data: (N, S) 静态特征
            temporal_data: (N, T, F) 时序特征
            sample_size: 背景样本数（用于拟合 SHAP explainer）

        Returns:
            (shap_values, explainer): shap_values 为双输入模型的 SHAP 值列表，
            explainer 为 SHAP explainer 对象
        """
        self.model.eval()
        if sample_size < 1 or sample_size >= len(static_data):
            raise ValueError(
                "sample_size must leave at least one sample to explain."
            )
        explain_size = min(50, len(static_data) - sample_size)

        # 创建背景数据
        background_static = static_data[:sample_size]
        background_temporal = temporal_data[:sample_size]

        # 使用包装后的模型
        wrapped_model = SHAPModelWrapper(self.model).to(device)
        wrapped_model.eval()

        # 转换为tensor
        background_static = torch.tensor(background_static).float().to(device)
        background_temporal = torch.tensor(background_temporal).float().to(device)

        test_sample_static = torch.tensor(
            static_data[sample_size:sample_size + explain_size]
        ).float().to(device)
        test_sample_temporal = torch.tensor(
            temporal_data[sample_size:sample_size + explain_size]
        ).float().to(device)

        try:
            # 尝试使用 DeepExplainer；用 catch_warnings 局部抑制 SHAP 的 LayerNorm 警告
            print("Attempting SHAP DeepExplainer...")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='.*unrecognized nn.Module.*',
                    category=UserWarning,
                )
                explainer = self.shap.DeepExplainer(
                    wrapped_model,
                    [background_static, background_temporal],
                )
                shap_values = explainer.shap_values([test_sample_static, test_sample_temporal])
            print("SHAP DeepExplainer succeeded.")
            return shap_values, explainer
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to KernelExplainer (slower but more robust)...")
            # 备用方案：使用 KernelExplainer
            return self._kernel_explain(static_data, temporal_data, sample_size)

    def _kernel_explain(self, static_data, temporal_data, sample_size):
        """使用 KernelExplainer 作为备用方案（扰动法，比 DeepExplainer 慢但兼容性更好）。

        KernelExplainer 需要展平的 (N, D) 单表输入，因此将时序特征 reshape 为
        (N, T*F) 后与静态特征拼接；预测函数内部再恢复三维形状。
        """
        wrapped_model = SHAPModelWrapper(self.model).to(device)
        wrapped_model.eval()

        # KernelExplainer 需要展平的 (N, D) 输入
        static_flat = static_data[:sample_size]
        temporal_flat = temporal_data[:sample_size].reshape(sample_size, -1)  # (sample, T*F)
        background_flat = np.concatenate([static_flat, temporal_flat], axis=1)

        def model_predict(flat_input):
            """KernelExplainer 回调：展平输入 → 恢复三维 → 前向 → softmax。
            返回完整 (N, 2) 概率矩阵而非单列，确保 shap_values 与
            DeepExplainer 的 per-class list 格式一致。"""
            n = flat_input.shape[0]
            static_dim = static_data.shape[1]
            temporal_dim_2d = temporal_data.shape[1]   # T
            temporal_dim_3d = temporal_data.shape[2]   # F

            static_part = torch.FloatTensor(flat_input[:, :static_dim]).to(device)
            temporal_part = torch.FloatTensor(
                flat_input[:, static_dim:].reshape(n, temporal_dim_2d, temporal_dim_3d)
            ).to(device)

            with torch.inference_mode():
                output = wrapped_model([static_part, temporal_part])
                probs = torch.softmax(output, dim=-1)
            return probs.cpu().numpy()  # 返回 (N, 2)，与 DeepExplainer per-class 格式一致

        # 测试数据展平
        explain_size = min(50, len(static_data) - sample_size)
        test_static_flat = static_data[sample_size:sample_size + explain_size]
        test_temporal_flat = temporal_data[
            sample_size:sample_size + explain_size
        ].reshape(explain_size, -1)
        test_flat = np.concatenate([test_static_flat, test_temporal_flat], axis=1)

        explainer = self.shap.KernelExplainer(
            model_predict,
            background_flat[:100],
        )
        # shap_values: 返回 list of 2 arrays（每类一个），每个 (explain_size, total_features)
        raw_shap = explainer.shap_values(test_flat, nsamples=100)
        if isinstance(raw_shap, list):
            # 堆叠为 (explain_size, total_features, 2)，与 DeepExplainer 输出对齐
            raw_shap = np.stack(raw_shap, axis=-1)

        # 拆分静态/时序 SHAP 值，保留类别维度
        static_shap = raw_shap[:, :static_data.shape[1], :]
        temporal_shap = raw_shap[:, static_data.shape[1]:, :].reshape(
            explain_size,
            temporal_data.shape[1],
            temporal_data.shape[2],
            -1,  # 类别维度
        )

        print("KernelExplainer succeeded.")
        return [static_shap, temporal_shap], explainer


def build_feature_names(data_loader, static_dim, temporal_steps, temporal_dim):
    """从 data_loader 构建静态和时序特征的完整名称列表。

    Args:
        data_loader: CreditDataLoader 实例
        static_dim: 静态特征维度
        temporal_steps: 时序步数
        temporal_dim: 时序特征通道数

    Returns:
        (static_names, temporal_names): 两个名称列表
    """
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
    """运行 SHAP 分析并输出 Top-10 特征重要性排名。

    对双输入模型分别计算静态特征和展平时序特征的 |SHAP| 均值，
    合并后取前 10 个最重要的特征输出。

    Args:
        model: 训练好的 AABiLSTM 模型
        data_loader: CreditDataLoader（含特征名）
        X_static_test/X_temporal_test: 评估数据
        dataset_name: 数据集名（用于输出标题）
        sample_size: SHAP 背景样本数（None 时自动根据数据量调整）

    Returns:
        shap_values 或 None（分析失败时）
    """
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

    # SHAP 对二分类模型返回 (n_samples, n_features, 2) 形状，
    # 每个样本每个特征有两个 SHAP 值（分别对应两个类别的 logit 贡献），
    # 取正类（index=1）的 SHAP 值。
    if static_values.ndim >= 3 and static_values.shape[-1] == 2:
        static_values = static_values[..., 1]
    if temporal_values.ndim >= 4 and temporal_values.shape[-1] == 2:
        temporal_values = temporal_values[..., 1]

    # 展平时序 SHAP 值与静态 SHAP 值合并，按 |SHAP| 均值排序
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


def bootstrap_metric_intervals(
    labels,
    predictions,
    probabilities,
    n_bootstrap=500,
    random_state=42,
):
    """为核心测试指标（Accuracy, Sensitivity, AUC, AUC-PR）提供非参数 95% bootstrap 置信区间。

    Args:
        labels: (N,) 真实标签
        predictions: (N,) 模型预测
        probabilities: (N,) 预测概率
        n_bootstrap: bootstrap 重采样次数
        random_state: 随机种子

    Returns:
        dict: 各指标的 {metric}_ci95_low / {metric}_ci95_high
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(random_state)
    values = {
        'accuracy': [],
        'sensitivity': [],
        'auc': [],
        'auc_pr': [],
    }
    dropped_auc_samples = 0
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        y_sample = labels[indices]
        pred_sample = predictions[indices]
        prob_sample = probabilities[indices]
        values['accuracy'].append(
            accuracy_score(y_sample, pred_sample)
        )
        values['sensitivity'].append(
            recall_score(
                y_sample,
                pred_sample,
                zero_division=0,
            )
        )
        if len(np.unique(y_sample)) == 2:
            values['auc'].append(
                roc_auc_score(y_sample, prob_sample)
            )
            values['auc_pr'].append(
                average_precision_score(y_sample, prob_sample)
            )
        else:
            dropped_auc_samples += 1

    if dropped_auc_samples > 0:
        print(
            f"Bootstrap: {dropped_auc_samples}/{n_bootstrap} AUC samples "
            f"dropped due to single-class resamples."
        )

    result = {}
    for name, samples in values.items():
        if not samples:
            continue
        lower, upper = np.quantile(samples, [0.025, 0.975])
        result[f'{name}_ci95_low'] = float(lower)
        result[f'{name}_ci95_high'] = float(upper)
    if dropped_auc_samples > 0:
        result['bootstrap_auc_dropped'] = dropped_auc_samples
    return result


# ==============================
# 6. 主程序
# ==============================

def run_experiment(dataset_name='taiwan', epoch=None, batch_size=None, data_path=None,
                   run_baselines=True, run_ablation=True, run_robustness=True,
                   run_shap=True, make_plots=True,
                   threshold_min_sensitivity=None,
                   seed=None,
                   split_seed=42,
                   hidden_dim=None, num_layers=None,
                   dropout=None, lr=None, weight_decay=None,
                   loss_name='dynamic_focal', selection_metric='auc_pr',
                   threshold_objective='hybrid',
                   class_balance_power=0.5,
                   focal_gamma_max=None,
                   ema_decay=None,
                   calibrate_probabilities=True,
                   analysis_on_test=False,
                   threshold_confidence=None,
                   auto_tune=False,
                   tune_epochs=10):
    """
    运行完整实验：数据加载 → 预处理 → 建模 → 训练 → 评估 → 分析。

    实验流水线：
    1. 加载数据集并做四层分割
    2. 初始化 AA-BiLSTM 模型
    3. 训练（含 auto-tune 可选超参搜索）
    4. 测试集评估 + Bootstrap CI
    5. 可选：baselines 对比、消融实验、不平衡鲁棒性
    6. 可选：SHAP 解释 + 训练历史可视化

    Args:
        dataset_name: 'taiwan'
        epoch: 训练轮数（None 时使用数据集默认值）
        batch_size: 批次大小（None 时使用数据集默认值）
        data_path: 本地数据文件路径
        run_baselines: 是否运行基线模型对比
        run_ablation: 是否运行消融实验
        run_robustness: 是否运行不平衡鲁棒性实验
        run_shap: 是否运行 SHAP 可解释性分析
        make_plots: 是否生成训练曲线和混淆矩阵图
        threshold_min_sensitivity: 阈值搜索的敏感性下限
        seed: 模型随机种子
        split_seed: 数据分割随机种子
        hidden_dim: 隐藏维度（None 时使用数据集默认值）
        num_layers: 循环层数（None 时使用数据集默认值）
        dropout: Dropout 比率（None 时使用数据集默认值）
        lr: 学习率（None 时使用数据集默认值）
        weight_decay: 权重衰减（None 时使用数据集默认值）
        loss_name: 损失函数类型 ('dynamic_focal' / 'weighted_ce' / 'cross_entropy')
        selection_metric: checkpoint 选择指标
        threshold_objective: 阈值搜索目标
        class_balance_power: 类别平衡幂指数
        focal_gamma_max: focal loss 的 gamma_max
        ema_decay: EMA 衰减率
        calibrate_probabilities: 是否做 Platt 概率校准
        analysis_on_test: 分析时使用测试集（默认使用验证集保护 holdout）
        threshold_confidence: Wilson 下界的置信水平
        auto_tune: 是否在最终训练前做轻量超参搜索
        tune_epochs: 每个候选配置的调优 epoch 预算

    Returns:
        (model, test_metrics, history): 训练好的模型、测试指标字典、训练历史
    """
    dataset_name = dataset_name.lower()
    if dataset_name != 'taiwan':
        raise ValueError("dataset_name must be 'taiwan'.")
    if loss_name not in {'dynamic_focal', 'weighted_ce', 'cross_entropy'}:
        raise ValueError(
            "loss_name must be 'dynamic_focal', 'weighted_ce' or 'cross_entropy'."
        )
    if threshold_min_sensitivity is None:
        threshold_min_sensitivity = (
            0.55
        )
    if not 0.0 <= threshold_min_sensitivity <= 1.0:
        raise ValueError("threshold_min_sensitivity must be in [0, 1].")
    if threshold_confidence is None:
        threshold_confidence = (
            0.80
        )
    if not 0.5 <= threshold_confidence < 1.0:
        raise ValueError("threshold_confidence must be in [0.5, 1).")
    if tune_epochs < 1:
        raise ValueError("tune_epochs must be at least 1.")

    print(f"\n{'=' * 60}")
    print(f"Running AA-BiLSTM Experiment on {dataset_name.upper()} Dataset")
    print(f"{'=' * 60}")
    print(f"Using device: {device}")
    seed = 42 if seed is None else seed
    set_seed(seed)

    # 1. 数据加载
    data_loader = CreditDataLoader()

    static_feat, temporal_feat, y = data_loader.load_taiwan_credit(data_path)
    batch_size = 256 if batch_size is None else batch_size
    epoch = 50 if epoch is None else epoch
    hidden_dim = 96 if hidden_dim is None else hidden_dim
    num_layers = 3 if num_layers is None else num_layers
    dropout = 0.30 if dropout is None else dropout
    lr = 7e-4 if lr is None else lr
    weight_decay = 2e-4 if weight_decay is None else weight_decay
    focal_gamma_max = (
        1.75 if focal_gamma_max is None else focal_gamma_max
    )
    ema_decay = 0.995 if ema_decay is None else ema_decay
    patience = 15

    if (
        batch_size < 1
        or epoch < 1
        or hidden_dim < 2
        or num_layers < 1
    ):
        raise ValueError(
            "batch_size, epoch, and num_layers must be positive; "
            "hidden_dim must be at least 2."
        )
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("lr must be positive and weight_decay non-negative.")

    print(f"Data shape: Static {static_feat.shape}, Temporal {temporal_feat.shape}")
    print(f"Class distribution: {Counter(y)}")
    print(f"Default rate: {y.mean() * 100:.2f}%")
    print(
        f"Config: seed={seed}, epochs={epoch}, batch={batch_size}, "
        f"hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.2f}, "
        f"lr={lr:g}, loss={loss_name}, select={selection_metric}, "
        f"split_seed={split_seed}"
    )

    # 2. 数据预处理
    (X_static_train, X_temporal_train, y_train,
     X_static_val, X_temporal_val, y_val,
     X_static_test, X_temporal_test, y_test) = data_loader.preprocess(
        static_feat,
        temporal_feat,
        y,
        split_seed=split_seed,
    )
    (
        X_static_calibration,
        X_temporal_calibration,
        y_calibration,
    ) = data_loader.calibration_data_

    # 2. 创建 DataLoader
    train_dataset = CreditDataset(X_static_train, X_temporal_train, y_train)
    val_dataset = CreditDataset(X_static_val, X_temporal_val, y_val)
    calibration_dataset = CreditDataset(
        X_static_calibration,
        X_temporal_calibration,
        y_calibration,
    )
    test_dataset = CreditDataset(X_static_test, X_temporal_test, y_test)

    pin_memory = torch.cuda.is_available()
    # Windows 上 spawn 多进程 worker 的导入和序列化成本通常高于收益；
    # 数据已常驻内存，单进程加载足够快。
    num_workers = (
        min(4, os.cpu_count() or 1)
        if torch.cuda.is_available() and os.name != 'nt'
        else 0
    )
    persistent = num_workers > 0
    eval_workers = num_workers // 2
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=persistent,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=eval_workers,
        persistent_workers=eval_workers > 0,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=eval_workers,
        persistent_workers=eval_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=eval_workers,
        persistent_workers=eval_workers > 0,
    )

    # 3. 模型初始化
    static_dim = X_static_train.shape[1]
    temporal_steps = X_temporal_train.shape[1]
    temporal_dim = X_temporal_train.shape[2]
    enable_local_attention = True

    # 模型工厂函数：支持 auto-tune 时用不同超参构建多个候选模型
    def build_model(candidate_hidden, candidate_layers, candidate_dropout):
        return AABiLSTM(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            temporal_steps=temporal_steps,
            hidden_dim=candidate_hidden,
            num_layers=candidate_layers,
            num_classes=2,
            dropout=candidate_dropout,
            use_fusion=True,
            use_ag_resunit=True,
            bidirectional=True,
            use_multiscale=True,
            local_window=min(3, temporal_steps),
            use_global_context=True,
            use_local_attention=enable_local_attention,
            temporal_categorical_index=0,
            temporal_category_min=-2,
            temporal_category_max=8,
            use_step_embedding=True,
            multiscale_mode=(
                'lightweight'
            ),
        )

    if auto_tune:
        # 轻量级超参搜索：默认配置 + 2 个变体，不触及测试集。
        # 候选 1：默认配置
        # 候选 2：更小模型（75% hidden_dim，少一层）+ 更高 lr + 更低 dropout
        # 候选 3：原尺寸但少一层 + 更高 dropout + 更低 lr
        compact_hidden = max(32, int(round(hidden_dim * 0.75 / 8)) * 8)
        candidates = [
            (hidden_dim, num_layers, dropout, lr),
            (
                compact_hidden,
                max(1, num_layers - 1),
                max(0.15, dropout - 0.05),
                lr * 1.15,
            ),
            (
                hidden_dim,
                max(1, num_layers - 1),
                min(0.45, dropout + 0.05),
                lr * 0.80,
            ),
        ]
        candidates = list(dict.fromkeys(candidates))
        candidate_scores = []
        print(
            f"Running leakage-safe lightweight tuning with "
            f"{len(candidates)} candidates..."
        )
        for candidate_index, candidate in enumerate(candidates):
            candidate_hidden, candidate_layers, candidate_dropout, candidate_lr = (
                candidate
            )
            set_seed(seed)
            if train_loader.generator is not None:
                train_loader.generator.manual_seed(seed)
            candidate_model = build_model(
                candidate_hidden,
                candidate_layers,
                candidate_dropout,
            )
            candidate_trainer = Trainer(
                candidate_model,
                train_loader,
                val_loader,
                val_loader,  # auto-tune 阶段不接触 test_loader，用 val_loader 代替
                calibration_loader=calibration_loader,
                num_epochs=min(tune_epochs, epoch),
                lr=candidate_lr,
                weight_decay=weight_decay,
                use_dynamic_focal=loss_name == 'dynamic_focal',
                use_class_weight=loss_name == 'weighted_ce',
                use_early_stopping=True,
                threshold_min_sensitivity=threshold_min_sensitivity,
                selection_metric=selection_metric,
                threshold_objective=threshold_objective,
                class_balance_power=class_balance_power,
                focal_gamma_max=focal_gamma_max,
                ema_decay=None,
                calibrate_probabilities=False,
                threshold_confidence=threshold_confidence,
            )
            candidate_trainer.early_stopping.patience = min(
                patience,
                max(4, tune_epochs // 2),
            )
            candidate_metrics, _ = candidate_trainer.train()
            candidate_score = float(
                candidate_metrics['best_validation_score']
            )
            candidate_scores.append((candidate_score, candidate))
            print(
                f"Tuning candidate {candidate_index + 1}: "
                f"hidden={candidate_hidden}, layers={candidate_layers}, "
                f"dropout={candidate_dropout:.2f}, lr={candidate_lr:g}, "
                f"selection={candidate_score:.4f}"
            )
        _, best_candidate = max(
            candidate_scores,
            key=lambda item: item[0],
        )
        hidden_dim, num_layers, dropout, lr = best_candidate
        print(
            f"Selected tuning configuration: hidden={hidden_dim}, "
            f"layers={num_layers}, dropout={dropout:.2f}, lr={lr:g}"
        )
        set_seed(seed)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(seed)

    model = build_model(hidden_dim, num_layers, dropout)

    total_parameters = sum(p.numel() for p in model.parameters())
    fusion_parameters = (
        sum(p.numel() for p in model.feature_fusion.parameters())
        if model.feature_fusion is not None else 0
    )
    print(f"\nModel parameters: {total_parameters:,}")
    print(
        f"Fusion parameters: {fusion_parameters:,} "
        f"({fusion_parameters / max(total_parameters, 1):.2%})"
    )

    # 4. 训练
    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        calibration_loader=calibration_loader,
        num_epochs=epoch,
        lr=lr,
        weight_decay=weight_decay,
        use_dynamic_focal=loss_name == 'dynamic_focal',
        use_class_weight=loss_name == 'weighted_ce',
        use_early_stopping=True,
        threshold_min_sensitivity=threshold_min_sensitivity,
        selection_metric=selection_metric,
        threshold_objective=threshold_objective,
        class_balance_power=class_balance_power,
        focal_gamma_max=focal_gamma_max,
        ema_decay=ema_decay,
        calibrate_probabilities=calibrate_probabilities,
        threshold_confidence=threshold_confidence,
    )
    trainer.early_stopping.patience = patience  # 动态设置耐心值

    test_metrics, history = trainer.train()

    test_metrics.update({
        'seed': int(seed),
        'split_seed': int(split_seed),
        'epochs_requested': int(epoch),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'dropout': float(dropout),
        'learning_rate': float(lr),
        'batch_size': int(batch_size),
        'loss_name': loss_name,
    })
    test_metrics.update(bootstrap_metric_intervals(
        test_metrics['labels'],
        test_metrics['predictions'],
        test_metrics['probabilities'],
        random_state=split_seed,
    ))
    # 将部署所需的辅助状态绑定到模型上，方便同一进程内进行推理。
    # CLI 可通过 --bundle / --checkpoint 分别保存。
    model.preprocessing_state = data_loader.export_preprocessing_state()
    model.training_config = {
        'dataset': dataset_name,
        'seed': int(seed),
        'split_seed': int(split_seed),
        'loss_name': loss_name,
        'selection_metric': selection_metric,
        'threshold_objective': threshold_objective,
        'threshold_min_sensitivity': float(
            threshold_min_sensitivity
        ),
        'class_balance_power': float(class_balance_power),
        'focal_gamma_max': float(focal_gamma_max),
        'ema_decay': float(ema_decay),
        'threshold_confidence': float(threshold_confidence),
        'calibration_method': 'cross_fitted_platt',
        'multiscale_mode': (
            'lightweight'
        ),
        'temporal_state_embedding': True,
        'step_embedding': True,
        'auto_tune': bool(auto_tune),
        'split_sizes': {
            name: int(len(indices))
            for name, indices in data_loader.split_indices_.items()
            if isinstance(indices, np.ndarray)
        },
    }

    # 5. 结果输出
    print(f"\n{'=' * 60}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Sensitivity (Recall): {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity'] * 100:.2f}%)")
    print(f"Precision:            {test_metrics['precision']:.4f}")
    print(f"Specificity:          {test_metrics['specificity']:.4f}")
    print(f"Accuracy:             {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"F1-Score:             {test_metrics['f1']:.4f}")
    print(f"F2-Score:             {test_metrics['f2']:.4f}")
    print(f"AUC-ROC:              {test_metrics['auc']:.4f}")
    print(f"AUC-PR:               {test_metrics['auc_pr']:.4f}")
    print(f"Brier Score:          {test_metrics['brier_score']:.4f}")
    print(f"Decision Threshold:   {test_metrics['threshold']:.3f}")
    print(
        "95% bootstrap CI | "
        f"Accuracy [{test_metrics['accuracy_ci95_low']:.4f}, "
        f"{test_metrics['accuracy_ci95_high']:.4f}] | "
        f"Sensitivity [{test_metrics['sensitivity_ci95_low']:.4f}, "
        f"{test_metrics['sensitivity_ci95_high']:.4f}]"
    )
    print(f"{'=' * 60}\n")

    # 分析集选择：默认用验证集做 SHAP/消融/鲁棒性分析，保护测试集的"最终答案"地位；
    # --analysis-on-test 可用于论文最终版本时在测试集上生成图表。
    analysis_static_eval = (
        X_static_test if analysis_on_test else X_static_val
    )
    analysis_temporal_eval = (
        X_temporal_test if analysis_on_test else X_temporal_val
    )
    analysis_y_eval = y_test if analysis_on_test else y_val
    analysis_loader = test_loader if analysis_on_test else val_loader
    analysis_split_name = 'test' if analysis_on_test else 'validation'

    # 6. 对比基准模型
    if run_baselines:
        print(
            f"Comparison with baseline models on {analysis_split_name} split..."
        )
        compare_baselines(X_static_train, X_temporal_train, y_train,
                          analysis_static_eval, analysis_temporal_eval,
                          analysis_y_eval,
                          deep_epochs=min(epoch, 30),
                          batch_size=batch_size)

    # 7. 消融实验
    if run_ablation:
        print("\nRunning ablation study...")
        run_ablation_study(static_dim, temporal_dim, temporal_steps,
                           train_loader, val_loader, analysis_loader,
                           calibration_loader=calibration_loader,
                           epoch=min(epoch, 50),
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           dropout=dropout,
                           lr=lr,
                           weight_decay=weight_decay,
                           enable_local_attention=enable_local_attention,
                           enable_temporal_categorical=enable_local_attention,
                           multiscale_mode=(
                               'lightweight'
                           ),
                           use_step_embedding=True,
                           threshold_confidence=threshold_confidence,
                           threshold_min_sensitivity=threshold_min_sensitivity,
                           selection_metric=selection_metric,
                           threshold_objective=threshold_objective,
                           class_balance_power=class_balance_power,
                           focal_gamma_max=focal_gamma_max,
                           ema_decay=ema_decay,
                           calibrate_probabilities=calibrate_probabilities,
                           seed=seed,
                           patience=patience)


    if run_robustness:
        run_imbalance_robustness_study(
            X_static_train,
            X_temporal_train,
            y_train,
            X_static_val,
            X_temporal_val,
            y_val,
            X_static_calibration,
            X_temporal_calibration,
            y_calibration,
            analysis_static_eval,
            analysis_temporal_eval,
            analysis_y_eval,
            epoch=min(epoch, 30),
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            enable_local_attention=enable_local_attention,
            enable_temporal_categorical=enable_local_attention,
            multiscale_mode=(
                'lightweight'
            ),
            use_step_embedding=True,
            threshold_confidence=threshold_confidence,
            threshold_min_sensitivity=threshold_min_sensitivity,
            selection_metric=selection_metric,
            threshold_objective=threshold_objective,
            class_balance_power=class_balance_power,
            focal_gamma_max=focal_gamma_max,
            ema_decay=ema_decay,
            calibrate_probabilities=calibrate_probabilities,
            seed=seed,
            patience=patience
        )

    if run_shap:
        run_shap_summary(
            model,
            data_loader,
            analysis_static_eval,
            analysis_temporal_eval,
            dataset_name,
        )
    # 8. 可视化训练历史
    if make_plots:
        plot_training_history(history, dataset_name)

    return model, test_metrics, history


def compare_baselines(X_static_train, X_temporal_train, y_train,
                      X_static_test, X_temporal_test, y_test,
                      deep_epochs=20, batch_size=128):
    """
    与传统 ML 和深度序列基线模型进行全面对比。

    传统模型（展平时序 → 单表输入）：
    - Random Forest、XGBoost

    深度序列模型（保留时序结构）：
    - Standard LSTM、Bi-LSTM、Attention-LSTM

    Args:
        X_static_train/X_temporal_train/y_train: 训练数据
        X_static_test/X_temporal_test/y_test: 测试数据
        deep_epochs: 深度学习基线的训练轮数
        batch_size: 批次大小

    Returns:
        dict: 各模型在测试集上的指标字典
    """
    from sklearn.ensemble import RandomForestClassifier

    # 展平时序特征用于传统 ML，保持 DataFrame 格式以保留特征名
    temporal_train_flat = X_temporal_train.reshape(X_temporal_train.shape[0], -1)
    temporal_test_flat = X_temporal_test.reshape(X_temporal_test.shape[0], -1)

    # 生成特征名
    static_cols = [f'static_{i}' for i in range(X_static_train.shape[1])]
    temporal_cols = [f'temporal_{i}' for i in range(temporal_train_flat.shape[1])]
    feature_names = static_cols + temporal_cols

    X_train_flat = pd.DataFrame(
        np.concatenate([X_static_train, temporal_train_flat], axis=1),
        columns=feature_names
    )
    X_test_flat = pd.DataFrame(
        np.concatenate([X_static_test, temporal_test_flat], axis=1),
        columns=feature_names
    )

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
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
    }
    for name, model_type in deep_models.items():
        try:
            set_seed(42)
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
                       train_loader, val_loader, test_loader,
                       calibration_loader=None, epoch=30,
                       hidden_dim=64, num_layers=4, dropout=0.3,
                       lr=1e-3, weight_decay=1e-4,
                       enable_local_attention=True,
                       enable_temporal_categorical=True,
                       multiscale_mode='lightweight',
                       use_step_embedding=True,
                       threshold_confidence=0.80,
                       threshold_min_sensitivity=0.40,
                       selection_metric='auc_pr',
                       threshold_objective='hybrid',
                       class_balance_power=0.5,
                       focal_gamma_max=2.0,
                       ema_decay=0.995,
                       calibrate_probabilities=True,
                       seed=42,
                       patience=10):
    """
    消融实验：按方法论模块逐一添加，量化各模块对性能的增量贡献。

    消融顺序（渐进累积）：
    1. Standard LSTM（基线，无任何增强）
    2. + Multi-Scale Encoding（细粒度 + 中粒度 + 全局摘要，对应 3.5 节）
    3. + AdaptiveFusion（全局上下文 + 局部冲击注意力 + 门控残差，对应 3.3 节）
    4. + AG-ResUnit + Bi-LSTM（λ 门控 + 残差 + 双向，对应 3.4 节）
    5. + DynamicFocalLoss（完整 AA-BiLSTM，对应 3.6 节）

    每个配置独立训练并报告 AUC/AUC-PR/Accuracy/F1/Sensitivity/Specificity/参数量。

    Args:
        static_dim/temporal_dim/temporal_steps: 特征维度信息
        train_loader/val_loader/test_loader: 数据加载器
        calibration_loader: 校准集加载器
        epoch: 每个消融配置的训练轮数
        hidden_dim/num_layers/dropout/lr/weight_decay: 模型超参
        enable_local_attention: 是否允许启用局部注意力
        enable_temporal_categorical: 是否启用时序类别 Embedding
        multiscale_mode: 'legacy' 或 'lightweight'
        use_step_embedding: 是否添加可学习的时间步 Embedding
        threshold_confidence: Wilson 下界的置信水平

    Returns:
        list of dict: 每个消融配置的评估结果
    """

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    # 消融配置：按组件逐一添加。use_cross 指旧版 TS-CrossAttention（被 AdaptiveFusion 替代）；
    # 其余参数直接映射到 AABiLSTM 构造参数。
    configs = [
        {
            'name': 'Standard LSTM (Baseline)',
            'use_cross': False,
            'use_multiscale': False,
            'use_global': False,
            'use_local': False,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'use_standard_lstm': True,
        },
        {
            'name': '+ Multi-Scale Encoding',
            'use_cross': False,
            'use_multiscale': True,
            'use_global': False,
            'use_local': False,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False
        },
        {
            'name': '+ AdaptiveFusion',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': True,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False
        }
    ]

    configs.extend([
        {
            'name': '+ AG-ResUnit + Bi-LSTM',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': enable_local_attention,
            'use_ag': True,
            'use_bi': True,
            'use_focal': False
        },
        {
            'name': '+ DynamicFocalLoss (Full AA-BiLSTM)',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': enable_local_attention,
            'use_ag': True,
            'use_bi': True,
            'use_focal': True
        }
    ])

    results = []
    for config in configs:
        print(f"\nTesting: {config['name']}")
        set_seed(seed)
        if getattr(train_loader, 'generator', None) is not None:
            train_loader.generator.manual_seed(seed)

        if config.get('use_standard_lstm'):
            # 标准 LSTM 基线：直接用 train_sequence_baseline（和基线对比完全一致）
            model = SequenceBaselineModel(
                static_dim=static_dim,
                temporal_dim=temporal_dim,
                hidden_dim=64,
                model_type='lstm',
                dropout=dropout,
            )
            train_ds = train_loader.dataset
            test_ds = test_loader.dataset
            metrics = train_sequence_baseline(
                config['name'],
                model,
                train_ds.static_features.numpy(),
                train_ds.temporal_features.numpy(),
                train_ds.labels.numpy(),
                test_ds.static_features.numpy(),
                test_ds.temporal_features.numpy(),
                test_ds.labels.numpy(),
                epochs=30,
                batch_size=train_loader.batch_size,
                lr=1e-3,
            )
            row = {
                'name': config['name'],
                'auc': metrics['auc'],
                'auc_pr': metrics['auc_pr'],
                'accuracy': metrics['acc'],
                'f1': metrics['f1'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'parameters': sum(p.numel() for p in model.parameters()),
            }
        else:
            model = AABiLSTM(
                static_dim=static_dim,
                temporal_dim=temporal_dim,
                temporal_steps=temporal_steps,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout,
                use_fusion=config['use_cross'],
                use_ag_resunit=config['use_ag'],
                bidirectional=config['use_bi'],
                use_multiscale=config['use_multiscale'],
                local_window=min(3, temporal_steps),
                use_global_context=config['use_global'],
                use_local_attention=config['use_local'],
                temporal_categorical_index=(
                    0 if enable_temporal_categorical else None
                ),
                use_step_embedding=use_step_embedding,
                multiscale_mode=multiscale_mode,
            )
            trainer = Trainer(
                model, train_loader, val_loader, test_loader,
                calibration_loader=calibration_loader,
                num_epochs=epoch,
                lr=lr,
                weight_decay=weight_decay,
                use_dynamic_focal=config['use_focal'],
                use_early_stopping=True,
                threshold_min_sensitivity=threshold_min_sensitivity,
                selection_metric=selection_metric,
                threshold_objective=threshold_objective,
                class_balance_power=class_balance_power,
                focal_gamma_max=focal_gamma_max,
                ema_decay=ema_decay,
                calibrate_probabilities=calibrate_probabilities,
                threshold_confidence=threshold_confidence,
            )
            trainer.early_stopping.patience = patience
            metrics, _ = trainer.train()
            row = {
                'name': config['name'],
                'auc': metrics['auc'],
                'auc_pr': metrics['auc_pr'],
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'parameters': sum(p.numel() for p in model.parameters()),
            }
        results.append(row)
        print(
            f"Result: AUC={row['auc']:.4f}, AUC-PR={row['auc_pr']:.4f}, "
            f"Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}, "
            f"Sens={row['sensitivity']:.4f}, Spec={row['specificity']:.4f}, "
            f"Params={row['parameters']:,}"
        )

    print("\n" + "=" * 145)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 145)
    print(
        f"{'Configuration':<28} {'AUC':<9} {'AUC-PR':<9} {'Acc':<9} "
        f"{'F1':<9} {'Sens':<9} {'Spec':<9} {'Params(M)':<11} "
        f"{'ΔAUC(pp)':<10} {'ΔSens(pp)':<10}"
    )
    print("-" * 145)

    base_auc = results[0]['auc']
    base_sensitivity = results[0]['sensitivity']
    for row in results:
        print(
            f"{row['name']:<28} {row['auc']:<9.4f} "
            f"{row['auc_pr']:<9.4f} {row['accuracy']:<9.4f} "
            f"{row['f1']:<9.4f} {row['sensitivity']:<9.4f} "
            f"{row['specificity']:<9.4f} "
            f"{row['parameters'] / 1e6:<11.3f} "
            f"{(row['auc'] - base_auc) * 100:<10.2f} "
            f"{(row['sensitivity'] - base_sensitivity) * 100:<10.2f}"
        )
    print("=" * 145)
    return results


def subsample_training_rate(X_static_train, X_temporal_train, y_train,
                            target_default_rate, random_state=42):
    """对某一类进行随机欠采样，调整训练集的违约率至目标水平。

    优先保持多数类完整，对少数类做欠采样；当少数类样本不足时反向操作。

    Args:
        X_static_train/X_temporal_train/y_train: 训练数据
        target_default_rate: 目标违约率（0~1）
        random_state: 随机种子

    Returns:
        三元组 (X_static, X_temporal, y): 欠采样后的训练数据
    """
    if not 0 < target_default_rate < 1:
        raise ValueError("target_default_rate must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    y_train = np.asarray(y_train)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes are required for imbalance robustness study.")

    # 策略：优先保持多数类完整，对少数类做欠采样；
    # 当少数类样本不足以达到目标比例时，反向操作——欠采样多数类。
    # 公式：target_rate = pos / (pos + neg) → pos = target_rate/(1-target_rate) * neg
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
                                   X_static_calibration,
                                   X_temporal_calibration, y_calibration,
                                   X_static_test, X_temporal_test, y_test,
                                   target_rates=(0.02, 0.05, 0.10, 0.22, 0.30),
                                   epoch=20, batch_size=128, hidden_dim=128,
                                   num_layers=8, dropout=0.4,
                                   lr=5e-4, weight_decay=2e-4,
                                   enable_local_attention=True,
                                   enable_temporal_categorical=True,
                                   multiscale_mode='lightweight',
                                   use_step_embedding=True,
                                   threshold_confidence=0.80,
                                   threshold_min_sensitivity=0.40,
                                   selection_metric='auc_pr',
                                   threshold_objective='hybrid',
                                   class_balance_power=0.5,
                                   focal_gamma_max=2.0,
                                   ema_decay=0.995,
                                   calibrate_probabilities=True,
                                   seed=42,
                                   patience=10):
    """
    类别不平衡鲁棒性实验：在不同训练集违约率下比较 Standard LSTM 与 AA-BiLSTM。

    通过 subsample_training_rate 将训练集违约率调整到 {2%, 5%, 10%, 22%, 30%}，
    每种失衡程度下独立训练 Standard LSTM 和 AA-BiLSTM，对比 AUC 和 Sensitivity，
    评估模型在极端少数类场景下的鲁棒性。

    Args:
        X_static_train/.../y_test: 完整数据集按四层分割后的各子集
        target_rates: 要测试的目标违约率序列
        epoch/batch_size/hidden_dim/.../lr/weight_decay: 模型超参
        enable_local_attention: 是否启用局部注意力
        enable_temporal_categorical: 是否启用时序类别 Embedding
        multiscale_mode: 'legacy' 或 'lightweight'
        use_step_embedding: 是否添加可学习的时间步 Embedding
        threshold_confidence: Wilson 下界的置信水平

    Returns:
        list of dict: 每种违约率下两个模型的评估结果
    """
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
    calibration_loader = DataLoader(
        CreditDataset(
            X_static_calibration,
            X_temporal_calibration,
            y_calibration,
        ),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CreditDataset(X_static_test, X_temporal_test, y_test),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    results = []
    for rate in target_rates:
        # 每种违约率使用 deterministic seed，不同 rate 独立但可复现
        set_seed(int(rate * 10000) + seed)
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
            hidden_dim=64,
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
            epochs=30,
            batch_size=batch_size,
            lr=1e-3,
        )

        set_seed(int(rate * 10000) + seed)

        aa_model = AABiLSTM(
            static_dim=X_static_train.shape[1],
            temporal_dim=X_temporal_train.shape[2],
            temporal_steps=X_temporal_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            use_fusion=True,
            use_ag_resunit=True,
            bidirectional=True,
            use_multiscale=True,
            local_window=min(3, X_temporal_train.shape[1]),
            use_global_context=True,
            use_local_attention=enable_local_attention,
            temporal_categorical_index=(
                0 if enable_temporal_categorical else None
            ),
            use_step_embedding=use_step_embedding,
            multiscale_mode=multiscale_mode,
        )
        trainer = Trainer(
            aa_model,
            train_loader,
            val_loader,
            test_loader,
            calibration_loader=calibration_loader,
            num_epochs=epoch,
            lr=lr,
            weight_decay=weight_decay,
            use_dynamic_focal=True,
            use_early_stopping=True,
            threshold_min_sensitivity=threshold_min_sensitivity,
            selection_metric=selection_metric,
            threshold_objective=threshold_objective,
            class_balance_power=class_balance_power,
            focal_gamma_max=focal_gamma_max,
            ema_decay=ema_decay,
            calibrate_probabilities=calibrate_probabilities,
            threshold_confidence=threshold_confidence,
        )
        trainer.early_stopping.patience = patience
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


def plot_training_history(history, dataset_name):
    """绘制训练历史四联图：Loss / AUC / AUC-PR / F1-Score。

    Args:
        history: Trainer.history 字典，含 train_loss, val_auc, val_auc_pr, val_f1
        dataset_name: 数据集名（用于图表标题和文件名）
    """
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
    """绘制并保存混淆矩阵热力图（Normal vs Default）。

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        dataset_name: 数据集名（用于图表标题和文件名）
    """
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


def save_experiment_bundle(path, model, metrics=None):
    """保存完整的实验与部署 bundle（PyTorch .pt 格式）。

    bundle 包含：
    - 模型配置与权重（state_dict）
    - 决策阈值、温度、校准参数
    - 预处理状态（特征名/缩放器/类别映射）
    - 训练配置与评估指标标量值

    Args:
        path: 保存路径
        model: 训练好的 AABiLSTM（需附加 model_config / preprocessing_state /
               training_config 等属性）
        metrics: 评估指标字典（可选）

    Returns:
        bundle: 被保存的完整字典
    """
    bundle = {
        'format_version': 3,
        'model_class': 'AABiLSTM',
        'model_config': copy.deepcopy(
            getattr(model, 'model_config', None)
        ),
        'model_state_dict': {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        'decision_threshold': float(
            getattr(model, 'decision_threshold', 0.5)
        ),
        'temperature': float(getattr(model, 'temperature', 1.0)),
        'calibration_scale': float(
            getattr(model, 'calibration_scale', 1.0)
        ),
        'calibration_bias': float(
            getattr(model, 'calibration_bias', 0.0)
        ),
        'entropy_threshold': float(
            getattr(model, 'entropy_threshold', 0.65)
        ),
        'preprocessing_state': copy.deepcopy(
            getattr(model, 'preprocessing_state', None)
        ),
        'training_config': copy.deepcopy(
            getattr(model, 'training_config', None)
        ),
        'metrics': scalar_metrics(metrics or {}),
    }
    if not bundle['model_config']:
        raise ValueError("model.model_config is required for bundle export.")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(bundle, path)
    return bundle


# ==============================
# 7. 运行入口
# ==============================

def parse_args():
    """解析命令行参数，返回论文对齐的 AA-BiLSTM 信用风险实验配置。

    Returns:
        argparse.Namespace: 含所有实验参数的对象
    """
    parser = argparse.ArgumentParser(
        description='Paper-aligned AA-BiLSTM credit-risk experiment.'
    )
    parser.add_argument('--dataset', choices=['taiwan'], default=None)
    parser.add_argument('--data-path', default=None,
                        help='Local Taiwan .xls/.xlsx/.csv file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Default: optimized setting (50 epochs).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Default: optimized setting (256 batch size).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Default: 42.')
    parser.add_argument('--split-seed', type=int, default=42,
                        help=(
                            'Seed for stratified train/selection/calibration/'
                            'test splits.'
                        ))
    parser.add_argument(
        '--repeat-runs',
        type=int,
        default=1,
        help='Repeated model runs used to report mean, std and CI.',
    )
    parser.add_argument(
        '--vary-split-seed',
        action='store_true',
        help=(
            'Also vary the data split across repeated runs. By default only '
            'the model seed changes, separating initialization variance.'
        ),
    )
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Override the dataset-specific hidden size.')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override the dataset-specific recurrent depth.')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout in [0, 1).')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override the dataset-specific learning rate.')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Override the dataset-specific AdamW weight decay.')
    parser.add_argument(
        '--loss',
        choices=['dynamic_focal', 'weighted_ce', 'cross_entropy'],
        default='dynamic_focal',
        help='Training loss; dynamic_focal is the recommended default.',
    )
    parser.add_argument(
        '--selection-metric',
        choices=['auc', 'auc_pr', 'hybrid', 'accuracy'],
        default='auc_pr',
        help='Validation metric used for checkpoint selection and early stopping.',
    )
    parser.add_argument(
        '--threshold-objective',
        choices=['hybrid', 'f1', 'balanced_accuracy', 'accuracy'],
        default='hybrid',
        help='Validation objective used to choose the operating threshold.',
    )
    parser.add_argument(
        '--min-sensitivity',
        type=float,
        default=None,
        help=(
            'Validation sensitivity floor (default: 0.55).'
        ),
    )
    parser.add_argument(
        '--threshold-confidence',
        type=float,
        default=None,
        help=(
            'One-sided Wilson confidence used for the sensitivity floor '
            '(default: 0.80).'
        ),
    )
    parser.add_argument(
        '--auto-tune',
        action='store_true',
        help=(
            'Run a small validation-only hyperparameter search before the '
            'final fit; the test split remains untouched.'
        ),
    )
    parser.add_argument(
        '--tune-epochs',
        type=int,
        default=10,
        help='Epoch budget per lightweight tuning candidate.',
    )
    parser.add_argument(
        '--class-balance-power',
        type=float,
        default=0.5,
        help='Focal class-balance strength in [0,1]; 0.5 is less aggressive.',
    )
    parser.add_argument(
        '--focal-gamma-max',
        type=float,
        default=None,
        help='Override final focal gamma (default: 1.75).',
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=None,
        help='Override EMA decay (default: 0.995).',
    )
    parser.add_argument('--result-json', default=None,
                        help='Optional path for test metrics and audit summary.')
    parser.add_argument('--checkpoint', default=None,
                        help='Optional path for the trained PyTorch state_dict.')
    parser.add_argument(
        '--bundle',
        default=None,
        help='Optional path for a complete deployable experiment bundle.',
    )
    parser.add_argument(
        '--baselines',
        dest='run_baselines',
        action='store_true',
        default=None,
        help='Run baseline model comparison (Random Forest, XGBoost, LSTM variants).'
    )
    parser.add_argument(
        '--no-baselines',
        dest='run_baselines',
        action='store_false',
        help='Skip baseline comparison.'
    )
    parser.add_argument(
        '--ablation',
        dest='run_ablation',
        action='store_true',
        default=None,
        help='Run component ablation study.'
    )
    parser.add_argument(
        '--no-ablation',
        dest='run_ablation',
        action='store_false',
        help='Skip ablation study.'
    )
    parser.add_argument(
        '--robustness',
        dest='run_robustness',
        action='store_true',
        default=None,
        help='Run imbalance robustness study.'
    )
    parser.add_argument(
        '--no-robustness',
        dest='run_robustness',
        action='store_false',
        help='Skip robustness study.'
    )
    parser.add_argument(
        '--shap',
        dest='run_shap',
        action='store_true',
        default=None,
        help='Run SHAP explainability analysis.'
    )
    parser.add_argument(
        '--no-shap',
        dest='run_shap',
        action='store_false',
        help='Skip SHAP analysis.'
    )
    parser.add_argument(
        '--analysis-on-test',
        action='store_true',
        help=(
            'Explicitly run additional analysis on the final test split. '
            'Default analysis uses validation data to protect the holdout.'
        ),
    )
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        '--calibrate',
        dest='calibrate_probabilities',
        action='store_true',
        help='Fit cross-fitted Platt calibration (recommended).',
    )
    calibration_group.add_argument(
        '--no-calibrate',
        dest='calibrate_probabilities',
        action='store_false',
        help='Disable probability calibration.',
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        '--plots',
        dest='make_plots',
        action='store_true',
        help='Generate training and confusion-matrix plots.'
    )
    plot_group.add_argument(
        '--no-plots',
        dest='make_plots',
        action='store_false',
        help='Skip plot generation.'
    )
    parser.set_defaults(
        run_baselines=None,
        run_ablation=None,
        run_robustness=None,
        run_shap=None,
        make_plots=None,
        calibrate_probabilities=True,
    )
    return parser.parse_args()


def scalar_metrics(metrics):
    """从指标字典中提取可序列化的标量值，排除大数组（predictions/probabilities/labels）。

    Args:
        metrics: 完整的评估指标字典

    Returns:
        dict: 仅含标量值的子集，适合 JSON 序列化
    """
    result = {}
    for key, value in metrics.items():
        if key in {'predictions', 'probabilities', 'labels'}:
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def attach_repeat_summary(primary_metrics, metric_rows):
    """把多次独立运行的均值、标准差和 95% 均值置信区间附加到主结果字典。

    对 Accuracy, Sensitivity, Specificity, Balanced Accuracy, F1, AUC, AUC-PR,
    Brier Score 这 8 个核心指标分别计算统计量。

    Args:
        primary_metrics: 主结果字典（原地修改）
        metric_rows: 各次运行的指标字典列表

    Returns:
        primary_metrics（已原地修改，含 repeat_* 前缀的汇总统计量）
    """
    keys = (
        'accuracy',
        'sensitivity',
        'specificity',
        'balanced_accuracy',
        'f1',
        'auc',
        'auc_pr',
        'brier_score',
    )
    primary_metrics['repeat_runs'] = len(metric_rows)
    print("\nREPEATED-RUN SUMMARY")
    print("-" * 72)
    for key in keys:
        values = np.asarray(
            [row[key] for row in metric_rows if np.isfinite(row.get(key, np.nan))],
            dtype=float,
        )
        if len(values) == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = (
            float(1.96 * std / np.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        primary_metrics[f'repeat_{key}_mean'] = mean
        primary_metrics[f'repeat_{key}_std'] = std
        primary_metrics[f'repeat_{key}_ci95_low'] = mean - half_width
        primary_metrics[f'repeat_{key}_ci95_high'] = mean + half_width
        print(
            f"{key:<20s} {mean:.4f} +/- {std:.4f} "
            f"(mean 95% CI [{mean - half_width:.4f}, "
            f"{mean + half_width:.4f}])"
        )
    print("-" * 72)
    return primary_metrics


# ============================================================
# 命令行入口：交互模式或参数模式
# ============================================================
if __name__ == '__main__':
    set_seed(42)  # 模块导入时不设种子，仅在直接运行时设定
    args = parse_args()

    dataset_name = args.dataset
    if dataset_name is None:
        print("=" * 40)
        print("Using Taiwan Credit dataset")
        dataset_name = "taiwan"

    run_baselines = args.run_baselines
    run_ablation = args.run_ablation
    run_robustness = args.run_robustness
    run_shap = args.run_shap
    if any(x is None for x in [run_baselines, run_ablation, run_robustness, run_shap]):
        if args.dataset is not None:
            run_baselines = False if run_baselines is None else run_baselines
            run_ablation = False if run_ablation is None else run_ablation
            run_robustness = False if run_robustness is None else run_robustness
            run_shap = False if run_shap is None else run_shap
        else:
            while True:
                ans = input("Run baseline comparison? (Y/N): ").strip().upper()
                if ans in {'Y', 'N'}:
                    run_baselines = (ans == 'Y')
                    break
                print("Invalid input. Enter Y or N.")
            while True:
                ans = input("Run ablation study? (Y/N): ").strip().upper()
                if ans in {'Y', 'N'}:
                    run_ablation = (ans == 'Y')
                    break
                print("Invalid input. Enter Y or N.")
            while True:
                ans = input("Run robustness study? (Y/N): ").strip().upper()
                if ans in {'Y', 'N'}:
                    run_robustness = (ans == 'Y')
                    break
                print("Invalid input. Enter Y or N.")
            while True:
                ans = input("Run SHAP analysis? (Y/N): ").strip().upper()
                if ans in {'Y', 'N'}:
                    run_shap = (ans == 'Y')
                    break
                print("Invalid input. Enter Y or N.")

    make_plots = args.make_plots
    if make_plots is None:
        if args.dataset is not None:
            make_plots = False
        else:
            while True:
                ans = input("Generate plots? (Y/N): ").strip().upper()
                if ans in {'Y', 'N'}:
                    make_plots = (ans == 'Y')
                    break
                print("Invalid input. Enter Y or N.")

    print("Starting experiments...")

    if args.repeat_runs < 1:
        raise ValueError("--repeat-runs must be at least 1.")
    experiment_kwargs = dict(
        dataset_name=dataset_name,
        epoch=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        run_baselines=run_baselines,
        run_ablation=run_ablation,
        run_robustness=run_robustness,
        run_shap=run_shap,
        make_plots=make_plots,
        threshold_min_sensitivity=args.min_sensitivity,
        seed=args.seed,
        split_seed=args.split_seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        selection_metric=args.selection_metric,
        threshold_objective=args.threshold_objective,
        class_balance_power=args.class_balance_power,
        focal_gamma_max=args.focal_gamma_max,
        ema_decay=args.ema_decay,
        calibrate_probabilities=args.calibrate_probabilities,
        analysis_on_test=args.analysis_on_test,
        threshold_confidence=args.threshold_confidence,
        auto_tune=args.auto_tune,
        tune_epochs=args.tune_epochs,
    )
    model, metrics, history = run_experiment(**experiment_kwargs)

    repeated_metrics = [metrics]
    # 多次独立运行：固定数据分割，仅改变模型 seed，分离初始化方差；
    # 若 --vary-split-seed 则同时改变分割，评估分割稳定性。
    if args.repeat_runs > 1:
        base_model_seed = (
            args.seed
            if args.seed is not None
            else 42
        )
        for repeat_idx in range(1, args.repeat_runs):
            print(
                f"\nStarting repeated run {repeat_idx + 1}/"
                f"{args.repeat_runs}..."
            )
            repeated_kwargs = dict(experiment_kwargs)
            repeated_kwargs.update({
                # 1009 是一个素数，避免多次运行 seed 产生周期性重复模式
                'seed': base_model_seed + 1009 * repeat_idx,
                'split_seed': (
                    args.split_seed + 1009 * repeat_idx
                    if args.vary_split_seed else args.split_seed
                ),
                'run_baselines': False,
                'run_ablation': False,
                'run_robustness': False,
                'run_shap': False,
                'make_plots': False,
                'analysis_on_test': False,
                'auto_tune': False,
                'hidden_dim': metrics['hidden_dim'],
                'num_layers': metrics['num_layers'],
                'dropout': metrics['dropout'],
                'lr': metrics['learning_rate'],
            })
            _, repeat_metrics, _ = run_experiment(**repeated_kwargs)
            repeated_metrics.append(repeat_metrics)
        attach_repeat_summary(metrics, repeated_metrics)

    if make_plots:
        plot_confusion_matrix(metrics['labels'], metrics['predictions'], dataset_name)

    if args.result_json:
        result_dir = os.path.dirname(os.path.abspath(args.result_json))
        os.makedirs(result_dir, exist_ok=True)
        result_payload = scalar_metrics(metrics)
        with open(args.result_json, 'w', encoding='utf-8') as result_file:
            json.dump(
                result_payload,
                result_file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved metrics to {args.result_json}")

    if args.checkpoint:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved model checkpoint to {args.checkpoint}")

    if args.bundle:
        save_experiment_bundle(args.bundle, model, metrics)
        print(f"Saved complete experiment bundle to {args.bundle}")

    print("\nExperiment completed successfully!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final AUC: {metrics['auc']:.4f}")
    print(f"Final Sensitivity: {metrics['sensitivity']:.4f}")