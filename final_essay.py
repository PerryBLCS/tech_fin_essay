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
from sklearn.linear_model import LogisticRegression
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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statistics import NormalDist
import warnings

warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
)

GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
TAIWAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


# 设置随机种子
def set_seed(seed, deterministic=True):
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


# ==============================
# 1. 数据加载与预处理
# ==============================

class CreditDataLoader:
    """
    处理 German Credit 和 Taiwan Credit 数据集
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
        self.audit_groups = {}
        self.audit_group_splits_ = {}

        # 原始静态字段会保留到数据切分之后；one-hot 词表只能用训练集拟合。
        self._raw_static_numeric = None
        self._raw_static_numeric_names = None
        self._raw_static_categorical = None
        self._raw_static_categorical_names = None
        self._raw_static_prefixes = None

        # German 伪时序的类别编码和频率也必须只由训练集确定。
        self._german_raw_temporal = None
        self._german_selected_cols = None
        self._german_risk_priors = None
        self.german_temporal_maps_ = None

    def load_german_credit(self, filepath=None):
        """
        German Credit Dataset (1000样本, 20特征, 30%违约率)
        将6个类别属性编码为伪时序数据
        """
        # 如果没有提供文件路径, 使用 UCI 在线数据
        self.dataset_name = 'german'
        self.temporal_categorical_indices_ = []
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
        self._raw_static_numeric = numeric_df.to_numpy(dtype=np.float32)
        self._raw_static_numeric_names = [str(c) for c in numeric_cols]
        self._raw_static_categorical = (
            df[static_categorical_cols].astype(str).to_numpy()
        )
        self._raw_static_categorical_names = [
            str(c) for c in static_categorical_cols
        ]
        self._raw_static_prefixes = [
            f"A{col}" for col in static_categorical_cols
        ]

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
        self._german_selected_cols = list(selected_cols)
        self._german_risk_priors = copy.deepcopy(risk_priors)
        self._german_raw_temporal = np.stack(
            [df[col].astype(str).to_numpy() for col in selected_cols],
            axis=1,
        )

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
        self.audit_groups = {
            'sex': df['SEX'].astype(str).to_numpy(),
            'age_band': pd.cut(
                df['AGE'].astype(float),
                bins=[-np.inf, 29, 39, 49, 59, np.inf],
                labels=['<30', '30-39', '40-49', '50-59', '60+'],
            ).astype(str).to_numpy(),
        }

        static_df = pd.concat(
            [base_static_df.reset_index(drop=True), engineered_static.reset_index(drop=True)],
            axis=1
        )
        static_features = static_df.astype(np.float32).values
        self.static_feature_names = [str(c) for c in static_df.columns]

        # 原始字段顺序为“最近月 -> 最早月”；统一反转为“最早月 -> 最近月”，
        # 使局部因果窗口和趋势方向符合真实时间顺序。
        # 金额是强长尾变量；有符号 log1p 保留方向并减少极端值的影响。
        bill_signal = np.sign(bill_values) * np.log1p(np.abs(bill_values))
        payment_signal = np.sign(payment_values) * np.log1p(
            np.abs(payment_values)
        )
        temporal_features = np.stack([
            status_values,
            bill_signal,
            payment_signal,
            utilization,
            payment_to_bill,
        ], axis=-1)[:, ::-1, :].copy().astype(np.float32)  # [N, 6, 5]
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

    def _encode_static_from_training(self, train_idx, *evaluation_indices):
        """使用训练集类别词表编码三个数据子集。"""
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

    def _encode_german_temporal_from_training(
        self,
        train_idx,
        *evaluation_indices,
    ):
        """训练集拟合 German 类别代码与频率，避免分布信息泄漏。"""
        if self._german_raw_temporal is None:
            return None

        raw = self._german_raw_temporal
        maps = []
        for step, col in enumerate(self._german_selected_cols):
            train_values = raw[train_idx, step]
            categories = sorted(np.unique(train_values).tolist())
            denominator = max(len(categories) - 1, 1)
            code_map = {
                category: idx / denominator
                for idx, category in enumerate(categories)
            }
            counts = pd.Series(train_values).value_counts(normalize=True)
            frequency_map = {
                str(category): float(value)
                for category, value in counts.items()
            }
            maps.append({
                'column': int(col),
                'code': code_map,
                'frequency': frequency_map,
            })
        self.german_temporal_maps_ = maps

        def encode(indices):
            encoded_steps = []
            for step, col in enumerate(self._german_selected_cols):
                values = raw[indices, step]
                code_map = maps[step]['code']
                frequency_map = maps[step]['frequency']
                priors = self._german_risk_priors.get(col, {})
                category_code = np.asarray(
                    [code_map.get(value, 0.5) for value in values],
                    dtype=np.float32,
                )
                risk_prior = np.asarray(
                    [
                        priors.get(value, code_map.get(value, 0.5))
                        for value in values
                    ],
                    dtype=np.float32,
                )
                category_frequency = np.asarray(
                    [frequency_map.get(value, 0.0) for value in values],
                    dtype=np.float32,
                )
                encoded_steps.append(np.stack(
                    [category_code, risk_prior, category_frequency],
                    axis=1,
                ))
            return np.stack(encoded_steps, axis=1)

        return tuple(
            encode(idx) for idx in (train_idx, *evaluation_indices)
        )

    @staticmethod
    def _scaler_state(scaler):
        if not hasattr(scaler, 'mean_'):
            return None
        return {
            'mean': np.asarray(scaler.mean_).tolist(),
            'scale': np.asarray(scaler.scale_).tolist(),
            'var': np.asarray(scaler.var_).tolist(),
            'n_features_in': int(scaler.n_features_in_),
            'n_samples_seen': np.asarray(scaler.n_samples_seen_).tolist(),
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
            'german_temporal_maps': copy.deepcopy(
                self.german_temporal_maps_
            ),
            'german_selected_cols': copy.deepcopy(
                self._german_selected_cols
            ),
            'german_risk_priors': copy.deepcopy(
                self._german_risk_priors
            ),
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
        数据预处理: 标准化、分割
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

        # 先分割样本索引，再拟合类别词表、频率和 scaler。
        all_indices = np.arange(len(y))
        development_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        heldout_fraction = val_size + calibration_size
        train_idx, operating_idx = train_test_split(
            development_idx,
            test_size=heldout_fraction / (1.0 - test_size),
            random_state=split_seed + 1,
            stratify=y[development_idx],
        )
        calibration_fraction = calibration_size / heldout_fraction
        val_idx, calibration_idx = train_test_split(
            operating_idx,
            test_size=calibration_fraction,
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

        encoded_german = self._encode_german_temporal_from_training(
            train_idx,
            val_idx,
            calibration_idx,
            test_idx,
        )
        if encoded_german is None:
            X_temporal_train = temporal_features[train_idx].copy()
            X_temporal_val = temporal_features[val_idx].copy()
            X_temporal_calibration = temporal_features[
                calibration_idx
            ].copy()
            X_temporal_test = temporal_features[test_idx].copy()
        else:
            (
                X_temporal_train,
                X_temporal_val,
                X_temporal_calibration,
                X_temporal_test,
            ) = encoded_german

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_calibration = y[calibration_idx]
        y_test = y[test_idx]
        self.audit_group_splits_ = {
            split: {
                name: np.asarray(values)[indices]
                for name, values in self.audit_groups.items()
            }
            for split, indices in (
                ('train', train_idx),
                ('validation', val_idx),
                ('calibration', calibration_idx),
                ('test', test_idx),
            )
        }

        binary_static_cols = np.all((X_static_train == 0) | (X_static_train == 1), axis=0)
        continuous_static_cols = ~binary_static_cols
        self.continuous_static_cols_ = continuous_static_cols.copy()
        if np.any(continuous_static_cols):
            X_static_train = X_static_train.copy()
            X_static_val = X_static_val.copy()
            X_static_calibration = X_static_calibration.copy()
            X_static_test = X_static_test.copy()
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
        continuous_temporal_cols = np.ones(n_features, dtype=bool)
        for categorical_index in self.temporal_categorical_indices_:
            if not 0 <= categorical_index < n_features:
                raise ValueError(
                    "Temporal categorical feature index is out of range."
                )
            continuous_temporal_cols[categorical_index] = False
        self.continuous_temporal_cols_ = continuous_temporal_cols.copy()

        def scale_temporal(x, fit=False):
            result = x.astype(np.float32, copy=True)
            if not np.any(continuous_temporal_cols):
                return result
            reshaped = result[:, :, continuous_temporal_cols].reshape(
                -1,
                int(continuous_temporal_cols.sum()),
            )
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


class BlackSwanMonitor:
    """
    与分类模型解耦的鲁棒分布外监测器。

    监测器仅使用训练集拟合特征中位数和 MAD，并在全局状态、波动、趋势
    与最大相邻突变描述符上计算异常分数。它不把未知样本强制归为违约，
    而是为人工复核或保守策略提供 OOD 标记。
    """

    def __init__(self, quantile=0.99, top_fraction=0.10, eps=1e-6):
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1.0.")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        self.quantile = float(quantile)
        self.top_fraction = float(top_fraction)
        self.eps = float(eps)
        self.center_ = None
        self.scale_ = None
        self.threshold_ = None

    @staticmethod
    def _descriptors(static_features, temporal_features):
        static_features = np.asarray(static_features, dtype=np.float32)
        temporal_features = np.asarray(temporal_features, dtype=np.float32)
        if static_features.ndim != 2:
            raise ValueError("static_features must have shape [N, S].")
        if temporal_features.ndim != 3:
            raise ValueError("temporal_features must have shape [N, T, F].")
        if static_features.shape[0] != temporal_features.shape[0]:
            raise ValueError("Static and temporal sample counts must match.")

        mean = temporal_features.mean(axis=1)
        std = temporal_features.std(axis=1)
        maximum = temporal_features.max(axis=1)
        minimum = temporal_features.min(axis=1)
        last = temporal_features[:, -1]
        trend = temporal_features[:, -1] - temporal_features[:, 0]
        if temporal_features.shape[1] > 1:
            max_abs_delta = np.abs(
                np.diff(temporal_features, axis=1)
            ).max(axis=1)
        else:
            max_abs_delta = np.zeros_like(last)

        return np.concatenate([
            static_features,
            mean,
            std,
            maximum,
            minimum,
            last,
            trend,
            max_abs_delta
        ], axis=1)

    def _raw_scores(self, descriptors):
        robust_z = np.abs(
            (descriptors - self.center_) / self.scale_
        )
        top_k = max(
            1,
            int(np.ceil(robust_z.shape[1] * self.top_fraction))
        )
        partitioned = np.partition(
            robust_z, robust_z.shape[1] - top_k, axis=1
        )
        return partitioned[:, -top_k:].mean(axis=1)

    def fit(self, static_features, temporal_features):
        descriptors = self._descriptors(
            static_features, temporal_features
        )
        self.center_ = np.median(descriptors, axis=0)
        mad = np.median(
            np.abs(descriptors - self.center_), axis=0
        )
        robust_scale = 1.4826 * mad
        fallback_scale = descriptors.std(axis=0)
        self.scale_ = np.where(
            robust_scale > self.eps,
            robust_scale,
            np.where(fallback_scale > self.eps, fallback_scale, 1.0)
        )
        train_scores = self._raw_scores(descriptors)
        self.threshold_ = float(
            np.quantile(train_scores, self.quantile)
        )
        return self

    def score(self, static_features, temporal_features):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("BlackSwanMonitor must be fitted first.")
        descriptors = self._descriptors(
            static_features, temporal_features
        )
        scores = self._raw_scores(descriptors)
        flags = scores > self.threshold_
        return scores.astype(np.float32), flags

    def calibrate_threshold(self, static_features, temporal_features):
        """固定训练中心/尺度，仅用验证集校准 ID 假阳性分位点。"""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("BlackSwanMonitor must be fitted first.")
        descriptors = self._descriptors(
            static_features,
            temporal_features,
        )
        calibration_scores = self._raw_scores(descriptors)
        self.threshold_ = float(
            np.quantile(calibration_scores, self.quantile)
        )
        return self

    def to_dict(self):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("BlackSwanMonitor must be fitted first.")
        return {
            'quantile': self.quantile,
            'top_fraction': self.top_fraction,
            'eps': self.eps,
            'center': self.center_.tolist(),
            'scale': self.scale_.tolist(),
            'threshold': self.threshold_
        }

    @classmethod
    def from_dict(cls, state):
        monitor = cls(
            quantile=state['quantile'],
            top_fraction=state['top_fraction'],
            eps=state.get('eps', 1e-6)
        )
        monitor.center_ = np.asarray(state['center'], dtype=np.float32)
        monitor.scale_ = np.asarray(state['scale'], dtype=np.float32)
        monitor.threshold_ = float(state['threshold'])
        return monitor


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
        if not np.isfinite(val_auc):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return
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
            # 0.5 次幂比完全逆频率更温和：通常可减少假阳性、改善 Accuracy，
            # 再由验证集阈值保证敏感性下限。
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
        """从训练数据集读取全局类别数，不消耗随机 DataLoader。"""
        labels = getattr(loader.dataset, 'labels', None)
        if labels is None:
            all_labels = []
            for batch in loader:
                all_labels.extend(batch['label'].cpu().numpy())
            labels = np.asarray(all_labels)
        elif torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
        if len(counts) != 2 or np.any(counts == 0):
            raise ValueError(
                f"Training data must contain both binary classes; got {counts.tolist()}."
            )
        return counts.astype(np.float64)

    @classmethod
    def _compute_class_weights(cls, loader):
        """计算全局 inverse-frequency 类别权重。"""
        counts = cls._compute_class_counts(loader)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 2  # 归一化
        return torch.FloatTensor(weights)

    def _update_ema(self):
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

    def fit_temperature(self):
        """在验证集上以 NLL 拟合单一温度，不改变概率排序。"""
        logits, labels = self._collect_logits(self.val_loader)
        candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))

        def nll_at(temperature):
            return float(F.cross_entropy(
                logits / float(temperature),
                labels,
            ).item())

        losses = np.asarray([nll_at(value) for value in candidates])
        best_idx = int(np.argmin(losses))
        best_temperature = float(candidates[best_idx])

        lower = candidates[max(best_idx - 1, 0)]
        upper = candidates[min(best_idx + 1, len(candidates) - 1)]
        refined = np.linspace(lower, upper, 41)
        refined_losses = np.asarray([nll_at(value) for value in refined])
        refined_idx = int(np.argmin(refined_losses))
        best_temperature = float(refined[refined_idx])
        return best_temperature, {
            'nll_before': nll_at(1.0),
            'nll_after': float(refined_losses[refined_idx]),
        }

    @staticmethod
    def _fit_platt_parameters(margins, labels):
        margins = np.asarray(margins, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64)
        if len(np.unique(labels)) != 2:
            return 1.0, 0.0
        calibrator = LogisticRegression(
            C=10.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        calibrator.fit(margins, labels)
        scale = float(calibrator.coef_[0, 0])
        bias = float(calibrator.intercept_[0])
        if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0.0:
            return 1.0, 0.0
        return scale, bias

    @staticmethod
    def _apply_platt(margins, scale, bias):
        calibrated_logit = np.clip(
            float(scale) * np.asarray(margins, dtype=np.float64)
            + float(bias),
            -40.0,
            40.0,
        )
        return 1.0 / (1.0 + np.exp(-calibrated_logit))

    def fit_platt_scaling(self):
        """
        Fit affine log-odds calibration on the dedicated operating split.

        Out-of-fold probabilities estimate calibration generalization.
        Operating-threshold search uses the uncalibrated margin ranking and
        is mapped through the final monotonic calibrator afterwards; this
        avoids fold-specific calibrators scrambling the global ranking.
        """
        logits, label_tensor = self._collect_logits(
            self.calibration_loader
        )
        labels = label_tensor.numpy().astype(np.int64)
        margins = (logits[:, 1] - logits[:, 0]).numpy()
        raw_probabilities = self._apply_platt(margins, 1.0, 0.0)
        class_counts = np.bincount(labels, minlength=2)
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

    @staticmethod
    def _expected_calibration_error(labels, probabilities, n_bins=15):
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.minimum(
            np.digitize(probabilities, edges[1:-1], right=False),
            n_bins - 1,
        )
        result = 0.0
        for bin_idx in range(n_bins):
            mask = bin_ids == bin_idx
            if not np.any(mask):
                continue
            result += (
                float(mask.mean())
                * abs(
                    float(probabilities[mask].mean())
                    - float(labels[mask].mean())
                )
            )
        return float(result)

    def train_epoch(self, epoch):
        """训练一个 epoch（混合精度 + 梯度裁剪）"""
        self.model.train()
        if isinstance(self.criterion, DynamicFocalLoss):
            self.criterion.set_epoch(epoch)

        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            static = batch['static'].to(device, non_blocking=True)
            temporal = batch['temporal'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播
            with torch.autocast(
                device_type=device.type,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(static, temporal)
                loss = self.criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss detected. Check input scaling "
                    "and the model's numerical operations."
                )

            # 混合精度反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪（防止梯度爆炸）
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite gradients detected before the optimizer step."
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()

            batch_samples = labels.shape[0]
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
        """评估模型，并返回排序、分类和概率校准指标。"""
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
            fpr, tpr, _ = roc_curve(flat_labels, flat_probs)
            ks = float(np.max(tpr - fpr))
        except ValueError:
            auc = np.nan
            ks = np.nan
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
        mcc = matthews_corrcoef(flat_labels, flat_preds)
        probability_log_loss = log_loss(
            flat_labels,
            np.column_stack([1.0 - clipped_probs, clipped_probs]),
            labels=[0, 1],
        )
        brier = brier_score_loss(flat_labels, clipped_probs)
        ece = self._expected_calibration_error(
            flat_labels,
            clipped_probs,
        )
        top_count = max(1, int(np.ceil(len(flat_labels) * 0.10)))
        top_indices = np.argsort(flat_probs)[-top_count:]
        base_rate = max(float(np.mean(flat_labels)), 1e-12)
        lift_at_10 = float(np.mean(flat_labels[top_indices]) / base_rate)

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
            'mcc': float(mcc),
            'ks': float(ks),
            'lift_at_10': float(lift_at_10),
            'log_loss': float(probability_log_loss),
            'brier_score': float(brier),
            'ece': float(ece),
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
        if total <= 0:
            return 0.0
        proportion = successes / total
        z_value = NormalDist().inv_cdf(confidence)
        denominator = 1.0 + z_value ** 2 / total
        centre = proportion + z_value ** 2 / (2.0 * total)
        radius = z_value * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value ** 2 / (4.0 * total ** 2)
        )
        return float((centre - radius) / denominator)

    def find_best_threshold(
        self,
        min_sensitivity=0.40,
        labels=None,
        probabilities=None,
    ):
        """仅在验证集上选择概率阈值"""
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

        unique_probabilities = np.unique(probabilities)
        if len(unique_probabilities) > 5000:
            unique_probabilities = np.quantile(
                unique_probabilities,
                np.linspace(0.0, 1.0, 5000),
            )
        if len(unique_probabilities) == 1:
            candidates = np.asarray([0.5], dtype=float)
        else:
            candidates = (
                unique_probabilities[:-1] + unique_probabilities[1:]
            ) / 2.0
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

        (
            objective,
            balanced_accuracy,
            accuracy,
            sensitivity,
            sensitivity_lower,
            f1,
            _,
            threshold,
        ) = max(scored)
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
        if self.ema_state is not None:
            self.model.load_state_dict(self.ema_state)
            ema_metrics = self.evaluate(
                self.val_loader,
                threshold=0.5,
                temperature=1.0,
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
                operating_probabilities,
                calibration_metrics,
            ) = self.fit_platt_scaling()
        else:
            operating_logits, operating_label_tensor = (
                self._collect_logits(self.calibration_loader)
            )
            operating_labels = operating_label_tensor.numpy()
            operating_margins = (
                operating_logits[:, 1] - operating_logits[:, 0]
            ).numpy()
            operating_probabilities = self._apply_platt(
                operating_margins,
                1.0,
                0.0,
            )
            self.calibration_scale = 1.0
            self.calibration_bias = 0.0
        self.temperature = 1.0 / max(self.calibration_scale, 1e-7)

        raw_decision_threshold, validation_threshold_metrics = self.find_best_threshold(
            min_sensitivity=self.threshold_min_sensitivity,
            labels=operating_labels,
            probabilities=operating_probabilities,
        )
        clipped_raw_threshold = float(np.clip(
            raw_decision_threshold,
            1e-7,
            1.0 - 1e-7,
        ))
        raw_threshold_logit = np.log(
            clipped_raw_threshold / (1.0 - clipped_raw_threshold)
        )
        self.decision_threshold = float(self._apply_platt(
            np.asarray([raw_threshold_logit]),
            self.calibration_scale,
            self.calibration_bias,
        )[0])
        operating_margin = np.log(
            np.clip(operating_probabilities, 1e-7, 1.0 - 1e-7)
            / np.clip(
                1.0 - operating_probabilities,
                1e-7,
                1.0,
            )
        )
        calibrated_operating_probabilities = self._apply_platt(
            operating_margin,
            self.calibration_scale,
            self.calibration_bias,
        )
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

        # The untouched test set is evaluated once with the validation cutoff.
        test_metrics = self.evaluate(self.test_loader, threshold=self.decision_threshold)
        test_metrics['validation_threshold_accuracy'] = validation_threshold_metrics['accuracy']
        test_metrics['validation_threshold_sensitivity'] = validation_threshold_metrics['sensitivity']
        test_metrics['validation_threshold_specificity'] = validation_threshold_metrics['specificity']
        test_metrics['validation_threshold_objective'] = validation_threshold_metrics['objective']
        test_metrics['operating_threshold_accuracy'] = (
            validation_threshold_metrics['accuracy']
        )
        test_metrics['operating_threshold_sensitivity'] = (
            validation_threshold_metrics['sensitivity']
        )
        test_metrics['operating_threshold_specificity'] = (
            validation_threshold_metrics['specificity']
        )
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


def specificity_score(y_true, y_pred):
    """计算特异性"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


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
    with torch.inference_mode():
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

    def forward(self, *inputs):
        """兼容 SHAP 的列表输入和多位置参数输入。"""
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            static_feat, temporal_feat = inputs[0]
        elif len(inputs) == 2:
            static_feat, temporal_feat = inputs
        else:
            raise ValueError("Expected static and temporal model inputs.")
        return self.model(static_feat, temporal_feat, return_attention=False)


class Explainer:
    """SHAP 可解释性分析"""

    def __init__(self, model, feature_names=None):
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
        """使用 SHAP 解释模型预测"""
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

        # 过滤 SHAP 对 LayerNorm 的警告
        warnings.filterwarnings(
            'ignore',
            message='.*unrecognized nn.Module.*',
            category=UserWarning,
        )

        try:
            # 尝试使用 DeepExplainer
            print("Attempting SHAP DeepExplainer...")
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

            with torch.inference_mode():
                output = wrapped_model([static_part, temporal_part])
                probs = torch.softmax(output, dim=-1)
            return probs[:, 1].cpu().numpy()

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
        shap_values = explainer.shap_values(test_flat, nsamples=100)

        # 重构 SHAP 值格式以匹配双输入
        static_shap = shap_values[:, :static_data.shape[1]]
        temporal_shap = shap_values[:, static_data.shape[1]:].reshape(
            explain_size,
            temporal_data.shape[1],
            temporal_data.shape[2],
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


def summarize_fusion_diagnostics(model, data_loader, X_static_test,
                                 X_temporal_test, sample_size=256):
    """汇总全局门、局部门、局部窗口权重与时间步突变强度。"""
    if not hasattr(model, 'forward'):
        return None

    n_samples = min(sample_size, len(X_static_test))
    if n_samples <= 0:
        return None

    model.eval()
    try:
        with torch.inference_mode():
            static = torch.FloatTensor(X_static_test[:n_samples]).to(device)
            temporal = torch.FloatTensor(X_temporal_test[:n_samples]).to(device)
            _, diagnostics = model(static, temporal, return_attention=True)
    except Exception as e:
        print(f"Fusion diagnostics skipped: {e}")
        return None

    if diagnostics is None:
        print("Fusion diagnostics skipped: model returned no diagnostics.")
        return None

    # 兼容 legacy_cross 返回的字典。
    if 'legacy_attention' in diagnostics:
        weights = diagnostics['legacy_attention'].detach().cpu().numpy()
        avg_weights = weights.mean(axis=0).reshape(-1)
        names = data_loader.temporal_step_names or [
            f't{i}' for i in range(X_temporal_test.shape[1])
        ]
        if len(avg_weights) > len(names):
            names = list(names) + ['multi_scale_context']
        print("\nLegacy TS-CrossAttention context weights:")
        for idx in np.argsort(avg_weights)[::-1][:min(10, len(avg_weights))]:
            print(f"{names[idx]}: {avg_weights[idx]:.6f}")
        return {'legacy_attention': avg_weights}

    result = {}
    local_attention = diagnostics.get('local_attention')
    if local_attention is not None:
        weights = local_attention.detach().cpu().numpy()
        mean_by_lag = weights.mean(axis=(0, 1))
        window_size = len(mean_by_lag)
        lag_names = [
            f't-{window_size - 1 - idx}'
            if idx < window_size - 1 else 'current'
            for idx in range(window_size)
        ]
        print("\nMean local shock-attention weight by lag:")
        for name, value in zip(lag_names, mean_by_lag):
            print(f"{name:>8s}: {value:.6f}")
        result['local_attention_by_lag'] = mean_by_lag

    for key, label in (
        ('local_gate', 'Local gate'),
        ('global_gate', 'Global gate')
    ):
        value = diagnostics.get(key)
        if value is not None:
            mean_value = float(value.mean().item())
            print(f"{label} mean activation: {mean_value:.6f}")
            result[f'{key}_mean'] = mean_value

    shock_score = diagnostics.get('shock_score')
    if shock_score is not None:
        mean_shock = shock_score.mean(dim=0).detach().cpu().numpy()
        names = data_loader.temporal_step_names or [
            f't{i}' for i in range(len(mean_shock))
        ]
        top_idx = np.argsort(mean_shock)[::-1][
            :min(5, len(mean_shock))
        ]
        print("Top average temporal shock positions:")
        for idx in top_idx:
            print(f"{names[idx]}: {mean_shock[idx]:.6f}")
        result['mean_shock_by_step'] = mean_shock

    return result


def summarize_cross_attention(model, data_loader, X_static_test,
                              X_temporal_test, sample_size=256):
    """向后兼容旧函数名。"""
    return summarize_fusion_diagnostics(
        model,
        data_loader,
        X_static_test,
        X_temporal_test,
        sample_size=sample_size
    )


def summarize_selective_risk(labels, predictions, ood_scores, ood_flags):
    """计算拒判覆盖率与风险-覆盖曲线摘要。"""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(ood_scores)
    flags = np.asarray(ood_flags, dtype=bool)
    accepted = ~flags
    errors = (predictions != labels).astype(np.float64)
    order = np.argsort(scores)
    cumulative_risk = (
        np.cumsum(errors[order])
        / np.arange(1, len(errors) + 1)
    )
    result = {
        'selective_coverage': float(accepted.mean()),
        'selective_risk': (
            float(errors[accepted].mean()) if np.any(accepted) else np.nan
        ),
        'risk_coverage_auc': float(cumulative_risk.mean()),
    }
    if np.any(accepted):
        result['selective_accuracy'] = float(
            accuracy_score(labels[accepted], predictions[accepted])
        )
        result['selective_sensitivity'] = float(
            recall_score(
                labels[accepted],
                predictions[accepted],
                zero_division=0,
            )
        )
    else:
        result['selective_accuracy'] = np.nan
        result['selective_sensitivity'] = np.nan
    return result


def summarize_group_fairness(labels, predictions, groups, min_group_size=20):
    """输出审计用途的组间 TPR/FPR/Accuracy 差异，不参与模型训练。"""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    summaries = {}
    scalar_gaps = {}
    for group_name, group_values in (groups or {}).items():
        group_values = np.asarray(group_values)
        rows = {}
        for value in np.unique(group_values):
            mask = group_values == value
            if int(mask.sum()) < min_group_size:
                continue
            tn, fp, fn, tp = confusion_matrix(
                labels[mask],
                predictions[mask],
                labels=[0, 1],
            ).ravel()
            rows[str(value)] = {
                'n': int(mask.sum()),
                'accuracy': float(
                    accuracy_score(labels[mask], predictions[mask])
                ),
                'tpr': float(tp / max(tp + fn, 1)),
                'fpr': float(fp / max(fp + tn, 1)),
                'positive_prediction_rate': float(
                    predictions[mask].mean()
                ),
            }
        if len(rows) < 2:
            continue
        summaries[group_name] = rows
        for metric in ('accuracy', 'tpr', 'fpr', 'positive_prediction_rate'):
            values = [row[metric] for row in rows.values()]
            scalar_gaps[f'{group_name}_{metric}_gap'] = float(
                max(values) - min(values)
            )
    return summaries, scalar_gaps


def bootstrap_metric_intervals(
    labels,
    predictions,
    probabilities,
    n_bootstrap=500,
    random_state=42,
):
    """为核心测试指标提供非参数 95% 置信区间。"""
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

    result = {}
    for name, samples in values.items():
        if not samples:
            continue
        lower, upper = np.quantile(samples, [0.025, 0.975])
        result[f'{name}_ci95_low'] = float(lower)
        result[f'{name}_ci95_high'] = float(upper)
    return result


# ==============================
# 6. 主程序
# ==============================

def run_experiment(dataset_name='german', epoch=None, batch_size=None, data_path=None,
                   run_analysis=True, make_plots=True,
                   threshold_min_sensitivity=None, ood_quantile=0.99,
                   fusion_type='global_local', seed=None,
                   split_seed=42,
                   hidden_dim=None, num_layers=None, num_heads=None,
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
    运行完整实验
    Args:
        dataset_name: 'german' 或 'taiwan'
        epoch: 训练轮数
        batch_size: 批次大小
        data_path: Taiwan 数据集路径，可传入 .xls/.xlsx/.csv
        ood_quantile: 训练分布异常分数阈值分位数
        fusion_type: 'global_local'（推荐）或 'legacy_cross'
        loss_name: 'dynamic_focal'、'weighted_ce' 或 'cross_entropy'
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in {'german', 'taiwan'}:
        raise ValueError("dataset_name must be 'german' or 'taiwan'.")
    if loss_name not in {'dynamic_focal', 'weighted_ce', 'cross_entropy'}:
        raise ValueError(
            "loss_name must be 'dynamic_focal', 'weighted_ce' or 'cross_entropy'."
        )
    if threshold_min_sensitivity is None:
        threshold_min_sensitivity = (
            0.5 if dataset_name == 'german' else 0.55
        )
    if not 0.0 <= threshold_min_sensitivity <= 1.0:
        raise ValueError("threshold_min_sensitivity must be in [0, 1].")
    if threshold_confidence is None:
        threshold_confidence = (
            0.60 if dataset_name == 'german' else 0.80
        )
    if not 0.5 <= threshold_confidence < 1.0:
        raise ValueError("threshold_confidence must be in [0.5, 1).")
    if tune_epochs < 1:
        raise ValueError("tune_epochs must be at least 1.")

    print(f"\n{'=' * 60}")
    print(f"Running AA-BiLSTM Experiment on {dataset_name.upper()} Dataset")
    print(f"{'=' * 60}")
    print(f"Using device: {device}")
    seed = (7 if dataset_name == 'german' else 42) if seed is None else seed
    set_seed(seed)

    # 1. 数据加载
    data_loader = CreditDataLoader()

    if dataset_name == 'german':
        static_feat, temporal_feat, y = data_loader.load_german_credit(data_path)
        batch_size = 32 if batch_size is None else batch_size
        epoch = 30 if epoch is None else epoch
        hidden_dim = 64 if hidden_dim is None else hidden_dim
        num_layers = 2 if num_layers is None else num_layers
        num_heads = 4 if num_heads is None else num_heads
        dropout = 0.25 if dropout is None else dropout
        lr = 8e-4 if lr is None else lr
        weight_decay = 2e-4 if weight_decay is None else weight_decay
        focal_gamma_max = (
            1.5 if focal_gamma_max is None else focal_gamma_max
        )
        ema_decay = 0.99 if ema_decay is None else ema_decay
        patience = 12
    else:
        static_feat, temporal_feat, y = data_loader.load_taiwan_credit(data_path)
        batch_size = 256 if batch_size is None else batch_size
        epoch = 50 if epoch is None else epoch
        hidden_dim = 96 if hidden_dim is None else hidden_dim
        num_layers = 3 if num_layers is None else num_layers
        num_heads = 4 if num_heads is None else num_heads
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
        or num_heads < 1
    ):
        raise ValueError(
            "batch_size, epoch, num_layers and num_heads must be positive; "
            "hidden_dim must be at least 2."
        )
    if fusion_type == 'legacy_cross' and hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")
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

    # 只用训练集拟合鲁棒分布边界，避免验证/测试信息泄漏。
    black_swan_monitor = BlackSwanMonitor(
        quantile=ood_quantile
    ).fit(
        X_static_train,
        X_temporal_train,
    ).calibrate_threshold(
        X_static_calibration,
        X_temporal_calibration,
    )

    # 3. 创建 DataLoader
    train_dataset = CreditDataset(X_static_train, X_temporal_train, y_train)
    val_dataset = CreditDataset(X_static_val, X_temporal_val, y_val)
    calibration_dataset = CreditDataset(
        X_static_calibration,
        X_temporal_calibration,
        y_calibration,
    )
    test_dataset = CreditDataset(X_static_test, X_temporal_test, y_test)

    pin_memory = torch.cuda.is_available()
    # 数据已常驻内存；Windows spawn worker 的导入和复制成本通常高于收益。
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

    # 4. 模型初始化
    static_dim = X_static_train.shape[1]
    temporal_steps = X_temporal_train.shape[1]
    temporal_dim = X_temporal_train.shape[2]
    # German Credit 的“时序”由不同类别字段构造，并非真实月份；
    # 局部突变注意力仅在 Taiwan 的真实月度序列上启用。
    enable_local_attention = dataset_name == 'taiwan'

    def build_model(candidate_hidden, candidate_layers, candidate_dropout):
        return AABiLSTM(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            temporal_steps=temporal_steps,
            hidden_dim=candidate_hidden,
            num_layers=candidate_layers,
            num_classes=2,
            dropout=candidate_dropout,
            num_heads=num_heads,
            use_cross_attention=True,
            use_ag_resunit=True,
            bidirectional=True,
            use_multiscale=True,
            fusion_type=fusion_type,
            local_window=min(3, temporal_steps),
            use_global_context=True,
            use_local_attention=(
                enable_local_attention
                if fusion_type == 'global_local' else True
            ),
            temporal_categorical_index=(
                0 if dataset_name == 'taiwan' else None
            ),
            temporal_category_min=-2,
            temporal_category_max=8,
            use_step_embedding=dataset_name == 'taiwan',
            multiscale_mode=(
                'lightweight' if dataset_name == 'taiwan' else 'legacy'
            ),
        )

    if auto_tune:
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
                val_loader,
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
        if model.feature_fusion is not None else
        sum(p.numel() for p in model.cross_attention.parameters())
        if model.cross_attention is not None else 0
    )
    print(f"\nModel parameters: {total_parameters:,}")
    print(
        f"Fusion parameters: {fusion_parameters:,} "
        f"({fusion_parameters / max(total_parameters, 1):.2%})"
    )

    # 5. 训练
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

    ood_scores, ood_flags = black_swan_monitor.score(
        X_static_test, X_temporal_test
    )
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
        'ood_rate': float(ood_flags.mean()),
        'in_distribution_coverage': float(1.0 - ood_flags.mean()),
        'ood_score_mean': float(ood_scores.mean()),
        'ood_score_max': float(ood_scores.max()),
        'ood_threshold': float(black_swan_monitor.threshold_),
        'ood_scores': ood_scores,
        'ood_flags': ood_flags
    })
    test_metrics.update(summarize_selective_risk(
        test_metrics['labels'],
        test_metrics['predictions'],
        ood_scores,
        ood_flags,
    ))
    fairness_detail, fairness_gaps = summarize_group_fairness(
        test_metrics['labels'],
        test_metrics['predictions'],
        data_loader.audit_group_splits_.get('test', {}),
    )
    test_metrics['fairness_detail'] = fairness_detail
    test_metrics.update(fairness_gaps)
    test_metrics.update(bootstrap_metric_intervals(
        test_metrics['labels'],
        test_metrics['predictions'],
        test_metrics['probabilities'],
        random_state=split_seed,
    ))
    # 便于同一 Python 进程内进行带拒判的部署推理；CLI 可另存 JSON 状态。
    model.black_swan_monitor = black_swan_monitor
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
            'lightweight' if dataset_name == 'taiwan' else 'legacy'
        ),
        'temporal_state_embedding': dataset_name == 'taiwan',
        'step_embedding': dataset_name == 'taiwan',
        'auto_tune': bool(auto_tune),
        'split_sizes': {
            name: int(len(indices))
            for name, indices in data_loader.split_indices_.items()
            if isinstance(indices, np.ndarray)
        },
    }

    # 6. 结果输出
    print(f"\n{'=' * 60}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"AUC-ROC:   {test_metrics['auc']:.4f}")
    print(f"AUC-PR:    {test_metrics['auc_pr']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Sensitivity (Recall): {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity'] * 100:.2f}%)")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"F2-Score:   {test_metrics['f2']:.4f}")
    print(f"KS:         {test_metrics['ks']:.4f}")
    print(f"Lift@10%:   {test_metrics['lift_at_10']:.3f}")
    print(
        f"Calibration: Brier={test_metrics['brier_score']:.4f} | "
        f"ECE={test_metrics['ece']:.4f} | "
        f"LogLoss={test_metrics['log_loss']:.4f}"
    )
    print(f"Decision Threshold: {test_metrics['threshold']:.3f}")
    print(
        f"OOD/Black-swan flags: {test_metrics['ood_rate']:.2%} | "
        f"threshold={test_metrics['ood_threshold']:.3f}"
    )
    print(
        f"Accepted-set Accuracy: {test_metrics['selective_accuracy']:.4f} | "
        f"coverage={test_metrics['selective_coverage']:.2%}"
    )
    print(
        "95% bootstrap CI | "
        f"Accuracy [{test_metrics['accuracy_ci95_low']:.4f}, "
        f"{test_metrics['accuracy_ci95_high']:.4f}] | "
        f"Sensitivity [{test_metrics['sensitivity_ci95_low']:.4f}, "
        f"{test_metrics['sensitivity_ci95_high']:.4f}]"
    )
    print(f"{'=' * 60}\n")

    analysis_static_eval = (
        X_static_test if analysis_on_test else X_static_val
    )
    analysis_temporal_eval = (
        X_temporal_test if analysis_on_test else X_temporal_val
    )
    analysis_y_eval = y_test if analysis_on_test else y_val
    analysis_loader = test_loader if analysis_on_test else val_loader
    analysis_split_name = 'test' if analysis_on_test else 'validation'

    # 7. 对比基准模型
    if run_analysis:
        print(
            f"Comparison with baseline models on {analysis_split_name} split..."
        )
        compare_baselines(X_static_train, X_temporal_train, y_train,
                          analysis_static_eval, analysis_temporal_eval,
                          analysis_y_eval,
                          deep_epochs=min(epoch, 30),
                          batch_size=batch_size)

    # 8. 消融实验
    if run_analysis:
        print("\nRunning ablation study...")
        run_ablation_study(static_dim, temporal_dim, temporal_steps,
                           train_loader, val_loader, analysis_loader,
                           calibration_loader=calibration_loader,
                           epoch=min(epoch, 50),
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           dropout=dropout,
                           lr=lr,
                           weight_decay=weight_decay,
                           enable_local_attention=enable_local_attention)

    if run_analysis and dataset_name == 'taiwan':
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
            preprocessor_template=data_loader,
            split_seed=split_seed,
            # 不用外推的伪月份证明长期依赖；超过真实长度的实验直接跳过。
            extend_method=None
        )
    elif run_analysis:
        print(
            "History-length study skipped for German: its six pseudo steps "
            "are heterogeneous attributes rather than calendar history."
        )

    if run_analysis:
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
            num_heads=num_heads,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            enable_local_attention=enable_local_attention
        )

    if run_analysis:
        run_shap_summary(
            model,
            data_loader,
            analysis_static_eval,
            analysis_temporal_eval,
            dataset_name,
        )
        summarize_fusion_diagnostics(
            model,
            data_loader,
            analysis_static_eval,
            analysis_temporal_eval,
        )

    # 9. 可视化训练历史
    if make_plots:
        plot_training_history(history, dataset_name)

    return model, test_metrics, history


def compare_baselines(X_static_train, X_temporal_train, y_train,
                      X_static_test, X_temporal_test, y_test,
                      deep_epochs=20, batch_size=128):
    """与传统ML和深度学习基准模型对比。"""
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
                       hidden_dim=64, num_layers=4, num_heads=4, dropout=0.3,
                       lr=1e-3, weight_decay=1e-4,
                       enable_local_attention=True):
    """
    严格消融实验。

    多尺度、旧交叉注意力、非线性全局聚合和局部冲击注意力分别控制，
    避免原实现把 TS-CrossAttention 与 MultiScale 同时开启而混淆贡献。
    """

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    configs = [
        {
            'name': 'Standard LSTM',
            'use_cross': False,
            'use_multiscale': False,
            'use_global': False,
            'use_local': False,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'global_local'
        },
        {
            'name': '+ MultiScale only',
            'use_cross': False,
            'use_multiscale': True,
            'use_global': False,
            'use_local': False,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'global_local'
        },
        {
            'name': 'Legacy CrossAttn',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': True,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'legacy_cross'
        },
        {
            'name': '+ Nonlinear Global',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': False,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'global_local'
        }
    ]
    if enable_local_attention:
        configs.append({
            'name': '+ Local Shock Attn',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': True,
            'use_ag': False,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'global_local'
        })

    configs.extend([
        {
            'name': '+ AG-ResUnit',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': enable_local_attention,
            'use_ag': True,
            'use_bi': False,
            'use_focal': False,
            'fusion_type': 'global_local'
        },
        {
            'name': '+ Bi-Directional',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': enable_local_attention,
            'use_ag': True,
            'use_bi': True,
            'use_focal': False,
            'fusion_type': 'global_local'
        },
        {
            'name': '+ Dynamic Focal (Full)',
            'use_cross': True,
            'use_multiscale': True,
            'use_global': True,
            'use_local': enable_local_attention,
            'use_ag': True,
            'use_bi': True,
            'use_focal': True,
            'fusion_type': 'global_local'
        }
    ])

    results = []
    for config in configs:
        print(f"\nTesting: {config['name']}")
        set_seed(42)
        if getattr(train_loader, 'generator', None) is not None:
            train_loader.generator.manual_seed(42)

        model = AABiLSTM(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            temporal_steps=temporal_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            num_heads=num_heads,
            use_cross_attention=config['use_cross'],
            use_ag_resunit=config['use_ag'],
            bidirectional=config['use_bi'],
            use_multiscale=config['use_multiscale'],
            fusion_type=config['fusion_type'],
            local_window=min(3, temporal_steps),
            use_global_context=config['use_global'],
            use_local_attention=config['use_local'],
            temporal_categorical_index=(
                0 if enable_local_attention else None
            ),
            use_step_embedding=True,
            multiscale_mode='lightweight',
        )

        trainer = Trainer(
            model, train_loader, val_loader, test_loader,
            calibration_loader=calibration_loader,
            num_epochs=epoch,
            lr=lr,
            weight_decay=weight_decay,
            use_dynamic_focal=config['use_focal'],
            use_early_stopping=True
        )

        metrics, _ = trainer.train()
        row = {
            'name': config['name'],
            'auc': metrics['auc'],
            'auc_pr': metrics['auc_pr'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'parameters': sum(p.numel() for p in model.parameters())
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
                                   X_static_calibration,
                                   X_temporal_calibration, y_calibration,
                                   X_static_test, X_temporal_test, y_test,
                                   target_rates=(0.02, 0.05, 0.10, 0.22, 0.30),
                                   epoch=20, batch_size=128, hidden_dim=128,
                                   num_layers=8, num_heads=8, dropout=0.4,
                                   lr=5e-4, weight_decay=2e-4,
                                   enable_local_attention=True):
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
        set_seed(int(rate * 10000) + 42)
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
            use_multiscale=True,
            fusion_type='global_local',
            local_window=min(3, X_temporal_train.shape[1]),
            use_global_context=True,
            use_local_attention=enable_local_attention,
            temporal_categorical_index=(
                0 if enable_local_attention else None
            ),
            use_step_embedding=True,
            multiscale_mode='lightweight',
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
                             num_heads=8, dropout=0.4, extend_method=None,
                             preprocessor_template=None, split_seed=42):
    """
    长期依赖能力验证。支持数据扩展以测试更长时间窗口。

    参数:
        extend_method: 仅用于显式的合成压力测试。正式结果应设为 None，
                       只评估数据集中真实存在的历史长度。
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
                'use_multiscale': True,
                'fusion_type': 'global_local',
                'local_window': 3,
                'use_global_context': True,
                'use_local_attention': dataset_name == 'taiwan',
                'temporal_categorical_index': (
                    0 if dataset_name == 'taiwan' else None
                ),
                'use_step_embedding': True,
                'multiscale_mode': 'lightweight',
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
        data_loader = (
            copy.deepcopy(preprocessor_template)
            if preprocessor_template is not None
            else CreditDataLoader()
        )
        data_loader.temporal_step_names = [f'month_{i + 1}' for i in range(length)]
        data_loader.temporal_feature_names = ['feature_' + str(i) for i in range(temporal_features.shape[2])]

        # 数据预处理
        (X_static_train, X_temporal_train, y_train,
         X_static_val, X_temporal_val, y_val,
         X_static_test, X_temporal_test, y_test) = data_loader.preprocess(
            static_features,
            subset_temporal,
            y,
            split_seed=split_seed,
        )
        (
            X_static_calibration,
            X_temporal_calibration,
            y_calibration,
        ) = data_loader.calibration_data_

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
                calibration_loader=calibration_loader,
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


def transform_encoded_features(
    static_features,
    temporal_features,
    preprocessing_state,
):
    """
    使用 bundle 中的训练缩放器变换已按保存特征顺序编码的数据。

    原始 CSV/Excel 仍应先经过对应的 load_german_credit/load_taiwan_credit
    特征工程；本函数负责最后一步连续变量标准化。
    """
    static = np.asarray(static_features, dtype=np.float32).copy()
    temporal = np.asarray(temporal_features, dtype=np.float32).copy()
    static_names = preprocessing_state.get('static_feature_names') or []
    if static.ndim != 2 or temporal.ndim != 3:
        raise ValueError("Expected static [N,S] and temporal [N,T,F].")
    if static_names and static.shape[1] != len(static_names):
        raise ValueError(
            f"Expected {len(static_names)} static features, "
            f"got {static.shape[1]}."
        )

    continuous = preprocessing_state.get('continuous_static_cols')
    static_scaler = preprocessing_state.get('static_scaler')
    if continuous is not None and static_scaler is not None:
        mask = np.asarray(continuous, dtype=bool)
        mean = np.asarray(static_scaler['mean'], dtype=np.float32)
        scale = np.asarray(static_scaler['scale'], dtype=np.float32)
        static[:, mask] = (static[:, mask] - mean) / np.maximum(
            scale,
            1e-7,
        )

    temporal_scaler = preprocessing_state.get('temporal_scaler')
    if temporal_scaler is not None:
        mean = np.asarray(temporal_scaler['mean'], dtype=np.float32)
        scale = np.asarray(temporal_scaler['scale'], dtype=np.float32)
        continuous_temporal = preprocessing_state.get(
            'continuous_temporal_cols'
        )
        if continuous_temporal is None:
            continuous_temporal = np.ones(
                temporal.shape[-1],
                dtype=bool,
            )
        continuous_temporal = np.asarray(
            continuous_temporal,
            dtype=bool,
        )
        if (
            len(continuous_temporal) != temporal.shape[-1]
            or int(continuous_temporal.sum()) != len(mean)
        ):
            raise ValueError(
                "Temporal scaler mask is incompatible with the supplied "
                f"feature dimension {temporal.shape[-1]}."
            )
        temporal[:, :, continuous_temporal] = (
            temporal[:, :, continuous_temporal]
            - mean.reshape(1, 1, -1)
        ) / np.maximum(scale.reshape(1, 1, -1), 1e-7)
    return static.astype(np.float32), temporal.astype(np.float32)


def _read_raw_credit_frame(raw_data, dataset_name):
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()
    path = str(raw_data).strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw inference file not found: {path}")
    if dataset_name == 'german':
        return pd.read_csv(path, sep=r"\s+", header=None)
    extension = os.path.splitext(path)[1].lower()
    if extension in {'.xls', '.xlsx'}:
        return pd.read_excel(path, header=1)
    return pd.read_csv(path)


def _engineer_raw_german(frame, preprocessing_state):
    if frame.shape[1] not in {20, 21}:
        raise ValueError(
            "German raw input must contain 20 feature columns and may "
            "optionally include the label column."
        )
    features = frame.iloc[:, :20].copy()
    features.columns = list(range(20))
    numeric_cols = [
        int(value)
        for value in preprocessing_state.get(
            'raw_static_numeric_names',
            [1, 4, 7, 10, 12, 15, 17],
        )
    ]
    categorical_cols = [
        int(value)
        for value in preprocessing_state.get(
            'raw_static_categorical_names',
            [9, 11, 13, 14, 16, 18, 19],
        )
    ]
    prefixes = preprocessing_state.get(
        'raw_static_prefixes',
        [f'A{column}' for column in categorical_cols],
    )
    numeric = features[numeric_cols].astype(np.float32).to_numpy()
    categorical = features[categorical_cols].astype(str)
    categorical.columns = [str(column) for column in categorical_cols]
    dummy = pd.get_dummies(
        categorical,
        prefix=prefixes,
        drop_first=False,
        dtype=np.float32,
    )
    expected_dummy = preprocessing_state.get('static_dummy_columns') or []
    dummy = dummy.reindex(columns=expected_dummy, fill_value=0.0)
    static = np.concatenate(
        [numeric, dummy.to_numpy(dtype=np.float32)],
        axis=1,
    )

    selected_cols = preprocessing_state.get('german_selected_cols')
    mappings = preprocessing_state.get('german_temporal_maps')
    risk_priors = preprocessing_state.get('german_risk_priors') or {}
    if not selected_cols or not mappings:
        raise ValueError(
            "The bundle does not contain German raw-category mappings."
        )
    temporal_steps = []
    for step, column in enumerate(selected_cols):
        values = features[int(column)].astype(str).to_numpy()
        mapping = mappings[step]
        code_map = {
            str(key): float(value)
            for key, value in mapping['code'].items()
        }
        frequency_map = {
            str(key): float(value)
            for key, value in mapping['frequency'].items()
        }
        column_priors = (
            risk_priors.get(int(column))
            or risk_priors.get(str(column))
            or {}
        )
        category_code = np.asarray(
            [code_map.get(value, 0.5) for value in values],
            dtype=np.float32,
        )
        risk_prior = np.asarray(
            [
                column_priors.get(
                    value,
                    code_map.get(value, 0.5),
                )
                for value in values
            ],
            dtype=np.float32,
        )
        category_frequency = np.asarray(
            [frequency_map.get(value, 0.0) for value in values],
            dtype=np.float32,
        )
        temporal_steps.append(np.stack(
            [category_code, risk_prior, category_frequency],
            axis=1,
        ))
    temporal = np.stack(temporal_steps, axis=1)
    return static, temporal


def _engineer_raw_taiwan(frame, preprocessing_state):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={
        'default payment next month': 'default.payment.next.month',
    })
    static_num_cols = ['LIMIT_BAL', 'AGE']
    static_cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = [
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    ]
    payment_cols = [
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    ]
    required = (
        static_num_cols + static_cat_cols + status_cols
        + bill_cols + payment_cols
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Taiwan raw input is missing required columns: {missing}"
        )

    limit_balance = np.maximum(
        frame['LIMIT_BAL'].astype(float).to_numpy(),
        1.0,
    )
    bills = frame[bill_cols].astype(float).to_numpy()
    payments = frame[payment_cols].astype(float).to_numpy()
    statuses = frame[status_cols].astype(float).to_numpy()
    utilization = np.clip(
        bills / limit_balance[:, None],
        -10.0,
        10.0,
    )
    payment_to_bill = np.clip(
        payments / np.maximum(np.abs(bills), 1.0),
        -10.0,
        10.0,
    )
    chronological_bill = utilization[:, ::-1]
    trend_axis = np.arange(6, dtype=np.float32)
    trend_axis -= trend_axis.mean()
    bill_trend = (
        chronological_bill @ trend_axis
    ) / max(float(np.square(trend_axis).sum()), 1e-7)
    engineered = pd.DataFrame({
        'recent_bill_to_limit': utilization[:, 0],
        'avg_bill_to_limit_6m': utilization.mean(axis=1),
        'max_bill_to_limit_6m': utilization.max(axis=1),
        'bill_trend_6m': bill_trend,
        'recent_payment_shock': statuses[:, 0] - statuses[:, 1],
        'delinquency_count_6m': (statuses > 0).sum(axis=1),
        'max_delinquency_6m': statuses.max(axis=1),
        'recent_payment_to_bill': payment_to_bill[:, 0],
        'avg_payment_to_bill_6m': payment_to_bill.mean(axis=1),
    }).clip(-10.0, 10.0)
    numeric = pd.concat(
        [
            frame[static_num_cols].astype(np.float32).reset_index(drop=True),
            engineered.astype(np.float32).reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=np.float32)
    categorical = frame[static_cat_cols].astype(str)
    dummy = pd.get_dummies(
        categorical,
        prefix=static_cat_cols,
        drop_first=False,
        dtype=np.float32,
    )
    expected_dummy = preprocessing_state.get('static_dummy_columns') or []
    dummy = dummy.reindex(columns=expected_dummy, fill_value=0.0)
    static = np.concatenate(
        [numeric, dummy.to_numpy(dtype=np.float32)],
        axis=1,
    )

    bill_signal = np.sign(bills) * np.log1p(np.abs(bills))
    payment_signal = np.sign(payments) * np.log1p(np.abs(payments))
    temporal = np.stack(
        [
            statuses,
            bill_signal,
            payment_signal,
            utilization,
            payment_to_bill,
        ],
        axis=-1,
    )[:, ::-1, :].copy().astype(np.float32)
    return static, temporal


def transform_raw_credit_data(raw_data, preprocessing_state, dataset_name=None):
    """
    Convert a raw German/Taiwan file or DataFrame to model-ready arrays.

    Category vocabularies, column order and scalers come only from the
    training bundle. Unknown categories are represented by all-zero dummy
    vectors (or the model's unknown temporal-state embedding).
    """
    if not preprocessing_state:
        raise ValueError("A preprocessing_state is required.")
    dataset_name = (
        dataset_name or preprocessing_state.get('dataset_name')
    )
    if dataset_name not in {'german', 'taiwan'}:
        raise ValueError(
            "dataset_name must be 'german' or 'taiwan'."
        )
    frame = _read_raw_credit_frame(raw_data, dataset_name)
    if dataset_name == 'german':
        static, temporal = _engineer_raw_german(
            frame,
            preprocessing_state,
        )
    else:
        static, temporal = _engineer_raw_taiwan(
            frame,
            preprocessing_state,
        )
    return transform_encoded_features(
        static,
        temporal,
        preprocessing_state,
    )


def predict_raw_credit(model, raw_data, dataset_name=None, **predict_kwargs):
    """End-to-end raw CSV/Excel/DataFrame inference with schema validation."""
    preprocessing_state = getattr(model, 'preprocessing_state', None)
    static, temporal = transform_raw_credit_data(
        raw_data,
        preprocessing_state,
        dataset_name=dataset_name,
    )
    result = predict_with_rejection(
        model,
        static,
        temporal,
        **predict_kwargs,
    )
    result['static_features'] = static
    result['temporal_features'] = temporal
    return result


def predict_with_rejection(model, static_features, temporal_features,
                           decision_threshold=None, monitor=None,
                           entropy_threshold=None, batch_size=512,
                           temperature=None, calibration_scale=None,
                           calibration_bias=None):
    """
    对已按训练 scaler 变换的数据进行带拒判的推理。

    decision=-1 表示输入分布异常或预测熵过高，应进入人工复核；0/1 分别
    表示正常/违约。OOD 检测不是分类标签，不能直接把异常样本当作违约。
    """
    static_features = np.asarray(static_features, dtype=np.float32)
    temporal_features = np.asarray(temporal_features, dtype=np.float32)
    if static_features.shape[0] != temporal_features.shape[0]:
        raise ValueError("Static and temporal sample counts must match.")
    if len(static_features) == 0:
        raise ValueError("At least one sample is required for inference.")
    if static_features.ndim != 2 or temporal_features.ndim != 3:
        raise ValueError(
            "Expected static [N, S] and temporal [N, T, F] arrays."
        )
    if decision_threshold is None:
        decision_threshold = getattr(model, 'decision_threshold', 0.5)
    if calibration_scale is None:
        calibration_scale = getattr(model, 'calibration_scale', None)
    if calibration_bias is None:
        calibration_bias = getattr(model, 'calibration_bias', 0.0)
    if temperature is not None:
        calibration_scale = 1.0 / float(temperature)
        calibration_bias = 0.0
    elif calibration_scale is None:
        temperature = getattr(model, 'temperature', 1.0)
        calibration_scale = 1.0 / float(temperature)
    if entropy_threshold is None:
        entropy_threshold = getattr(model, 'entropy_threshold', 0.65)
    decision_threshold = float(decision_threshold)
    calibration_scale = float(calibration_scale)
    calibration_bias = float(calibration_bias)
    temperature = 1.0 / max(calibration_scale, 1e-7)
    entropy_threshold = float(entropy_threshold)
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be in [0, 1].")
    if (
        not np.isfinite(calibration_scale)
        or calibration_scale <= 0.0
        or not np.isfinite(calibration_bias)
    ):
        raise ValueError(
            "Calibration scale must be positive and calibration parameters "
            "must be finite."
        )
    if not 0.0 <= entropy_threshold <= np.log(2.0):
        raise ValueError("entropy_threshold must be in [0, log(2)].")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    monitor = monitor or getattr(model, 'black_swan_monitor', None)
    if monitor is None:
        raise ValueError(
            "A fitted BlackSwanMonitor is required for rejection inference."
        )

    model_device = next(model.parameters()).device
    model.eval()
    probability_batches = []
    with torch.inference_mode():
        for start in range(0, len(static_features), batch_size):
            end = start + batch_size
            static = torch.as_tensor(
                static_features[start:end],
                device=model_device
            )
            temporal = torch.as_tensor(
                temporal_features[start:end],
                device=model_device
            )
            logits = model(static, temporal).float()
            margin = logits[:, 1] - logits[:, 0]
            positive_probability = torch.sigmoid(
                calibration_scale * margin + calibration_bias
            )
            probability_batches.append(
                torch.stack(
                    [1.0 - positive_probability, positive_probability],
                    dim=-1,
                ).cpu().numpy()
            )

    probabilities = np.concatenate(probability_batches, axis=0)
    positive_probability = probabilities[:, 1]
    predictions = (
        positive_probability >= decision_threshold
    ).astype(np.int64)
    predictive_entropy = -np.sum(
        probabilities * np.log(probabilities + 1e-12), axis=1
    )
    ood_scores, ood_flags = monitor.score(
        static_features, temporal_features
    )
    uncertainty_flags = predictive_entropy > entropy_threshold
    review_flags = ood_flags | uncertainty_flags
    decisions = predictions.copy()
    decisions[review_flags] = -1

    return {
        'decisions': decisions,
        'predictions': predictions,
        'probabilities': positive_probability,
        'predictive_entropy': predictive_entropy,
        'ood_scores': ood_scores,
        'ood_flags': ood_flags,
        'uncertainty_flags': uncertainty_flags,
        'review_flags': review_flags,
        'decision_threshold': decision_threshold,
        'temperature': temperature,
        'calibration_scale': calibration_scale,
        'calibration_bias': calibration_bias,
        'entropy_threshold': entropy_threshold,
    }


def save_experiment_bundle(path, model, metrics=None):
    """保存可复现实验和部署决策所需的完整状态。"""
    monitor = getattr(model, 'black_swan_monitor', None)
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
        'black_swan_monitor': (
            monitor.to_dict() if monitor is not None else None
        ),
        'preprocessing_state': copy.deepcopy(
            getattr(model, 'preprocessing_state', None)
        ),
        'training_config': copy.deepcopy(
            getattr(model, 'training_config', None)
        ),
        'metrics': scalar_metrics(metrics or {}),
        'fairness_detail': copy.deepcopy(
            (metrics or {}).get('fairness_detail')
        ),
    }
    if not bundle['model_config']:
        raise ValueError("model.model_config is required for bundle export.")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(bundle, path)
    return bundle


def load_experiment_bundle(path, map_location='cpu'):
    """恢复模型、阈值、温度和 OOD 监控器。"""
    try:
        bundle = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        bundle = torch.load(path, map_location=map_location)
    format_version = bundle.get('format_version')
    if format_version not in {2, 3}:
        raise ValueError("Unsupported experiment bundle format.")
    model = AABiLSTM(**bundle['model_config'])
    model.load_state_dict(
        bundle['model_state_dict'],
        strict=format_version >= 3,
    )
    model.decision_threshold = float(bundle['decision_threshold'])
    model.temperature = float(bundle['temperature'])
    model.calibration_scale = float(
        bundle.get(
            'calibration_scale',
            1.0 / max(model.temperature, 1e-7),
        )
    )
    model.calibration_bias = float(
        bundle.get('calibration_bias', 0.0)
    )
    model.entropy_threshold = float(
        bundle.get('entropy_threshold', 0.65)
    )
    model.preprocessing_state = bundle.get('preprocessing_state')
    model.training_config = bundle.get('training_config')
    monitor_state = bundle.get('black_swan_monitor')
    if monitor_state is not None:
        model.black_swan_monitor = BlackSwanMonitor.from_dict(
            monitor_state
        )
    return model.to(map_location), bundle


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
                        help='Default: optimized setting (30 German / 50 Taiwan).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Default: optimized setting (32 German / 256 Taiwan).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Default: 7 for German, 42 for Taiwan.')
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
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Attention heads used only by legacy_cross fusion.')
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
            'Validation sensitivity floor (default: 0.65 German / '
            '0.55 Taiwan).'
        ),
    )
    parser.add_argument(
        '--threshold-confidence',
        type=float,
        default=None,
        help=(
            'One-sided Wilson confidence used for the sensitivity floor '
            '(default: 0.60 German / 0.80 Taiwan).'
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
        help='Override final focal gamma (default: 1.5 German / 1.75 Taiwan).',
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=None,
        help='Override EMA decay (default: 0.99 German / 0.995 Taiwan).',
    )
    parser.add_argument(
        '--fusion',
        choices=['global_local', 'legacy_cross'],
        default='global_local',
        help='Feature fusion: recommended nonlinear global-local or legacy cross-attention.'
    )
    parser.add_argument(
        '--ood-quantile',
        type=float,
        default=0.99,
        help='Training-score quantile used as the black-swan/OOD threshold.'
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
        '--monitor-json',
        default=None,
        help='Optional path for the fitted black-swan monitor state.'
    )
    analysis_group = parser.add_mutually_exclusive_group()
    analysis_group.add_argument(
        '--analysis',
        dest='run_analysis',
        action='store_true',
        help='Run baselines, strict ablation, robustness, SHAP and diagnostics.'
    )
    analysis_group.add_argument(
        '--no-analysis',
        dest='run_analysis',
        action='store_false',
        help='Skip the additional analysis suite.'
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
        run_analysis=None,
        make_plots=None,
        calibrate_probabilities=True,
    )
    return parser.parse_args()


def scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if key in {
            'predictions', 'probabilities', 'labels',
            'ood_scores', 'ood_flags'
        }:
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def attach_repeat_summary(primary_metrics, metric_rows):
    """把多次独立运行的均值、标准差和均值置信区间加入主结果。"""
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

    run_analysis = args.run_analysis
    if run_analysis is None:
        if args.dataset is not None:
            run_analysis = False
        else:
            while True:
                ans = input("是否进行验证环节（包含基准模型对比、消融实验）（Y/N）：").strip().upper()
                if ans in {'Y', 'N'}:
                    run_analysis = (ans == 'Y')
                    break
                print("输入无效，请输入 Y 或 N。")

    make_plots = args.make_plots
    if make_plots is None:
        if args.dataset is not None:
            make_plots = False
        else:
            while True:
                ans = input("是否绘图（Y/N）：").strip().upper()
                if ans in {'Y', 'N'}:
                    make_plots = (ans == 'Y')
                    break
                print("输入无效，请输入 Y 或 N。")

    print("Starting experiments...")

    if args.repeat_runs < 1:
        raise ValueError("--repeat-runs must be at least 1.")
    experiment_kwargs = dict(
        dataset_name=dataset_name,
        epoch=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        run_analysis=run_analysis,
        make_plots=make_plots,
        threshold_min_sensitivity=args.min_sensitivity,
        ood_quantile=args.ood_quantile,
        fusion_type=args.fusion,
        seed=args.seed,
        split_seed=args.split_seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
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
    if args.repeat_runs > 1:
        base_model_seed = (
            args.seed
            if args.seed is not None
            else (7 if dataset_name == 'german' else 42)
        )
        for repeat_idx in range(1, args.repeat_runs):
            print(
                f"\nStarting repeated run {repeat_idx + 1}/"
                f"{args.repeat_runs}..."
            )
            repeated_kwargs = dict(experiment_kwargs)
            repeated_kwargs.update({
                'seed': base_model_seed + 1009 * repeat_idx,
                'split_seed': (
                    args.split_seed + 1009 * repeat_idx
                    if args.vary_split_seed else args.split_seed
                ),
                'run_analysis': False,
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
        if metrics.get('fairness_detail'):
            result_payload['fairness_detail'] = metrics['fairness_detail']
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

    if args.monitor_json:
        monitor_dir = os.path.dirname(os.path.abspath(args.monitor_json))
        os.makedirs(monitor_dir, exist_ok=True)
        with open(args.monitor_json, 'w', encoding='utf-8') as monitor_file:
            json.dump(
                model.black_swan_monitor.to_dict(),
                monitor_file,
                ensure_ascii=False,
                indent=2
            )
        print(f"Saved black-swan monitor to {args.monitor_json}")

    print("\nExperiment completed successfully!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final AUC: {metrics['auc']:.4f}")
    print(f"Final Sensitivity: {metrics['sensitivity']:.4f}")
