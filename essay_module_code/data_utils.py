"""数据加载与预处理模块"""

import numpy as np
import pandas as pd
import os
import copy
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import GERMAN_CREDIT_URL, TAIWAN_CREDIT_URL, device


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
