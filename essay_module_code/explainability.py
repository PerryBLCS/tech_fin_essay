"""可解释性分析 (SHAP)"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from config import device


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
            import explainability
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
