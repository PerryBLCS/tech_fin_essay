"""训练与评估模块"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score,
    log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from statistics import NormalDist

from config import device
from data_utils import CreditDataset
from loss import DynamicFocalLoss


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
