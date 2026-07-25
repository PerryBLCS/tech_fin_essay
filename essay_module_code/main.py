"""主程序与运行入口"""

import argparse
import copy
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from config import device, set_seed
from data_utils import CreditDataLoader, CreditDataset, BlackSwanMonitor
from model import AABiLSTM
from trainer import (
    Trainer, SequenceBaselineModel,
    train_sequence_baseline, specificity_score,
)
from explainability import (
    run_shap_summary, summarize_fusion_diagnostics,
    summarize_selective_risk, summarize_group_fairness,
    bootstrap_metric_intervals,
)


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
