import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy.spatial import ConvexHull
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

# ==============================================
# 全局设置
# ==============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2

BO_INIT_POINTS = 15
BO_N_ITER = 75
BO_CV_FOLDS = 10

ENABLE_LOESS = True
MAIN_MODEL_NAME = "Exp1_SMOTE+BO+CatBoost"
MAIN_MODEL_DISPLAY_NAME = "Best Model"
DATA_PATH = r"C:\Users\h\Desktop\5.4.xlsx"

FEATURE_LABELS = [
    "DBR", "DPC", "FSSR", "CFR", "PCGDP",
    "PCI", "GDPgr", "PTI", "PGR", "RTA",
    "NPMS", "DCR", "ALR", "EGR", "QR",
    "CR", "PPWMB", "PRCG", "GS", "PC"
]


# ==============================================
# SAFE 评分函数
# ==============================================
def calculate_lorenz_zonoid(data_vector):
    """计算 Lorenz Zonoid 值"""
    data = np.array(data_vector).flatten()
    if len(data) == 0:
        return 0.0
    if np.max(data) == np.min(data):
        return 0.0

    data_sorted = np.sort(data)
    data_normalized = (data_sorted - data_sorted.min()) / (data_sorted.max() - data_sorted.min() + 1e-12)

    if np.sum(data_normalized) == 0:
        return 0.0

    n = len(data_normalized)
    F = np.arange(1, n + 1) / n
    L = np.cumsum(data_normalized) / np.sum(data_normalized)

    points = np.column_stack((F, L))
    points = np.vstack([[0, 0], points])
    try:
        hull = ConvexHull(points)
        lz_value = hull.volume
    except Exception:
        lz_value = np.trapz(L, F)
    return float(lz_value)


def _extract_shap_array(shap_values):
    """兼容不同 shap 版本的返回格式"""
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        # 二分类中常取正类
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.array(shap_values)

    # 某些版本会返回 (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values[:, :, 0]

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    return shap_values


def calculate_shapley_lorenz(model, X_data, feature_names):
    """计算每个特征的 Shapley-Lorenz 值"""
    explainer = shap.TreeExplainer(model)
    X_df = pd.DataFrame(X_data, columns=feature_names)
    shap_values = explainer.shap_values(X_df)
    shap_values = _extract_shap_array(shap_values)

    sl_values = [calculate_lorenz_zonoid(shap_values[:, i]) for i in range(len(feature_names))]
    return np.array(sl_values)


class SAFEScoreCalculator:
    """SAFE 评分计算器：Ex / Ac / Sust"""
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names):
        self.model = model
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.feature_names = feature_names
        self.k = len(feature_names)

        self.shapley_lorenz = calculate_shapley_lorenz(model, X_test, feature_names)

        y_pred_proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            self.y_pred_proba = y_pred_proba[:, 1]
        else:
            self.y_pred_proba = y_pred_proba[np.arange(len(y_test)), y_test]

    def ex_score(self):
        LZ_Y = calculate_lorenz_zonoid(self.y_test)
        sum_SL = np.sum(self.shapley_lorenz)
        ex = sum_SL / ((LZ_Y + 1e-10) * self.k)
        return float(np.clip(ex, 0.0, 1.0))

    def ac_score(self):
        LZ_Y_pred = calculate_lorenz_zonoid(self.y_pred_proba)
        LZ_Y_test = calculate_lorenz_zonoid(self.y_test)
        ac = LZ_Y_pred / (LZ_Y_test + 1e-10)
        return float(np.clip(ac, 0.0, 1.0))

    def sust_score(self, G=5):
        pred_quantiles = np.quantile(self.y_pred_proba, np.linspace(0, 1, G + 1))
        groups = np.digitize(self.y_pred_proba, pred_quantiles[1:-1])
        V_G_SL = []

        for g in range(G):
            group_idx = np.where(groups == g)[0]
            if len(group_idx) == 0:
                V_G_SL.append(0.0)
                continue

            X_group = self.X_test[group_idx]
            explainer = shap.TreeExplainer(self.model)
            X_group_df = pd.DataFrame(X_group, columns=self.feature_names)
            shap_group = explainer.shap_values(X_group_df)
            shap_group = _extract_shap_array(shap_group)

            if shap_group.shape[1] != self.k:
                sl_group = [0.0] * self.k
            else:
                sl_group = [calculate_lorenz_zonoid(shap_group[:, i]) for i in range(self.k)]

            V_G_SL.append(np.sum(sl_group))

        LZ_V_G_SL = calculate_lorenz_zonoid(V_G_SL)
        sust = 1 - LZ_V_G_SL
        return float(np.clip(sust, 0.0, 1.0))

    def get_all_scores(self):
        return {
            "Ex-score": round(self.ex_score(), 6),
            "Ac-score": round(self.ac_score(), 6),
            "Sust-score": round(self.sust_score(), 6)
        }

# ==============================================
# RGA 函数
# ==============================================
def binary_ranked_grading_accuracy(y_true, y_pred_proba, pos_label=1):
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred_proba) >= 0.5).astype(int)

    y_true_bin = np.where(y_true == pos_label, 1, 0)
    y_pred_bin = np.where(y_pred == pos_label, 1, 0)

    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    rga_score = (TP + TN + 0.5 * (FP + FN)) / len(y_true)
    return round(float(rga_score), 6), {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}

# ==============================================
# 排序指标：Precision@Top-K
# ==============================================
def precision_at_top_k(y_true, y_score, k):
    """
    Precision@Top-K：按预测风险从高到低排序，取前 K 个样本计算正类比例。
    这里的 K 是绝对数量，不是百分比。
    """
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score)

    n = len(y_true)
    if n == 0:
        return np.nan

    k = int(max(1, min(k, n)))
    top_idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_idx] == 1))

# ==============================================
# 画图函数
# ==============================================
def plot_confusion(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc(y_test, y_score, title="ROC Curve"):
    if len(np.unique(y_test)) != 2:
        return
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.6f}')
    plt.plot([0, 1], [0, 1], '--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_test, y_score, title="Precision-Recall Curve"):
    if len(np.unique(y_test)) != 2:
        return
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, label=f'AP = {ap:.6f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================================
# 数据准备
# ==============================================
def load_and_prepare_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在，请检查路径：{data_path}")

    df = pd.read_excel(data_path)
    X_raw = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].copy().astype(int)
    if X_raw.shape[1] != len(FEATURE_LABELS):
        raise ValueError(f"特征数量与预设符号数量不一致：数据中有 {X_raw.shape[1]} 个特征，但 FEATURE_LABELS 中有 {len(FEATURE_LABELS)} 个标签。")
    feature_names = FEATURE_LABELS.copy()

    print("Original class distribution:")
    print(y_raw.value_counts())

    # ---------- 原始数据流：不做 SMOTE ----------
    scaler_raw = MinMaxScaler(feature_range=(-1, 1))
    X_raw_scaled = scaler_raw.fit_transform(X_raw)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw_scaled, y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw if len(np.unique(y_raw)) > 1 else None
    )

    # ---------- 主数据流：先 SMOTE，再划分 ----------
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_raw, y_raw)

    print("\nResampled class distribution:")
    print(pd.Series(y_resampled).value_counts())

    scaler_smote = MinMaxScaler(feature_range=(-1, 1))
    X_resampled_scaled = scaler_smote.fit_transform(X_resampled)

    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
        X_resampled_scaled, y_resampled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_resampled if len(np.unique(y_resampled)) > 1 else None
    )

    # 类别分布图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_raw, palette='viridis')
    plt.title('Class Distribution Before SMOTE')

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_resampled, palette='magma')
    plt.title('Class Distribution After SMOTE')
    plt.tight_layout()
    plt.show()

    return {
        'feature_names': feature_names,
        'raw': {
            'X_train': np.array(X_train_raw),
            'X_test': np.array(X_test_raw),
            'y_train': np.array(y_train_raw),
            'y_test': np.array(y_test_raw),
        },
        'smote_before_split': {
            'X_train': np.array(X_train_sm),
            'X_test': np.array(X_test_sm),
            'y_train': np.array(y_train_sm),
            'y_test': np.array(y_test_sm),
            'X_all': np.array(X_resampled_scaled),
            'y_all': np.array(y_resampled)
        }
    }

# ==============================================
# class_weight
# ==============================================
def get_class_weights(y_train):
    y_train = np.array(y_train).astype(int)
    counts = pd.Series(y_train).value_counts()
    n_neg = counts.get(0, 0)
    n_pos = counts.get(1, 0)

    if n_pos == 0:
        return [1.0, 1.0]

    pos_weight = n_neg / n_pos
    return [1.0, float(pos_weight)]

# ==============================================
# 默认参数（供不做 BO 的实验组使用）
# ==============================================
def get_default_params():
    return {
        'iterations': 350,
        'learning_rate': 0.03,
        'depth': 4,
        'l2_leaf_reg': 3.5,
        'border_count': 64,
        'random_strength': 1.5,
        'bagging_temperature': 0.3,
        'rsm': 0.8
    }

# ==============================================
# CatBoost 构建器
# ==============================================
def build_cat_model(params, use_class_weight=False, y_train=None, use_early_stopping=False, verbose=0):
    model_kwargs = {
        'iterations': int(params['iterations']),
        'learning_rate': float(params['learning_rate']),
        'depth': int(params['depth']),
        'l2_leaf_reg': float(params['l2_leaf_reg']),
        'border_count': int(params['border_count']),
        'random_strength': float(params['random_strength']),
        'bagging_temperature': float(params['bagging_temperature']),
        'rsm': float(params['rsm']),
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': RANDOM_STATE,
        'verbose': verbose,
        'thread_count': 1
    }

    if use_class_weight and y_train is not None:
        model_kwargs['class_weights'] = get_class_weights(y_train)

    if use_early_stopping:
        model_kwargs['early_stopping_rounds'] = 50

    return CatBoostClassifier(**model_kwargs)

# ==============================================
# 评估函数
# ==============================================
def evaluate_binary_metrics(model, X_test, y_test, exp_name="Experiment"):
    y_test = np.array(y_test).astype(int)
    y_pred = np.array(model.predict(X_test)).astype(int).ravel()
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        p_at_50 = precision_at_top_k(y_test, y_score, 50)
        p_at_100 = precision_at_top_k(y_test, y_score, 100)
        p_at_150 = precision_at_top_k(y_test, y_score, 150)
        rga, rga_metrics = binary_ranked_grading_accuracy(y_test, y_score, pos_label=1)
    else:
        auc, ap, p_at_50, p_at_100, p_at_150 = np.nan, np.nan, np.nan, np.nan, np.nan
        rga, rga_metrics = np.nan, None

    result = {
        'Experiment': exp_name,
        'Accuracy': round(float(acc), 6),
        'Precision': round(float(pre), 6),
        'Recall': round(float(rec), 6),
        'F1': round(float(f1), 6),
        'AUC': round(float(auc), 6) if pd.notna(auc) else np.nan,
        'AveragePrecision': round(float(ap), 6) if pd.notna(ap) else np.nan,
        'P@50': round(float(p_at_50), 6) if pd.notna(p_at_50) else np.nan,
        'P@100': round(float(p_at_100), 6) if pd.notna(p_at_100) else np.nan,
        'P@150': round(float(p_at_150), 6) if pd.notna(p_at_150) else np.nan,
        'RGA': round(float(rga), 6) if pd.notna(rga) else np.nan
    }
    return result, y_pred, y_score, rga_metrics

# ==============================================
# BO 目标函数
# ==============================================
def make_bo_objective(X_train, y_train, use_class_weight=False):
    def objective_cbc(iterations, learning_rate, depth, l2_leaf_reg,
                      border_count, random_strength, bagging_temperature, rsm):
        params = {
            'iterations': int(iterations),
            'learning_rate': learning_rate,
            'depth': int(depth),
            'l2_leaf_reg': l2_leaf_reg,
            'border_count': int(border_count),
            'random_strength': random_strength,
            'bagging_temperature': bagging_temperature,
            'rsm': rsm
        }

        model = build_cat_model(
            params=params,
            use_class_weight=use_class_weight,
            y_train=y_train,
            use_early_stopping=False,
            verbose=0
        )

        cv = StratifiedKFold(n_splits=BO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=1
        )
        return float(np.mean(scores))

    return objective_cbc

# ==============================================
# 贝叶斯优化
# ==============================================
def run_bayesian_optimization(X_train, y_train, use_class_weight=False):
    pbounds = {
        'iterations': (200, 500),
        'learning_rate': (0.01, 0.05),
        'depth': (3, 5),
        'l2_leaf_reg': (2.0, 8.0),
        'border_count': (50, 100),
        'random_strength': (1.0, 3.0),
        'bagging_temperature': (0.1, 0.5),
        'rsm': (0.6, 0.9)
    }

    objective = make_bo_objective(X_train, y_train, use_class_weight=use_class_weight)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=RANDOM_STATE,
        verbose=2
    )

    print("Starting Bayesian Optimization...")
    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)

    best_params = optimizer.max['params']
    best_params['iterations'] = int(best_params['iterations'])
    best_params['depth'] = int(best_params['depth'])
    best_params['border_count'] = int(best_params['border_count'])

    print(f"Best BO target (weighted F1): {optimizer.max['target']:.6f}")
    print("Best params:", best_params)
    return best_params

# ==============================================
# 10折交叉验证：分类 + 排序指标
# ==============================================
def cross_validate_ranking_metrics(params, X_train, y_train, use_class_weight=False, n_splits=10):
    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_f1 = []
    fold_rga = []
    fold_ap = []
    fold_p50 = []
    fold_p100 = []
    fold_p150 = []

    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = build_cat_model(
            params=params,
            use_class_weight=use_class_weight,
            y_train=y_tr,
            use_early_stopping=False,
            verbose=0
        )
        model.fit(X_tr, y_tr, verbose=0)

        y_pred = np.array(model.predict(X_val)).astype(int).ravel()
        y_score = model.predict_proba(X_val)[:, 1]

        fold_f1.append(f1_score(y_val, y_pred, zero_division=0))
        fold_ap.append(average_precision_score(y_val, y_score))
        fold_p50.append(precision_at_top_k(y_val, y_score, 50))
        fold_p100.append(precision_at_top_k(y_val, y_score, 100))
        fold_p150.append(precision_at_top_k(y_val, y_score, 150))
        rga, _ = binary_ranked_grading_accuracy(y_val, y_score, pos_label=1)
        fold_rga.append(rga)

    cv_result = {
        'Mean F1': round(float(np.mean(fold_f1)), 6),
        'Std F1': round(float(np.std(fold_f1)), 6),
        'Mean RGA': round(float(np.mean(fold_rga)), 6),
        'Std RGA': round(float(np.std(fold_rga)), 6),
        'Mean AP': round(float(np.mean(fold_ap)), 6),
        'Std AP': round(float(np.std(fold_ap)), 6),
        'Mean P@50': round(float(np.mean(fold_p50)), 6),
        'Std P@50': round(float(np.std(fold_p50)), 6),
        'Mean P@100': round(float(np.mean(fold_p100)), 6),
        'Std P@100': round(float(np.std(fold_p100)), 6),
        'Mean P@150': round(float(np.mean(fold_p150)), 6),
        'Std P@150': round(float(np.std(fold_p150)), 6),
    }

    return cv_result, {
        'fold_f1': fold_f1,
        'fold_rga': fold_rga,
        'fold_ap': fold_ap,
        'fold_p50': fold_p50,
        'fold_p100': fold_p100,
        'fold_p150': fold_p150,
    }

# ==============================================
# 单个实验组运行
# ==============================================
def run_experiment(exp_name, dataset_pack, use_bo=True, use_class_weight=False, final_use_early_stopping=True):
    X_train = dataset_pack['X_train']
    X_test = dataset_pack['X_test']
    y_train = dataset_pack['y_train']
    y_test = dataset_pack['y_test']

    print("\n" + "#" * 100)
    print(f"开始实验: {exp_name}")
    print(f"use_bo={use_bo}, use_class_weight={use_class_weight}")
    print("#" * 100)

    if use_bo:
        best_params = run_bayesian_optimization(X_train, y_train, use_class_weight=use_class_weight)
    else:
        best_params = get_default_params()
        print("使用默认参数：", best_params)

    model = build_cat_model(
        params=best_params,
        use_class_weight=use_class_weight,
        y_train=y_train,
        use_early_stopping=final_use_early_stopping,
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    result, y_pred, y_score, rga_metrics = evaluate_binary_metrics(model, X_test, y_test, exp_name=exp_name)

    cv_result, cv_detail = cross_validate_ranking_metrics(
        params=best_params,
        X_train=X_train,
        y_train=y_train,
        use_class_weight=use_class_weight,
        n_splits=10
    )

    print("\n--- Test Metrics ---")
    for k, v in result.items():
        if k != 'Experiment':
            print(f"{k}: {v}")
    if rga_metrics is not None:
        print(f"RGA Confusion Detail: {rga_metrics}")

    print("\n--- 10-Fold Cross-Validation Ranking Metrics ---")
    for k, v in cv_result.items():
        print(f"{k}: {v}")

    return {
        'name': exp_name,
        'model': model,
        'params': best_params,
        'result': result,
        'cv_result': cv_result,
        'cv_detail': cv_detail,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score
    }

# ==============================================
# SHAP 函数
# ==============================================
def get_mean_abs_shap(model, X_test, feature_names):
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    shap_values = _extract_shap_array(shap_values)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    return mean_abs.sort_values(ascending=False), shap_values, X_test_df


def plot_shap_outputs(model, X_test, feature_names, top_n=10, model_name='Main Model'):
    mean_abs_shap, shap_values, X_test_df = get_mean_abs_shap(model, X_test, feature_names)

    print(f"\n{model_name} Mean |SHAP|:")
    print(mean_abs_shap.to_string())

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type='bar', show=False)
    title1 = 'Feature Importance Ranking (SHAP)' if model_name == MAIN_MODEL_DISPLAY_NAME else f'{model_name} Feature Importance Ranking (SHAP)'
    plt.title(title1)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    title2 = 'SHAP Summary Plot' if model_name == MAIN_MODEL_DISPLAY_NAME else f'{model_name} SHAP Summary Plot'
    plt.title(title2)
    plt.tight_layout()
    plt.show()

    for feature_name in mean_abs_shap.head(top_n).index:
        display_name = feature_name
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(feature_name, shap_values, X_test_df, interaction_index=None, show=False)
        dep_title = f'SHAP Dependence Plot for {display_name}' if model_name == MAIN_MODEL_DISPLAY_NAME else f'{model_name} - SHAP Dependence Plot for {display_name}'
        plt.title(dep_title)
        plt.tight_layout()
        plt.show()

    return mean_abs_shap, shap_values, X_test_df

# ==============================================
# LOESS 分析（可选）
# ==============================================
def plot_loess_for_top_features(model, X_all, y_all, feature_names, top_features, frac_values=(0.2, 0.3, 0.4)):
    print("\n--- LOESS Local Weighted Regression Analysis ---")
    y_proba = model.predict_proba(X_all)[:, 1]

    fig, axes = plt.subplots(len(top_features), len(frac_values), figsize=(18, 4 * len(top_features)))
    if len(top_features) == 1:
        axes = np.array([axes])

    for i, feature in enumerate(top_features):
        feature_idx = feature_names.index(feature)
        display_name = feature
        x_vals = X_all[:, feature_idx]

        for j, frac in enumerate(frac_values):
            ax = axes[i, j] if len(top_features) > 1 else axes[j]
            loess_data = pd.DataFrame({'feature': x_vals, 'target_proba': y_proba}).sort_values('feature')
            loess_result = lowess(loess_data['target_proba'], loess_data['feature'], frac=frac)

            ax.scatter(loess_data['feature'], loess_data['target_proba'], alpha=0.3, s=10, color='gray')
            ax.plot(loess_result[:, 0], loess_result[:, 1], color='red', linewidth=2)
            ax.set_xlabel(f'{display_name} (Scaled)')
            ax.set_ylabel('Predicted Probability')
            ax.set_ylim([-0.1, 1.1])
            ax.set_title(f'LOESS Curve for {display_name} (frac={frac})')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('LOESS Local Weighted Regression Analysis', y=1.02, fontsize=16, fontweight='bold')
    plt.show()



def plot_single_feature_loess_bandwidths(model, X_all, feature_names, feature_name="EGR",
                                         frac_values=(0.1, 0.2, 0.3, 0.4, 0.5)):
    print(f"\n--- LOESS Curve with Different Bandwidth Parameters for {feature_name} ---")

    if feature_name not in feature_names:
        print(f"Feature {feature_name} not found in feature_names.")
        return

    y_proba = model.predict_proba(X_all)[:, 1]
    feature_idx = feature_names.index(feature_name)
    x_vals = X_all[:, feature_idx]

    loess_data = pd.DataFrame({
        'feature': x_vals,
        'target_proba': y_proba
    }).sort_values('feature')

    plt.figure(figsize=(8, 6))
    plt.scatter(loess_data['feature'], loess_data['target_proba'], alpha=0.25, s=10, color='gray')

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(frac_values)))

    for frac, color in zip(frac_values, colors):
        loess_result = lowess(loess_data['target_proba'], loess_data['feature'], frac=frac)
        plt.plot(loess_result[:, 0], loess_result[:, 1], linewidth=2, label=f'frac={frac}', color=color)

    plt.xlabel(f'{feature_name} (Scaled)')
    plt.ylabel('Target Value')
    plt.title(f'LOESS Curve with Different Bandwidth Parameters\nfor Feature {feature_name}')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(frac_values), vmax=max(frac_values)))
    sm.set_array([])
    plt.colorbar(sm, label='Bandwidth (frac)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================
# 主模型交叉验证图
# ==============================================
def plot_main_cv_f1_boxplot(fold_f1):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[fold_f1], width=0.35, color='#66c2a5')
    sns.swarmplot(data=[fold_f1], color='red', size=4, alpha=0.9)
    plt.xticks([0], ['10-Fold CV'])
    plt.ylabel('F1 Score')
    plt.title('K-Fold Cross-Validation Performance (Best Model)')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_f1_rga_trend(fold_f1, fold_rga):
    plt.figure(figsize=(10, 6))
    folds = [f'Fold {i + 1}' for i in range(len(fold_f1))]

    f1_mean = np.mean(fold_f1)
    f1_std = np.std(fold_f1)
    rga_mean = np.mean(fold_rga)
    rga_std = np.std(fold_rga)

    plt.plot(
        folds, fold_f1, marker='o', linewidth=2, markersize=5, color='#E64B35',
        label=f'F1 (Mean: {f1_mean:.5f} ± {f1_std:.5f})'
    )
    plt.plot(
        folds, fold_rga, marker='s', linewidth=2, markersize=5, color='#3CB371',
        label=f'RGA (Mean: {rga_mean:.5f} ± {rga_std:.5f})'
    )

    plt.xlabel('10-Fold Cross-Validation')
    plt.ylabel('Score')
    plt.title('10-Fold Cross-Validation: F1 vs RGA Trend')
    plt.xticks(rotation=45)
    plt.ylim(0.94, 1.0)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================
# 主程序
# ==============================================
def main():
    print("=" * 120)
    print("LGFV 二修完整版实验脚本启动")
    print("当前主线：保留‘先 SMOTE，再划分’")
    print("=" * 120)

    data_dict = load_and_prepare_data(DATA_PATH)
    feature_names = data_dict['feature_names']

    all_runs = []

    # Exp1 主模型：SMOTE + BO + CatBoost（保留你当前主线）
    run1 = run_experiment(
        exp_name='Exp1_SMOTE+BO+CatBoost',
        dataset_pack=data_dict['smote_before_split'],
        use_bo=True,
        use_class_weight=False
    )
    all_runs.append(run1)

    # Exp2 去掉 BO：SMOTE + CatBoost
    run2 = run_experiment(
        exp_name='Exp2_SMOTE+CatBoost',
        dataset_pack=data_dict['smote_before_split'],
        use_bo=False,
        use_class_weight=False
    )
    all_runs.append(run2)

    # Exp3 去掉 SMOTE：BO + CatBoost（原始数据）
    run3 = run_experiment(
        exp_name='Exp3_BO+CatBoost_raw',
        dataset_pack=data_dict['raw'],
        use_bo=True,
        use_class_weight=False
    )
    all_runs.append(run3)

    # Exp4 去掉 SMOTE 和 BO：原始 CatBoost
    run4 = run_experiment(
        exp_name='Exp4_CatBoost_raw',
        dataset_pack=data_dict['raw'],
        use_bo=False,
        use_class_weight=False,
        final_use_early_stopping=False
    )
    all_runs.append(run4)

    # Exp5 替代方案：class_weight + BO + CatBoost（原始数据）
    run5 = run_experiment(
        exp_name='Exp5_class_weight+BO+CatBoost',
        dataset_pack=data_dict['raw'],
        use_bo=True,
        use_class_weight=True
    )
    all_runs.append(run5)

    # ------------------------------
    # 汇总结果表
    # ------------------------------
    results_df = pd.DataFrame([x['result'] for x in all_runs])
    cv_results_df = pd.DataFrame([{'Experiment': x['name'], **x['cv_result']} for x in all_runs])
    print("\n" + "=" * 120)
    print("五组实验汇总结果")
    print("=" * 120)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("五组实验 10 折交叉验证汇总结果")
    print("=" * 120)
    print(cv_results_df.to_string(index=False))

    print("\n五组实验结果已输出到控制台，不导出Excel文件。")

    # ------------------------------
    # 汇总柱状图
    # ------------------------------
    plot_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AveragePrecision', 'P@50', 'P@100']
    plot_df = results_df.set_index('Experiment')[plot_cols]

    plot_df.plot(kind='bar', figsize=(14, 7))
    plt.title('Ablation Study and Alternative Baseline Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # 主模型详细输出
    # ------------------------------
    main_run = next(x for x in all_runs if x['name'] == MAIN_MODEL_NAME)
    main_model = main_run['model']
    main_X_test = main_run['X_test']
    main_y_test = main_run['y_test']
    main_y_pred = main_run['y_pred']
    main_y_score = main_run['y_score']

    plot_confusion(main_y_test, main_y_pred, title='Confusion Matrix')
    plot_roc(main_y_test, main_y_score, title='ROC Curve')
    plot_pr_curve(main_y_test, main_y_score, title='Precision-Recall Curve')

    # ------------------------------
    # 主模型 SAFE
    # ------------------------------
    print("\n" + "=" * 60)
    print("主模型 SAFE 评分")
    print("=" * 60)

    main_dataset = data_dict['smote_before_split']
    safe_calculator = SAFEScoreCalculator(
        model=main_model,
        X_train=main_dataset['X_train'],
        X_test=main_dataset['X_test'],
        y_train=main_dataset['y_train'],
        y_test=main_dataset['y_test'],
        feature_names=feature_names
    )
    safe_scores = safe_calculator.get_all_scores()
    for score_name, score_value in safe_scores.items():
        print(f'{score_name}: {score_value:.6f}')

    # ------------------------------
    # 主模型 10 折交叉验证（含排序指标）
    # ------------------------------
    print("\n" + "=" * 60)
    print("主模型 10-fold CV（含排序指标）")
    print("=" * 60)

    main_cv_result = main_run['cv_result']
    main_cv_detail = main_run['cv_detail']
    for k, v in main_cv_result.items():
        print(f'{k}: {v}')

    plot_main_cv_f1_boxplot(main_cv_detail['fold_f1'])
    plot_f1_rga_trend(main_cv_detail['fold_f1'], main_cv_detail['fold_rga'])

    plt.figure(figsize=(12, 6))
    metric_arrays = [
        main_cv_detail['fold_f1'],
        main_cv_detail['fold_rga'],
        main_cv_detail['fold_ap'],
        main_cv_detail['fold_p50'],
        main_cv_detail['fold_p100']
    ]
    sns.boxplot(data=metric_arrays, width=0.5)
    sns.swarmplot(data=metric_arrays, color='black', size=4, alpha=0.7)
    plt.xticks([0, 1, 2, 3, 4], ['F1', 'RGA', 'AP', 'P@50', 'P@100'])
    plt.title('10-Fold Cross-Validation Metrics')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # SHAP：主模型
    # ------------------------------
    print("\n" + "=" * 60)
    print("主模型 SHAP 分析")
    print("=" * 60)

    main_mean_abs_shap, main_shap_values, main_X_test_df = plot_shap_outputs(
        main_model, main_X_test, feature_names, top_n=10, model_name=MAIN_MODEL_DISPLAY_NAME
    )

    # ------------------------------
    # SHAP 稳健性比较：主模型 vs class_weight 替代组
    # ------------------------------
    print("\n" + "=" * 60)
    print("SHAP 稳健性比较（主模型 vs class_weight 替代组）")
    print("=" * 60)

    alt_run = next(x for x in all_runs if x['name'] == 'Exp5_class_weight+BO+CatBoost')
    alt_mean_abs_shap, _, _ = get_mean_abs_shap(alt_run['model'], alt_run['X_test'], feature_names)

    top10_main = set(main_mean_abs_shap.head(10).index)
    top10_alt = set(alt_mean_abs_shap.head(10).index)
    overlap = len(top10_main & top10_alt)

    print('Main Top10 Features:', list(main_mean_abs_shap.head(10).index))
    print('Alt  Top10 Features:', list(alt_mean_abs_shap.head(10).index))
    print(f'Top10 overlap count = {overlap}/10')

    shap_compare_df = pd.DataFrame({
        'Main_SHAP': main_mean_abs_shap,
        'ClassWeight_SHAP': alt_mean_abs_shap
    }).fillna(0)

    corr_val = shap_compare_df['Main_SHAP'].corr(shap_compare_df['ClassWeight_SHAP'], method='spearman')
    print(f'Spearman correlation of SHAP importance = {corr_val:.6f}')

    plt.figure(figsize=(8, 8))
    plt.scatter(shap_compare_df['Main_SHAP'], shap_compare_df['ClassWeight_SHAP'], alpha=0.7)
    for feat in shap_compare_df.index:
        plt.text(
            shap_compare_df.loc[feat, 'Main_SHAP'],
            shap_compare_df.loc[feat, 'ClassWeight_SHAP'],
            feat,
            fontsize=8
        )
    plt.xlabel('Main Model Mean |SHAP|')
    plt.ylabel('ClassWeight Model Mean |SHAP|')
    plt.title('SHAP Importance Robustness Comparison')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # LOESS（可选）
    # ------------------------------
    if ENABLE_LOESS:
        top5_features = list(main_mean_abs_shap.head(5).index)
        plot_loess_for_top_features(
            model=main_model,
            X_all=data_dict['smote_before_split']['X_all'],
            y_all=data_dict['smote_before_split']['y_all'],
            feature_names=feature_names,
            top_features=top5_features,
            frac_values=(0.2, 0.3, 0.4)
        )

        plot_single_feature_loess_bandwidths(
            model=main_model,
            X_all=data_dict['smote_before_split']['X_all'],
            feature_names=feature_names,
            feature_name='EGR',
            frac_values=(0.1, 0.2, 0.3, 0.4, 0.5)
        )

    print("\n脚本运行完成。")


if __name__ == '__main__':
    main()
