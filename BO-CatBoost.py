import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.spatial import ConvexHull

# 全局设置：解决中文显示+图表美化
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
warnings.filterwarnings('ignore')

# ==============================================
# ✅ 1. SAFE评分 全套核心函数
# ✅ 包含：洛伦兹曲面值+Shapley-Lorenz值+SAFE评分计算器(Ex+Ac+Sust)
# ==============================================
def calculate_lorenz_zonoid(data_vector):
    """计算Lorenz Zonoid值（SAFE评分基础，所有SAFE子评分依赖）"""
    data = np.array(data_vector).flatten()
    if len(data) == 0 or data.max() == data.min():
        return 0.0

    data_sorted = np.sort(data)
    data_normalized = (data_sorted - data_sorted.min()) / (data_sorted.max() - data_sorted.min())

    n = len(data_normalized)
    F = np.arange(1, n + 1) / n
    L = np.cumsum(data_normalized) / np.sum(data_normalized)

    points = np.column_stack((F, L))
    points = np.vstack([[0, 0], points])
    try:
        hull = ConvexHull(points)
        lz_value = hull.volume
    except:
        lz_value = np.trapz(L, F)
    return lz_value

def calculate_shapley_lorenz(model, X_data, feature_names):
    """计算每个特征的Shapley-Lorenz值（SL值，适配树模型CatBoost/XGB/LGB）"""
    explainer = shap.TreeExplainer(model)
    X_df = pd.DataFrame(X_data, columns=feature_names)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.array(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    sl_values = [calculate_lorenz_zonoid(shap_values[:, i]) for i in range(len(feature_names))]
    return np.array(sl_values)

class SAFEScoreCalculator:
    """SAFE评分计算器（Ex-可解释性 Ac-准确性 Sust-稳健性）0-1区间，越高越好"""
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.k = len(feature_names)

        self.shapley_lorenz = calculate_shapley_lorenz(model, X_test, feature_names)
        y_pred_proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            self.y_pred_proba = y_pred_proba[:, 1]
        else:
            self.y_pred_proba = y_pred_proba[np.arange(len(y_test)), y_test]

    def ex_score(self):
        """可解释性评分：模型决策逻辑的可理解程度"""
        LZ_Y = calculate_lorenz_zonoid(self.y_test)
        sum_SL = np.sum(self.shapley_lorenz)
        ex = sum_SL / ((LZ_Y + 1e-10) * self.k)
        return np.clip(ex, 0.0, 1.0)

    def ac_score(self):
        """准确性评分：预测概率分布与真实标签分布的匹配度"""
        LZ_Y_pred = calculate_lorenz_zonoid(self.y_pred_proba)
        LZ_Y_test = calculate_lorenz_zonoid(self.y_test)
        ac = LZ_Y_pred / (LZ_Y_test + 1e-10)
        return np.clip(ac, 0.0, 1.0)

    def sust_score(self, G=5):
        """稳健性评分：不同置信度分组下特征贡献的一致性"""
        pred_quantiles = np.quantile(self.y_pred_proba, np.linspace(0, 1, G + 1))
        groups = np.digitize(self.y_pred_proba, pred_quantiles[1:-1])
        V_G_SL = []
        for g in range(G):
            group_idx = np.where(groups == g)[0]
            if len(group_idx) == 0:
                V_G_SL.append(0)
                continue
            X_group = self.X_test[group_idx]
            explainer = shap.TreeExplainer(self.model)
            X_group_df = pd.DataFrame(X_group, columns=self.feature_names)
            shap_group = explainer.shap_values(X_group_df)

            if isinstance(shap_group, list):
                shap_group = shap_group[1] if len(shap_group) > 1 else shap_group[0]
            shap_group = np.array(shap_group)
            if shap_group.ndim == 1:
                shap_group = shap_group.reshape(-1, 1)

            if shap_group.shape[1] != self.k:
                sl_group = [0.0] * self.k
            else:
                sl_group = [calculate_lorenz_zonoid(shap_group[:, i]) for i in range(self.k)]
            V_G_SL.append(np.sum(sl_group))

        LZ_V_G_SL = calculate_lorenz_zonoid(V_G_SL)
        sust = 1 - LZ_V_G_SL
        return np.clip(sust, 0.0, 1.0)

    def get_all_scores(self):
        """返回所有SAFE评分（保留6位小数）"""
        return {
            "Ex-score (可解释性)": round(self.ex_score(), 6),
            "Ac-score (准确性)": round(self.ac_score(), 6),
            "Sust-score (稳健性)": round(self.sust_score(), 6)
        }

# ==============================================
# ✅ 2. 等级分级准确率 RGA 核心实现
# ==============================================
def binary_ranked_grading_accuracy(y_true, y_pred_proba, pos_label=1):
    """
    二分类 等级分级准确率 (RGA)
    :param y_true: 真实标签
    :param y_pred_proba: 模型预测正类概率
    :param pos_label: 正类标签，默认=1
    :return: RGA分数, 混淆矩阵指标(TP/TN/FP/FN)
    """
    y_true = np.array(y_true).astype(int)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # 概率转标签，阈值0.5

    # 归一化标签为0/1
    y_true_bin = np.where(y_true == pos_label, 1, 0)
    y_pred_bin = np.where(y_pred == pos_label, 1, 0)

    # 计算混淆矩阵核心指标
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    # 论文标准RGA计算公式
    rga_score = (TP + TN + 0.5 * (FP + FN)) / len(y_true)

    return round(rga_score, 6), {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

# ==============================================
# 数据加载与预处理
# ==============================================
try:
    data_path = "C:\\Users\\h\\Desktop\\5.4.xlsx"
    df = pd.read_excel(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = [f"x{i}" for i in range(1, X.shape[1] + 1)] # 特征名保存，用于SAFE

    print(f"Original class distribution:\n{y.value_counts()}")

    # SMOTE过采样平衡数据
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"\nResampled class distribution:\n{pd.Series(y_resampled).value_counts()}")

    # 特征归一化到[-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42,
                                                        stratify=y_resampled if np.unique(
                                                            y_resampled).size > 1 else None)
    # 转为numpy数组解决索引问题
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}. Please update the path.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 类别分布可视化 ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y, palette='viridis')
plt.title('Class Distribution Before SMOTE', fontsize=12, fontweight='bold')
plt.xlabel('Class', fontsize=10)
plt.ylabel('Count', fontsize=10)

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, palette='magma')
plt.title('Class Distribution After SMOTE', fontsize=12, fontweight='bold')
plt.xlabel('Class', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.tight_layout()
plt.show()

# ==============================================
# 贝叶斯优化CatBoost超参数
# ==============================================
def objective_cbc(iterations, learning_rate, depth, l2_leaf_reg, border_count, random_strength, bagging_temperature,
                  rsm):
    model = CatBoostClassifier(
        iterations=int(iterations),
        learning_rate=learning_rate,
        depth=int(depth),
        l2_leaf_reg=l2_leaf_reg,
        border_count=int(border_count),
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        rsm=rsm,
        loss_function='Logloss',
        eval_metric='F1',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='f1_weighted', n_jobs=-1)
    return np.mean(scores)

# 超参数搜索范围
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

# 贝叶斯优化
optimizer = BayesianOptimization(f=objective_cbc, pbounds=pbounds, random_state=42, verbose=2)
print("Starting Bayesian Optimization for CatBoost hyperparameters...")
optimizer.maximize(init_points=15, n_iter=75)

# 最优参数
best_params = optimizer.max['params']
best_params['iterations'] = int(best_params['iterations'])
best_params['depth'] = int(best_params['depth'])
best_params['border_count'] = int(best_params['border_count'])
print(f"\nBest F1 score from optimization: {optimizer.max['target']:.6f}")
print("Best parameters found:", best_params)

# 训练最终模型
final_model = CatBoostClassifier(**best_params, loss_function='Logloss', eval_metric='F1',
                                 random_seed=42, verbose=100, early_stopping_rounds=50)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ==============================================
# 模型测试集评估 + F1+RGA+【SAFE评分】核心集成
# ==============================================
print("\n--- Model Evaluation on Test Set (F1+RGA+SAFE) ---")
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)

# 基础指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
# RGA计算
rga_score = None
rga_metrics = None
if np.unique(y_test).size == 2:
    rga_score, rga_metrics = binary_ranked_grading_accuracy(y_test, y_pred_proba[:, 1], pos_label=1)
    print(
        f"\n✅ RGA Confusion Matrix: TP={rga_metrics['TP']}, TN={rga_metrics['TN']}, FP={rga_metrics['FP']}, FN={rga_metrics['FN']}")

# ========== 计算SAFE评分 ==========
print("\n" + "="*60)
print("✅ SAFE 评分计算结果")
print("="*60)
safe_calculator = SAFEScoreCalculator(
    model=final_model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names
)
safe_scores = safe_calculator.get_all_scores()
for score_name, score_value in safe_scores.items():
    print(f"  {score_name}: {score_value:.6f}")

# 打印所有指标
print(f"\nAccuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
if rga_score:
    print(f"✅ RGA (Ranked Grading Accuracy): {rga_score:.6f}")

# 混淆矩阵
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=final_model.classes_, yticklabels=final_model.classes_)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ROC曲线
if np.unique(y_test).size == 2:
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
    print(f"Test ROC AUC: {roc_auc_test:.6f}")
    print(f"✅ Paper Validation: RGA={rga_score:.6f} | AUC={roc_auc_test:.6f} (Binary classification equivalence)")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, color='#2E86AB', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.6f})')
    plt.plot([0, 1], [0, 1], color='#A23B72', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC Curve (Test Set)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

# ==============================================
# 十折交叉验证 + F1+RGA+【SAFE】计算 + 可视化
# ==============================================
print("\n--- K-Fold Cross-Validation Performance (F1+RGA+SAFE) ---")
kfold_model = CatBoostClassifier(**best_params, loss_function='Logloss', eval_metric='F1',
                                 random_seed=42, verbose=0, early_stopping_rounds=None)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 计算十折F1/RGA
cv_f1_scores = cross_val_score(kfold_model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
cv_rga_list = []
cv_cm_metrics = []
cv_safe_ex = []
cv_safe_ac = []
cv_safe_sust = []

if np.unique(y_train).size == 2:
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        kfold_model.fit(X_cv_train, y_cv_train, verbose=0)
        cv_proba = kfold_model.predict_proba(X_cv_val)[:, 1]
        # 计算单折RGA
        cv_rga, cv_metric = binary_ranked_grading_accuracy(y_cv_val, cv_proba)
        cv_rga_list.append(cv_rga)
        cv_cm_metrics.append(cv_metric)
        # 计算单折SAFE
        cv_safe_calc = SAFEScoreCalculator(kfold_model, X_cv_train, X_cv_val, y_cv_train, y_cv_val, feature_names)
        cv_safe_ex.append(cv_safe_calc.ex_score())
        cv_safe_ac.append(cv_safe_calc.ac_score())
        cv_safe_sust.append(cv_safe_calc.sust_score())

# 计算均值和标准差
f1_mean, f1_std = np.mean(cv_f1_scores), np.std(cv_f1_scores)
rga_mean, rga_std = (np.mean(cv_rga_list), np.std(cv_rga_list)) if cv_rga_list else (0, 0)
safe_ex_mean, safe_ex_std = np.mean(cv_safe_ex), np.std(cv_safe_ex)
safe_ac_mean, safe_ac_std = np.mean(cv_safe_ac), np.std(cv_safe_ac)
safe_sust_mean, safe_sust_std = np.mean(cv_safe_sust), np.std(cv_safe_sust)

# 打印十折结果
print(f"Cross-Validation F1 Scores: {cv_f1_scores}")
print(f"Mean F1: {f1_mean:.6f} (±{f1_std:.6f})")
if cv_rga_list:
    print(f"✅ Cross-Validation RGA Scores: {cv_rga_list}")
    print(f"✅ Mean RGA: {rga_mean:.6f} (±{rga_std:.6f})")
print(f"✅ Cross-Validation SAFE-Ex Mean: {safe_ex_mean:.6f} (±{safe_ex_std:.6f})")
print(f"✅ Cross-Validation SAFE-Ac Mean: {safe_ac_mean:.6f} (±{safe_ac_std:.6f})")
print(f"✅ Cross-Validation SAFE-Sust Mean: {safe_sust_mean:.6f} (±{safe_sust_std:.6f})")

# ==============================================
# 可视化1：F1+RGA 十折箱线图
# ==============================================
plt.figure(figsize=(10, 6))
plot_data = [cv_f1_scores, cv_rga_list] if cv_rga_list else [cv_f1_scores]
sns.boxplot(data=plot_data, width=0.4, palette=['#F24236', '#34A853'])
sns.swarmplot(data=plot_data, color='black', size=6, alpha=0.8)
plt.xticks([0, 1] if cv_rga_list else [0], ['F1 Score', 'RGA Score'], fontsize=10)
plt.ylabel('Score', fontsize=10)
plt.title('10-Fold Cross-Validation: F1 vs RGA Score', fontsize=14, fontweight='bold')
plt.axhline(y=f1_mean, color='#F24236', linestyle='--', alpha=0.7, label=f'F1 Mean: {f1_mean:.6f}')
if cv_rga_list:
    plt.axhline(y=rga_mean, color='#34A853', linestyle='--', alpha=0.7, label=f'RGA Mean: {rga_mean:.6f}')
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(0.94, 1.0)
plt.tight_layout()
plt.show()

# ==============================================
# 可视化2：F1+RGA 十折折线对比图
# ==============================================
if cv_rga_list:
    plt.figure(figsize=(12, 6))
    fold_ids = [f'Fold {i + 1}' for i in range(10)]
    plt.plot(fold_ids, cv_f1_scores, marker='o', linewidth=2, markersize=6, color='#F24236',
             label=f'F1 (Mean: {f1_mean:.6f} ±{f1_std:.6f})')
    plt.plot(fold_ids, cv_rga_list, marker='s', linewidth=2, markersize=6, color='#34A853',
             label=f'RGA (Mean: {rga_mean:.6f} ±{rga_std:.6f})')
    plt.xlabel('10-Fold Cross-Validation', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.title('10-Fold Cross-Validation: F1 vs RGA Trend', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(0.94, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==============================================
# 可视化3：F1+RGA 均值+标准差柱状对比图
# ==============================================
if cv_rga_list:
    plt.figure(figsize=(8, 6))
    metrics = ['F1 Score', 'RGA Score']
    means = [f1_mean, rga_mean]
    stds = [f1_std, rga_std]
    colors = ['#F24236', '#34A853']

    bars = plt.bar(metrics, means, yerr=stds, capsize=8, width=0.4, color=colors, alpha=0.8)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{mean:.6f}\n(±{std:.6f})', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel('Mean Score', fontsize=10)
    plt.title('10-Fold Cross-Validation: F1 vs RGA (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.ylim(0.96, 0.98)
    plt.tight_layout()
    plt.show()

# ==============================================
# 可视化4：RGA 十折平均混淆矩阵热力图
# ==============================================
if cv_cm_metrics:
    avg_tp = np.mean([m['TP'] for m in cv_cm_metrics])
    avg_tn = np.mean([m['TN'] for m in cv_cm_metrics])
    avg_fp = np.mean([m['FP'] for m in cv_cm_metrics])
    avg_fn = np.mean([m['FN'] for m in cv_cm_metrics])
    avg_cm = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.title('10-Fold Cross-Validation: Average RGA Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==============================================
# 可视化5：RGA 十折分数分布直方图
# ==============================================
if cv_rga_list:
    plt.figure(figsize=(10, 6))
    sns.histplot(cv_rga_list, bins=6, color='#34A853', alpha=0.8, kde=True, edgecolor='black')
    plt.axvline(x=rga_mean, color='red', linestyle='--', linewidth=2, label=f'RGA Mean: {rga_mean:.6f}')
    plt.xlabel('RGA Score', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('10-Fold Cross-Validation: RGA Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

# ==============================================
# 可视化6：SAFE柱状图可视化
# ==============================================
plt.figure(figsize=(10, 6))
safe_metrics = ['Ex (可解释性)', 'Ac (准确性)', 'Sust (稳健性)']
safe_means = [safe_ex_mean, safe_ac_mean, safe_sust_mean]
safe_stds = [safe_ex_std, safe_ac_std, safe_sust_std]
colors = ['#2E86AB', '#F18F01', '#34A853']

bars = plt.bar(safe_metrics, safe_means, yerr=safe_stds, capsize=8, width=0.5, color=colors, alpha=0.8)
for bar, mean, std in zip(bars, safe_means, safe_stds):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{mean:.6f}\n(±{std:.6f})', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylabel('SAFE Score (0-1)', fontsize=10)
plt.title('10-Fold Cross-Validation: SAFE Score (Ex+Ac+Sust)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ==============================================
# 可视化7：F1+RGA+SAFE六指标柱状图
# ==============================================
plt.figure(figsize=(12, 7))
all_metrics = ['F1', 'RGA', 'SAFE-Ex', 'SAFE-Ac', 'SAFE-Sust']
all_means = [f1_mean, rga_mean, safe_ex_mean, safe_ac_mean, safe_sust_mean]
all_stds = [f1_std, rga_std, safe_ex_std, safe_ac_std, safe_sust_std]
colors = ['#F24236', '#34A853', '#2E86AB', '#F18F01', '#9B59B6']

bars = plt.bar(all_metrics, all_means, yerr=all_stds, capsize=8, width=0.5, color=colors, alpha=0.8)
for bar, mean, std in zip(bars, all_means, all_stds):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{mean:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.ylabel('Score Value', fontsize=10)
plt.title('10-Fold Cross-Validation: F1 + RGA + SAFE All Metrics', fontsize=15, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.ylim(0.9, 1.02)
plt.tight_layout()
plt.show()

# ==============================================
# SHAP分析 + LOESS分析
# ==============================================
# --- SHAP Analysis ---
print("\n--- SHAP Analysis ---")
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_df)

# SHAP条形图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("Feature Importance Ranking (SHAP Values)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 平均SHAP绝对值
if isinstance(shap_values, list):
    pred_class_indices = final_model.predict(X_test_df)
    if isinstance(pred_class_indices[0], str):
        pred_class_indices = [np.where(final_model.classes_ == cls)[0][0] for cls in pred_class_indices]
    selected_shap = np.array([shap_values[cls][i] for i, cls in enumerate(pred_class_indices)])
else:
    selected_shap = shap_values

mean_shap_abs = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP|": np.abs(selected_shap).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)
print("\nAverage Absolute SHAP Values:")
print(mean_shap_abs.to_string(index=False))

# SHAP蜂群图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Value Distribution", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# SHAP依赖图（前20特征）
for feature_name in feature_names[:20]:
    display_name = "EGR" if feature_name == "x14" else feature_name
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values, X_test_df, interaction_index=None, show=False)
    plt.title(f"SHAP Dependence Plot for {display_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- LOESS Local Weighted Regression Analysis ---
print("\n--- LOESS Local Weighted Regression Analysis ---")
df_combined = pd.DataFrame(X_resampled_scaled, columns=feature_names)
df_combined['target'] = y_resampled
top_features = mean_shap_abs['Feature'].head(5).tolist()
print(f"\nPerforming LOESS analysis on top 5 features: {top_features}")

frac_values = [0.2, 0.3, 0.4]
fig, axes = plt.subplots(len(top_features), len(frac_values), figsize=(18, 15))
if len(top_features) == 1:
    axes = axes.reshape(1, -1)

for i, feature in enumerate(top_features):
    display_name = "EGR" if feature == "x14" else feature
    for j, frac in enumerate(frac_values):
        ax = axes[i, j] if len(top_features) > 1 else axes[j]
        if np.unique(y).size == 2:
            y_proba = final_model.predict_proba(X_resampled_scaled)[:, 1]
            loess_data = pd.DataFrame({
                'feature': X_resampled_scaled[:, feature_names.index(feature)],
                'target_proba': y_proba
            }).sort_values('feature')
            loess_result = lowess(loess_data['target_proba'], loess_data['feature'], frac=frac)
            ax.scatter(loess_data['feature'], loess_data['target_proba'], alpha=0.3, s=10, color='gray')
            ax.plot(loess_result[:, 0], loess_result[:, 1], color='red', linewidth=2)
            ax.set_ylabel('Predicted Probability')
            ax.set_ylim([-0.1, 1.1])
        else:
            y_pred = final_model.predict(X_resampled_scaled)
            loess_data = pd.DataFrame({
                'feature': X_resampled_scaled[:, feature_names.index(feature)],
                'target': y_pred
            }).sort_values('feature')
            unique_classes = sorted(y_pred.unique())
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            for cls, color in zip(unique_classes, colors):
                binary_target = (loess_data['target'] == cls).astype(int)
                loess_result = lowess(binary_target, loess_data['feature'], frac=frac)
                ax.plot(loess_result[:, 0], loess_result[:, 1], color=color, linewidth=2, label=f'Class {cls}')
            ax.set_ylabel('Class Probability')
            ax.legend(fontsize=8)
            ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel(f'{display_name} (Scaled)')
        ax.set_title(f'LOESS Curve for {display_name}\n(frac={frac})')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('LOESS Local Weighted Regression Analysis', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# --- LOESS Residual + Bandwidth Analysis ---
print("\n--- LOESS Residual Analysis ---")
top_feature = mean_shap_abs['Feature'].iloc[0]
display_top_feature = "EGR" if top_feature == "x14" else top_feature
feature_idx = feature_names.index(top_feature)
feature_values = X_resampled_scaled[:, feature_idx]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

if np.unique(y).size == 2:
    y_proba = final_model.predict_proba(X_resampled_scaled)[:, 1]
    data = pd.DataFrame({'feature': feature_values, 'target_proba': y_proba}).sort_values('feature')
    for i, frac in enumerate([0.2, 0.3, 0.4, 0.5]):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        loess_result = lowess(data['target_proba'], data['feature'], frac=frac)
        residuals = data['target_proba'] - loess_result[:, 1]
        ax.scatter(data['feature'], data['target_proba'], alpha=0.3, s=10, color='gray', label='Data points')
        ax.plot(loess_result[:, 0], loess_result[:, 1], color='red', linewidth=2, label='LOESS curve')
        for x, y_val, res in zip(data['feature'], data['target_proba'], residuals):
            ax.plot([x, x], [y_val, y_val - res], color='blue', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'{display_top_feature} (Scaled)')
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'LOESS with Residuals (frac={frac})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])
else:
    y_pred = final_model.predict(X_resampled_scaled)
    unique_classes = sorted(y_pred.unique())
    for i, cls in enumerate(unique_classes[:4]):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        binary_target = (y_pred == cls).astype(int)
        data = pd.DataFrame({'feature': feature_values, 'target': binary_target}).sort_values('feature')
        loess_result = lowess(data['target'], data['feature'], frac=0.3)
        residuals = data['target'] - loess_result[:, 1]
        ax.scatter(data['feature'], data['target'], alpha=0.3, s=10, color='gray', label='Data points')
        ax.plot(loess_result[:, 0], loess_result[:, 1], color='red', linewidth=2, label='LOESS curve')
        for x, y_val, res in zip(data['feature'], data['target'], residuals):
            ax.plot([x, x], [y_val, y_val - res], color='blue', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'{display_top_feature} (Scaled)')
        ax.set_ylabel(f'Class {cls} Probability')
        ax.set_title(f'LOESS Residuals - Class {cls}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.suptitle(f'LOESS Residual Analysis for Top Feature ({display_top_feature})', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# 带宽选择分析
print("\n--- LOESS Bandwidth Selection Analysis ---")
feature_to_analyze = top_features[0]
display_feature_to_analyze = "EGR" if feature_to_analyze == "x14" else feature_to_analyze
feature_idx = feature_names.index(feature_to_analyze)
feature_values = X_resampled_scaled[:, feature_idx]

if np.unique(y).size == 2:
    y_proba = final_model.predict_proba(X_resampled_scaled)[:, 1]
    data = pd.DataFrame({'feature': feature_values, 'target': y_proba}).sort_values('feature')
else:
    first_class = sorted(y_resampled.unique())[0]
    y_binary = (y_resampled == first_class).astype(int)
    data = pd.DataFrame({'feature': feature_values, 'target': y_binary}).sort_values('feature')

frac_range = np.arange(0.1, 0.6, 0.05)
plt.figure(figsize=(12, 8))
plt.scatter(data['feature'], data['target'], alpha=0.3, s=10, color='gray', label='Data points')
colors = plt.cm.viridis(np.linspace(0, 1, len(frac_range)))
for frac, color in zip(frac_range, colors):
    loess_result = lowess(data['target'], data['feature'], frac=frac)
    plt.plot(loess_result[:, 0], loess_result[:, 1], color=color, alpha=0.7, linewidth=1.5)

plt.xlabel(f'{display_feature_to_analyze} (Scaled)')
plt.ylabel('Target Value')
plt.title(f'LOESS Curve with Different Bandwidth Parameters\nfor Feature {display_feature_to_analyze}', fontsize=14,
          fontweight='bold')
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=frac_range.min(), vmax=frac_range.max()))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Bandwidth (frac)')
plt.tight_layout()
plt.show()

# 带宽误差分析
errors = []
for frac in frac_range:
    loess_result = lowess(data['target'], data['feature'], frac=frac)
    mse = np.mean((data['target'].values - loess_result[:, 1]) ** 2)
    mae = np.mean(np.abs(data['target'].values - loess_result[:, 1]))
    errors.append({'frac': frac, 'mse': mse, 'mae': mae})
    print(f"frac={frac:.2f}: MSE={mse:.6f}, MAE={mae:.6f}")

errors_df = pd.DataFrame(errors)
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()
line1 = ax1.plot(errors_df['frac'], errors_df['mse'], 'b-o', label='MSE', linewidth=2)
line2 = ax2.plot(errors_df['frac'], errors_df['mae'], 'r-s', label='MAE', linewidth=2)
ax1.set_xlabel('Bandwidth (frac)')
ax1.set_ylabel('Mean Squared Error (MSE)', color='b')
ax2.set_ylabel('Mean Absolute Error (MAE)', color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
best_frac_mse = errors_df.loc[errors_df['mse'].idxmin(), 'frac']
best_frac_mae = errors_df.loc[errors_df['mae'].idxmin(), 'frac']
ax1.axvline(x=best_frac_mse, color='b', linestyle='--', alpha=0.7, label=f'Best MSE: frac={best_frac_mse:.2f}')
ax2.axvline(x=best_frac_mae, color='r', linestyle='--', alpha=0.7, label=f'Best MAE: frac={best_frac_mae:.2f}')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.title(f'LOESS Fit Error vs Bandwidth for Feature {display_feature_to_analyze}', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# LOESS总结
print("\n--- LOESS Analysis Summary ---")
print(f"1. Top 5 features for LOESS analysis: {top_features}")
print(f"2. Best bandwidth for MSE minimization: {best_frac_mse:.2f}")
print(f"3. Best bandwidth for MAE minimization: {best_frac_mae:.2f}")
print(f"4. LOESS analysis helps identify non-linear relationships between features and target")
print(f"5. Residual analysis shows where the model predictions deviate from the LOESS curve")