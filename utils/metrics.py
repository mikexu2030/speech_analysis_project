"""
评估指标计算
支持: EER, UAR, WA, MAE, 准确率
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from typing import Tuple, List


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    计算等错误率 (Equal Error Rate)
    
    Args:
        scores: 相似度分数 (越高越相似)
        labels: 1=同一人, 0=不同人
    
    Returns:
        (eer, threshold)
    """
    # 按分数排序
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # 计算FPR和FNR
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    fnrs = np.zeros(len(scores) + 1)
    fprs = np.zeros(len(scores) + 1)
    
    # 遍历所有可能的阈值
    for i in range(len(scores)):
        fnrs[i] = np.sum(sorted_labels[i:] == 1) / n_pos if n_pos > 0 else 0
        fprs[i] = np.sum(sorted_labels[:i] == 0) / n_neg if n_neg > 0 else 0
    
    # 找到FPR和FNR最接近的点
    diff = np.abs(fprs - fnrs)
    min_idx = np.argmin(diff)
    
    eer = (fprs[min_idx] + fnrs[min_idx]) / 2
    threshold = sorted_scores[min_idx] if min_idx < len(sorted_scores) else 0.0
    
    return eer, threshold


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算无加权平均召回率 (Unweighted Average Recall)
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        UAR (0-1)
    """
    cm = confusion_matrix(y_true, y_pred)
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    uar = np.mean(recalls)
    return uar


def compute_war(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算加权平均召回率 (Weighted Average Recall) = 准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        WAR (0-1)
    """
    return accuracy_score(y_true, y_pred)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        准确率 (0-1)
    """
    return accuracy_score(y_true, y_pred)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                              labels: List[str] = None) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签名称
    
    Returns:
        混淆矩阵
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: List[str] = None) -> dict:
    """
    计算每个类别的详细指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    
    Returns:
        每个类别的precision, recall, f1
    """
    cm = confusion_matrix(y_true, y_pred)
    
    n_classes = cm.shape[0]
    metrics = {}
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(cm[i, :])
        }
    
    return metrics


def compute_speaker_verification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    计算说话人验证的完整指标
    
    Args:
        scores: 相似度分数
        labels: 1=同一人, 0=不同人
    
    Returns:
        包含EER, minDCF, threshold的字典
    """
    eer, threshold = compute_eer(scores, labels)
    
    # 计算minDCF (简化版)
    # 使用P_target=0.01, C_miss=1, C_fa=1
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    min_dcf = float('inf')
    best_threshold = 0.0
    
    for i in range(len(scores)):
        fn = np.sum(sorted_labels[i:] == 1)
        fp = np.sum(sorted_labels[:i] == 0)
        
        miss_rate = fn / n_pos if n_pos > 0 else 0
        fa_rate = fp / n_neg if n_neg > 0 else 0
        
        # DCF = 0.01 * miss_rate + 0.99 * fa_rate
        dcf = 0.01 * miss_rate + 0.99 * fa_rate
        
        if dcf < min_dcf:
            min_dcf = dcf
            best_threshold = sorted_scores[i]
    
    return {
        'eer': eer,
        'min_dcf': min_dcf,
        'threshold': threshold,
        'best_threshold': best_threshold
    }


def compute_age_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算年龄估计的指标
    
    Args:
        y_true: 真实年龄
        y_pred: 预测年龄
    
    Returns:
        包含MAE, RMSE, 5年准确率, 10年准确率
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 5年准确率
    acc_5yr = np.mean(np.abs(y_true - y_pred) <= 5)
    # 10年准确率
    acc_10yr = np.mean(np.abs(y_true - y_pred) <= 10)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'acc_5yr': acc_5yr,
        'acc_10yr': acc_10yr
    }


if __name__ == "__main__":
    # 测试
    print("Metrics module loaded successfully")
    print("Functions: compute_eer, compute_uar, compute_war, compute_accuracy, compute_mae")
    
    # 简单测试
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    
    print(f"UAR: {compute_uar(y_true, y_pred):.4f}")
    print(f"WAR: {compute_war(y_true, y_pred):.4f}")
    
    # EER测试
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    labels = np.array([1, 1, 0, 0, 1, 0])
    eer, thresh = compute_eer(scores, labels)
    print(f"EER: {eer:.4f}, Threshold: {thresh:.4f}")
