
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_curve, auc, 
    precision_recall_curve,
    confusion_matrix, 
    classification_report
)
from torch.utils.data import Dataset, DataLoader

from models.model import CosmeticEffectPredictorWithTokenizer


class CosmeticDataset(Dataset):
    def __init__(self, ingredients_lists, labels):
        """
        화장품 데이터셋 클래스
        Args:
            ingredients_lists: 성분 리스트의 리스트
            labels: 레이블 텐서 (num_samples, num_classes)
        """
        self.ingredients_lists = ingredients_lists
        self.labels = labels
        
    def __len__(self):
        return len(self.ingredients_lists)
    
    def __getitem__(self, idx):
        return self.ingredients_lists[idx], self.labels[idx]


def evaluate_model(model, test_data, effect_names, batch_size=32, num_workers=4, threshold=0.5, output_dir=None):
    """
    모델 평가 및 성능 지표 계산
    Args:
        model: 평가할 모델 (CosmeticEffectPredictorWithTokenizer)
        test_data: (ingredients_lists, labels) 튜플
        effect_names: 효과 이름 리스트
        batch_size: 배치 크기
        num_workers: 데이터 로더 워커 수
        threshold: 분류 임계값
        output_dir: 평가 결과 저장 디렉토리
    Returns:
        평가 지표 사전
    """
    ingredients_lists, labels = test_data
    
    # 토크나이저 추출
    tokenizer = model.tokenizer
    device = model.device
    
    # 데이터셋 및 데이터로더 생성
    test_dataset = CosmeticDataset(ingredients_lists, labels)
    
    # 콜레이트 함수
    def collate_fn(batch):
        ingredients_lists, labels = zip(*batch)
        
        # 토큰화
        input_ids = tokenizer.encode_batch(ingredients_lists)
        
        # 어텐션 마스크 생성
        attention_mask = tokenizer.get_attention_mask(input_ids)
        
        # 레이블 텐서 생성
        labels = torch.stack(labels)
        
        return input_ids, attention_mask, labels

    # 데이터 로더
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    # 예측 및 정답 레이블 수집
    model.model.eval()
    all_preds_prob = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # 예측
            outputs = model.model(input_ids, attention_mask)
            
            # 임계값 적용
            preds = (outputs >= threshold).float()
            
            all_preds_prob.append(outputs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # 배열 결합
    all_preds_prob = np.concatenate(all_preds_prob, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 성능 지표 계산 (매크로 평균)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # 효과별 성능 지표
    class_metrics = {}
    for i, effect in enumerate(effect_names):
        p, r, f, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        class_metrics[effect] = {
            'precision': p,
            'recall': r,
            'f1': f
        }
    
    # 종합 성능 지표
    metrics = {
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'class_metrics': class_metrics
    }
    
    # 출력 디렉토리가 제공된 경우 결과 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 성능 지표 저장
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 2. 클래스별 성능 시각화
        plot_class_performance(effect_names, class_metrics, output_dir)
        
        # 3. ROC 곡선 및 PR 곡선 시각화
        plot_roc_curves(all_labels, all_preds_prob, effect_names, output_dir)
        plot_pr_curves(all_labels, all_preds_prob, effect_names, output_dir)
        
        # 4. 혼동 행렬 시각화
        confusion_matrix(all_labels, all_preds, effect_names, output_dir)
        
        # 5. 상세 분류 보고서 저장
        classification_report(all_labels, all_preds, effect_names, output_dir)
    
    return metrics


def plot_class_performance(effect_names, class_metrics, output_dir):
    """
    클래스별 성능 지표 시각화
    """
    # 데이터 준비
    effects = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for effect, metrics in class_metrics.items():
        effects.append(effect)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Effect': effects,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })
    
    # 성능 지표별 플롯
    plt.figure(figsize=(12, 8))
    plot = sns.barplot(x='Effect', y='value', hue='metric', 
                     data=pd.melt(df, id_vars=['Effect'], value_vars=['Precision', 'Recall', 'F1 Score'], 
                                 var_name='metric', value_name='value'))
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Performance Metrics by Effect Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'))
    plt.close()


def plot_roc_curves(all_labels, all_preds_prob, effect_names, output_dir):
    """
    ROC 곡선 시각화
    """
    plt.figure(figsize=(12, 10))
    
    # 각 효과에 대한 ROC 곡선
    for i, effect in enumerate(effect_names):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{effect} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()


def plot_pr_curves(all_labels, all_preds_prob, effect_names, output_dir):
    """
    정밀도-재현율 곡선 시각화
    """
    plt.figure(figsize=(12, 10))
    
    # 각 효과에 대한 PR 곡선
    for i, effect in enumerate(effect_names):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{effect} (AUC = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])