import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        가중치가 적용된 바이너리 크로스 엔트로피 손실 함수
        Args:
            pos_weight: 각 클래스의 양성 샘플에 대한 가중치 (클래스별 가중치 텐서)
            reduction: 손실 감소 방식 ('mean', 'sum', 'none')
        """
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        손실 계산
        Args:
            logits: 모델 출력 (batch_size, num_classes)
            targets: 타겟 레이블 (batch_size, num_classes), 이진값 (0 또는 1)
        Returns:
            손실 값
        """
        # 이미 시그모이드가 적용된 출력이므로 BCELoss 사용
        if self.pos_weight is not None:
            # 클래스별 가중치 적용
            weight = self.pos_weight.expand_as(targets)
            
            # 양성(1) 및 음성(0) 샘플에 대한 가중치 적용
            weight_pos = weight * targets
            weight_neg = (1 - targets)
            weights = weight_pos + weight_neg
            
            # 가중치가 적용된 이진 크로스 엔트로피 손실
            loss = F.binary_cross_entropy(logits, targets, weight=weights, reduction='none')
        else:
            loss = F.binary_cross_entropy(logits, targets, reduction='none')
        
        # 감소 방식에 따라 손실 처리
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def calculate_class_weights(labels, beta=0.999):
    """
    클래스 불균형을 다루기 위한 가중치 계산 
    (Cui et al., 2019) "Class-balanced loss based on effective number of samples"
    
    Args:
        labels: 훈련 데이터의 레이블 텐서 (num_samples, num_classes)
        beta: 클래스 불균형 조정을 위한 스무딩 팩터
    Returns:
        클래스별 가중치 텐서
    """
    # 각 클래스의 샘플 수 계산
    num_samples_per_class = torch.sum(labels, dim=0)  # (num_classes)
    
    # 0으로 나누기 방지
    num_samples_per_class = torch.clamp(num_samples_per_class, min=1)
    
    # 효과적인 샘플 수 계산
    effective_num = 1.0 - torch.pow(beta, num_samples_per_class)
    effective_num = torch.clamp(effective_num, min=1e-8)
    
    # 클래스 가중치 계산
    weights = (1.0 - beta) / effective_num
    
    # 가중치 정규화 (합이 클래스 수가 되도록)
    weights = weights / torch.sum(weights) * len(num_samples_per_class)
    
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        멀티 레이블 분류를 위한 포컬 손실 함수
        Args:
            alpha: 양성 샘플 가중치
            gamma: 포커싱 파라미터
            reduction: 손실 감소 방식 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        포컬 손실 계산
        Args:
            logits: 모델 출력 확률 (batch_size, num_classes)
            targets: 타겟 레이블 (batch_size, num_classes), 이진값 (0 또는 1)
        Returns:
            손실 값
        """
        # 확률 및 반대 확률
        p = logits
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 알파 가중치
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 포컬 가중치
        focal_weight = (1 - p_t) ** self.gamma
        
        # 포컬 손실 계산
        loss = -alpha_t * focal_weight * torch.log(torch.clamp(p_t, min=1e-8))
        
        # 감소 방식에 따라 손실 처리
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss