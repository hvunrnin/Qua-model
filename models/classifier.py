import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, fc_hidden_dims=[512, 256, 128], dropout=0.1):
        """
        멀티레이블 분류를 위한 클래스
        Args:
            embedding_dim: 인코더의 출력 임베딩 차원
            num_classes: 분류할 클래스 개수
            fc_hidden_dims: 완전 연결 층의 은닉층 차원 리스트
            dropout: 드롭아웃 비율
        """
        super(MultiLabelClassifier, self).__init__()
        
        # 분류 헤드 (완전 연결 층)
        layers = []
        
        # 첫 번째 완전 연결 층
        layers.append(nn.Linear(embedding_dim, fc_hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 중간 완전 연결 층
        for i in range(len(fc_hidden_dims) - 1):
            layers.append(nn.Linear(fc_hidden_dims[i], fc_hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 출력 층
        layers.append(nn.Linear(fc_hidden_dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        분류기 순전파
        Args:
            x: 인코더 출력 표현 (batch_size, embedding_dim)
        Returns:
            각 클래스에 대한 로짓 (batch_size, num_classes)
        """
        return self.classifier(x)


class PooledMultiLabelClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes, pooling='mean',
                 fc_hidden_dims=[512, 256, 128], dropout=0.1):
        """
        인코더 출력을 풀링하고 멀티레이블 분류를 수행하는 클래스
        Args:
            encoder: 트랜스포머 인코더 모델
            embedding_dim: 인코더의 출력 임베딩 차원
            num_classes: 분류할 클래스 개수
            pooling: 풀링 방식 ('mean', 'cls', 'max')
            fc_hidden_dims: 완전 연결 층의 은닉층 차원 리스트
            dropout: 드롭아웃 비율
        """
        super(PooledMultiLabelClassifier, self).__init__()
        
        self.encoder = encoder
        self.pooling = pooling
        
        # 분류기
        self.classifier = MultiLabelClassifier(
            embedding_dim, num_classes, fc_hidden_dims, dropout
        )
        
        # 시그모이드 출력층 (멀티레이블 분류를 위한 별도 층)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask=None):
        """
        모델 순전파
        Args:
            x: 입력 텐서 (batch_size, seq_length)
            mask: 어텐션 마스크 (optional)
        Returns:
            각 클래스에 대한 확률 (batch_size, num_classes)
        """
        # 인코더를 통과한 출력
        encoder_output = self.encoder(x, mask)  # (batch_size, seq_length, embedding_dim)
        
        # 인코더 출력 풀링
        if self.pooling == 'cls':
            # [CLS] 토큰 표현 사용 (첫 번째 토큰)
            pooled_output = encoder_output[:, 0, :]  # (batch_size, embedding_dim)
        elif self.pooling == 'max':
            # 최대 풀링
            pooled_output = torch.max(encoder_output, dim=1)[0]  # (batch_size, embedding_dim)
        else:  # 'mean' (기본값)
            # 평균 풀링
            pooled_output = torch.mean(encoder_output, dim=1)  # (batch_size, embedding_dim)
        
        # 분류기를 통과한 로짓
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        # 시그모이드를 통과한 확률
        probs = self.sigmoid(logits)
        
        return probs