import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        """
        멀티헤드 어텐션 모듈
        Args:
            embedding_dim: 임베딩 차원
            num_heads: 어텐션 헤드 개수
            dropout: 드롭아웃 비율
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0, "임베딩 차원은 헤드 개수로 나누어 떨어져야 합니다."
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # 쿼리, 키, 밸류 가중치 행렬
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # 출력 가중치 행렬
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        멀티헤드 어텐션 순전파
        Args:
            query: 쿼리 텐서 (batch_size, seq_length, embedding_dim)
            key: 키 텐서 (batch_size, seq_length, embedding_dim)
            value: 밸류 텐서 (batch_size, seq_length, embedding_dim)
            mask: 어텐션 마스크 (optional)
        Returns:
            output: 어텐션 결과 (batch_size, seq_length, embedding_dim)
            attention_weights: 어텐션 가중치
        """
        batch_size = query.size(0)
        
        # 선형 투영 및 헤드로 분할
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len_q, head_dim)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len_k, head_dim)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len_v, head_dim)
        
        # 스케일드 닷-프로덕트 어텐션
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len_q, seq_len_k)
        
        # 마스크 적용 (필요한 경우)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 어텐션 가중치 계산
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 가중치와 밸류의 곱
        attention_output = torch.matmul(attention_weights, v)  # (batch, num_heads, seq_len_q, head_dim)
        
        # 헤드 결합 및 출력 투영
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)  # (batch, seq_len_q, embedding_dim)
        output = self.out_linear(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        """
        피드포워드 네트워크 모듈
        Args:
            embedding_dim: 임베딩 차원
            ff_dim: 피드포워드 네트워크 내부 차원
            dropout: 드롭아웃 비율
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        피드포워드 네트워크 순전파
        Args:
            x: 입력 텐서 (batch_size, seq_length, embedding_dim)
        Returns:
            출력 텐서 (batch_size, seq_length, embedding_dim)
        """
        # 첫 번째 선형 변환 및 활성화 함수 (ReLU)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # 두 번째 선형 변환
        x = self.linear2(x)
        
        return x