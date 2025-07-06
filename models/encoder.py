import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, FeedForward
from .embeddings import IngredientEmbeddingWithPositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        """
        트랜스포머 인코더 레이어
        Args:
            embedding_dim: 임베딩 차원
            num_heads: 어텐션 헤드 개수
            ff_dim: 피드포워드 네트워크 내부 차원
            dropout: 드롭아웃 비율
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)
        
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        인코더 레이어 순전파
        Args:
            x: 입력 텐서 (batch_size, seq_length, embedding_dim)
            mask: 어텐션 마스크 (optional)
        Returns:
            출력 텐서 (batch_size, seq_length, embedding_dim)
        """
        # 멀티헤드 셀프 어텐션 (서브레이어 1)
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 잔차 연결 및 레이어 정규화
        
        # 피드포워드 네트워크 (서브레이어 2)
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output)
        out2 = self.layernorm2(out1 + ff_output)  # 잔차 연결 및 레이어 정규화
        
        return out2


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2, num_heads=8, ff_dim=2048, max_seq_length=100, dropout=0.1):
        """
        트랜스포머 인코더 모델
        Args:
            vocab_size: 어휘 크기 (성분 개수)
            embedding_dim: 임베딩 차원
            num_layers: 인코더 레이어 개수
            num_heads: 어텐션 헤드 개수
            ff_dim: 피드포워드 네트워크 내부 차원
            max_seq_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super(TransformerEncoder, self).__init__()
        
        # 임베딩 레이어 (및 포지셔널 인코딩 포함)
        self.embedding = IngredientEmbeddingWithPositionalEncoding(
            vocab_size, embedding_dim, max_seq_length, dropout
        )
        
        # 인코더 레이어 스택
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        """
        트랜스포머 인코더 순전파
        Args:
            x: 입력 텐서 (batch_size, seq_length)
            mask: 어텐션 마스크 (optional)
        Returns:
            출력 텐서 (batch_size, seq_length, embedding_dim)
        """
        # 임베딩 및 포지셔널 인코딩 적용
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # 인코더 레이어 통과
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        return x