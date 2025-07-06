import torch
import torch.nn as nn
import numpy as np
import math

class IngredientEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        화장품 성분 임베딩 클래스
        Args:
            vocab_size: 어휘 크기 (성분 개수)
            embedding_dim: 임베딩 차원
        """
        super(IngredientEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        """
        입력 텐서를 임베딩 벡터로 변환
        Args:
            x: 입력 텐서 (batch_size, sequence_length)
        Returns:
            임베딩 벡터 (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        """
        포지셔널 인코딩 모듈
        Args:
            embedding_dim: 임베딩 차원
            max_seq_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 포지셔널 인코딩 행렬 계산
        self.pos_encoding = self._create_positional_encoding(max_seq_length, embedding_dim)
        
    def _create_positional_encoding(self, seq_length, embedding_dim):
        """
        포지셔널 인코딩 생성
        Args:
            seq_length: 시퀀스 길이
            embedding_dim: 임베딩 차원
        Returns:
            포지셔널 인코딩 텐서 (1, seq_length, embedding_dim)
        """
        # 위치 인코딩 행렬 초기화
        pos_encoding = torch.zeros(seq_length, embedding_dim)
        
        # 위치 인덱스와 차원 인덱스
        positions = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        
        # 사인 함수와 코사인 함수 적용
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        
        # 배치 차원 추가 (1, seq_length, embedding_dim)
        pos_encoding = pos_encoding.unsqueeze(0)
        
        # 모델 파라미터로 등록하되 업데이트는 하지 않음
        self.register_buffer('pe', pos_encoding)
        
        return pos_encoding
        
    def forward(self, x):
        """
        임베딩에 포지셔널 인코딩 적용
        Args:
            x: 입력 임베딩 텐서 (batch_size, seq_length, embedding_dim)
        Returns:
            포지셔널 인코딩이 적용된 텐서 (batch_size, seq_length, embedding_dim)
        """
        # 시퀀스 길이에 맞게 포지셔널 인코딩 자르기
        seq_length = x.size(1)
        
        # 임베딩에 포지셔널 인코딩 더하기
        x = x + self.pos_encoding[:, :seq_length, :]
        
        return self.dropout(x)


class IngredientEmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=512, dropout=0.1):
        """
        화장품 성분 임베딩과 포지셔널 인코딩을 결합한 클래스
        Args:
            vocab_size: 어휘 크기 (성분 개수)
            embedding_dim: 임베딩 차원
            max_seq_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super(IngredientEmbeddingWithPositionalEncoding, self).__init__()
        
        # 임베딩 레이어
        self.embedding = IngredientEmbedding(vocab_size, embedding_dim)
        
        # 포지셔널 인코딩 레이어
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
    def forward(self, x):
        """
        입력 텐서에 대해 임베딩과 포지셔널 인코딩을 적용
        Args:
            x: 입력 텐서 (batch_size, sequence_length)
        Returns:
            임베딩 + 포지셔널 인코딩 벡터 (batch_size, sequence_length, embedding_dim)
        """
        # 임베딩 계산
        embedded = self.embedding(x)
        
        # 포지셔널 인코딩 적용
        encoded = self.pos_encoding(embedded)
        
        return encoded