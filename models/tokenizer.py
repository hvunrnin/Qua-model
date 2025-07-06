
import torch
import numpy as np
from collections import Counter

class IngredientTokenizer:
    """
    화장품 성분을 토큰화하는 클래스
    """
    def __init__(self, vocab_size=None, min_freq=2, special_tokens=None):
        """
        초기화
        Args:
            vocab_size: 어휘 크기 (최대 단어 수)
            min_freq: 최소 등장 빈도
            special_tokens: 특수 토큰 (예: [PAD], [UNK], [CLS], [SEP])
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # 특수 토큰 설정
        self.special_tokens = special_tokens or {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]', 
            'cls_token': '[CLS]',
            'sep_token': '[SEP]'
        }
        
        # 어휘 사전 및 반대 매핑
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 특수 토큰 추가
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """특수 토큰을 어휘 사전에 추가"""
        idx = 0
        for token in self.special_tokens.values():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            idx += 1
    
    def fit(self, ingredients_lists):
        """
        성분 데이터로 토크나이저 학습
        Args:
            ingredients_lists: 성분 리스트의 리스트
        """
        # 각 성분의 빈도 계산
        counter = Counter()
        for ingredients in ingredients_lists:
            counter.update(ingredients)
        
        # 최소 빈도 이상인 성분만 선택
        valid_ingredients = [
            ingredient for ingredient, count in counter.items() 
            if count >= self.min_freq
        ]
        
        # vocab_size 제한 적용 (특수 토큰 제외)
        start_idx = len(self.special_tokens)
        if self.vocab_size is not None:
            valid_ingredients = valid_ingredients[:self.vocab_size - start_idx]
        
        # 어휘 사전 구축
        for idx, ingredient in enumerate(valid_ingredients, start=start_idx):
            self.vocab[ingredient] = idx
            self.inverse_vocab[idx] = ingredient
    
    def encode(self, ingredients_list, max_length=None, padding=True, truncation=True):
        """
        성분 리스트를 인덱스 리스트로 변환
        Args:
            ingredients_list: 성분 리스트
            max_length: 최대 시퀀스 길이
            padding: 패딩 적용 여부
            truncation: 길이 초과 시 자르기 여부
        Returns:
            인덱스 리스트
        """
        # CLS 토큰으로 시작
        encoded = [self.vocab[self.special_tokens['cls_token']]]
        
        # 성분을 인덱스로 변환
        for ingredient in ingredients_list:
            if ingredient in self.vocab:
                encoded.append(self.vocab[ingredient])
            else:
                # OOV 단어는 UNK 토큰으로 대체
                encoded.append(self.vocab[self.special_tokens['unk_token']])
        
        # SEP 토큰으로 종료
        encoded.append(self.vocab[self.special_tokens['sep_token']])
        
        # 길이 제한
        if max_length is not None and truncation and len(encoded) > max_length:
            encoded = encoded[:max_length]
        
        # 패딩
        if max_length is not None and padding and len(encoded) < max_length:
            encoded.extend([self.vocab[self.special_tokens['pad_token']]] * (max_length - len(encoded)))
        
        return encoded
    
    def encode_batch(self, batch_ingredients, max_length=None, padding=True, truncation=True):
        """
        배치 데이터 인코딩
        Args:
            batch_ingredients: 성분 리스트의 리스트
            max_length: 최대 시퀀스 길이
            padding: 패딩 적용 여부
            truncation: 길이 초과 시 자르기 여부
        Returns:
            인코딩된 텐서 (batch_size, seq_length)
        """
        # 각 성분 리스트를 인코딩
        encoded_lists = [
            self.encode(ingredients, max_length, padding, truncation) 
            for ingredients in batch_ingredients
        ]
        
        # 배치의 최대 길이 계산
        if max_length is None and padding:
            max_len = max(len(encoded) for encoded in encoded_lists)
            # 패딩 적용
            for i, encoded in enumerate(encoded_lists):
                encoded_lists[i].extend(
                    [self.vocab[self.special_tokens['pad_token']]] * (max_len - len(encoded))
                )
        
        # 텐서로 변환
        return torch.tensor(encoded_lists, dtype=torch.long)
    
    def decode(self, indices):
        """
        인덱스 리스트를 성분 리스트로 변환
        Args:
            indices: 인덱스 리스트
        Returns:
            성분 리스트
        """
        return [
            self.inverse_vocab.get(idx, self.special_tokens['unk_token']) 
            for idx in indices
        ]
    
    def decode_batch(self, batch_indices):
        """
        배치 인덱스를 성분 리스트로 변환
        Args:
            batch_indices: 인덱스 배치 (batch_size, seq_length)
        Returns:
            성분 리스트의 리스트
        """
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.cpu().numpy()
            
        return [self.decode(indices) for indices in batch_indices]
    
    def get_vocab_size(self):
        """어휘 사전 크기 반환"""
        return len(self.vocab)
    
    def get_pad_token_id(self):
        """PAD 토큰 ID 반환"""
        return self.vocab[self.special_tokens['pad_token']]
    
    def get_attention_mask(self, input_ids):
        """
        어텐션 마스크 생성
        Args:
            input_ids: 입력 토큰 ID (batch_size, seq_length)
        Returns:
            어텐션 마스크 (batch_size, seq_length)
        """
        pad_token_id = self.get_pad_token_id()
        return (input_ids != pad_token_id).long()