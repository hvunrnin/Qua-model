import torch
import torch.nn as nn

from .tokenizer import IngredientTokenizer
from .encoder import TransformerEncoder
from .classifier import PooledMultiLabelClassifier

class CosmeticEffectPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, 
                 num_layers=2, num_heads=8, ff_dim=2048, 
                 max_seq_length=100, pooling='mean', dropout=0.1,
                 fc_hidden_dims=[512, 256, 128]):
        """
        화장품 성분 기반 효과 예측 모델
        Args:
            vocab_size: 어휘 크기 (성분 개수)
            embedding_dim: 임베딩 차원
            num_classes: 분류할 효과 클래스 개수
            num_layers: 인코더 레이어 개수
            num_heads: 어텐션 헤드 개수
            ff_dim: 피드포워드 네트워크 내부 차원
            max_seq_length: 최대 시퀀스 길이
            pooling: 인코더 출력 풀링 방식 ('mean', 'cls', 'max')
            dropout: 드롭아웃 비율
            fc_hidden_dims: 분류기 완전 연결 층의 은닉층 차원 리스트
        """
        super(CosmeticEffectPredictor, self).__init__()
        
        # 트랜스포머 인코더
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # 풀링 및 멀티레이블 분류기
        self.classifier = PooledMultiLabelClassifier(
            encoder=self.encoder,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            pooling=pooling,
            fc_hidden_dims=fc_hidden_dims,
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        """
        모델 순전파
        Args:
            x: 입력 텐서 (batch_size, seq_length)
            mask: 어텐션 마스크 (optional)
        Returns:
            각 효과 클래스에 대한 확률 (batch_size, num_classes)
        """
        return self.classifier(x, mask)
    
    
class CosmeticEffectPredictorWithTokenizer:
    def __init__(self, model, tokenizer, device='cpu'):
        """
        토크나이저가 포함된 화장품 효과 예측 모델 래퍼 클래스
        Args:
            model: CosmeticEffectPredictor 모델
            tokenizer: IngredientTokenizer 토크나이저
            device: 모델이 로드될 장치 ('cpu' 또는 'cuda')
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def predict(self, ingredients_list, threshold=0.5, max_length=None):
        """
        화장품 성분 리스트로부터 효과 예측
        Args:
            ingredients_list: 성분 리스트 또는 성분 리스트의 리스트
            threshold: 분류 임계값
            max_length: 최대 시퀀스 길이
        Returns:
            예측된 효과 리스트 또는 효과 리스트의 리스트
        """
        # 입력이 단일 성분 리스트인지 여러 성분 리스트의 리스트인지 확인
        is_single_sample = not isinstance(ingredients_list[0], list)
        
        # 단일 샘플을 배치 형태로 변환
        if is_single_sample:
            ingredients_list = [ingredients_list]
        
        # 토큰화
        inputs = self.tokenizer.encode_batch(ingredients_list, max_length=max_length)
        inputs = inputs.to(self.device)
        
        # 어텐션 마스크 생성
        attention_mask = self.tokenizer.get_attention_mask(inputs)
        attention_mask = attention_mask.to(self.device)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, attention_mask)
        
        # 임계값 적용
        predictions = (outputs >= threshold).float().cpu().numpy()
        
        # 단일 샘플인 경우 첫 번째 예측 결과만 반환
        if is_single_sample:
            return predictions[0]
        
        return predictions
    
    def predict_proba(self, ingredients_list, max_length=None):
        """
        화장품 성분 리스트로부터 효과 확률 예측
        Args:
            ingredients_list: 성분 리스트 또는 성분 리스트의 리스트
            max_length: 최대 시퀀스 길이
        Returns:
            예측된 효과 확률 리스트 또는 확률 리스트의 리스트
        """
        # 입력이 단일 성분 리스트인지 여러 성분 리스트의 리스트인지 확인
        is_single_sample = not isinstance(ingredients_list[0], list)
        
        # 단일 샘플을 배치 형태로 변환
        if is_single_sample:
            ingredients_list = [ingredients_list]
        
        # 토큰화
        inputs = self.tokenizer.encode_batch(ingredients_list, max_length=max_length)
        inputs = inputs.to(self.device)
        
        # 어텐션 마스크 생성
        attention_mask = self.tokenizer.get_attention_mask(inputs)
        attention_mask = attention_mask.to(self.device)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, attention_mask)
        
        probas = outputs.cpu().numpy()
        
        # 단일 샘플인 경우 첫 번째 예측 결과만 반환
        if is_single_sample:
            return probas[0]
        
        return probas
    
    @classmethod
    def from_pretrained(cls, model_path, tokenizer_path=None, device='cpu'):
        """
        사전 학습된 모델 및 토크나이저 로드
        Args:
            model_path: 모델 파일 경로
            tokenizer_path: 토크나이저 파일 경로 (없는 경우 모델 파일에서 함께 로드)
            device: 모델이 로드될 장치 ('cpu' 또는 'cuda')
        Returns:
            CosmeticEffectPredictorWithTokenizer 인스턴스
        """
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 초기화
        model = CosmeticEffectPredictor(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            num_classes=checkpoint['num_classes'],
            num_layers=checkpoint['num_layers'],
            num_heads=checkpoint['num_heads'],
            ff_dim=checkpoint['ff_dim'],
            max_seq_length=checkpoint['max_seq_length'],
            pooling=checkpoint['pooling'],
            dropout=checkpoint['dropout'],
            fc_hidden_dims=checkpoint['fc_hidden_dims']
        )
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 토크나이저 로드
        if tokenizer_path is not None:
            tokenizer = torch.load(tokenizer_path)
        else:
            tokenizer = checkpoint.get('tokenizer')
            
            # 체크포인트에 토크나이저가 없는 경우
            if tokenizer is None:
                raise ValueError("토크나이저가 제공되지 않았습니다. 별도의 토크나이저 파일을 제공하세요.")
        
        return cls(model, tokenizer, device)
    
    def save_pretrained(self, model_path, save_tokenizer=True):
        """
        모델 및 토크나이저 저장
        Args:
            model_path: 저장할 파일 경로
            save_tokenizer: 토크나이저도 함께 저장할지 여부
        """
        # 모델 설정 및 가중치 저장
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.encoder.embedding.embedding.embedding.num_embeddings,
            'embedding_dim': self.model.encoder.embedding.embedding.embedding_dim,
            'num_classes': self.model.classifier.classifier.classifier[-1].out_features,
            'num_layers': len(self.model.encoder.encoder_layers),
            'num_heads': self.model.encoder.encoder_layers[0].self_attention.num_heads,
            'ff_dim': self.model.encoder.encoder_layers[0].feed_forward.linear1.out_features,
            'max_seq_length': self.model.encoder.embedding.pos_encoding.pe.size(1),
            'pooling': self.model.classifier.pooling,
            'dropout': self.model.encoder.dropout.p,
            'fc_hidden_dims': [
                layer.out_features for layer in self.model.classifier.classifier.classifier 
                if isinstance(layer, nn.Linear)
            ][:-1]  # 마지막 출력층 제외
        }
        
        # 토크나이저 함께 저장
        if save_tokenizer:
            checkpoint['tokenizer'] = self.tokenizer
        
        # 파일로 저장
        torch.save(checkpoint, model_path)