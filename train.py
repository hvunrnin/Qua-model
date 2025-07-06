import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from models.tokenizer import IngredientTokenizer
from models.model import CosmeticEffectPredictor, CosmeticEffectPredictorWithTokenizer
from models.loss import WeightedBinaryCrossEntropyLoss, calculate_class_weights


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


def collate_fn(batch, tokenizer, max_length=None):
    """
    배치 데이터 처리 함수
    Args:
        batch: (ingredients_list, label) 튜플의 리스트
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
    Returns:
        입력 텐서, 어텐션 마스크, 레이블 텐서
    """
    ingredients_lists, labels = zip(*batch)
    
    # 토큰화
    input_ids = tokenizer.encode_batch(ingredients_lists, max_length=max_length)
    
    # 어텐션 마스크 생성
    attention_mask = tokenizer.get_attention_mask(input_ids)
    
    # 레이블 텐서 생성
    labels = torch.stack(labels)
    
    return input_ids, attention_mask, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 에폭 동안 모델 학습
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        device: 학습 장치 ('cpu' 또는 'cuda')
    Returns:
        평균 손실
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        # 데이터를 지정된 장치로 이동
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # 순전파
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 진행 상황 출력
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """
    모델 평가
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 평가 장치
        threshold: 분류 임계값
    Returns:
        평균 손실, 정밀도, 재현율, F1 점수
    """
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # 순전파
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 임계값 적용하여 예측 결과 변환 (0.5 이상이면 1, 미만이면 0)
            preds = (outputs >= threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 예측값과 타겟값 결합
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 평가 지표 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return total_loss / len(dataloader), precision, recall, f1


def make_collate_fn(tokenizer, max_length):
    def collate_fn_inner(batch):
        return collate_fn(batch, tokenizer, max_length)
    return collate_fn_inner

def main(args):
    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("Loading data...")
    # 여기에 데이터 로드 코드를 추가해야 합니다.
    # 예시: CSV 파일에서 성분 리스트와 효과 레이블 로드
    df = pd.read_csv(args.data_path)
    
    # 각 행에서 성분 리스트와 효과 레이블 추출
    # 이 부분은 데이터 형식에 따라 조정 필요
    ingredients_lists = df['ingredients'].apply(lambda x: re.split(r'(?<=[a-zA-Z0-9.]),\s+', x)).tolist()
    
    # 효과 레이블 컬럼들
    effect_columns = [col for col in df.columns if col.startswith('effect_')]
    labels = df[effect_columns].values
    
    # 훈련/검증/테스트 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        ingredients_lists, labels, test_size=0.3, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    
    # 레이블을 텐서로 변환
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)
    
    # 토크나이저 학습
    print("Training tokenizer...")
    tokenizer = IngredientTokenizer(vocab_size=args.vocab_size)
    tokenizer.fit(X_train)
    
    # 데이터셋 생성
    train_dataset = CosmeticDataset(X_train, y_train)
    val_dataset = CosmeticDataset(X_val, y_val)
    test_dataset = CosmeticDataset(X_test, y_test)
    
    # 커스텀 콜레이트 함수
    train_collate = make_collate_fn(tokenizer, args.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=train_collate, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=train_collate, num_workers=args.num_workers
    )
    
    # 모델 생성
    print("Creating model...")
    model = CosmeticEffectPredictor(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=args.embedding_dim,
        num_classes=len(effect_columns),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_seq_length=args.max_seq_length,
        pooling=args.pooling,
        dropout=args.dropout,
        fc_hidden_dims=args.fc_hidden_dims
    ).to(device)
    
    # 클래스 가중치 계산
    if args.weighted_loss:
        print("Calculating class weights...")
        pos_weight = calculate_class_weights(y_train, beta=args.class_weight_beta)
        pos_weight = pos_weight.to(device)
        print(f"Class weights: {pos_weight}")
    else:
        pos_weight = None
    
    # 손실 함수 및 옵티마이저
    criterion = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 학습률 스케줄러
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 설정 저장
    config = {
        'vocab_size': tokenizer.get_vocab_size(),
        'embedding_dim': args.embedding_dim,
        'num_classes': len(effect_columns),
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'max_seq_length': args.max_seq_length,
        'pooling': args.pooling,
        'dropout': args.dropout,
        'fc_hidden_dims': args.fc_hidden_dims,
        'effect_names': effect_columns
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 학습 시작
    print("Starting training...")
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")
        
        # 검증
        val_loss, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device, args.threshold
        )
        print(f"Validation - Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # 학습률 스케줄러 업데이트
        if args.use_scheduler:
            scheduler.step(val_loss)
        
        # 최고 성능 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            # 모델 래퍼 생성 및 저장
            model_with_tokenizer = CosmeticEffectPredictorWithTokenizer(model, tokenizer, device)
            model_with_tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model.pt'))
            
            print(f"New best model saved (F1: {val_f1:.4f})")
    
    print(f"Training completed. Best model at epoch {best_epoch} with F1: {best_val_f1:.4f}")
    
    # 최고 성능 모델 로드
    print("Evaluating best model on test set...")
    best_model_with_tokenizer = CosmeticEffectPredictorWithTokenizer.from_pretrained(
        os.path.join(args.output_dir, 'best_model.pt'), device=device
    )
    best_model = best_model_with_tokenizer.model
    
    # 테스트 세트에서 평가
    test_loss, test_precision, test_recall, test_f1 = evaluate(
        best_model, test_loader, criterion, device, args.threshold
    )
    print(f"Test - Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # 결과 저장
    results = {
        'best_epoch': best_epoch,
        'val_f1': best_val_f1,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cosmetic effect predictor model")
    
    # 데이터 인자
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model and results")
    parser.add_argument("--max_seq_length", type=int, default=100,
                        help="Maximum sequence length for tokenizer")
    
    # 모델 인자
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Maximum vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Dimension of embeddings")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=1024,
                        help="Dimension of feed-forward layer")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy for encoder output")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--fc_hidden_dims", type=int, nargs="+", default=[512, 256, 128],
                        help="Hidden dimensions of classifier layers")
    
    # 훈련 인자
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use class-weighted loss")
    parser.add_argument("--class_weight_beta", type=float, default=0.999,
                        help="Beta factor for class weight calculation")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use learning rate scheduler")
    
    # 기타 인자
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    main(args)