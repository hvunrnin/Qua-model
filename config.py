"""
모델 및 훈련에 관한 설정 파일
"""

# 데이터 경로
DATA_PATH = "data/cosmetics_data.csv"
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"
TEST_DATA_PATH = "data/test.csv"

# 모델 하이퍼파라미터
VOCAB_SIZE = 10000  # 어휘 크기
EMBEDDING_DIM = 256  # 임베딩 차원
NUM_LAYERS = 2  # 인코더 레이어 수
NUM_HEADS = 8  # 어텐션 헤드 수
FF_DIM = 1024  # 피드포워드 레이어 차원
MAX_SEQ_LENGTH = 100  # 최대 시퀀스 길이
POOLING = "mean"  # 풀링 방식 ('mean', 'cls', 'max')
DROPOUT = 0.1  # 드롭아웃 비율
FC_HIDDEN_DIMS = [512, 256, 128]  # 분류기 은닉층 차원

# 훈련 하이퍼파라미터
BATCH_SIZE = 32  # 배치 크기
LEARNING_RATE = 1e-4  # 학습률
WEIGHT_DECAY = 1e-5  # 가중치 감쇠
NUM_EPOCHS = 50  # 에폭 수
PATIENCE = 5  # 조기 종료 인내심
THRESHOLD = 0.5  # 분류 임계값
WEIGHTED_LOSS = True  # 가중치 손실 사용 여부
CLASS_WEIGHT_BETA = 0.999  # 클래스 가중치 계산을 위한 베타 값

# 기타 설정
SEED = 42  # 랜덤 시드
NUM_WORKERS = 4  # 데이터 로더 워커 수
SAVE_DIR = "saved_models"  # 모델 저장 디렉토리
LOGS_DIR = "logs"  # 로그 디렉토리

# 효과 레이블 목록 (예시)
EFFECT_LABELS = [
    "hydrating",        # 보습 효과
    "anti_aging",       # 노화 방지
    "brightening",      # 미백/브라이트닝
    "soothing",         # 진정 효과
    "pore_minimizing",  # 모공 축소
    "anti_acne",        # 여드름 개선
    "exfoliating",      # 각질 제거
    "oil_control",      # 유분 조절
    "anti_wrinkle",     # 주름 개선
    "uv_protection",    # 자외선 차단
    "anti_oxidant",     # 항산화
    "anti_inflammatory", # 항염
    "firming",          # 탄력 개선
    "pigmentation",     # 색소 침착 개선
    "barrier_repair"    # 피부 장벽 강화
]