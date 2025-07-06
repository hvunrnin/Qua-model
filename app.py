import os
import json
from flask import Flask, request, jsonify
import torch
import re

from models.model import CosmeticEffectPredictorWithTokenizer

app = Flask(__name__)

# 모델 로드
MODEL_PATH = 'output/best_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 글로벌 변수로 모델 선언
model = None
effect_names = []

def load_model():
    """모델 로드 함수"""
    global model, effect_names
    
    # 모델 로드
    model = CosmeticEffectPredictorWithTokenizer.from_pretrained(MODEL_PATH, device=device)
    
    # 설정 파일에서 효과 이름 로드
    config_path = os.path.join(os.path.dirname(MODEL_PATH), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        effect_names = config.get('effect_names', [])
    except FileNotFoundError:
        effect_names = [f'effect_{i}' for i in range(model.model.classifier.classifier.classifier[-1].out_features)]
    
    return model, effect_names

# 앱 초기화 시 모델 로드
with app.app_context():
    model, effect_names = load_model()
    print(f"Model loaded successfully. Effects: {effect_names}")

@app.route('/predict', methods=['POST'])
def predict():
    """성분 리스트로부터 효과 예측"""
    if request.method == 'POST':
        
        data = request.get_json()
        ingredients = data.get('ingredients', '')
        
        # 성분 리스트 전처리
        if isinstance(ingredients, str):
            ingredients_list = re.split(r'(?<=[a-zA-Z0-9.]),\s+', ingredients)
        else:
            ingredients_list = ingredients
        
        # 예측
        prediction_probs = model.predict_proba(ingredients_list)
        
        # 결과 정리
        results = {}
        for i, effect in enumerate(effect_names):
            results[effect] = {
                'probability': float(prediction_probs[i]),
            }
        
        # 결과 반환 (JSON API 또는 웹 페이지)
     
        return jsonify({
            'predictions': results
        })

if __name__ == '__main__':
    # 개발 환경에서는 debug=True, 배포 환경에서는 debug=False
    app.run(host='0.0.0.0', port=5000, debug=True)