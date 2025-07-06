import argparse
import json
import pandas as pd
import torch
import numpy as np
from models.model import CosmeticEffectPredictorWithTokenizer

def load_model(model_path, device):
    """
    저장된 모델 로드
    Args:
        model_path: 모델 파일 경로
        device: 모델이 로드될 장치
    Returns:
        모델, 효과 이름 리스트
    """
    # 모델 로드
    model = CosmeticEffectPredictorWithTokenizer.from_pretrained(model_path, device=device)
    
    # 설정 파일 로드
    config_path = '/'.join(model_path.split('/')[:-1] + ['config.json'])
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        effect_names = config.get('effect_names', [f'effect_{i}' for i in range(model.model.classifier.classifier.classifier[-1].out_features)])
    except FileNotFoundError:
        # 설정 파일이 없는 경우 기본 효과 이름 사용
        effect_names = [f'effect_{i}' for i in range(model.model.classifier.classifier.classifier[-1].out_features)]
    
    return model, effect_names

def predict_from_ingredients(model, ingredients_list, effect_names, threshold=0.5, return_proba=False):
    """
    성분 리스트에서 효과 예측
    Args:
        model: 예측 모델
        ingredients_list: 성분 리스트
        effect_names: 효과 이름 리스트
        threshold: 분류 임계값
        return_proba: 확률 값 반환 여부
    Returns:
        예측 결과 (효과 이름 및 확률)
    """
    # 성분이 문자열인 경우 리스트로 변환
    if isinstance(ingredients_list, str):
        ingredients_list = [ing.strip() for ing in ingredients_list.split(',')]
    
    # 예측
    if return_proba:
        predictions = model.predict_proba(ingredients_list)
    else:
        predictions = model.predict(ingredients_list, threshold=threshold)
    
    # 결과 정리
    results = {}
    for i, effect in enumerate(effect_names):
        if return_proba:
            results[effect] = float(predictions[i])
        else:
            results[effect] = bool(predictions[i])
    
    return results

def predict_from_file(model, file_path, effect_names, threshold=0.5, return_proba=False, ingredients_col='ingredients'):
    """
    파일에서 여러 화장품의 효과 예측
    Args:
        model: 예측 모델
        file_path: 입력 파일 경로 (CSV)
        effect_names: 효과 이름 리스트
        threshold: 분류 임계값
        return_proba: 확률 값 반환 여부
        ingredients_col: 성분 컬럼 이름
    Returns:
        예측 결과가 포함된 데이터프레임
    """
    # 파일 로드
    df = pd.read_csv(file_path)
    
    # 성분 리스트 추출
    ingredients_lists = []
    for ingredients in df[ingredients_col]:
        if isinstance(ingredients, str):
            ingredients_lists.append([ing.strip() for ing in ingredients.split(',')])
        else:
            ingredients_lists.append([])
    
    # 예측
    if return_proba:
        batch_predictions = model.predict_proba(ingredients_lists)
    else:
        batch_predictions = model.predict(ingredients_lists, threshold=threshold)
    
    # 결과 데이터프레임에 추가
    for i, effect in enumerate(effect_names):
        if return_proba:
            df[f'pred_{effect}'] = batch_predictions[:, i]
        else:
            df[f'pred_{effect}'] = batch_predictions[:, i].astype(bool)
    
    return df

def main(args):
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    print(f"Loading model from {args.model_path}...")
    model, effect_names = load_model(args.model_path, device)
    
    # 출력 형식
    return_proba = args.output_mode == 'probability'
    
    # 입력 소스에 따라 처리
    if args.input_file:
        # 파일에서 예측
        print(f"Predicting from file {args.input_file}...")
        results_df = predict_from_file(
            model, args.input_file, effect_names, 
            threshold=args.threshold, return_proba=return_proba,
            ingredients_col=args.ingredients_col
        )
        
        # 결과 저장
        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
            print(f"Results saved to {args.output_file}")
        else:
            print(results_df)
            
    elif args.ingredients:
        # 단일 성분 리스트에서 예측
        ingredients_list = [ing.strip() for ing in args.ingredients.split(',')]
        print(f"Predicting effects for: {ingredients_list}")
        
        results = predict_from_ingredients(
            model, ingredients_list, effect_names, 
            threshold=args.threshold, return_proba=return_proba
        )
        
        # 결과 출력
        print("\nPredicted Effects:")
        if return_proba:
            # 확률 높은 순으로 정렬
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for effect, prob in sorted_results:
                print(f"{effect}: {prob:.4f}")
        else:
            # 긍정적인 효과만 출력
            positive_effects = [effect for effect, has_effect in results.items() if has_effect]
            if positive_effects:
                for effect in positive_effects:
                    print(f"- {effect}")
            else:
                print("No effects predicted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cosmetic effects from ingredients")
    
    # 입력 인자
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ingredients", type=str,
                            help="Comma-separated list of ingredients")
    input_group.add_argument("--input_file", type=str,
                            help="Path to input CSV file with ingredients")
    
    # 모델 인자
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    
    # 출력 인자
    parser.add_argument("--output_file", type=str,
                        help="Path to save output CSV file (when using --input_file)")
    parser.add_argument("--output_mode", type=str, default="binary",
                        choices=["binary", "probability"],
                        help="Output format: binary (0/1) or probability")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for binary output")
    
    # 기타 인자
    parser.add_argument("--ingredients_col", type=str, default="ingredients",
                        help="Column name for ingredients in input file")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    main(args)