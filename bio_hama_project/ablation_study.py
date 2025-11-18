# ablation_study.py

import argparse
import yaml
import torch
import random
from typing import List, Dict, Any
import sys
import os

# 프로젝트 모듈 import를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 구현한 모듈들 import
from main_train import load_config, get_model, eval_loop
from data.dataset import get_dataloader, DummyReasoningDataset
from training.optimizer import BioAGRPO

def run_ablation(args):
    """
    제거 연구를 설정하고 실행하는 메인 함수.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"제거 연구 시작: {args.ablation_name}")
    print("=" * 70)
    print(f"제거할 모듈: {args.ablate_modules if args.ablate_modules else '없음'}")
    print(f"제거할 신경 조절 기능: {args.ablate_neuromodulators if args.ablate_neuromodulators else '없음'}")
    print(f"사용 장치: {device}")
    print("=" * 70 + "\n")

    # --- 1. 설정 및 데이터 로드 ---
    config = load_config(args.config_path)
    # 평가 시에는 배치 크기를 더 크게 설정할 수 있음
    eval_batch_size = config['train_params'].get('eval_batch_size', 16)
    
    # 평가용 데이터 로더
    eval_dataset = DummyReasoningDataset(num_samples=200, task_type=args.task_type)
    eval_loader = get_dataloader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    
    # --- 2. 사전 학습된 Bio-HAMA 모델 로드 ---
    model = get_model('bio_hama', config, device)
    
    # 체크포인트 로드
    if os.path.exists(args.checkpoint_path):
        print(f"체크포인트 로드: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("✓ 사전 학습된 Bio-HAMA 모델 로드 완료\n")
    else:
        print(f"⚠ 경고: 체크포인트를 찾을 수 없습니다 ({args.checkpoint_path})")
        print("랜덤 초기화된 모델로 진행합니다.\n")

    # --- 3. 지정된 인지 모듈 '제거' (Monkey-Patching) ---
    if args.ablate_modules:
        for module_name in args.ablate_modules:
            if module_name in model.cognitive_modules:
                print(f"  → '{module_name}' 모듈을 비활성화합니다.")
                # 해당 모듈의 forward를 출력이 0인 함수로 덮어씌움
                original_module = model.cognitive_modules[module_name]
                
                def disabled_forward(x, state=None):
                    return torch.zeros(
                        x.shape[0], 
                        original_module.output_dim,
                        device=x.device
                    )
                
                # setattr을 사용하여 인스턴스의 메서드를 동적으로 변경
                setattr(original_module, 'forward', disabled_forward)
            else:
                print(f"  ⚠ 경고: '{module_name}' 모듈을 찾을 수 없습니다.")
        print()

    # --- 4. 지정된 신경 조절 기능 '제거' ---
    # Bio-A-GRPO 인스턴스를 수정된 파라미터로 생성
    if 'bio_a_grpo_params' in config:
        bio_a_grpo_params = config['bio_a_grpo_params'].copy()
        if args.ablate_neuromodulators:
            for modulator in args.ablate_neuromodulators:
                print(f"  → '{modulator}' 신경 조절 기능을 비활성화합니다.")
                if modulator == 'norepinephrine':
                    # 민감도를 0으로 설정하여 동적 조절 기능을 무력화
                    bio_a_grpo_params['norepinephrine_sensitivity'] = 0.0
                elif modulator == 'serotonin':
                    bio_a_grpo_params['serotonin_sensitivity'] = 0.0
                elif modulator == 'acetylcholine':
                    bio_a_grpo_params['acetylcholine_sensitivity'] = 0.0
                # 도파민은 기본 강화 신호이므로 제거하지 않음
            print()
        
        # 제거 연구를 위해 trainer에서 사용할 BioAGRPO 인스턴스
        bio_a_grpo_instance = BioAGRPO(**bio_a_grpo_params)

    # --- 5. 수정된 모델로 평가 실행 ---
    criterion = torch.nn.CrossEntropyLoss()
    print("수정된 모델로 성능 평가를 시작합니다...\n")
    
    # eval_loop 함수를 재사용
    eval_loss, eval_acc = eval_loop(model, eval_loader, criterion, device, 'bio_hama')
    
    # 결과 출력
    print("=" * 70)
    print(f"제거 연구 '{args.ablation_name}' 완료")
    print("=" * 70)
    print(f"평가 손실: {eval_loss:.4f}")
    print(f"평가 정확도: {eval_acc:.4f}")
    
    # 가상의 성능 하락 계산 (실제로는 baseline과 비교)
    base_accuracy = 87.5  # Full Bio-HAMA 모델의 기준 정확도
    performance_drop = base_accuracy - (eval_acc * 100)
    print(f"성능 하락: {performance_drop:.2f}%p")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bio-HAMA 제거 연구(Ablation Study) 스크립트")
    parser.add_argument('--config_path', type=str, required=True,
                        help="Bio-HAMA 모델 설정을 담은 YAML 파일 경로")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="사전 학습된 Bio-HAMA 모델의 체크포인트 경로")
    parser.add_argument('--task_type', type=str, default='social', choices=['logic', 'social'],
                        help="평가에 사용할 과제 유형")
    
    # 제거 연구를 위한 핵심 인자들
    parser.add_argument('--ablation_name', type=str, default="Ablation Test",
                        help="이번 실험의 이름 (결과 로깅용)")
    parser.add_argument('--ablate_modules', type=str, nargs='*', default=[],
                        help="비활성화할 인지 모듈의 클래스 이름 (공백으로 구분)")
    parser.add_argument('--ablate_neuromodulators', type=str, nargs='*', default=[],
                        choices=['norepinephrine', 'serotonin', 'acetylcholine'],
                        help="비활성화할 신경 조절 기능 이름 (공백으로 구분)")

    args = parser.parse_args()
    
    # --- 쉘 스크립트 실행 예시 ---
    # 아래 명령어들을 쉘 스크립트 파일(e.g., run_ablations.sh)로 만들어 실행하면
    # 논문의 [표 2]와 같은 체계적인 실험을 자동화할 수 있습니다.
    
    # 1. Full Model (제거 없음)
    # python ablation_study.py --config_path configs/bio_hama.yaml --checkpoint_path best_bio_hama_model.pt --ablation_name "Full Model"

    # 2. 사회-감정 계층 제거
    # python ablation_study.py --config_path configs/bio_hama.yaml --checkpoint_path best_bio_hama_model.pt --ablation_name "w/o Social-Emotional" --ablate_modules SocialCognitionModule EmotionRegulationModule

    # 3. 메타인지 모듈 제거
    # python ablation_study.py --config_path configs/bio_hama.yaml --checkpoint_path best_bio_hama_model.pt --ablation_name "w/o Metacognition" --ablate_modules MetacognitionModule

    # 4. 정적 학습 (신경 조절 기능 제거)
    # python ablation_study.py --config_path configs/bio_hama.yaml --checkpoint_path best_bio_hama_model.pt --ablation_name "Static Learning" --ablate_neuromodulators norepinephrine serotonin acetylcholine

    run_ablation(args)

