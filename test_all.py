#!/usr/bin/env python
# test_all.py
"""
Bio-HAMA 프로젝트 전체 테스트 스크립트
각 Phase의 모듈들이 제대로 작동하는지 검증합니다.
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_phase2_dataset():
    """Phase 2: 데이터셋 테스트"""
    print("\n" + "=" * 70)
    print("Phase 2 테스트: 데이터셋 및 CognitiveState")
    print("=" * 70)
    try:
        from data.dataset import DummyReasoningDataset, CognitiveState, get_dataloader
        import torch
        
        # CognitiveState 테스트
        state = CognitiveState()
        print("✓ CognitiveState 생성 성공")
        
        # Dataset 테스트
        dataset = DummyReasoningDataset(num_samples=10, task_type='logic')
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        print(f"✓ Dataset 생성 성공 (샘플 수: {len(dataset)})")
        print(f"✓ DataLoader 생성 성공 (배치 크기: {len(batch['input_text'])})")
        print("Phase 2: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 2 테스트 실패: {e}\n")
        return False

def test_phase3_baselines():
    """Phase 3: 베이스라인 모델 테스트"""
    print("=" * 70)
    print("Phase 3 테스트: 베이스라인 모델")
    print("=" * 70)
    try:
        from models.baselines import BaselineLSTM, BaselineGRU
        import torch
        
        # LSTM 테스트
        lstm = BaselineLSTM(vocab_size=1000, embed_size=128, hidden_size=256, num_layers=2)
        dummy_input = torch.randint(0, 1000, (4, 20))
        lstm_output = lstm(dummy_input)
        print(f"✓ LSTM 모델 생성 및 forward 성공 (출력 shape: {lstm_output['logits'].shape})")
        
        # GRU 테스트
        gru = BaselineGRU(vocab_size=1000, embed_size=128, hidden_size=256, num_layers=2)
        gru_output = gru(dummy_input)
        print(f"✓ GRU 모델 생성 및 forward 성공 (출력 shape: {gru_output['logits'].shape})")
        
        print("Phase 3: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 3 테스트 실패: {e}\n")
        return False

def test_phase4_1_modules():
    """Phase 4-1: 인지 모듈 테스트"""
    print("=" * 70)
    print("Phase 4-1 테스트: 12개 인지 모듈")
    print("=" * 70)
    try:
        from models.bio_hama.modules import (
            SocialCognitionModule, PlanningModule, MetacognitionModule
        )
        import torch
        
        # 모듈 생성 테스트
        social_module = SocialCognitionModule(input_dim=256, output_dim=256)
        planning_module = PlanningModule(input_dim=256, output_dim=256)
        meta_module = MetacognitionModule(input_dim=256, output_dim=256)
        
        # Forward 테스트
        dummy_input = torch.randn(4, 256)
        social_output = social_module(dummy_input)
        
        print(f"✓ 인지 모듈 생성 성공 (3개 테스트)")
        print(f"✓ Forward 연산 성공 (출력 shape: {social_output.shape})")
        print("Phase 4-1: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 4-1 테스트 실패: {e}\n")
        return False

def test_phase4_2_meta_router():
    """Phase 4-2: 메타-라우터 테스트"""
    print("=" * 70)
    print("Phase 4-2 테스트: 계층적 메타-라우터")
    print("=" * 70)
    try:
        from models.bio_hama.meta_router import HierarchicalMetaRouter, CognitiveState
        import torch
        
        # 메타-라우터 생성
        router = HierarchicalMetaRouter(num_modules=12, state_dim=128, num_sub_goals=10)
        
        # 더미 상태 생성
        state = CognitiveState(
            working_memory=torch.randn(4, 256),
            affective_context=torch.randn(4, 256)
        )
        
        # Forward 테스트
        module_logits, goal_vec = router(state)
        
        print(f"✓ 메타-라우터 생성 성공")
        print(f"✓ Forward 연산 성공")
        print(f"  - 모듈 정책 로짓 shape: {module_logits.shape}")
        print(f"  - 선택된 목표 shape: {goal_vec.shape}")
        print("Phase 4-2: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 4-2 테스트 실패: {e}\n")
        return False

def test_phase4_3_bio_a_grpo():
    """Phase 4-3: Bio-A-GRPO 테스트"""
    print("=" * 70)
    print("Phase 4-3 테스트: Bio-A-GRPO 학습 알고리즘")
    print("=" * 70)
    try:
        from training.optimizer import BioAGRPO, CognitiveState
        import torch
        
        # BioAGRPO 인스턴스 생성
        bio_a_grpo = BioAGRPO(num_modules=12)
        
        # 더미 인지 상태
        state = CognitiveState(
            metacognition={'prediction_uncertainty': 0.5, 'cognitive_load': 0.3},
            attention_allocation=torch.ones(12)
        )
        
        # 동적 파라미터 계산
        params = bio_a_grpo.calculate_dynamic_params(state)
        
        print(f"✓ Bio-A-GRPO 인스턴스 생성 성공")
        print(f"✓ 동적 파라미터 계산 성공")
        print(f"  - dynamic_lr: {params['dynamic_lr']:.6f}")
        print(f"  - dynamic_gamma: {params['dynamic_gamma']:.6f}")
        print(f"  - dynamic_epsilon: {params['dynamic_epsilon']:.6f}")
        print("Phase 4-3: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 4-3 테스트 실패: {e}\n")
        return False

def test_phase4_4_bio_hama():
    """Phase 4-4: Bio-HAMA 아키텍처 테스트"""
    print("=" * 70)
    print("Phase 4-4 테스트: 전체 Bio-HAMA 아키텍처")
    print("=" * 70)
    try:
        from models.bio_hama.architecture import BioHAMA
        from models.bio_hama.meta_router import CognitiveState
        import torch
        
        # 설정
        config = {
            'vocab_size': 1000,
            'embed_dim': 128,
            'state_dim': 64,
            'num_sub_goals': 5,
            'routing_top_k': 3,
            'module_names': [
                "SocialCognitionModule", "PlanningModule", "MetacognitionModule",
                "EmotionRegulationModule", "AdaptiveMemoryModule", "MultiExpertsModule",
                "SparseAttentionModule", "AttentionControlModule", "MultimodalModule",
                "TerminationModule", "TopologyLearningModule", "EvolutionaryEngineModule"
            ]
        }
        
        # 모델 생성
        model = BioHAMA(config)
        
        # 더미 입력
        input_ids = torch.randint(0, 1000, (4, 10))
        state = CognitiveState(
            working_memory=torch.randn(4, 128),
            affective_context=torch.randn(4, 128)
        )
        
        # Forward 테스트
        model.eval()
        final_output, next_state, logits, activations = model(input_ids, state)
        
        print(f"✓ Bio-HAMA 모델 생성 성공")
        print(f"✓ Forward 연산 성공")
        print(f"  - 최종 출력 shape: {final_output.shape}")
        print(f"  - 모듈 정책 로짓 shape: {logits.shape}")
        print(f"  - 활성화 가중치 shape: {activations.shape}")
        print(f"  - 활성화된 모듈 수: {activations.sum(dim=1)[0].item():.0f}")
        print("Phase 4-4: 성공 ✓\n")
        return True
    except Exception as e:
        print(f"✗ Phase 4-4 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """모든 테스트 실행"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Bio-HAMA 전체 테스트 시작" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = []
    
    # 각 Phase 테스트 실행
    results.append(("Phase 2: 데이터셋", test_phase2_dataset()))
    results.append(("Phase 3: 베이스라인", test_phase3_baselines()))
    results.append(("Phase 4-1: 인지 모듈", test_phase4_1_modules()))
    results.append(("Phase 4-2: 메타-라우터", test_phase4_2_meta_router()))
    results.append(("Phase 4-3: Bio-A-GRPO", test_phase4_3_bio_a_grpo()))
    results.append(("Phase 4-4: Bio-HAMA", test_phase4_4_bio_hama()))
    
    # 결과 요약
    print("=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = "✓ 통과" if result else "✗ 실패"
        print(f"{name}: {status}")
    
    print("=" * 70)
    print(f"전체 결과: {passed_tests}/{total_tests} 테스트 통과")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트 통과! 프로젝트가 정상적으로 작동합니다.")
    else:
        print(f"⚠ {total_tests - passed_tests}개의 테스트가 실패했습니다.")
    
    print("=" * 70 + "\n")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

