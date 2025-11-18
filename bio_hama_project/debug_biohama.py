#!/usr/bin/env python
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.bio_hama.architecture import BioHAMA
from models.bio_hama.meta_router import CognitiveState
import torch

print("=" * 70)
print("Bio-HAMA 아키텍처 직접 테스트")
print("=" * 70)

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

print("\n1. 모델 생성...")
model = BioHAMA(config)
print("✓ 모델 생성 성공")

print(f"\n2. 생성된 모듈 확인...")
print(f"  총 모듈 수: {len(model.cognitive_modules)}")
first_module_name = list(model.cognitive_modules.keys())[0]
first_module = model.cognitive_modules[first_module_name]
print(f"  첫 번째 모듈 ({first_module_name}):")
print(f"    {first_module}")

print("\n3. Forward 테스트...")
input_ids = torch.randint(0, 1000, (4, 10))
state = CognitiveState(
    working_memory=torch.randn(4, 128),
    affective_context=torch.randn(4, 128)
)

model.eval()
try:
    final_output, next_state, logits, activations = model(input_ids, state)
    print("✓ Forward 성공!")
    print(f"  최종 출력 shape: {final_output.shape}")
    print(f"  모듈 정책 로짓 shape: {logits.shape}")
    print(f"  활성화 가중치 shape: {activations.shape}")
    print(f"  활성화된 모듈 수: {activations.sum(dim=1)[0].item():.0f}")
    print("\n🎉 Bio-HAMA 테스트 성공!")
except Exception as e:
    print(f"✗ Forward 실패: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)

