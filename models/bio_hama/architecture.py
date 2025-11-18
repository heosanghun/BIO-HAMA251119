# models/bio_hama/architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import sys
import os

# 이전에 정의한 컴포넌트들을 import 합니다.
# from .meta_router import HierarchicalMetaRouter, CognitiveState
# from .modules import ...

# 상대 import 문제 해결을 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.bio_hama.meta_router import HierarchicalMetaRouter, CognitiveState
from models.bio_hama.modules import (
    MultiExpertsModule, SparseAttentionModule, AttentionControlModule, MultimodalModule,
    TerminationModule, TopologyLearningModule, EvolutionaryEngineModule, PlanningModule,
    MetacognitionModule, SocialCognitionModule, EmotionRegulationModule, AdaptiveMemoryModule
)

class BioHAMA(nn.Module):
    """
    Bio-HAMA 전체 아키텍처.
    계층적 메타-라우터와 12개의 인지 모듈을 통합하여,
    뇌과학적 원리에 기반한 동적이고 효율적인 추론을 수행합니다.
    """
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): 모델의 하이퍼파라미터 및 설정을 담은 딕셔너리.
                           (예: state_dim, embed_dim, 각 모듈의 차원 등)
        """
        super().__init__()
        self.config = config
        
        # 1. 계층적 메타-라우터 초기화
        self.meta_router = HierarchicalMetaRouter(
            num_modules=len(config['module_names']),
            state_dim=config['state_dim'],
            num_sub_goals=config['num_sub_goals']
        )
        
        # 2. 12개 인지 모듈을 담을 ModuleDict 초기화
        # ModuleDict를 사용하면 모듈들이 BioHAMA의 파라미터로 올바르게 등록됩니다.
        self.cognitive_modules = nn.ModuleDict({
            name: self._create_module(name, config) for name in config['module_names']
        })

        # 3. 입력 텍스트를 임베딩하기 위한 레이어
        self.input_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])

    def _create_module(self, module_name: str, config: Dict) -> nn.Module:
        """모듈 이름에 따라 해당 인지 모듈 클래스의 인스턴스를 생성하는 헬퍼 함수."""
        # 이름으로 클래스 객체 찾기
        module_classes = {
            'MultiExpertsModule': MultiExpertsModule,
            'SparseAttentionModule': SparseAttentionModule,
            'AttentionControlModule': AttentionControlModule,
            'MultimodalModule': MultimodalModule,
            'TerminationModule': TerminationModule,
            'TopologyLearningModule': TopologyLearningModule,
            'EvolutionaryEngineModule': EvolutionaryEngineModule,
            'PlanningModule': PlanningModule,
            'MetacognitionModule': MetacognitionModule,
            'SocialCognitionModule': SocialCognitionModule,
            'EmotionRegulationModule': EmotionRegulationModule,
            'AdaptiveMemoryModule': AdaptiveMemoryModule
        }
        
        module_class = module_classes.get(module_name)
        if module_class is None:
            raise ValueError(f"알 수 없는 모듈 이름: {module_name}")
        
        # 모든 모듈을 동일한 차원으로 생성 (나중에 multimodal 입력 처리 시 수정 필요)
        return module_class(input_dim=config['embed_dim'], output_dim=config['embed_dim'])

    def forward(
        self, 
        input_ids: torch.Tensor, 
        previous_state: CognitiveState,
        gumbel_tau: float = 1.0
    ) -> Tuple[torch.Tensor, CognitiveState, torch.Tensor, torch.Tensor]:
        """
        Bio-HAMA의 전체 순방향 연산 과정.

        Args:
            input_ids (torch.Tensor): 사용자 입력의 토큰 ID 텐서.
            previous_state (CognitiveState): 이전 타임스텝의 인지 상태.
            gumbel_tau (float, optional): Gumbel-Softmax의 온도 파라미터. Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, CognitiveState, torch.Tensor, torch.Tensor]:
                - final_output: 최종 처리 결과 텐서.
                - next_state: 업데이트된 다음 인지 상태.
                - module_policy_logits: 메타-라우터가 출력한 원시 정책 로짓 (학습용).
                - module_activations: 실제 모듈 활성화 가중치 (해석용).
        """
        # --- 1. 초기 상태 준비 ---
        input_embedding = self.input_embedding(input_ids).mean(dim=1) # (batch, embed_dim)
        # TODO: 더 정교한 입력 인코딩 (e.g., Transformer Encoder) 필요
        
        # 현재 입력을 작업 기억에 추가 (단순화된 버전)
        current_state = previous_state
        current_state.working_memory = input_embedding # 이전 기억과 결합하는 로직 필요
        current_state.affective_context = input_embedding # 감정 분석 로직 필요

        # --- 2. 메타-라우터의 모듈 선택 ---
        module_policy_logits, _ = self.meta_router(current_state)

        # --- 3. Gumbel-Softmax를 이용한 조건부 계산 ---
        # training 모드에서는 stochastic, eval 모드에서는 deterministic(argmax)
        if self.training:
            module_activations = F.gumbel_softmax(module_policy_logits, tau=gumbel_tau, hard=False)
        else:
            # 추론 시에는 가장 확률이 높은 모듈을 선택 (hard routing)
            # top-k를 사용하여 여러 모듈을 선택할 수도 있음
            top_k = self.config.get('routing_top_k', 3)
            _, top_indices = torch.topk(module_policy_logits, k=top_k, dim=-1)
            module_activations = torch.zeros_like(module_policy_logits).scatter_(-1, top_indices, 1.0)
            
        # --- 4. 선택된 모듈만 희소하게 활성화 ---
        # 각 모듈의 출력을 계산하고 활성화 가중치를 곱함
        module_outputs = []
        for i, name in enumerate(self.config['module_names']):
            # i번째 모듈의 활성화 가중치 (batch_size, 1)
            activation_weight = module_activations[:, i].unsqueeze(1)
            
            # 가중치가 0에 가까우면 실제 연산을 건너뛸 수 있음 (효율성)
            # 여기서는 모든 모듈을 실행하되, 출력을 가중치로 조절
            module_instance = self.cognitive_modules[name]
            
            # TODO: 각 모듈에 필요한 입력을 동적으로 전달하는 로직 필요
            # 예: 'AdaptiveMemoryModule'은 working_memory 전체를 필요로 할 수 있음
            output = module_instance(input_embedding, state=current_state)
            
            module_outputs.append(output * activation_weight)
        
        # --- 5. 모듈 출력 통합 ---
        # 모든 활성화된 모듈의 출력을 합산 (또는 다른 통합 방식 사용 가능)
        final_output = torch.stack(module_outputs, dim=0).sum(dim=0)
        
        # --- 6. 다음 인지 상태 업데이트 ---
        # TODO: 메타인지 모듈 등을 실행하여 confidence, uncertainty 등을 업데이트하는 로직 필요
        next_state = self._update_cognitive_state(current_state, final_output)
        
        return final_output, next_state, module_policy_logits, module_activations

    def _update_cognitive_state(self, current_state: CognitiveState, final_output: torch.Tensor) -> CognitiveState:
        """
        처리 결과를 바탕으로 다음 스텝의 인지 상태를 업데이트합니다.
        """
        # TODO: 메타인지 모듈을 호출하여 'model_confidence', 'prediction_uncertainty' 업데이트
        # 예: metacognition_module = self.cognitive_modules['MetacognitionModule']
        # meta_output = metacognition_module(final_output) ...
        next_state = current_state # 단순화를 위해 그대로 전달
        return next_state


# 이 파일이 직접 실행될 때 모델 구조를 테스트하는 코드
if __name__ == '__main__':
    print("--- 전체 Bio-HAMA 아키텍처 테스트 ---")
    
    # 더미 설정
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
    
    BATCH_SIZE = 4
    SEQ_LEN = 10

    # 모델 인스턴스 생성
    bio_hama_model = BioHAMA(config)
    
    # 더미 입력 데이터
    dummy_input_ids = torch.randint(0, config['vocab_size'], (BATCH_SIZE, SEQ_LEN))
    dummy_state = CognitiveState(
        working_memory=torch.randn(BATCH_SIZE, config['embed_dim']),
        affective_context=torch.randn(BATCH_SIZE, config['embed_dim'])
    )

    # 1. 학습 모드 테스트
    bio_hama_model.train()
    print("\n[학습 모드 테스트 (Gumbel-Softmax 사용)]")
    final_output, next_state, logits, activations = bio_hama_model(dummy_input_ids, dummy_state)
    
    print("입력 ID shape:", dummy_input_ids.shape)
    print("최종 출력 shape:", final_output.shape) # (Batch, embed_dim)
    print("정책 로짓 shape:", logits.shape)     # (Batch, 12)
    print("활성화 가중치 shape:", activations.shape) # (Batch, 12)
    print("샘플 활성화 가중치 (합이 1에 가까움):", activations[0])
    print("활성화된 모듈 수 (이론상):", (activations > 0.01).sum(dim=1))
    
    # 2. 평가 모드 테스트
    bio_hama_model.eval()
    print("\n[평가 모드 테스트 (Top-k Hard Routing 사용)]")
    final_output, next_state, logits, activations = bio_hama_model(dummy_input_ids, dummy_state)

    print("최종 출력 shape:", final_output.shape)
    print("활성화 가중치 (0 또는 1):", activations[0])
    print(f"활성화된 모듈 수 (Top-k={config['routing_top_k']}):", activations.sum(dim=1))

    print("\nBio-HAMA 모델 구조:")
    print(bio_hama_model)

