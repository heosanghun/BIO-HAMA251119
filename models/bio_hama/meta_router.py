# models/bio_hama/meta_router.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# Phase 2에서 정의했던 CognitiveState를 가져옵니다.
# 실제 프로젝트에서는 from data.dataset import CognitiveState 와 같이 import 합니다.

@dataclass
class CognitiveState:
    working_memory: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    affective_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    attention_allocation: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    metacognition: Dict[str, float] = field(default_factory=lambda: {
        'model_confidence': 1.0, 'prediction_uncertainty': 0.0, 'cognitive_load': 0.0
    })

# --- 메타-라우터의 각 계층 구현 ---

class StrategyLayer(nn.Module):
    """
    상위층: 전략적 목표 설정 (복내측 전전두피질(vmPFC) 모방)
    장기적인 목표나 대화의 전체 의도를 파악하여 하위 목표를 설정합니다.
    """
    def __init__(self, state_dim: int, num_sub_goals: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, num_sub_goals),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector (torch.Tensor): 현재 인지 상태를 나타내는 벡터.

        Returns:
            torch.Tensor: 각 하위 목표에 대한 로짓(logits).
        """
        sub_goal_logits = self.network(state_vector)
        return sub_goal_logits

class TacticsLayer(nn.Module):
    """
    중위층: 전술적 실행 계획 (배외측 전전두피질(dlPFC) 모방)
    설정된 하위 목표를 달성하기 위해 인지 모듈의 실행 계획을 수립합니다.
    """
    def __init__(self, state_dim: int, num_modules: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, num_modules)
        )

    def forward(self, state_vector_with_goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector_with_goal (torch.Tensor): 인지 상태와 현재 목표가 결합된 벡터.

        Returns:
            torch.Tensor: 12개 인지 모듈을 활성화할지에 대한 로짓(정책).
        """
        module_logits = self.network(state_vector_with_goal)
        return module_logits

class ResponseLayer(nn.Module):
    """
    하위층: 즉각적 반응 및 오류 제어 (전방 대상피질(ACC) 모방)
    실행 과정을 모니터링하고 충돌이나 오류를 감지합니다.
    """
    def __init__(self, state_dim: int, num_control_signals: int = 2): # 예: [continue, replan]
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, num_control_signals)
        )

    def forward(self, state_vector_with_feedback: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector_with_feedback (torch.Tensor): 모듈 실행 후 피드백이 포함된 상태 벡터.

        Returns:
            torch.Tensor: 제어 신호에 대한 로짓 (계속 진행할지, 재계획할지 등).
        """
        control_logits = self.network(state_vector_with_feedback)
        return control_logits

# --- 계층적 메타-라우터 통합 ---

class HierarchicalMetaRouter(nn.Module):
    """
    Bio-HAMA의 중앙 제어 장치. 3계층 구조를 통해 인지 과정을 조율합니다.
    """
    def __init__(self, num_modules: int, state_dim: int, num_sub_goals: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.num_sub_goals = num_sub_goals

        self.strategy_layer = StrategyLayer(state_dim, num_sub_goals)
        # 전술층은 상태와 목표를 함께 입력받으므로 입력 차원이 더 큼
        self.tactics_layer = TacticsLayer(state_dim + num_sub_goals, num_modules)
        self.response_layer = ResponseLayer(state_dim) # 단순화를 위해 피드백 없이 상태만 사용

        # CognitiveState 객체를 고정된 크기의 벡터로 변환하는 인코더
        self.state_encoder = nn.Linear(1024, state_dim) # 예시 차원, 실제로는 각 필드를 임베딩하고 합산

    def _encode_state(self, state: CognitiveState) -> torch.Tensor:
        """CognitiveState 객체를 단일 벡터로 인코딩하는 헬퍼 함수."""
        # TODO: 실제 구현에서는 working_memory, affective_context 등 텐서들을
        # MLP나 어텐션을 통해 하나의 벡터로 압축하는 정교한 로직이 필요.
        # 여기서는 단순화를 위해 더미 텐서를 생성.
        # 가정: working_memory, affective_context 등이 concat되어 (batch, 1024) 형태
        
        batch_size = state.working_memory.shape[0] if state.working_memory.nelement() > 0 else 1
        device = state.working_memory.device if state.working_memory.nelement() > 0 else torch.device('cpu')
        
        # 실제로는 상태 벡터들을 결합해야 하지만, 여기서는 랜덤 벡터로 대체
        # 실제 구현 시 이 부분을 수정해야 함
        if state.working_memory.nelement() > 0 and state.affective_context.nelement() > 0:
            dummy_state_tensors = torch.cat([
                state.working_memory,
                state.affective_context
            ], dim=-1)
            # 1024 차원으로 맞춤
            if dummy_state_tensors.shape[-1] < 1024:
                padding = torch.zeros(batch_size, 1024 - dummy_state_tensors.shape[-1], device=device)
                dummy_state_tensors = torch.cat([dummy_state_tensors, padding], dim=-1)
            elif dummy_state_tensors.shape[-1] > 1024:
                dummy_state_tensors = dummy_state_tensors[:, :1024]
            state_vector = self.state_encoder(dummy_state_tensors)
        else:
            state_vector = torch.randn(batch_size, self.state_dim, device=device)
        
        return state_vector

    def forward(self, state: CognitiveState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        메타-라우터의 전체 제어 흐름.

        Args:
            state (CognitiveState): 현재 시스템의 인지 상태.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - module_policy_logits: 모듈 선택을 위한 정책 로짓.
                - selected_sub_goal: 선택된 하위 목표 벡터.
        """
        # 1. 현재 인지 상태를 벡터로 인코딩
        state_vector = self._encode_state(state)

        # 2. 전략층(StrategyLayer)을 통해 하위 목표 설정
        sub_goal_logits = self.strategy_layer(state_vector)
        # 가장 확률이 높은 하위 목표를 선택 (실제로는 샘플링 가능)
        selected_sub_goal = torch.softmax(sub_goal_logits, dim=-1)

        # 3. 전술층(TacticsLayer)을 위해 상태와 목표를 결합
        state_with_goal = torch.cat([state_vector, selected_sub_goal], dim=-1)
        
        # 4. 전술층을 통해 활성화할 모듈 정책 결정
        module_policy_logits = self.tactics_layer(state_with_goal)
        
        # 5. 반응층(ResponseLayer)은 실제 모듈 실행 후에 호출되어야 함
        #    (여기서는 순방향 흐름만 정의)

        return module_policy_logits, selected_sub_goal


# 이 파일이 직접 실행될 때 모델 구조를 테스트하는 코드
if __name__ == '__main__':
    print("--- 계층적 메타-라우터 테스트 ---")
    
    # 테스트용 파라미터
    NUM_MODULES = 12
    STATE_DIM = 256
    NUM_SUB_GOALS = 10
    BATCH_SIZE = 4

    # 더미 인지 상태 생성 (실제로는 이전 스텝에서 계산됨)
    dummy_state = CognitiveState(
        working_memory=torch.randn(BATCH_SIZE, 512),
        affective_context=torch.randn(BATCH_SIZE, 512)
    )

    # 메타-라우터 인스턴스 생성
    meta_router = HierarchicalMetaRouter(NUM_MODULES, STATE_DIM, NUM_SUB_GOALS)
    
    # 순방향 연산 테스트
    module_logits, goal_vec = meta_router(dummy_state)
    
    print(f"입력: {BATCH_SIZE}개의 CognitiveState 객체")
    print(f"출력 (모듈 정책 로짓) shape: {module_logits.shape}") # (Batch, Num_modules)
    print(f"출력 (선택된 목표) shape: {goal_vec.shape}")   # (Batch, Num_sub_goals)
    print("-" * 30)

    print("메타-라우터의 각 계층:")
    print(meta_router)

