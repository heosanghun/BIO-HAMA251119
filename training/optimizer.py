# training/optimizer.py

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass, field

# Phase 2에서 정의한 CognitiveState를 import
# from data.dataset import CognitiveState

@dataclass
class CognitiveState:
    working_memory: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    affective_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    attention_allocation: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    metacognition: Dict[str, float] = field(default_factory=lambda: {
        'model_confidence': 1.0, 'prediction_uncertainty': 0.0, 'cognitive_load': 0.0
    })


class BioAGRPO:
    """
    Bio-Adaptive Group Relative Policy Optimization.
    뇌의 신경전달물질 시스템에서 영감을 받아 학습 파라미터를 동적으로 조절하는
    메타학습 알고리즘. Actor-Critic 학습 루프 내에서 사용되는 헬퍼 클래스입니다.
    """
    def __init__(
        self,
        base_lr: float = 1e-4,
        base_gamma: float = 0.99,
        base_epsilon: float = 0.2,
        norepinephrine_sensitivity: float = 1.0,
        serotonin_sensitivity: float = 0.1,
        acetylcholine_sensitivity: float = 1.5,
        num_modules: int = 12
    ):
        """
        Args:
            base_lr (float): 기본 학습률 (alpha_base).
            base_gamma (float): 기본 할인율 (gamma_0).
            base_epsilon (float): 기본 탐험률 (epsilon_base).
            norepinephrine_sensitivity (float): 불확실성에 대한 학습률/탐험률 민감도.
            serotonin_sensitivity (float): 인지 부하에 대한 할인율 민감도.
            acetylcholine_sensitivity (float): 주의 할당에 대한 모듈별 학습률 민감도.
            num_modules (int): 전체 인지 모듈의 수.
        """
        self.base_lr = base_lr
        self.base_gamma = base_gamma
        self.base_epsilon = base_epsilon
        self.norepinephrine_sensitivity = norepinephrine_sensitivity
        self.serotonin_sensitivity = serotonin_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.num_modules = num_modules
    
    def calculate_dynamic_params(
        self, 
        cognitive_state: CognitiveState
    ) -> Dict[str, any]:
        """
        현재 인지 상태를 기반으로 핵심 학습 파라미터를 동적으로 계산합니다.

        Args:
            cognitive_state (CognitiveState): 현재 스텝의 인지 상태.

        Returns:
            Dict[str, any]: 동적으로 조절된 학습 파라미터 딕셔너리.
        """
        meta_cog = cognitive_state.metacognition
        uncertainty = meta_cog.get('prediction_uncertainty', 0.0)
        cognitive_load = meta_cog.get('cognitive_load', 0.0)
        
        # 1. 노르에피네프린 시스템 모방 (Norepinephrine): 불확실성 기반 각성 조절
        # 불확실성이 높을수록(새로운 상황) 더 빠르게 학습하고(학습률 증가) 더 많이 탐험(탐험률 증가).
        arousal_level = torch.sigmoid(torch.tensor(uncertainty * self.norepinephrine_sensitivity))
        dynamic_lr = self.base_lr * (1 + arousal_level)
        dynamic_epsilon = self.base_epsilon * (1 + arousal_level)
        
        # 2. 세로토닌 시스템 모방 (Serotonin): 인내심 조절
        # 인지 부하가 높을수록(복잡한 장기 과제) 미래 보상을 더 중요하게 고려 (할인율 증가).
        patience_level = torch.sigmoid(torch.tensor(cognitive_load * self.serotonin_sensitivity))
        dynamic_gamma = self.base_gamma + (1 - self.base_gamma) * patience_level

        # 3. 아세틸콜린 시스템 모방 (Acetylcholine): 주의 기반 학습 가중치 조절
        # '주의 할당' 벡터를 기반으로 특정 모듈의 학습을 강화.
        # attention_allocation이 (batch, num_modules) 형태의 가중치 벡터라고 가정.
        attention_weights = cognitive_state.attention_allocation
        if attention_weights is None or attention_weights.nelement() == 0:
            attention_weights = torch.ones(self.num_modules)
        
        per_module_lr_factor = 1 + (attention_weights * self.acetylcholine_sensitivity)
        
        return {
            "dynamic_lr": dynamic_lr.item(),
            "dynamic_gamma": dynamic_gamma.item(),
            "dynamic_epsilon": dynamic_epsilon.item(),
            "per_module_lr_factor": per_module_lr_factor # 벡터 형태
        }

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        dynamic_gamma: float,
        gae_lambda: float = 0.95
    ) -> torch.Tensor:
        """
        Generalized Advantage Estimation (GAE)을 동적 할인율(gamma)을 사용하여 계산합니다.
        
        Args:
            rewards (torch.Tensor): 에피소드의 보상 시퀀스.
            values (torch.Tensor): 에피소드의 각 상태 가치 시퀀스.
            dones (torch.Tensor): 에피소드 종료 여부 시퀀스.
            dynamic_gamma (float): 현재 상태에서 계산된 동적 할인율.
            gae_lambda (float): GAE 람다 파라미터.
            
        Returns:
            torch.Tensor: 계산된 Advantage 값 텐서.
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            
            # 1. TD-Error 계산 (도파민 시스템 모방)
            # 다음 상태의 value가 없는 마지막 스텝 처리
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            td_error = rewards[t] + dynamic_gamma * next_value * mask - values[t]
            
            # 2. Advantage 계산 (동적 gamma 사용)
            last_advantage = td_error + dynamic_gamma * gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
            
        return advantages


# 이 파일이 직접 실행될 때 알고리즘 로직을 테스트하는 코드
if __name__ == "__main__":
    print("--- Bio-A-GRPO 동적 파라미터 계산 테스트 ---")
    
    # 시나리오 1: 안정적이고 확실한 상황
    stable_state = CognitiveState(
        metacognition={'prediction_uncertainty': 0.1, 'cognitive_load': 0.2},
        attention_allocation=torch.tensor([0.1, 0.1, 0.8]) # 3개 모듈, 3번째 모듈에 집중
    )
    
    # 시나리오 2: 불확실하고 복잡한 상황
    uncertain_state = CognitiveState(
        metacognition={'prediction_uncertainty': 0.9, 'cognitive_load': 0.8},
        attention_allocation=torch.tensor([0.6, 0.3, 0.1]) # 1번째 모듈에 집중
    )

    bio_optimizer = BioAGRPO(num_modules=3)

    print("\n[시나리오 1: 안정적 상황]")
    stable_params = bio_optimizer.calculate_dynamic_params(stable_state)
    for key, val in stable_params.items():
        print(f"  - {key}: {val}")

    print("\n[시나리오 2: 불확실한 상황]")
    uncertain_params = bio_optimizer.calculate_dynamic_params(uncertain_state)
    for key, val in uncertain_params.items():
        print(f"  - {key}: {val}")

    # Advantage 계산 테스트
    print("\n--- Advantage 계산 테스트 ---")
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.1, 0.2, 0.3])
    dones = torch.tensor([0, 0, 1])
    
    # 안정적 상황의 동적 gamma 사용
    adv = bio_optimizer.compute_advantages(rewards, values, dones, stable_params['dynamic_gamma'])
    print(f"계산된 Advantages (안정적 gamma): {adv}")

