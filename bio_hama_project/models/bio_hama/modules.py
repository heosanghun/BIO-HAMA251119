# models/bio_hama/modules.py

import torch
import torch.nn as nn
from typing import Dict, Any

class CognitiveModule(nn.Module):
    """
    모든 인지 모듈이 상속받는 추상 기본 클래스(Abstract Base Class).
    각 모듈은 특정 뇌 영역의 전문화된 기능을 계산적으로 모델링합니다.
    """
    def __init__(self, module_name: str, input_dim: int, output_dim: int):
        super().__init__()
        self.module_name = module_name
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, state: Dict[str, Any] = None) -> torch.Tensor:
        """
        각 모듈의 정방향 연산을 정의하는 인터페이스.

        Args:
            x (torch.Tensor): 주 입력 텐서.
            state (Dict[str, Any], optional): 추가적인 상태 정보 (e.g., CognitiveState).

        Returns:
            torch.Tensor: 모듈의 처리 결과 텐서.
        """
        raise NotImplementedError("모든 인지 모듈은 forward 메소드를 구현해야 합니다.")

# --- 간단한 MLP 플레이스홀더 ---
class PlaceholderMLP(CognitiveModule):
    """초기 구현을 위한 간단한 MLP 기반 인지 모듈."""
    def __init__(self, module_name: str, input_dim: int, output_dim: int, hidden_dim: int = None):
        super().__init__(module_name, input_dim, output_dim)
        # hidden_dim이 지정되지 않으면 input_dim과 output_dim의 평균값 사용
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, state: Dict[str, Any] = None) -> torch.Tensor:
        # TODO: Implement detailed logic based on paper for each module
        return self.network(x)

# --- 12개 인지 모듈 클래스 정의 (플레이스홀더 구현) ---
# 각 모듈의 input_dim과 output_dim은 전체 아키텍처 설계에 따라 달라지므로,
# 여기서는 동일한 값으로 가정하고 나중에 수정합니다.
DIM = 512 # 예시 차원

# 1. 기본 처리 계층
class MultiExpertsModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("MultiExpertsModule", input_dim, output_dim)

class SparseAttentionModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("SparseAttentionModule", input_dim, output_dim)

class AttentionControlModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("AttentionControlModule", input_dim, output_dim)

class MultimodalModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim): # 예: 텍스트+이미지 입력
        super().__init__("MultimodalModule", input_dim, output_dim)

class TerminationModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim): # 출력 생성 담당
        super().__init__("TerminationModule", input_dim, output_dim)

# 2. 고차 인지 계층
class TopologyLearningModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("TopologyLearningModule", input_dim, output_dim)

class EvolutionaryEngineModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("EvolutionaryEngineModule", input_dim, output_dim)

class PlanningModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("PlanningModule", input_dim, output_dim)

class MetacognitionModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("MetacognitionModule", input_dim, output_dim)

# 3. 사회-감정 계층
class SocialCognitionModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("SocialCognitionModule", input_dim, output_dim)
        
class EmotionRegulationModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("EmotionRegulationModule", input_dim, output_dim)

class AdaptiveMemoryModule(PlaceholderMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__("AdaptiveMemoryModule", input_dim, output_dim)


# 이 파일이 직접 실행될 때 모듈 구조를 테스트하는 코드
if __name__ == '__main__':
    print("--- 인지 모듈 플레이스홀더 테스트 ---")

    # 예시 모듈 인스턴스 생성
    social_module = SocialCognitionModule(input_dim=512, output_dim=512)
    planning_module = PlanningModule(input_dim=512, output_dim=512)

    # 더미 입력
    dummy_input = torch.randn(4, 512) # (Batch_size, dim)

    # 모듈 실행 테스트
    social_output = social_module(dummy_input)
    planning_output = planning_module(dummy_input)

    print(f"모듈 이름: {social_module.module_name}")
    print(f"입력 텐서 shape: {dummy_input.shape}")
    print(f"출력 텐서 shape: {social_output.shape}")
    print("-" * 30)
    
    print(f"모듈 이름: {planning_module.module_name}")
    print(f"입력 텐서 shape: {dummy_input.shape}")
    print(f"출력 텐서 shape: {planning_output.shape}")
    print("-" * 30)

    # 전체 모듈 리스트 생성 및 확인
    all_modules = {
        "multi_experts": MultiExpertsModule(),
        "sparse_attention": SparseAttentionModule(),
        "attention_control": AttentionControlModule(),
        "multimodal": MultimodalModule(input_dim=DIM*2),
        "termination": TerminationModule(),
        "topology_learning": TopologyLearningModule(),
        "evolutionary_engine": EvolutionaryEngineModule(),
        "planning": PlanningModule(),
        "metacognition": MetacognitionModule(),
        "social_cognition": SocialCognitionModule(),
        "emotion_regulation": EmotionRegulationModule(),
        "adaptive_memory": AdaptiveMemoryModule()
    }

    print(f"총 {len(all_modules)}개의 인지 모듈이 정의되었습니다.")
    print(list(all_modules.keys()))

