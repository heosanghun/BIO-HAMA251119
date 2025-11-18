# data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import random

# 1. '인지 상태(Cognitive State)'를 표현하는 데이터 클래스 정의
@dataclass
class CognitiveState:
    """
    Bio-HAMA 모델의 내적 상태를 표현하는 다차원적 정보 벡터.
    뇌의 인지 상태를 기능적으로 모방합니다.
    """
    # 현재 과제 목표, 이전 대화 이력 등을 저장하는 작업 기억 (dlPFC 기능 모방)
    working_memory: torch.Tensor = field(
        default_factory=lambda: torch.empty(0)
    )

    # 사용자의 텍스트나 반응에서 추론된 감정적 맥락 (Amygdala-PFC 회로 모방)
    affective_context: torch.Tensor = field(
        default_factory=lambda: torch.empty(0)
    )

    # 현재 시스템이 어떤 정보에 집중하고 있는지를 나타내는 벡터 (주의 네트워크 모방)
    attention_allocation: torch.Tensor = field(
        default_factory=lambda: torch.empty(0)
    )

    # 자기 성찰 및 모니터링 정보를 담는 딕셔너리 (전전두피질의 메타인지 기능 모방)
    metacognition: Dict[str, float] = field(default_factory=lambda: {
        'model_confidence': 1.0,  # 모델 출력의 확신도 (1.0 - 엔트로피)
        'prediction_uncertainty': 0.0, # 예측의 불확실성 (MC Dropout 분산)
        'cognitive_load': 0.0, # 현재 과제의 복잡도
    })

# 2. 입출력 데이터 형식을 위한 데이터 클래스 (선택 사항이지만 명시성을 위해 추가)
@dataclass
class BioHamaInput:
    """Bio-HAMA 모델의 단일 스텝 입력을 정의합니다."""
    user_input_text: str
    previous_state: CognitiveState
    multimodal_data: Optional[torch.Tensor] = None

@dataclass
class BioHamaOutput:
    """Bio-HAMA 모델의 단일 스텝 출력을 정의합니다."""
    response_text: str
    next_state: CognitiveState
    activated_modules: List[str] # 해석 가능성을 위한 활성화 모듈 기록


# 3. 테스트를 위한 더미 데이터셋 클래스 구현
class DummyReasoningDataset(Dataset):
    """
    복합 논리 추론 및 사회적 맥락 이해 과제를 모방하는 더미 데이터셋.
    초기 개발 및 단위 테스트를 위해 사용됩니다.
    """
    def __init__(self, num_samples: int = 1000, task_type: str = 'logic'):
        """
        Args:
            num_samples (int): 생성할 총 샘플의 수.
            task_type (str): 생성할 과제의 유형 ('logic' 또는 'social').
        """
        self.num_samples = num_samples
        self.task_type = task_type

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        하나의 더미 데이터 샘플(입력 텍스트, 정답 텍스트)을 생성합니다.
        """
        if self.task_type == 'logic':
            # 예시: "숫자 73과 28 중 더 큰 것은?" -> "73"
            num1 = random.randint(1, 100)
            num2 = random.randint(1, 100)
            while num1 == num2:
                num2 = random.randint(1, 100)
            
            question = f"숫자 {num1}과 {num2} 중 더 큰 것은?"
            answer = str(max(num1, num2))

        elif self.task_type == 'social':
            # 예시: "회의가 '또' 늦어지다니 '참' 잘됐네요." -> "비꼼"
            sarcastic_phrases = [
                ("회의가 '또' 늦어지다니 '참' 잘됐네요.", "비꼼"),
                ("네 덕분에 일이 '아주' 쉬워졌어.", "비꼼"),
                ("정말 고마워. 너는 진정한 친구야.", "진심"),
                ("오늘 발표 정말 훌륭했어요.", "진심"),
            ]
            question, answer = random.choice(sarcastic_phrases)

        else:
            raise ValueError("task_type은 'logic' 또는 'social'이어야 합니다.")

        return {"input_text": question, "target_text": answer}

# 4. 데이터 로더 생성 헬퍼 함수
def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    주어진 데이터셋으로부터 PyTorch DataLoader를 생성합니다.
    
    Args:
        dataset (Dataset): PyTorch 데이터셋 객체.
        batch_size (int): 배치 크기.
        shuffle (bool): 데이터를 섞을지 여부.
        
    Returns:
        DataLoader: 설정된 DataLoader 객체.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 이 파일이 직접 실행될 때 테스트하는 코드
if __name__ == '__main__':
    print("--- 논리 추론 더미 데이터셋 테스트 ---")
    logic_dataset = DummyReasoningDataset(num_samples=5, task_type='logic')
    logic_loader = get_dataloader(logic_dataset, batch_size=2)

    for i, batch in enumerate(logic_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs: {batch['input_text']}")
        print(f"  Targets: {batch['target_text']}")
        print("-" * 20)

    print("\n--- 사회적 맥락 이해 더미 데이터셋 테스트 ---")
    social_dataset = DummyReasoningDataset(num_samples=5, task_type='social')
    social_loader = get_dataloader(social_dataset, batch_size=2)
    
    # 초기 인지 상태 예시 생성
    initial_state = CognitiveState()
    print(f"초기 인지 상태 예시:\n{initial_state}\n")

    for i, batch in enumerate(social_loader):
        print(f"Batch {i+1}:")
        # 실제 모델에서는 이 텍스트와 초기 상태가 BioHamaInput으로 래핑되어 전달됨
        print(f"  Inputs: {batch['input_text']}")
        print(f"  Targets: {batch['target_text']}")
        print("-" * 20)

