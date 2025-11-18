# models/baselines.py

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Dict

class BaselineLSTM(nn.Module):
    """
    LSTM 기반의 간단한 Encoder-Decoder Seq2Seq 모델.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        """
        Args:
            vocab_size (int): 전체 어휘 사전의 크기.
            embed_size (int): 임베딩 벡터의 차원.
            hidden_size (int): LSTM hidden state의 차원.
            num_layers (int): LSTM 레이어의 수.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # 디코더는 간단하게 인코더의 마지막 hidden state를 받아 처리하는 것으로 단순화
        # 실제 Seq2Seq에서는 별도의 디코더 LSTM이 필요
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        입력 시퀀스를 받아 출력 시퀀스의 로짓을 반환합니다.

        Args:
            input_ids (torch.Tensor): 입력 텍스트의 토큰 ID 텐서. (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): 어텐션 마스크. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: 'logits' 키를 포함하는 딕셔너리.
        """
        embedded = self.embedding(input_ids)
        
        # LSTM Encoder
        # h_0, c_0는 0으로 초기화됨
        _, (hidden, cell) = self.lstm_encoder(embedded)
        
        # 마지막 레이어의 hidden state를 사용하여 예측
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        logits = self.fc(last_hidden) # (batch_size, vocab_size)
        # Seq2Seq 출력을 모방하기 위해 차원 확장
        logits = logits.unsqueeze(1) # (batch_size, 1, vocab_size)

        return {'logits': logits}


class BaselineGRU(nn.Module):
    """
    GRU 기반의 간단한 Encoder-Decoder Seq2Seq 모델.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        """
        Args:
            vocab_size (int): 전체 어휘 사전의 크기.
            embed_size (int): 임베딩 벡터의 차원.
            hidden_size (int): GRU hidden state의 차원.
            num_layers (int): GRU 레이어의 수.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru_encoder = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        입력 시퀀스를 받아 출력 시퀀스의 로짓을 반환합니다.

        Args:
            input_ids (torch.Tensor): 입력 텍스트의 토큰 ID 텐서. (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): 어텐션 마스크. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: 'logits' 키를 포함하는 딕셔너리.
        """
        embedded = self.embedding(input_ids)
        
        # GRU Encoder
        _, hidden = self.gru_encoder(embedded)
        
        # 마지막 레이어의 hidden state를 사용하여 예측
        last_hidden = hidden[-1]
        
        logits = self.fc(last_hidden)
        logits = logits.unsqueeze(1)

        return {'logits': logits}


class BaselineTransformer(nn.Module):
    """
    Hugging Face의 T5 모델을 활용한 Transformer 기반 Seq2Seq 모델.
    단일 블록, 고성능 베이스라인의 역할을 합니다.
    """
    def __init__(self, model_name: str = "t5-small", tokenizer: T5Tokenizer = None):
        """
        Args:
            model_name (str, optional): 사용할 사전 학습된 T5 모델 이름. Defaults to "t5-small".
            tokenizer (T5Tokenizer, optional): T5 토크나이저. generate 메서드 사용 시 필요.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Hugging Face T5 모델의 forward pass를 직접 호출합니다.

        Args:
            input_ids (torch.Tensor): 인코더 입력 토큰 ID.
            attention_mask (torch.Tensor): 인코더 어텐션 마스크.
            labels (torch.Tensor, optional): 디코더 타겟 토큰 ID. 
                                            학습 시에는 loss 계산을 위해 필요.

        Returns:
            Dict[str, torch.Tensor]: T5 모델의 출력을 그대로 반환 (loss, logits 등 포함).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
        
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
        추론 시 텍스트 생성을 위한 메서드.
        """
        if self.tokenizer is None:
            raise ValueError("generate를 사용하려면 토크나이저가 필요합니다.")
        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# 이 파일이 직접 실행될 때 모델 구조를 테스트하는 코드
if __name__ == '__main__':
    # 테스트용 파라미터
    BATCH_SIZE = 4
    SEQ_LEN = 20
    VOCAB_SIZE = 1000
    EMBED_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2

    # 더미 입력 데이터
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # 1. LSTM 모델 테스트
    print("--- LSTM 모델 테스트 ---")
    lstm_model = BaselineLSTM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    lstm_output = lstm_model(dummy_input_ids)
    print("Input shape:", dummy_input_ids.shape)
    print("Output logits shape:", lstm_output['logits'].shape) # (Batch, 1, Vocab)
    print("-" * 30)

    # 2. GRU 모델 테스트
    print("--- GRU 모델 테스트 ---")
    gru_model = BaselineGRU(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    gru_output = gru_model(dummy_input_ids)
    print("Input shape:", dummy_input_ids.shape)
    print("Output logits shape:", gru_output['logits'].shape) # (Batch, 1, Vocab)
    print("-" * 30)

    # 3. Transformer (T5) 모델 테스트
    print("--- Transformer (T5) 모델 테스트 ---")
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        transformer_model = BaselineTransformer(model_name="t5-small", tokenizer=tokenizer)
        
        dummy_text = ["이것은 테스트 문장입니다.", "이것도 다른 문장입니다."]
        inputs = tokenizer(dummy_text, return_tensors="pt", padding=True)
        labels = tokenizer(dummy_text, return_tensors="pt", padding=True).input_ids

        transformer_output = transformer_model(**inputs, labels=labels)
        print("Input shapes:", {k: v.shape for k, v in inputs.items()})
        print("Output logits shape:", transformer_output.logits.shape)
        print("Calculated loss:", transformer_output.loss.item())
        
        # 생성 테스트
        print("\nGenerate test:")
        generated_text = transformer_model.generate(inputs['input_ids'], inputs['attention_mask'])
        print(f"Generated text: {generated_text}")

    except Exception as e:
        print(f"Transformer 모델 테스트 중 오류 발생: {e}")
        print("Hugging Face 라이브러리가 올바르게 설치되었는지, 인터넷 연결이 되어있는지 확인하세요.")
    print("-" * 30)

