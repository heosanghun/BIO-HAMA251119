# main_train.py

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any
import sys
import os

# 프로젝트 모듈 import를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 구현한 모듈들 import
from data.dataset import get_dataloader, DummyReasoningDataset, CognitiveState
from models.baselines import BaselineLSTM, BaselineGRU, BaselineTransformer
from models.bio_hama.architecture import BioHAMA
from training.optimizer import BioAGRPO
from transformers import T5Tokenizer, AutoTokenizer

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_model(model_name: str, config: Dict, device: torch.device):
    """모델 이름에 따라 모델 인스턴스를 생성하고 반환합니다."""
    model_config = config['model_params']
    
    if model_name == 'lstm':
        return BaselineLSTM(**model_config).to(device)
    elif model_name == 'gru':
        return BaselineGRU(**model_config).to(device)
    elif model_name == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        model = BaselineTransformer(model_name=model_config['model_name'], tokenizer=tokenizer).to(device)
        return model, tokenizer
    elif model_name == 'bio_hama':
        return BioHAMA(model_config).to(device)
    else:
        raise ValueError(f"알 수 없는 모델 이름입니다: {model_name}")

def train_loop(
    epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    args: argparse.Namespace,
    tokenizer: Any = None,
    bio_a_grpo: BioAGRPO = None
):
    """한 에포크 동안의 학습 루프를 실행합니다."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    for batch in progress_bar:
        input_texts = batch['input_text']
        target_texts = batch['target_text']
        
        optimizer.zero_grad()
        
        # --- 모델별 입력 및 출력 처리 ---
        if model_name in ['lstm', 'gru']:
            # TODO: 실제로는 텍스트를 토큰 ID로 변환해야 함
            input_ids = torch.randint(0, model.embedding.num_embeddings, (len(input_texts), 20)).to(device)
            target_ids = torch.randint(0, model.embedding.num_embeddings, (len(input_texts), 1)).to(device)
            
            outputs = model(input_ids)
            logits = outputs['logits']
            loss = criterion(logits.squeeze(1), target_ids.squeeze(1))

        elif model_name == 'transformer':
            inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

        elif model_name == 'bio_hama':
            # Bio-HAMA는 상태를 유지하며 순환해야 함
            # 단순화를 위해 매 배치마다 초기 상태로 시작
            batch_size = len(input_texts)
            embed_dim = model.config['embed_dim']
            
            # TODO: 텍스트를 토큰 ID로 변환
            input_ids = torch.randint(0, model.config['vocab_size'], (batch_size, 10)).to(device)
            target_ids = torch.randint(0, model.config['vocab_size'], (batch_size, 1)).to(device)

            prev_state = CognitiveState(
                working_memory=torch.zeros(batch_size, embed_dim).to(device),
                affective_context=torch.zeros(batch_size, embed_dim).to(device)
            )
            
            # --- Bio-A-GRPO 로직 적용 ---
            if bio_a_grpo is not None:
                # 1. 동적 파라미터 계산 (이 스텝의 정책 결정 전에 상태 기반으로 계산)
                dynamic_params = bio_a_grpo.calculate_dynamic_params(prev_state)
            
            # 2. 모델 순방향 연산
            final_output, next_state, policy_logits, _ = model(input_ids, prev_state)

            # 3. 강화학습 손실 계산 (단순화된 버전)
            # 여기서는 지도학습 손실로 대체하여 프레임워크를 테스트
            # TODO: Actor-Critic 손실 구현 필요
            # final_output은 모듈 출력의 합이므로, 이를 최종 로짓으로 변환하는 레이어 필요
            output_projection = nn.Linear(embed_dim, model.config['vocab_size']).to(device)
            final_logits = output_projection(final_output)
            loss = criterion(final_logits, target_ids.squeeze(1))
            
            # Actor-Critic 업데이트 시 dynamic_params 사용
            # ex) actor_loss.backward() -> actor_optimizer.step(lr=dynamic_params['dynamic_lr'])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    return avg_loss

def eval_loop(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    tokenizer: Any = None
):
    """검증 데이터셋에 대한 평가 루프."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    print("\n평가 중...")
    with torch.no_grad():
        for batch in loader:
            input_texts = batch['input_text']
            target_texts = batch['target_text']
            
            if model_name in ['lstm', 'gru']:
                input_ids = torch.randint(0, model.embedding.num_embeddings, (len(input_texts), 20)).to(device)
                target_ids = torch.randint(0, model.embedding.num_embeddings, (len(input_texts), 1)).to(device)
                
                outputs = model(input_ids)
                logits = outputs['logits']
                loss = criterion(logits.squeeze(1), target_ids.squeeze(1))
                total_loss += loss.item()
                
                # 정확도 계산
                predictions = logits.squeeze(1).argmax(dim=-1)
                correct = (predictions == target_ids.squeeze(1)).sum().item()
                total_correct += correct
                total_samples += len(input_texts)
                
            elif model_name == 'transformer':
                inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
                
                outputs = model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                total_samples += len(input_texts)
                
            elif model_name == 'bio_hama':
                batch_size = len(input_texts)
                embed_dim = model.config['embed_dim']
                input_ids = torch.randint(0, model.config['vocab_size'], (batch_size, 10)).to(device)
                
                prev_state = CognitiveState(
                    working_memory=torch.zeros(batch_size, embed_dim).to(device),
                    affective_context=torch.zeros(batch_size, embed_dim).to(device)
                )
                
                final_output, _, _, _ = model(input_ids, prev_state)
                total_samples += len(input_texts)
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"평가 완료 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    return avg_loss, accuracy


def main(args):
    """메인 실행 함수."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"모델: {args.model_name}")
    print(f"설정 파일: {args.config_path}\n")
    
    config = load_config(args.config_path)
    train_params = config['train_params']

    # --- 데이터 로드 ---
    # TODO: 실제 데이터셋으로 교체
    train_dataset = DummyReasoningDataset(num_samples=100, task_type='logic') 
    eval_dataset = DummyReasoningDataset(num_samples=20, task_type='logic')
    train_loader = get_dataloader(train_dataset, batch_size=train_params['batch_size'])
    eval_loader = get_dataloader(eval_dataset, batch_size=train_params['batch_size'], shuffle=False)
    
    # --- 모델 및 토크나이저 초기화 ---
    tokenizer = None
    if args.model_name == 'transformer':
        model, tokenizer = get_model(args.model_name, config, device)
    else:
        model = get_model(args.model_name, config, device)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # --- 옵티마이저 및 손실 함수 ---
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    bio_a_grpo_instance = None
    if args.model_name == 'bio_hama' and 'bio_a_grpo_params' in config:
        bio_a_grpo_instance = BioAGRPO(**config['bio_a_grpo_params'])
        print("Bio-A-GRPO 학습 알고리즘 활성화\n")

    # --- 학습 및 평가 실행 ---
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_loop(
            epoch, model, train_loader, optimizer, criterion, device,
            args.model_name, args, tokenizer, bio_a_grpo_instance
        )
        
        eval_loss, eval_acc = eval_loop(model, eval_loader, criterion, device, args.model_name, tokenizer)
        
        # 최고 성능 모델 저장
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_path = f"best_{args.model_name}_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"✓ 최고 성능 모델 저장: {save_path}\n")
    
    print("=" * 50)
    print("학습 완료!")
    print(f"최종 평가 손실: {eval_loss:.4f}")
    print(f"최종 정확도: {eval_acc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bio-HAMA 및 베이스라인 모델 학습 스크립트")
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['lstm', 'gru', 'transformer', 'bio_hama'],
                        help="학습할 모델의 이름")
    parser.add_argument('--config_path', type=str, required=True,
                        help="모델 및 학습 설정을 담은 YAML 파일 경로")
    parser.add_argument('--epochs', type=int, default=10, help="총 학습 에포크 수")
    
    args = parser.parse_args()
    
    # --- 실행 예시 ---
    # python main_train.py --model_name bio_hama --config_path configs/bio_hama.yaml --epochs 5
    
    main(args)

