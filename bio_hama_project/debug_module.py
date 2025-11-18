import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.bio_hama.modules import SocialCognitionModule
import torch

print("=" * 50)
print("모듈 디버깅")
print("=" * 50)

# 모듈 생성
module = SocialCognitionModule(input_dim=128, output_dim=128)

print(f"\n모듈 구조:")
print(module)

print(f"\n네트워크 레이어:")
for name, layer in module.network.named_children():
    print(f"  {name}: {layer}")
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        print(f"    입력: {layer.in_features}, 출력: {layer.out_features}")

# 테스트 입력
test_input = torch.randn(4, 128)
print(f"\n테스트 입력 shape: {test_input.shape}")

try:
    output = module(test_input)
    print(f"✓ Forward 성공! 출력 shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward 실패: {e}")

