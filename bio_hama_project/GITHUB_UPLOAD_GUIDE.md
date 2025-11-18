# 📤 GitHub 업로드 가이드

Bio-HAMA 프로젝트를 GitHub에 업로드하는 단계별 가이드입니다.

---

## 🔑 사전 준비

1. **GitHub 계정 확인**: https://github.com/heosanghun
2. **Git 설치 확인**:
   ```bash
   git --version
   ```
3. **프로젝트 디렉토리 위치**: `D:\AI\BIOHAMA251118\bio_hama_project`

---

## 📝 업로드 단계

### 1단계: Git 초기화

프로젝트 디렉토리에서 다음 명령어를 실행하세요:

```bash
cd D:\AI\BIOHAMA251118\bio_hama_project
git init
```

### 2단계: 불필요한 파일 정리 (선택)

디버그 파일들을 삭제하거나 `.gitignore`에 추가:

```bash
# 디버그 파일 삭제 (선택)
del debug_module.py
del debug_biohama.py

# 또는 .gitignore에 추가
echo debug_*.py >> .gitignore
```

### 3단계: 모든 파일 스테이징

```bash
git add .
```

### 4단계: 첫 번째 커밋

```bash
git commit -m "Initial commit: Bio-HAMA project complete implementation

- Implemented 12 cognitive modules with brain-inspired architecture
- Hierarchical meta-router with 3-layer control (Strategy, Tactics, Response)
- Bio-A-GRPO learning algorithm with dynamic parameter adjustment
- Complete baseline models (LSTM, GRU, Transformer)
- Ablation study automation
- Visualization and analysis tools
- All tests passing (6/6)
"
```

### 5단계: GitHub 저장소 연결

**옵션 A: 새 저장소 생성**

1. GitHub 웹사이트에서 새 저장소 생성:
   - 저장소 이름: `Bio-HAMA` 또는 `BIO-HAMA_MAIN`
   - 설명: "Bio-inspired Hyper-Adaptive Multimodal Architecture"
   - Public/Private 선택
   - README, .gitignore, LICENSE는 **체크하지 않기** (이미 있음)

2. 저장소 URL 복사 (예: `https://github.com/heosanghun/Bio-HAMA.git`)

3. 로컬 저장소와 연결:
   ```bash
   git remote add origin https://github.com/heosanghun/Bio-HAMA.git
   git branch -M main
   ```

**옵션 B: 기존 저장소 업데이트**

기존 `BIO-HAMA_MAIN` 저장소를 업데이트하려면:

```bash
git remote add origin https://github.com/heosanghun/BIO-HAMA_MAIN.git
git branch -M main
```

### 6단계: GitHub에 푸시

```bash
git push -u origin main
```

**인증이 필요한 경우**:
- Username: `heosanghun`
- Password: Personal Access Token (PAT) 사용
  - Settings → Developer settings → Personal access tokens → Generate new token

---

## 🎯 간단한 업로드 스크립트

다음 명령어들을 한 번에 실행:

```bash
# PowerShell
cd D:\AI\BIOHAMA251118\bio_hama_project
git init
git add .
git commit -m "Initial commit: Bio-HAMA complete implementation"
git remote add origin https://github.com/heosanghun/Bio-HAMA.git
git branch -M main
git push -u origin main
```

---

## 🔄 업데이트 방법

이후 변경사항을 업로드하려면:

```bash
git add .
git commit -m "Update: [변경 내용 설명]"
git push
```

---

## 📋 업로드 후 체크리스트

- [ ] README.md가 제대로 표시되는지 확인
- [ ] 파일 구조가 올바르게 업로드되었는지 확인
- [ ] LICENSE 파일이 보이는지 확인
- [ ] .gitignore가 적용되어 불필요한 파일이 제외되었는지 확인
- [ ] 저장소 설명 추가
- [ ] Topics/Tags 추가: `ai`, `deep-learning`, `pytorch`, `cognitive-architecture`, `bio-inspired`

---

## 🌟 저장소 꾸미기 (선택)

### 1. 저장소 설명 추가
```
Bio-inspired Hyper-Adaptive Multimodal Architecture for Human-like Intelligence
```

### 2. Topics 추가
```
ai, deep-learning, pytorch, neural-networks, cognitive-architecture, 
bio-inspired, meta-learning, reinforcement-learning, brain-inspired, 
modular-ai, adaptive-systems
```

### 3. About 섹션 설정
- Website: (프로젝트 웹사이트가 있다면)
- Description: (위 설명 사용)

### 4. README.md 배지 추가 (선택)
저장소 상단에 상태 배지를 추가할 수 있습니다:
```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-6%2F6%20passing-brightgreen.svg)
```

---

## 🚨 문제 해결

### 오류: "remote origin already exists"
```bash
git remote remove origin
git remote add origin [새 URL]
```

### 오류: "failed to push some refs"
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### 대용량 파일 문제
`.gitignore`에서 제외되었는지 확인하거나, Git LFS 사용:
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
```

---

## ✅ 완료!

축하합니다! Bio-HAMA 프로젝트가 GitHub에 성공적으로 업로드되었습니다.

저장소 주소: `https://github.com/heosanghun/Bio-HAMA`

이제 다른 연구자들과 프로젝트를 공유하고, Star를 받고, 기여를 받을 수 있습니다!

---

**참고**: 이 가이드는 Windows PowerShell 기준으로 작성되었습니다.  
다른 운영체제에서는 명령어가 약간 다를 수 있습니다.

