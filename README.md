# NVIDIA GPU 학습 시간 · 비용 계산기

데이터 양과 모델 속성을 입력하면 NVIDIA GPU 모델별 예상 학습 시간과 클라우드 비용을 계산하는 정적 웹 앱입니다.

## 기능

- **학습 유형 6종**: LLM 사전학습 / 전체 파인튜닝 / LoRA·QLoRA, 이미지 분류(CNN·ViT), 확산 모델(SD 계열), 기상·지구 시뮬레이션(NVIDIA Earth-2: FourCastNet 3, CorrDiff)
- **유명 오픈소스 모델 프리셋**: NVIDIA Nemotron, Llama 3.1, DeepSeek-V3/R1, Qwen, Mixtral, Gemma, Phi-4, gpt-oss 등 (MoE는 활성 파라미터 기준 연산)
- **NVIDIA GPU 28종 비교**: GB300/GB200/B200/B100, H200/H100/H20, L-시리즈, A100 등 Ampere·Volta·Turing, DGX Spark, RTX PRO/워크스테이션, GeForce RTX 30/40/50
- 데이터 양을 토큰 수와 **텍스트 용량(GB/TB)** 으로 상호 환산 표시
- BF16/FP8 정밀도, MFU(실효 효율), GPU 개수, 원/달러 환율 조정
- GPU별 시간당 단가 직접 수정 가능, 메모리 부족 경고

## 계산 방식

- LLM: `총 FLOPs ≈ 6 × 파라미터 × 토큰` (LoRA ≈ 0.67배, MoE는 활성 파라미터 기준)
- 비전/확산/기상: `샘플당 순전파 FLOPs × 3 × 샘플 수 × 에포크`
- 학습 시간 = `총 FLOPs ÷ (GPU 수 × 이론 TFLOPS × MFU)`
- Earth-2 프리셋은 공개된 학습 비용(FCN3 ≈ 9만 GPU·시간, CorrDiff ≈ 5,000 A100 GPU·시간)에 맞춰 보정

> ⚠️ 모든 수치는 추정치입니다. 실제 학습 시간은 데이터 파이프라인, 통신 병목, 시퀀스 길이 등에 따라 달라질 수 있으며, 시간당 단가는 2025~2026년 온디맨드 시세 기준 기본값입니다.

## 실행

빌드 도구 없는 단일 `index.html` 파일입니다. 브라우저에서 바로 열거나 아무 정적 서버로 서빙하면 됩니다.

```bash
python -m http.server 4173
```
