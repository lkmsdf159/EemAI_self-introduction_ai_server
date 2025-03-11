# AI 자기소개서 분석 프로젝트 🤖

실제 개발 환경 구축 과정과 팁을 공유합니다. WSL2 환경에서 Hugging Face 모델을 손쉽게 사용하는 방법을 중심으로 설명합니다.

> 본 프로젝트의 개발 환경 설정 가이드는 테디노트님의 작업을 기반으로 작성되었습니다. 

## 📋 프로젝트 개요

### 개발 의도
이 프로젝트는 취업 준비생들이 더 효과적인 자기소개서를 작성할 수 있도록 돕기 위해 개발되었습니다. 기업별 합격 자기소개서 데이터를 분석하여 맞춤형 피드백과 개선 방향을 제공함으로써, 취업 준비생들의 자기소개서 작성 과정을 지원합니다.

### 주요 기능
합격 자기소개서 제공: 기업별/직무별 합격 자기소개서 데이터를 분석하여 주요 키워드와 패턴을 파악
맞춤형 피드백 제공: 사용자가 작성한 자기소개서를 분석 및 합격 자기소개서의 참고할점,개선점과 제공
AI 기반 자기소개서 개선 제안: Gemma 2B-IT 모델을 활용하여 더 효과적인 표현과 구성을 제안

### 기술적 접근
- WSL2와 CUDA를 활용한 고성능 추론 환경 구축
- LangChain과 LangServe를 활용한 확장 가능한 API 서비스 구현
- Hugging Face의 Gemma 2B-IT 모델을 경량화된 GGUF 형식으로 최적화
- RAG(Retrieval-Augmented Generation) 아키텍처를 활용한 맥락 기반 분석 구현
- ngrok을 활용한 안전한 API 엔드포인트 제공

> 해당 UI는 https://github.com/eFOROW/EmpAI 이곳에서 확인 가능합니다.

## 개발 환경 세팅하기 

### 🔧 필수 환경 (테스트 완료 버전)
- Python 3.10
- PyTorch 2.5.1
- CUDA 11.8
- cuDNN 9.1.0

### CUDA & PyTorch 설정
> 💡 CUDA 설정이 까다로울 수 있는데, [이 블로그](https://limitsinx.tistory.com/317)를 차근차근 따라하시면 됩니다!

CUDA가 제대로 설치됐는지 확인하려면:
```bash
nvcc -V
```

아래와 같은 출력이 나와야 정상입니다:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## WSL2로 개발하기

Hugging Face 모델을 더 편하게 사용하기 위해 WSL2 환경을 추천드립니다. Windows에서 직접 설정하는 것보다 훨씬 수월합니다.

### Gemma 2B-IT 모델 설치하기

#### GGUF 파일 형식에 대해
GGUF는 딥러닝 모델을 효율적으로 저장하고 배포하기 위한 새로운 파일 형식입니다. 기존 모델 저장 방식들의 호환성 문제를 해결했죠.

#### 모델 다운로드 방법
Hugging Face에서 편하게 다운받는 방법:
```bash
huggingface-cli download \
  [허깅페이스 gguf저장소 이름] \
  [다운받을 gguf 파일명] \
  --local-dir [다운로드_경로] \
  --local-dir-use-symlinks False
```

> 💡 **Tip**: `symlinks=False` 설정은 파일을 직접 복사하는 방식으로 동작해서 시스템 호환성 문제를 피할 수 있습니다.

#### Ollama 모델 등록
```bash
ollama create [등록할_이름] -f [Modelfile_경로]
```

## API 서버 구성하기

### LangServe로 API 만들기
LangChain으로 만든 LLM Chain을 API로 손쉽게 배포할 수 있습니다. Flask나 FastAPI로 직접 구성할 필요 없이, LangServe가 제공하는 도구로 간편하게 배포 가능합니다.

### ngrok으로 외부 접속 허용하기
로컬에서 돌아가는 서버를 외부에서 접속 가능하게 만들어주는 도구입니다.

#### 설치 방법
1. [ngrok 설치 페이지](https://dashboard.ngrok.com/get-started/setup/linux)에서 본인 운영체제에 맞는 버전을 선택
2. 안내대로 설치하면 끝!

#### 사용 방법
로컬 서버 포워딩하기:
```bash
ngrok http http://localhost:8080
```

실행하면 아래와 같은 주소가 생성됩니다:
```
https://cicada-musical-donkey.ngrok-free.app
```
이 주소로 API 통신이 가능합니다.

> ⚠️ **주의사항**: 
> - Ctrl+C로 종료하고 다시 실행하면 도메인이 바뀝니다
> - 무료 계정은 고정 도메인 하나를 제공하니, 처음 생성된 static domain을 사용하세요

## 실제 실행하기

1. ngrok 실행
2. 서버 코드 디렉토리로 이동
3. `python server.py` 실행

## 자기소개서 데이터 준비
CSV 형식으로 데이터를 준비해야 합니다. 데이터 수집 후 전처리 작업을 거쳐 지정된 양식에 맞춰주세요.

## 🤝 도움이 필요하다면
- CUDA나 PyTorch 설치에서 막힌다면 -> [블로그 가이드](https://limitsinx.tistory.com/317) 참고
- WSL2 설치가 안 된다면 -> Windows 버전 확인
- ngrok 연결이 안 된다면 -> 포트 번호 확인
- 해당 내용을 세세하게 영상 강의로 보고싶다 -> https://python.langchain.com/docs/langserve/

## 📚 참고 자료
- [LangChain 공식 문서 - LangServe](https://python.langchain.com/docs/langserve/)
- [테디노트님의 LangServe & Ollama 가이드](https://github.com/teddylee777/langserve_ollama)
