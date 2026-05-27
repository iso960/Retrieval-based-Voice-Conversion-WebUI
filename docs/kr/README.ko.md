<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITS 기반의 간단하고 사용하기 쉬운 음성 변환 프레임워크.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**업데이트 로그**](./Changelog_KO.md) | [**자주 묻는 질문**](./faq_ko.md) | [**AutoDL·5원으로 AI 가수 훈련**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**대조 실험 기록**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**온라인 데모**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> [데모 영상](https://www.bilibili.com/video/BV1pm4y1z7Gm/)을 확인해 보세요!

> RVC를 활용한 실시간 음성변환: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 기본 모델은 50시간 가량의 고퀄리티 오픈 소스 VCTK 데이터셋을 사용하였으므로, 저작권상의 염려가 없으니 안심하고 사용하시기 바랍니다.

> 더 큰 매개변수, 더 큰 데이터, 더 나은 효과, 기본적으로 동일한 추론 속도, 더 적은 양의 훈련 데이터가 필요한 RVCv3의 기본 모델을 기대해 주십시오.

<table>
   <tr>
		<td align="center">훈련 및 추론 인터페이스</td>
		<td align="center">실시간 음성 변환 인터페이스</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-web.bat</td>
		<td align="center">go-realtime-gui.bat</td>
	</tr>
  <tr>
    <td align="center">원하는 작업을 자유롭게 선택할 수 있습니다.</td>
		<td align="center">우리는 이미 끝에서 끝까지 170ms의 지연을 실현했습니다. ASIO 입력 및 출력 장치를 사용하면 끝에서 끝까지 90ms의 지연을 달성할 수 있지만, 이는 하드웨어 드라이버 지원에 매우 의존적입니다.</td>
	</tr>
</table>

## 소개

본 Repo는 다음과 같은 특징을 가지고 있습니다:

- top1 검색을 이용하여 입력 음색 특징을 훈련 세트 음색 특징으로 대체하여 음색의 누출을 방지
- 상대적으로 낮은 성능의 GPU에서도 빠른 훈련 가능
- 적은 양의 데이터로 훈련해도 좋은 결과를 얻을 수 있음 (최소 10분 이상의 저잡음 음성 데이터를 사용하는 것을 권장)
- 모델 융합을 통한 음색의 변조 가능 (ckpt 처리 탭->ckpt 병합 선택)
- 사용하기 쉬운 WebUI (웹 인터페이스)
- UVR5 모델을 이용하여 목소리와 배경음악의 빠른 분리;
- 최첨단 [음성 피치 추출 알고리즘 InterSpeech2023-RMVPE](#参考项目)을 사용하여 무성음 문제를 해결합니다. 효과는 최고(압도적)이며 crepe_full보다 더 빠르고 리소스 사용이 적음
- A카드와 I카드 가속을 지원

해당 프로젝트의 [데모 비디오](https://www.bilibili.com/video/BV1pm4y1z7Gm/)를 확인해보세요!

## 환경 설정

### 사전 요구사항

- Python 3.11 이상
- Git
- Nvidia GPU 권장 (CUDA 11.8 이상), CPU로도 실행 가능하나 속도 저하 있음
- Visual C++ Build Tools (Windows 필수) — https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 1. 저장소 클론

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
```

### 2. 의존성 설치

**Nvidia GPU (CUDA 11.8) 사용 시**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**CPU만 사용하는 경우**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. 사전 훈련 모델 다운로드

[Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)에서 아래 파일을 다운로드하여 지정 경로에 배치하세요.

| 파일 | 배치 경로 |
|------|-----------|
| `hubert_base.pt` | `assets/hubert/` |
| `pretrained/` (폴더 전체) | `assets/pretrained/` |
| `pretrained_v2/` (폴더 전체) | `assets/pretrained_v2/` |
| `uvr5_weights/` (폴더 전체) | `assets/uvr5_weights/` |
| `rmvpe.pt` | 루트 디렉토리 |

**추론 테스트를 하려면 음색 모델이 별도로 필요합니다**

- 음색 모델(.pth)은 `assets/weights/`에 배치
- 직접 훈련(훈련 탭)하거나 커뮤니티에서 공개된 모델을 받아서 넣어야 함
- `assets/pretrained_v2/`의 모델은 훈련용 베이스 모델로 음색 모델과 다름

**Windows 사용자 — ffmpeg 배치**

- [ffmpeg.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)
- [ffprobe.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

두 파일을 저장소 루트 디렉토리에 배치하세요.

### 4. 실행

**WebUI**

```bat
go-web.bat
```

**실시간 음성 변환 GUI**

실행 전 아래 패키지를 추가로 설치하세요.

```bash
pip install FreeSimpleGUI "sounddevice<0.5.0"
```

```bat
go-realtime-gui.bat
```

## 참조 프로젝트

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  - 사전 훈련된 모델은 [yxlllc](https://github.com/yxlllc/RMVPE)와 [RVC-Boss](https://github.com/RVC-Boss)에 의해 훈련되고 테스트되었습니다.

## 모든 기여자들의 노력에 감사드립니다

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
