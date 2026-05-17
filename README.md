# audio-mastering

![Release](https://img.shields.io/github/v/release/kyejin1991/AUDIO-MASTERING?include_prereleases)
![License](https://img.shields.io/github/license/kyejin1991/AUDIO-MASTERING)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-community%20release-green)

## English

audio-mastering Community Edition is the open-source audio analysis and manual workflow layer of AI Mastering Lab.

It can create projects, inspect audio files, and run loudness, spectrum, dynamics, and stereo analysis.

### Quick Start Guide

For a more detailed guide, see [docs/QUICK_START.md](docs/QUICK_START.md).

For a Korean release summary, see [docs/ANNOUNCEMENT.md](docs/ANNOUNCEMENT.md).

### Features

- Project creation from local audio files
- Loudness analysis
- Spectrum analysis
- Dynamics analysis
- Stereo analysis
- Streamlit-based Community UI
- Basic DSP module structure

### Community Scope

audio-mastering Community Edition includes the open-source analysis and manual workflow layer of AI Mastering Lab.

The proprietary Pro engine is not included in this repository. This includes the AI Master Assistant, AI mastering decision logic, Pro rendering orchestration, genre-specific profiles, and commercial presets.

Community is not an unfinished version. It is the open-source layer.

Pro is a separate proprietary product layer.

### What You Can Do

- Load audio files
- Create analysis projects
- Inspect loudness, spectrum, dynamics, and stereo information
- Use the Community UI as a base for manual workflow foundations
- Explore the basic DSP module structure

### What Is Not Included

This public repository does not include the Pro mastering engine.

The following components are intentionally excluded:

- AI Master Assistant
- AI mastering decision engine
- Pro rendering orchestration
- Genre-specific profiles
- Commercial presets
- Pro QC workflow

### Quick Start

For Windows:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For macOS / Linux:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Screenshots

#### Overview

![audio-mastering overview](assets/images/hero.png)

#### Audio Upload

![Upload screen](assets/images/upload_screen.png)

#### Audio Analysis

![Analysis screen](assets/images/analysis_screen.png)

#### Manual Mastering Rack

![Module rack](assets/images/module_rack.png)

#### Render Result

![Render result](assets/images/render_result.png)

### Quick Start Video

The 15-second quick start video is available in the GitHub Release assets:

[View release assets](https://github.com/kyejin1991/AUDIO-MASTERING/releases/tag/v0.1.0-community)

### Third-Party Components

This project uses third-party open-source components including Demucs and pyloudnorm.

See:

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- [NOTICE](NOTICE)

### Community / Pro Boundary

| Edition | Status | Included |
|---|---|---|
| Community | Open Source | Analysis, manual workflow layer, basic DSP structure |
| Pro | Proprietary | AI mastering engine, rendering orchestration, genre profiles |

The Community Edition intentionally includes only the open-source audio analysis and manual workflow layer.

The AI mastering decision engine, Pro rendering orchestration, genre-specific profiles, and commercial presets are part of the separate Pro engine and are not included in this public repository.

### Support

If you find this project useful, please consider giving it a star on GitHub.

It helps the project reach more developers and audio creators.

### License

See [LICENSE](LICENSE).

### Author

Developed by [kyejin1991](https://github.com/kyejin1991)  
© 2026 Arcapps

---

## 한국어

audio-mastering Community Edition은 AI Mastering Lab의 공개 가능한 오픈소스 오디오 분석 및 수동 워크플로우 레이어입니다.

프로젝트를 생성하고, 오디오 파일을 점검하고, loudness, spectrum, dynamics, stereo 분석을 실행할 수 있습니다.

### 빠른 시작 가이드

더 자세한 실행 방법은 [docs/QUICK_START.md](docs/QUICK_START.md)를 확인하세요.

공개 요약 문서는 [docs/ANNOUNCEMENT.md](docs/ANNOUNCEMENT.md)를 참고하세요.

### 주요 기능

- 로컬 오디오 파일 기반 프로젝트 생성
- 라우드니스 분석
- 스펙트럼 분석
- 다이내믹스 분석
- 스테레오 분석
- Streamlit 기반 Community UI
- 기본 DSP 모듈 구조

### Community 범위

audio-mastering Community Edition은 AI Mastering Lab에서 공개 가능한 분석 및 수동 워크플로우 레이어만 포함합니다.

비공개 Pro 엔진은 이 저장소에 포함되어 있지 않습니다. 여기에는 AI Master Assistant, AI 마스터링 의사결정 로직, Pro 렌더링 오케스트레이션, 장르별 프로필, 상업용 프리셋이 포함됩니다.

Community는 미완성 버전이 아니라 공개 가능한 오픈소스 레이어입니다.

Pro는 별도의 비공개 제품 레이어입니다.

### 할 수 있는 것

- 오디오 파일 불러오기
- 분석 프로젝트 생성
- loudness, spectrum, dynamics, stereo 정보 확인
- Community UI를 수동 워크플로우 기반으로 활용
- 기본 DSP 모듈 구조 탐색

### 포함되지 않는 것

이 공개 저장소에는 Pro 마스터링 엔진이 포함되어 있지 않습니다.

다음 구성요소는 의도적으로 제외되었습니다:

- AI Master Assistant
- AI 마스터링 의사결정 엔진
- Pro 렌더링 오케스트레이션
- 장르별 프로필
- 상업용 프리셋
- Pro QC 워크플로우

### 빠른 시작

Windows:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 스크린샷

#### 개요

![audio-mastering overview](assets/images/hero.png)

#### 오디오 업로드

![Upload screen](assets/images/upload_screen.png)

#### 오디오 분석

![Analysis screen](assets/images/analysis_screen.png)

#### 수동 마스터링 랙

![Module rack](assets/images/module_rack.png)

#### 렌더 결과

![Render result](assets/images/render_result.png)

### 빠른 시작 영상

15초 Quick Start 영상은 GitHub Release assets에서 확인할 수 있습니다:

[View release assets](https://github.com/kyejin1991/AUDIO-MASTERING/releases/tag/v0.1.0-community)

### 서드파티 구성요소

이 프로젝트는 Demucs와 pyloudnorm을 포함한 서드파티 오픈소스 구성요소를 사용합니다.

참고:

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- [NOTICE](NOTICE)

### Community / Pro 경계

| 에디션 | 상태 | 포함 범위 |
|---|---|---|
| Community | 오픈소스 | 분석, 수동 워크플로우 레이어, 기본 DSP 구조 |
| Pro | 비공개 | AI 마스터링 엔진, 렌더링 오케스트레이션, 장르별 프로필 |

Community Edition은 의도적으로 오픈소스 오디오 분석 및 수동 워크플로우 레이어만 포함합니다.

AI 마스터링 의사결정 엔진, Pro 렌더링 오케스트레이션, 장르별 프로필, 상업용 프리셋은 별도의 Pro 엔진에 속하며 이 공개 저장소에는 포함되지 않습니다.

### 응원하기

프로젝트가 도움이 되었다면 GitHub Star를 눌러주세요.

더 많은 개발자와 오디오 크리에이터에게 프로젝트가 닿는 데 큰 도움이 됩니다.

### 라이선스

See [LICENSE](LICENSE).

### 작성자

Developed by [kyejin1991](https://github.com/kyejin1991)  
© 2026 Arcapps
