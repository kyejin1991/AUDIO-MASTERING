# audio-mastering

![Release](https://img.shields.io/github/v/release/kyejin1991/AUDIO-MASTERING?include_prereleases)
![License](https://img.shields.io/github/license/kyejin1991/AUDIO-MASTERING)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-community%20release-green)

audio-mastering Community Edition is the open-source audio analysis and manual workflow layer of AI Mastering Lab.

audio-mastering Community Edition은 AI Mastering Lab의 공개 가능한 오픈소스 오디오 분석 및 수동 워크플로우 레이어입니다.

It can create projects, inspect audio files, and run loudness, spectrum, dynamics, and stereo analysis.

프로젝트를 생성하고, 오디오 파일을 점검하고, loudness, spectrum, dynamics, stereo 분석을 실행할 수 있습니다.

## Quick Start Guide / 빠른 시작 가이드

For a more detailed guide, see [docs/QUICK_START.md](docs/QUICK_START.md).

더 자세한 실행 방법은 [docs/QUICK_START.md](docs/QUICK_START.md)를 확인하세요.

For a Korean release summary, see [docs/ANNOUNCEMENT.md](docs/ANNOUNCEMENT.md).

한국어 공개 요약은 [docs/ANNOUNCEMENT.md](docs/ANNOUNCEMENT.md)를 참고하세요.

## Features / 주요 기능

- Project creation from local audio files
- Loudness analysis
- Spectrum analysis
- Dynamics analysis
- Stereo analysis
- Streamlit-based Community UI
- Basic DSP module structure

- 로컬 오디오 파일 기반 프로젝트 생성
- 라우드니스 분석
- 스펙트럼 분석
- 다이내믹스 분석
- 스테레오 분석
- Streamlit 기반 Community UI
- 기본 DSP 모듈 구조

## Community Scope / Community 범위

audio-mastering Community Edition includes the open-source analysis and manual workflow layer of AI Mastering Lab.

audio-mastering Community Edition은 AI Mastering Lab에서 공개 가능한 분석 및 수동 워크플로우 레이어만 포함합니다.

It can create projects, inspect audio files, and run loudness, spectrum, dynamics, and stereo analysis.

프로젝트 생성, 오디오 점검, loudness, spectrum, dynamics, stereo 분석이 가능합니다.

The proprietary Pro engine is not included in this repository. This includes the AI Master Assistant, AI mastering decision logic, Pro rendering orchestration, genre-specific profiles, and commercial presets.

비공개 Pro 엔진은 이 저장소에 포함되어 있지 않습니다. 여기에는 AI Master Assistant, AI 마스터링 의사결정 로직, Pro 렌더링 오케스트레이션, 장르별 프로필, 상업용 프리셋이 포함됩니다.

Community is not an unfinished version. It is the open-source layer.

Community는 미완성 버전이 아니라 공개 가능한 오픈소스 레이어입니다.

Pro is a separate proprietary product layer.

Pro는 별도의 비공개 제품 레이어입니다.

## What You Can Do / 할 수 있는 것

- Load audio files
- Create analysis projects
- Inspect loudness, spectrum, dynamics, and stereo information
- Use the Community UI as a base for manual workflow foundations
- Explore the basic DSP module structure

- 오디오 파일 불러오기
- 분석 프로젝트 생성
- loudness, spectrum, dynamics, stereo 정보 확인
- Community UI를 수동 워크플로우 기반으로 활용
- 기본 DSP 모듈 구조 탐색

## What Is Not Included / 포함되지 않는 것

This public repository does not include the Pro mastering engine.

이 공개 저장소에는 Pro 마스터링 엔진이 포함되어 있지 않습니다.

The following components are intentionally excluded:

다음 구성요소는 의도적으로 제외되었습니다:

- AI Master Assistant
- AI mastering decision engine
- Pro rendering orchestration
- Genre-specific profiles
- Commercial presets
- Pro QC workflow

- AI Master Assistant
- AI 마스터링 의사결정 엔진
- Pro 렌더링 오케스트레이션
- 장르별 프로필
- 상업용 프리셋
- Pro QC 워크플로우

## Quick Start / 빠른 시작

For Windows:

Windows:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For macOS / Linux:

macOS / Linux:

```bash
git clone https://github.com/kyejin1991/AUDIO-MASTERING.git
cd AUDIO-MASTERING
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots / 스크린샷

### Overview / 개요

![audio-mastering overview](assets/images/hero.png)

### Audio Upload / 오디오 업로드

![Upload screen](assets/images/upload_screen.png)

### Audio Analysis / 오디오 분석

![Analysis screen](assets/images/analysis_screen.png)

### Manual Mastering Rack / 수동 마스터링 랙

![Module rack](assets/images/module_rack.png)

### Render Result / 렌더 결과

![Render result](assets/images/render_result.png)

## Quick Start Video / 빠른 시작 영상

The 15-second quick start video is available in the GitHub Release assets:

15초 Quick Start 영상은 GitHub Release assets에서 확인할 수 있습니다:

[View release assets](https://github.com/kyejin1991/AUDIO-MASTERING/releases/tag/v0.1.0-community)

## Third-Party Components / 서드파티 구성요소

This project uses third-party open-source components including Demucs and pyloudnorm.

이 프로젝트는 Demucs와 pyloudnorm을 포함한 서드파티 오픈소스 구성요소를 사용합니다.

See:

참고:

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- [NOTICE](NOTICE)

## Community / Pro Boundary / Community / Pro 경계

| Edition | Status | Included |
|---|---|---|
| Community | Open Source | Analysis, manual workflow layer, basic DSP structure |
| Pro | Proprietary | AI mastering engine, rendering orchestration, genre profiles |

| 에디션 | 상태 | 포함 범위 |
|---|---|---|
| Community | 오픈소스 | 분석, 수동 워크플로우 레이어, 기본 DSP 구조 |
| Pro | 비공개 | AI 마스터링 엔진, 렌더링 오케스트레이션, 장르별 프로필 |

The Community Edition intentionally includes only the open-source audio analysis and manual workflow layer.

Community Edition은 의도적으로 오픈소스 오디오 분석 및 수동 워크플로우 레이어만 포함합니다.

The AI mastering decision engine, Pro rendering orchestration, genre-specific profiles, and commercial presets are part of the separate Pro engine and are not included in this public repository.

AI 마스터링 의사결정 엔진, Pro 렌더링 오케스트레이션, 장르별 프로필, 상업용 프리셋은 별도의 Pro 엔진에 속하며 이 공개 저장소에는 포함되지 않습니다.

## Support / 응원하기

If you find this project useful, please consider giving it a ⭐ on GitHub.

프로젝트가 도움이 되었다면 GitHub Star를 눌러주세요.

It helps the project reach more developers and audio creators.

더 많은 개발자와 오디오 크리에이터에게 프로젝트가 닿는 데 큰 도움이 됩니다.

## License / 라이선스

See [LICENSE](LICENSE).

## Author / 작성자

Developed by [kyejin1991](https://github.com/kyejin1991)  
© 2026 Arcapps
