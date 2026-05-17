# audio-mastering

![Release](https://img.shields.io/github/v/release/kyejin1991/AUDIO-MASTERING?include_prereleases)
![License](https://img.shields.io/github/license/kyejin1991/AUDIO-MASTERING)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-community%20release-green)

audio-mastering Community Edition is the open-source audio analysis and manual workflow layer of AI Mastering Lab.

It can create projects, inspect audio files, and run loudness, spectrum, dynamics, and stereo analysis.

## Quick Start Guide

For a more detailed guide, see [docs/QUICK_START.md](docs/QUICK_START.md).

## Features

- Project creation from local audio files
- Loudness analysis
- Spectrum analysis
- Dynamics analysis
- Stereo analysis
- Streamlit-based Community UI
- Basic DSP module structure

## Community Scope

audio-mastering Community Edition includes the open-source analysis and manual workflow layer of AI Mastering Lab.

It can create projects, inspect audio files, and run loudness, spectrum, dynamics, and stereo analysis.

The proprietary Pro engine is not included in this repository. This includes the AI Master Assistant, AI mastering decision logic, Pro rendering orchestration, genre-specific profiles, and commercial presets.

Community is not an unfinished version. It is the open-source layer.

Pro is a separate proprietary product layer.

## What You Can Do

- Load audio files
- Create analysis projects
- Inspect loudness, spectrum, dynamics, and stereo information
- Use the Community UI as a base for manual workflow foundations
- Explore the basic DSP module structure

## What Is Not Included

This public repository does not include the Pro mastering engine.

The following components are intentionally excluded:

- AI Master Assistant
- AI mastering decision engine
- Pro rendering orchestration
- Genre-specific profiles
- Commercial presets
- Pro QC workflow

## Quick Start

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

## Screenshots

### Overview

![audio-mastering overview](assets/images/hero.png)

### Audio Upload

![Upload screen](assets/images/upload_screen.png)

### Audio Analysis

![Analysis screen](assets/images/analysis_screen.png)

### Manual Mastering Rack

![Module rack](assets/images/module_rack.png)

### Render Result

![Render result](assets/images/render_result.png)

## Quick Start Video

The 15-second quick start video is available in the GitHub Release assets:

[View release assets](https://github.com/kyejin1991/AUDIO-MASTERING/releases/tag/v0.1.0-community)

## Third-Party Components

This project uses third-party open-source components including Demucs and pyloudnorm.

See:

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- [NOTICE](NOTICE)

## Community / Pro Boundary

| Edition | Status | Included |
|---|---|---|
| Community | Open Source | Analysis, manual workflow layer, basic DSP structure |
| Pro | Proprietary | AI mastering engine, rendering orchestration, genre profiles |

The Community Edition intentionally includes only the open-source audio analysis and manual workflow layer.

The AI mastering decision engine, Pro rendering orchestration, genre-specific profiles, and commercial presets are part of the separate Pro engine and are not included in this public repository.

## License

See [LICENSE](LICENSE).

## Author

Developed by [kyejin1991](https://github.com/kyejin1991)  
© 2026 Arcapps
