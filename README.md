# AI Mastering Lab Community Edition

![Hero](assets/images/hero.png)

Community edition of AI Mastering Lab for audio analysis, DSP, and manual mastering.

## Features

- Loudness analysis
- Spectrum analysis
- Dynamics analysis
- Stereo analysis
- Basic DSP modules
- Manual mastering workflow
- Optional stem separation interface

## What This Is

AI Mastering Lab Community Edition is a manual audio mastering toolkit.

## What This Is Not

This repository does not include:

- AI Master Assistant
- Automatic mastering decision engine
- Genre-specific Pro profiles
- Pro rendering orchestration
- Commercial presets
- Pro QC workflow

## Quick Start

```bash
git clone https://github.com/YOUR_NAME/ai-mastering-lab-community.git
cd ai-mastering-lab-community
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots

### Audio Upload

![Upload Screen](assets/images/upload_screen.png)

### Audio Analysis

![Analysis Screen](assets/images/analysis_screen.png)

### Manual Mastering Rack

![Module Rack](assets/images/module_rack.png)

### Render Result

![Render Result](assets/images/render_result.png)

## Quick Start Video

https://github.com/user-attachments/assets/...

## Third-Party Components

This project uses third-party open-source components including Demucs and pyloudnorm.

See:
- `THIRD_PARTY_LICENSES.md`
- `NOTICE`

## Community / Pro Boundary

| Edition | Status | Included |
|---|---|---|
| Community | Open Source | Analysis, basic DSP, manual mastering |
| Pro | Proprietary | AI decision engine, automatic mastering, genre profiles |

## License

See `LICENSE`.
