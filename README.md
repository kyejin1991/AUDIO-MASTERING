# audio-mastering Community Edition

![Hero](assets/images/hero.png)

`audio-mastering` is an open-source audio analysis and manual mastering toolkit.

## Features

- Loudness analysis
- Spectrum analysis
- Dynamics analysis
- Stereo analysis
- Basic DSP modules
- Manual mastering workflow
- Optional stem separation interface

## What This Is Not

This repository does not include the proprietary Pro engine:

- AI Master Assistant
- Automatic mastering decision engine
- Genre-specific profiles
- Pro rendering orchestration
- Commercial presets
- Pro QC workflow

## Quick Start

```bash
git clone https://github.com/YOUR_NAME/audio-mastering.git
cd audio-mastering
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
