# 🎼 MuseAI — AI Music Generation with LSTM

MuseAI is a professional educational AI music-generation application that learns musical patterns from MIDI event sequences using a **PyTorch LSTM** and generates new compositions as **MIDI + WAV audio**.

## ✨ Features

- 🎹 LSTM-based deep-learning music generation
- 🎼 MIDI preprocessing with `music21`
- 🎧 Browser audio preview without requiring an external music API
- ⬇️ MIDI and WAV export
- 🎛️ Style presets: Classical, Jazz, Ambient, Cinematic, Folk
- 🌡️ Temperature and Top-K sampling controls
- 📈 Interactive composition/piano-roll-style preview
- 🔁 Reproducible generation with random seeds
- ☁️ Streamlit-ready deployment
- 🔒 No API keys required

## 🧠 Architecture

```text
MIDI Collection
      ↓
music21 preprocessing
      ↓
Pitch + Duration event sequences
      ↓
PyTorch LSTM
      ↓
Next Pitch + Next Duration prediction
      ↓
Sampling (Temperature + Top-K)
      ↓
Generated MIDI
      ↓
Lightweight WAV renderer
      ↓
Browser playback + downloads
```

## 📁 Project Structure

```text
AI_Music_Generator/
├── app/
│   └── app.py
├── music_ai/
│   ├── config.py
│   ├── generation.py
│   ├── model.py
│   └── preprocess.py
├── scripts/
│   ├── create_demo_dataset.py
│   ├── prepare_dataset.py
│   └── train.py
├── data/
│   └── raw/
├── models/
│   └── lstm_music_generator.pt
├── outputs/
├── docs/
├── requirements.txt
└── README.md
```

## 🚀 Run locally

### 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app

```powershell
streamlit run app/app.py
```

Open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## 🎓 Train with your own MIDI dataset

Put legally obtained MIDI files into:

```text
data/raw/
```

Then preprocess:

```powershell
python scripts/prepare_dataset.py
```

Train:

```powershell
python scripts/train.py --epochs 30
```

The trained checkpoint will be saved to:

```text
models/lstm_music_generator.pt
```

Then launch:

```powershell
streamlit run app/app.py
```

### Demo dataset

The repository also contains a generator for small synthetic MIDI examples:

```powershell
python scripts/create_demo_dataset.py
python scripts/prepare_dataset.py
python scripts/train.py --epochs 30
```

The included checkpoint is already trained on a small bundled demo MIDI collection so the app can be launched immediately. For the strongest academic result, replace it by retraining on a larger, properly licensed MIDI collection.

## ☁️ Streamlit deployment

1. Push this repository to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Select your GitHub repository.
4. Set the main file to:

```text
app/app.py
```

5. Deploy.

The included `requirements.txt` installs the model and audio dependencies. The app does **not** require an external API key or system SoundFont.

## ⚠️ Dataset and copyright

Only use MIDI files that you have permission to use. For a serious research/demo result, use a properly licensed dataset and document its source, license, preprocessing, and train/validation split.

## 🧪 Academic task mapping

| Task requirement | Implementation |
|---|---|
| Collect MIDI music data | `data/raw/` + dataset workflow |
| Preprocess MIDI | `music_ai/preprocess.py` with music21 |
| Deep learning model | PyTorch LSTM |
| Train model | `scripts/train.py` |
| Generate new music | `music_ai/generation.py` |
| Convert to MIDI | `events_to_midi()` |
| Play/save as audio | WAV renderer + Streamlit audio |
| Deploy live | Streamlit Community Cloud |

## 👨‍💻 Project

**MuseAI — Music Generation with AI**

Built as an academic AI/deep-learning project demonstrating sequence modeling, MIDI processing, neural music generation, and deployment.
