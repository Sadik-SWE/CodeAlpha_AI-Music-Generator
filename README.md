# 🎼 MuseAI

### AI-Powered Music Generation with PyTorch LSTM

MuseAI is an educational **AI music generation application** that learns musical patterns from MIDI sequences using a **PyTorch LSTM neural network** and generates new compositions as **MIDI and WAV audio**.

Built to demonstrate the practical application of **deep learning, sequence modeling, MIDI processing, generative AI, and ML deployment** in a complete end-to-end project.

<p align="center">

### 🚀 [Live Demo](https://codealphaai-music-generator-hkxxr8oacqwqtgfobeh8yf.streamlit.app/) · [GitHub Repository](https://github.com/Sadik-SWE/CodeAlpha_AI-Music-Generator)

</p>

---

## ✨ Highlights

* 🎹 **LSTM-based music generation**
* 🧠 PyTorch deep-learning architecture
* 🎼 MIDI preprocessing with `music21`
* 🎧 Built-in browser audio playback
* 🎵 MIDI composition generation
* 🔊 WAV audio rendering
* 🎛️ Classical, Jazz, Ambient, Cinematic & Folk presets
* 🌡️ Temperature-based sampling
* 🎯 Top-K sampling
* 🎲 Reproducible generation with random seeds
* 📊 Interactive musical composition visualization
* ☁️ Streamlit Community Cloud deployment
* 🔐 No external API keys required

---

## 🎯 How It Works

MuseAI transforms MIDI music into sequences of musical events, trains an LSTM model to learn those patterns, and uses the trained model to generate new compositions.

```text
             MIDI Dataset
                  │
                  ▼
          MIDI Preprocessing
             (music21)
                  │
                  ▼
       Pitch + Duration Events
                  │
                  ▼
          Sequence Modeling
                  │
                  ▼
           PyTorch LSTM
                  │
                  ▼
      Next Event Prediction
                  │
                  ▼
       Temperature + Top-K
             Sampling
                  │
                  ▼
          Generated Music
                  │
             ┌────┴────┐
             ▼         ▼
           MIDI       WAV
             │         │
             └────┬────┘
                  ▼
          Streamlit Web App
```

---

## 🧠 Model Architecture

The core generation model uses a **Long Short-Term Memory (LSTM)** network designed for sequential musical data.

The model learns relationships between:

* Musical pitch
* Note duration
* Previous musical events
* Temporal sequence patterns

During generation, the model predicts the next pitch and duration based on previously generated events.

```text
Previous Events
      │
      ▼
┌───────────────┐
│   LSTM Model  │
└───────┬───────┘
        │
   ┌────┴─────┐
   ▼          ▼
Pitch       Duration
Prediction  Prediction
   │          │
   └────┬─────┘
        ▼
   New Musical Event
```

---

## 🎛️ Generation Controls

### Temperature

Controls the randomness of generated music.

| Setting | Behavior         |
| ------- | ---------------- |
| Lower   | More predictable |
| Higher  | More diverse     |

### Top-K Sampling

Limits the prediction space to the most probable candidates, helping balance **musical consistency and creativity**.

### Random Seed

Allows reproducible generations for experimentation and academic evaluation.

---

## 🛠️ Tech Stack

| Technology     | Role                     |
| -------------- | ------------------------ |
| **Python**     | Core development         |
| **PyTorch**    | LSTM deep-learning model |
| **music21**    | MIDI processing          |
| **Streamlit**  | Web application          |
| **NumPy**      | Numerical computation    |
| **Pandas**     | Data processing          |
| **Matplotlib** | Visualization            |
| **MIDI**       | Musical representation   |
| **WAV**        | Audio output             |

---

## 📁 Project Structure

```text
MuseAI/
│
├── app/
│   └── app.py                 # Streamlit application
│
├── music_ai/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── generation.py          # Music generation
│   ├── model.py               # LSTM architecture
│   └── preprocess.py          # MIDI preprocessing
│
├── scripts/
│   ├── create_demo_dataset.py # Demo MIDI generation
│   ├── prepare_dataset.py     # Dataset preparation
│   └── train.py               # Model training
│
├── data/
│   └── raw/                   # MIDI dataset
│
├── models/
│   └── lstm_music_generator.pt
│
├── docs/
│   ├── DEPLOYMENT_CHECKLIST.md
│   └── PROJECT_REPORT.md
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

---

# 🚀 Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Sadik-SWE/CodeAlpha_AI-Music-Generator.git
cd CodeAlpha_AI-Music-Generator
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Launch the Application

```powershell
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

# 🎼 Train with Your Own MIDI Dataset

Place legally obtained MIDI files inside:

```text
data/raw/
```

Then run:

```powershell
python scripts/prepare_dataset.py
```

Train the LSTM:

```powershell
python scripts/train.py --epochs 30
```

The trained model will be saved to:

```text
models/lstm_music_generator.pt
```

Launch the application again:

```powershell
streamlit run app/app.py
```

---

## 🧪 Demo Dataset

For a quick experiment, the project includes a synthetic MIDI dataset generator.

```powershell
python scripts/create_demo_dataset.py
python scripts/prepare_dataset.py
python scripts/train.py --epochs 30
```

The repository also contains a pre-trained checkpoint, allowing the application to run without training from scratch.

---

# ☁️ Live Deployment

MuseAI is deployed on **Streamlit Community Cloud**.

### 🌐 Live Application

**https://codealphaai-music-generator-hkxxr8oacqwqtgfobeh8yf.streamlit.app/**

### Entry Point

```text
app/app.py
```

### Deployment Requirements

```text
requirements.txt
packages.txt
```

The current application does not require an external API key.

---

# 📊 Project Workflow

```text
Dataset
   ↓
MIDI Parsing
   ↓
Event Extraction
   ↓
Sequence Preparation
   ↓
LSTM Training
   ↓
Music Generation
   ↓
MIDI Creation
   ↓
WAV Rendering
   ↓
Streamlit Deployment
```

---

# 🎓 Academic Scope

MuseAI demonstrates practical concepts in:

* Artificial Intelligence
* Deep Learning
* Generative AI
* Sequence Modeling
* Recurrent Neural Networks
* LSTM Networks
* Music Information Processing
* MIDI Processing
* Model Training
* ML Application Deployment

---

# ⚠️ Limitations

MuseAI is primarily an **educational and experimental project**.

The current model is trained on a relatively small demo dataset, so generated compositions may have limited musical complexity and long-term structure.

For improved results, future versions can use:

* Larger licensed MIDI datasets
* Transformer-based architectures
* Multi-track generation
* Instrument-aware generation
* Chord-conditioned generation
* More advanced audio synthesis

---

# 📌 Dataset & Copyright

Only use MIDI files that you have the legal right or appropriate permission to use.

For research or academic work, document the dataset source, license, preprocessing method, and training configuration.

---

# 👨‍💻 Project

**MuseAI — AI Music Generation with LSTM**

Developed by **Shahariar Sadik**

**Focus:** AI Engineering · Deep Learning · Generative AI · Python · ML Deployment

---

## 🔗 Links

🌐 **Live Demo**
https://codealphaai-music-generator-hkxxr8oacqwqtgfobeh8yf.streamlit.app/

💻 **GitHub**
https://github.com/Sadik-SWE/CodeAlpha_AI-Music-Generator

---

<p align="center">

**🎼 MuseAI — Exploring Creativity Through Deep Learning**

Built with Python · PyTorch · music21 · Streamlit

</p>
