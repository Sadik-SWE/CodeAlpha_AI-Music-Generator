
from pathlib import Path
import sys, random, uuid, base64
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music_ai.generation import load_model, generate_events, events_to_midi, midi_to_wav, seed_from_style

st.set_page_config(page_title="MuseAI — Music Generator", page_icon="🎼", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 2rem 2.2rem; border: 1px solid rgba(128,128,128,.18); border-radius: 24px;
       background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(236,72,153,.08)); margin-bottom: 1.5rem;}
.hero h1 {font-size: 3rem; margin: 0;}
.hero p {font-size: 1.08rem; opacity: .82;}
.card {padding: 1.15rem; border: 1px solid rgba(128,128,128,.18); border-radius: 18px;}
.small {font-size: .88rem; opacity: .72;}
.stMetric {border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

MODEL = ROOT/"models"/"lstm_music_generator.pt"
st.markdown("""
<div class="hero">
<h1>🎼 MuseAI</h1>
<p>AI-powered MIDI music generation with an LSTM neural network — create, preview, and export original musical sequences.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL.exists():
    st.error("Model checkpoint is missing. Run the training pipeline before deploying this app.")
    st.stop()

@st.cache_resource
def get_model():
    return load_model(MODEL)

model, ckpt = get_model()

with st.sidebar:
    st.header("🎛️ Generation Studio")
    style = st.selectbox("Musical style", ["Classical", "Jazz", "Ambient", "Cinematic", "Folk"])
    length = st.slider("Generated notes", 24, 180, 80, 8)
    temperature = st.slider("Creativity / temperature", 0.35, 1.5, 0.85, 0.05)
    top_k = st.slider("Top-K sampling", 4, 24, 12)
    tempo = st.slider("Tempo (BPM)", 60, 180, 110, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    generate = st.button("✨ Generate Music", use_container_width=True, type="primary")
    st.divider()
    st.caption("Model: LSTM • 2 layers • 128 hidden units")
    st.caption("Output: MIDI + WAV")
    st.caption("No external music API required")

if "result" not in st.session_state:
    st.session_state.result = None

if generate:
    random.seed(int(seed))
    seed_p, seed_d = seed_from_style(style)
    events = generate_events(model, seed_p, seed_d, length=length,
                             temperature=temperature, top_k=top_k)
    run_id = uuid.uuid4().hex[:8]
    out_dir = ROOT/"outputs"
    out_dir.mkdir(exist_ok=True)
    midi_path = out_dir/f"museai_{run_id}.mid"
    wav_path = out_dir/f"museai_{run_id}.wav"
    events_to_midi(events, midi_path, tempo=tempo)
    midi_to_wav(midi_path, wav_path)
    st.session_state.result = {
        "events": events, "midi": midi_path, "wav": wav_path,
        "style": style, "tempo": tempo, "seed": seed
    }

r = st.session_state.result

col1, col2, col3 = st.columns(3)
col1.metric("Model", "LSTM")
col2.metric("Style", r["style"] if r else "—")
col3.metric("Notes", len(r["events"]) if r else "—")

if r:
    st.subheader("🎧 Your AI composition")
    st.audio(str(r["wav"]), format="audio/wav")
    st.caption(f"Style: {r['style']}  •  Tempo: {r['tempo']} BPM  •  Seed: {r['seed']}")

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download MIDI", r["midi"].read_bytes(),
                       file_name=f"MuseAI_{r['style']}.mid", mime="audio/midi",
                       use_container_width=True)
    c2.download_button("⬇️ Download WAV Audio", r["wav"].read_bytes(),
                       file_name=f"MuseAI_{r['style']}.wav", mime="audio/wav",
                       use_container_width=True)

    st.subheader("🎹 Composition preview")
    try:
        import pandas as pd
        import plotly.graph_objects as go
        rows, t = [], 0.0
        for idx, (pitch, dur_id) in enumerate(r["events"]):
            from music_ai.preprocess import id_to_duration, REST
            dur = id_to_duration(dur_id)
            if pitch != REST:
                rows.append((idx, pitch, t, t+dur))
            t += dur
        fig = go.Figure()
        for idx, pitch, start, end in rows:
            fig.add_trace(go.Scatter(x=[start,end], y=[pitch,pitch],
                                     mode="lines", line=dict(width=9),
                                     hovertemplate=f"Note {idx}<br>MIDI pitch: {pitch}<extra></extra>",
                                     showlegend=False))
        fig.update_layout(height=360, xaxis_title="Time (seconds)",
                          yaxis_title="MIDI pitch", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Preview chart unavailable; audio and MIDI export are still available.")

else:
    st.info("Choose your style and generation settings, then click **Generate Music**.")
    st.markdown("""
### How it works
1. A seed melody is selected from the chosen musical style.
2. The LSTM predicts the next pitch and note duration repeatedly.
3. The generated event sequence is converted to a standard MIDI file.
4. A lightweight local synthesizer renders the MIDI as WAV for instant browser playback.

> **Academic note:** This project uses an LSTM-based sequence model and is designed as an educational music-generation system. The included checkpoint is a small demo model so the deployed app can run immediately. For a stronger result, retrain it with a larger, properly licensed MIDI collection.
""")

with st.expander("📚 About the model"):
    st.write("The network learns two distributions at each time step: the next MIDI pitch (0–127 + REST) and the next quantized duration. Temperature controls sampling randomness and Top-K limits candidate notes.")
    st.write("Training pipeline: MIDI → music21 preprocessing → event sequences → PyTorch LSTM → checkpoint → generation → MIDI/WAV.")
