# MuseAI — Task 3 Project Report

## Objective
Build an AI system capable of learning musical patterns from MIDI sequences and generating new music.

## Method
The system converts MIDI notes into a sequence of two features: MIDI pitch and quantized note duration. A two-layer LSTM receives previous events and predicts the next pitch and duration. During generation, temperature and Top-K sampling control creativity.

## Output
The generated sequence is converted into a standard MIDI file and rendered into WAV audio for browser playback.

## Deployment
The Streamlit interface provides an interactive generation studio, audio preview, MIDI/WAV downloads, and a visual composition preview.

## Limitations
The included checkpoint is intentionally small so the live application can run on a free cloud environment. A larger licensed dataset and longer training run should be used for research-quality musical coherence.

## Future Work
- Transformer-based music generation
- Multi-instrument/polyphonic modeling
- Chord-aware conditioning
- Genre-conditioned training
- Larger licensed MIDI datasets
- Human evaluation and music-theory metrics
