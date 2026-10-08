
from __future__ import annotations
import random
from pathlib import Path
import torch
import numpy as np

from .model import MusicLSTM
from .preprocess import id_to_duration, REST

def load_model(checkpoint: str | Path, device="cpu"):
    ckpt = torch.load(checkpoint, map_location=device)
    model = MusicLSTM(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt

def _sample(logits, temperature=1.0, top_k=12):
    logits = logits / max(0.15, float(temperature))
    k = min(top_k, logits.shape[-1])
    vals, inds = torch.topk(logits, k)
    probs = torch.softmax(vals, dim=-1)
    choice = torch.multinomial(probs, 1)
    return int(inds[choice].item())

@torch.no_grad()
def generate_events(model, seed_pitches, seed_durations, length=96,
                    temperature=0.9, top_k=12, device="cpu"):
    pitches = list(map(int, seed_pitches))
    durations = list(map(int, seed_durations))
    if not pitches:
        pitches, durations = [60], [5]

    p = torch.tensor([pitches[-64:]], dtype=torch.long, device=device)
    d = torch.tensor([durations[-64:]], dtype=torch.long, device=device)
    pitch_logits, dur_logits, hidden = model(p, d)

    for _ in range(int(length)):
        next_p = _sample(pitch_logits[:, -1, :][0], temperature, top_k)
        next_d = _sample(dur_logits[:, -1, :][0], temperature, min(8, top_k))
        pitches.append(next_p)
        durations.append(next_d)
        ip = torch.tensor([[next_p]], dtype=torch.long, device=device)
        idur = torch.tensor([[next_d]], dtype=torch.long, device=device)
        pitch_logits, dur_logits, hidden = model(ip, idur, hidden)

    return list(zip(pitches, durations))

def events_to_midi(events, out_path: str | Path, tempo=120):
    import pretty_midi
    out_path = Path(out_path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=0, name="AI Piano")
    t = 0.0
    for pitch, dur_id in events:
        dur = id_to_duration(dur_id)
        if int(pitch) != REST:
            note = pretty_midi.Note(velocity=82, pitch=int(pitch),
                                    start=t, end=t + dur)
            inst.notes.append(note)
        t += dur
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return out_path

def midi_to_wav(midi_path: str | Path, wav_path: str | Path):
    """Dependency-light piano-ish renderer for browser playback.
    This is intentionally simple and deterministic; no external SoundFont is required.
    """
    import pretty_midi
    from scipy.io.wavfile import write

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    fs = 22050
    audio = pm.fluidsynth(fs=fs) if False else None  # never requires fluidsynth
    # Render from note events with a simple additive synth.
    duration = max((n.end for inst in pm.instruments for n in inst.notes), default=1.0) + 0.2
    samples = int(duration * fs)
    y = np.zeros(samples, dtype=np.float32)
    for inst in pm.instruments:
        for n in inst.notes:
            start, end = int(n.start*fs), min(samples, int(n.end*fs))
            if end <= start: 
                continue
            tt = np.arange(end-start, dtype=np.float32) / fs
            f = 440.0 * (2.0 ** ((n.pitch-69)/12.0))
            env = np.ones_like(tt)
            attack = min(int(0.015*fs), len(env)//2)
            release = min(int(0.12*fs), len(env)//2)
            if attack:
                env[:attack] = np.linspace(0, 1, attack)
            if release:
                env[-release:] *= np.linspace(1, 0, release)
            wave = (np.sin(2*np.pi*f*tt) +
                    0.28*np.sin(2*np.pi*2*f*tt) +
                    0.12*np.sin(2*np.pi*3*f*tt)) / 1.4
            y[start:end] += wave * env * (n.velocity/127.0) * 0.28
    peak = max(1e-6, float(np.max(np.abs(y))))
    y = np.clip(y / peak * 0.9, -1, 1)
    wav_path = Path(wav_path)
    write(str(wav_path), fs, (y*32767).astype(np.int16))
    return wav_path

def seed_from_style(style: str):
    scales = {
        "Classical": [60, 62, 64, 65, 67, 69, 71, 72],
        "Jazz": [60, 62, 63, 65, 67, 69, 70, 72],
        "Ambient": [57, 60, 64, 67, 71, 72],
        "Cinematic": [48, 55, 60, 62, 67, 69, 74],
        "Folk": [60, 62, 65, 67, 69, 72, 74],
    }
    notes = scales.get(style, scales["Classical"])
    seed = [random.choice(notes) for _ in range(12)]
    durs = [random.choice([3, 4, 5, 6]) for _ in seed]
    return seed, durs
