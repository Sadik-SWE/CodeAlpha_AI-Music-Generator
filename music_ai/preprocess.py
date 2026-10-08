
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np

REST = 128
DURATION_BINS = np.array([0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0,
                          2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0], dtype=float)

def duration_to_id(seconds: float) -> int:
    idx = int(np.argmin(np.abs(DURATION_BINS - max(0.125, seconds))))
    return idx

def id_to_duration(idx: int) -> float:
    return float(DURATION_BINS[int(idx)])

def midi_to_events(midi_path: str | Path, max_events: int = 2500) -> List[Tuple[int, int]]:
    """Extract monophonic note events from MIDI using music21.
    Returns (pitch, duration_id). Chords are reduced to their top note.
    """
    from music21 import converter, chord, note, stream

    score = converter.parse(str(midi_path))
    flat = score.flatten()
    events = []
    for element in flat.notesAndRests:
        try:
            dur = float(element.duration.quarterLength) * 0.5  # approx seconds at 120 BPM
        except Exception:
            continue
        if isinstance(element, note.Note):
            pitch = int(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            if not element.pitches:
                continue
            pitch = max(int(p.midi) for p in element.pitches)
        else:
            pitch = REST
        events.append((pitch, duration_to_id(dur)))
        if len(events) >= max_events:
            break
    return events

def build_dataset(raw_dir: str | Path, out_file: str | Path,
                  seq_len: int = 64, max_events_per_file: int = 2500):
    raw_dir, out_file = Path(raw_dir), Path(out_file)
    all_events = []
    files = sorted(list(raw_dir.rglob("*.mid")) + list(raw_dir.rglob("*.midi")))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {raw_dir}")
    used = []
    for path in files:
        try:
            ev = midi_to_events(path, max_events_per_file)
            if len(ev) >= seq_len + 1:
                all_events.extend(ev)
                used.append(str(path))
        except Exception as exc:
            print(f"[WARN] Could not parse {path}: {exc}")

    if len(all_events) < seq_len + 1:
        raise RuntimeError("Not enough valid MIDI events to build a dataset.")

    pitches = np.array([x[0] for x in all_events], dtype=np.int64)
    durations = np.array([x[1] for x in all_events], dtype=np.int64)

    payload = {
        "pitches": pitches.tolist(),
        "durations": durations.tolist(),
        "seq_len": seq_len,
        "files": used,
        "duration_bins": DURATION_BINS.tolist(),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved {len(all_events)} events from {len(used)} MIDI files -> {out_file}")
    return payload

def load_dataset(dataset_file: str | Path):
    data = json.loads(Path(dataset_file).read_text(encoding="utf-8"))
    return np.asarray(data["pitches"], dtype=np.int64), np.asarray(data["durations"], dtype=np.int64), int(data["seq_len"])
