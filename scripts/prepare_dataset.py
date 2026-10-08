
from pathlib import Path
import sys, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from music_ai.preprocess import build_dataset

ap = argparse.ArgumentParser(description="Preprocess collected MIDI files into an LSTM training dataset.")
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--output", default="data/processed/dataset.json")
ap.add_argument("--seq-len", type=int, default=64)
args = ap.parse_args()
build_dataset(ROOT/args.raw, ROOT/args.output, seq_len=args.seq_len)
