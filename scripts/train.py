
from pathlib import Path
import sys, argparse, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from music_ai.model import MusicLSTM
from music_ai.preprocess import load_dataset

class SequenceDataset(Dataset):
    def __init__(self, pitches, durations, seq_len):
        self.p, self.d, self.seq_len = pitches, durations, seq_len
    def __len__(self):
        return max(1, len(self.p) - self.seq_len - 1)
    def __getitem__(self, i):
        x_p = torch.tensor(self.p[i:i+self.seq_len], dtype=torch.long)
        x_d = torch.tensor(self.d[i:i+self.seq_len], dtype=torch.long)
        y_p = torch.tensor(self.p[i+1:i+self.seq_len+1], dtype=torch.long)
        y_d = torch.tensor(self.d[i+1:i+self.seq_len+1], dtype=torch.long)
        return x_p, x_d, y_p, y_d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset.json")
    ap.add_argument("--output", default="models/lstm_music_generator.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    p, d, seq_len = load_dataset(ROOT/args.dataset)
    ds = SequenceDataset(p, d, seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    config = dict(pitch_vocab=129, duration_vocab=16, hidden_size=128, num_layers=2, dropout=0.15)
    model = MusicLSTM(**config)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        for xp, xd, yp, yd in dl:
            opt.zero_grad()
            lp, ld, _ = model(xp, xd)
            loss = loss_fn(lp.reshape(-1,129), yp.reshape(-1)) + 0.7*loss_fn(ld.reshape(-1,16), yd.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={total/max(1,len(dl)):.4f}")

    out = ROOT/args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": config,
        "seq_len": seq_len,
        "trained_on": "User-provided MIDI dataset",
    }, out)
    print(f"Saved model -> {out}")

if __name__ == "__main__":
    main()
