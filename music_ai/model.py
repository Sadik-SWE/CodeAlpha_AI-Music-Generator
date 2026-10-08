
from __future__ import annotations
import torch
from torch import nn

class MusicLSTM(nn.Module):
    def __init__(self, pitch_vocab=129, duration_vocab=16,
                 hidden_size=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, 48)
        self.duration_emb = nn.Embedding(duration_vocab, 16)
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.pitch_head = nn.Linear(hidden_size, pitch_vocab)
        self.duration_head = nn.Linear(hidden_size, duration_vocab)

    def forward(self, pitches, durations, hidden=None):
        x = torch.cat([self.pitch_emb(pitches), self.duration_emb(durations)], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return self.pitch_head(out), self.duration_head(out), hidden
