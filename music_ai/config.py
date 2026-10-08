
from dataclasses import dataclass

@dataclass
class Config:
    seq_len: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 0.002
    pitch_vocab: int = 129   # 0-127 MIDI pitches + 128 = REST
    duration_vocab: int = 16
    checkpoint_name: str = "lstm_music_generator.pt"
