
from pathlib import Path
import random, sys
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data"/"raw"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

scales = [
    [60,62,64,65,67,69,71,72],
    [57,59,60,62,64,65,67,69],
    [48,50,52,55,57,59,60,62],
    [60,62,65,67,69,72,74,76],
]
for i in range(36):
    pm = pretty_midi.PrettyMIDI(initial_tempo=random.choice([90,100,110,120,128]))
    inst = pretty_midi.Instrument(program=0)
    scale = random.choice(scales)
    t = 0.0
    for _ in range(random.randint(90, 150)):
        p = random.choice(scale)
        if random.random() < 0.08:
            t += 0.25
            continue
        dur = random.choice([0.25,0.5,0.75,1.0,1.5])
        inst.notes.append(pretty_midi.Note(velocity=random.randint(65,105), pitch=p,
                                           start=t, end=t+dur))
        t += dur
    pm.instruments.append(inst)
    pm.write(str(OUT/f"demo_{i:02d}.mid"))
print(f"Created demo MIDI dataset in {OUT}")
