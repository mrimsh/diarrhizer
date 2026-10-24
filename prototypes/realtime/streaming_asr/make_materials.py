"""Synthesize Russian test materials with Silero TTS from plain-text scripts (scripts and audio stay in ../_local, not committed).

    python make_materials.py            # reads ../_local/texts/*.txt, writes ../_local/materials/<name>.{wav,json,txt}

Script format: one turn per line, "<speaker>[<offset>]:<lang>: text"
  speaker  A | B | C            (voice mapping below), a line without a speaker prefix is the monologue
  [-0.8]   optional: the turn starts 0.8 s BEFORE the previous turn ends (interruption / overlap)
  :en      optional: the turn is English (Silero v3_en), default Russian
Outputs per material:
  <name>.wav       16 kHz mono mix (what the ASR variants listen to), also <name>.48k.wav for the Diarrhizer pipeline
  <name>.stem_<S>.wav   16 kHz stem of every speaker (ground truth for channel / diarization experiments)
  <name>.json      turns with start/end time, speaker, language, text
  <name>.ref.txt   true text in chronological order of turn start (the script itself)
The "noisy" variant of a material (--noisy) adds pink noise (SNR 12 dB), a quiet babble of another voice and 300-3400 Hz band limit.
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfilt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.abspath(os.path.join(HERE, "..", "_local"))
TXT, OUT = os.path.join(LOCAL, "texts"), os.path.join(LOCAL, "materials")
os.makedirs(OUT, exist_ok=True)
SR = 24000
VOICES = {"ru": {"A": "aidar", "B": "baya", "C": "xenia", "M": "aidar"}, "en": {"A": "en_3", "B": "en_21", "C": "en_8", "M": "en_3"}}
rng = np.random.default_rng(7)
dev = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}

def tts_model(lang):
    if lang not in _models:
        from silero import silero_tts
        m, _ = silero_tts(language=lang, speaker="v4_ru" if lang == "ru" else "v3_en")
        m.to(dev)          # Silero's .to() returns None
        _models[lang] = m
    return _models[lang]

def say(text, speaker, lang):
    """Silero takes ~1000 characters per call: split on sentences."""
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(cur) + len(s) > 700 and cur:
            parts.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        parts.append(cur)
    m = tts_model(lang)
    out = []
    for p in parts:
        kw = dict(put_accent=True, put_yo=True) if lang == "ru" else {}
        a = m.apply_tts(text=p, speaker=VOICES[lang][speaker], sample_rate=SR, **kw).numpy().astype("float32")
        out += [a, np.zeros(int(0.12 * SR), "float32")]
    return np.concatenate(out)

def parse(path):
    turns = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^(?:([ABC])(?:\[(-?[\d.]+)\])?(?::(\w\w))?:\s*)?(.*)$", ln)
        spk, off, lang, text = m.group(1) or "M", float(m.group(2) or 0), m.group(3) or "ru", m.group(4)
        turns.append((spk, off, lang, text))
    return turns

def pink(n):
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = 1
    x = np.fft.irfft((rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f), n)
    return (x / np.std(x)).astype("float32")

def build(name, noisy=False):
    turns = parse(os.path.join(TXT, name + ".txt"))
    t_end, rows, stems = 0.0, [], {}
    for spk, off, lang, text in turns:
        a = say(text, spk, lang)
        gap = float(rng.uniform(0.25, 0.7)) if off == 0 else off      # off < 0: start before the previous end
        start = max(0.0, t_end + gap)
        i0 = int(start * SR)
        st = stems.setdefault(spk, np.zeros(0, "float32"))
        if len(st) < i0 + len(a):
            st = np.pad(st, (0, i0 + len(a) - len(st)))
        st[i0:i0 + len(a)] += a
        stems[spk] = st
        t_end = start + len(a) / SR
        rows.append({"speaker": spk, "lang": lang, "start": round(start, 2), "end": round(t_end, 2), "text": text})
    n = max(len(s) for s in stems.values())
    stems = {k: np.pad(v, (0, n - len(v))) for k, v in stems.items()}
    mix = sum(stems.values())
    tag = name
    if noisy:
        tag = name + "_noisy"
        sos = butter(4, [300, 3400], btype="band", fs=SR, output="sos")
        sig_rms = np.sqrt(np.mean(mix ** 2))
        noise = pink(n) * sig_rms / (10 ** (12 / 20))
        bab = np.zeros(n, "float32")
        b = say(" ".join(t[3] for t in parse(os.path.join(TXT, "dialog.txt"))[:14]), "C", "ru")
        bab[:min(n, len(b))] = b[:n]
        mix = sosfilt(sos, mix + noise + 0.12 * bab).astype("float32")
        stems = {k: sosfilt(sos, v).astype("float32") for k, v in stems.items()}
    peak = np.abs(mix).max()
    g = 0.8 / peak
    mix *= g
    stems = {k: v * g for k, v in stems.items()}
    r16 = lambda x: resample_poly(x, 2, 3).astype("float32")
    sf.write(os.path.join(OUT, tag + ".wav"), r16(mix), 16000, subtype="PCM_16")
    sf.write(os.path.join(OUT, tag + ".48k.wav"), resample_poly(mix, 2, 1).astype("float32"), 48000, subtype="PCM_16")
    for k, v in stems.items():
        sf.write(os.path.join(OUT, f"{tag}.stem_{k}.wav"), r16(v), 16000, subtype="PCM_16")
    json.dump({"duration_s": round(n / SR, 2), "turns": rows}, open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, tag + ".ref.txt"), "w", encoding="utf-8").write(" ".join(r["text"] for r in sorted(rows, key=lambda r: r["start"])))
    print(f"{tag}: {n / SR / 60:.2f} min, {len(rows)} turns, speakers {sorted(stems)}, {sum(len(r['text'].split()) for r in rows)} words", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["mono", "dialog", "overlap", "mixed"])
    ap.add_argument("--noisy", action="store_true")
    a = ap.parse_args()
    for nme in a.names:
        build(nme, a.noisy)
