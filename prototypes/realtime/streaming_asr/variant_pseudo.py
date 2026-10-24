"""Variant A: pseudo-stream. Silero VAD cuts the live audio into utterances, faster-whisper decodes each utterance when it ends.

    python variant_pseudo.py <material> [--model large-v3] [--compute float16] [--silence-ms 500] [--max-utt 15]
                             [--beam 5] [--lang ru|auto] [--prompt] [--speed 1.0] [--out run.json]

Latency of a phrase = VAD end-of-speech wait (--silence-ms) + queue wait + decode time. Nothing is shown while a person
is still talking (there are no partial results), a long monologue is cut by force every --max-utt seconds.
"""
import argparse, os, sys, time
import numpy as np
import torch
from stream_lib import Monitor, load16, run_stream, save, SR

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("material"); ap.add_argument("--model", default="large-v3"); ap.add_argument("--compute", default="float16")
ap.add_argument("--silence-ms", type=int, default=500); ap.add_argument("--max-utt", type=float, default=15.0)
ap.add_argument("--beam", type=int, default=5); ap.add_argument("--lang", default="ru"); ap.add_argument("--prompt", action="store_true")
ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--out"); ap.add_argument("--vad-thr", type=float, default=0.5)
ap.add_argument("--device", default="cuda")
ap.add_argument("--adaptive", action="store_true", help="end-of-utterance silence shrinks from --silence-ms to --silence-min-ms between 5 and 10 s of speech")
ap.add_argument("--silence-min-ms", type=int, default=150)
a = ap.parse_args()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

mon = Monitor().start()
t_load = time.perf_counter()
asr = WhisperModel(a.model, device=a.device, compute_type=a.compute)
vad_model = load_silero_vad()
load_s = round(time.perf_counter() - t_load, 1)
x = load16(a.material)
# warm-up so the first phrase does not pay for CUDA initialisation
list(asr.transcribe(x[:SR * 3], language=None if a.lang == "auto" else a.lang, beam_size=a.beam)[0])
PAD = int(0.25 * SR)


def worker(q, log):
    vad = VADIterator(vad_model, threshold=a.vad_thr, sampling_rate=SR, min_silence_duration_ms=a.silence_ms, speech_pad_ms=100)
    buf = np.zeros(0, "float32"); base = 0            # samples of the stream kept in buf, buf[0] is stream sample `base`
    start = None                                      # stream sample where the current utterance began
    prev_text = ""
    jobs = []

    def decode(s, e):
        nonlocal prev_text
        seg = buf[max(0, s - base - PAD): e - base + PAD]
        t = time.perf_counter()
        segs, _ = asr.transcribe(seg, language=None if a.lang == "auto" else a.lang, beam_size=a.beam, vad_filter=False,
                                 condition_on_previous_text=False, initial_prompt=prev_text[-200:] if a.prompt else None, temperature=0.0)
        text = " ".join(s_.text.strip() for s_ in segs)
        log.emit(text, s / SR, e / SR, decode_s=round(time.perf_counter() - t, 3), utt_s=round((e - s) / SR, 2))
        prev_text = text

    pending = np.zeros(0, "float32")
    last_speech = 0
    while True:
        item = q.get()
        if item is None:
            break
        t_end, c = item
        buf = np.concatenate([buf, c]); pending = np.concatenate([pending, c])
        while len(pending) >= 512:
            frame, pending = pending[:512], pending[512:]
            pos = base + len(buf) - len(pending)               # stream sample after this frame
            if a.adaptive:
                p = float(vad_model(torch.from_numpy(frame), SR))
                speech = p >= (a.vad_thr if start is None else a.vad_thr - 0.15)
                if speech:
                    last_speech = pos
                    if start is None:
                        start = max(0, pos - 512 - int(0.1 * SR))
                elif start is not None:
                    dur = (pos - start) / SR
                    need = a.silence_ms if dur <= 5 else max(a.silence_min_ms, a.silence_ms - (a.silence_ms - a.silence_min_ms) * (dur - 5) / 5)
                    if (pos - last_speech) / SR * 1000 >= need:
                        decode(start, last_speech + int(0.1 * SR)); start = None
                if start is not None and (pos - start) / SR >= a.max_utt:
                    decode(start, pos); start = pos
            else:
                r = vad(torch.from_numpy(frame), return_seconds=False)
                if r and "start" in r and start is None:
                    start = max(0, r["start"])
                if r and "end" in r and start is not None:
                    decode(start, r["end"]); start = None
                elif start is not None and (pos - start) / SR >= a.max_utt:
                    decode(start, pos); start = pos         # forced cut inside a long monologue
        if start is None and len(buf) > 5 * SR:        # keep only the tail while nobody speaks
            cut = len(buf) - 2 * SR; buf = buf[cut:]; base += cut
        elif start is not None and start - base > 0 and start - base > 30 * SR:
            cut = start - base - SR; buf = buf[cut:]; base += cut
    if start is not None:
        decode(start, base + len(buf))


log = run_stream(x, worker, speed=a.speed)
res = mon.stop()
meta = dict(variant="pseudo", material=os.path.basename(a.material), model=a.model, compute=a.compute, silence_ms=a.silence_ms,
            max_utt=a.max_utt, adaptive=a.adaptive, silence_min_ms=a.silence_min_ms, beam=a.beam, lang=a.lang, prompt=a.prompt, model_load_s=load_s, audio_s=round(len(x) / SR, 1))
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.material)), "..", "stream_runs", os.path.basename(a.material).replace(".wav", "") + ".pseudo.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
save(out, log, meta, res)
print(f"{meta['material']}: {len(log.events)} segments, wall {log.total_wall}s, audio {meta['audio_s']}s, resources {res}")
