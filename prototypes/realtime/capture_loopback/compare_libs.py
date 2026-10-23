"""Compare loopback capture in PyAudioWPatch vs soundcard (sounddevice has no loopback).

Usage: python compare_libs.py <lib> <scenario> [seconds]
  lib: pyaw | soundcard | sounddevice
  scenario: play (plays test speech to default output) | silence (nothing playing)
Prints one JSON line with stats; run each combination in its own process (a hang is a result too).
"""
import json, sys, threading, time
import numpy as np

lib, scenario = sys.argv[1], sys.argv[2]
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
SPEECH = "../_local/speech48k.wav"
events = []  # (t_perf, nframes, rms)

def start_playback():
    import soundfile as sf, sounddevice as sd
    data, sr = sf.read(SPEECH, dtype="float32")
    data *= 0.25
    sd.play(data, sr)  # default output device, WASAPI/MME whichever default host

def run_pyaw():
    import pyaudiowpatch as pa
    p = pa.PyAudio()
    wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
    out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    lb = next(d for d in p.get_loopback_device_info_generator() if out["name"] in d["name"])
    rate, ch = int(lb["defaultSampleRate"]), lb["maxInputChannels"]
    def cb(in_data, frames, ti, status):
        a = np.frombuffer(in_data, dtype=np.int16)
        events.append((time.perf_counter(), frames, float(np.sqrt(np.mean((a/32768.0)**2))), status))
        return (None, pa.paContinue)
    s = p.open(format=pa.paInt16, channels=ch, rate=rate, input=True, input_device_index=lb["index"],
               frames_per_buffer=rate // 20, stream_callback=cb)
    t0 = time.perf_counter()
    if scenario == "play": start_playback()
    time.sleep(SECONDS)
    s.stop_stream(); s.close(); p.terminate()
    return rate, ch, t0

def run_soundcard():
    import soundcard as sc
    spk = sc.default_speaker()
    lb = sc.get_microphone(spk.name, include_loopback=True)
    rate, ch = 48000, lb.channels
    t0 = time.perf_counter()
    if scenario == "play": start_playback()
    with lb.recorder(samplerate=rate, blocksize=rate // 20) as rec:
        end = t0 + SECONDS
        while time.perf_counter() < end:
            a = rec.record(numframes=rate // 20)
            events.append((time.perf_counter(), len(a), float(np.sqrt(np.mean(a**2))), 0))
    return rate, ch, t0

def run_sounddevice():
    import sounddevice as sd
    try:
        # no 'loopback' in WasapiSettings of the bundled PortAudio -> only the mic works here
        sd.WasapiSettings(loopback=True)
    except TypeError as e:
        raise RuntimeError("WasapiSettings has no loopback: " + str(e))

runner = {"pyaw": run_pyaw, "soundcard": run_soundcard, "sounddevice": run_sounddevice}[lib]
res = {"lib": lib, "scenario": scenario, "seconds": SECONDS}
try:
    rate, ch, t0 = runner()
    n = sum(e[1] for e in events)
    ts = [e[0] for e in events]
    gaps = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    res.update(rate=rate, channels=ch, callbacks=len(events), frames=n, audio_s=round(n / rate, 3),
               wall_s=round(SECONDS, 3), first_cb_s=round(ts[0] - t0, 3) if ts else None,
               max_gap_s=round(float(gaps.max()), 3), rms_max=round(max((e[2] for e in events), default=0), 5),
               rms_mean=round(float(np.mean([e[2] for e in events])) if events else 0, 5),
               status_flags=sorted({e[3] for e in events}))
except Exception as e:
    res.update(error=f"{type(e).__name__}: {e}", callbacks=len(events))
print(json.dumps(res, ensure_ascii=False))
