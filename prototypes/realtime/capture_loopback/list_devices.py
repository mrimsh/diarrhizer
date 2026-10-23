"""Enumerate audio endpoints as seen by PyAudioWPatch, soundcard and sounddevice."""
import sys

def pyaw():
    import pyaudiowpatch as pa
    p = pa.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        print("[PyAudioWPatch] default out:", p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"])
        print("[PyAudioWPatch] default in :", p.get_device_info_by_index(wasapi["defaultInputDevice"])["name"])
        for lb in p.get_loopback_device_info_generator():
            print("  loopback:", lb["index"], lb["name"], int(lb["defaultSampleRate"]), lb["maxInputChannels"])
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == wasapi["index"] and not d.get("isLoopbackDevice") and d["maxInputChannels"] > 0:
                print("  mic     :", i, d["name"], int(d["defaultSampleRate"]), d["maxInputChannels"])
    finally:
        p.terminate()

def sc():
    import soundcard as s
    print("[soundcard] default speaker:", s.default_speaker().name)
    print("[soundcard] default mic    :", s.default_microphone().name)
    for m in s.all_microphones(include_loopback=True):
        print("  ", "loopback" if m.isloopback else "mic     ", m.name)

def sd():
    import sounddevice as d
    print("[sounddevice] hostapis:", [h["name"] for h in d.query_hostapis()])
    print("  has WasapiSettings loopback arg:", "loopback" in d.WasapiSettings.__init__.__code__.co_varnames)
    print("  default:", d.default.device)

for f in (pyaw, sc, sd):
    try:
        f()
    except Exception as e:
        print(f"{f.__name__} FAILED: {type(e).__name__}: {e}")
