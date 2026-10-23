"""Feasibility check of Windows process loopback (proc-tap) vs system loopback.

Starts Chrome with a throw-away profile playing a looping speech wav, then for 12 s records
  (a) the Chrome process tree via proc-tap (WASAPI process loopback, native C++ extension),
  (b) the whole system mix via PyAudioWPatch,
while a *different* process plays a 1 kHz tone between t=4 s and t=8 s (stands for a notification /
another app). Reports per-window RMS and the 1 kHz tone level in each capture.
"""
import os, subprocess, sys, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.abspath(os.path.join(HERE, "..", "_local"))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
RATE = 48000

def tone_in_other_process():
    code = ("import numpy as np, sounddevice as sd, time\n"
            "t=np.arange(int(4*48000))/48000\n"
            "x=(0.2*np.sin(2*np.pi*1000*t)).astype('float32')\n"
            "time.sleep(4); sd.play(np.stack([x,x],1),48000); sd.wait()\n")
    return subprocess.Popen([sys.executable, "-c", code])

def tone_level(x, lo=950, hi=1050):
    """fraction of spectral energy in 950-1050 Hz"""
    if len(x) < RATE:
        return float("nan")
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return float(X[(f >= lo) & (f <= hi)].sum() / X.sum())

def windows(x, name):
    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")
    pre, mid, post = x[RATE:4 * RATE], x[int(4.5 * RATE):int(7.5 * RATE)], x[int(8.5 * RATE):]
    print(f"{name:14s} rms before/during/after tone: {rms(pre):.4f} / {rms(mid):.4f} / {rms(post):.4f}   "
          f"1kHz energy share during tone: {tone_level(mid):.3f}   before: {tone_level(pre):.4f}")

def main():
    from proctap import ProcessAudioCapture
    import pyaudiowpatch as pa
    prof = os.path.join(LOCAL, "chrome_profile")
    ch = subprocess.Popen([CHROME, f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                           "--autoplay-policy=no-user-gesture-required", "--new-window",
                           "file:///" + os.path.join(LOCAL, "chrome_test.html").replace("\\", "/")])
    print("chrome pid", ch.pid)
    try:
        time.sleep(6)
        tap_chunks, sys_chunks = [], []
        cap = ProcessAudioCapture(ch.pid, on_data=lambda pcm, n: tap_chunks.append(np.frombuffer(pcm, np.float32).reshape(-1, 2).mean(1)))
        print("proc-tap format:", cap.get_format())
        p = pa.PyAudio(); lb = p.get_default_wasapi_loopback()
        def cb(d, n, ti, st):
            sys_chunks.append(np.frombuffer(d, np.float32).reshape(-1, lb["maxInputChannels"]).mean(1)); return (None, pa.paContinue)
        s = p.open(format=pa.paFloat32, channels=lb["maxInputChannels"], rate=int(lb["defaultSampleRate"]), input=True,
                   input_device_index=lb["index"], frames_per_buffer=2400, stream_callback=cb)
        tp = tone_in_other_process()
        t0 = time.perf_counter(); cap.start()
        time.sleep(12)
        cap.stop(); cap.close(); s.stop_stream(); s.close(); p.terminate(); tp.wait()
        tap = np.concatenate(tap_chunks) if tap_chunks else np.zeros(0, np.float32)
        sysm = np.concatenate(sys_chunks)
        print(f"proc-tap frames: {len(tap)} ({len(tap)/RATE:.2f}s), chunks={len(tap_chunks)}; system frames: {len(sysm)} ({len(sysm)/RATE:.2f}s)")
        windows(tap, "proc-tap(chrome)")
        windows(sysm, "system loopback")
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(ch.pid)], capture_output=True)

main()
