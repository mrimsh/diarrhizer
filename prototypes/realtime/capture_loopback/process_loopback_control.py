"""Negative control: proc-tap on a PID that renders nothing (this very process) while Chrome plays.
If a build honours process filtering the result must be silence; if RMS equals the system mix, filtering is not applied."""
import os, subprocess, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
LOCAL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local"))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
from proctap import ProcessAudioCapture
ch = subprocess.Popen([CHROME, f"--user-data-dir={os.path.join(LOCAL,'chrome_profile')}", "--no-first-run", "--autoplay-policy=no-user-gesture-required",
                       "file:///" + os.path.join(LOCAL, "chrome_test.html").replace("\\", "/")])
try:
    time.sleep(6)
    for label, pid in (("this python (silent)", os.getpid()), ("chrome main", ch.pid)):
        buf = []
        cap = ProcessAudioCapture(pid, on_data=lambda pcm, n: buf.append(np.frombuffer(pcm, np.float32)))
        cap.start(); time.sleep(5); cap.stop(); cap.close()
        x = np.concatenate(buf) if buf else np.zeros(1, np.float32)
        print(f"{label:22s} pid={pid} samples={len(x)} rms={np.sqrt(np.mean(x**2)):.5f} peak={np.abs(x).max():.4f}")
finally:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(ch.pid)], capture_output=True)
