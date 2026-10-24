"""Third-party GPU load for the 'ASR while something else uses the GPU' test.

    python gpu_load.py <seconds> [--duty 0.5] [--nvenc]

* compute: a CUDA matmul loop that keeps the GPU busy for `duty` of every second (stands for a game / video call effects / a model)
* --nvenc: additionally ffmpeg h264_nvenc encodes a synthetic 1080p60 stream to NUL (stands for screen recording; uses the video
  encoder engine and memory bandwidth, not the CUDA cores)
Prints the load it produced; stop with Ctrl+C or when the time is over.
"""
import argparse, subprocess, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("seconds", type=float); ap.add_argument("--duty", type=float, default=0.5); ap.add_argument("--nvenc", action="store_true")
a = ap.parse_args()
enc = None
if a.nvenc:
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-re", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=60", "-t", str(a.seconds),
                            "-c:v", "h264_nvenc", "-b:v", "12M", "-f", "null", "-"], stdout=subprocess.DEVNULL)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
t_end = time.time() + a.seconds
while time.time() < t_end:
    t0 = time.time()
    while time.time() - t0 < a.duty:
        for _ in range(8):
            (x @ x).sum().item()
    time.sleep(max(0.0, 1 - (time.time() - t0)))
if enc:
    enc.terminate()
