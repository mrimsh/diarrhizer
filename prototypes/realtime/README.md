# prototypes/realtime

Research prototypes for working with calls in real time (Telemost, Google Meet, Discord, Teams).
The findings are collected in [`docs/realtime.md`](../../docs/realtime.md).

* Nothing here is imported from `src/` and nothing here is packaged. No dependencies are added to `pyproject.toml`;
  the libraries a prototype needs are pinned in [`requirements.txt`](requirements.txt) and installed into a separate `.venv-research`; see [environment setup](../../docs/development.md).
* Recordings, runs and browser profiles go to `prototypes/realtime/_local/` (git-ignored, together with `*.wav`).
  Never record calls of other people; use your own test material and a call with yourself.

## capture_loopback/ - step B1: system loopback + microphone (Windows)

| file | what it does |
|---|---|
| `list_devices.py` | endpoints as seen by PyAudioWPatch / soundcard / sounddevice |
| `compare_libs.py` | one loopback library x scenario (`play` / `silence`) per process, prints a JSON line |
| `capture.py` | **the prototype**: records microphone and loopback into two aligned mono WAVs (+ `stats.csv`, `summary.json`) |
| `play_bg.py` | plays test speech or very quiet noise to the default output so that the loopback has a signal |
| `analyze_drift.py` | clock drift of the two tracks from `stats.csv` |
| `analyze_tracks.py` | activity of "me" / "others" and echo (loopback signal leaking into the mic track) |
| `process_loopback_check.py`, `process_loopback_control.py` | feasibility of per-process capture (`proc-tap`) incl. a negative control |

Run (from `prototypes/realtime/capture_loopback`, interpreter `..\..\..\.venv-research\Scripts\python.exe`, set `PYTHONUTF8=1`):

```
python list_devices.py
python capture.py --seconds 60 --out ..\_local\run1                  # PyAudioWPatch (default)
python capture.py --seconds 60 --out ..\_local\run1 --backend soundcard
python capture.py --seconds 120 --out ..\_local\run2 --follow-devices   # reopen streams when the default device changes
python analyze_tracks.py ..\_local\run1\mic.wav ..\_local\run1\loopback.wav
python analyze_drift.py ..\_local\run1
```

`..\_local\speech48k.wav` (test speech, 48 kHz stereo) is made from the repository's `test_speech.mp3`:
`ffmpeg -i test_speech.mp3 -ar 48000 -ac 2 prototypes/realtime/_local/speech48k.wav`.

### Checking a real client (needs a call with yourself / a second device)

1. Join a test room, make the other side play a recording of speech (or talk from a second device).
2. `python capture.py --seconds 120 --out ..\_local\client_x` while you speak a few sentences and the other side answers.
3. `python analyze_tracks.py ..\_local\client_x\mic.wav ..\_local\client_x\loopback.wav`
4. Listen to both files: the loopback must contain the other side and must not contain your own voice;
   the mic must contain your voice. Make sure the client's output device is the Windows default one
   (otherwise the loopback of the default device is silent).
