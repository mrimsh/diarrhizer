# prototypes/realtime/browser_extension

Step B3, approach B: a Chrome/Edge extension (Manifest V3) that sends the audio of the current tab and/or the microphone to a
receiver on `127.0.0.1`. A research prototype: not published, not packaged, nothing here is imported from `src/`.
Findings: [`docs/realtime.md`](../../../docs/realtime.md), sections "Подходы B–D" and "Расширение браузера".

| file | role |
|---|---|
| `manifest.json` | MV3; install-time permissions `tabCapture`, `offscreen`, `activeTab`, `storage`, `scripting`; call sites are only **optional** host permissions |
| `background.js` | service worker: start / stop, badge, tab-closed cleanup, relays DOM-probe events |
| `offscreen.html/.js`, `pcm-worklet.js` | the audio side: `getUserMedia` (tab stream id or microphone) -> AudioWorklet -> 50 ms Int16 packets -> WebSocket; the tab keeps playing to the user (passthrough) |
| `popup.html/.js` | buttons: tab audio / microphone / both / stop |
| `options.html/.js` | receiver port + secret, one-time microphone grant, enable/disable the DOM probe (asks Chrome for the host permissions at that moment) |
| `dom_probe.js` | feasibility probe of "participants" / "active speaker" in the page DOM: counts and timings per element signature, never names |
| `receiver.py` | the local receiver, standard library only: 127.0.0.1, Host + Origin check, shared secret, writes `<source>_<n>.wav` + stats |

## Use it by hand

1. `python receiver.py` (interpreter `..\..\..\.venvcuda\Scripts\python.exe`, run from this directory). The first start creates
   `..\_local\receiver_secret.txt` (the secret is not printed).
2. Chrome/Edge -> `chrome://extensions` -> Developer mode -> **Load unpacked** -> this folder.
3. Extension options: paste the secret, press "Grant microphone", save.
4. Open the call tab, click the extension button -> "Tab audio" (or "+ microphone"). Files appear in `..\_local\ext_run\`.
5. Optional: options -> "Enable on call sites" (Chrome shows its own host-access prompt) and reload the call tab; DOM events go to `dom.jsonl`.

Branded Chrome 137+ ignores `--load-extension` (announced 2025-04-04 on the chromium-extensions list), so automation needs
Chromium / Chrome for Testing / Edge (Edge 155 loads it, but `tabCapture` still needs a real click or shortcut by the user), or a manual "Load unpacked" once per profile.

## Safety notes

* The receiver accepts only `chrome-extension://...` origins, only `127.0.0.1`/`localhost` Host headers and only a client that
  knows the secret; web pages cannot connect (checked: web origin, no origin, wrong host, wrong secret are all refused).
* The capture is started only by a click on the extension (activeTab) - there is no background recording.
* Recording other people needs their notice/consent, see the legal note in `docs/realtime.md`.
