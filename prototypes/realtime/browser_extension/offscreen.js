// Offscreen document: owns the MediaStreams, the AudioContexts and the WebSocket to the local receiver.
// One session per source ("tab" | "mic"). Packet: u32 seq | f64 context time ms | int16 LE mono PCM (50 ms @ 48 kHz).
const sessions = {};

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.target !== "offscreen") return;
  if (msg.type === "start") {
    start(msg).then(sendResponse, (e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true;
  }
  if (msg.type === "stop") { stop(msg.source, "stopped by user"); sendResponse({ ok: true }); }
});

async function start({ source, streamId, wsUrl, secret, passthrough }) {
  if (sessions[source]) stop(source, "restart");
  const constraints = source === "tab"
    ? { audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } }, video: false }
    // raw microphone: our own pipeline does its own processing, so the browser's DSP is off
    : { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const ctx = new AudioContext({ sampleRate: 48000 });
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  // A captured tab goes silent for the user unless we play it back ourselves.
  if (source === "tab" && passthrough) src.connect(ctx.destination);
  const node = new AudioWorkletNode(ctx, "pcm-capture", { numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { chunk: 2400 } });
  const mute = ctx.createGain(); mute.gain.value = 0;               // keeps the worklet in the rendering graph, outputs silence
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);

  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  await new Promise((resolve, reject) => {
    ws.onopen = () => {
      ws.send(JSON.stringify({ secret, source, sampleRate: ctx.sampleRate, channels: 1, format: "s16le", epoch_ms: Date.now(), ctx_ms: ctx.currentTime * 1000 }));
      resolve();
    };
    ws.onerror = () => reject(new Error("cannot connect to the receiver (is receiver.py running? port?)"));
  });
  const s = { source, stream, ctx, ws, seq: 0, dropped: 0, sent: 0 };
  sessions[source] = s;
  node.port.onmessage = (e) => {
    if (ws.readyState !== 1) { s.dropped++; return; }
    const pcm = new Int16Array(e.data.pcm);
    const buf = new ArrayBuffer(12 + pcm.byteLength);
    const dv = new DataView(buf);
    dv.setUint32(0, s.seq++ >>> 0, true); dv.setFloat64(4, e.data.t, true);
    new Int16Array(buf, 12).set(pcm);
    ws.send(buf); s.sent++;
  };
  ws.onclose = () => end(source, "websocket closed");
  stream.getAudioTracks()[0].addEventListener("ended", () => end(source, "track ended (tab closed / device lost)"));
  return { ok: true, sampleRate: ctx.sampleRate };
}

function stop(source, why) {
  const s = sessions[source];
  if (!s) return;
  delete sessions[source];
  try { s.ws.close(1000, why); } catch {}
  s.stream.getTracks().forEach((t) => t.stop());
  s.ctx.close();
}

function end(source, why) {
  if (!sessions[source]) return;
  stop(source, why);
  chrome.runtime.sendMessage({ type: "ended", source, why });
}
