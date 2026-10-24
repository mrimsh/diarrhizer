// Service worker: starts / stops the capture sessions. The audio itself lives in the offscreen document
// (a service worker has no DOM, no AudioContext and gets suspended). State is kept in chrome.storage.session.
const DEFAULTS = { port: 8765, secret: "", passthrough: true };

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function hasOffscreen() {
  const ctxs = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  return ctxs.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture tab audio / microphone as PCM and keep playing the tab audio to the user",
  });
}

async function getState() {
  return (await chrome.storage.session.get({ sessions: {} })).sessions;
}
async function setState(sessions) {
  await chrome.storage.session.set({ sessions });
  const n = Object.keys(sessions).length;
  await chrome.action.setBadgeText({ text: n ? String(n) : "" });
  await chrome.action.setBadgeBackgroundColor({ color: "#c62828" });
}

async function start({ tabId, tab, mic }) {
  const s = await settings();
  if (!s.secret) throw new Error("secret is not set (options page)");
  const wsUrl = `ws://127.0.0.1:${s.port}/`;
  await ensureOffscreen();
  const sessions = await getState();
  const started = [];
  if (tab) {
    // Needs activeTab (the action was invoked on this tab) or a host permission for the tab's origin.
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    const r = await chrome.runtime.sendMessage({ target: "offscreen", type: "start", source: "tab", streamId, wsUrl, secret: s.secret, passthrough: s.passthrough });
    if (!r?.ok) throw new Error("tab: " + r?.error);
    sessions.tab = { tabId, since: Date.now(), sampleRate: r.sampleRate };
    started.push("tab");
  }
  if (mic) {
    const r = await chrome.runtime.sendMessage({ target: "offscreen", type: "start", source: "mic", wsUrl, secret: s.secret });
    if (!r?.ok) throw new Error("mic: " + r?.error);
    sessions.mic = { since: Date.now(), sampleRate: r.sampleRate };
    started.push("mic");
  }
  await setState(sessions);
  return started;
}

async function stop(which) {
  const sessions = await getState();
  for (const src of which ? [which] : ["tab", "mic"]) {
    if (sessions[src]) {
      await chrome.runtime.sendMessage({ target: "offscreen", type: "stop", source: src }).catch(() => {});
      delete sessions[src];
    }
  }
  await setState(sessions);
  if (!Object.keys(sessions).length && (await hasOffscreen())) await chrome.offscreen.closeDocument();
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.target === "offscreen") return;           // addressed to the offscreen document
  (async () => {
    try {
      if (msg.cmd === "start") sendResponse({ ok: true, started: await start(msg) });
      else if (msg.cmd === "stop") { await stop(msg.source); sendResponse({ ok: true }); }
      else if (msg.cmd === "state") sendResponse({ ok: true, sessions: await getState() });
      else if (msg.type === "ended") {                 // the offscreen document tells us a stream ended (tab closed, device lost)
        const sessions = await getState(); delete sessions[msg.source]; await setState(sessions); sendResponse({ ok: true });
      } else if (msg.type === "dom") { await relayDom(msg.event); sendResponse({ ok: true }); }
    } catch (e) {
      sendResponse({ ok: false, error: String(e && e.message || e) });
    }
  })();
  return true;
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  const sessions = await getState();
  if (sessions.tab && sessions.tab.tabId === tabId) await stop("tab");
});

// ---- DOM probe events -> receiver (a content script must not open its own socket: page CSP, mixed content)
let domWs = null, domReady = null;
async function relayDom(event) {
  const s = await settings();
  if (!domWs || domWs.readyState > 1) {
    domReady = new Promise((resolve) => {
      domWs = new WebSocket(`ws://127.0.0.1:${s.port}/`);
      domWs.onopen = () => { domWs.send(JSON.stringify({ secret: s.secret, source: "dom" })); resolve(); };
      domWs.onerror = () => resolve();
    });
  }
  await domReady;
  if (domWs.readyState === 1) domWs.send(JSON.stringify(event));
}

// Keyboard shortcuts: invoking the extension this way grants activeTab for the current tab, just like a toolbar click.
chrome.commands.onCommand.addListener(async (command, tab) => {
  try {
    if (command === "capture-tab-and-mic") {
      const started = await start({ tabId: tab.id, tab: true, mic: true });
      await chrome.storage.session.set({ lastCommand: { command, ok: true, started, at: Date.now() } });
    } else if (command === "stop-capture") {
      await stop();
      await chrome.storage.session.set({ lastCommand: { command, ok: true, at: Date.now() } });
    }
  } catch (e) {
    await chrome.storage.session.set({ lastCommand: { command, ok: false, error: String((e && e.message) || e), at: Date.now() } });
  }
});
