const st = document.getElementById("st");

async function refresh() {
  const r = await chrome.runtime.sendMessage({ cmd: "state" });
  const s = (r && r.sessions) || {};
  st.textContent = Object.keys(s).length ? "capturing: " + Object.keys(s).join(", ") : "idle";
}

async function go(opts) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const r = await chrome.runtime.sendMessage({ cmd: "start", tabId: tab.id, ...opts });
  st.textContent = r.ok ? "started: " + r.started.join(", ") : "error: " + r.error;
}

document.getElementById("tab").onclick = () => go({ tab: true, mic: false });
document.getElementById("mic").onclick = () => go({ tab: false, mic: true });
document.getElementById("both").onclick = () => go({ tab: true, mic: true });
document.getElementById("stop").onclick = async () => { await chrome.runtime.sendMessage({ cmd: "stop" }); refresh(); };
document.getElementById("opt").onclick = () => chrome.runtime.openOptionsPage();
refresh();
