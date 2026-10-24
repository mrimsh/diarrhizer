const $ = (id) => document.getElementById(id);
const msg = (t) => ($("msg").textContent = t);
const ORIGINS = chrome.runtime.getManifest().optional_host_permissions;

(async () => {
  const s = await chrome.storage.local.get({ port: 8765, secret: "", passthrough: true });
  $("port").value = s.port; $("secret").value = s.secret; $("pt").checked = s.passthrough;
})();

$("save").onclick = async () => {
  await chrome.storage.local.set({ port: Number($("port").value) || 8765, secret: $("secret").value.trim(), passthrough: $("pt").checked });
  msg("saved");
};

$("grant").onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach((t) => t.stop());
    msg("microphone granted");
  } catch (e) { msg("microphone: " + e.name); }
};

$("probe_on").onclick = async () => {                      // runs from a click = user gesture, required by permissions.request
  const ok = await chrome.permissions.request({ origins: ORIGINS });
  if (!ok) return msg("host access was not granted");
  await chrome.scripting.unregisterContentScripts({ ids: ["dom-probe"] }).catch(() => {});
  await chrome.scripting.registerContentScripts([{ id: "dom-probe", matches: ORIGINS, js: ["dom_probe.js"], runAt: "document_idle", persistAcrossSessions: true }]);
  msg("DOM probe registered for:\n" + ORIGINS.join("\n") + "\nReload the call tab.");
};

$("probe_off").onclick = async () => {
  await chrome.scripting.unregisterContentScripts({ ids: ["dom-probe"] }).catch(() => {});
  await chrome.permissions.remove({ origins: ORIGINS });
  msg("DOM probe removed, host access revoked");
};
