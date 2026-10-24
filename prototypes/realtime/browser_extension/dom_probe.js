// DOM probe (feasibility only): finds out whether a call page exposes "who is in the call" and "who speaks now" in its DOM and how
// stable those elements are. It reports counts and timings - never names or text. Participant identity is a 32-bit hash of the
// element's id/label, only to see that the *same* tile changes (so events of one tile can be correlated with the audio offline).
// Selectors of these sites are not documented and change without notice, so instead of hard-coding them the probe discovers
// candidate elements by attribute / class vocabulary and keeps statistics per "signature" (tag + attribute + matched token).
(() => {
  if (window.__diarrhizerProbe) return;
  window.__diarrhizerProbe = true;
  const HOST = location.host;
  const WORDS = /(participant|speaker|speaking|roster|voice.?user|active.?speaker|member|tile|avatar)/i;
  const send = (event) => chrome.runtime.sendMessage({ type: "dom", event: { host: HOST, t: Date.now(), ...event } }).catch(() => {});
  const hash = (s) => { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0; return h >>> 0; };

  const sigsOf = (el) => {
    const out = [];
    for (const a of el.attributes) {
      if (a.name === "class") {
        for (const c of a.value.split(/\s+/)) if (WORDS.test(c)) out.push(`${el.tagName.toLowerCase()}.class~${c.replace(/[A-Za-z0-9_-]{6,}$/, (m) => (/\d|[A-Z]/.test(m) && m.length > 8 ? "<hash>" : m))}`);
      } else if (WORDS.test(a.name)) out.push(`${el.tagName.toLowerCase()}[${a.name}]`);
      else if (/^(data-|aria-|role)/.test(a.name) && WORDS.test(a.value) && a.value.length < 40) out.push(`${el.tagName.toLowerCase()}[${a.name}=${a.value}]`);
    }
    return out;
  };

  const stats = new Map();           // sig -> {count, muts, nonEmpty, ids:Set}
  const get = (sig) => stats.get(sig) || stats.set(sig, { count: 0, muts: 0, nonEmpty: 0, ids: new Set(), attrs: {} }).get(sig);

  function scan() {
    for (const v of stats.values()) { v.count = 0; v.nonEmpty = 0; v.ids.clear(); }
    for (const el of document.querySelectorAll("*")) {
      if (!el.attributes.length) continue;
      for (const sig of sigsOf(el)) {
        const v = get(sig); v.count++;
        if ((el.textContent || "").trim()) v.nonEmpty++;
        v.ids.add(hash(el.getAttribute("data-participant-id") || el.getAttribute("aria-label") || el.id || String(v.count)));
      }
    }
  }

  const mo = new MutationObserver((records) => {
    for (const r of records) {
      const el = r.target.nodeType === 1 ? r.target : r.target.parentElement;
      if (!el || !el.attributes) continue;
      for (const sig of sigsOf(el)) {
        const v = get(sig); v.muts++;
        if (r.type === "attributes") v.attrs[r.attributeName] = (v.attrs[r.attributeName] || 0) + 1;
        // per-tile change event: lets the offline step line up "tile X changed" with the audio activity of the tab track
        send({ kind: "change", sig, tile: hash(el.getAttribute("data-participant-id") || el.getAttribute("aria-label") || el.id || sig), attr: r.attributeName || "text" });
      }
    }
  });
  mo.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });

  setInterval(() => {
    scan();
    const rows = [...stats.entries()].filter(([, v]) => v.count || v.muts)
      .sort((a, b) => b[1].muts - a[1].muts || b[1].count - a[1].count).slice(0, 40)
      .map(([sig, v]) => ({ sig, count: v.count, tiles: v.ids.size, nonEmpty: v.nonEmpty, muts5s: v.muts, topAttr: Object.entries(v.attrs).sort((a, b) => b[1] - a[1])[0]?.[0] || null }));
    for (const v of stats.values()) { v.muts = 0; v.attrs = {}; }
    send({ kind: "summary", sigs: rows, title_len: document.title.length });
  }, 5000);
  send({ kind: "start", ua: navigator.userAgent.slice(0, 80) });
})();
