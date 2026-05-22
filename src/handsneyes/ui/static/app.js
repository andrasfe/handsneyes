"use strict";

// ── DOM refs ───────────────────────────────────────────────────
const $frame = document.getElementById("frame");
const $frameEmpty = document.getElementById("frame-empty");
const $frameMeta = document.getElementById("frame-meta");
const $btnPrev = document.getElementById("btn-prev");
const $btnNext = document.getElementById("btn-next");
const $btnLive = document.getElementById("btn-live");
const $btnRefresh = document.getElementById("btn-refresh");

const $chatLog = document.getElementById("chat-log");
const $chatForm = document.getElementById("chat-form");
const $chatInput = document.getElementById("chat-input");
const $btnSend = document.getElementById("btn-send");
const $optDryRun = document.getElementById("opt-dry-run");
const $optNoFocus = document.getElementById("opt-no-focus");
const $optPlatform = document.getElementById("opt-platform");
const $optVault = document.getElementById("opt-vault");
const $btnLock = document.getElementById("btn-lock");
const $btnUnlock = document.getElementById("btn-unlock");
const $btnExecScript = document.getElementById("btn-exec-script");
const $execModal = document.getElementById("exec-modal");
const $execScriptBody = document.getElementById("exec-script-body");
const $execModalClose = document.getElementById("exec-modal-close");
const $execModalCancel = document.getElementById("exec-modal-cancel");
const $execModalRun = document.getElementById("exec-modal-run");

const $logView = document.getElementById("log-view");
const $btnClearLogs = document.getElementById("btn-clear-logs");

const $clickMarker = document.getElementById("click-marker");
const $frameBusy = document.getElementById("frame-busy");
const $frameBusyLabel = $frameBusy
  ? $frameBusy.querySelector(".frame-busy-label")
  : null;
const $optClickToMove = document.getElementById("opt-click-to-move");
const $btnMouseLeft = document.getElementById("btn-mouse-left");
const $btnMouseMiddle = document.getElementById("btn-mouse-middle");
const $btnMouseRight = document.getElementById("btn-mouse-right");

// ── state ──────────────────────────────────────────────────────
const state = {
  liveMode: true,
  currentId: null,
  knownIds: [],          // ascending order, fed by /api/frames listings
  busy: false,
  logsAtBottom: true,
  globalSrc: null,
  runChats: new Map(),   // run_id -> chat <li>
  // True after the operator has performed at least one ``click_at``
  // this session. Until then we warn on the first keystroke that
  // typing-before-clicking is the most common reason "the host
  // didn't see what I typed" — the host window is probably not
  // focused. Persists in sessionStorage so a tab reload doesn't
  // re-show the banner if the user has already acknowledged it.
  hadClickAt: false,
  // True while the mouse pointer is over the screenshot. Used to
  // light up the keyboard-passthrough indicator so the operator
  // knows their keypresses will land on the host.
  mouseOverFrame: false,
  warnedPassthrough: (() => {
    try {
      return sessionStorage.getItem("te-warned-passthrough") === "1";
    } catch (_) { return false; }
  })(),
};

// ── frame fetching ─────────────────────────────────────────────
$frame.addEventListener("load", () => {
  $frame.classList.remove("empty");
  $frameEmpty.style.display = "none";
});
// One-shot resync guard so we don't spin forever if every frame
// 404s (e.g. server has zero frames).
let _imageErrorResyncing = false;
$frame.addEventListener("error", async () => {
  $frame.classList.add("empty");
  $frameEmpty.style.display = "";
  $frameEmpty.textContent =
    `image load failed for id ${state.currentId} — refreshing frame list…`;
  if (_imageErrorResyncing) return;
  _imageErrorResyncing = true;
  try {
    // Stale id: the FrameStore was rebuilt (cc restart, ring-buffer
    // eviction, etc.) so our cached id no longer maps to anything on
    // disk. Re-list from the server and jump to the new latest.
    const before = state.currentId;
    state.currentId = null;     // force setFrameSrc to actually swap
    await refreshKnownIds();
    if (state.currentId === before || state.currentId === null) {
      $frameEmpty.textContent =
        "no frames available — send an instruction.";
    }
  } finally {
    _imageErrorResyncing = false;
  }
});

function setFrameSrc(id) {
  if (id === null || id === undefined) {
    $frame.classList.add("empty");
    $frameEmpty.style.display = "";
    $frameMeta.textContent = "—";
    return;
  }
  // Cache-bust per id so a re-render picks up a fresh fetch even if
  // the URL is the same.
  $frame.src = `/api/frames/${id}?t=${Date.now()}`;
  state.currentId = id;
  updateFrameMeta();
}

function updateFrameMeta() {
  const id = state.currentId;
  if (id === null) {
    $frameMeta.textContent = "—";
    return;
  }
  const idx = state.knownIds.indexOf(id);
  const total = state.knownIds.length;
  if (idx >= 0 && total > 0) {
    $frameMeta.textContent =
      `${idx + 1}/${total}` + (state.liveMode ? "  • live" : "");
  } else {
    $frameMeta.textContent = state.liveMode ? "live" : "history";
  }
  $btnPrev.disabled = idx <= 0;
  $btnNext.disabled = idx < 0 || idx >= total - 1;
  $btnLive.classList.toggle("active", state.liveMode);
}

async function refreshKnownIds() {
  // Fetch up to 500 frames (newest-first), reverse to ascending.
  try {
    const r = await fetch("/api/frames?limit=500");
    if (!r.ok) {
      $frameEmpty.textContent =
        `frames API returned ${r.status} ${r.statusText}`;
      return;
    }
    const data = await r.json();
    const ids = (data.items || []).map(m => m.id).reverse();
    state.knownIds = ids;
    if (ids.length === 0) {
      $frameEmpty.textContent = "No frames yet — send an instruction.";
      return;
    }
    if (state.liveMode) {
      const latest = ids[ids.length - 1];
      if (latest !== state.currentId) setFrameSrc(latest);
      else updateFrameMeta();
    } else {
      updateFrameMeta();
    }
  } catch (e) {
    $frameEmpty.textContent = "frames API failed: " + (e && e.message);
  }
}

async function pollLatest() {
  // Long-poll: server holds for up to 10s if no change.
  while (true) {
    try {
      const since = state.currentId ?? "";
      const url = `/api/frames/latest?wait=1&since=${encodeURIComponent(since)}`;
      const r = await fetch(url);
      if (r.ok) {
        const { item } = await r.json();
        if (item && item.id !== state.currentId) {
          if (!state.knownIds.includes(item.id)) {
            state.knownIds.push(item.id);
            // Cap mirror of server's ring buffer.
            if (state.knownIds.length > 600) {
              state.knownIds.splice(0, state.knownIds.length - 500);
            }
          }
          if (state.liveMode) setFrameSrc(item.id);
          else updateFrameMeta();
        }
      }
    } catch (_) {
      await sleep(1000);
    }
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ── frame controls ─────────────────────────────────────────────
$btnPrev.addEventListener("click", async () => {
  if (state.knownIds.length === 0) return;
  const idx = state.knownIds.indexOf(state.currentId);
  if (idx > 0) {
    state.liveMode = false;
    setFrameSrc(state.knownIds[idx - 1]);
  } else if (idx < 0 && state.knownIds.length > 0) {
    state.liveMode = false;
    setFrameSrc(state.knownIds[state.knownIds.length - 1]);
  }
});

$btnNext.addEventListener("click", () => {
  if (state.knownIds.length === 0) return;
  const idx = state.knownIds.indexOf(state.currentId);
  if (idx >= 0 && idx < state.knownIds.length - 1) {
    const next = state.knownIds[idx + 1];
    setFrameSrc(next);
    if (idx + 1 === state.knownIds.length - 1) state.liveMode = true;
    updateFrameMeta();
  }
});

$btnLive.addEventListener("click", () => {
  state.liveMode = true;
  if (state.knownIds.length > 0) {
    setFrameSrc(state.knownIds[state.knownIds.length - 1]);
  } else {
    updateFrameMeta();
  }
});

// Manual refresh: fire /api/snapshot and let the poll-until-stable
// loop on the server stream new frames into the watch dir. We jump
// to live so the long-poll surfaces them as they land.
$btnRefresh?.addEventListener("click", async () => {
  if ($btnRefresh.disabled) return;
  $btnRefresh.disabled = true;
  $btnRefresh.classList.add("spinning");
  state.liveMode = true;
  try {
    const res = await fetch("/api/snapshot", { method: "POST" });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      console.warn("refresh failed:", res.status, t);
    }
  } catch (e) {
    console.warn("refresh fetch error:", e);
  } finally {
    $btnRefresh.disabled = false;
    $btnRefresh.classList.remove("spinning");
  }
});

// ── active refresh: idle auto-capture every minute ─────────────
// When checked, POST /api/snapshot?dedup=1 every ACTIVE_REFRESH_MS.
// The server's dedup mode only persists a frame if it differs from
// the most recent stored frame, so an unchanged screen produces zero
// writes (but still pays for the webcam grab — that's the cost we
// warn about). Choice persists in localStorage so a reload picks
// the same setting back up. Tab/window hidden → pause to avoid
// burning the camera when nothing's watching.
const $optActiveRefresh = document.getElementById("opt-active-refresh");
const ACTIVE_REFRESH_KEY = "cc.activeRefresh";
const ACTIVE_REFRESH_MS = 60_000;
let activeRefreshTimer = null;
let activeRefreshInFlight = false;

async function activeRefreshTick() {
  if (activeRefreshInFlight) return;
  if (document.hidden) return;
  activeRefreshInFlight = true;
  try {
    await fetch("/api/snapshot?dedup=1", { method: "POST" });
  } catch (e) {
    console.warn("active-refresh tick failed:", e);
  } finally {
    activeRefreshInFlight = false;
  }
}
function startActiveRefresh() {
  if (activeRefreshTimer != null) return;
  $optActiveRefresh?.parentElement?.classList.add("armed");
  // First tick is immediate so the user sees the loop is alive,
  // subsequent ones honour the cadence.
  activeRefreshTick();
  activeRefreshTimer = setInterval(activeRefreshTick, ACTIVE_REFRESH_MS);
}
function stopActiveRefresh() {
  if (activeRefreshTimer == null) return;
  clearInterval(activeRefreshTimer);
  activeRefreshTimer = null;
  $optActiveRefresh?.parentElement?.classList.remove("armed");
}
$optActiveRefresh?.addEventListener("change", () => {
  const on = !!$optActiveRefresh.checked;
  if (on) {
    const ok = window.confirm(
      "Active Refresh will grab a webcam frame every 60 seconds " +
      "even when you're idle. This keeps the camera awake and the " +
      "host's UI continuously analysed — resource intensive on " +
      "battery / Pi. Identical frames are discarded server-side.\n\n" +
      "Enable it?"
    );
    if (!ok) {
      $optActiveRefresh.checked = false;
      return;
    }
    try { localStorage.setItem(ACTIVE_REFRESH_KEY, "1"); } catch (_) {}
    startActiveRefresh();
  } else {
    try { localStorage.removeItem(ACTIVE_REFRESH_KEY); } catch (_) {}
    stopActiveRefresh();
  }
});
// Restore previous setting on load — but DON'T re-prompt, the user
// already opted in. Restoring is silent.
try {
  if (localStorage.getItem(ACTIVE_REFRESH_KEY) === "1" && $optActiveRefresh) {
    $optActiveRefresh.checked = true;
    startActiveRefresh();
  }
} catch (_) {}
// Pause/resume on tab visibility so we don't burn the webcam in a
// hidden tab.
document.addEventListener("visibilitychange", () => {
  if (!$optActiveRefresh?.checked) return;
  if (document.hidden) stopActiveRefresh();
  else startActiveRefresh();
});

// ── chat / runs ────────────────────────────────────────────────
function appendChat({ runId, intent, status, reason }) {
  let li = state.runChats.get(runId);
  if (!li) {
    li = document.createElement("li");
    li.dataset.runId = runId;
    const intentEl = document.createElement("div");
    intentEl.className = "intent";
    intentEl.textContent = intent;
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    li.append(intentEl, metaEl);
    $chatLog.appendChild(li);
    state.runChats.set(runId, li);
  }
  const meta = li.querySelector(".meta");
  meta.textContent = `${runId.slice(0, 6)} · ${status}` +
    (reason ? ` — ${reason}` : "");
  li.classList.toggle("outcome-ok", status === "succeeded");
  li.classList.toggle("outcome-bad", status === "failed");
  li.classList.toggle("outcome-err", status === "error");
  $chatLog.scrollTop = $chatLog.scrollHeight;
}

async function pollRunStatus(runId) {
  while (true) {
    try {
      const r = await fetch(`/api/runs/${runId}`);
      if (!r.ok) return;
      const rec = await r.json();
      appendChat({
        runId, intent: rec.intent,
        status: rec.status, reason: rec.reason,
      });
      if (rec.status !== "running" && rec.status !== "pending") {
        state.busy = false;
        $btnSend.disabled = false;
        if ($btnLock) $btnLock.disabled = false;
        if ($btnUnlock) $btnUnlock.disabled = false;
        if ($btnExecScript) $btnExecScript.disabled = false;
        return;
      }
    } catch (_) {}
    await sleep(700);
  }
}

$chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const intent = $chatInput.value.trim();
  if (!intent || state.busy) return;
  state.busy = true;
  $btnSend.disabled = true;
  const body = {
    intent,
    no_focus: $optNoFocus.checked,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
  };
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.status === 409) {
      appendSystemLog("ERROR", "another run is already in progress");
      state.busy = false;
      $btnSend.disabled = false;
      if ($btnLock) $btnLock.disabled = false;
      if ($btnUnlock) $btnUnlock.disabled = false;
      if ($btnExecScript) $btnExecScript.disabled = false;
      return;
    }
    if (!r.ok) {
      const t = await r.text();
      appendSystemLog("ERROR", `run rejected: ${t}`);
      state.busy = false;
      $btnSend.disabled = false;
      if ($btnLock) $btnLock.disabled = false;
      if ($btnUnlock) $btnUnlock.disabled = false;
      if ($btnExecScript) $btnExecScript.disabled = false;
      return;
    }
    const rec = await r.json();
    $chatInput.value = "";
    appendChat({
      runId: rec.run_id, intent: rec.intent, status: rec.status,
    });
    pollRunStatus(rec.run_id);
  } catch (err) {
    appendSystemLog("ERROR", String(err));
    state.busy = false;
    $btnSend.disabled = false;
    if ($btnLock) $btnLock.disabled = false;
    if ($btnUnlock) $btnUnlock.disabled = false;
    if ($btnExecScript) $btnExecScript.disabled = false;
  }
});

// ── quick actions: Lock / Unlock ──────────────────────────────
async function startRun(body, fallbackIntent) {
  if (state.busy) return;
  state.busy = true;
  $btnSend.disabled = true;
  $btnLock.disabled = true;
  $btnUnlock.disabled = true;
  if ($btnExecScript) $btnExecScript.disabled = true;
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.status === 409) {
      appendSystemLog("ERROR", "another run is already in progress");
      return;
    }
    if (!r.ok) {
      const t = await r.text();
      appendSystemLog("ERROR", `run rejected: ${t}`);
      return;
    }
    const rec = await r.json();
    appendChat({
      runId: rec.run_id,
      intent: rec.intent || fallbackIntent,
      status: rec.status,
    });
    pollRunStatus(rec.run_id);
  } catch (err) {
    appendSystemLog("ERROR", String(err));
  } finally {
    // pollRunStatus re-enables Send on terminal status, but for the
    // synchronous error paths above we have to release the buttons
    // here. pollRunStatus also re-enables, so calling twice is fine.
    if (state.busy) {
      // poll is still running; leave buttons disabled.
    } else {
      $btnSend.disabled = false;
      $btnLock.disabled = false;
      $btnUnlock.disabled = false;
    }
  }
}

$btnLock.addEventListener("click", () => {
  startRun({
    intent: "lock the screen",
    no_focus: true,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
  }, "lock the screen");
});

// Cache the vault passphrase for the cc session so the operator
// types it once per cc restart, not once per Unlock click. Lives
// only in this tab's JS heap — never persisted, never sent over the
// wire except inline with each /api/run.
let _vaultPassphraseCache = "";

$btnUnlock.addEventListener("click", () => {
  const vault = ($optVault.value || "").trim();
  if (!vault) {
    appendSystemLog("ERROR", "Unlock requires a vault entry name");
    return;
  }
  // If the operator hasn't supplied a passphrase yet this session,
  // prompt for it. Empty input means "just type the password
  // directly" — we fall through to a second prompt.
  if (!_vaultPassphraseCache) {
    const vp = window.prompt(
      `Vault passphrase for entry "${vault}"\n` +
      "(Cancel to type the unlock password directly instead.)",
    );
    if (vp === null) {
      // Operator chose direct-password path.
      const pw = window.prompt(
        `Unlock password for "${vault}"\n` +
        "(Stored only for this run.)",
      );
      if (!pw) return;
      startRun({
        intent: "unlock the screen",
        no_focus: true,
        dry_run: $optDryRun.checked,
        platform: $optPlatform.value,
        vault,
        password: pw,
      }, "unlock the screen");
      return;
    }
    _vaultPassphraseCache = vp;
  }
  startRun({
    intent: "unlock the screen",
    no_focus: true,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
    vault,
    vault_passphrase: _vaultPassphraseCache,
  }, "unlock the screen");
});

// ── Execute Script modal ──────────────────────────────────────
function openExecModal() {
  $execModal.classList.remove("hidden");
  $execModal.setAttribute("aria-hidden", "false");
  // Focus the textarea, but don't clobber an existing draft.
  setTimeout(() => $execScriptBody.focus(), 30);
}

function closeExecModal() {
  $execModal.classList.add("hidden");
  $execModal.setAttribute("aria-hidden", "true");
}

$btnExecScript.addEventListener("click", openExecModal);
$execModalClose.addEventListener("click", closeExecModal);
$execModalCancel.addEventListener("click", closeExecModal);

// ── homer retrain (online training) ────────────────────────────
const $btnRetrain = document.getElementById("btn-retrain");
const $btnRollback = document.getElementById("btn-rollback");
const $retrainCount = document.getElementById("retrain-count");
let _retrainInFlight = false;

async function refreshRetrainState() {
  try {
    const r = await fetch("/api/homer/training-state");
    if (!r.ok) return;
    const j = await r.json();
    const n = j.n_trajectories_since_train || 0;
    $retrainCount.textContent = `(${n} new)`;
    if (j.is_retraining || _retrainInFlight) {
      $btnRetrain.disabled = true;
      $btnRetrain.textContent = "Retraining…";
    } else {
      $btnRetrain.disabled = false;
      // Restore button text + the span.
      $btnRetrain.textContent = "Retrain homer ";
      $btnRetrain.appendChild($retrainCount);
    }
  } catch (e) { /* ignore */ }
}

$btnRetrain?.addEventListener("click", async () => {
  if (_retrainInFlight) return;
  const confirmMsg =
    "Retrain pointer-accel + long-jump models on accumulated clicks?\n\n" +
    "• Build the dataset (canary-excluded, sanity-gated).\n" +
    "• Train new vN+1 checkpoints.\n" +
    "• Canary eval: if the new model regresses by >1.5× on the\n" +
    "  held-out set, it gets rejected and the previous stays.\n" +
    "• Training takes ~30-60s; clicks continue working in parallel.";
  if (!window.confirm(confirmMsg)) return;
  _retrainInFlight = true;
  refreshRetrainState();
  try {
    const r = await fetch("/api/homer/retrain", { method: "POST" });
    if (!r.ok) {
      const t = await r.text().catch(() => "");
      window.alert(`retrain failed to start: ${r.status} ${t}`);
    }
  } catch (e) {
    window.alert(`retrain error: ${e}`);
  } finally {
    _retrainInFlight = false;
    refreshRetrainState();
  }
});

$btnRollback?.addEventListener("click", async () => {
  if (!window.confirm(
    "Roll back homer checkpoints?\n\n" +
    "Removes the newest pointer-accel and long-jump checkpoints. " +
    "The homer falls back to the previous ones on the next click. " +
    "Use this if a fresh retrain made clicks visibly worse."
  )) return;
  try {
    const r = await fetch("/api/homer/rollback", { method: "POST" });
    const j = await r.json();
    window.alert(
      "Rolled back: " +
      (j.rolled_back && j.rolled_back.length
        ? j.rolled_back.join(", ")
        : "(nothing to roll back — already at oldest)")
    );
  } catch (e) {
    window.alert(`rollback error: ${e}`);
  }
});

// Poll retrain state every 5s so the counter stays fresh and we
// can show "Retraining…" if another tab kicked off training.
setInterval(refreshRetrainState, 5000);
refreshRetrainState();

// ESC closes the modal when it's open.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$execModal.classList.contains("hidden")) {
    closeExecModal();
  }
});

$execModalRun.addEventListener("click", () => {
  const body = ($execScriptBody.value || "").trim();
  if (!body) {
    appendSystemLog("ERROR", "Execute Script: empty body");
    return;
  }
  // Envelope the body so the controller's _partial_plan bypasses
  // chain-split + LLM and routes straight to ExecScriptAgent.
  const intent =
    "__EXEC_SCRIPT__\n" + body + "\n__EXEC_SCRIPT_END__";
  // chat label: short preview, not the marker envelope.
  const preview = body.split("\n", 1)[0].slice(0, 60) || "script";
  closeExecModal();
  startRun({
    intent,
    no_focus: false,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
  }, `exec script: ${preview}…`);
});

$chatInput.addEventListener("keydown", (e) => {
  // Enter sends. Shift+Enter inserts a newline so multi-line intents
  // are still possible. ⌘/Ctrl+Enter is kept for muscle memory.
  if (e.key === "Enter" && !e.isComposing) {
    if (e.shiftKey) return;            // Shift+Enter → newline
    e.preventDefault();
    $chatForm.requestSubmit();
  }
});

// ── logs ───────────────────────────────────────────────────────
function appendSystemLog(level, msg) {
  appendLogLine({ ts: Date.now() / 1000, level, source: "system", msg });
}

function tsString(ts) {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function appendLogLine(ev) {
  const line = document.createElement("div");
  line.className = `log-line lvl-${ev.level} src-${ev.source}`;
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = tsString(ev.ts) + "  ";
  line.appendChild(ts);
  line.appendChild(document.createTextNode(ev.msg));
  $logView.appendChild(line);
  // Trim DOM if it grows huge.
  while ($logView.childNodes.length > 1500) {
    $logView.removeChild($logView.firstChild);
  }
  if (state.logsAtBottom) {
    $logView.scrollTop = $logView.scrollHeight;
  }
}

$logView.addEventListener("scroll", () => {
  const nearBottom =
    $logView.scrollHeight - $logView.scrollTop - $logView.clientHeight < 30;
  state.logsAtBottom = nearBottom;
});

$btnClearLogs.addEventListener("click", () => {
  $logView.innerHTML = "";
});

// ── manual mouse control ───────────────────────────────────────
// The screenshot is laid out with object-fit:contain inside its
// wrapper, so the rendered image fills only part of the element box
// in one axis. To map a click to a screen percentage we need the
// rendered image rect, not the box rect.
function imageRect() {
  const w = $frame.naturalWidth;
  const h = $frame.naturalHeight;
  const box = $frame.getBoundingClientRect();
  if (!w || !h || !box.width || !box.height) return null;
  const scale = Math.min(box.width / w, box.height / h);
  const renderedW = w * scale;
  const renderedH = h * scale;
  const offsetX = (box.width - renderedW) / 2;
  const offsetY = (box.height - renderedH) / 2;
  return {
    left: box.left + offsetX,
    top: box.top + offsetY,
    width: renderedW,
    height: renderedH,
  };
}

function showClickMarker(clientX, clientY) {
  const wrap = $frame.parentElement.getBoundingClientRect();
  $clickMarker.style.left = (clientX - wrap.left) + "px";
  $clickMarker.style.top = (clientY - wrap.top) + "px";
  $clickMarker.classList.remove("hidden");
  // Restart animation
  $clickMarker.style.animation = "none";
  // Force reflow to restart CSS animation.
  void $clickMarker.offsetWidth;
  $clickMarker.style.animation = "";
  setTimeout(() => $clickMarker.classList.add("hidden"), 700);
}

// One in-flight manual-mouse call at a time; the homer holds the
// webcam, and the UI shouldn't queue conflicting requests.
let _mouseBusy = false;

function setMouseBusy(busy, label) {
  _mouseBusy = !!busy;
  if ($frameBusy) {
    $frameBusy.classList.toggle("hidden", !busy);
    if ($frameBusyLabel && label) $frameBusyLabel.textContent = label;
  }
  for (const b of [$btnMouseLeft, $btnMouseMiddle, $btnMouseRight]) {
    if (b) b.disabled = !!busy;
  }
}

async function postMouse(path, body, busyLabel) {
  if (_mouseBusy) {
    appendSystemLog("INFO", "mouse busy — skipping click");
    return null;
  }
  setMouseBusy(true, busyLabel || "working…");
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let t = "";
      try { t = await r.text(); } catch (_) {}
      appendSystemLog("ERROR", `${path} → ${r.status} ${t}`);
      return null;
    }
    return await r.json();
  } catch (e) {
    appendSystemLog("ERROR", `${path} failed: ${e}`);
    return null;
  } finally {
    setMouseBusy(false);
  }
}

// Single-click on the screenshot = home-and-click (visual servo).
// Double-click = double-click at the current red-cursor position
// (no re-homing — the host's red cursor is already where the user
// wants the dblclick to land). We detect the pair ourselves with a
// debounce timer so the dblclick suppresses the single-click home;
// otherwise the homer would fire on the first click AND the second.
const DBL_CLICK_MS = 300;
let _pendingSingleClickTimer = null;

async function _fireHomingClick(x_pct, y_pct, clientX, clientY) {
  if ($frame.classList.contains("empty")) return;
  if (_mouseBusy) return;
  showClickMarker(clientX, clientY);
  appendSystemLog(
    "INFO",
    `mouse click_at (${x_pct.toFixed(3)}, ${y_pct.toFixed(3)})`,
  );
  await postMouse(
    "/api/mouse/click_at",
    { x_pct, y_pct, button: "left" },
    "homing cursor…",
  );
  state.hadClickAt = true;
  if ($passInput) {
    $passInput.value = "";
    $passInput.focus();
  }
}

$frame.addEventListener("click", (e) => {
  if (!$optClickToMove || !$optClickToMove.checked) return;
  if ($frame.classList.contains("empty")) return;
  const rect = imageRect();
  if (!rect) return;
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return;
  const x_pct = Math.max(0, Math.min(1, x / rect.width));
  const y_pct = Math.max(0, Math.min(1, y / rect.height));
  // Wait DBL_CLICK_MS for a possible second click. If one comes,
  // the dblclick handler cancels this timer and fires its own action
  // instead — no homing, just a double-click at the current cursor.
  if (_pendingSingleClickTimer != null) {
    clearTimeout(_pendingSingleClickTimer);
  }
  _pendingSingleClickTimer = setTimeout(() => {
    _pendingSingleClickTimer = null;
    _fireHomingClick(x_pct, y_pct, e.clientX, e.clientY);
  }, DBL_CLICK_MS);
});

// Double-click on the screenshot → home the cursor to the
// double-click location AND fire two clicks there. Semantically the
// same as "open the thing under my pointer" — the operator usually
// wants the host to act on whatever they're aiming at, not on
// whatever the host's red cursor happens to be hovering. The
// single-click home is cancelled by clearing the debounce timer so
// the homer only fires once (with count=2) instead of twice.
$frame.addEventListener("dblclick", async (e) => {
  if (!$optClickToMove || !$optClickToMove.checked) return;
  if ($frame.classList.contains("empty")) return;
  if (_pendingSingleClickTimer != null) {
    clearTimeout(_pendingSingleClickTimer);
    _pendingSingleClickTimer = null;
  }
  e.preventDefault();
  const rect = imageRect();
  if (!rect) return;
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return;
  const x_pct = Math.max(0, Math.min(1, x / rect.width));
  const y_pct = Math.max(0, Math.min(1, y / rect.height));
  if (_mouseBusy) return;
  showClickMarker(e.clientX, e.clientY);
  appendSystemLog(
    "INFO",
    `mouse dblclick_at (${x_pct.toFixed(3)}, ${y_pct.toFixed(3)})`,
  );
  await postMouse(
    "/api/mouse/click_at",
    { x_pct, y_pct, button: "left", count: 2 },
    "homing + double-clicking…",
  );
  state.hadClickAt = true;
  if ($passInput) {
    $passInput.value = "";
    $passInput.focus();
  }
});

// Block the default context menu when right-clicking on the
// screenshot — operators usually want to fire a remote right-click
// instead of opening the browser menu. We expose right-click via
// the Right button; suppress the menu so it's not a distraction.
$frame.addEventListener("contextmenu", (e) => {
  if ($optClickToMove && $optClickToMove.checked) e.preventDefault();
});

async function fireButton(button) {
  appendSystemLog("INFO", `mouse click button=${button}`);
  await postMouse(
    "/api/mouse/click", { button }, `clicking ${button}…`,
  );
}

if ($btnMouseLeft)
  $btnMouseLeft.addEventListener("click", () => fireButton("left"));
if ($btnMouseMiddle)
  $btnMouseMiddle.addEventListener("click", () => fireButton("middle"));
if ($btnMouseRight)
  $btnMouseRight.addEventListener("click", () => fireButton("right"));

// ── mouse wheel → remote scroll ────────────────────────────────
// When the operator scrolls while hovering over the screenshot, we
// forward wheel ticks to the target via /api/mouse/scroll. Browsers
// emit many wheel events per gesture (often dozens of small
// deltaY samples), so we accumulate the pixel delta and flush via
// a single coalesced POST per ~120 ms. Without coalescing, a single
// trackpad gesture would spam the Pi with 30+ HTTP requests.
//
// Position (x_pct, y_pct) is included for telemetry / snapshot
// labelling — the actual scroll happens at the target's current
// cursor location. Hovering-over-region routing would need a fast
// open-loop home, which is on the to-do list, not in this MVP.
let _wheelPxAccum = 0;
let _wheelLastPos = { x: null, y: null };
let _wheelFlushTimer = null;
let _wheelFlushing = false;
// Browsers report deltaY in pixels (DOM_DELTA_PIXEL); ~30 px is a
// typical trackpad two-finger increment, ~100 px is one mouse-wheel
// notch. Map ~30 px to one Pi wheel tick so trackpad gestures
// actually cross the threshold, and clamp the per-POST amount so
// the Pi never receives a single scroll larger than ±10 ticks.
const WHEEL_PX_PER_TICK = 30;
const WHEEL_MAX_TICKS_PER_POST = 10;
const WHEEL_FLUSH_DELAY_MS = 80;

async function flushScroll() {
  if (_wheelFlushing) return;
  // If the accumulated delta hasn't crossed one Pi tick yet, this
  // is just trackpad noise (every micro-pan emits wheel events with
  // sub-pixel deltaY). Showing the "scrolling…" hourglass for those
  // flickers makes the operator think a scroll is happening when
  // nothing actually fires. Bail before touching the busy state.
  if (Math.abs(_wheelPxAccum) < WHEEL_PX_PER_TICK) return;
  // Another mouse action is in flight (click, click_at, paste-file).
  // Don't barge — re-arm the debounce timer so we try again once
  // the in-flight action releases the busy state.
  if (_mouseBusy) {
    if (_wheelFlushTimer === null) {
      _wheelFlushTimer = setTimeout(() => {
        _wheelFlushTimer = null;
        flushScroll();
      }, WHEEL_FLUSH_DELAY_MS);
    }
    return;
  }
  _wheelFlushing = true;
  // Show the same busy hourglass other manual mouse actions use —
  // user feedback on every action, per the cc UX contract.
  setMouseBusy(true, "scrolling…");
  try {
    while (Math.abs(_wheelPxAccum) >= WHEEL_PX_PER_TICK) {
      const sign = Math.sign(_wheelPxAccum);
      const ticks = Math.min(
        WHEEL_MAX_TICKS_PER_POST,
        Math.floor(Math.abs(_wheelPxAccum) / WHEEL_PX_PER_TICK),
      );
      const amount = sign * ticks;
      // Drain the pixels we're about to send so simultaneous wheel
      // events accumulate only the REMAINDER, not the whole gesture.
      _wheelPxAccum -= amount * WHEEL_PX_PER_TICK;
      const body = {
        amount,
        x_pct: _wheelLastPos.x,
        y_pct: _wheelLastPos.y,
      };
      appendSystemLog(
        "INFO",
        `mouse scroll amount=${amount} ` +
        `at (${_wheelLastPos.x?.toFixed(3)}, ${_wheelLastPos.y?.toFixed(3)})`,
      );
      try {
        const r = await fetch("/api/mouse/scroll", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          let t = "";
          try { t = await r.text(); } catch (_) {}
          appendSystemLog("ERROR", `/api/mouse/scroll → ${r.status} ${t}`);
          _wheelPxAccum = 0;
          break;
        }
      } catch (e) {
        appendSystemLog("ERROR", `/api/mouse/scroll failed: ${e}`);
        _wheelPxAccum = 0;
        break;
      }
    }
  } finally {
    setMouseBusy(false);
    _wheelFlushing = false;
  }
}

// Normalise a wheel event's deltaY to pixels regardless of
// deltaMode. A real mouse with a notched wheel typically reports
// `deltaMode = 1` (DOM_DELTA_LINE) with deltaY = ±3 — those 3 are
// lines, not pixels, and treating them as 3 px never crosses the
// 30-px flush threshold so the user observes "scroll does nothing"
// even though the handler fires. Trackpads usually report
// deltaMode = 0 (pixels) with much larger deltaY values.
function _wheelDeltaPx(e) {
  switch (e.deltaMode) {
    case 1: return e.deltaY * 38;     // line ≈ ~38 px (Firefox default)
    case 2: return e.deltaY * 800;    // page ≈ viewport-ish; unusual
    case 0:
    default: return e.deltaY;
  }
}

$frame.addEventListener("wheel", (e) => {
  // Scrolling over the screenshot is unambiguous intent (you're
  // trying to scroll the content shown there), so unlike click_at
  // we do NOT gate on the click-to-move toggle. Operators who want
  // native page scroll can move the cursor off the screenshot
  // pane.
  if ($frame.classList.contains("empty")) return;
  // Don't let the browser also scroll the cc UI's pane.
  e.preventDefault();
  const rect = imageRect();
  if (rect) {
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x >= 0 && y >= 0 && x <= rect.width && y <= rect.height) {
      _wheelLastPos.x = Math.max(0, Math.min(1, x / rect.width));
      _wheelLastPos.y = Math.max(0, Math.min(1, y / rect.height));
    }
  }
  _wheelPxAccum += _wheelDeltaPx(e);
  if (_wheelFlushTimer !== null) {
    clearTimeout(_wheelFlushTimer);
  }
  _wheelFlushTimer = setTimeout(() => {
    _wheelFlushTimer = null;
    flushScroll();
  }, WHEEL_FLUSH_DELAY_MS);
}, { passive: false });

// Expose for the test harness — playwright can call window.__teTest
// to bypass UI gating during automated tests.
window.__teTest = window.__teTest || {};
window.__teTest.flushScroll = flushScroll;
window.__teTest.peekScrollState = () => ({
  px: _wheelPxAccum,
  pos: { ..._wheelLastPos },
  flushing: _wheelFlushing,
});

// ── keyboard passthrough ───────────────────────────────────────
// Each keystroke in the passthrough field is forwarded to the host.
// Requests are serialized so typing fast doesn't reorder them: HTTP
// + Pi+BT HID don't guarantee in-order delivery across concurrent
// requests, but a single in-flight FIFO does.
const $passInput = document.getElementById("passthrough-input");
const $btnPassEnter = document.getElementById("btn-passthrough-enter");
const $btnPassTab = document.getElementById("btn-passthrough-tab");
const $btnPassEsc = document.getElementById("btn-passthrough-esc");
const $btnPassClear = document.getElementById("btn-passthrough-clear");
const $btnPassUp = document.getElementById("btn-passthrough-up");
const $btnPassDown = document.getElementById("btn-passthrough-down");
const $btnPassLeft = document.getElementById("btn-passthrough-left");
const $btnPassRight = document.getElementById("btn-passthrough-right");

const _kbQueue = [];
let _kbDraining = false;

// Typing-snapshot loop: while keystrokes are flowing to the host we
// want a much tighter refresh cadence than the 60s Active Refresh —
// the operator wants to see the field fill in. We piggyback on
// /api/snapshot?dedup=1 so unchanged frames still write nothing.
// The loop:
//   - any keystroke pushes lastKeystrokeAt forward
//   - while (now - lastKeystrokeAt) < TYPING_IDLE_MS, fire a dedup
//     snapshot every TYPING_SNAPSHOT_MS
//   - after idle, the timer stops; the long-poll surfaces frames the
//     way it normally does
const TYPING_SNAPSHOT_MS = 2000;
const TYPING_IDLE_MS = 5000;
let lastKeystrokeAt = 0;
let typingSnapshotTimer = null;

async function _typingSnapshotTick() {
  if (activeRefreshInFlight) return;
  if (document.hidden) return;
  if (Date.now() - lastKeystrokeAt > TYPING_IDLE_MS) {
    clearInterval(typingSnapshotTimer);
    typingSnapshotTimer = null;
    return;
  }
  activeRefreshInFlight = true;
  try {
    await fetch("/api/snapshot?dedup=1", { method: "POST" });
  } catch (e) { /* ignore — next tick will retry */ }
  finally { activeRefreshInFlight = false; }
}
function _markTypingActivity() {
  lastKeystrokeAt = Date.now();
  state.liveMode = true;
  if (typingSnapshotTimer == null) {
    // Fire one immediately so the operator gets feedback within ~1s
    // of the first keystroke instead of waiting a full interval.
    _typingSnapshotTick();
    typingSnapshotTimer = setInterval(
      _typingSnapshotTick, TYPING_SNAPSHOT_MS,
    );
  }
}

function _kbEnqueue(job) {
  _kbQueue.push(job);
  _markTypingActivity();
  _kbDrain();
}

async function _kbDrain() {
  if (_kbDraining) return;
  _kbDraining = true;
  if ($passInput) $passInput.classList.add("busy");
  try {
    while (_kbQueue.length > 0) {
      const job = _kbQueue.shift();
      try {
        const r = await fetch(job.path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(job.body),
        });
        if (!r.ok) {
          let t = "";
          try { t = await r.text(); } catch (_) {}
          appendSystemLog("ERROR", `${job.path} → ${r.status} ${t}`);
          _kbQueue.length = 0;       // give up on this typing burst
          break;
        }
      } catch (e) {
        appendSystemLog("ERROR", `${job.path} failed: ${e}`);
        _kbQueue.length = 0;
        break;
      }
    }
  } finally {
    _kbDraining = false;
    if ($passInput) $passInput.classList.remove("busy");
  }
}

// Map browser key names to Pi keystroke names. Pi's HID-codes map is
// case-sensitive PascalCase ("Enter", not "enter") for special keys —
// see raspi/hid_codes.py KEY_CODES.
const _PASS_SPECIAL = {
  "Enter": "Enter",
  "Backspace": "Backspace",
  "Tab": "Tab",
  "Escape": "Escape",
  "ArrowUp": "Up",
  "ArrowDown": "Down",
  "ArrowLeft": "Left",
  "ArrowRight": "Right",
  "Home": "Home",
  "End": "End",
  "PageUp": "PageUp",
  "PageDown": "PageDown",
  "Delete": "Delete",
};

// Warning banner about typing-without-clicking-first.
const $passWarn = document.getElementById("passthrough-warn");
const $passWarnDismiss = document.getElementById(
  "passthrough-warn-dismiss",
);

function _maybeShowPassthroughWarn() {
  if (state.hadClickAt) return;
  if (state.warnedPassthrough) return;
  if (!$passWarn) return;
  $passWarn.classList.remove("hidden");
}

if ($passWarnDismiss) {
  $passWarnDismiss.addEventListener("click", () => {
    state.warnedPassthrough = true;
    try {
      sessionStorage.setItem("te-warned-passthrough", "1");
    } catch (_) {}
    if ($passWarn) $passWarn.classList.add("hidden");
  });
}

function _passthroughHandleKey(e) {
  if (!$passInput) return;
  // Let the browser handle navigation modifier-only events.
  if (e.key === "Shift" || e.key === "Control" || e.key === "Alt"
      || e.key === "Meta") {
    return;
  }
  // First "real" keystroke without a prior click_at this session?
  // Surface the warning so the operator notices when host focus is
  // wrong and they're typing into the void.
  _maybeShowPassthroughWarn();
  e.preventDefault();

  let hasCtrl = e.ctrlKey;
  const hasMeta = e.metaKey;
  const hasAlt = e.altKey;
  const hasShift = e.shiftKey;
  // Mac-target shortcut convenience: Mac uses Cmd, not Ctrl, for
  // copy/paste/cut/select-all/etc. An operator coming from Linux
  // or Windows muscle-memory presses Ctrl-A and expects select-all,
  // but their literal Ctrl-A reaches the Mac as Ctrl-A which means
  // "beginning of line" (or nothing) on Mac. When the target is
  // macOS (read from $optPlatform — defaults to "macos") and the
  // keystroke is a *plain* Ctrl-letter/digit chord with no other
  // modifier, swap the ctrl bit for super (Cmd). Ctrl-Shift-letter
  // and Ctrl-arrow are left literal because Mac apps do bind those
  // (Ctrl-arrow = Mission Control, Ctrl-Shift-Tab = previous tab,
  // Ctrl-A in iTerm = bash beginning-of-line, etc.).
  const targetIsMac = !$optPlatform || $optPlatform.value === "macos";
  if (
    hasCtrl && !hasMeta && !hasAlt
    && e.key && e.key.length === 1
    && /[a-zA-Z0-9]/.test(e.key)
    && targetIsMac
  ) {
    hasCtrl = false;
    // We'll push "super" below.
    e._teMacRemap = true;
  }
  const mods = [];
  if (hasCtrl) mods.push("ctrl");
  // Cmd on macOS / Super on Linux. The Pi modifier map at
  // raspi/hid_codes.py::MODIFIER_MAP only accepts "super" / "meta"
  // / "win" (no "cmd"), so sending "cmd" here used to silently drop
  // the modifier and the host received an unmodified keystroke —
  // which is why Cmd-C / Ctrl-Tab / etc. appeared not to work even
  // though the rest of the pipeline was firing.
  if (hasMeta || e._teMacRemap) mods.push("super");
  if (hasAlt) mods.push("alt");
  if (hasShift) mods.push("shift");

  // Mirror locally regardless of focus: the operator wants to see
  // what they typed, whether they were focused on the passthrough
  // field or hovering over the screenshot with global capture
  // active. The mirror is purely visual — the host has already
  // received the literal keystroke via /api/keyboard above.
  const special = _PASS_SPECIAL[e.key];
  if (special) {
    _kbEnqueue({
      path: "/api/keyboard/key",
      body: { key: special, modifiers: mods },
    });
    _applySpecialLocally(special, hasShift);
    return;
  }

  // Single printable character.
  if (e.key.length === 1) {
    // If modifier present (other than shift), send as combo.
    const nonShiftMods = mods.filter(m => m !== "shift");
    if (nonShiftMods.length > 0) {
      _kbEnqueue({
        path: "/api/keyboard/key",
        body: { key: e.key.toLowerCase(), modifiers: mods },
      });
      // Local mirror of shortcut-letter chords would mislead more
      // than it helps (Ctrl-A locally selects, but on the host it
      // depends on the focused app). Leave the field alone.
      return;
    }
    _kbEnqueue({
      path: "/api/keyboard/text",
      body: { text: e.key },
    });
    _insertCharLocally(e.key);
  }
}

// ── Local-mirror caret editing ────────────────────────────────────
// The passthrough field's keydown listener calls preventDefault() on
// every key (so the host receives the literal keystroke, not the
// browser's interpretation). That intercepts the browser's built-in
// caret + edit handling, so without this we'd have:
//   - Left/Right arrow do nothing locally
//   - Backspace always deletes from the end
//   - Typing mid-string appends to the end
// These helpers re-apply the navigation/edit on the local <input>
// so the operator sees the same caret + text the host is receiving.

function _setCaret(start, end) {
  if (end === undefined) end = start;
  try {
    $passInput.setSelectionRange(start, end);
  } catch (_) {}
}

function _getCaret() {
  return {
    start: $passInput.selectionStart ?? $passInput.value.length,
    end:   $passInput.selectionEnd   ?? $passInput.value.length,
  };
}

function _insertCharLocally(ch) {
  const { start, end } = _getCaret();
  const v = $passInput.value;
  $passInput.value = v.slice(0, start) + ch + v.slice(end);
  _setCaret(start + ch.length);
}

function _applySpecialLocally(special, hasShift) {
  const v = $passInput.value;
  const { start, end } = _getCaret();
  switch (special) {
    case "Backspace": {
      if (start !== end) {
        $passInput.value = v.slice(0, start) + v.slice(end);
        _setCaret(start);
      } else if (start > 0) {
        $passInput.value = v.slice(0, start - 1) + v.slice(start);
        _setCaret(start - 1);
      }
      return;
    }
    case "Delete": {
      if (start !== end) {
        $passInput.value = v.slice(0, start) + v.slice(end);
        _setCaret(start);
      } else if (start < v.length) {
        $passInput.value = v.slice(0, start) + v.slice(start + 1);
        _setCaret(start);
      }
      return;
    }
    case "Left": {
      if (hasShift) _setCaret(Math.max(0, start - 1), end);
      else if (start !== end) _setCaret(Math.min(start, end));
      else _setCaret(Math.max(0, start - 1));
      return;
    }
    case "Right": {
      if (hasShift) _setCaret(start, Math.min(v.length, end + 1));
      else if (start !== end) _setCaret(Math.max(start, end));
      else _setCaret(Math.min(v.length, start + 1));
      return;
    }
    case "Home": {
      if (hasShift) _setCaret(0, end);
      else _setCaret(0);
      return;
    }
    case "End": {
      if (hasShift) _setCaret(start, v.length);
      else _setCaret(v.length);
      return;
    }
    case "Tab": {
      $passInput.value = v.slice(0, start) + "\t" + v.slice(end);
      _setCaret(start + 1);
      return;
    }
    case "Enter":
    case "Escape": {
      // Both reset the local buffer — the host has consumed what we
      // had, the local field starts over.
      $passInput.value = "";
      _setCaret(0);
      return;
    }
    // Up / Down / PageUp / PageDown have no meaningful single-line
    // input editing — leave the local field untouched.
  }
}

// Global key capture: when the user hasn't focused another text input
// (chat, vault, exec body), forward keystrokes to the host. Makes the
// "click on icon → start typing" flow work without the operator
// remembering to click the passthrough field.
const _GLOBAL_PASSTHRU_SKIP_IDS = new Set([
  "chat-input", "opt-vault", "exec-script-body",
]);
function _shouldGlobalCapture() {
  const a = document.activeElement;
  if (!a) return true;
  if (a === $passInput) return false;     // field handles it itself
  if (a.id && _GLOBAL_PASSTHRU_SKIP_IDS.has(a.id)) return false;
  const tag = (a.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") {
    return false;
  }
  if (a.isContentEditable) return false;
  return true;
}
// What the user originally asked for: "capture any special keys
// like Ctrl-C, etc. when on top of the remote image". That meant
// modifier combos and special keys (arrows, F-keys, Tab, Enter,
// Esc, ...), NOT plain letters. Plain typing while hovering the
// image used to silently forward to the host even when the user
// thought they were typing locally — surprising and easy to do by
// accident, because the screenshot fills most of the viewport so
// the mouse is "over the image" by default.
//
// New rule: the global hover-capture only fires for keystrokes that
// have an explicit reason to land on the host:
//   - any key with a modifier (Ctrl / Cmd / Alt)
//   - any non-printable / named key (Arrow*, F*, Escape, Tab, Enter,
//     Backspace, Delete, Home, End, PageUp, PageDown)
// Plain printable characters fall through to the browser. If the
// operator actually wants to type into the host, they click into
// the passthrough field — its own listener still forwards
// everything regardless of mouse position.
function _isSpecialOrModifiedKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return true;
  if (e.key && e.key.length > 1) {
    // Don't fire on modifier-only press; _passthroughHandleKey
    // skips those anyway but no point doing the work.
    if (e.key === "Shift" || e.key === "Control"
        || e.key === "Alt" || e.key === "Meta") return false;
    return true;
  }
  return false;
}
document.addEventListener("keydown", (e) => {
  if (!_shouldGlobalCapture()) return;
  if (!state.mouseOverFrame) return;
  if (!_isSpecialOrModifiedKey(e)) return;
  _passthroughHandleKey(e);
}, true);

// Visual cue: when the mouse is hovering over the screenshot AND
// no UI input is focused, any keystroke (including Ctrl-C, arrow
// keys, F-keys, Cmd-V, etc.) is being forwarded to the host. Show
// a thin accent border on the image so the operator can tell at a
// glance that the next keypress will land on the target machine,
// not in the browser.
const $framePane = document.getElementById("frame-pane");
function _refreshHostKbCue() {
  if (!$framePane) return;
  const on = state.mouseOverFrame && _shouldGlobalCapture();
  $framePane.classList.toggle("host-kb", !!on);
}
if ($frame) {
  $frame.addEventListener("mouseenter", () => {
    state.mouseOverFrame = true;
    _refreshHostKbCue();
  });
  $frame.addEventListener("mouseleave", () => {
    state.mouseOverFrame = false;
    _refreshHostKbCue();
  });
}
// Focus changes (clicking into chat-input, away from it, etc.)
// flip whether global capture is active — update the cue too.
document.addEventListener("focusin", _refreshHostKbCue);
document.addEventListener("focusout", _refreshHostKbCue);

if ($passInput) {
  $passInput.addEventListener("keydown", _passthroughHandleKey);
  // Block paste/cut/contextmenu autofill — only keystrokes go through.
  $passInput.addEventListener("paste", (e) => {
    e.preventDefault();
    const text = (e.clipboardData || window.clipboardData).getData("text");
    if (!text) return;
    _kbEnqueue({
      path: "/api/keyboard/text",
      body: { text },
    });
    $passInput.value += text;
  });
}

if ($btnPassEnter)
  $btnPassEnter.addEventListener("click", () => {
    _kbEnqueue({
      path: "/api/keyboard/key", body: { key: "Enter", modifiers: [] },
    });
    if ($passInput) $passInput.value = "";
  });
if ($btnPassTab)
  $btnPassTab.addEventListener("click", () => {
    _kbEnqueue({
      path: "/api/keyboard/key", body: { key: "Tab", modifiers: [] },
    });
    if ($passInput) $passInput.value += "\t";
  });
if ($btnPassEsc)
  $btnPassEsc.addEventListener("click", () => {
    _kbEnqueue({
      path: "/api/keyboard/key", body: { key: "Escape", modifiers: [] },
    });
  });
if ($btnPassClear)
  $btnPassClear.addEventListener("click", () => {
    if ($passInput) $passInput.value = "";
  });

// Arrow-key buttons — send the key directly without needing to focus
// the passthrough field or hover over the screenshot.
for (const [btn, key] of [
  [$btnPassUp, "Up"],
  [$btnPassDown, "Down"],
  [$btnPassLeft, "Left"],
  [$btnPassRight, "Right"],
]) {
  if (!btn) continue;
  btn.addEventListener("click", () => {
    _kbEnqueue({
      path: "/api/keyboard/key", body: { key, modifiers: [] },
    });
  });
}

// ── paste-file: pick a local file, type it on the host ────────
const $btnPasteFile = document.getElementById("btn-paste-file");
const $pasteFilePicker = document.getElementById("paste-file-picker");
const $pasteModal = document.getElementById("paste-modal");
const $pasteModalClose = document.getElementById("paste-modal-close");
const $pasteModalCancel = document.getElementById("paste-modal-cancel");
const $pasteModalSend = document.getElementById("paste-modal-send");
const $pasteMetaName = document.getElementById("paste-meta-name");
const $pasteMetaStats = document.getElementById("paste-meta-stats");
const $pastePath = document.getElementById("paste-path");
const $pasteContent = document.getElementById("paste-content");
const $pasteOptMaximize = document.getElementById("paste-opt-maximize");
const $pasteOptVerify = document.getElementById("paste-opt-verify");
const $pasteOptBodyReadback = document.getElementById("paste-opt-body-readback");
const $pasteOptPlatform = document.getElementById("paste-opt-platform");

const PASTE_MAX_BYTES = 50_000;

function _pasteOpenModal() {
  $pasteModal.classList.remove("hidden");
  $pasteModal.setAttribute("aria-hidden", "false");
  setTimeout(() => $pasteContent && $pasteContent.focus(), 30);
}

function _pasteCloseModal() {
  $pasteModal.classList.add("hidden");
  $pasteModal.setAttribute("aria-hidden", "true");
}

function _pasteShowStats(name, content) {
  const lines = content.split("\n").length;
  const bytes = new Blob([content]).size;
  $pasteMetaName.textContent = name;
  $pasteMetaStats.textContent =
    `${bytes} bytes · ${lines} lines · ~${Math.round(bytes / 35)}s to type at 35 cps`;
}

if ($btnPasteFile)
  $btnPasteFile.addEventListener("click", () => {
    if ($pasteFilePicker) $pasteFilePicker.click();
  });

if ($pasteFilePicker)
  $pasteFilePicker.addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    if (file.size > PASTE_MAX_BYTES) {
      appendSystemLog(
        "ERROR",
        `${file.name}: ${file.size} bytes exceeds ${PASTE_MAX_BYTES} cap`,
      );
      $pasteFilePicker.value = "";
      return;
    }
    let text = "";
    try {
      text = await file.text();
    } catch (err) {
      appendSystemLog("ERROR", `read failed: ${err}`);
      return;
    }
    $pasteContent.value = text;
    // Default host filename = local basename, under /tmp.
    const safe = (file.name || "cc_paste.txt")
      .replace(/[^A-Za-z0-9._-]/g, "_")
      .slice(0, 80);
    $pastePath.value = `/tmp/${safe}`;
    _pasteShowStats(file.name, text);
    _pasteOpenModal();
    // Reset so picking the same file again still fires change.
    $pasteFilePicker.value = "";
  });

// Re-compute stats live as the operator edits the buffer.
if ($pasteContent)
  $pasteContent.addEventListener("input", () => {
    const name = $pasteMetaName.textContent || "buffer";
    _pasteShowStats(name, $pasteContent.value);
  });

if ($pasteModalClose) $pasteModalClose.addEventListener("click", _pasteCloseModal);
if ($pasteModalCancel) $pasteModalCancel.addEventListener("click", _pasteCloseModal);

async function _pasteSubmit() {
  const content = $pasteContent.value;
  if (!content) {
    appendSystemLog("ERROR", "paste-file: empty content");
    return;
  }
  if (new Blob([content]).size > PASTE_MAX_BYTES) {
    appendSystemLog("ERROR", "paste-file: content over 50 KB cap");
    return;
  }
  const body = {
    content,
    path: ($pastePath.value || "/tmp/cc_paste.txt").trim(),
    platform: $pasteOptPlatform.value,
    maximize: !!$pasteOptMaximize.checked,
    verify: !!$pasteOptVerify.checked,
    body_readback: !!($pasteOptBodyReadback && $pasteOptBodyReadback.checked),
  };
  _pasteCloseModal();
  setMouseBusy(true, "pasting file…");
  appendSystemLog(
    "INFO",
    `paste-file → ${body.path} (${content.length} chars, verify=${body.verify})`,
  );
  try {
    const r = await fetch("/api/paste-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let t = "";
      try { t = await r.text(); } catch (_) {}
      appendSystemLog("ERROR", `paste-file → ${r.status} ${t}`);
      return;
    }
    const data = await r.json();
    if (data.verify) {
      const v = data.verify;
      const tag = v.match ? "✓ SHA MATCH" : "✗ SHA MISMATCH";
      const rounds = Array.isArray(v.rounds) ? v.rounds : [];
      const lastRound = rounds[rounds.length - 1] || {};
      appendSystemLog(
        v.match ? "INFO" : "ERROR",
        `${tag} after ${rounds.length} round(s) ` +
          `(${v.n_chunks} chunks @ ${v.chunk_size}B). ` +
          `local=${(v.local_sha || "").slice(0, 12)}… ` +
          `host=${(lastRound.host_sha || "?").slice(0, 12)}…`,
      );
      // Surface each repair round's bad-chunk set so failures are
      // diagnosable from the chat log alone.
      for (const r of rounds) {
        const bad = Array.isArray(r.bad_indices) ? r.bad_indices : null;
        if (bad && bad.length) {
          appendSystemLog(
            "INFO",
            `round ${r.round}: ${bad.length} chunk(s) repaired ` +
              `[${bad.slice(0, 24).join(",")}${bad.length > 24 ? ",…" : ""}]`,
          );
        }
        if (r.abort_reason) {
          appendSystemLog("ERROR", `round ${r.round} abort: ${r.abort_reason}`);
        }
      }
      // Per-chunk retransmit summary — useful when the channel is
      // noisy and a chunk needed several attempts to land.
      const rx = v.chunk_retransmits || {};
      const rxKeys = Object.keys(rx);
      if (rxKeys.length) {
        const max = Math.max(...rxKeys.map(k => rx[k]));
        const total = rxKeys.reduce((s, k) => s + rx[k], 0);
        appendSystemLog(
          "INFO",
          `retransmits: ${total} total across ${rxKeys.length} chunk(s), ` +
            `max ${max} for one chunk` +
            (v.per_chunk_retry_cap
              ? ` (cap ${v.per_chunk_retry_cap})` : ""),
        );
      }
    } else {
      appendSystemLog(
        "INFO",
        `paste-file ok: wrote ${data.wrote_path} (${data.sent_chars} chars)`,
      );
    }
    // Body readback (the `more`-based visual confirmation) is
    // separate from the SHA verdict — surface it on its own line.
    if (data.body_readback) {
      const rb = data.body_readback;
      const ok = rb.similarity >= 0.85;
      appendSystemLog(
        ok ? "INFO" : "ERROR",
        `${ok ? "✓" : "≈"} body readback: ` +
          `similarity=${rb.similarity} over ${rb.pages} page(s) ` +
          `(expected=${rb.expected_chars}c, ocr=${rb.ocr_chars}c)`,
      );
    }
  } catch (e) {
    appendSystemLog("ERROR", `paste-file failed: ${e}`);
  } finally {
    setMouseBusy(false);
  }
}

if ($pasteModalSend) $pasteModalSend.addEventListener("click", _pasteSubmit);

// ESC closes paste modal when open.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $pasteModal && !$pasteModal.classList.contains("hidden")) {
    _pasteCloseModal();
  }
});

function connectGlobalLogs() {
  if (state.globalSrc) state.globalSrc.close();
  const src = new EventSource("/api/logs?tail=200");
  src.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      appendLogLine(ev);
    } catch (_) {}
  };
  src.onerror = () => {
    src.close();
    setTimeout(connectGlobalLogs, 1500);
  };
  state.globalSrc = src;
}

// ── chat history (replay prior runs on load) ─────────────────────
async function loadChatHistory() {
  try {
    const r = await fetch("/api/runs?limit=50");
    if (!r.ok) return;
    const data = await r.json();
    const items = data.items || [];
    // The server returns newest-first; chat reads top-to-bottom
    // oldest-to-newest, so reverse before appending.
    for (const rec of items.reverse()) {
      appendChat({
        runId: rec.run_id,
        intent: rec.intent,
        status: rec.status,
        reason: rec.reason,
      });
      // If a run somehow ended up still "running" while the page
      // was reloaded, resume polling its status so the chat updates
      // when it eventually completes.
      if (rec.status === "running" || rec.status === "pending") {
        pollRunStatus(rec.run_id);
      }
    }
  } catch (_) {}
}

// ── column divider drag (desktop layout) ───────────────────────
// Right column width is stored in --right-col-width on <main>. We
// persist it to localStorage so the user's split survives reloads.
// Mobile (<800px) hides the divider and stacks columns vertically;
// the resize handlers are no-ops there because the divider isn't
// in the DOM hit-test.
(function initColDivider() {
  const root = document.getElementById("app");
  const divider = document.getElementById("col-divider");
  if (!root || !divider) return;
  const STORAGE_KEY = "cc.rightColWidth";
  const MIN_LEFT = 320;
  const MIN_RIGHT = 240;

  function applyWidth(px) {
    const max = Math.max(MIN_RIGHT, window.innerWidth - MIN_LEFT - 24);
    const clamped = Math.max(MIN_RIGHT, Math.min(max, px));
    root.style.setProperty("--right-col-width", clamped + "px");
    return clamped;
  }
  function clampAndPersist(px) {
    const v = applyWidth(px);
    try { localStorage.setItem(STORAGE_KEY, String(v)); } catch (_) {}
  }
  // Restore from storage (if any) on boot, before first paint settles.
  try {
    const saved = parseFloat(localStorage.getItem(STORAGE_KEY) || "");
    if (Number.isFinite(saved) && saved > 0) applyWidth(saved);
  } catch (_) {}
  // Re-clamp on window resize so a previously-saved width can't
  // leave the left column below MIN_LEFT.
  window.addEventListener("resize", () => {
    const cur = parseFloat(
      getComputedStyle(root).getPropertyValue("--right-col-width") || "0"
    );
    if (Number.isFinite(cur) && cur > 0) applyWidth(cur);
  });

  let dragging = false;
  function onPointerDown(e) {
    dragging = true;
    divider.classList.add("dragging");
    document.body.classList.add("col-resizing");
    divider.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }
  function onPointerMove(e) {
    if (!dragging) return;
    // Right column width = distance from pointer to right edge of
    // the viewport, minus the app's right padding (6px).
    const width = window.innerWidth - e.clientX - 6;
    applyWidth(width);
  }
  function onPointerUp(e) {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove("dragging");
    document.body.classList.remove("col-resizing");
    try { divider.releasePointerCapture?.(e.pointerId); } catch (_) {}
    const cur = parseFloat(
      getComputedStyle(root).getPropertyValue("--right-col-width") || "0"
    );
    if (Number.isFinite(cur) && cur > 0) {
      try { localStorage.setItem(STORAGE_KEY, String(cur)); } catch (_) {}
    }
  }
  divider.addEventListener("pointerdown", onPointerDown);
  divider.addEventListener("pointermove", onPointerMove);
  divider.addEventListener("pointerup", onPointerUp);
  divider.addEventListener("pointercancel", onPointerUp);
  // Double-click resets to default split.
  divider.addEventListener("dblclick", () => {
    clampAndPersist(Math.round(window.innerWidth * 0.38));
  });
  // Keyboard nudging when the divider is focused.
  divider.addEventListener("keydown", (e) => {
    const cur = parseFloat(
      getComputedStyle(root).getPropertyValue("--right-col-width") || "0"
    ) || (window.innerWidth * 0.38);
    const step = e.shiftKey ? 64 : 16;
    if (e.key === "ArrowLeft") { clampAndPersist(cur + step); e.preventDefault(); }
    else if (e.key === "ArrowRight") { clampAndPersist(cur - step); e.preventDefault(); }
  });
})();

// ── boot ───────────────────────────────────────────────────────
async function init() {
  await loadChatHistory();
  await refreshKnownIds();
  connectGlobalLogs();
  pollLatest();
  // Periodic re-list to catch ring-buffer evictions / deep history.
  setInterval(refreshKnownIds, 5000);
}

init();
