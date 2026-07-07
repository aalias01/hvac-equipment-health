const PRODUCTION_API_BASE = "https://alvinalias-portfolio-ml-api.hf.space/hvac";
const API_BASE = window.HVAC_API_BASE
  || new URLSearchParams(window.location.search).get("api")
  || localStorage.getItem("HVAC_API_BASE")
  || PRODUCTION_API_BASE;

const FEATURE_GLOSSES = {
  rolling_cop_std_24h: "COP volatility over 24h",
  cop_proxy: "COP level",
  delta_t_supply_proxy: "supply delta-T",
  delta_t_refrigerant_proxy: "refrigerant delta-T",
  load_ratio: "load vs rated",
};

const state = {
  health: null,
  units: [],
  demoScenarios: [],
  recent: [],
  currentScore: null,
  currentTier: null,
  warmups: {},
};

const els = {
  apiDocsLink: document.getElementById("api-docs-link"),
  footerDocsLink: document.getElementById("footer-docs-link"),
  statusText: document.getElementById("status-text"),
  modeToggle: document.getElementById("mode-toggle"),
  refreshFleet: document.getElementById("refresh-fleet"),
  fleetTitle: document.getElementById("fleet-title"),
  fleetMessage: document.getElementById("fleet-message"),
  counts: {
    critical: document.getElementById("count-critical"),
    warning: document.getElementById("count-warning"),
    monitor: document.getElementById("count-monitor"),
    healthy: document.getElementById("count-healthy"),
    total: document.getElementById("count-total"),
  },
  scoreFleet: document.getElementById("score-fleet"),
  scoreManual: document.getElementById("score-manual"),
  scoreError: document.getElementById("score-error"),
  runLog: document.getElementById("run-log"),
  recentPulls: document.getElementById("recent-pulls"),
  scoreValue: document.getElementById("score-value"),
  scoreLabel: document.getElementById("score-label"),
  tierChip: document.getElementById("tier-chip"),
  healthScale: document.getElementById("health-scale"),
  crossCheck: document.getElementById("cross-check"),
  anomalyLine: document.getElementById("anomaly-line"),
  deflectionList: document.getElementById("deflection-list"),
};

window.addEventListener("DOMContentLoaded", () => {
  els.apiDocsLink.href = `${API_BASE}/docs`;
  els.footerDocsLink.href = `${API_BASE}/docs`;
  initMode();
  renderHealthScale();
  bindEvents();
  checkHealth();
  loadUnits();
  loadDemoReadings();
});

function bindEvents() {
  els.modeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "night" ? "day" : "night";
    setTheme(next, true);
  });
  els.refreshFleet.addEventListener("click", loadUnits);
  els.scoreFleet.addEventListener("click", scoreFleetUnit);
  els.scoreManual.addEventListener("click", scoreManualReading);
}

function initMode() {
  const theme = document.documentElement.dataset.theme || "day";
  setTheme(theme, false);
}

function setTheme(theme, persist) {
  const clean = theme === "night" ? "night" : "day";
  const mode = clean === "night" ? "dark" : "light";
  document.documentElement.dataset.theme = clean;
  els.modeToggle.setAttribute("aria-pressed", clean === "night" ? "true" : "false");
  els.modeToggle.setAttribute("aria-label", clean === "night" ? "Switch to light mode" : "Switch to dark mode");
  if (!persist) return;
  localStorage.setItem("mode", mode);
  if (location.hostname.endsWith("alvinalias.com")) {
    document.cookie = `mode=${mode}; Domain=.alvinalias.com; Path=/; Max-Age=31536000; SameSite=Lax`;
  }
}

async function checkHealth() {
  const started = performance.now();
  const warmup = scheduleWarmup("mini", started, false);
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    if (data.scorer_loaded) {
      state.health = data;
      els.statusText.textContent = `scorer ready · ${data.feature_count} features`;
    } else {
      els.statusText.textContent = "scorer not loaded";
    }
    finishWarmup(warmup);
    renderHealthScale(state.currentScore, state.currentTier);
  } catch {
    cancelWarmup(warmup);
    els.statusText.textContent = "server unreachable right now";
  }
}

async function loadUnits() {
  els.fleetMessage.classList.remove("is-error");
  els.fleetMessage.textContent = "pulling the fleet";
  setFleetCounts();
  try {
    const response = await fetch(`${API_BASE}/units`);
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const data = await response.json();
    state.units = Array.isArray(data.units) ? data.units : [];
    renderFleet(data);
  } catch (error) {
    state.units = [];
    els.fleetMessage.classList.add("is-error");
    els.fleetMessage.textContent = `Could not load the fleet. ${error.message}`;
  }
}

async function loadDemoReadings() {
  try {
    const response = await fetch(`${API_BASE}/demo-readings`);
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const data = await response.json();
    state.demoScenarios = Array.isArray(data.scenarios) ? data.scenarios : [];
  } catch (error) {
    state.demoScenarios = [];
    console.warn("Could not load demo readings", error);
  }
}

function renderFleet(data) {
  const stamp = data.snapshot_generated ? ` · scored ${data.snapshot_generated}` : "";
  els.fleetTitle.textContent = `The fleet · ${data.total} units, relative snapshot from the meter study${stamp}`;
  els.fleetMessage.textContent = "";
  setFleetCounts(data);
}

function setFleetCounts(data = {}) {
  const units = Array.isArray(data.units) ? data.units : state.units;
  const bands = summarizeFleetBands(units);
  els.counts.critical.textContent = formatCount(bands.highAttention);
  els.counts.warning.textContent = formatCount(bands.elevated);
  els.counts.monitor.textContent = formatCount(bands.watch);
  els.counts.healthy.textContent = formatCount(bands.normal);
  els.counts.total.textContent = formatCount(data.total ?? (units.length > 0 ? units.length : undefined));
}

async function scoreFleetUnit() {
  clearError();
  if (state.demoScenarios.length === 0) {
    await loadDemoReadings();
  }
  if (state.demoScenarios.length === 0) {
    showError("Could not load the demo readings.", "GET /demo-readings returned no scenarios");
    return;
  }

  const scenario = drawDemoScenario();

  setLog([
    `> ${scenario.title.toLowerCase()} · unit ${scenario.source_unit_id} drawn from curated readings`,
    "> complete historical features scored against this unit's baseline",
  ]);
  await scoreReading(scenario.reading, true, scenario);
}

async function scoreManualReading() {
  clearError();
  const reading = buildManualReading();
  if (!reading) {
    showError("Enter at least COP, both delta-T values, and load ratio.");
    return;
  }
  setLog(["> manual readings sent with estimated rolling context"]);
  await scoreReading(reading, false);
}

async function scoreReading(reading, recordRecent, context = null) {
  const started = performance.now();
  const warmup = scheduleWarmup("main", started, true);
  setBusy(true);
  try {
    const response = await fetch(`${API_BASE}/score?shap=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reading),
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const data = await response.json();
    finishWarmup(warmup);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    appendLog(`> scored in ${seconds} s · both detectors ran`);
    renderScore(data, context);
    if (recordRecent) {
      addRecent(data, context);
    }
  } catch (error) {
    cancelWarmup(warmup);
    const network = error instanceof TypeError;
    if (network) {
      showError(
        "Could not score this unit. This ML demo sleeps after extended inactivity; the Space may still be waking. Try again in a moment.",
        "network error",
      );
    } else {
      showError(`Could not score this unit. ${error.message}`);
    }
  } finally {
    setBusy(false);
  }
}

function buildManualReading() {
  const cop = readNumber("cop");
  const supply = readNumber("delta-t-supply");
  const refrigerant = readNumber("delta-t-refrigerant");
  const load = readNumber("load-ratio");
  if ([cop, supply, refrigerant, load].some((value) => value === null)) {
    return null;
  }
  const reading = {
    ...dateFields(),
    cop_proxy: cop,
    delta_t_supply_proxy: supply,
    delta_t_refrigerant_proxy: refrigerant,
    load_ratio: load,
    rolling_cop_mean_24h: cop,
    rolling_cop_std_24h: Math.max(0.05, Math.min(0.8, cop * 0.05)),
    rolling_load_mean_24h: load,
    rolling_cop_mean_168h: cop,
    cop_deviation_from_baseline: 0,
  };
  const airTemperature = readNumber("air-temperature");
  const hourOfDay = readInteger("hour-of-day");
  if (airTemperature !== null) {
    reading.air_temperature = airTemperature;
  }
  if (hourOfDay !== null) {
    reading.hour_of_day = hourOfDay;
  }
  return reading;
}

function dateFields() {
  const now = new Date();
  const day = now.getDay();
  return {
    hour_of_day: now.getHours(),
    day_of_week: day,
    is_weekend: day === 0 || day === 6 ? 1 : 0,
    month: now.getMonth() + 1,
  };
}

function drawDemoScenario() {
  const drawn = getDrawnScenarios();
  let pool = state.demoScenarios.filter((scenario) => !drawn.has(scenario.id));
  if (pool.length === 0) {
    sessionStorage.removeItem("hvacDrawnScenarios");
    pool = state.demoScenarios.slice();
  }
  const index = Math.floor(Math.random() * pool.length);
  const scenario = pool[index];
  drawn.add(scenario.id);
  sessionStorage.setItem("hvacDrawnScenarios", JSON.stringify(Array.from(drawn)));
  return scenario;
}

function getDrawnScenarios() {
  try {
    return new Set(JSON.parse(sessionStorage.getItem("hvacDrawnScenarios") || "[]"));
  } catch {
    return new Set();
  }
}

function renderScore(data, context = null) {
  const score = Number(data.health_score ?? 0);
  const tier = sanitizeTier(data.health_tier);
  state.currentScore = score;
  state.currentTier = tier;

  els.scoreValue.textContent = score.toFixed(1);
  els.scoreValue.classList.toggle("is-critical", tier === "critical");
  const label = context?.title ? context.title.toLowerCase() : `${tier} tier`;
  els.scoreLabel.textContent = `reading health · ${label}`;
  els.tierChip.hidden = false;
  els.tierChip.textContent = tier;
  els.tierChip.className = `tier-chip tier-chip--${tier}`;
  renderHealthScale(score, tier);
  renderCrossCheck(data);
  renderDeflections(data.top_shap_factors || []);

  if (Number(data.anomaly_flag) === 1) {
    els.anomalyLine.hidden = false;
  } else {
    els.anomalyLine.hidden = true;
  }
}

function renderHealthScale(score = null, tier = null) {
  const minX = 24;
  const maxX = 576;
  const y = 62;
  const scaleX = (value) => minX + (Math.max(0, Math.min(100, value)) / 100) * (maxX - minX);
  const minorTicks = [];
  for (let value = 0; value <= 100; value += 5) {
    const major = value % 25 === 0;
    minorTicks.push(`<line class="${major ? "major" : "minor"}" x1="${scaleX(value)}" y1="${major ? 50 : 54}" x2="${scaleX(value)}" y2="${major ? 74 : 70}" />`);
  }
  const labels = [0, 25, 50, 75, 100]
    .map((value) => `<text x="${scaleX(value)}" y="98">${value}</text>`)
    .join("");
  const boundaries = state.health?.tiers
    ? Object.entries(state.health.tiers)
        .map(([name, value]) => `
          <line class="boundary" x1="${scaleX(value)}" y1="24" x2="${scaleX(value)}" y2="76" />
          <text class="threshold-label" x="${scaleX(value)}" y="15">${name}</text>
        `)
        .join("")
    : "";
  const marker = score === null
    ? ""
    : `<polygon class="marker ${tier === "critical" ? "is-critical" : ""}" points="${scaleX(score)},44 ${scaleX(score) - 7},30 ${scaleX(score) + 7},30" />`;

  els.healthScale.innerHTML = `
    <svg class="health-svg" viewBox="0 0 600 112" role="img" aria-label="Health score scale: warning at 50, monitor at 70, healthy at 90">
      <line class="baseline" x1="${minX}" y1="${y}" x2="${maxX}" y2="${y}" />
      ${minorTicks.join("")}
      ${boundaries}
      ${marker}
      ${labels}
    </svg>
  `;
}

function renderCrossCheck(data) {
  const agree = Number(data.if_lof_agree) === 1;
  const hasValue = data.if_lof_agree !== null && data.if_lof_agree !== undefined;
  const flagged = Number(data.anomaly_flag) === 1;
  els.crossCheck.hidden = false;
  els.crossCheck.classList.toggle("is-care", hasValue && !agree);
  const strong = els.crossCheck.querySelector("strong");
  const detail = els.crossCheck.querySelector("span");
  if (!hasValue) {
    strong.textContent = "Cross-check unavailable · LOF did not return a comparison";
    detail.textContent = "review the Isolation Forest score without the secondary detector";
  } else if (agree) {
    strong.textContent = flagged
      ? "Cross-check · both detectors flag this reading"
      : "Cross-check · both detectors see a normal pattern";
    detail.textContent = "Isolation Forest and LOF agree; sample agreement is 91.3%";
  } else {
    strong.textContent = "Low-confidence cross-check · the detectors disagree";
    detail.textContent = "review this as an algorithm-dependent triage signal";
  }
}

function renderDeflections(factors) {
  if (factors.length === 0) {
    els.deflectionList.innerHTML = "";
    return;
  }
  const maxAbs = Math.max(...factors.map((factor) => Math.abs(Number(factor.shap_value) || 0)), 0.001);
  els.deflectionList.innerHTML = factors.map((factor) => {
    const value = Number(factor.shap_value) || 0;
    const width = Math.max(4, Math.abs(value) / maxAbs * 48);
    const negative = value < 0;
    return `
      <div class="deflection__row">
        <div class="deflection__name">${escapeHtml(glossFeature(factor.feature))}</div>
        <div class="deflection__track" title="API field: top_shap_factors">
          <span class="deflection__bar ${negative ? "is-negative" : "is-positive"}" style="width: ${width}%"></span>
        </div>
        <div class="deflection__value">${formatSigned(value)}</div>
      </div>
    `;
  }).join("");
}

function addRecent(data, context = null) {
  const unit = context
    ? `${context.band.replace("_", " ")} · unit ${context.source_unit_id}`
    : data.building_id === null || data.building_id === undefined ? "manual" : `unit ${data.building_id}`;
  state.recent.unshift({
    unit,
    score: Number(data.health_score || 0).toFixed(1),
    tier: sanitizeTier(data.health_tier),
  });
  state.recent = state.recent.slice(0, 5);
  els.recentPulls.innerHTML = state.recent.map((item) => `
    <tr>
      <td>${escapeHtml(item.unit)}</td>
      <td>${item.score}</td>
      <td>${item.tier}</td>
    </tr>
  `).join("");
}

function summarizeFleetBands(units) {
  const total = Array.isArray(units) ? units.length : 0;
  if (total === 0) {
    return {};
  }
  const highAttention = Math.ceil(total * 0.15);
  const elevated = Math.ceil(total * 0.35);
  const watch = Math.ceil(total * 0.30);
  const normal = Math.max(0, total - highAttention - elevated - watch);
  return { highAttention, elevated, watch, normal };
}

function scheduleWarmup(target, started, logToRun) {
  const handle = { target, started, active: false, timer: null, logToRun };
  handle.timer = window.setTimeout(() => startWarmup(handle), 2500);
  return handle;
}

function startWarmup(handle) {
  handle.active = true;
  const box = document.getElementById(`warmup-${handle.target}`);
  box.hidden = false;
  setWarmMarker(handle.target, 60, false);
  setWarmText(handle.target, "60", "estimated seconds to warm");
  document.getElementById(`warm-log-${handle.target}`).textContent = "> warm-up estimate counting · this is an estimate, not progress";
  if (handle.logToRun) {
    appendLog("> server was asleep · sent the wake call");
    appendLog("> warm-up estimate counting · this is an estimate, not progress");
  }

  handle.interval = window.setInterval(() => {
    const elapsed = Math.floor((performance.now() - handle.started) / 1000);
    const remaining = 60 - elapsed;
    if (remaining > 0) {
      setWarmMarker(handle.target, remaining, false);
      setWarmText(handle.target, String(remaining), "estimated seconds to warm");
    } else {
      setWarmMarker(handle.target, 0, true);
      setWarmText(handle.target, String(elapsed), "seconds elapsed · still starting");
      document.getElementById(`warm-log-${handle.target}`).textContent = "> past the usual window · still waiting, counting up honestly";
      if (handle.logToRun && !handle.overrunLogged) {
        appendLog("> past the usual window · still waiting, counting up honestly");
        handle.overrunLogged = true;
      }
    }
  }, 1000);
  state.warmups[handle.target] = handle;
}

function finishWarmup(handle) {
  window.clearTimeout(handle.timer);
  if (!handle.active) return;
  window.clearInterval(handle.interval);
  const measured = ((performance.now() - handle.started) / 1000).toFixed(1);
  setWarmMarker(handle.target, 0, false);
  setWarmText(handle.target, "0", "ready");
  document.getElementById(`warm-log-${handle.target}`).textContent = `> awake · measured wake time ${measured} s`;
  if (handle.logToRun) {
    appendLog(`> awake · measured wake time ${measured} s`);
  }
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.setTimeout(() => {
    document.getElementById(`warmup-${handle.target}`).hidden = true;
  }, reduce ? 0 : 4000);
  delete state.warmups[handle.target];
}

function cancelWarmup(handle) {
  window.clearTimeout(handle.timer);
  if (handle.active) {
    window.clearInterval(handle.interval);
    document.getElementById(`warmup-${handle.target}`).hidden = true;
  }
  delete state.warmups[handle.target];
}

function setWarmText(target, number, label) {
  document.getElementById(`warm-number-${target}`).textContent = number;
  document.getElementById(`warm-label-${target}`).textContent = label;
}

function setWarmMarker(target, seconds, overrun) {
  const marker = document.getElementById(`warm-marker-${target}`);
  const x = 20 + (Math.max(0, Math.min(60, seconds)) / 60) * 260;
  marker.setAttribute("x1", x);
  marker.setAttribute("x2", x);
  marker.classList.toggle("is-overrun", overrun);
}

function setBusy(isBusy) {
  els.scoreFleet.disabled = isBusy;
  els.scoreManual.disabled = isBusy;
}

function setLog(lines) {
  els.runLog.textContent = lines.join("\n");
}

function appendLog(line) {
  const prefix = els.runLog.textContent ? "\n" : "";
  els.runLog.textContent += `${prefix}${line}`;
}

function clearError() {
  els.scoreError.hidden = true;
  els.scoreError.textContent = "";
}

function showError(message, detail = "") {
  els.scoreError.hidden = false;
  els.scoreError.textContent = message;
  if (detail) {
    const code = document.createElement("code");
    code.textContent = detail;
    els.scoreError.appendChild(code);
  }
}

async function readError(response) {
  try {
    const data = await response.json();
    return typeof data.detail === "string" ? data.detail : response.statusText;
  } catch {
    return response.statusText;
  }
}

function readNumber(id) {
  const raw = document.getElementById(id).value.trim();
  if (raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function readInteger(id) {
  const raw = document.getElementById(id).value.trim();
  if (raw === "") return null;
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) ? value : null;
}

function sanitizeTier(value) {
  return ["healthy", "monitor", "warning", "critical"].includes(value) ? value : "critical";
}

function glossFeature(feature) {
  return FEATURE_GLOSSES[feature] || String(feature || "").replaceAll("_", " ");
}

function formatCount(value) {
  return Number.isFinite(Number(value)) ? String(value) : "--";
}

function formatSigned(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
