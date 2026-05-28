/**
 * app.js — HVAC Equipment Health Dashboard
 *
 * Responsibilities:
 *   - Health check on load (GET /health)
 *   - Load fleet summary (GET /units) → fleet banner + alert table
 *   - Score a unit (POST /score) → health gauge + SHAP panel
 *   - Health gauge SVG arc animation
 *   - Alert table row click → auto-fill and score
 */

// ── Config ────────────────────────────────────────────────────────────────
// Override with: localStorage.setItem("HVAC_API_BASE", "https://your-api.onrender.com")
const API_BASE = window.HVAC_API_BASE || localStorage.getItem("HVAC_API_BASE") || "http://localhost:8000";

const TIER_COLORS = {
  healthy:  "#3ecf8e",
  monitor:  "#63b3ed",
  warning:  "#f5a623",
  critical: "#e53e3e",
};

// Gauge arc path total length (half-circle: π × r = π × 80 ≈ 251.3)
const GAUGE_ARC_LENGTH = 251.3;

// ── DOM refs ──────────────────────────────────────────────────────────────
const $statusDot  = document.getElementById("status-dot");
const $statusText = document.getElementById("status-text");
const $scoreBtn   = document.getElementById("score-btn");
const $scoreError = document.getElementById("score-error");
const $gaugeFill  = document.getElementById("gauge-fill");
const $gaugeScore = document.getElementById("gauge-score");
const $tierBadge  = document.getElementById("tier-badge");
const $anomalyFlag = document.getElementById("anomaly-flag");
const $shapList   = document.getElementById("shap-list");
const $scoreStats = document.getElementById("score-stats");
const $alertTbody = document.getElementById("alert-tbody");
const $unitSelect = document.getElementById("unit-select");
const $refreshBtn = document.getElementById("refresh-btn");
const $apiDocsLink = document.getElementById("api-docs-link");

// ── Startup ───────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  $apiDocsLink.href = `${API_BASE}/docs`;
  checkHealth();
  loadUnits();
  $scoreBtn.addEventListener("click", handleScore);
  $refreshBtn.addEventListener("click", loadUnits);
  $unitSelect.addEventListener("change", handleUnitSelectChange);
});

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.scorer_loaded) {
      setStatus("ok", `API ready · ${data.feature_count} features`);
    } else {
      setStatus("degraded", "API running · models not loaded");
    }
  } catch {
    setStatus("error", "API unreachable");
  }
}

function setStatus(state, text) {
  $statusDot.className = `status-dot ${state}`;
  $statusText.textContent = text;
}

// ── Fleet overview ────────────────────────────────────────────────────────
async function loadUnits() {
  $alertTbody.innerHTML = `<tr><td colspan="4" class="table-placeholder"><span class="spinner"></span>Loading…</td></tr>`;
  try {
    const res = await fetch(`${API_BASE}/units`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderFleetBanner(data);
    renderAlertTable(data.units);
    populateUnitSelect(data.units);
  } catch (e) {
    $alertTbody.innerHTML = `<tr><td colspan="4" class="table-placeholder">Could not load units (${e.message})</td></tr>`;
  }
}

function renderFleetBanner(data) {
  document.getElementById("count-critical").textContent = data.n_critical;
  document.getElementById("count-warning").textContent  = data.n_warning;
  document.getElementById("count-monitor").textContent  = data.n_monitor;
  document.getElementById("count-healthy").textContent  = data.n_healthy;
  document.getElementById("count-total").textContent    = data.total;
}

function renderAlertTable(units) {
  if (!units || units.length === 0) {
    $alertTbody.innerHTML = `<tr><td colspan="4" class="table-placeholder">No units loaded yet.</td></tr>`;
    return;
  }
  $alertTbody.innerHTML = units.map(u => `
    <tr data-building="${escapeHtml(u.building_id ?? "")}" class="unit-row">
      <td style="font-family:monospace;font-size:11px;">${escapeHtml(u.building_id ?? "—")}</td>
      <td style="font-weight:600;color:${TIER_COLORS[sanitizeTier(u.health_tier)] || "#e2e8f0"};">
        ${u.health_score?.toFixed(1) ?? "—"}
      </td>
      <td><span class="tier-chip ${sanitizeTier(u.health_tier)}">${escapeHtml(u.health_tier ?? "unknown")}</span></td>
      <td>${u.anomaly_flag ? '<span class="anomaly-dot" title="Anomaly">●</span>' : ''}</td>
    </tr>
  `).join("");

  // Row click → auto-select and score
  document.querySelectorAll(".unit-row").forEach(row => {
    row.addEventListener("click", () => {
      const bid = row.dataset.building;
      $unitSelect.value = bid;
      handleUnitSelectChange();
      handleScore();
    });
  });
}

function populateUnitSelect(units) {
  $unitSelect.innerHTML = `<option value="">— pick a unit or enter manual readings —</option>`;
  (units || []).forEach(u => {
    const opt = document.createElement("option");
    opt.value = u.building_id;
    const tierEmoji = { healthy: "✅", monitor: "🔵", warning: "⚠️", critical: "🔴" }[u.health_tier] ?? "";
    opt.textContent = `${tierEmoji} ${u.building_id}  (${u.health_score?.toFixed(1) ?? "?"})`;
    $unitSelect.appendChild(opt);
  });
}

// ── Unit select → fill hidden field for API call ──────────────────────────
let _selectedUnit = null;

function handleUnitSelectChange() {
  const bid = $unitSelect.value;
  _selectedUnit = bid || null;
}

// ── Scoring ───────────────────────────────────────────────────────────────
async function handleScore() {
  clearError();
  $scoreBtn.disabled = true;
  $scoreBtn.innerHTML = `<span class="spinner"></span>Scoring…`;

  const reading = buildReading();
  if (!reading) {
    showError("Enter at least COP, ΔT Supply, ΔT Refrigerant, and Load Ratio.");
    resetBtn();
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/score?shap=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reading),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    renderGauge(data);
    renderSHAP(data);
    renderStats(data);
  } catch (e) {
    showError(e.message);
  } finally {
    resetBtn();
  }
}

function buildReading() {
  // If a unit is selected from the table, only send building_id + any manual overrides
  const cop     = parseFloat(document.getElementById("cop").value);
  const dtSupp  = parseFloat(document.getElementById("delta-t-supply").value);
  const dtRef   = parseFloat(document.getElementById("delta-t-refrig").value);
  const load    = parseFloat(document.getElementById("load-ratio").value);

  // Require at least one manual value OR a selected unit
  if (_selectedUnit && isNaN(cop) && isNaN(dtSupp)) {
    // Unit selected but no manual reading — use default demo values
    return {
      building_id: _selectedUnit,
      cop_proxy: 3.0,
      delta_t_supply_proxy: 9.0,
      delta_t_refrigerant_proxy: 17.0,
      load_ratio: 0.70,
      hour_of_day: new Date().getHours(),
      day_of_week: new Date().getDay(),
      is_weekend: [0,6].includes(new Date().getDay()) ? 1 : 0,
      month: new Date().getMonth() + 1,
    };
  }

  if (isNaN(cop) || isNaN(dtSupp) || isNaN(dtRef) || isNaN(load)) return null;

  return {
    building_id: _selectedUnit || null,
    cop_proxy: cop,
    delta_t_supply_proxy: dtSupp,
    delta_t_refrigerant_proxy: dtRef,
    load_ratio: load,
    air_temperature: parseFloatOrNull("air-temp"),
    hour_of_day: parseIntOrNull("hour"),
    day_of_week: new Date().getDay(),
    is_weekend: [0,6].includes(new Date().getDay()) ? 1 : 0,
    month: new Date().getMonth() + 1,
  };
}

function parseFloatOrNull(id) {
  const v = parseFloat(document.getElementById(id)?.value);
  return isNaN(v) ? null : v;
}
function parseIntOrNull(id) {
  const v = parseInt(document.getElementById(id)?.value, 10);
  return isNaN(v) ? null : v;
}

// ── Gauge rendering ───────────────────────────────────────────────────────
function renderGauge(data) {
  const score = data.health_score ?? 0;
  const tier  = sanitizeTier(data.health_tier);
  const color = TIER_COLORS[tier] || "#e2e8f0";

  // Arc: fill = (score / 100) × total arc length
  const filled = (score / 100) * GAUGE_ARC_LENGTH;
  const empty  = GAUGE_ARC_LENGTH - filled;
  $gaugeFill.setAttribute("stroke-dasharray", `${filled} ${empty}`);
  $gaugeFill.setAttribute("stroke", color);

  $gaugeScore.textContent = score.toFixed(0);
  $gaugeScore.setAttribute("fill", color);

  // Tier badge
  $tierBadge.textContent  = tier.toUpperCase();
  $tierBadge.className    = `tier-badge ${tier}`;
  $tierBadge.classList.remove("hidden");

  // Anomaly flag
  if (data.anomaly_flag === 1) {
    $anomalyFlag.classList.remove("hidden");
  } else {
    $anomalyFlag.classList.add("hidden");
  }
}

// ── SHAP panel ────────────────────────────────────────────────────────────
function renderSHAP(data) {
  const factors = data.top_shap_factors;
  if (!factors || factors.length === 0) {
    $shapList.innerHTML = `<div class="shap-placeholder">No SHAP data returned.</div>`;
    return;
  }

  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

  $shapList.innerHTML = factors.map(f => {
    const widthPct = (Math.abs(f.shap_value) / maxAbs * 100).toFixed(1);
    const dirClass = f.direction === "worsens_health" ? "worsens" : "improves";
    const dirLabel = f.direction === "worsens_health" ? "↑ worsens" : "↓ improves";
    return `
      <div class="shap-item">
        <div class="shap-feature">
          <span class="shap-feature-name">${escapeHtml(f.feature)}</span>
          <span class="shap-direction ${dirClass}">${dirLabel}</span>
        </div>
        <div class="shap-bar-bg">
          <div class="shap-bar-fill ${dirClass}" style="width:${widthPct}%"></div>
        </div>
        <div class="shap-value">value: ${f.feature_value} · SHAP: ${f.shap_value.toFixed(4)}</div>
      </div>
    `;
  }).join("");
}

// ── Score stats ───────────────────────────────────────────────────────────
function renderStats(data) {
  $scoreStats.classList.remove("hidden");
  document.getElementById("stat-iforest").textContent = data.iforest_score?.toFixed(4) ?? "—";
  document.getElementById("stat-anomaly").textContent = data.anomaly_flag === 1 ? "⚠ Yes" : "✓ No";
  document.getElementById("stat-lof").textContent     = data.lof_flag != null
    ? (data.lof_flag === 1 ? "⚠ Yes" : "✓ No") : "—";
  document.getElementById("stat-agree").textContent   = data.if_lof_agree != null
    ? (data.if_lof_agree === 1 ? "✓ Yes" : "✗ No") : "—";
}

// ── Helpers ───────────────────────────────────────────────────────────────
function showError(msg) {
  $scoreError.textContent = msg;
  $scoreError.classList.remove("hidden");
}
function clearError() {
  $scoreError.textContent = "";
  $scoreError.classList.add("hidden");
}
function resetBtn() {
  $scoreBtn.disabled = false;
  $scoreBtn.textContent = "Score Unit";
}

function sanitizeTier(value) {
  return ["healthy", "monitor", "warning", "critical"].includes(value) ? value : "critical";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
