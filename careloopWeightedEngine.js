'use strict';

// CareLoop weighted risk engine — the single scoring brain.
//
// Pure, stateless, ZERO npm dependencies, so the exact same logic runs in:
//   • the local Express server (server.js), and
//   • the Vercel serverless functions (api/risk/score-signals.js, etc.)
//
//   Z = β0 + Σ(βi · xi)      linear risk index (β = configurable weights)
//   P = 1 / (1 + e^(-Z))     sigmoid → probability in [0,1]
//   score = round(P × 100)   0–100 risk score
//
// The β weights are surfaced in the UI as the "Risk Detection Settings" sliders,
// so adjusting a slider directly changes how much a signal moves the score.

const DEFAULT_RISK_WEIGHTS = {
  // ── Power BI–derived (Africa fit) ──
  intercept:                -3.18,  // β0   log-odds of baseline non-avail rate
  cost_barrier:              0.9,   // β1   high-cost / affordability concern
  missed_appointments:       0.7,   // β2   past non-availability (capped at 3)
  silence_days:              0.35,  // β3   infrequent/unreachable (per day, capped 14)
  // ── CareLoop OPD extensions ──
  medication_nonadherence:   1.2,   // β4   2+ consecutive missed doses
  lab_overdue:               0.5,   // β5   per overdue lab (capped at 5)
  chronic_condition:         0.4,   // β6   chronic episode flag
  anxiety_flag:              0.6,   // β7   anxiety/fear inferred from sentiment
  tier2_signal:              4.0,   // β8   red-flag clinical symptom (forces emergent)
  tier1_signal:              1.0,   // β9   yellow-flag clinical symptom
};

// [feature key, human label] — label must match the client FACTOR_DISPLAY keys
// so the breakdown panel can render each contribution with a friendly name.
const CONTRIBUTION_LABELS = [
  ['cost_barrier',            'Cost barrier flagged'],
  ['medication_nonadherence', 'Medication non-adherence'],
  ['lab_overdue',             'Lab orders overdue'],
  ['silence_days',            'Days silent'],
  ['chronic_condition',       'Chronic condition'],
  ['missed_appointments',     'Missed appointments'],
  ['anxiety_flag',            'Anxiety / fear inferred'],
  ['tier2_signal',            'Red-flag clinical signal'],
  ['tier1_signal',            'Yellow-flag clinical signal'],
];

// Clamp/normalise a loosely-typed signals object into the feature shape the
// engine expects (same caps as the event-derived features in server.js).
function normalizeSignals(sig) {
  sig = sig || {};
  const n = (v, cap) => Math.min(cap, Math.max(0, parseInt(v, 10) || 0));
  return {
    cost_barrier:            sig.cost_barrier ? 1 : 0,
    medication_nonadherence: sig.medication_nonadherence ? 1 : 0,
    lab_overdue:             n(sig.lab_overdue, 5),
    silence_days:            n(sig.silence_days, 14),
    chronic_condition:       sig.chronic_condition ? 1 : 0,
    missed_appointments:     n(sig.missed_appointments, 3),
    anxiety_flag:            sig.anxiety_flag ? 1 : 0,
    tier2_signal:            sig.tier2_signal ? 1 : 0,
    tier1_signal:            sig.tier1_signal ? 1 : 0,
  };
}

// Merge a (possibly partial / untrusted) weights object onto the defaults,
// keeping only finite numeric overrides for known keys.
function sanitizeWeights(weights) {
  const out = { ...DEFAULT_RISK_WEIGHTS };
  if (weights && typeof weights === 'object') {
    for (const k of Object.keys(DEFAULT_RISK_WEIGHTS)) {
      const v = Number(weights[k]);
      if (Number.isFinite(v)) out[k] = v;
    }
  }
  return out;
}

// Apply the given weights to a feature set → full breakdown + 0–100 score.
function computeRiskBreakdownFromFeatures(features, weights) {
  const w = sanitizeWeights(weights);

  const contributions = [
    { name: 'Baseline (intercept)', weight: w.intercept, value: 1, contribution: w.intercept },
  ];
  for (const [key, label] of CONTRIBUTION_LABELS) {
    const value = features[key] == null ? 0 : features[key];
    contributions.push({ name: label, weight: w[key], value, contribution: w[key] * value });
  }

  const Z = contributions.reduce((sum, c) => sum + c.contribution, 0);
  const P = 1 / (1 + Math.exp(-Z));
  const score = Math.round(P * 100);

  return { features, weights: w, contributions, Z, probability: P, score };
}

function triageFromScore(score) {
  if (score > 80) return 'emergent';
  if (score >= 60) return 'urgent';
  return 'normal';
}

module.exports = {
  DEFAULT_RISK_WEIGHTS,
  CONTRIBUTION_LABELS,
  normalizeSignals,
  sanitizeWeights,
  computeRiskBreakdownFromFeatures,
  triageFromScore,
};
