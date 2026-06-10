'use strict';

// Vercel serverless function — POST /api/risk/score-signals
//
// Risk scoring is pure, stateless math, so it runs perfectly as a serverless
// function (no database, no secrets). This makes the risk score work on the
// deployed Vercel URL, not just the local Node server.
//
// Body: { signals: {...}, weights?: {...} }
//   • signals  — the patient's risk signals (lab_overdue, cost_barrier, etc.)
//   • weights  — optional override (the client passes the live slider values so
//                tuning the Risk Detection Settings changes the score here too).

const { normalizeSignals, computeRiskBreakdownFromFeatures, triageFromScore } = require('../../careloopWeightedEngine');

module.exports = (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch (_) { body = {}; }
  }
  body = body || {};

  const features = normalizeSignals(body.signals);
  const breakdown = computeRiskBreakdownFromFeatures(features, body.weights);
  res.status(200).json({ ok: true, ...breakdown, triage: triageFromScore(breakdown.score) });
};
