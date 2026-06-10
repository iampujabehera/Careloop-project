'use strict';

// Vercel serverless function — /api/settings/risk-weights
//   GET   → return the default weights so the sliders can initialise.
//   PATCH → accept + echo the posted weights.
//
// This is stateless on Vercel (no shared server memory across invocations).
// That's fine: the client keeps the current weights in CURRENT_RISK_WEIGHTS and
// passes them to /api/risk/score-signals on every score, so the displayed score
// always reflects the sliders even though we don't persist them server-side.

const { DEFAULT_RISK_WEIGHTS, sanitizeWeights } = require('../../careloopWeightedEngine');

module.exports = (req, res) => {
  if (req.method === 'GET') {
    res.status(200).json({ weights: { ...DEFAULT_RISK_WEIGHTS }, defaults: { ...DEFAULT_RISK_WEIGHTS } });
    return;
  }

  if (req.method === 'PATCH') {
    let body = req.body;
    if (typeof body === 'string') {
      try { body = JSON.parse(body); } catch (_) { body = {}; }
    }
    res.status(200).json({ ok: true, weights: sanitizeWeights(body) });
    return;
  }

  res.status(405).json({ error: 'Method not allowed' });
};
