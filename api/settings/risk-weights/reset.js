'use strict';

// Vercel serverless function — POST /api/settings/risk-weights/reset
// Returns the recommended default weights (stateless reset).

const { DEFAULT_RISK_WEIGHTS } = require('../../../careloopWeightedEngine');

module.exports = (req, res) => {
  res.status(200).json({ ok: true, weights: { ...DEFAULT_RISK_WEIGHTS } });
};
