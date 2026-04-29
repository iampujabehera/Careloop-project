require('dotenv').config();

const path = require('path');
const express = require('express');
const cors = require('cors');
const Database = require('better-sqlite3');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const DB_PATH = path.join(__dirname, 'careloop.db');

// ---------------------------------------------------------------------------
// Database setup
// ---------------------------------------------------------------------------

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patientName     TEXT    NOT NULL,
    mrn             TEXT    NOT NULL,
    age             INTEGER,
    mobile          TEXT,
    language        TEXT,
    insurance       TEXT,
    doctorName      TEXT,
    department      TEXT,
    diagnosis       TEXT,
    icdCode         TEXT,
    episodeType     TEXT    NOT NULL DEFAULT 'acute',
    medications     TEXT    NOT NULL DEFAULT '[]',
    labs            TEXT    NOT NULL DEFAULT '[]',
    instructions    TEXT    NOT NULL DEFAULT '[]',
    followUp        TEXT,
    riskScore       INTEGER NOT NULL DEFAULT 0,
    triageSegment   TEXT    NOT NULL DEFAULT 'normal',
    status          TEXT    NOT NULL DEFAULT 'active',
    careGap         TEXT,
    createdAt       TEXT    NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episodeId   INTEGER NOT NULL,
    eventType   TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    createdAt   TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (episodeId) REFERENCES episodes(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_events_episode ON events(episodeId, createdAt);
`);

// ---------------------------------------------------------------------------
// Risk Engine — Logistic Regression
// ---------------------------------------------------------------------------
//
// Z = β0 + Σ(βi · xi)         linear risk index
// P = 1 / (1 + e^(-Z))        sigmoid → probability in [0,1]
// score = round(P × 100)      0–100 risk score
//
// Methodology aligned with the Power BI regression model fitted on Africa
// hospital data (Dump_Africa_1):
//   • Intercept = LN(base_rate / (1 - base_rate))  — log-odds of baseline non-availability
//   • Coefficients = lift ratios (non-avail rate within risk segment ÷ baseline rate),
//     converted to log-odds form for use in the linear index.
//   • Three factors from Power BI fit: high-cost, infrequent-visitor, past-non-avail.
//   • Extended for CareLoop OPD context with: medication adherence, lab overdue,
//     silence days, chronic, anxiety, and clinical tier signals.
//
// Coefficients (β) are configurable per client via Settings → Risk Engine.
//
// triageSegment:
//   score >  80           → emergent (red)
//   score 60..80          → urgent   (amber)
//   score <  60           → normal   (green)

const DEFAULT_RISK_WEIGHTS = {
  // ── Power BI–derived (Africa fit) ──
  intercept:                -3.18,  // β0   evaluated log-odds of baseline non-avail rate (Power BI)
  cost_barrier:              0.9,   // β1   maps to Coeff_High_Cost (lift ratio in log-odds form)
  missed_appointments:       0.7,   // β2   maps to Coeff_Past_Non_Avail (per occurrence, capped at 3)
  silence_days:              0.35,  // β3   maps to Coeff_Infrequent_Visitor (per day, capped at 14)
  // ── CareLoop OPD extensions ──
  medication_nonadherence:   1.2,   // β4   2+ consecutive missed doses
  lab_overdue:               0.5,   // β5   per overdue lab (capped at 5)
  chronic_condition:         0.4,   // β6   chronic episode flag
  anxiety_flag:              0.6,   // β7   anxiety/fear inferred from LLM sentiment
  tier2_signal:              4.0,   // β8   red-flag clinical symptom (forces emergent)
  tier1_signal:              1.0,   // β9   yellow-flag clinical symptom
};

// Mutable copy — updated from Settings UI via PATCH /api/settings/risk-weights.
let RISK_WEIGHTS = { ...DEFAULT_RISK_WEIGHTS };

function extractFeatures(events) {
  const labOverdue = events.filter(e => e.eventType === 'LAB_OVERDUE').length;
  const contactFailed = events.filter(e => e.eventType === 'CONTACT_ATTEMPT_FAILED').length;

  let maxConsecutiveMissed = 0;
  let currentRun = 0;
  for (const e of events) {
    if (e.eventType === 'MEDICATION_MISSED') {
      currentRun += 1;
      if (currentRun > maxConsecutiveMissed) maxConsecutiveMissed = currentRun;
    } else if (e.eventType === 'MEDICATION_TAKEN' || e.eventType === 'MEDICATION_ACKNOWLEDGED') {
      currentRun = 0;
    }
  }

  const silenceDays = Math.min(contactFailed, 14);
  const labOverdueCapped = Math.min(labOverdue, 5);
  const missedAppointments = events.filter(e => e.eventType === 'MISSED_APPOINTMENT').length;

  return {
    cost_barrier:            events.some(e => e.eventType === 'COST_BARRIER_FLAGGED') ? 1 : 0,
    medication_nonadherence: maxConsecutiveMissed >= 2 ? 1 : 0,
    lab_overdue:             labOverdueCapped,
    silence_days:            silenceDays,
    chronic_condition:       events.some(e => e.eventType === 'CHRONIC_FLAG') ? 1 : 0,
    missed_appointments:     Math.min(missedAppointments, 3),
    anxiety_flag:            events.some(e => e.eventType === 'ANXIETY_FLAG') ? 1 : 0,
    tier2_signal:            events.some(e => e.eventType === 'TIER2_SIGNAL') ? 1 : 0,
    tier1_signal:            events.some(e => e.eventType === 'TIER1_SIGNAL') ? 1 : 0,
  };
}

function computeRiskFromEvents(events) {
  return computeRiskBreakdown(events).score;
}

function computeRiskBreakdown(events) {
  const features = extractFeatures(events);
  const w = RISK_WEIGHTS;

  const contributions = [
    { name: 'Baseline (intercept)',         weight: w.intercept,              value: 1,                          contribution: w.intercept },
    { name: 'Cost barrier flagged',         weight: w.cost_barrier,           value: features.cost_barrier,      contribution: w.cost_barrier * features.cost_barrier },
    { name: 'Medication non-adherence',     weight: w.medication_nonadherence,value: features.medication_nonadherence, contribution: w.medication_nonadherence * features.medication_nonadherence },
    { name: 'Lab orders overdue',           weight: w.lab_overdue,            value: features.lab_overdue,       contribution: w.lab_overdue * features.lab_overdue },
    { name: 'Days silent',                  weight: w.silence_days,           value: features.silence_days,      contribution: w.silence_days * features.silence_days },
    { name: 'Chronic condition',            weight: w.chronic_condition,      value: features.chronic_condition, contribution: w.chronic_condition * features.chronic_condition },
    { name: 'Missed appointments',          weight: w.missed_appointments,    value: features.missed_appointments, contribution: w.missed_appointments * features.missed_appointments },
    { name: 'Anxiety / fear inferred',      weight: w.anxiety_flag,           value: features.anxiety_flag,      contribution: w.anxiety_flag * features.anxiety_flag },
    { name: 'Red-flag clinical signal',     weight: w.tier2_signal,           value: features.tier2_signal,      contribution: w.tier2_signal * features.tier2_signal },
    { name: 'Yellow-flag clinical signal',  weight: w.tier1_signal,           value: features.tier1_signal,      contribution: w.tier1_signal * features.tier1_signal },
  ];

  const Z = contributions.reduce((sum, c) => sum + c.contribution, 0);
  const P = 1 / (1 + Math.exp(-Z));
  const score = Math.round(P * 100);

  return {
    features,
    weights: { ...w },
    contributions,
    Z,
    probability: P,
    score,
  };
}

function triageFromScore(score) {
  if (score > 80) return 'emergent';
  if (score >= 60) return 'urgent';
  return 'normal';
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

function parseEpisodeRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    patientName: row.patientName,
    mrn: row.mrn,
    age: row.age,
    mobile: row.mobile,
    language: row.language,
    insurance: row.insurance,
    doctorName: row.doctorName,
    department: row.department,
    diagnosis: row.diagnosis,
    icdCode: row.icdCode,
    episodeType: row.episodeType,
    medications: JSON.parse(row.medications || '[]'),
    labs: JSON.parse(row.labs || '[]'),
    instructions: JSON.parse(row.instructions || '[]'),
    followUp: row.followUp ? JSON.parse(row.followUp) : null,
    riskScore: row.riskScore,
    triageSegment: row.triageSegment,
    status: row.status,
    careGap: row.careGap,
    createdAt: row.createdAt,
  };
}

function getEventsForEpisode(episodeId) {
  const rows = db
    .prepare(
      'SELECT id, episodeId, eventType, metadata, createdAt FROM events WHERE episodeId = ? ORDER BY id ASC'
    )
    .all(episodeId);
  return rows.map(r => ({
    id: r.id,
    episodeId: r.episodeId,
    eventType: r.eventType,
    metadata: JSON.parse(r.metadata || '{}'),
    createdAt: r.createdAt,
  }));
}

function recomputeAndPersistRisk(episodeId) {
  const events = getEventsForEpisode(episodeId);
  const riskScore = computeRiskFromEvents(events);
  const triageSegment = triageFromScore(riskScore);
  db.prepare(
    'UPDATE episodes SET riskScore = ?, triageSegment = ? WHERE id = ?'
  ).run(riskScore, triageSegment, episodeId);
  return { riskScore, triageSegment };
}

function loadEpisode(id) {
  const row = db.prepare('SELECT * FROM episodes WHERE id = ?').get(id);
  if (!row) return null;
  const episode = parseEpisodeRow(row);
  episode.events = getEventsForEpisode(id);
  return episode;
}

const insertEpisodeStmt = db.prepare(`
  INSERT INTO episodes (
    patientName, mrn, age, mobile, language, insurance,
    doctorName, department, diagnosis, icdCode, episodeType,
    medications, labs, instructions, followUp,
    riskScore, triageSegment, status, careGap
  ) VALUES (
    @patientName, @mrn, @age, @mobile, @language, @insurance,
    @doctorName, @department, @diagnosis, @icdCode, @episodeType,
    @medications, @labs, @instructions, @followUp,
    @riskScore, @triageSegment, @status, @careGap
  )
`);

function createEpisode(payload) {
  const row = {
    patientName: payload.patientName,
    mrn: String(payload.mrn),
    age: payload.age ?? null,
    mobile: payload.mobile ?? null,
    language: payload.language ?? null,
    insurance: payload.insurance ?? null,
    doctorName: payload.doctorName ?? null,
    department: payload.department ?? null,
    diagnosis: payload.diagnosis ?? null,
    icdCode: payload.icdCode ?? null,
    episodeType: payload.episodeType || 'acute',
    medications: JSON.stringify(payload.medications || []),
    labs: JSON.stringify(payload.labs || []),
    instructions: JSON.stringify(payload.instructions || []),
    followUp: payload.followUp ? JSON.stringify(payload.followUp) : null,
    riskScore: 0,
    triageSegment: 'normal',
    status: payload.status || 'active',
    careGap: payload.careGap ?? null,
  };
  const result = insertEpisodeStmt.run(row);
  const id = result.lastInsertRowid;
  recomputeAndPersistRisk(id);
  return loadEpisode(id);
}

const insertEventStmt = db.prepare(
  'INSERT INTO events (episodeId, eventType, metadata) VALUES (?, ?, ?)'
);

function appendEvent(episodeId, eventType, metadata = {}) {
  insertEventStmt.run(episodeId, eventType, JSON.stringify(metadata || {}));
  recomputeAndPersistRisk(episodeId);
  return loadEpisode(episodeId);
}

// ---------------------------------------------------------------------------
// Seed data
// ---------------------------------------------------------------------------

function seedIfEmpty() {
  const { count } = db.prepare('SELECT COUNT(*) AS count FROM episodes').get();
  if (count > 0) return;

  const nora = createEpisode({
    patientName: 'Nora Al-Ansari',
    mrn: '44201',
    age: 38,
    mobile: '+971551234567',
    language: 'Arabic',
    insurance: 'Daman Enhanced',
    doctorName: 'Dr. Sam Willis',
    department: 'General Practice',
    diagnosis: 'Abdominal Pain, Query Appendicitis',
    icdCode: 'R10.0',
    episodeType: 'acute',
    medications: [
      { name: 'Inj. Ranitidine 50mg', dose: '50mg', frequency: 'Stat IV', duration: 'Given at clinic' },
      { name: 'Tab. Hyoscine Butylbromide 10mg', dose: '10mg', frequency: 'Stat + SOS', duration: 'As needed' },
    ],
    labs: [
      { name: 'USG Abdomen', priority: 'Urgent' },
    ],
    instructions: [
      'Watch out for vomiting or severe stomach pain — especially right lower abdomen',
      'Take bland diet only. Avoid spicy food.',
    ],
    followUp: { date: 'tomorrow', location: 'American Hospital Dubai', notes: 'Bring USG report' },
  });

  const zayn = createEpisode({
    patientName: 'Zayn Al-Rashidi',
    mrn: '40821',
    age: 34,
    mobile: '+971501234567',
    language: 'Arabic',
    insurance: 'Daman Enhanced',
    doctorName: 'Dr. Sam Willis',
    department: 'General Practice',
    diagnosis: 'Fever Investigation, Query URTI',
    icdCode: 'J06.9',
    episodeType: 'acute',
    medications: [
      { name: 'Tab. Paracetamol 500mg', dose: '500mg', frequency: 'Stat + SOS', duration: 'As needed' },
      { name: 'Tab. Azithromycin 500mg', dose: '500mg', frequency: 'Once daily', duration: '5 days' },
    ],
    labs: [
      { name: 'CBC with Differential Count', priority: 'Routine' },
      { name: 'Malarial Parasite MP Smear', priority: 'Routine' },
    ],
    instructions: [
      'Monitor temperature every 4-6 hours',
      'Watch out for shivering or recurring high fever — especially if it repeats every 6-7 hours',
    ],
    followUp: { date: 'Mar 14', location: 'American Hospital Dubai', notes: 'Bring CBC and malarial parasite reports' },
  });

  for (let i = 1; i <= 4; i++) {
    appendEvent(zayn.id, 'LAB_OVERDUE', { day: i });
  }
  for (let i = 1; i <= 5; i++) {
    appendEvent(zayn.id, 'CONTACT_ATTEMPT_FAILED', { attempt: i });
  }
  appendEvent(zayn.id, 'TIER1_SIGNAL', { symptom: 'shivering reported' });
  appendEvent(zayn.id, 'COST_BARRIER_FLAGGED', { source: 'inferred from message' });
  appendEvent(zayn.id, 'ANXIETY_FLAG', { source: 'inferred from message: nervous about results' });
  appendEvent(zayn.id, 'MEDICATION_MISSED', { dose: 1 });
  appendEvent(zayn.id, 'MEDICATION_MISSED', { dose: 2 });

  console.log(`  Seeded 2 episodes: Nora (id=${nora.id}) and Zayn (id=${zayn.id})`);
}

seedIfEmpty();

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// API ----------------------------------------------------------------------

app.post('/api/episodes', (req, res) => {
  const payload = req.body || {};
  if (!payload.patientName || !payload.mrn) {
    return res.status(400).json({ error: 'patientName and mrn are required' });
  }
  const episode = createEpisode(payload);
  res.status(201).json(episode);
});

app.get('/api/episodes', (req, res) => {
  const rows = db
    .prepare(
      `SELECT id, patientName, mrn, episodeType, diagnosis, icdCode, doctorName,
              riskScore, triageSegment, status, careGap, createdAt
         FROM episodes
        WHERE status = 'active'
        ORDER BY riskScore DESC, id ASC`
    )
    .all();
  res.json(rows);
});

app.get('/api/episodes/:id', (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) return res.status(400).json({ error: 'invalid id' });
  const episode = loadEpisode(id);
  if (!episode) return res.status(404).json({ error: 'episode not found' });
  res.json(episode);
});

app.post('/api/episodes/:id/event', (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) return res.status(400).json({ error: 'invalid id' });
  const { eventType, metadata } = req.body || {};
  if (!eventType) return res.status(400).json({ error: 'eventType is required' });
  const exists = db.prepare('SELECT id FROM episodes WHERE id = ?').get(id);
  if (!exists) return res.status(404).json({ error: 'episode not found' });
  const episode = appendEvent(id, eventType, metadata || {});
  res.json(episode);
});

// Risk breakdown for one patient — variables, weights, contributions, Z, P, score.
app.get('/api/episodes/:id/risk', (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) return res.status(400).json({ error: 'invalid id' });
  const exists = db.prepare('SELECT id FROM episodes WHERE id = ?').get(id);
  if (!exists) return res.status(404).json({ error: 'episode not found' });
  const events = getEventsForEpisode(id);
  const breakdown = computeRiskBreakdown(events);
  res.json({ episodeId: id, ...breakdown, triage: triageFromScore(breakdown.score) });
});

// Settings → Risk Weights (configurable per client).
app.get('/api/settings/risk-weights', (req, res) => {
  res.json({ weights: { ...RISK_WEIGHTS }, defaults: { ...DEFAULT_RISK_WEIGHTS } });
});

app.patch('/api/settings/risk-weights', (req, res) => {
  const updates = req.body || {};
  const allowed = Object.keys(DEFAULT_RISK_WEIGHTS);
  for (const key of Object.keys(updates)) {
    if (!allowed.includes(key)) {
      return res.status(400).json({ error: `unknown weight: ${key}` });
    }
    const v = Number(updates[key]);
    if (!Number.isFinite(v)) {
      return res.status(400).json({ error: `weight ${key} must be a number` });
    }
    RISK_WEIGHTS[key] = v;
  }
  // Recompute every episode with new weights.
  const ids = db.prepare('SELECT id FROM episodes').all().map(r => r.id);
  for (const id of ids) recomputeAndPersistRisk(id);
  res.json({ ok: true, weights: { ...RISK_WEIGHTS } });
});

app.post('/api/settings/risk-weights/reset', (req, res) => {
  RISK_WEIGHTS = { ...DEFAULT_RISK_WEIGHTS };
  const ids = db.prepare('SELECT id FROM episodes').all().map(r => r.id);
  for (const id of ids) recomputeAndPersistRisk(id);
  res.json({ ok: true, weights: { ...RISK_WEIGHTS } });
});

// WhatsApp via Twilio.
//   .env keys required:
//     TWILIO_ACCOUNT_SID
//     TWILIO_AUTH_TOKEN
//     TWILIO_WHATSAPP_FROM   e.g.  whatsapp:+14155238886   (sandbox default)
app.post('/api/whatsapp/send', async (req, res) => {
  const { to, message, episodeId } = req.body || {};
  if (!to || !message) {
    return res.status(400).json({ error: 'to and message are required' });
  }

  const sid = process.env.TWILIO_ACCOUNT_SID;
  const token = process.env.TWILIO_AUTH_TOKEN;
  const from = process.env.TWILIO_WHATSAPP_FROM;

  if (!sid || !token || !from) {
    return res.status(503).json({
      error: 'Twilio not configured',
      hint: 'Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM to .env',
    });
  }

  try {
    const twilio = require('twilio')(sid, token);
    const toFormatted = String(to).startsWith('whatsapp:') ? to : `whatsapp:${to}`;
    const result = await twilio.messages.create({
      from,
      to: toFormatted,
      body: message,
    });
    if (episodeId) {
      appendEvent(Number(episodeId), 'WHATSAPP_SENT', {
        to: toFormatted,
        sid: result.sid,
        body: message,
      });
    }
    res.json({ ok: true, sid: result.sid, status: result.status, to: toFormatted });
  } catch (err) {
    res.status(500).json({ error: err.message, code: err.code });
  }
});

app.get('/api/whatsapp/status', (req, res) => {
  const configured = Boolean(
    process.env.TWILIO_ACCOUNT_SID &&
    process.env.TWILIO_AUTH_TOKEN &&
    process.env.TWILIO_WHATSAPP_FROM
  );
  res.json({
    configured,
    from: configured ? process.env.TWILIO_WHATSAPP_FROM : null,
  });
});

app.delete('/api/episodes/reset', (req, res) => {
  const resetTxn = db.transaction(() => {
    db.exec('DELETE FROM events;');
    db.exec('DELETE FROM episodes;');
    db.exec("DELETE FROM sqlite_sequence WHERE name IN ('events','episodes');");
  });
  resetTxn();
  seedIfEmpty();
  const rows = db
    .prepare(
      `SELECT id, patientName, mrn, episodeType, diagnosis, icdCode, doctorName,
              riskScore, triageSegment, status, careGap, createdAt
         FROM episodes
        WHERE status = 'active'
        ORDER BY riskScore DESC, id ASC`
    )
    .all();
  res.json({ ok: true, episodes: rows });
});

// Static --------------------------------------------------------------------

app.use(express.static(PUBLIC_DIR));

// SPA fallback for any unknown non-API path.
app.use((req, res, next) => {
  if (req.path.startsWith('/api/')) return next();
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.listen(PORT, () => {
  console.log('');
  console.log('  ╔══════════════════════════════════════╗');
  console.log('  ║   CareLoop Workbench                 ║');
  console.log('  ║   Al-Hilal Hospital · OPD Care       ║');
  console.log('  ╠══════════════════════════════════════╣');
  console.log(`  ║   Running at: http://localhost:${PORT}   ║`);
  console.log('  ╚══════════════════════════════════════╝');
  console.log('');
});
