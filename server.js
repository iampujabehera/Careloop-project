require('dotenv').config();

const path = require('path');
const express = require('express');
const cors = require('cors');
const Database = require('better-sqlite3');
const multer = require('multer');
const OpenAI = require('openai');
const pdfParse = require('pdf-parse');
const careloopRiskEngine = require('./careloopRiskEngine');
const careloopWeightedEngine = require('./careloopWeightedEngine');
const { buildSystemPrompt } = require('./careloopMessagingPrompt');
const reminders = require('./careloopReminders');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const DB_PATH = path.join(__dirname, 'careloop.db');

console.log('OpenAI key loaded:', process.env.OPENAI_API_KEY ? '✓ yes' : '✗ no — set OPENAI_API_KEY in .env');
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

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

  CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    direction    TEXT    NOT NULL,
    fromNumber   TEXT,
    toNumber     TEXT,
    body         TEXT,
    twilioSid    TEXT,
    patientMrn   TEXT,
    patientName  TEXT,
    episodeId    INTEGER,
    createdAt    TEXT    NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_messages_phone ON messages(fromNumber, toNumber, createdAt);
  CREATE INDEX IF NOT EXISTS idx_messages_mrn ON messages(patientMrn, createdAt);

  CREATE TABLE IF NOT EXISTS patient_state (
    mrn                       TEXT    PRIMARY KEY,
    patientName               TEXT,
    phone                     TEXT,
    careActivated             INTEGER NOT NULL DEFAULT 0,
    careStartedAt             TEXT,
    medicationConfirmedToday  INTEGER NOT NULL DEFAULT 0,
    lastMedicationConfirmedAt TEXT,
    treatmentAdvice           TEXT,
    condition                 TEXT,
    doctor                    TEXT,
    hospital                  TEXT,
    medicationsJson           TEXT NOT NULL DEFAULT '[]',
    labsJson                  TEXT NOT NULL DEFAULT '[]',
    followUp                  TEXT,
    riskSegment               TEXT NOT NULL DEFAULT 'Low',
    updatedAt                 TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS adherence_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    mrn           TEXT    NOT NULL,
    label         TEXT    NOT NULL,
    scheduledAt   TEXT    NOT NULL,
    respondedAt   TEXT,
    taken         INTEGER,
    barrier       TEXT,
    createdAt     TEXT    NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_adherence_mrn ON adherence_log(mrn, createdAt);
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

// Weights, scoring math and triage live in the shared module so the local
// server and the Vercel serverless functions score identically.
const {
  DEFAULT_RISK_WEIGHTS,
  normalizeSignals,
  computeRiskBreakdownFromFeatures,
  triageFromScore,
} = careloopWeightedEngine;

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
  // Use the live (slider-tuned) RISK_WEIGHTS for persisted episodes.
  return computeRiskBreakdownFromFeatures(extractFeatures(events), RISK_WEIGHTS);
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

// Idempotent: ensure the Rehan autism chronic-care demo patient exists,
// even on DBs that were seeded before he was added. Runs every boot but only
// inserts if MRN 50231 is missing, so it never duplicates.
function ensureRehan() {
  const existing = db.prepare('SELECT id FROM episodes WHERE mrn = ?').get('50231');
  if (existing) return;

  const ashok = createEpisode({
    patientName: 'Rehan',
    mrn: '50231',
    age: 4,
    mobile: '+971509876543',
    language: 'English',
    insurance: 'Daman Enhanced',
    doctorName: 'Dr. Layla Haddad',
    department: 'Developmental Pediatrics',
    diagnosis: 'Autism Spectrum Disorder',
    icdCode: 'F84.0',
    episodeType: 'chronic',
    careGap: '3 caregiver forms pending · caregiver silent',
    medications: [],
    labs: [
      { name: 'Developmental pediatrician review', priority: 'Due' },
      { name: 'Hearing & Vision Evaluation', priority: 'Annual' },
    ],
    instructions: [
      'Keep caregiver engaged through automated reminders and forms',
      'Monitor for sleep disturbance, GI/feeding concerns, and seizure-like episodes',
    ],
    followUp: { date: 'Developmental review due', location: 'Good Health Hospital OPD', notes: 'Caregiver re-engagement required' },
  });

  // Drive the engine to a HIGH-risk score: chronic flag, missed appointments,
  // caregiver silence, anxiety signal and a yellow-flag clinical signal.
  appendEvent(ashok.id, 'CHRONIC_FLAG', { condition: 'Autism Spectrum Disorder' });
  appendEvent(ashok.id, 'MISSED_APPOINTMENT', { what: 'speech therapy follow-up' });
  appendEvent(ashok.id, 'MISSED_APPOINTMENT', { what: 'developmental review' });
  for (let i = 1; i <= 6; i++) {
    appendEvent(ashok.id, 'CONTACT_ATTEMPT_FAILED', { attempt: i });
  }
  appendEvent(ashok.id, 'ANXIETY_FLAG', { source: 'caregiver reported sleep disturbance and restlessness' });
  appendEvent(ashok.id, 'TIER1_SIGNAL', { symptom: 'sleep disturbance reported by caregiver' });

  // Pin the displayed risk score to 88 (matches the detail screen) so the
  // queue and detail header stay in sync regardless of weight tuning.
  db.prepare('UPDATE episodes SET riskScore = ?, triageSegment = ? WHERE id = ?')
    .run(88, 'emergent', ashok.id);

  console.log(`  Ensured Rehan autism chronic-care patient (id=${ashok.id})`);
}

ensureRehan();

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

// Score an ad-hoc signals object with the CURRENT configurable weights.
// Client-side patients (uploaded via prescription, demo patients) aren't
// persisted as episodes, so they post their derived signals here and get
// scored by the same slider-controlled engine as seeded episodes. Adjusting
// any weight in Risk Detection Settings changes the result of this endpoint.
app.post('/api/risk/score-signals', (req, res) => {
  const body = req.body || {};
  const features = normalizeSignals(body.signals);
  // Honour an explicit weights override (e.g. unsaved slider values); fall back
  // to the server's live RISK_WEIGHTS.
  const breakdown = computeRiskBreakdownFromFeatures(features, body.weights || RISK_WEIGHTS);
  res.json({ ok: true, ...breakdown, triage: triageFromScore(breakdown.score) });
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

// Demo defaults — the review screen auto-fills the phone field with this so
// the upload-to-WhatsApp demo flow doesn't require typing on every test.
app.get('/api/config/demo-defaults', (req, res) => {
  let demoPhone = process.env.PATIENT_PHONE_NUMBER || '';
  // Strip whatsapp: prefix if user added it in .env — the UI field expects bare phone
  if (demoPhone.startsWith('whatsapp:')) demoPhone = demoPhone.slice('whatsapp:'.length);
  res.json({ demoPhone });
});

// ---------------------------------------------------------------------------
// WhatsApp — LLM-driven messaging + state tracking + inbound auto-reply
// ---------------------------------------------------------------------------

// Static fallback when OpenAI is unavailable. Same shape as before so we
// degrade gracefully instead of failing the demo.
function buildStaticCareplanMessage(rx) {
  const firstName = String(rx.name || 'there').trim().split(/\s+/)[0];
  const doctor = rx.consultant || 'Your doctor';
  let msg = `Hi ${firstName} 👋 ${doctor} has shared your care plan via CareLoop.\n\n`;
  if (Array.isArray(rx.medications) && rx.medications.length) {
    msg += '💊 *Medications*\n';
    for (const m of rx.medications) {
      if (!m || !m.name) continue;
      let line = '• ' + m.name;
      if (m.dose) line += ' ' + m.dose;
      if (m.frequency) line += ' · ' + m.frequency;
      if (m.duration) line += ' · ' + m.duration;
      msg += line + '\n';
    }
    msg += '\n';
  }
  if (rx.followupDate) msg += '📅 Follow-up: ' + rx.followupDate + '\n\n';
  msg += '— CareLoop · Good Health Hospital';
  return msg;
}

function formatMedicationsForPrompt(rx) {
  return (rx.medications || [])
    .filter(m => m && m.name)
    .map(m => {
      const parts = [m.name];
      if (m.dose) parts.push(m.dose);
      if (m.frequency) parts.push(m.frequency);
      if (m.duration) parts.push(m.duration);
      return parts.join(' — ');
    });
}

function formatLabsForPrompt(rx) {
  return (rx.labs || [])
    .filter(l => l && l.test)
    .map(l => {
      const parts = [l.test];
      if (l.lab) parts.push('at ' + l.lab);
      if (l.dueWithin) parts.push('due within ' + l.dueWithin);
      return parts.join(' · ');
    });
}

function getPatientState(mrn) {
  if (!mrn) return null;
  const row = db.prepare('SELECT * FROM patient_state WHERE mrn = ?').get(String(mrn));
  if (!row) return null;
  return {
    mrn: row.mrn,
    patientName: row.patientName,
    name: (row.patientName || '').split(/\s+/)[0],
    phone: row.phone,
    careActivated: !!row.careActivated,
    careStartedAt: row.careStartedAt,
    medicationConfirmedToday: !!row.medicationConfirmedToday,
    lastMedicationConfirmedAt: row.lastMedicationConfirmedAt,
    treatmentAdvice: row.treatmentAdvice || '',
    condition: row.condition || '',
    doctor: row.doctor || '',
    hospital: row.hospital || 'Good Health Hospital',
    medications: JSON.parse(row.medicationsJson || '[]'),
    labs: JSON.parse(row.labsJson || '[]'),
    followUp: row.followUp || '',
    riskSegment: row.riskSegment || 'Low',
    dayInJourney: computeDayInJourney(row.careStartedAt),
  };
}

function computeDayInJourney(careStartedAt) {
  if (!careStartedAt) return 1;
  const start = new Date(careStartedAt + (careStartedAt.includes('T') ? '' : 'Z'));
  if (isNaN(start.getTime())) return 1;
  const days = Math.floor((Date.now() - start.getTime()) / 86400000) + 1;
  return Math.max(1, days);
}

function upsertPatientState(rx, riskSegment) {
  const meds = formatMedicationsForPrompt(rx);
  const labs = formatLabsForPrompt(rx);
  const followUp = rx.followupDate
    ? (rx.followupTime ? `${rx.followupDate} at ${rx.followupTime}` : rx.followupDate)
    : '';
  db.prepare(
    `INSERT INTO patient_state (mrn, patientName, phone, careActivated, careStartedAt, treatmentAdvice, condition, doctor, hospital, medicationsJson, labsJson, followUp, riskSegment, updatedAt)
     VALUES (?, ?, ?, 1, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
     ON CONFLICT(mrn) DO UPDATE SET
       patientName=excluded.patientName,
       phone=excluded.phone,
       careActivated=1,
       careStartedAt=COALESCE(patient_state.careStartedAt, excluded.careStartedAt),
       treatmentAdvice=excluded.treatmentAdvice,
       condition=excluded.condition,
       doctor=excluded.doctor,
       hospital=excluded.hospital,
       medicationsJson=excluded.medicationsJson,
       labsJson=excluded.labsJson,
       followUp=excluded.followUp,
       riskSegment=excluded.riskSegment,
       updatedAt=datetime('now')`
  ).run(
    String(rx.mrn || ''),
    rx.name || '',
    rx.phone || '',
    rx.instructionsEn || '',
    rx.diagnosis || '',
    rx.consultant || '',
    'Good Health Hospital',
    JSON.stringify(meds),
    JSON.stringify(labs),
    followUp,
    riskSegment || 'Low'
  );
}

function markMedicationConfirmed(mrn) {
  db.prepare(
    "UPDATE patient_state SET medicationConfirmedToday=1, lastMedicationConfirmedAt=datetime('now'), updatedAt=datetime('now') WHERE mrn=?"
  ).run(String(mrn));
}

function setRiskSegment(mrn, segment) {
  db.prepare(
    "UPDATE patient_state SET riskSegment=?, updatedAt=datetime('now') WHERE mrn=?"
  ).run(segment, String(mrn));
}

async function generateLLMMessage(state, incomingPatientText) {
  if (!openai) {
    return null; // fall back to static template
  }
  const systemPrompt = buildSystemPrompt(state);
  const userContent = incomingPatientText
    ? incomingPatientText
    : 'Begin the first contact message now. Deliver the care plan as instructed.';
  const resp = await openai.chat.completions.create({
    model: 'gpt-4o',
    temperature: 0.7,
    max_tokens: 400,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userContent },
    ],
  });
  return resp.choices[0].message.content.trim();
}

async function buildCareplanMessage(rx, riskSegment = 'Low') {
  try {
    const meds = formatMedicationsForPrompt(rx);
    const labs = formatLabsForPrompt(rx);
    const state = {
      name: (rx.name || '').split(/\s+/)[0] || 'there',
      condition: rx.diagnosis || '',
      doctor: rx.consultant || '',
      hospital: 'Good Health Hospital',
      medications: meds,
      labs,
      followUp: rx.followupDate
        ? (rx.followupTime ? `${rx.followupDate} at ${rx.followupTime}` : rx.followupDate)
        : '',
      treatmentAdvice: rx.instructionsEn || '',
      dayInJourney: 1,
      careActivated: false,
      medicationConfirmedToday: false,
      riskSegment: riskSegment || 'Low',
    };
    const llmMsg = await generateLLMMessage(state, null);
    if (llmMsg && llmMsg.length > 20) return llmMsg;
  } catch (err) {
    console.warn('[careplan] LLM generation failed, using static template:', err.message);
  }
  return buildStaticCareplanMessage(rx);
}

async function sendWhatsAppRaw(toPhone, body, mrn, patientName) {
  const sid = process.env.TWILIO_ACCOUNT_SID;
  const token = process.env.TWILIO_AUTH_TOKEN;
  const from = process.env.TWILIO_WHATSAPP_FROM;
  if (!sid || !token || !from) throw new Error('Twilio not configured');

  const cleanPhone = String(toPhone).replace(/\s+/g, '');
  const toFormatted = cleanPhone.startsWith('whatsapp:') ? cleanPhone : `whatsapp:${cleanPhone}`;
  const twilioClient = require('twilio')(sid, token);
  const result = await twilioClient.messages.create({ from, to: toFormatted, body });

  db.prepare(
    'INSERT INTO messages (direction, fromNumber, toNumber, body, twilioSid, patientMrn, patientName) VALUES (?, ?, ?, ?, ?, ?, ?)'
  ).run('outbound', from, toFormatted, body, result.sid || '', String(mrn || ''), patientName || '');

  return result;
}

app.post('/api/whatsapp/send-careplan', async (req, res) => {
  const rx = req.body || {};
  if (!rx.phone) {
    return res.status(400).json({ error: 'phone is required' });
  }
  if (!process.env.TWILIO_ACCOUNT_SID || !process.env.TWILIO_AUTH_TOKEN || !process.env.TWILIO_WHATSAPP_FROM) {
    return res.status(503).json({ error: 'Twilio not configured' });
  }

  try {
    // Score the patient first so the message tone matches the risk segment
    let riskSegment = 'Low';
    let riskScore = null;
    try {
      const scored = await careloopRiskEngine.scorePatient({
        patient: {
          name: rx.name, mrn: rx.mrn, age: rx.age,
          insurance_type: rx.insurance, chronic_diseases: rx.diagnosis,
          prescriptions: (rx.medications || []).map(m => m.name),
          past_non_availability_rate: 0,
        },
        whatsappMessages: [],
        openai,
      });
      riskSegment = scored.risk_segment;
      riskScore = scored.risk_score;
    } catch (riskErr) {
      console.warn('[careplan] Risk scoring failed, defaulting to Low:', riskErr.message);
    }

    // Persist patient state so the inbound webhook can use it for replies
    try { upsertPatientState(rx, riskSegment); } catch (e) { console.warn('[careplan] state persist failed:', e.message); }

    // Generate the care-plan message via LLM (with static fallback inside)
    const message = await buildCareplanMessage(rx, riskSegment);

    const result = await sendWhatsAppRaw(rx.phone, message, rx.mrn, rx.name);

    // Schedule the first medication reminder. For demo visibility we default to
    // 2 minutes after onboarding (configurable via REMINDER_FIRST_DELAY_MS in .env).
    let reminderLogId = null;
    try {
      const firstMed = (rx.medications || [])[0];
      const medName = firstMed ? firstMed.name : 'your medication';
      const delayMs = Number(process.env.REMINDER_FIRST_DELAY_MS) || reminders.DEFAULT_FIRST_REMINDER_MS;
      reminderLogId = reminders.scheduleMedicationReminder(
        { db, sendFn: sendWhatsAppRaw, patient: { mrn: rx.mrn, patientName: rx.name, phone: rx.phone, medicationName: medName } },
        { delayMs, label: medName + ' — first dose' }
      );
    } catch (remErr) {
      console.warn('[careplan] reminder scheduling failed:', remErr.message);
    }

    res.json({
      ok: true,
      sid: result.sid,
      status: result.status,
      to: result.to,
      bodyPreview: message.slice(0, 140),
      bodyLength: message.length,
      risk: { score: riskScore, segment: riskSegment },
      reminderLogId,
    });
  } catch (err) {
    console.error('send-careplan failed:', err.message, 'code:', err.code);
    res.status(500).json({ error: err.message, code: err.code });
  }
});

// Twilio sends form-encoded payloads, NOT JSON — use a specific middleware for this route.
const twilioFormParser = express.urlencoded({ extended: false });

app.post('/api/whatsapp/incoming', twilioFormParser, (req, res) => {
  const { From, To, Body, MessageSid, NumMedia, ProfileName } = req.body || {};
  console.log(`📩 Inbound WhatsApp from ${From} (${ProfileName || 'unknown'}): "${Body}"`);

  // Look up the patient by phone (strip whatsapp: prefix)
  const cleanFrom = String(From || '').replace(/^whatsapp:/, '').replace(/\s+/g, '');
  let patientName = '';
  let patientMrn = '';

  try {
    const recentOutbound = db.prepare(
      "SELECT patientName, patientMrn FROM messages WHERE direction='outbound' AND toNumber LIKE ? ORDER BY id DESC LIMIT 1"
    ).get('%' + cleanFrom + '%');
    if (recentOutbound) {
      patientName = recentOutbound.patientName || '';
      patientMrn = recentOutbound.patientMrn || '';
    }
  } catch (lookupErr) {
    console.warn('Patient lookup failed:', lookupErr.message);
  }

  try {
    db.prepare(
      'INSERT INTO messages (direction, fromNumber, toNumber, body, twilioSid, patientMrn, patientName) VALUES (?, ?, ?, ?, ?, ?, ?)'
    ).run('inbound', From || '', To || '', Body || '', MessageSid || '', patientMrn, patientName);
  } catch (dbErr) {
    console.error('Failed to store inbound message:', dbErr.message);
  }

  // Empty TwiML response immediately — we'll send the LLM reply via the Messages API instead,
  // so Twilio doesn't time out and the orchestrator's terminal sees the full reasoning.
  res.set('Content-Type', 'text/xml');
  res.send('<?xml version="1.0" encoding="UTF-8"?><Response></Response>');

  // Background pipeline: rescore risk, then generate + send the LLM reply.
  if (patientMrn) {
    (async () => {
      try {
        const conversation = buildMessagesForMrn(patientMrn);

        // 1) Re-score risk with the new message included
        const scored = await careloopRiskEngine.scorePatient({
          patient: { mrn: patientMrn, name: patientName },
          whatsappMessages: conversation,
          openai,
        });
        setRiskSegment(patientMrn, scored.risk_segment);
        db.prepare(
          'INSERT INTO events (episodeId, eventType, metadata) VALUES (?, ?, ?)'
        ).run(0, 'risk_rescored_after_inbound', JSON.stringify({
          mrn: patientMrn,
          score: scored.risk_score,
          segment: scored.risk_segment,
          flags: scored.flags,
          next_action: scored.next_action,
        }));
        console.log(`📊 Re-scored ${patientName || patientMrn}: ${scored.risk_score} (${scored.risk_segment}) — ${scored.next_action}`);

        // 2) Generate the reply. Three fast-path commands handled before the LLM:
        //    a) Pending check-in yes/no → mark adherence + brief confirmation
        //    b) "summary" / "status" → adherence summary
        //    c) "remind me in 10m" → parse delay + schedule reminder
        //    Otherwise → LLM reply with full prompt context
        const state = getPatientState(patientMrn);
        if (!state) {
          console.warn('No patient_state for MRN', patientMrn, '— skipping auto-reply.');
          return;
        }
        state.riskSegment = scored.risk_segment;
        state.careActivated = true;

        const phone = state.phone || cleanFrom;
        let replyBody = null;
        let fastPath = null;

        // (a) Adherence check-in response
        const adherenceResult = reminders.handleAdherenceResponse(db, patientMrn, Body || '');
        if (adherenceResult) {
          fastPath = adherenceResult.taken === 1 ? 'adherence_yes' : 'adherence_no';
          const firstName = state.name;
          replyBody = adherenceResult.taken === 1
            ? `Great job ${firstName}! ✅ Marked your ${adherenceResult.label} as taken. Keep it up 💚`
            : `Thanks for letting me know ${firstName}. ❌ Logged as missed. Your care team will check in shortly 💚`;
          if (adherenceResult.taken === 1) markMedicationConfirmed(patientMrn);
        }

        // (b) Summary request
        if (!replyBody && reminders.isSummaryRequest(Body || '')) {
          fastPath = 'summary';
          replyBody = reminders.buildAdherenceSummary(db, patientMrn);
        }

        // (c) Schedule reminder request — "remind me in 10 minutes"
        if (!replyBody && reminders.isScheduleRequest(Body || '')) {
          const delayMs = reminders.parseReminderDelay(Body || '');
          if (delayMs) {
            const firstMed = (state.medications || [])[0];
            const medName = firstMed ? String(firstMed).split('—')[0].trim() : 'your medication';
            reminders.scheduleMedicationReminder(
              { db, sendFn: sendWhatsAppRaw, patient: { mrn: patientMrn, patientName: state.patientName, phone, medicationName: medName } },
              { delayMs, label: medName + ' — patient-requested' }
            );
            const minutes = Math.round(delayMs / 60000);
            fastPath = 'reminder_scheduled';
            replyBody = `✅ Got it — I'll remind you in ${minutes >= 1 ? minutes + ' minute' + (minutes !== 1 ? 's' : '') : Math.round(delayMs/1000) + ' seconds'}. 💚`;
          } else {
            fastPath = 'reminder_invalid';
            replyBody = `I can set a reminder up to 24 hours away. Try "remind me in 30 minutes" or "remind me in 2 hours" 💚`;
          }
        }

        // (d) LLM fallback — full prompt context
        if (!replyBody) {
          replyBody = await generateLLMMessage(state, Body || '');
        }
        if (!replyBody) {
          console.warn('LLM produced no reply for', patientMrn);
          return;
        }

        // 3) Detect markers BEFORE stripping them (state updates), then strip from outgoing text
        const hasConfirmed = /✅\s*CONFIRMED/.test(replyBody);
        const hasMissed = /❌\s*MISSED/.test(replyBody);
        const hasEscalate = /🚨\s*ESCALATE/.test(replyBody);

        if (hasConfirmed) markMedicationConfirmed(patientMrn);
        if (hasMissed || hasEscalate) {
          db.prepare(
            'INSERT INTO events (episodeId, eventType, metadata) VALUES (?, ?, ?)'
          ).run(0, hasEscalate ? 'patient_escalation' : 'medication_missed', JSON.stringify({
            mrn: patientMrn,
            patientName,
            patientMessage: Body,
            llmReply: replyBody,
            riskSegment: scored.risk_segment,
            timestamp: new Date().toISOString(),
          }));
        }

        // 4) Send the reply back to the patient
        if (phone) {
          await sendWhatsAppRaw(phone, replyBody, patientMrn, patientName);
          const tags = [];
          if (fastPath) tags.push(fastPath);
          if (hasConfirmed) tags.push('✅ CONFIRMED');
          if (hasMissed) tags.push('❌ MISSED');
          if (hasEscalate) tags.push('🚨 ESCALATE');
          console.log(`💬 Reply sent to ${phone}${tags.length ? ' [' + tags.join(', ') + ']' : ''}`);
        }
      } catch (err) {
        console.warn('Auto-reply pipeline failed:', err.message);
      }
    })();
  }
});

app.get('/api/messages', (req, res) => {
  const limit = Math.min(Number(req.query.limit) || 50, 200);
  const sinceId = Number(req.query.sinceId) || 0;
  const rows = db.prepare(
    'SELECT id, direction, fromNumber, toNumber, body, twilioSid, patientMrn, patientName, createdAt FROM messages WHERE id > ? ORDER BY id DESC LIMIT ?'
  ).all(sinceId, limit);
  res.json({ ok: true, messages: rows });
});

// ---------------------------------------------------------------------------
// Risk Scoring — LLM flag extraction + logistic regression
// ---------------------------------------------------------------------------

function buildMessagesForMrn(mrn, limit = 20) {
  if (!mrn) return [];
  const rows = db.prepare(
    "SELECT direction, body, createdAt FROM messages WHERE patientMrn = ? AND body != '' ORDER BY id DESC LIMIT ?"
  ).all(String(mrn), limit);
  return rows
    .reverse() // chronological order, oldest first
    .map(r => ({
      sender: r.direction === 'inbound' ? 'patient' : 'orchestrator',
      text: r.body || '',
    }));
}

app.get('/api/adherence/:mrn', (req, res) => {
  const mrn = String(req.params.mrn || '');
  const rows = db.prepare(
    'SELECT id, label, scheduledAt, respondedAt, taken, barrier, createdAt FROM adherence_log WHERE mrn = ? ORDER BY id DESC LIMIT 100'
  ).all(mrn);
  const summary = reminders.buildAdherenceSummary(db, mrn);
  res.json({ ok: true, mrn, rows, summaryText: summary });
});

app.post('/api/risk/score', async (req, res) => {
  const { patient = {}, adherence = null, whatsapp_messages = null, use_llm = true } = req.body || {};

  try {
    // If caller didn't supply messages, build them from the DB by MRN
    const messages = Array.isArray(whatsapp_messages)
      ? whatsapp_messages
      : buildMessagesForMrn(patient.mrn);

    const result = await careloopRiskEngine.scorePatient({
      patient,
      adherence,
      whatsappMessages: messages,
      openai: use_llm ? openai : null,
    });

    // Audit
    try {
      db.prepare(
        'INSERT INTO events (episodeId, eventType, metadata) VALUES (?, ?, ?)'
      ).run(0, 'risk_scored', JSON.stringify({
        mrn: patient.mrn || '',
        score: result.risk_score,
        segment: result.risk_segment,
        message_count: messages.length,
      }));
    } catch (_) { /* non-fatal */ }

    res.json({ ok: true, ...result });
  } catch (err) {
    console.error('Risk scoring failed:', err);
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Prescription PDF/image parsing via OpenAI GPT-4o
// ---------------------------------------------------------------------------

const RX_EXTRACTION_SCHEMA_PROMPT = `You are a medical prescription parser for CareLoop, a patient-engagement platform at hospitals in the UAE.
Extract structured data from the prescription provided and return STRICT JSON matching this schema:

{
  "name": "Full patient name as printed",
  "mrn": "Medical record number (digits only, no MRN: prefix)",
  "age": "Age as a human-readable string e.g. '26 years' or '54 years'",
  "phone": "Patient's mobile if present, else empty string",
  "lang": "Most likely patient language ('Arabic', 'English', 'Hindi', 'Marathi', etc.). Infer from script used or location.",
  "insurance": "Insurance plan if present, else empty string",
  "consultant": "Prescribing doctor's full name with title (e.g. 'Dr. Nitin G. Barde')",
  "dept": "Department or speciality inferred from doctor's qualification (e.g. 'Dermatology', 'Cardiology', 'Pulmonology')",
  "diagnosis": "Primary diagnosis or reason for visit, in English. If only medications are listed, infer the condition from them.",
  "episodeType": "EXACTLY one of: 'OPD Acute' (single short course <= 30 days), 'OPD Chronic' (long-term/maintenance/lifestyle conditions like hypertension, diabetes, hair-loss maintenance), or 'OPD · Acute + Chronic' (both)",
  "medications": [
    {
      "name": "Brand name as written",
      "dose": "Strength + unit e.g. '500mg', '5%', '1 capsule'",
      "frequency": "Dosing schedule TRANSLATED TO ENGLISH e.g. 'Once daily', 'Twice daily (morning & night)', '3 times per week'",
      "duration": "Length of course e.g. '7 days', '30 days', 'ongoing'"
    }
  ],
  "labs": [
    {
      "test": "Lab test name",
      "lab": "Lab provider if specified, else empty string",
      "dueWithin": "Time window e.g. '5 days', '2 weeks'"
    }
  ],
  "followupDate": "Follow-up date in 'MMM D' format e.g. 'Jul 2', else empty string",
  "followupTime": "Follow-up time e.g. '10:30 AM', else empty string",
  "instructionsEn": "Any general instructions (diet, lifestyle) TRANSLATED TO ENGLISH, else empty string"
}

RULES:
- Return ONLY the JSON object. No prose, no markdown fences.
- Translate any non-English text (Arabic, Marathi, Hindi) to clear English in 'frequency', 'instructionsEn', 'diagnosis'.
- If a field is genuinely absent, use empty string "" (or empty array for medications/labs). Never invent values.
- Ignore clinic phone numbers, QR codes, addresses, branding — only extract PATIENT and PRESCRIPTION data.
- For Indian prescriptions where 'OD' = once daily, 'BD/BID' = twice daily, 'TDS' = three times daily, 'HS' = at bedtime.
- For Arabic prescriptions: same logic with Arabic dosing conventions.`;

async function callOpenAIVision(imageBuffer, mimeType) {
  const base64 = imageBuffer.toString('base64');
  const dataUrl = `data:${mimeType};base64,${base64}`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: RX_EXTRACTION_SCHEMA_PROMPT },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Parse this prescription. Return ONLY the JSON object specified in the system prompt.' },
          { type: 'image_url', image_url: { url: dataUrl, detail: 'high' } },
        ],
      },
    ],
    max_tokens: 2000,
    temperature: 0,
  });

  return response.choices[0].message.content;
}

async function callOpenAIText(text) {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: RX_EXTRACTION_SCHEMA_PROMPT },
      { role: 'user', content: 'Parse the prescription text below. Return ONLY the JSON object specified in the system prompt.\n\n---\n' + text },
    ],
    max_tokens: 2000,
    temperature: 0,
  });

  return response.choices[0].message.content;
}

// OCR path for scanned/image-only PDFs: upload to OpenAI Files API and ask
// GPT-4o to read the document directly. Handles both text-PDFs and scanned-PDFs.
// File is deleted after extraction to avoid storage cost accumulation.
async function callOpenAIFilePDF(buffer, filename) {
  const file = await openai.files.create({
    file: new File([buffer], filename || 'prescription.pdf', { type: 'application/pdf' }),
    purpose: 'user_data',
  });
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: RX_EXTRACTION_SCHEMA_PROMPT },
        { role: 'user', content: [
          { type: 'text', text: 'Parse the attached prescription PDF. Return ONLY the JSON object specified in the system prompt.' },
          { type: 'file', file: { file_id: file.id } },
        ]},
      ],
      max_tokens: 2000,
      temperature: 0,
    });
    return response.choices[0].message.content;
  } finally {
    openai.files.delete(file.id).catch(() => {});
  }
}

app.post('/api/parse-prescription', upload.single('prescription'), async (req, res) => {
  if (!openai) {
    return res.status(503).json({ error: 'OpenAI not configured. Set OPENAI_API_KEY in .env and restart the server.' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded. Send a multipart form with a "prescription" field.' });
  }

  const { buffer, mimetype, originalname, size } = req.file;
  const t0 = Date.now();

  try {
    let rawJson;
    let parseMethod;

    if (mimetype === 'application/pdf' || originalname.toLowerCase().endsWith('.pdf')) {
      // Cheap path first: EMR-exported PDFs have a real text layer.
      let pdfText = '';
      try {
        const pdfData = await pdfParse(buffer);
        pdfText = (pdfData.text || '').trim();
      } catch (_) { pdfText = ''; }

      if (pdfText && pdfText.length >= 30) {
        rawJson = await callOpenAIText(pdfText);
        parseMethod = 'pdf-text';
      } else {
        // OCR layer: scanned/photo-PDF — upload to OpenAI Files API and let
        // GPT-4o read the document directly (handles both image-only PDFs and
        // mixed PDFs without us having to render anything locally).
        rawJson = await callOpenAIFilePDF(buffer, originalname);
        parseMethod = 'pdf-ocr-via-files-api';
      }
    } else if (mimetype && mimetype.startsWith('image/')) {
      rawJson = await callOpenAIVision(buffer, mimetype);
      parseMethod = 'image-vision';
    } else {
      return res.status(415).json({ error: `Unsupported file type: ${mimetype}. Use PDF, JPG, or PNG.` });
    }

    let extracted;
    try {
      extracted = JSON.parse(rawJson);
    } catch (parseErr) {
      return res.status(502).json({
        error: 'OpenAI returned malformed JSON. Try uploading again.',
        rawResponse: rawJson,
      });
    }

    const elapsedMs = Date.now() - t0;

    // Audit trail: log the parse event (without storing PHI in clear text)
    try {
      db.prepare(
        'INSERT INTO events (episodeId, eventType, metadata) VALUES (?, ?, ?)'
      ).run(0, 'prescription_parsed', JSON.stringify({
        filename: originalname,
        sizeBytes: size,
        method: parseMethod,
        elapsedMs,
        extractedFields: Object.keys(extracted),
      }));
    } catch (auditErr) {
      console.warn('Audit log failed (non-fatal):', auditErr.message);
    }

    res.json({
      ok: true,
      extracted,
      meta: {
        filename: originalname,
        sizeBytes: size,
        method: parseMethod,
        elapsedMs,
        model: 'gpt-4o',
      },
    });
  } catch (err) {
    console.error('Prescription parse failed:', err);
    res.status(500).json({
      error: err.message || 'Parsing failed. Check server logs.',
      code: err.code,
    });
  }
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
