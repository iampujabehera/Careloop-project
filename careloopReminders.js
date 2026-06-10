'use strict';

// CareLoop medication reminders + adherence tracking.
//
// Adapted from the original reminders.js POC to:
//   - persist adherence in SQLite (adherence_log table) instead of in-memory state
//   - use server.js's sendWhatsAppRaw() helper instead of a separate twilio.js module
//   - track per-MRN active timers in a Node-process Map (lost on restart — fine for demo)
//   - keep the original yes/no parsing and reminder cadence semantics

const CHECKIN_DELAY_MS = 120_000;        // 2 minutes — gap between reminder and check-in
const MAX_DELAY_MS = 86_400_000;         // 24h cap (Node setTimeout fires immediately above 2^31ms)
const DEFAULT_FIRST_REMINDER_MS = 120_000; // 2 minutes after onboarding for demo visibility

// Per-MRN active timer handles. Keyed by mrn so cleanupTimers() can clear them.
const ACTIVE_TIMERS = new Map();
// MRN → { logEntryId } for the most recent pending check-in question.
const PENDING_CHECKINS = new Map();

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

function parseReminderDelay(text) {
  const t = String(text || '').trim().toLowerCase();
  let ms = null;

  const hourMatch = t.match(/(\d+)\s*(?:hours?|hrs?|h)\b/);
  const minMatch  = t.match(/(\d+)\s*(?:minutes?|mins?|m)\b/);
  const secMatch  = t.match(/(\d+)\s*(?:seconds?|secs?|s)\b/);

  if (hourMatch)      ms = parseInt(hourMatch[1], 10) * 3_600_000;
  else if (minMatch)  ms = parseInt(minMatch[1], 10) * 60_000;
  else if (secMatch)  ms = parseInt(secMatch[1], 10) * 1_000;

  if (ms === null || ms <= 0) return null;
  if (ms > MAX_DELAY_MS) return null;
  return ms;
}

function isYesResponse(text) {
  return /^(yes|yeah|yep|yup|taken|done|already took|just took|took it|i did|✅|👍|1️⃣|1\b)/i.test(String(text || '').trim());
}

function isNoResponse(text) {
  return /^(no|nope|not yet|forgot|didn'?t|missed|haven'?t|not taken|❌|👎|2️⃣|2\b)/i.test(String(text || '').trim());
}

function isSummaryRequest(text) {
  return /^(summary|status|stats|adherence|progress|report)\b/i.test(String(text || '').trim());
}

function isScheduleRequest(text) {
  return /^(remind|reminder|schedule)/i.test(String(text || '').trim());
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

function pushTimer(mrn, handle, label) {
  if (!ACTIVE_TIMERS.has(mrn)) ACTIVE_TIMERS.set(mrn, []);
  ACTIVE_TIMERS.get(mrn).push({ handle, label });
}

function pruneTimer(mrn, handle) {
  const arr = ACTIVE_TIMERS.get(mrn);
  if (!arr) return;
  ACTIVE_TIMERS.set(mrn, arr.filter(t => t.handle !== handle));
}

/**
 * Schedule a medication reminder for a patient.
 *
 * @param {object} ctx
 * @param {object} ctx.db        — better-sqlite3 instance
 * @param {function} ctx.sendFn  — async (phone, body, mrn, patientName) => result
 * @param {object} ctx.patient   — { mrn, patientName, phone, medicationName }
 * @param {object} opts          — { delayMs, label }
 * @returns {number} logEntryId
 */
function scheduleMedicationReminder({ db, sendFn, patient }, opts = {}) {
  const { mrn, patientName, phone, medicationName } = patient;
  const { delayMs = DEFAULT_FIRST_REMINDER_MS, label = 'morning dose' } = opts;
  if (!mrn || !phone) throw new Error('scheduleMedicationReminder requires mrn + phone');

  const scheduledAtIso = new Date(Date.now() + delayMs).toISOString();
  const result = db.prepare(
    'INSERT INTO adherence_log (mrn, label, scheduledAt) VALUES (?, ?, ?)'
  ).run(String(mrn), label, scheduledAtIso);
  const logEntryId = Number(result.lastInsertRowid);

  const firstName = (patientName || '').split(/\s+/)[0] || 'there';
  const medName = medicationName || 'your medication';

  const reminderHandle = setTimeout(async () => {
    pruneTimer(mrn, reminderHandle);
    try {
      const stillActive = db.prepare('SELECT careActivated FROM patient_state WHERE mrn = ?').get(String(mrn));
      if (!stillActive || !stillActive.careActivated) return;

      await sendFn(phone, `⏰ Reminder, ${firstName}: Time to take your ${medName}. Reply *yes* once taken, or *no* if you can't. Take care 💚`, mrn, patientName);
      console.log(`[REMINDER_SENT] mrn=${mrn} label=${label}`);

      // Follow-up check-in 2 minutes later if no response received
      const checkinHandle = setTimeout(async () => {
        pruneTimer(mrn, checkinHandle);
        try {
          const entry = db.prepare('SELECT taken FROM adherence_log WHERE id = ?').get(logEntryId);
          if (entry && entry.taken !== null) return; // patient already responded

          PENDING_CHECKINS.set(String(mrn), { logEntryId, label });
          await sendFn(phone, `Did you take your ${medName}, ${firstName}?\n\n1️⃣ Yes, taken\n2️⃣ No, not yet`, mrn, patientName);
          console.log(`[CHECKIN_SENT] mrn=${mrn} logEntryId=${logEntryId}`);
        } catch (err) {
          console.error(`[REMINDER_FAIL] mrn=${mrn} checkin:`, err.message);
        }
      }, CHECKIN_DELAY_MS);
      pushTimer(mrn, checkinHandle, `checkin-${logEntryId}`);
    } catch (err) {
      console.error(`[REMINDER_FAIL] mrn=${mrn} reminder:`, err.message);
    }
  }, delayMs);
  pushTimer(mrn, reminderHandle, `reminder-${logEntryId}`);

  console.log(`[REMINDER_SCHEDULED] mrn=${mrn} label=${label} delayMs=${delayMs}`);
  return logEntryId;
}

// ---------------------------------------------------------------------------
// Adherence response handler — called from the inbound webhook
// ---------------------------------------------------------------------------

/**
 * If there's a pending check-in for this MRN and the inbound text is yes/no,
 * mark adherence in the DB and return { taken, logEntryId, label }. Otherwise null.
 */
function handleAdherenceResponse(db, mrn, text) {
  if (!mrn) return null;
  const pending = PENDING_CHECKINS.get(String(mrn));
  if (!pending) return null;

  let taken = null;
  if (isYesResponse(text)) taken = 1;
  else if (isNoResponse(text)) taken = 0;
  if (taken === null) return null;

  db.prepare(
    "UPDATE adherence_log SET respondedAt = datetime('now'), taken = ? WHERE id = ?"
  ).run(taken, pending.logEntryId);
  PENDING_CHECKINS.delete(String(mrn));

  return { taken, logEntryId: pending.logEntryId, label: pending.label };
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

function formatTimestamp(iso) {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const day = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  return `${day} at ${time}`;
}

function buildAdherenceSummary(db, mrn) {
  const stateRow = db.prepare('SELECT patientName FROM patient_state WHERE mrn = ?').get(String(mrn));
  const firstName = stateRow ? (stateRow.patientName || 'there').split(/\s+/)[0] : 'there';

  const rows = db.prepare(
    'SELECT label, taken, respondedAt, scheduledAt FROM adherence_log WHERE mrn = ? ORDER BY id DESC'
  ).all(String(mrn));

  if (!rows.length) {
    return `No medication tracking data yet, ${firstName}. Reminders are scheduled — you'll get one shortly! 💚`;
  }

  const groups = {};
  for (const r of rows) {
    if (!groups[r.label]) groups[r.label] = { taken: 0, missed: 0, pending: 0, lastMissed: null };
    if (r.taken === 1) groups[r.label].taken++;
    else if (r.taken === 0) {
      groups[r.label].missed++;
      groups[r.label].lastMissed = r.respondedAt || r.scheduledAt;
    } else {
      groups[r.label].pending++;
    }
  }

  const lines = [`📊 Your medication summary, ${firstName}:\n`];
  for (const [label, stats] of Object.entries(groups)) {
    let line = `💊 ${label}: ✅ ${stats.taken} taken · ❌ ${stats.missed} missed${stats.pending ? ` · ⏳ ${stats.pending} pending` : ''}`;
    if (stats.lastMissed) line += `\n   Last missed: ${formatTimestamp(stats.lastMissed)}`;
    lines.push(line);
  }
  lines.push('\nKeep it up! 💚');
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

function cleanupTimers(mrn) {
  const arr = ACTIVE_TIMERS.get(String(mrn));
  if (!arr) return;
  for (const t of arr) clearTimeout(t.handle);
  ACTIVE_TIMERS.delete(String(mrn));
  PENDING_CHECKINS.delete(String(mrn));
  console.log(`[TIMERS_CLEANED] mrn=${mrn} count=${arr.length}`);
}

module.exports = {
  scheduleMedicationReminder,
  handleAdherenceResponse,
  buildAdherenceSummary,
  cleanupTimers,
  parseReminderDelay,
  isYesResponse,
  isNoResponse,
  isSummaryRequest,
  isScheduleRequest,
  CHECKIN_DELAY_MS,
  DEFAULT_FIRST_REMINDER_MS,
  ACTIVE_TIMERS,
  PENDING_CHECKINS,
};
