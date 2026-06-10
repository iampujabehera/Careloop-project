// CareLoop Risk Engine — JavaScript port of the Python POC.
//
// LLM extracts behavioural flags from WhatsApp conversation context.
// Logistic regression deterministically computes the final risk score.
//
// The Python POC at port 8088 is replaced by /api/risk/score in server.js so
// the project stays Node-only (Python is not installed on this workstation).

// Model coefficients from risk.js (calibrated set). LLM flags arrive as 0/1/3
// per spec; we scale them by /3 before feeding the regression, matching risk.js.
const DEFAULT_INTERCEPT = -1.25;

const DEFAULT_WEIGHTS = {
  past_non_availability_rate: 2.1,
  chronic_disease_flag:       0.7,
  cost_sensitivity_flag:      0.6,
  age_flag:                   0.5,
  medication_adherence_risk:  1.1,
  procedure_flag:             1.3,
  mental_stage_flag:          1.0,
};

const SEGMENT_THRESHOLDS = { medium: 0.35, high: 0.65 };

const CHRONIC_KEYWORDS = [
  'diabetes', 'hypertension', 'asthma', 'copd', 'epilepsy', 'arthritis',
  'depression', 'anxiety', 'cancer', 'heart', 'renal', 'kidney', 'chronic',
  'thyroid', 'cholesterol', 'hair loss', 'alopecia',
];

const COST_SENSITIVE_INSURANCE_HINTS = [
  'self pay', 'self-pay', 'cash', 'uninsured', 'out of pocket', 'no insurance',
];

function inferChronicFlag(patient) {
  const text = (
    String(patient.chronic_diseases || '') + ' ' +
    String(patient.diagnosis || '')
  ).toLowerCase();
  return CHRONIC_KEYWORDS.some(k => text.includes(k)) ? 1 : 0;
}

function inferCostSensitivityFlag(patient) {
  const ins = String(patient.insurance_type || patient.insurance || '').toLowerCase().trim();
  if (!ins) return 1; // missing insurance = treat as cost-sensitive
  return COST_SENSITIVE_INSURANCE_HINTS.some(k => ins.includes(k)) ? 1 : 0;
}

function inferAgeFlag(patient) {
  // Scaled per risk.js: 65+ = 1, 50-64 = 1/3, else 0
  const digits = String(patient.age || '').match(/\d+/);
  const ageNum = digits ? Number(digits[0]) : NaN;
  if (isNaN(ageNum)) return 1 / 3;
  if (ageNum >= 65) return 1;
  if (ageNum >= 50) return 1 / 3;
  return 0;
}

function buildFeatures(patient, adherence, flags) {
  const rawMed   = flags && flags.medication_adherence_risk != null ? flags.medication_adherence_risk : 1;
  const rawProc  = flags && flags.procedure_flag != null            ? flags.procedure_flag            : 1;
  const rawMent  = flags && flags.mental_stage_flag != null         ? flags.mental_stage_flag         : 1;

  // Hard adherence data overrides LLM-inferred adherence flag
  let medAdherenceRaw = rawMed;
  if (adherence && adherence.medicine_taken_ratio != null) {
    const r = Number(adherence.medicine_taken_ratio);
    if (!isNaN(r)) medAdherenceRaw = r >= 0.85 ? 0 : (r >= 0.5 ? 1 : 3);
  }

  // risk.js scales flag values by /3 to put them on a 0-1 scale before applying the coefficient
  return {
    past_non_availability_rate: Number(patient.past_non_availability_rate || 0),
    chronic_disease_flag:       inferChronicFlag(patient),
    cost_sensitivity_flag:      inferCostSensitivityFlag(patient),
    age_flag:                   inferAgeFlag(patient),
    medication_adherence_risk:  medAdherenceRaw / 3,
    procedure_flag:             rawProc / 3,
    mental_stage_flag:          rawMent / 3,
  };
}

function computeRiskScore(features) {
  let z = DEFAULT_INTERCEPT;
  const contributions = {};
  for (const [key, weight] of Object.entries(DEFAULT_WEIGHTS)) {
    const x = features[key] == null ? 0 : features[key];
    const contribution = weight * x;
    z += contribution;
    contributions[key] = {
      weight: Number(weight.toFixed(3)),
      value: x,
      contribution: Number(contribution.toFixed(3)),
    };
  }
  const probability = 1 / (1 + Math.exp(-z));
  const score = Math.round(probability * 100);
  return { z: Number(z.toFixed(3)), probability: Number(probability.toFixed(3)), score, contributions };
}

function getRiskSegment(score) {
  // Score is 0–100. Convert to 0–1 to compare against risk.js probability thresholds.
  const p = score / 100;
  if (p >= SEGMENT_THRESHOLDS.high) return 'High';
  if (p >= SEGMENT_THRESHOLDS.medium) return 'Medium';
  return 'Low';
}

function getNextAction(segment, flags) {
  if (segment === 'High') return 'Escalate to consultant within 24 hours · orchestrator call same day';
  if (segment === 'Medium') {
    const barrier = flags && flags.barrier_type;
    if (barrier === 'Cost')      return 'Schedule cost-barrier conversation';
    if (barrier === 'Fear')      return 'Schedule reassurance check-in';
    if (barrier === 'Logistics') return 'Help patient with logistics';
    return 'Schedule wellbeing check-in within 48 hours';
  }
  return 'Continue automated monitoring';
}

const FLAG_EXTRACTION_PROMPT = `You are a care-plan adherence-risk analyser. Given WhatsApp messages between a care orchestrator and a patient, extract the following SIGNAL FLAGS and return STRICT JSON only:

{
  "mental_stage_flag": 0,
  "medication_adherence_risk": 0,
  "procedure_flag": 0,
  "barrier_type": "None",
  "reasoning": "one short sentence"
}

VALUES:
- mental_stage_flag:        0 = positive (engaged, optimistic), 1 = neutral / unclear, 3 = negative (anxious, depressed, hopeless, distressed)
- medication_adherence_risk: 0 = adherent (confirmed taking), 1 = partial / unclear, 3 = non-adherent (skipped, refused, silent on meds)
- procedure_flag:           0 = labs/procedures completed, 1 = not discussed yet, 3 = explicitly missed or refused
- barrier_type:             one of "Cost", "Fear", "Logistics", "None", "Other"

If the conversation is empty or too short to judge, return all 1s and barrier_type "None".

Return ONLY the JSON object, no prose, no markdown.`;

async function extractFlagsViaLLM(openai, messages) {
  if (!openai || !Array.isArray(messages) || messages.length === 0) {
    return {
      mental_stage_flag: 1,
      medication_adherence_risk: 1,
      procedure_flag: 1,
      barrier_type: 'None',
      reasoning: 'No conversation yet — neutral baseline.',
    };
  }

  const transcript = messages
    .filter(m => m && m.text)
    .map(m => `${m.sender || 'unknown'}: ${m.text}`)
    .join('\n')
    .slice(0, 4000); // hard cap so we don't blow the context window

  try {
    const resp = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: FLAG_EXTRACTION_PROMPT },
        { role: 'user', content: 'Conversation transcript:\n\n' + transcript },
      ],
      temperature: 0,
      max_tokens: 200,
    });
    return JSON.parse(resp.choices[0].message.content);
  } catch (err) {
    console.warn('[risk] LLM flag extraction failed:', err.message);
    return {
      mental_stage_flag: 1,
      medication_adherence_risk: 1,
      procedure_flag: 1,
      barrier_type: 'None',
      reasoning: 'LLM extraction failed — neutral baseline.',
    };
  }
}

async function scorePatient({ patient = {}, adherence = null, whatsappMessages = [], openai = null }) {
  const flags = await extractFlagsViaLLM(openai, whatsappMessages);
  const features = buildFeatures(patient, adherence, flags);
  const { z, probability, score, contributions } = computeRiskScore(features);
  const segment = getRiskSegment(score);
  const nextAction = getNextAction(segment, flags);

  return {
    risk_score: score,
    risk_segment: segment,
    next_action: nextAction,
    flags,
    features,
    model: {
      intercept: DEFAULT_INTERCEPT,
      z,
      probability,
      contributions,
    },
    scored_at_utc: new Date().toISOString(),
  };
}

module.exports = {
  scorePatient,
  extractFlagsViaLLM,
  computeRiskScore,
  buildFeatures,
  getRiskSegment,
  getNextAction,
  DEFAULT_WEIGHTS,
  DEFAULT_INTERCEPT,
};
