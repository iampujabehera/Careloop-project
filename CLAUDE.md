# CareLoop Workbench — Claude Code Instructions

## What this project is

CareLoop is a patient engagement and care continuity platform for **OPD (outpatient) patients** at Middle East hospitals. This workbench is the **Care Orchestrator's daily tool** — the person (a nurse or coordinator) who monitors patients after they leave an OPD consultation.

The product is scoped to:
- Patients who left an OPD consultation with a prescription
- Active episodes = medication courses, lab orders, follow-up appointments
- Geography = UAE / GCC (WhatsApp primary channel, AED currency, HAAD/DHA compliance)

---

## Project structure

```
careloop-workbench/
├── server.js          ← Node.js server (no dependencies needed)
├── package.json
├── CLAUDE.md          ← You are here
└── public/
    └── index.html     ← The entire workbench (single self-contained file)
```

The entire product lives in `public/index.html`. All CSS, JS, and HTML are in one file. There is no build step, no framework, no npm install needed.

---

## How to run

```bash
node server.js
# Opens at http://localhost:3000
```

---

## Screens in the workbench

| Screen ID | What it is | How to navigate |
|---|---|---|
| `screen-queue` | OPD Priority Queue — main view | Default on load |
| `screen-detail` | Patient detail — Ahmed Al-Rashidi | Click any red/amber row |
| `screen-close-episode` | Episode closure — Sara Al-Zaabi | Click Sara's green row |
| `screen-inbox` | Patient Replies inbox | Click 📩 in sidebar |
| `screen-escalation` | Active escalations list | Click 🔺 in sidebar |
| `screen-escalation-form` | Escalation form to doctor | Click Escalate button in detail |
| `screen-notifications` | WhatsApp/SMS/voice templates | Click 💬 in sidebar |
| `screen-exec` | Revenue dashboard (CEO view) | Click 📊 in sidebar |

---

## Patient data (hardcoded for prototype)

All patient data is hardcoded in `public/index.html`. The five patients are:

| Patient | Status | Episode type | Key scenario |
|---|---|---|---|
| Ahmed Al-Rashidi | 🔴 Critical | OPD Acute + Chronic | CBC lab 5 days overdue, cost barrier suspected |
| Fatima Al-Zahrawi | 🔴 Critical | OPD Acute | X-ray result available, doctor not notified |
| Khalid Al-Mansoori | 🟡 Watch | OPD Acute | Cost barrier, lab due in 2 days |
| Mariam Al-Suwaidi | 🟡 Watch | OPD Chronic | Medication unacknowledged, 4 reminders sent |
| Sara Al-Zaabi | 🟢 Ready to Close | OPD Acute | All care items complete |

---

## Episode end logic

Episodes close in three ways:
1. **Natural Completion** — all care items done, orchestrator confirms
2. **Clinical Discharge** — doctor clears patient mid-episode
3. **Abandoned** — 14 days unreachable, 5 attempts, orchestrator logs reason

Closure always requires human confirmation. CareLoop never auto-closes.

---

## Engagement timeline logic

- **Acute episode** → full day-by-day linear timeline, always expanded
- **Chronic episode** → collapsed by default, shows 6-month compliance grid on expand + last 30 days events
- **Both active** → two collapsible tracks, acute expanded, chronic collapsed

---

## Validation layer (do not remove these)

The following safeguards are intentional and must be preserved in any changes:

1. **Safeguard banner** — top of every screen, HAAD/DHA compliance
2. **AI suggestion strip** — blue box above action buttons, labelled "review before acting"
3. **Risk score disclaimer** — "engagement patterns only, not a clinical assessment"
4. **Escalation review checkbox** — send button disabled until orchestrator checks box
5. **Escalation confirmation modal** — fires before WhatsApp actually sends
6. **Episode close checklist** — orchestrator reviews each care item before confirming
7. **Barrier inference disclaimer** — barriers are unconfirmed until spoken with patient

---

## Key design rules

- **WhatsApp is always the primary channel** (not SMS, not call)
- **AED everywhere** — no dollar signs
- **Arabic patients** — names, insurance (Daman, Thiqa, AXA Gulf, ADNIC, MSH)
- **OPD only** — no inpatient, no post-surgical, no ward language
- **HAAD/DHA** — not HIPAA
- **Care Orchestrator** — not Care Coordinator
- **Consultant** — not Doctor or Physician
- **Next of Kin** — not Emergency Contact

---

## What to build next (suggested)

1. **Episode data model** — replace hardcoded patients with a JSON data layer
2. **Filter logic** — wire up the All / Critical / At Risk / On Track filter buttons
3. **Barrier follow-up routing** — after logging a barrier, trigger the right next action
4. **Inbox unread state** — mark messages as read when patient row is clicked
5. **Closed episodes counter** — update stat card when episode is closed in session
6. **Mariam's detail screen** — Mariam currently has no clickable detail (only Ahmed does)

---

## What NOT to do

- Do not add inpatient/surgical episodes
- Do not replace HAAD with HIPAA
- Do not make AI suggestions sound like clinical decisions
- Do not remove the validation checkpoints or confirmation modals
- Do not auto-close episodes without orchestrator confirmation
- Do not add a dashboard tab for the orchestrator — that is the exec view only
