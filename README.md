# CareLoop Workbench

**OPD Care Orchestrator · Al-Hilal Hospital · Abu Dhabi**

A patient engagement workbench for care coordinators managing outpatient episodes — medications, lab orders, and follow-up appointments — after patients leave an OPD consultation.

---

## Run in 10 seconds

```bash
# No npm install needed — pure Node.js
node server.js
```

Open **http://localhost:3000**

---

## What you'll see

**Priority Queue** — 5 OPD patients ordered by care gap risk. Red patients need action now. Click any row to open the patient detail.

**Patient Detail (Ahmed)** — dual-track engagement timeline. Acute episode fully expanded. Chronic episode collapsed with 6-month compliance grid. AI-suggested next action. 4-button action panel.

**Close Episode (Sara)** — all care items verified complete. 3-step closure with HAAD audit trail.

**Patient Replies Inbox** — 5 unread WhatsApp replies from patients.

**Escalations** — 2 active cases pending doctor response.

**Revenue Dashboard** — CEO view. AED 521K realized MTD.

---

## Stack

- Pure HTML / CSS / JavaScript — no framework
- Node.js server — no dependencies
- Single file at `public/index.html`

---

## For Claude Code

See `CLAUDE.md` for full product context, screen map, design rules, and what to build next.
