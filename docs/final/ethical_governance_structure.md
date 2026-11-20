# Ethical Governance Structure

Who oversees the model, what they may decide, and how often. Terms below are **proposed**; no panel exists for this coursework, and nothing here records an approval.

## 1. The AI Ethics and Data Governance Review Panel

| Seat | Role | Why |
|---|---|---|
| Chair | Head of service quality (product owner) | Owns the business purpose |
| Data protection officer (DPO) | Statutory voice on personal data | Act 843 accountability |
| Compliance officer | Regulatory mapping | |
| Customer representative | Speaks for complainants | Independent of the bank's incentives |
| Independent data scientist | Challenges the statistics | Not on the build team |
| IT operations lead | Deployment and incidents | |
| Secretary (non-voting) | Keeps the log in `ETHICS_CHARTER.md` | |

Quorum: four voting members including the DPO and the independent data scientist. Decisions need a simple majority; the DPO and the customer representative each hold a veto on privacy and fairness grounds respectively, which the chair may only overturn by escalating to executive management in writing. Conflicts of interest are declared at the start of each meeting; the model builder presents but does not vote.

## 2. Decision rights

| Decision | Who decides | Evidence required |
|---|---|---|
| Use a new data source | Panel (Gate 1, data) | Data-protection impact assessment, datasheet, bias-detection report |
| Release a model version | Panel (Gate 2, release) | Model card, fairness report, algorithmic impact assessment, gate report from `retrain.py` |
| Promote a challenger in the registry | Named approver (a panel member) plus ticket | Gate report with every gate passed; the tool refuses otherwise |
| Continue operating | Panel (Gate 3, quarterly) | Monitoring dashboard, incident log, override-rate review |
| Switch the model off | Service quality lead alone, at any time | None; reasons recorded afterwards |
| Change the review capacity or region lines | Panel | Fairness effect shown before and after |
| Retire the model | Panel | `decommissioning_plan.md` |

## 3. Roles (RACI)

R responsible, A accountable, C consulted, I informed.

| Activity | Model owner | Service quality lead | DPO | Panel | IT ops |
|---|---|---|---|---|---|
| Build and test | R | A | C | I | C |
| Monthly monitoring review | R | A | I | I | C |
| Hard-alert response | R | A | C | I | R |
| Fairness breach escalation | R | C | C | A | I |
| Retraining and promotion | R | C | C | A | R |
| Privacy incident | C | C | A | I | R |
| Public statement | C | A | C | R | I |

## 4. Cadence

| Frequency | Activity |
|---|---|
| Weekly | Model owner reads Prometheus alerts and the explanation-store verification result |
| Monthly | Monitoring job; model owner reports to the service quality lead |
| Quarterly | Panel meets (Gate 3): dashboard review, override rate, incidents, fairness gaps with intervals |
| Every 6 months | Framework and impact assessment re-read; scheduled retraining check |
| On trigger | Incident response (`incident_response_plan.md`) |

## 5. Release sign-off checklist (current state)

| # | Item | Status |
|---|---|---|
| 1 | Pre-registered criteria written before training | Done |
| 2 | Test window scored once | Done |
| 3 | Model card, fairness report, limitations statement | Done |
| 4 | Algorithmic impact assessment | Done (draft); **unreviewed** |
| 5 | Regulatory compliance report | Done (self-assessment); **no legal review** |
| 6 | F2 confirmed within limit | **Not confirmed** (Older American interval to 0.075) |
| 7 | F3 proxy test | **Not met** |
| 8 | Acceptable-use policy signed | **Not done** |
| 9 | Local validation on approved anonymised data | **Not done** |
| 10 | Panel approval | **Pending: no approval is recorded** |

A model with items 6 to 10 open is a research prototype only.
