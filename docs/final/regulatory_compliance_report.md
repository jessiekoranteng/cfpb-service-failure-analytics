# Regulatory Compliance Report

Scope: the complaint service-failure risk system as built (public CFPB data, generic Ghanaian retail-bank framing).
Author's note: this is a self-assessment by a student analyst, **not legal advice and not a compliance certification.** A qualified data-protection adviser must review it before any use with real customer data.

## 1. Headline

| Regime | Applies to this system today? | Why | If real bank data were introduced |
|---|---|---|---|
| Ghana Data Protection Act, 2012 (Act 843) | **Yes, as the governing standard for the intended deployer**; today the data are public and already published | The target setting is a Ghanaian bank; the CFPB records are used as published, with no names, accounts, narratives or ZIP codes | Fully applies: registration, lawful basis, purpose limitation, security, data-subject rights, automated-decision rights |
| EU GDPR | **No, assessed by analogy** | No EU establishment, no EU data subjects targeted; CFPB complainants are U.S. consumers | Applies if EU residents' data were processed or EU customers served |
| California CCPA / CPRA and CPPA regulations | **No, assessed by analogy** | No California business is operating the system; complainants are not being collected from | Would apply to a covered business; automated-decisionmaking rules phase in from 1 January 2027 |
| U.S. HIPAA | **Not applicable** | No protected health information; the operator is not a covered entity or business associate | Applies only if complaints carried health information handled by a covered entity |

## 2. Ghana Data Protection Act, 2012 (Act 843)

Section numbers are from the Act as published on GhaLII. The Act applies to a data controller (the bank) once personal data are processed.

| Requirement (section) | How the system meets it | Gap |
|---|---|---|
| Eight privacy principles (s. 17: accountability, lawfulness, purpose, compatible further processing, quality, openness, security, data-subject participation) | Mapped line by line in section 2.1 | See gaps below |
| Minimality (s. 19) | 9 intake fields in the model; 12 of 15 source columns loaded; ZIP, narrative and company public response excluded | None for current data |
| Specified purpose and compatibility (ss. 22, 25) | Purpose: service-quality prioritisation, never a decision about an individual; written in the Charter | Acceptable-use policy not yet signed |
| Retention (s. 24) | Explanation Store 90 days; raw layer course plus 12 months; policy reviewed every 90 days (enforced in `governance/policy.yaml`) | Retention for real bank data would follow the bank's schedule |
| Quality of information (s. 26) | 37 Great Expectations checks; label caveat (outcome is a company decision) | Label bias cannot be removed (risk H6) |
| Security (s. 28) | HMAC-SHA256 pseudonyms with environment-held key; role matrix; hash-chained logs; non-root containers | No encryption-at-rest or access control on the public demo; the public API has no authentication |
| Registration (ss. 27, 47, 53) | Not applicable to public data | The bank must be registered with the Data Protection Commission before processing; the processing here would be listed |
| Data-subject access and correction (ss. 32, 33, 35) | Stored explanation retrievable by id | Complainants are not currently told a score exists |
| Automated decision-taking (s. 41) and explanation of logic (s. 35) | Design avoids sole reliance: a reviewer decides every case; reasons accompany every score | Panel to confirm that "reorders human reading" is not a decision "based solely on automated processing" |
| Research and statistics exemption (s. 65) | Relevant to the coursework use of public data | Not a basis for operational use |
| Transfer abroad | The CFPB data originate in the U.S.; model files are static | Hosting region of the public demo is not in Ghana (reconciliation log C9) |

### 2.1 The eight principles

Accountability: named owner and panel (pending). Lawfulness: public data today; consent or other lawful basis needed for bank data. Purpose: stated and tested by the F4 contract.
Compatible further processing: no secondary use is built in. Quality: validation suites. Openness: Public Trust Statement. Security: controls above. Data-subject participation: explanation lookup; correction route is the bank's complaint process.

## 3. GDPR (by analogy)

| Article | Requirement | Position |
|---|---|---|
| 5 | Lawfulness, purpose limitation, minimisation, accuracy, storage limitation, integrity, accountability | Same as Act 843 principles above |
| 6, 9 | Lawful basis; no special-category data without a condition | No special-category data are model inputs. The protected tags (age 62+, military status) are audit-only; a bank-held equivalent of a protected characteristic would need its own basis |
| 13 to 15 | Information and access | Explanation lookup supports access; no privacy notice exists yet |
| 22 | No decision with legal or similarly significant effect based solely on automated processing; meaningful information about the logic | Reviewer decides; reasons given. Prioritisation of reading order is unlikely to be a "decision", but the point should be confirmed |
| 25, 32 | Data protection by design; security | Minimal inputs; audit log; pseudonyms |
| 35 | Data protection impact assessment for high-risk processing | `algorithmic_impact_assessment.md` serves as a draft; a formal DPIA is required before real data |
| 89 | Research safeguards | Relevant only to coursework use |

## 4. CCPA / CPRA (by analogy)

| Topic | Position |
|---|---|
| Who is covered | For-profit businesses meeting revenue or data thresholds that do business in California. This project is not one. |
| Personal information | The CFPB file is published government data, treated as publicly available information; the model never receives identifiers. |
| Rights (know, delete, correct, opt out of sale or sharing, limit use of sensitive data) | Nothing is sold or shared; deletion of stored requests is covered by 90-day retention. |
| Automated decisionmaking technology (ADMT) and risk assessments | The CPPA's finalised regulations apply risk-assessment duties from 1 January 2026 and ADMT duties for significant decisions from 1 January 2027. A complaint-reading order is unlikely to be a "significant decision", but a deployer should record that reasoning in its risk assessment. |

## 5. HIPAA

Not applicable. The system handles no protected health information and the operator is neither a covered entity nor a business associate. Complaint narratives, which could contain health details, are excluded from the model and the dashboards by design. Should a deployer pass narratives to the system, HIPAA, and in Ghana the Act 843 special-personal-data rules, would need review first.

## 6. Open compliance actions

| # | Action | Owner | Before |
|---|---|---|---|
| 1 | Obtain written advice on Act 843 s. 41 and whether reading-order prioritisation is a decision based solely on automated processing | DPO | Any use with bank data |
| 2 | Register the processing with the Data Protection Commission | DPO | Any use with bank data |
| 3 | Sign an acceptable-use policy (no use to delay complainants or rank staff) | Service quality lead | Pilot |
| 4 | Add authentication, rate limiting and TLS termination to the public API | Engineering | Any non-public hosting |
| 5 | Write a complainant-facing notice (draft in `public_trust_statement.md`) | Compliance | Pilot |
| 6 | Formal DPIA | DPO | Any use with bank data |

## 7. Sources

Data Protection Act, 2012 (Act 843) (Ghana): https://ghalii.org/akn/gh/act/2012/843/eng@2012-05-18 .
Regulation (EU) 2016/679 (GDPR), https://eur-lex.europa.eu/eli/reg/2016/679/oj .
California Privacy Protection Agency (2025), finalised CCPA regulations announcement, https://www.cppa.ca.gov/announcements/2025/20250923.html .
U.S. HHS, HIPAA Privacy Rule summary, https://www.hhs.gov/hipaa/for-professionals/privacy/index.html .
