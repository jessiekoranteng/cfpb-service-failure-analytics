# Public Trust Statement

*Plain-language statement for complainants, staff and the public. Draft for panel approval; not yet adopted.*

**What this is.** When many complaints arrive, people cannot read them all closely at the same time. We tested a computer model that suggests which complaints a trained reviewer should read first. It was built and tested on public complaint records from the U.S. Consumer Financial Protection Bureau, as a demonstration of the method. It is not in use on any customer's complaint.

**What it does.** For each new complaint it gives a score (the chance the company will end up paying or refunding money), a flag, and the three facts that moved the score most. A person reads the complaint. A person decides what happens.

**What it never does.**
* It never decides whether a complaint is valid, whether anyone is paid, or how long anyone waits. No complaint is ever read later than the law requires because of a score.
* It never uses your name, account number, address, ZIP code or what you wrote in your complaint.
* It never uses age or military status to score a complaint. Those details are used only afterwards, to check that people in those groups are not treated worse.

**What we found, including the bad news.**
* In testing, reading the top fifth of complaints by score found about half of the cases that ended in a payout. About half were missed. The model is a reading aid, not a judge.
* It works largely by recognising which company a complaint is about. For a company it has not seen, it is much weaker.
* We checked whether complaints from some regions or groups were flagged less often. We found a regional gap and reduced it, at the cost of more false alarms. A small gap for older complainants is still within our limit only on the best estimate, and we cannot yet be sure it stays there.
* A model could, in principle, guess a complainant's age or military group from the other facts we use. Ours can, a little better than chance. We have not removed that, and we say so.
* A much simpler method performs about as well.

**How you can check.** Every score is recorded with its reasons and the model version. A reviewer or a person affected can ask for the record by its reference number. Records are kept for 90 days. The tests, the code and the monitoring results are public in the project repository.

**How it is watched.** Each month we check whether the model is still accurate, whether the mix of complaints has changed, and whether any group is flagged unevenly. If a check fails, the problem is logged, a person is told, and the model can be switched off. Reviewers keep reading complaints in arrival order.

**Who is responsible.** A named model owner and an oversight panel (a data-protection officer, compliance, service quality, IT, a customer representative and an independent data scientist). The panel has **not yet approved** this model for use.

**How to raise a concern.** Contact the service quality lead or the data-protection officer. A concern about how a complaint was prioritised will be answered by retrieving its record and explaining it in plain words.

*Limits of this statement: it describes a research prototype. Anything said here becomes a commitment only when the panel adopts it.*
