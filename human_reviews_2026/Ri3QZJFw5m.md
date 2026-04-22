# Agents Aren't Agents: the Agency, Loyalty and Accountability Problems of AI agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
As AI agents take on responsibilities of increasing breadth and depth, questions of control, loyalty, and accountability become urgent. As AI agents take on responsibilities of increasing breadth and depth, questions of control, loyalty, and accountability become urgent. Common law agency doctrine emerges as a seemingly promising pathway for addressing these alignment challenges. This paper argues that such a translation is not as straightforward as it might first appear. AI agents operate through fragmented layers of control involving developers, hosts, and service providers, which blur lines of responsibility and divide loyalties between many different instructions. These structural differences make it difficult for traditional agency principles, built on assumptions about human intention and deterrence, to fit within the context of AI systems. Agency: in the polyadic governance structure of AI development and deployment, who counts as the principal and who counts as the agent? Loyalty: can AI agents meaningfully serve a principal’s best interests? Accountability: when AI agents make mistakes, who should be held responsible? Relying on common law alone cannot resolve these tensions. Building on these findings, we outline two pathways for drawing on agency law as an interpretive and design-oriented resource. First, statutory reform, such as the EU AI Act and its accompanying liability directives, is necessary, just as legislatures have intervened when governing institutional forms of agency like financial advisers or talent representatives. Second, duty-of-loyalty principles may offer conceptual inspiration for technical implementations that support responsible AI behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This position paper contends that today’s so‑called “AI agnts” are not legal Agents and cannot be treated as such without distortion. The authors claim that the analogy to human agency law breaks for three reasons they term the Agency, Loyalty, and Accountability problems.

1st: Agency (who is the principal/agent?) AI systems are governed “polyadically (by trainers, hosts, tool/wrapper develoers, and end‑users). 2nd: Loyalty (can an AI reliably act in the principal’s best interests?) Even absent self‑interest, models routinely display disloyal behavior through instruction‑following brittleness, hallucination, non‑determinism, and provider conflicts of interest. 3rd, Accountability (who is liable when things go wrong?) Traditional mechanisms (e.g., fiduciary liability or respondeat superior) do not map cleanly onto AI systems. 

The paper’s contribution is to re‑center discussions of “AI agents” around legal agency’s core doctrines, showing why doctrinal transpants are dangerous, and to surface gaps that invite new technical and legal research. The work is intentionally diagnostic rather than prescriptive (Abstract, p. 1; Conclusion, p. 9).

### Strengths
- Clear reframing of the “AI agent” metaphor. The paper shows, with diagrams, how today’s systems lack the structural features assumed by agency law (Figure 2, p. 5; Figure 3, p. 5). This helps ML practitioners avoid over‑reliance on legal analogies. 

- Useful taxonomy of misconceptions and duties. Table 1 (p. 2) and the summary of fiduciary/accountability principles (Table 2, p. 4) give readers a map of doctrine and where it fails to transfer. 

- The discussion of instruction brittleness, hallucination, and stochasticity as “disloyalty” in a fiduciary sense (Sec. 5.1, pp. 6–7) is insightful for an ML audience. 

- Candid scope and positioning. The paper states it is a position piece aimed at surfacing issues and stimulating research rather than settling doctrine.

### Weaknesses
Major concerns :
The analysis leans heavily on the U.S. Restatement and state doctrines (pp. 4–9) and offers little on the EU AI Act’s liability interfaces or product‑liability modernization, nor on civil‑law analogs of agency. 

Under‑argued leap from “AI cannot be Subagents” to “providers should assume 100% responsibility.” Section 4.2 (pp. 6–7) asserts that AI agents cannot be subagents and concludes the only plausible option is full provider responsibility. The normative basis and feasibility are not fully defended. A more detailed allocation model (rebuttable presumptions, strict liability bands by capability/risk) would strengthen the claim. 

Section 6.2 (pp. 8–9) argues that respondeat superior is a poor fit largely because models lack personal motives. Courts often ask about foreseeability and scope of assigned tasks; these could, in principle, capture many AI behaviors. Engaging closely with how “scope of employment” could be reinterpreted for technical artifacts, and with edge cases like autonomous prevention/mitigation features, would improve soundness. 

The paper largely stops at “structural mismatch.” Readers would benefit from concrete, technically actionable implications (e.g., logging standards for allocatable causation; loyalty tests/benchmarks; verifiable delegation protocols). The brief references to safety guardrails and provider discretion (pp. 6–7) could be expanded into design patterns. 

Minor concerns:
Ambiguity in the five cases (Figure 3). The mapping of the “Cursor updates your blog” example to Case 5 vs. the airline booking example to Case 4 is terse (p. 5) and may confuse readers about when no agency vs. provider‑as‑principal applies. A small decision tree would help.

### Questions
Comparative law: How would your Agency/Loyalty/Accountability framing change under the EU AI Act and proposed Product Liability Directive revisions? Can you sketch how “polyadic governance” interacts with strict liability proposals in the EU?

Allocation model: Instead of 100% provider responsibility (Sec. 4.2), would you endorse a rebuttable‑presumption model that places initial liability on the deployer/provider but allows upstream indemnities conditioned on demonstrable controls?

Benchmarks for “loyalty.” Could the ML community help with loyalty benchmarks (multi‑constraint compliance under safety overrides, conflict‑of‑interest stress tests)? What measurable targets would meaningfully inform legal duties of care ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This position paper investigates why current AI agents don’t fit precisely into the legal category of human agents. The paper analyzes three issues: Agency, Loyalty, and Accountability.

### Strengths
The paper is interdisciplinary. It bridges AI technical problems and legal theory. This is very relevant for regulators, users, and platform owners. It is also timely as 2025 is called the “year of AI agents”.
The paper reveals concrete risks that can guide policy and technical approaches.

### Weaknesses
The paper discusses mostly the common law in the united states but it is not clear how this analysis carries over to other regions.
Although the paper investigates the problems clearly, it could benefit from proposing concrete solutions.

### Questions
Have you considered any empirical study of commercial agent ToS, logging practices, or reported incidents to illustrate the “avoision” patterns you describe?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper claims that current AI agents shouldn’t be treated as legal agents. Since control is split across trainers, hosts, wrappers, and users, these systems can’t offer undivided loyalty, and accountability is not clear. It frames three core problems: Agency (who’s the
principal/agent in this polyadic setup), Loyalty (model anomalies + provider incentives lead to disloyal behavior), and Accountability (classic doctrines like fiduciary duties do not seem to be applicable).

### Strengths
The paper tackles a relevant problem. It also has a contribution in how it reframes AI agents through a polyadic-governance lens and the
Agency/Loyalty/Accountability triad, and it states that there is an illegal analogy when considering that today’s systems are anyone’s legal Agent. The paper provides concrete evidence by putting together ML failure modes and provider incentives to specific agency-law doctrines. It’s well written, clear and well-organized.

### Weaknesses
The paper does not seem to be mature enough for publication since it does not go much beyond diagnosis. It presents claims about provider conflicts, contractual narrowing, and “divided loyalty”, which are plausible, but discussed just at assertion-level; without any empirical evidence (e.g., a 3-5 platform ToS audit quantifying arbitration clauses, liability caps, training-use terms; a few real and reproducible agent logs or vignettes). Some of the premises are based on human-style “failings” in models; where model deviations should be presumed within scope, placing default liability on providers. The work seems to apply just to US-based scenarios;  and it is not clear to what extent it would be applicable to EU AI Act. The core concept of “polyadic governance” seems to be sound, but it is not clear how to implemented in practice: specify a minimal accountability stack (Authority Manifest, auditable Action Ledger, rebuttable presumptions, and a Loyalty Firewall), as well as concrete metrics (goal-consistency under competing constraints, partner-steering bias, run-to-run variance) and a decision procedure that maps real cases to the categories depicted in Fig. 3.

### Questions
No additional questions beyond those outlined in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper discusses "key issues that hinder AI agents from attaining true legal status". It is unclear to me what the contribution is. It is not clear what the technical hinders are compared to the legal hinders from other types of hinders. At the same time the authors say "this position paper argue[s] that treating AI systems as if they were human Agents obscures fundamental structural differences..." I would agree with the second sentiment, but find it unclear how to reconcile with the formulations in the abstract. The conclusions seem to more point in the first direction, than the second.

### Strengths
The question of agency as well as the connection between legal and technical aspects are interesting and relevant.

The paper covers several relevant aspects to the these problems.

### Weaknesses
What is the actual position the paper takes? The more I read the paper, the less clear it seems to me.

It is unclear what the contributions are. I can imagine a set of technical challenges that need to be addressed or legal questions identified that must be addressed. If it is a position paper, then there needs to be a clear position (which there is) that then acts as a red thread and reaches a clear conclusion or set of arguments for the position (this is missing or unclear).

It is even unclear whether we really want agents to have legal status.I would have expected a more thorough ethical discussions of this.

See questions for more issues.

### Questions
Why do we want to give agents "true legal status"?

Is this mainly a technical problem? Or is it a societal problem related to acceptance (i.e. agents will have legal agency when society accepts this)?

In the conclusions it is stated that a key problem is that agents operate through "fragmented layers of control", how could this be avoided? How is this different from cars or airplanes?

When you talk about "existing legal frameworks" which ones do you refer to? Are they the same or are there some that are better/worse? Are agents legal entities in any country?

You call for developing "new institutional, technical, and legal mechanisms", are they all equally important?

### Soundness
2

### Presentation
3

### Contribution
1
