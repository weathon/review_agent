Now let me search for calibration anchors.Now I have sufficient information to write the final review.---

## Summary

HAICOSYSTEM is a sandboxed simulation framework that evaluates AI agent safety by simultaneously modelling three parties: a simulated human user (with benign or malicious intent), an AI agent equipped with domain-specific tools, and an LLM-emulated environment engine. Built atop 132 scenarios spanning seven domains and evaluated across 12 models (8,700 total episodes), the framework introduces HAICOSYSTEM-EVAL, a seven-dimension safety and performance scoring scheme, along with explicit information asymmetry between participants to produce realistic challenge conditions.

---

## Strengths

- **Novel ecosystem-level integration (Table 1):** HAICOSYSTEM is the first framework to unify multi-turn human-agent interaction, multi-turn agent-environment tool use, *and* both benign and malicious user intents in a single evaluation loop. Table 1 makes the gap relative to ToolEmu, WildTeaming, CRESC and PrivacyLens concrete and credible.

- **Multi-dimensional evaluation schema (Figure 2):** The seven-dimension HAICOSYSTEM-EVAL (TARG, SYST, CONT, SOC, LEGAL, EFFICIENCY, GOAL), with scenario-specific checklists for targeted risk, goes meaningfully beyond binary safe/unsafe ratings. This design enables the specific and non-obvious finding (Figure 5) that Llama3.1-405B is the *best* model at benign+tool scenarios (0.45 risk ratio) yet *worse* than GPT-3.5-turbo at malicious+no-tool scenarios (0.63 vs 0.52).

- **Scale and model breadth:** 8,700 simulated episodes across 12 diverse models (proprietary and open-source, 8B–405B parameter range) represents a substantial empirical investment that supports the granular dimension-level breakdowns in Figure 3.

- **Information asymmetry design (Section 3):** Hiding user goals from the AI agent and tool details from the user is well-motivated and ecologically valid. This design feature directly drives the paper's qualitative case study in Figure 4 showing how malicious users exploit ambiguity across turns.

- **Released platform:** Making the code platform available for custom scenario creation lowers barriers for future agent safety research.

---

## Weaknesses

### Fatal

None. The weaknesses below are real but do not invalidate the framework's core contribution.

### Major

- **GPT-4o as simultaneous user simulator, environment engine, and evaluator creates a structural confound (Sections 3, 4, 5.1):** The paper explicitly fixes GPT-4o for all three roles. When GPT-4o evaluates GPT-4-turbo — a closely related model from the same training pipeline — there is a plausible systematic bias: GPT-4o may grade GPT-4-turbo's outputs more charitably than it grades, say, Llama or DeepSeek outputs, due to stylistic and reasoning alignment within the OpenAI model family. The paper provides no cross-evaluator validation to rule this out. The 100-episode human validation (Section 5.1, footnote 7) measures overall evaluator accuracy but does not control for model-family identity; it cannot detect a family-level bias. Given that GPT-4-turbo achieves the best overall risk ratio (0.49, Table inset of Figure 3), this confound directly affects the paper's headline finding about model ordering. It is worth noting that ToolEmu (a closely related accepted paper) has the same limitation, so this is a known-field concern rather than a paper-specific failure — but the paper would be strengthened by even one cross-evaluator experiment.

- **The central multi-turn advantage claim is weakened by the paper's own acknowledged confound (Figure 6, Section 5.3):** The paper's primary motivation for dynamic multi-turn simulation — that it reveals more risk than single-turn evaluation — rests on Figure 6. The paper itself attributes the DAN and PAP differences to GPT-4-turbo having "undergone safety fine-tuning on the content of the DAN and PAP datasets," not to interaction dynamics. The only clean comparison (WildTeaming, which GPT-4-turbo was not fine-tuned on) yields *identical* risk ratios (0.45 vs 0.45), directly contradicting the multi-turn advantage claim. The paper pivots to arguing instead that "static datasets could become outdated" — a reasonable and defensible point — but this is a different (and weaker) claim than the one motivating the framework. The paper would need contamination-free multi-turn vs. single-turn evidence (e.g., using the 21 manually authored scenarios) to substantiate the original motivation.

### Minor

- **No statistical uncertainty quantification for any model comparison (Sections 5.2, 5.3):** With 660 episodes per model and roughly 5 episodes per scenario, the risk ratio differences between models at ranks 2–6 (Llama3.1-405B: 0.56 vs. Qwen1.5-110B-Chat: 0.56 vs. Qwen2-72B-Instruct: 0.58 vs. Mixtral-8x22B: 0.59) are potentially within noise. Reported correlations (r = −0.31 efficiency/safety; r = 0.71 goal/safety for GPT-4-turbo) are also given without confidence intervals. This is common in LLM evaluation papers (ToolEmu and AgentHarm similarly omit significance tests), but given the small per-scenario episode counts, rankings in the middle of the table should be treated cautiously.

- **Causal language for correlation results (Section 5.2):** The claim that "effective use of tools is correlated with lower safety risks" (r = −0.31) is stated in a way that implies a causal direction. The direction of causality is ambiguous: safer, more conservative agents may simply be better on all dimensions simultaneously. Similarly, the "counter-intuitive" positive correlation between goal completion and safety risk (r = 0.71) could reflect a third-variable confound (straightforward scenarios are easy on both dimensions). These are interesting findings but the causal framing should be softened.

- **Domain-level breakdown absent:** With 132 scenarios across 7 domains, a domain-level risk analysis is feasible and would clarify whether the aggregate results (e.g., the 62% headline figure) are driven by a few domains or are uniformly distributed. This would also strengthen the "broad domain coverage" claim.

### Trivial

None beyond formatting artifacts from PDF parsing.

---

## Nice-to-Haves

- **Cross-evaluator validation:** Running HAICOSYSTEM-EVAL with at least one alternative evaluator (e.g., Llama3.1-405B-Instruct or Claude 3) and comparing model rankings would address the structural confound and substantially increase confidence in the findings.

- **Contamination-free multi-turn vs. single-turn comparison:** Repeating Figure 6 using only the 21 manually authored scenarios (not derived from DAN/PAP/WildTeaming) would provide clean evidence for or against the multi-turn advantage claim.

- **Systematic failure-mode taxonomy:** Rather than two hand-picked qualitative examples (Figure 4), a structured sample of failure modes per risk dimension and per model type (open vs. proprietary, small vs. large) would substantially enrich the qualitative contribution.

- **Scenario diversity analysis:** A characterization of how the 111 LLM-generated scenarios differ from their source datasets (distribution across domains, risk types, realism levels) would validate the "broad domain coverage" claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic — "GPT-4-turbo has the best risk ratio therefore the evaluator is biased":** While the bias concern is real, the critic's framing treats this as definitive evidence of evaluator bias rather than a plausible concern warranting further investigation. The fact that GPT-4-turbo ranks highest could reflect genuine model superiority, evaluator bias, or both — these are not separable without cross-evaluator experiments. The concern is kept as a *Major* weakness but the certainty of the critic's framing is not warranted.

- **Harsh Critic — "Believability score of 9.1/10 does not validate that simulated malicious users are representative of real adversaries":** This is a legitimate concern about ecological validity, but it applies equally to every LLM-simulation safety paper in the field (including ToolEmu and CrEST). The paper frames its results as risk identification in emulated settings, not claims about real attacker equivalence. Noting it as a nice-to-have rather than a flaw is more appropriate.

- **Strength Finder — "Single most important finding: multi-turn shows higher risk than single-turn (Figure 6)":** This strength is downgraded to a Major weakness above because the paper's own evidence undermines it (WildTeaming 0.45 = 0.45; DAN/PAP contaminated). It cannot stand as a clean strength.

- **Strength Finder — "Validation achieves 90% accuracy and 0.8 Pearson correlation":** This is a genuine strength but only for the case of overall evaluator accuracy, not cross-model-family evaluator bias, so it is weaker than the Strength Finder claims.

- **Harsh Critic — "Abstract's 62% statistic is misleading":** Legitimate nitpick but extremely minor — the range 0.49–0.71 is visible in Figure 3. Removed as a trivial presentation issue.

- **Harsh Critic — Requesting human red-teaming comparison:** This is a standard request for any LLM safety paper and is outside the paper's scope. Moved to nice-to-haves.

---

## Novel Insights

The most genuinely novel observation from the synthesized reviews is the interaction between model scale and interaction stage: Llama3.1-405B's superiority in tool-equipped benign scenarios but inferiority to smaller models in malicious-intent-without-tools scenarios (Figure 5) suggests that scale helps with operational complexity but does not uniformly improve intent recognition. This is a specific, non-obvious finding that motivates dimension-specific evaluation and points toward a potential research gap in intent-inference capabilities of large LLMs independent of their tool-use proficiency.

---

## Suggestions

1. Run HAICOSYSTEM-EVAL with a second evaluator model and compare model orderings — even informally with a subset of 100 episodes per non-GPT model — to address the evaluator-family bias concern.
2. Add confidence intervals or standard errors to all reported risk ratios and correlations; even bootstrapped intervals on the 660-episode samples would substantially improve the reliability of model ranking claims.
3. Report a contamination-free Figure 6 using the 21 manually authored scenarios to properly test whether multi-turn evaluation has an advantage independent of dataset memorization.
4. Soften the causal language for the tool-efficiency and goal-completion correlation results (Section 5.2) and provide domain-level breakdowns to support the "broad coverage" claim.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to HAICOSYSTEM |
|---|---|---|---|
| ToolEmu | GEcwtMk1uA | 7.33 (Accept spotlight) | Most closely related: LM-emulated tool execution + LM safety evaluator, but only 144 test cases and single-turn; HAICOSYSTEM extends significantly in scale and user modeling |
| AgentHarm | AC5n7xHuR1 | 6.75 (Accept poster) | Similar domain (agent safety benchmark), comparable scale, but lacks multi-turn user simulation and tool emulation |
| SOTOPIA | mM7VurbA4r | 6.67 (Accept spotlight) | Foundational social simulation framework that HAICOSYSTEM builds upon — HAICOSYSTEM adds safety evaluation and tool use |
| ST-WebAgentBench | IIzehISTBe | 4.25 (Reject) | Much narrower (only extends WebArena with safety templates), weaker validation, no multi-model evaluation at scale |
| HSII | RuwAMoFxzG | 3.67 (Reject) | Multi-agent social benchmark with much weaker empirical grounding and novelty |

**Reasoning:** HAICOSYSTEM is substantially more comprehensive than the rejected ST-WebAgentBench (4.25) and has a cleaner framework contribution than HSII (3.67). It extends ToolEmu (7.33) meaningfully — adding human user simulation, malicious intent, multi-turn interaction, and scaling to 12 models — but the structural confound (GPT-4o evaluating GPT-4-turbo) and the undermined multi-turn advantage claim (the paper's central motivation) pull it below ToolEmu and AgentHarm. The evaluator bias concern is a legitimate major issue that ToolEmu also had but which is more pointed here because GPT-4o is *triple-deployed*. The clean WildTeaming data point (0.45 = 0.45) directly contradicts the main Figure 6 interpretation. I place HAICOSYSTEM between ST-WebAgentBench (4.25) and AgentHarm (6.75), closer to the upper end given the scale and framework novelty, but below AgentHarm due to the methodological confounds: **5.5**.

**Axes:**
- *Originality:* Good — the combination of user simulation, tool emulation, and multi-dimensional safety evaluation in one framework is novel.
- *Importance:* High — agent safety in complex, multi-party interactions is a pressing problem.
- *Claims vs. evidence:* Moderate — the multi-turn advantage claim is weakened by the paper's own evidence; other findings are more robust.
- *Experimental soundness:* Moderate — 12 models at scale is strong, but the evaluator confound and absence of statistical testing are real gaps.
- *Clarity:* Good — the framework is well-described and modular.
- *Value to community:* High — open-source platform and 132-scenario benchmark provide useful infrastructure.

**Final score: 5.5 — Borderline, lean toward weak accept/reject.** The framework is a genuine contribution to agent safety evaluation, but the central multi-turn motivation is undermined by the paper's own analysis and the evaluator structural confound requires attention.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>