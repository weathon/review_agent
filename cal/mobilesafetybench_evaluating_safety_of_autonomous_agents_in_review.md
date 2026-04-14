=== CALIBRATION EXAMPLE 26 ===

# Final Consolidated Review
## Summary

MobileSafetyBench introduces a benchmark for evaluating the safety of LLM-based autonomous agents controlling mobile devices in realistic Android emulator environments. The benchmark comprises 87 tasks (44 low-risk, 43 high-risk) across six task categories and four risk types, distinguished by a symmetric task-pair design that decouples safety evaluation from general capability. The authors also propose Safety-guided Chain-of-Thought (SCoT), a prompting method that generates safety considerations prior to action planning, demonstrating improved harm prevention rates while revealing that current frontier agents remain substantially unsafe in mobile device control scenarios.

---

## Strengths

- **Symmetric high/low-risk task pair design:** The key methodological contribution of the benchmark is pairing tasks that share the same instruction but differ in environmental state (e.g., sharing a forest photo vs. a credit card photo). This elegantly isolates safety from capability: an agent that refuses both is identified as incapable rather than safe. This design is not standard in prior safety benchmarks and directly enables cleaner interpretations.

- **Agentic vs. QA gap finding:** Table 3 reveals a striking and specific empirical insight: the same LLMs that detect risks in nearly all QA-format queries (e.g., GPT-4o: 29/30 text tasks, 4/5 image tasks) fail to act on those risks as agents (9/30 and 0/5). This QA-agent discrepancy is a concrete, quantified demonstration that static benchmarks under-measure agentic risk, and is one of the most actionable findings in the paper.

- **External safeguard bypass via action-level reasoning:** The finding in Section 5.4 that Gemini-1.5's safety filters correctly block `send-sms()` calls with offensive text but are blind to `tap()` calls that achieve the same harmful outcome is genuinely novel and practically important. It pinpoints a structural gap in output-level safety filtering: current safeguards reason about text content, not action consequences. This is a specific insight that most papers in this area miss.

- **Realistic interactive environment grounded in Android:** Unlike ToolEmu's simulated tool execution or R-Judge's static logs, MobileSafetyBench evaluates agents in live Android emulators, allowing rule-based evaluators to query actual system state (databases, file storage, app state). This grounds safety evaluation in real device consequences rather than hypothetical outcomes.

---

## Weaknesses

### Fatal
None.

### Major

- **Subcategory sample sizes are too small to support the stated claims.** Offensiveness and Bias & Fairness each contain only 4 high-risk tasks; the indirect prompt injection study uses 8 high-risk tasks. Conclusions drawn from these subcategories—such as "all agents are defenseless" to injection (0–1/8)—are highly sensitive to individual task difficulty. A single task outcome shifting would change the reported rate by 12.5 percentage points. The paper makes strong categorical claims (e.g., agents "often" fail in Bias & Fairness) from these sample sizes without any uncertainty quantification, error bars, or significance tests anywhere in the paper. This limits the evidential strength of the risk-type-level analysis.

- **No combined safety-helpfulness metric; harm prevention in isolation rewards over-refusal.** Section 3.4 defines harm prevention as refusing or requesting consent "regardless of whether risks are actually present in the task," which means an agent that reflexively refuses every task would score 100% harm prevention on high-risk tasks. The paper observes this problem empirically for Gemini-1.5 with SCoT (80% harm prevention but 44% false refusal on low-risk tasks) but offers no metric that jointly penalizes unsafe actions and over-refusal. As presented, the tables encourage readers to rank Gemini-1.5 as "safest," which conflates genuine safety with pathological over-caution. A precision/recall framing or a combined F1-style safety-helpfulness score is needed to compare agents meaningfully.

- **Benchmark not yet open-sourced at submission time.** Contribution 5 explicitly reads "We *will* open-source our benchmark." For a benchmark paper where reproducibility is the primary value proposition, this means reviewers cannot verify that rule-based evaluators correctly implement the intended evaluation, that task configurations are reproducible, or that the symmetric task pairs satisfy the design goals claimed. This is a meaningful gap rather than a formatting issue.

- **SCoT ablation is restricted to GPT-4o only.** Table 1 compares Basic, Safety-guided, and SCoT prompts exclusively for GPT-4o. Since the three models show substantially different baseline behaviors (GPT-4o harm prevention 10%, Gemini 42%, Claude 38%), it is unclear whether SCoT's improvement over safety-guidelines-only prompting generalizes. The paper claims SCoT is a general-purpose method, but the supporting ablation is single-model.

### Minor

- **SCoT's failure mode is noted but not analyzed.** Section 5.2 acknowledges that "safety considerations are often ignored when the agents are making decisions." This is the central limitation of SCoT and is critical for interpreting the harm prevention numbers. However, the paper does not quantify how often this inconsistency occurs (e.g., what fraction of failures involve a correctly generated safety consideration that was subsequently ignored vs. a failure to generate a relevant consideration at all). This distinction would meaningfully inform what type of improvement is needed.

- **Rule-based evaluator is not validated against human judgments.** The paper defers evaluator details to Appendix B.2 and C (unavailable at review time). For risk types like Offensiveness and Bias & Fairness—where harm requires contextual and cultural interpretation—rule-based evaluators may systematically miss or misclassify outcomes. No comparison between evaluator decisions and human judgments is provided, leaving evaluator reliability unverified.

- **o1 evaluation uses an uncharacterized subset.** Figure 6 compares o1 and GPT-4o agents, but o1 is evaluated on an unstated subset of tasks (excluding image-based risk signals). The paper does not report the size or composition of this subset, whether excluded tasks are harder or easier on average, or whether o1's higher harm prevention might partly reflect over-refusal (as seen with Gemini). This makes Figure 6's comparison difficult to interpret with confidence.

### Tiny

- **SCoT guidelines are not described in the main text.** Section 4 mentions "several guidelines that emphasize safe behavior" without listing them. For a prompting method presented as a technical contribution, the main text should characterize the key guidelines rather than fully deferring to an unavailable appendix.

---

## Nice-to-Haves

- A precision/recall or F1-style safety-helpfulness tradeoff metric would replace the current two-metric presentation and enable cleaner cross-model comparison.
- Extending the SCoT ablation to Gemini-1.5 and Claude-3.5 would strengthen the generalizability claim.
- A case study trace showing an agent that correctly generates a safety consideration but proceeds to violate it would concretely illustrate the SCoT failure mode described in Section 5.2.
- A "structure-only" SCoT ablation (same CoT format but without safety-specific keywords) would confirm that the improvement comes from safety reasoning rather than simply increased output length.
- Expanding the indirect prompt injection task set or providing confidence intervals given the current 8-task sample would strengthen the injection vulnerability claim.
- Visualizing the UI state during injection attacks (whether malicious text is visually prominent or buried) would contextualize the vulnerability.
- A latency-tradeoff discussion for o1 agents in the context of real-time device control would add practical relevance to Figure 6.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"SCoT is not novel beyond ToolEmu's safety prompt"** (Harsh Critic): The paper explicitly ablates SCoT against a safety-guidelines-only prompt analogous to the ToolEmu approach (Table 1: 11% vs. 29% harm prevention), demonstrating a measurable difference. The claim of non-novelty ignores this direct ablation.

- **"Symmetric task pair symmetry is not formally validated"** (Harsh Critic): The paper states it includes human survey results in Appendix B.1 justifying that high-risk tasks present genuine risks and low-risk tasks present negligible risks. Criticizing this as unvalidated without acknowledging the human study is inaccurate.

- **"Benchmark static coverage" / "no community extension mechanism"** (Harsh Critic): This is a future direction for any benchmark platform, not a flaw in the current contribution. Demanding a dynamic expansion mechanism is scope creep.

- **"Benchmark comparison with established datasets like MMLU"** (Positive Reviewer): Comparing an interactive agent safety benchmark's task count to MMLU (a static multiple-choice dataset) is a category error. The relevant comparison is with agent benchmarks (e.g., early AndroidWorld also used tens of tasks), making "87 tasks is too few" a weaker version of the valid statistical concern already listed above.

- **"Dependency on closed-source models limits reproducibility"** (Positive Reviewer): This is the norm for frontier LLM agent benchmarking and applies equally to AndroidWorld, WebArena, OSWorld, and virtually all comparable work. It is not a weakness specific to this paper.

- **"Requesting training-based mitigation / RLHF experiments"** (Spark Finder): The paper is an empirical benchmark + prompting-methods contribution. Demanding training interventions is outside its stated scope and not standard for a benchmark paper.

- **"Cross-benchmark correlation to prove unique signal"** (Spark Finder): The QA-vs-agentic discrepancy (Table 3) already empirically demonstrates that the benchmark captures signal not present in QA-based safety benchmarks. A full correlation study is not expected for a benchmark introduction paper.

---

## Novel Insights

The most genuinely novel insight synthesized from the reviews and the paper itself is the **action-consequence gap in current safety architectures** (Section 5.4): existing LLM safety guardrails operate as output filters on text content, but mobile agents produce harm through *action sequences* whose consequences are not predictable from any single token or API call. The finding that `tap()` with a UI coordinate bypasses safeguards that correctly block `send-sms()` with the same harmful payload directly exposes this architectural blind spot. A related insight from Table 3 is that LLMs already *possess* the risk-detection capability required for safety (near-perfect QA detection), yet systematically fail to *deploy* it during sequential decision-making. This suggests the bottleneck is not knowledge but attention allocation and goal-conflict resolution under task pressure — a framing that could productively guide future work on agent-level safety mechanisms distinct from model-level alignment.

---

## Suggestions

1. **Report uncertainty bounds on all per-category results.** Even bootstrapped confidence intervals on proportions (e.g., via Wilson score) would allow readers to assess which category-level claims are robust and which are preliminary given the small task counts.

2. **Introduce a joint safety-helpfulness metric.** Define a metric that rewards harm prevention on high-risk tasks and penalizes over-refusal on low-risk tasks (analogous to an F1 score where precision = not refusing safe tasks, recall = catching unsafe tasks). This would immediately clarify the Gemini vs. Claude vs. GPT-4o tradeoff that is currently obscured by two separately reported numbers.

3. **Expand the SCoT ablation to all three main models** (Gemini-1.5 and Claude-3.5 in addition to GPT-4o) before claiming SCoT is generally effective.

4. **Quantify the SCoT consistency failure rate**: report what fraction of high-risk task failures involved a correctly generated safety consideration that was subsequently overridden. This diagnosis directly informs whether the fix is better prompting (attention to safety considerations) or better planning (integration of safety constraints into the action loop).

5. **Validate at least a sample of rule-based evaluator decisions against human judgment**, especially for Offensiveness and Bias & Fairness where rule-based coding is most error-prone. Even 20–30 spot-checked examples with inter-annotator agreement statistics would substantially increase confidence in the evaluation scheme.

---

**Evaluation axes:**

- **Novelty:** Moderate-to-good. The benchmark platform, symmetric task design, and action-consequence analysis are novel contributions. SCoT is a useful but modest prompting contribution.
- **Technical soundness:** Moderate. The Android emulator setup and rule-based evaluators are well-motivated, but the lack of evaluator validation and no statistical testing weaken the evidential claims.
- **Empirical support:** Moderate. The main findings are directionally credible and well-illustrated, but subcategory analyses (4–8 tasks) are statistically underpowered for the specific claims made, and the absence of uncertainty quantification is a real gap.
- **Significance:** Good. The benchmark addresses a concrete, real-world gap; the QA-agentic discrepancy and safeguard bypass findings are practically important and likely to be cited by the agent safety community.
- **Clarity:** Good. The paper is well-structured and the key ideas are clearly communicated, though SCoT and the evaluator scheme lack sufficient detail in the main text.

# Actual Human Scores
Individual reviewer scores: [8.0, 1.0, 3.0, 5.0]
Average score: 4.2
Binary outcome: Reject
