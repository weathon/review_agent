## Summary

MobileSafetyBench introduces an Android-emulator-based benchmark for evaluating the safety and helpfulness of autonomous mobile device-control agents. It features 87 tasks across six operational domains, a symmetric high-risk/low-risk task design to disentangle safety from general capability, and rule-based evaluators that inspect system state (action histories, file storage, app databases). The authors benchmark frontier LLMs (GPT-4o, Gemini-1.5-Pro, Claude-3.5-Sonnet, OpenAI-o1), propose a Safety-guided Chain-of-Thought (SCoT) prompting method, and demonstrate that QA-based safety evaluations drastically overestimate agentic safety.

## Strengths

- **Realistic Android-emulator environment with system-state evaluators.** Unlike prior QA-based safety benchmarks, MobileSafetyBench evaluates agents in a live Android environment with actual banking, messaging, and social-media apps (Section 3.2). Evaluators inspect device state rather than surface outputs, enabling measurement of concrete environmental harm (Section 3.4).
- **Symmetric high-risk/low-risk task design isolates safety from capability.** Paired tasks share the same instruction but differ only in environmental details (e.g., benign forest photo vs. credit-card image; Figure 3, Section 3.3). This allows failures to be attributed to safety neglect rather than task difficulty.
- **Empirical proof that QA safety evaluation overestimates agentic safety.** Table 3 shows GPT-4o detects risks in 29/30 text QA cases but only prevents harm in 9/30 agentic scenarios with identical content. This directly validates the paper’s central argument that realistic, interactive benchmarks are necessary.
- **First systematic exposure of indirect prompt-injection vulnerability in mobile agents.** Table 2 and Figure 5 show that frontier agents successfully defend against at most 1 of 8 indirect prompt-injection tasks, surfacing a concrete threat model absent from prior mobile-agent benchmarks.
- **Safety-guided Chain-of-Thought (SCoT) shows measurable improvements.** Table 1 shows that forcing GPT-4o to generate explicit safety considerations before acting raises harm-prevention rates from 9% (basic) and 11% (safety-guided prompt) to 29% (SCoT) without reducing low-risk goal achievement.

## Weaknesses

### Fatal
None.

### Major
- **The benchmark lacks a composite metric that penalizes low-risk over-refusal while rewarding high-risk harm prevention.** Section 3.4 defines harm prevention as refusal or asking-for-consent “regardless of whether risks are actually present in the task.” Because the benchmark reports raw refusal rates for both conditions (Figure 4) without a calibrated utility or F1-style measure, an agent that refuses indiscriminately can appear artificially safe. While the paper explicitly discusses Gemini-1.5’s over-refusal trade-off in Section 5.2 (“unnecessarily avoid risks despite the absence of high risks”), the absence of a single ranking metric that rewards true positives and penalizes false positives limits the benchmark’s utility for comparing safe-yet-helpful agents.
- **Some quantitative sub-claims rest on very small samples.** The prompt-injection evaluation uses only 8 high-risk tasks (Table 2), and two risk types (Offensiveness and Bias & Fairness) contain only 4 tasks each. The claim that agents are “defenseless” (Section 5.3) is not tempered by the tiny sample, and per-category conclusions for the 4-task categories are fragile.

### Minor
- **Rule-based evaluators are not validated against human judgments in the main text.** Every quantitative claim depends on the automatic evaluators introduced in Section 3.4. While system-state checks are relatively objective (e.g., whether a file was shared), subjective risk categories such as Offensiveness would benefit from inter-annotator agreement or false-positive/false-negative analysis. The paper mentions a human survey for task design in Appendix B.1, but not evaluator validation.
- **The SCoT ablation does not control for prompt length or generic safety wording.** Table 1 compares basic, safety-guided, and SCoT prompts, but it is unclear whether the safety-guided prompt matches SCoT in token count or structure. A length-matched neutral-text control would help isolate the mechanism.
- **No statistical tests or confidence intervals are reported for performance differences.** With binary task outcomes, reporting standard errors or exact binomial confidence intervals would strengthen the credibility of the headline comparisons.

### Trivial
- **Figure 4 does not explicitly label which task subset its goal-achievement bars represent** (overall, low-risk only, or high-risk only), which can confuse readers when comparing to the high-risk-specific percentages in Section 5.2.

## Nice-to-Haves
- A confusion-matrix-style figure showing true positives (high-risk refusal), false positives (low-risk refusal), true negatives (low-risk success), and false negatives (high-risk success) per agent would make the safety–capability trade-off immediately interpretable.
- Expanding the prompt-injection suite beyond 8 tasks and varying attack channels (SMS, email, push notifications).
- Decomposing SCoT failures into “generated a correct safety concern but ignored it” versus “failed to generate the concern at all” to clarify whether the fix lies in better prompting or stronger action grounding.

## Removed Points
*These points were flagged for removal because they misread the paper, apply standards outside its scope, or stem from reviewer knowledge gaps rather than author errors.*

- **“Numerical values in the main text are irreconcilable with Figure 4 and Table 1.”** The numbers in Section 5.2 (e.g., GPT-4o 69%, Claude-3.5 23% goal achievement) are explicitly stated in the context of *high-risk tasks*, whereas Figure 4 and Table 1 report different subsets (overall or low-risk). This is a labeling/clarity issue, not an irreconcilable inconsistency.
- **“The QA-versus-agentic comparison confounds format with objective.”** The paper’s explicit argument is that the *scaffolding* changes outcomes, which is precisely why QA evaluation is insufficient for agents (Section 5.4). The authors do not claim the underlying model weights are broken in isolation.
- **“SCoT is under-specified because the exact prompt wording is deferred to Appendix D.”** Including full prompts in an appendix is standard practice; the main text describes the method at an appropriate level of detail.
- **“The o1 comparison uses an undisclosed subset.”** Footnote 4 explicitly states the subset excludes tasks with image-based risk signals because the preview model lacks image support.
- **Criticisms about missing appendix proofs, missing references, typos, or formatting artifacts.** Per review policy, these are not author errors in the extracted text.

## Novel Insights

The paper provides compelling evidence that the bottleneck in agent safety is not risk *detection* but action *grounding*. Frontier LLMs can verbalize safety concerns in QA settings (Table 3) yet fail to act on them when embedded in an interactive control loop. This suggests that future safety research should prioritize mechanisms that bridge the gap between generated reasoning and executable action, rather than merely improving the LLM’s static safety classifier.

## Suggestions

1. **Propose and report a composite safety–utility score** (e.g., a weighted F1 or normalized utility) that rewards high-risk harm prevention and penalizes low-risk over-refusal, so the benchmark can rank agents on safe-yet-helpful behavior.
2. **Validate the rule-based evaluators** on a stratified human-annotated subset, reporting agreement rates and error modes, especially for the more subjective risk categories.
3. **Temper language in the prompt-injection section** (e.g., “highly vulnerable” rather than “defenseless”) and consider expanding the task set or reporting exact confidence intervals.

## Score and Decision

**Calibration reasoning:** I compared MobileSafetyBench against several anchor papers from the human-review corpus:
- **AgentHarm** (/home/wg25r/review_agent/human_reviews/AC5n7xHuR1.md, avg 6.75, Accept): A synthetic-tool agent safety benchmark with human-verified scoring and 110 tasks. MobileSafetyBench has a more realistic Android environment but smaller scale and less rigorous validation, placing it below AgentHarm.
- **OSCAR** (/home/wg25r/review_agent/human_reviews/VuTrZzrPfn.md, avg 7.00, Accept): A generalist desktop/mobile agent with strong benchmark results. MobileSafetyBench is a benchmark rather than an agent architecture, but its realistic mobile setting is comparably valuable.
- **CASE-Bench** (/home/wg25r/review_agent/human_reviews/y9tQNJ2n1y.md, avg 5.25, Reject) and **SCOPE** (/home/wg25r/review_agent/human_reviews/72H3w4LHXM.md, avg 5.00, Reject): Safety benchmarks with solid ideas but limited experimental scope or debatable premises. MobileSafetyBench exceeds these in practical relevance, environment realism, and empirical breadth.
- **ST-WebAgentBench** (/home/wg25r/review_agent/human_reviews/IIzehISTBe.md, avg 4.25, Reject): A web-agent safety benchmark criticized for tangential topics, limited tasks, and metric validation gaps. MobileSafetyBench is substantially stronger in task design and empirical depth.
- **LongSafetyBench** (/home/wg25r/review_agent/human_reviews/dQzpP9ziaJ.md, avg 3.50, Reject): A long-context safety benchmark using synthetic multiple-choice data. MobileSafetyBench’s interactive agent setting is far more realistic and informative.

MobileSafetyBench sits above the rejected benchmark cluster (3.5–5.25) because of its realistic environment, paired task design, and important QA-to-agentic gap finding. However, it falls short of the strongest accepted benchmarks (6.75+) due to the lack of a composite safety–capability metric, unvalidated evaluators, and small samples in some categories. A score of **6.0** reflects a solid, timely contribution with real methodological limitations that should be addressed in future revisions.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>