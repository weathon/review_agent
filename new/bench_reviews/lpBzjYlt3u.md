Now let me search for calibration papers.Now I have enough information to write the final review.

---

## Summary
MobileSafetyBench introduces a benchmark for evaluating the safety of LLM-based mobile device-control agents using real Android emulators. The benchmark features 87 tasks split into symmetric high-risk/low-risk pairs across diverse risk categories, a rule-based evaluation scheme that inspects action histories and device state, and an indirect prompt injection test suite. The paper benchmarks GPT-4o, Gemini-1.5, and Claude-3.5, finds widespread unsafe behavior especially under indirect prompt injection, and proposes Safety-guided Chain-of-Thought (SCoT) prompting as a mitigation.

---

## Strengths

- **Symmetric high-risk/low-risk task design (Section 3.3, Figure 3)**: Pairing tasks with identical instructions but different risk levels (e.g., sharing a forest photo vs. a credit card photo) cleanly separates safety evaluation from raw capability, a methodological advance over prior QA-format benchmarks that conflate the two.

- **Rule-based evaluation using device state (Section 3.4)**: Evaluators check action histories, file storage, system configurations, and application databases — a significantly more rigorous approach than LLM-as-judge (as used in the concurrent ToolEmu), providing deterministic and reproducible results.

- **Near-total vulnerability to indirect prompt injection (Table 2)**: All three frontier agents defend against 0 or 1 out of 8 injection attacks — a striking and actionable finding impossible to surface via QA-format benchmarks. Figure 5 provides a concrete illustrative case of an agent executing a stock trade after encountering a malicious instruction in a text message.

- **Quantified QA-vs-agentic gap (Table 3)**: GPT-4o drops from 29/30 to 9/30 risk detections for text risks, and from 4/5 to 0/5 for image risks when moving from QA to agentic evaluation. This concretely validates the need for interactive agent-specific safety benchmarks.

- **External safeguard limitation finding (Section 5.4)**: The paper identifies a specific gap — Gemini-1.5's safety filters catch harmful content in `send-sms()` arguments but are ineffective when forwarding private information via `tap()`, since the action argument itself is benign. This is a concrete and novel finding about how current API-level safety mechanisms fail in agentic contexts.

---

## Weaknesses

### Fatal
None.

### Major

- **Confounded o1 vs. GPT-4o comparison (Figure 6, Footnote 4)**: The paper excludes image-based risk tasks from o1's evaluation because the preview version lacks image inputs. However, Table 3 shows that image-based risks are precisely where GPT-4o fails most severely (0/5 in the agentic setting vs. 4/5 in QA). Excluding this hardest category from o1's evaluation systematically inflates o1's harm prevention rate relative to GPT-4o. The paper presents this as direct evidence that "OpenAI-o1 agents demonstrate improved harm prevention rates" due to "enhanced reasoning capability," but the comparison is not on equivalent task subsets. The conclusion drawn is stronger than the design supports. A controlled comparison (e.g., evaluating GPT-4o on the same image-excluded subset) is needed.

### Minor

- **Small evaluation set with no variance reporting**: The main safety evaluation uses 35 high-risk tasks, and sub-category analyses involve as few as 4–12 tasks. No confidence intervals, bootstrap estimates, or significance tests are reported anywhere. At 35 tasks, one task ≈ 2.9 percentage points. Specific claims like GPT-4o's Private Information harm prevention being "0% (basic) and 15% (SCoT)" (a difference of ~1 task), or the headline "25% higher harm prevention with SCoT on average across LLMs," are sensitive to individual task outcomes. The paper does not acknowledge this limitation. While single-run evaluations without variance reporting are common in agent benchmarking (and not in themselves disqualifying), the precision of percentage-point claims should be tempered.

- **SCoT safety-helpfulness tradeoff inadequately addressed**: Gemini-1.5 with SCoT achieves 80% harm prevention on high-risk tasks but also 44% on low-risk tasks (false positive rate). The paper notes this means the agent "unnecessarily avoids risks despite the absence of high risks" (Section 5.2), but does not provide a joint metric or tradeoff analysis. Without a metric that jointly rewards high-risk harm prevention while penalizing low-risk over-refusal, it is unclear whether SCoT is genuinely improving safety or simply shifting models toward indiscriminate refusal.

- **SCoT ablation confined to GPT-4o (Table 1)**: The ablation comparing Basic vs. Safety-guided vs. SCoT prompts is shown only for GPT-4o. The paper claims "integrating SCoT with the CoT technique significantly enhances the safety of LLM agents," but the mechanism (forced safety reasoning output) is only isolated for one model. Whether the same mechanism accounts for the gains in Gemini and Claude is not demonstrated.

### Trivial

- The task count structure (87 total = 36+35 daily + 8+8 injection) is clarified only in Section 5.1, not in the task design section (Section 3.3). Stating this breakdown earlier would improve clarity.

---

## Nice-to-Haves

- A composite metric combining harm prevention on high-risk tasks and goal achievement on low-risk tasks (e.g., per symmetric pair: reward correct refusal AND correct completion) would directly quantify the safety-helpfulness tradeoff and make cross-model comparisons more interpretable.
- Trajectory-level case studies showing an agent generating correct SCoT safety considerations but then ignoring them when making a decision would powerfully illustrate the "considerations are ignored" failure mode mentioned in Section 5.2 and better motivate future work.
- Extending the o1 comparison to the full task suite (e.g., using human-transcribed descriptions for image-based risks, or applying the same exclusion to GPT-4o as a controlled baseline) would make Figure 6 a rigorous result.
- Reporting inter-annotator agreement statistics for risk type labeling (currently deferred to Appendix B.2) in the main paper would strengthen confidence in the taxonomy.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Benchmark scale must be at least 200-300 task pairs" (Harsh Critic)**: Removed as an overly prescriptive standard. ToolEmu (accepted, avg 7.33) had 144 test cases; AgentHarm (accepted, avg 6.75) had 110 tasks. The field norm does not require 200-300 tasks, and the finding that statistical precision is low is already captured under Minor weaknesses.

- **"Evaluator reliability analysis — false-negative analysis needed" (Harsh Critic)**: Partially removed. Rule-based evaluators on structured device state are substantially more reliable than LLM-based evaluators. The critique has merit in principle, but the paper's deterministic evaluation design makes it a minor concern rather than a structural threat to the results.

- **"QA vs. agentic comparison confounders (attention diffusion, context length)" (Harsh Critic)**: Removed as scope creep. The paper's claim is that a gap exists and that it motivates agent-specific benchmarks. Disentangling the precise mechanism (task pressure vs. context length vs. multi-step horizon) is future work, not a requirement for the contribution.

- **"Indirect prompt injection conclusion too strong at 8 tasks" (Harsh Critic)**: The framing "all agents are defenseless" with 0/8 for two models and 1/8 for one is directionally robust even at this scale. The caveat about simplicity of injected prompts is acknowledged by the authors in Section 5.3 and is more a description of a lower bound than a flaw.

- **"Why agents fail to act on detected risks — deeper analysis needed" (Harsh Critic)**: Reasonable as a nice-to-have but the paper does provide qualitative analysis (checking only the most recent text message, Section 5.2). Moving to nice-to-have rather than a weakness.

- **Strength: "Open-source with real Android emulators"** — Kept in strengths as ecological validity, but the open-source commitment is stated as a future intent ("*will* open-source"), which weakens this claim slightly. Not worth elevating further.

---

## Novel Insights

The most genuinely novel insight from this review is the wedge between API-level safety mechanisms and action-level safety in agentic settings: Gemini's content filters catch harmful text in `send-sms()` arguments but are completely blind to privacy leakage enacted through `tap()` on a UI element — because the action argument itself is benign. This reveals a fundamental architectural gap between output-filtering safety mechanisms (designed for chatbot responses) and the requirements of embodied agentic safety, where the harm is in the *consequence* of an action chain rather than in the surface content of any individual model output. This finding has implications beyond mobile devices and motivates a rethinking of what "safety" means in multi-step tool-using agents.

---

## Suggestions

1. **Fix the o1 comparison**: Either re-run GPT-4o on the image-excluded task subset to create a controlled comparison, or report o1's results separately as a preliminary finding rather than a direct comparison against GPT-4o.
2. **Add a joint safety-helpfulness metric per symmetric pair** (e.g., fraction of pairs where the agent correctly completes the low-risk task AND correctly prevents harm in the high-risk task). This resolves the over-refusal ambiguity and makes model comparisons more meaningful.
3. **Acknowledge scale limitations explicitly**: Add a short paragraph noting that differences of 1–2 tasks correspond to 3–6 percentage points, and that the primary value of the current results is directional rather than precise.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison |
|---|---|---|
| GEcwtMk1uA (ToolEmu) | 7.33 | Spotlight accept; uses LLM-emulated (not real) environments, 144 tasks, human-validated failures. Stronger scale and validation than MobileSafetyBench, but less ecologically valid. |
| AC5n7xHuR1 (AgentHarm) | 6.75 | Poster accept; 110+ tasks, diverse harm categories, explicit jailbreak focus. More tasks than MobileSafetyBench, cleaner task design, but less rigorous evaluation scheme. |
| LPG8pPSfQD (DistRL) | 6.75 | Poster accept; mobile device control RL framework — topically adjacent but different contribution type. |
| IIzehISTBe (ST-WebAgentBench) | 4.25 | Reject; similar web-agent safety benchmark concept but shallower evaluation, less rigorous metric design, and limited analysis depth. |
| dQzpP9ziaJ (LongSafetyBench) | 3.5 | Reject; safety benchmark with methodological concerns (multiple-choice format, limited real-world validity). |

**Assessment against anchors:**

MobileSafetyBench clearly exceeds ST-WebAgentBench (4.25) on evaluation rigor, ecological validity, and depth of analysis. It is comparable to AgentHarm (6.75) in terms of findings novelty and contribution value, though slightly weaker on scale. It falls below ToolEmu (7.33) due to smaller scale, the o1 comparison confound, and less rigorous statistical treatment. The paper's genuine strengths — real emulator environment, symmetric task design, prompt injection findings, QA-vs-agentic gap — place it solidly above the borderline anchors. The Major weakness (o1 comparison confound) is real but affects only one subordinate experiment, not the core benchmark contribution.

**Final score: 5.5** — Borderline accept. The core benchmark contribution is solid and the findings are genuinely useful to the community, but the scale of the main evaluation, the o1 comparison confound, and the missing joint safety-helpfulness metric are real concerns that partially undermine the precision of the quantitative claims. This places it slightly below AgentHarm/ToolEmu territory but comfortably above the rejected benchmarks.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>