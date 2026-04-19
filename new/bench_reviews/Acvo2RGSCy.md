Now let me run calibration searches in parallel.Now I have enough information to write the consolidated review.---

## Summary

DeLLMa proposes a four-step decision-theoretic framework for LLM decision making under uncertainty: state enumeration, probabilistic state forecasting (via verbalized probabilities), utility function elicitation (via pairwise Bradley-Terry fitting over sampled state-action pairs), and expected utility maximization. It is evaluated on two real-world domains — agricultural planning (USDA data) and financial stock selection — with 120 problem instances each drawn from 7 items, showing consistent improvements over zero-shot, self-consistency, and chain-of-thought baselines across multiple LLM families (GPT-4, Claude 3, Gemini 1.5).

---

## Strengths

1. **Principled decision-theoretic scaffold with human auditability (§3, Equations 1–2, Figure 3 right):** Mapping LLM inference onto state enumeration → forecasting → utility elicitation → expected utility maximization produces genuinely interpretable intermediate outputs — sampled states, posterior weights, elicited utilities — which prior prompt-based methods cannot provide.

2. **Consistent cross-model improvements (Figure 2 right):** DeLLMa outperforms zero-shot prompting by 15–25 percentage points consistently across GPT-4, Claude 3, and Gemini 1.5, demonstrating the framework is not model-specific.

3. **Well-calibrated state forecasts validated directly (Table 1):** The paper directly measures ECE and NLL of the state forecast distributions, with GPT-4 achieving ECE = 0.062 and NLL = 1.11 — a stronger empirical test of an intermediate component than most prompting papers attempt.

4. **Effective variance reduction via shared state sampling (§3.3):** The trick of sharing sampled states across actions to reduce variance in Monte Carlo utility estimates is a concrete, well-motivated engineering contribution that is properly ablated.

5. **Documented baseline failure mode (§4.1):** The finding that zero-shot, SC, and CoT baselines reliably underperform random guessing on larger action sets is a clear empirical observation that motivates the structured approach.

---

## Weaknesses

### Fatal
None.

### Major

- **The CoT baseline is a custom-designed proxy for DeLLMa, not standard CoT (§4, baselines).** The paper explicitly states: "there is no standard CoT pipeline. Inspired by workflows from decision theory, we create a prompting chain consisting of three steps: (1) ask for unknown factors... (2) ...their possibility of occurrence; (3) ...a final decision. Such a mechanism is similar to the DeLLMa pipeline (see §3) but only consists of prompting." This baseline is, in essence, DeLLMa with the formal machinery removed — a degraded version of the authors' own method rather than an independent prior-art CoT approach. The margin DeLLMa gains over this "CoT" is therefore at least partly a gap between the full method and a strawman it was designed to beat, and the claim of beating "chain-of-thought" as a general technique is not justified.

- **The o1 comparison tests a full multi-call pipeline against a single zero-shot prompt (Table 3, §4.3).** The paper gives o1-preview only the zero-shot prompt used for baselines, while DeLLMa runs 64 samples per action through multi-step pairwise batching and Bradley-Terry fitting. The paper is transparent about this ("outperform o1 with the *zero-shot* prompt"), but the framing — "outperforms o1-preview by a wide margin" — is claimed as evidence that "specialized inference-time reasoning" beats "general-purpose inference-time reasoning." This confounds method quality with inference budget. A valid test would either place o1 as the LLM backbone inside DeLLMa, or give o1 the same structured multi-step prompt DeLLMa uses. As written, the headline comparison does not disentangle these factors.

### Minor

- **Authors included as human evaluators in the preference ranking study (§4.3, Table 4).** The paper states that annotators are "the paper authors and 5 external volunteers." Authors who designed the DeLLMa framework, chose the training prompts, and understand the system's expected behavior cannot be considered unbiased evaluators of whether the LLM's rankings reflect general human preferences. The 67% inter-annotator agreement is used to argue that human-LLM agreement is "on par" with human-human agreement, but this agreement rate could partly reflect annotator noise from the authors' familiarity with the task. Conducting this study with fully blinded external annotators would substantially strengthen it.

- **Independence assumption in state forecasting is unablated (§3.2, Algorithm 1).** The paper acknowledges that independent marginals across latent factors are assumed "for computational simplicity," and the final line of Algorithm 1 explicitly shows the factored product. No analysis is provided of how much probability mass this independence assumption assigns to unrealistic joint states (e.g., a drought state combined with a high-yield state for the same crop), or how the assumption degrades forecast quality. The ablation in Table 2 varies *what* states are used but never tests a jointly correlated distribution against the factored one.

- **No statistical significance testing on accuracy differences (§4.1, §4.2).** All 120 problem instances in each domain are combinations of the same 7 items, so instances are not independent. The accuracy differences between DeLLMa and baselines outside the 6-action condition are often 2–5 percentage points. The paper presents no confidence intervals or significance tests, making it unclear whether small margins constitute genuine performance differences or sampling variance.

### Trivial

- **"Scaling laws" terminology is overstated (§4.3, Figure 3 left).** The monotonic accuracy increase with sample size (4→64) and overlap percentage (0%→75%) is the expected behavior of a Monte Carlo estimator and Bradley-Terry fitting with more data. Calling this a "scaling law" imports connotations from the neural scaling literature that are not earned here.

- **The "up to 40% accuracy improvement" headline (abstract) is from the largest action-set condition only.** Average improvements across all action set sizes are substantially more modest; this selectivity is not flagged in the abstract.

---

## Nice-to-Haves

- Using o1 or a strong reasoning model as the LLM backbone inside the DeLLMa pipeline (rather than as a baseline) would cleanly separate the value of the framework from the value of model capability.
- A sensitivity analysis of the verbalized probability dictionary $\mathcal{V}$ (the mapping from qualitative labels to numerical values) would clarify how load-bearing this design choice is.
- A third domain with genuinely independent decision problems would substantially strengthen the generalization claim.
- Portfolio/multi-action extension, acknowledged as future work, would greatly increase practical relevance.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"December 2023 stock prices could be in GPT-4's training data" (data leakage concern):** The paper explicitly states it uses `gpt-4-1106-preview` with a training cutoff before December 2023 to prevent leakage, and provides 24 months of historical prices as context. Speculating about ambiguous training cutoffs constitutes a reproducibility concern rooted in doubting a cited entity's specification — removed per hard rules.
- **Missing API cost breakdown (Footnote 5):** The "implausibly low" cost argument requires external price data to verify; this is a reproducibility concern about an implementation detail. Removed per hard rules.
- **"Scaling across more than 7 items" as missing experiment:** This is scope creep beyond the paper's stated evaluation design.
- **Strength: "Strong performance versus OpenAI o1 (Table 3)** cited as a core strength": Removed. This directly conflicts with the verified Major weakness that the o1 comparison is structurally asymmetric.

---

## Novel Insights

The paper's most underappreciated finding is that baseline LLMs (including CoT and Self-Consistency) systematically **underperform random guessing** on larger action sets — not merely converge to chance level but go below it. This is concrete evidence that unconstrained LLM prompting has a systematic bias (presumably toward options that "echo" the contextual sentiment) that the structured expected-utility approach explicitly corrects. The variance reduction technique — sharing sampled world-states across all actions rather than sampling independently per action — is a clean engineering insight that transfers directly to any framework that computes Monte Carlo expected utilities for multiple choices.

---

## Suggestions

1. Replace the custom three-step CoT baseline with a genuine few-shot chain-of-thought using canonical CoT examples (or acknowledge explicitly that no standard baseline exists and argue why the custom proxy is the appropriate upper bound).
2. Report the o1 comparison as a supplementary "cost-matching" analysis (not a primary headline) and add an experiment using o1 as the DeLLMa backbone.
3. Recruit fully external annotators (e.g., via crowdsourcing) for the preference ranking study and report inter-annotator agreement separately from author annotations.
4. Add bootstrapped confidence intervals over the 120-instance accuracy estimates.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| BIRD (fAAaT826Vv) | Bayesian inference framework for LLM probabilities | 8, 8, 8 | Accept Oral |
| StructRAG (GhexuBLxbO) | Structured knowledge reconstruction for LLM reasoning | 8, 5, 8, 6 | Accept Poster |
| Hint Marginalization (DzKdjWe59v) | Iterative Monte Carlo reasoning pipeline for LLMs | 5, 6, 6, 6 | Reject |
| DetermLR (Sbi8BdKcob) | Structured LLM logical reasoning pipeline | 5, 5, 5, 3 | Reject |

**Reasoning:** DeLLMa sits above the Hint Marginalization / DetermLR cluster (scored ~5): it covers two real domains rather than one, has a more principled framework, provides calibration evaluation, and shows results across three LLM families. However, it sits clearly below BIRD (8, 8, 8) — which has tighter theoretical grounding and rigorously validated probability estimation — and below StructRAG's core reviewers (~7+): DeLLMa's CoT baseline and o1 comparison are methodologically compromised in ways that overstate two of its three headline claims. Its genuine contribution is the principled decision-theory scaffold that beats zero-shot across models; this is real, but narrower than claimed. The center of the comparable-quality cluster is approximately 5.5, and the paper does not clearly outperform that cluster given its comparison methodology issues.

**Axes summary:** Originality: moderate-good (novel application of decision theory to LLM inference); Research question importance: strong; Claims support: weaker than needed (o1 and CoT comparisons overstated); Experimental soundness: adequate but missing significance tests; Writing clarity: clear and well-organized; Community value: solid, especially the auditable decision-tree outputs.

**Score: 5.5 — Weak Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>