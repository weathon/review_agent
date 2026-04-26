Now let me search for calibration papers.Now I have sufficient calibration material. Let me write the final review.

---

## Summary

GHPO (Guided Hybrid Policy Optimization) is a difficulty-aware RL framework for LLM training that addresses reward sparsity in GRPO-based RLVR. The core idea is to detect "difficult" queries online (all G rollouts receive zero reward) and switch to guided imitation learning by injecting partial ground-truth solution traces into the prompt for those queries. The paper demonstrates 3–5% average gains over GRPO on six mathematical reasoning benchmarks using two model families.

---

## Strengths

- **Well-motivated problem diagnosis**: The paper provides a concrete, quantified characterization of the reward sparsity problem — 52% of NuminaMath-1.5 problems yield zero reward for Qwen2.5-7B-Instruct, directly supporting the need for adaptive guidance. The mechanistic explanation (zero group rewards → zero advantages → zero gradient) is crisp.
- **Adaptive switching versus static/manual methods**: Table 2 shows that GHPO (avg 0.442) outperforms both GRPO-CL (0.415) and the fixed-hint GRPO-CL-H(0.5) baseline (0.422), providing evidence that the adaptive switching mechanism — not merely providing hints — accounts for the improvement.
- **Two-model-family evaluation**: Testing on both Qwen2.5-Base-7B and Qwen2.5-Math-7B provides some evidence that gains are not specific to a single pretraining regime. Both show consistent improvement over vanilla GRPO.
- **Informative training dynamics analysis**: Figure 4's joint tracking of format reward, accuracy reward, response length, and gradient norm across training provides diagnostic value beyond just reporting final benchmark numbers.
- **Computationally lightweight design**: GHPO uses only the ground-truth solution traces already present in the training data and a simple group-level reward check, adding negligible overhead compared to off-policy approaches requiring auxiliary LLMs.

---

## Weaknesses

### Fatal
None.

### Major

- **Algorithmic description is internally inconsistent and ambiguous.** The objective in Eq. 1 shows rollouts {o_i} sampled from π_{θ_old}(·|**q**), but the IS ratio in the constraint uses **q*** = **q** + ω·h (the hint-augmented prompt) in *both* the numerator and denominator. These two are equal only when **q*** = **q** (non-difficult case). For difficult samples where **q*** ≠ **q**, using π_{θ_old}(o_{i,t}|**q***, o_{i,<t}) in the denominator while the generating distribution is π_{θ_old}(·|**q**) produces an invalid importance weight. The only coherent interpretation is that when difficulty is detected, a *new* set of rollouts is drawn from π_{θ_old}(·|**q***) and used in the update — but the paper never states this explicitly, and the objective's expectation annotation still points to **q**. If re-sampling occurs, the method works correctly (responses come from q*, IS ratio denominator uses q*, advantages are non-zero if hints help). If it does not occur, the "guided imitation" claim is empty — all-zero-reward groups have Â_{i,t} = 0, so the guided update reduces to pure KL regularization. This ambiguity cannot be resolved without a pseudocode or explicit clarification, and in its current form the algorithm description does not accurately convey what the method does.

- **Absence of DAPO and LUFFY baselines.** The paper explicitly positions GHPO against DAPO (dynamic sampling to discard zero-reward problems) and LUFFY (hybrid on-policy RL + off-policy demonstrations) in Related Work, citing them as the most relevant prior methods. Neither appears in any experimental table. The baselines used (vanilla GRPO and GRPO-CL) predate the targeted reward-sparsity literature. Without showing GHPO's advantage over DAPO in particular — which also addresses zero-reward prompts but by filtering rather than guiding — the central claim of superiority over "existing reward-sparsity-aware" methods is unsubstantiated.

### Minor

- **No variance estimates on small benchmarks.** AIME2024 contains 30 problems; the reported improvement 0.122 → 0.163 (Table 2) corresponds to roughly 1–2 problems. OlympiadBench shows a small *regression* for GHPO vs. GRPO (0.396 → 0.389), which the paper covers with "five of the six benchmarks." Without multi-seed runs or at least confidence intervals, the aggregated ~5% gain claimed in the abstract could partially reflect variance on small-N benchmarks. A table reporting the same runs from multiple seeds would resolve this and is standard practice in RLVR evaluation.

- **Multi-stage hint ratio ω schedule entirely deferred to appendix.** The multi-stage guidance mechanism is described as a core component of GHPO, but no formula, schedule, or stage count is given in the main text. Section 3.4 only says "details provided in Appendix B.3." Readers cannot assess how the schedule interacts with difficulty detection or judge its sensitivity from the main paper alone.

- **Qwen2.5-Math-7B comparison is weakened by absent curriculum learning baselines.** Section 4.3 shows Qwen2.5-Math-7B-GRPO vs. GHPO but omits the GRPO-CL and GRPO-CL-H(0.5) baselines present for the base model. The comparison for the math-specialized model thus regresses to the weakest baseline only.

### Trivial

- The abstract claims "approximately 5%" gain but the improvement on MATH-500 is ~0.2% (0.774 → 0.776 in Table 2). The 5% figure is driven largely by GPQA-Diamond and a small-N benchmark; calling it a "mathematics benchmark" average requires at minimum noting GPQA-Diamond's broader scope.

---

## Nice-to-Haves

- An explicit pseudocode block for the full training loop — including whether rollouts are re-sampled after difficulty detection — would make the algorithm reproducible and resolve the IS validity ambiguity.
- A direct ablation: apply the same partial hints at test time (not just training) to verify that the learned skill transfers to hint-free inference, validating Assumption 1 more directly.
- Scaling to 14B/32B models to assess whether reward sparsity (and GHPO's remedy) persists at larger capacity.
- Figure 4's gradient-norm interpretation should acknowledge that smaller norms could indicate reduced update informativeness rather than purely "stability" — both readings should be discussed.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Strength Finder — "consistent gains across all six benchmarks"**: The paper itself acknowledges only five of six benchmarks improve (OlympiadBench shows a small regression in Table 2: 0.396 → 0.389). This strength was weakened, not fully removed, but the unqualified form is inaccurate.

**Harsh Critic — "advantage collapse makes guided imitation inert"**: This is partially valid but only under the literal reading of the equations (no re-sampling). If re-sampling occurs (the sensible interpretation), advantages are non-zero and the method works. The underlying concern about algorithmic clarity is kept as a Major weakness; the "inert guidance" framing is too strong as an absolute claim and is removed.

**Harsh Critic — criticism of gradient norm interpretation as evidence of less-informative updates**: This is a reasonable interpretive nuance but speculative and not backed by evidence from this paper. Retained only as a nice-to-have.

**Harsh Critic — Section 3.5 cold-start ablation demand (N=20 steps)**: The absence of an ablation for a fixed 20-step warm-up is a nitpick unlikely to affect core claims. Demoted to out-of-scope.

**Harsh Critic — Figure 3 "60% difficult" implies model not learning via RL**: The inference that persistent difficulty means the model "is spending most of its training in guided mode" is itself an argument for why GHPO is needed, not a flaw. Removed as a misreading.

---

## Novel Insights

The paper's online difficulty detection criterion (all-zero group rewards as the trigger signal) is notably simple and requires no external oracle or separate difficulty model. This makes it more practical than curriculum-based methods that require offline dataset partitioning. The empirical evidence in Figure 3 — showing ~60% of batches require guidance even in late training — is a striking observation that challenges the common assumption that models "grow into" harder training data over time. This persistence of difficulty has implications beyond GHPO: it suggests that reward-sparsity-aware methods may need to remain active throughout training, not just in early stages.

---

## Calibration

| Paper | Avg Score | Comparison |
|---|---|---|
| Auto-CEI (`3ogIALgghF`) | 7.0 (Accept) | Curriculum expert iteration for LLM reasoning; clearer algorithmic description, broader task evaluation, comparable scope. GHPO has similar motivation but weaker experimental rigor and unclear algorithm. |
| mmSmQ0gNyZ (Contrastive Curriculum) | 4.0 (Reject) | Curriculum learning for LLM alignment; rejected for lack of novel contribution and scattered focus. GHPO has more algorithmic novelty but comparable evaluation weaknesses. |
| DQO (`k2q0rUX2lx`) | 3.5 (Reject) | RL for LLM reasoning with algorithmic ambiguity and unfair comparisons. GHPO's missing baselines and IS ambiguity parallel this, but GHPO's empirical consistency is stronger. |
| WizardMath (`mMPMHWOdOy`) | 8.0 (Accept Oral) | Strong math LLM training; comprehensive experiments, clear algorithm, fair comparisons. GHPO falls substantially below this bar on experimental rigor. |

GHPO's quality places it between the Auto-CEI paper (7.0) and the curriculum curriculum paper (4.0). The algorithmic ambiguity and the missing DAPO/LUFFY comparisons are genuine major weaknesses that would weigh against acceptance at a top venue. The paper is closer to the 4.0–4.5 range than to 6–7, primarily because the two major weaknesses (IS consistency and missing contemporaneous baselines) are both things a reviewer should require to be fixed before acceptance. The empirical gains are modest on most benchmarks, and the one result that drives the headline "5%" number (GPQA-Diamond improvement) is on a benchmark that is not primarily mathematical.

Final score: **4.5** — the paper has a real, practical idea and consistent (if modest) improvements, but the incoherent IS formulation and absence of comparison to the two most relevant contemporary baselines prevent acceptance in current form.

## Score and Decision

**Originality**: Moderate. The idea of injecting hints when reward is sparse is natural, and similar ideas appear in LUFFY and curriculum work. The online detection trigger is clean but not deeply novel.

**Importance**: The research question (reward sparsity in RLVR) is timely and important.

**Claims supported**: Partially. Consistent gains over vanilla GRPO are shown, but the headline claim of outperforming "strong" reward-sparsity-aware baselines is unsupported since DAPO and LUFFY are not evaluated.

**Soundness of experiments**: Weak for small benchmarks without variance estimates; adequate for MATH-500 and OlympiadBench trends.

**Clarity of writing**: Adequate in most places; the algorithm section is unclear on the re-sampling question, which is the core mechanism.

**Value to community**: Positive if the implementation is clarified and the baseline comparisons are added.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>