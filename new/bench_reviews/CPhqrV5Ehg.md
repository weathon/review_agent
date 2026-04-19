Now let me run calibration searches to anchor the score.Now I have sufficient information to write the consolidated review. Let me synthesize everything.

---

## Summary

This paper proposes Autoregressive Reward Models (ARM), a low-rank parametrization of reward models for decoding-time controlled text generation. The authors reformulate RAD's (Reward Augmented Decoding) training objective as a matrix completion problem, empirically observe that the learned reward matrix is low-rank (rank ~10² vs. vocabulary size 50,257), and propose a bilinear factorization that reduces per-token decoding cost from O(k) forward passes (RAD) to O(1). Experiments on detoxification and sentiment control tasks show that ARM, especially when distilled from RAD, matches RAD's quality while being substantially faster.

---

## Strengths

- **Matrix completion reformulation (§3.1.1, Eq. 5)**: The reframing of RAD's training objective as approximating an incomplete reward matrix is analytically clean and provides a principled motivation for why low-rank structure might be sufficient. It also clarifies what RAD implicitly optimizes.

- **Dueling decomposition with well-motivated regularization (Eq. 6–7, 11)**: The decomposition into a prefix-level baseline and a token-level marginal reward has a natural interpretation: the model can "abstain" by learning small Δr̂, which is a well-motivated inductive bias. Figure 5 validates that regularization both lowers the rank of R̂_ARM and measurably improves fluency.

- **Real efficiency gains demonstrated on actual hardware (Figure 6, Table 1)**: ARM's latency stays at ~0.001 s/token while RAD scales linearly to ~0.010 s/token at top-k=80. Using wall-clock time rather than FLOP counts is the right measurement for practitioners.

- **Experiments extend beyond GPT-2 to LLaMA-2 (7B/13B)**: The inclusion of larger open-weight models (§5.1, Figure 14 in appendix) adds evidence that the approach is not restricted to toy-scale experiments.

- **Open-weight toxicity classifier cross-check (Appendix F.3.1)**: Using a secondary open-weight classifier to corroborate Perspective API scores is good methodological hygiene given the time-variability of closed-source APIs.

---

## Weaknesses

### Fatal
None.

### Major

- **The quality comparison is primarily supported by a circular distillation result.** The headline claim that ARM "performs on par with the more flexible RAD parametrization" rests primarily on ARM-distilled (Eq. 10), which is explicitly trained to mimic the RAD teacher. A distilled model matching its teacher is expected by construction. The more informative from-scratch comparison ("ARM resp. only") consistently shows slightly worse fluency on both tasks (Figure 3: "slightly worse fluency w.r.t. average perplexity"; Figure 4: "lags slightly behind"). The magnitude of this gap is never quantified with point estimates or significance — only visual Pareto curves are provided, making it impossible to determine if the gap is negligible or meaningful. The abstract and conclusion should either be softened (honest claim: "distilled ARM matches RAD quality at k× lower cost") or the from-scratch gap should be quantified to justify the stronger claim.

### Minor

- **The low-rank motivation has an acknowledged but unresolved confound.** The paper itself notes in §3.1.3: "the incompleteness of PΩ(R) makes it easier for a reward model to learn a low-rank approximation," and proves that data sparsity alone guarantees a rank-1 solution compatible with training observations. This means the empirically observed low-rank structure of RAD (Figure 1) could be an artifact of sparse supervision rather than an intrinsic property of the reward function. The paper argues that the data's minimal rank is also low (Appendix B.2), which would resolve the confound, but this argument is deferred to the appendix. The practical empirical results are not invalidated, but the theoretical narrative is weakened if the low-rank structure is primarily a data-sparsity artifact rather than an inductive property of the task.

- **The Han et al. (2024) direct empirical contradiction is not investigated.** §4 notes that Han et al. (2024) observe value-function (RAD-style) outperforming Q-function (ARM-style), which "disagrees with our work," and devotes only two sentences to it. This is a direct empirical contradiction from closely related work; the paper should at minimum hypothesize and discuss what variable (task type, dataset scale, model size) determines which parametrization wins, even if a full empirical reconciliation is not possible.

- **Only two simple binary attribute tasks.** Both detoxification and sentiment are simple binary attribute control tasks. It is not demonstrated that the low-rank assumption holds for more compositional or fine-grained control objectives. A claim about general controlled generation efficiency would benefit from at least one additional task (e.g., style transfer with content preservation, or topic control).

### Trivial

- Figure 1 caption reports d=768 but the main text also mentions d=764 in one place — minor inconsistency.
- The paper claims "scales poorly" for RAD without specifying the regime: at the experimental setting (k=20), RAD is ~3× slower than ARM, not ~10×. The ~10× applies at k=80. The framing in the introduction slightly overstates this.

---

## Nice-to-Haves

- An ablation comparing ARM against a pure DExperts/GeDi-style architecture (using H·E without the learned W matrix) would clarify whether the bilinear W term is driving quality or whether ARM's gains come from the training objective alone.
- Qualitative generation examples comparing ARM and RAD at matched β values would help verify whether ARM's slightly worse fluency corresponds to perceptible quality differences.
- The distilled ARM outperforming RAD on sentiment (Figure 4) is a counterintuitive result: a student model outperforming its teacher. The single-sentence explanation (deterministic distillation targets reduce noise) is plausible but deserves more investigation or at least a sensitivity test.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Rank estimation SVD threshold unspecified.** The paper explicitly follows Finlayson et al. (2024) and defers the threshold details to Appendix C.4 — a standard reference methodology. The appendix was stripped by the parser, not omitted by authors. Removed per the "missing appendix" rule.

- **Harsh Critic: Computational cost of N·|V| RAD calls for rank estimation is unreported.** This is a methodological detail that does not affect the paper's core claims. Removed as a trivial reproducibility nitpick.

- **Harsh Critic: 10× efficiency framing.** Partially kept (as trivial weakness above) because it is a real mild overstatement in the introduction, but framed gently.

- **Harsh Critic: Freezing input/output embeddings is unablated.** The paper states the rationale ("we hope that, this way, the reward model generalizes better to unseen tokens") and this is a supporting design choice, not a core claim. Not ablating every design choice is normal; moved to nice-to-haves territory.

- **Harsh Critic: No confidence intervals or significance testing.** Single-run evaluation without confidence intervals is the norm in controlled generation evaluation papers following Deng & Raffel (2023) and Liu et al. (2021). Moved to nice-to-have per community standards.

- **Strength Finder: "Distillation yields slightly better models than training from data" listed as a strength.** Conflicts with a verified Major weakness (this comparison is circular) and lacks independent validation. Removed per the "weakness wins when strength and weakness disagree" rule.

---

## Novel Insights

The matrix completion reframing of RAD's training objective is a useful analytical lens that the field has not widely applied. The specific observation that data sparsity alone guarantees low-rank-compatible solutions (rank-1 when all prefixes appear once) is a clean, non-obvious theoretical result that could inform future work on reward model design and the general efficiency/expressivity tradeoff in token-level scoring models. The interplay between regularization, output rank, and generation fluency (Figure 5) is also a practically actionable insight for practitioners designing reward models.

---

## Suggestions

1. **Quantify the ARM (resp. only) vs. RAD quality gap** with a table of point estimates at fixed β values. Even without p-values, concrete numbers (e.g., "ARM resp. only achieves toxicity of 0.14 vs. RAD's 0.12 at perplexity 30") would let readers judge whether the gap is practically meaningful.
2. **Address the Han et al. discrepancy more substantively** — propose and discuss at least one testable hypothesis for when Q-function vs. value-function parametrization wins.
3. **Revise the abstract** to accurately reflect the asymmetry: "ARM distilled from RAD matches RAD quality; ARM trained from scratch is slightly but measurably inferior in fluency."
4. **Move the minimal-rank argument** (Appendix B.2) into the main text if space allows, since it is the key response to the data-sparsity confound that motivates the entire paper.

---

## Score and Decision

**Calibration:**
- *SASA* (jY5oml9fe9): Controlled decoding for detoxification without external models; scores 6,6,6,6 → Accept Poster. Similar topic, similar experimental scale, comparable analytical depth.
- *Collab* (7ohlQUbTpp): Controlled decoding via mixture of agents; scores 5,6,6,8 → Accept Poster. Similar application domain; has stronger theoretical results but weaker presentation.
- *LM Decoding as Direct Metrics Optimization* (488A64eOf6): Decoding as optimization; scores 8,5,6,6 → Accept Poster. More theoretically novel; serves as a high-end anchor.
- *LWMS4pk2vK* (Palu KV-cache via low-rank): scores 5,6,6,6 → Accept Poster. Low-rank efficiency contribution; similar scope and contribution level.

**Assessment relative to anchors:** This paper is squarely in the same quality band as SASA and Palu — a solid, incrementally novel efficiency contribution to a well-defined problem with clean experiments and honest reporting. It is modestly below the high-end anchor (488A64eOf6) due to its narrower experimental scope and the unresolved distillation circularity. The Major weakness (distillation comparison) prevents a 7, but the clean reformulation, solid empirical results across two tasks and two model families, and well-designed ablation study justify acceptance at the poster level, putting this at **6.0**.

**Axes summary:**
- *Originality*: Moderate — the matrix completion framing and bilinear decomposition are non-trivial, but the core efficiency idea (replace k forward passes with 1) is straightforward once the low-rank observation is made.
- *Importance*: Moderate — decoding efficiency for controlled generation is a genuine practical concern.
- *Claim support*: Mixed — efficiency claims are well-supported; quality parity claims are oversimplified (distillation circular, from-scratch gap unquantified).
- *Experimental soundness*: Good — follows established evaluation protocols, includes multiple models, ablation is informative.
- *Clarity*: Good — paper is well-organized and honest about limitations.
- *Value to community*: Moderate-to-good — provides a cleaner, faster alternative to RAD with a useful analytical perspective.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>