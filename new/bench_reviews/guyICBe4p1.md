Now let me run calibration searches to score this paper properly.Now I have enough information to write the final review. Let me synthesize everything.## Summary

The paper investigates the context sensitivity of belief probes in large language models (LLMs). It introduces a framework of four error scores (E1–E4) to systematically characterize how belief probe outputs respond to relevant versus irrelevant in-context premises, proposes Contrast Consistent Reflection (CCR) as a more stable variant of CCS, and conducts a causal intervention experiment showing that moving premise representations along belief directions shifts hypothesis probabilities in the logically expected direction. The central findings are: (1) belief probes are sensitive to relevant context but also to irrelevant contexts to a problematic degree, and (2) belief directions partially causally mediate natural language inference.

---

## Strengths

- **Novel E1–E4 error framework (Table 1, Section 3.3)**: Decomposing context sensitivity into four distinct types of consistency errors—E1 (corrupted premise), E2 (unrelated premise), E3 (conditional belief violation), E4 (marginal belief violation)—is a principled and more precise characterization than prior work, which evaluated probes purely on accuracy. The explicit recognition that E3 and E4 are opposing measures (acknowledged in Section 3.3) demonstrates theoretical care.

- **No-prem probes show premise sensitivity (Figure 2a, Section 4.1)**: Probes trained without any premise in context still respond coherently to premises at evaluation time, supporting the finding that LLMs do not represent prior beliefs independently from contextual beliefs. This directly challenges a natural assumption and is a substantive empirical result.

- **Instruction-tuned vs. pretrained comparison (Figure 3)**: The finding that OLMo-Instruct exhibits more E4 errors in later layers—consistent with instruction-tuning reinforcing the model to treat in-context assertions as true—is interpretively compelling and connects mechanistic findings to a known training difference.

- **Causal intervention experiment (Figure 4, Section 4.2)**: Moving premise representations backward along the belief direction causes entailed hypothesis probabilities to decrease and contradicted hypothesis probabilities to increase. This direction-specific behavior is qualitatively consistent with causal mediation, extending Marks & Tegmark (2023) by evaluating logical consistency of the changes rather than just the magnitude of token-probability shifts.

- **CCR theoretical motivation (Section 3.1, Eq. 2)**: The theoretical argument that CCR avoids CCS's degenerate solution (where θ orthogonal to both x⁺ and x⁻ satisfies p(x⁺) = p(x⁻) = 0.5 but violates the reflection constraint) is mathematically sound and concisely argued.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing control direction in the causal intervention experiment (Section 4.2)**: The paper's second headline contribution is that "belief directions causally mediate natural language inference." The intervention moves premise representations along the belief direction and observes ~10 percentage-point shifts in hypothesis probability. However, there is no control condition testing whether moving the premise along a random or orthogonal direction of the same magnitude produces a different effect. Transformer residual streams propagate perturbations non-specifically; without this control, the result is consistent with "any sufficiently large perturbation to the premise representation influences downstream token probabilities" rather than "the belief direction is specifically the causal mediator." The use of do(·) notation implicitly invokes a structural causal model, which is not warranted by the design. This control is not supplementary—it is necessary for the central causal claim.

- **E1 and E2 may be normalization artifacts; no absolute values reported (Section 3.3, Table 2)**: The error scores E1 and E2 are normalized by the Premise Effect (PE), the absolute probability shift from an affirmed supporting premise. For no-prem probes, PE is small by construction (Figure 2a). When PE is small, even small absolute responses to corrupted or unrelated premises produce normalized scores near or exceeding 1.0. The paper interprets E1 ≈ E2 ≈ 1.0 for no-prem probes as evidence that "contexts which should not affect the truth often still impact the probe outputs"—but no absolute (unnormalized) values are reported anywhere in the main paper. Without these, it is impossible to distinguish a substantive contextual sensitivity effect from noise inflated by a near-zero denominator. This is a significant transparency gap in one of the paper's primary empirical findings.

### Minor

- **Intervention experiment run on Llama2-13b while probing experiments are primarily on Llama2-7b (Sections 4.1–4.2)**: The paper presents probing results primarily for Llama2-7b (Table 2, Figure 2), while the causal intervention is run on Llama2-13b. The paper does not explain this switch, and since the paper itself notes that error scores "show no sign of scaling with model size," this creates an unexplained inconsistency between Experiments 1 and 2.

- **CCS omitted from Table 2 in main paper (Table 2 caption)**: The paper explicitly states "CCS omitted, full table in Appendix B." Given that CCR is proposed as an improvement over CCS, the absence of a head-to-head comparison in the main paper makes it difficult for readers to assess whether CCR represents a genuine improvement or is simply comparable to CCS. This is mitigated by the appendix containing the full table.

- **Layer selection in Table 2 cherry-picks best operating points**: Table 2 reports results for two hand-selected layers per method (best accuracy, best error rank). This obscures layer-by-layer variability that the authors themselves identify as a major finding. Per-layer plots for E1–E4 separately are deferred to the appendix.

- **Intervention magnitude fixed to |θ_mm| without justification**: All interventions use the same magnitude regardless of method or direction geometry. Since this magnitude was calibrated for MMP's direction, it may systematically disadvantage LR and CCR, which have different norms.

### Trivial
None.

---

## Nice-to-Haves

- Report absolute (unnormalized) premise effects alongside normalized E1/E2 scores, at minimum as a supplementary table. This would resolve the normalization artifact concern and strengthen the paper's empirical claims considerably.
- Include a control causal intervention using a random or orthogonal direction of the same magnitude to establish specificity of the belief-direction effect.
- Extend the model range beyond 7B and 13B (Llama2 family only) before drawing conclusions about scaling behavior of error scores. The paper correctly notes this in limitations but states the "no scaling" finding more confidently in the results section than the evidence warrants.
- Per-layer E1–E4 plots in the main paper to support the claim that errors depend strongly on the layer.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"CCR contribution is empirically unsubstantiated"** (Harsh Critic, Section 3.1): The paper provides a mathematically sound theoretical argument for why CCR avoids CCS's degenerate solution. The comparison to CCS exists in Appendix B (stripped by parser). That the argument is theoretical rather than purely empirical does not invalidate it as a contribution; the claim is modest (more stable convergence, similar performance), not that CCR is dramatically better. *Removed: mischaracterizes the contribution's scope.*

- **"Hallucination mitigation connection is underdeveloped"** (Harsh Critic, Introduction): The paper explicitly frames itself as "a first step" and does not claim to deliver a hallucination mitigation system. Evaluating the paper on whether it also does Y (mitigation) when it explicitly scopes to understanding X (context sensitivity of probes) is scope creep. *Removed: outside stated scope.*

- **"SNLI hypothesis-only bias undermines results"** (Harsh Critic, Section 4): The paper acknowledges the Poliak et al. (2018) SNLI annotation artifact explicitly and argues that probes showing premise sensitivity must encode more than the artifact. This is a reasonable, if incomplete, response. *Removed: paper addresses this adequately.*

- **"Marginal beliefs definition is operationally unclear"** (Harsh Critic, Section 3.2): This is a genuine conceptual tension but the paper acknowledges in its Limitations section that fully separating prior, conditional, and marginal belief directions requires data where entailment/truth features can be varied independently—and defers this to future work. The framework is useful even if the mapping from theory to practice is imperfect. *Weakened to nice-to-have / future work.*

- **"The Strength Finder's point about code availability"**: Generic, no specific section/figure citation. *Removed per filter rules.*

---

## Novel Insights

The paper's most genuinely novel observation is the trichotomy-in-practice finding: probes trained without premises (no-prem) still respond coherently to premises at evaluation time, suggesting that LLMs represent prior and contextual beliefs in a shared rather than orthogonal subspace. This has implications beyond consistency evaluation—it suggests that the "prior belief" baseline one might construct by simply removing context from the probe's training set cannot be treated as orthogonal to the contextual direction. The instruction-tuned/pretrained comparison (Figure 3) is a compelling mechanistic trace of this: instruction-tuning specifically amplifies E4 errors (marginal belief violations), consistent with the model learning to treat asserted context as true. Together, these findings provide a concrete mechanistic anchor for why inconsistency hallucinations are hard to fix by post-hoc prompting alone.

---

## Suggestions

1. **Add a control intervention**: Run the same causal intervention along a randomly sampled direction of the same magnitude |θ_mm| as a control. Report whether the entailment/contradiction differential effect is significantly larger for the belief direction. This single addition would substantiate the causal mediation claim.
2. **Report absolute PE and E1/E2 magnitudes**: Add a supplementary table (or small inset in Table 2) showing mean absolute |p(h; q̃⁻) − p(h)| alongside normalized E1/E2 values. This separates the normalization artifact concern from the substantive finding.
3. **Align intervention and probing model**: Either run the probing experiments on Llama2-13b or run the intervention experiment on Llama2-7b, to enable direct comparison.
4. **Soften the no-scaling claim**: In Section 4.1, the statement "error scores show no sign of scaling with model size" is based on a single model family at two sizes (7B, 13B). Rephrase as "we do not observe consistent improvement from 7B to 13B within Llama2" rather than a general scaling finding.

---

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Path | Avg Score | Comparison to paper under review |
|---|---|---|---|
| Linear representations of political perspective | rwqShzb9li | 7.50 | High anchor: cleaner linear representation finding with causal evidence; stronger methodology |
| Linearity of relation decoding in transformers | w7LU2s14kE | 7.33 | High anchor: similar probing of LM representations but stronger causal identification |
| Do LLMs "know" internally when they follow instructions? | qIN5VDdEOr | 6.00 | Medium anchor (accepted poster): similar structure—linear dimension in LLM, intervention; accepted despite generalization limits |
| Entity binding in LMs | zb3b6oKO77 | 5.50 | Medium anchor (accepted poster): causal experiments for context sensitivity; comparable rigor |
| Unifying interpretability and control via intervention | uOrfve3prk | 5.25 | Medium anchor (rejected): interventions as interpretability goal, similar causal framing but rejected |
| Circuit probing for causal variables | sZq3lDDETp | 4.20 | Medium-low anchor (withdrawn): causal analysis in transformers, weaker empirical grounding |
| CALM probing framework | InWaCoIMMN | 3.00 | Low anchor (rejected): probing framework for LLMs but much weaker empirically, older models, no novel findings |

**Scoring rationale**: The paper under review sits between the 5.25–6.00 band. Compared to qIN5VDdEOr (6.0, accepted poster), this paper has a similarly useful representational finding and multi-model evaluation, but qIN5VDdEOr included a practical intervention application that validated the dimension's utility. This paper's causal claim lacks the key control condition, and the normalization transparency gap weakens the E1/E2 finding. Compared to zb3b6oKO77 (5.50, accepted poster), the paper is comparable in scope and analytical depth. Both have genuine contributions and real methodological concerns. The missing causal control is more severe than what sunk papers at 5.25, but the E1-E4 framework is a more systematic contribution than most papers at that level. The paper is clearly above the 3.0 papers (which had weaker empirical foundations, no novel findings, or outdated models) and below the 7.0+ papers (which had cleaner methodology and stronger causal identification). Final score: **5.0** — the paper has genuine merit but both headline contributions are weaker than presented; substantive revision is required before acceptance, and fixing the causal experiment requires new results rather than clarification.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>