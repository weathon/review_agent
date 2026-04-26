Now let me search for calibration papers.Now I have enough information to write the review. Let me synthesize everything.Now I have sufficient information to write the final review.

---

## Summary

The paper introduces a cosine-similarity-based taxonomy of read-write (RW) functionalities for gated MLP neurons (SwiGLU/GEGLU) in transformer LLMs, classifying neurons based on the geometric relationships among their gate, input, and output weight vectors. Applied to 12 diverse LLMs, the method reveals a universal strengthening-then-weakening pattern by layer. Behavioral experiments on OLMo-7B show that "weakening" neurons have disproportionate impact on output entropy and attribute rate, and a novel conditional ablation analysis reveals that this effect is partially driven by activations with negative gate values — the first documented functional mechanism involving negative Swish activations.

---

## Strengths

- **Cross-model universality of the structural finding.** Figure 1(a) shows median cos(w_in, w_out) is positive in early-middle layers and crosses below zero in late layers for all 12 architecturally diverse models (Llama, OLMo, Gemma, Mistral, Qwen, Yi). This is a clean, replicable empirical regularity obtained from a computationally trivial weight-based method, with principled significance baselines (95% randomness regions via two independent approaches, Section 4.3).

- **Novel negative-gate-value mechanism.** The conditional ablation in Section 6.2 is the paper's most innovative experimental contribution. The finding that case (iii) — negative gate and negative input — drives the entropy-sharpening effect is both non-obvious and mathematically explained. Section 7 confirms negative gate activations are rare in weakening neurons, making their influence doubly surprising. The paper correctly positions this as the first demonstration of a functional role for negative Swish gate values in mechanistic interpretability.

- **Elegant, principled method.** The RW taxonomy follows directly from the SwiGLU formula (Eq. 2): the contribution of each neuron to the residual stream factorizes into Swish(x_gate) · x_in · w_out, making the cosine relationships among w_gate, w_in, w_out a natural lens. The sign preprocessing (Section 3.2) resolves a genuine ambiguity without changing model behavior. The method requires no training data, no forward passes for classification, and is applicable to any gated architecture — a practical advantage.

- **Activation-frequency correlation as independent validation.** Section 7 demonstrates a strong negative correlation (r ≥ −0.71 in all but the last two layers) between cos(w_in, w_out) and activation frequency across OLMo-7B layers, extending Gurnee et al. (2024) to gated architectures and providing data-independent support for the weight-geometric taxonomy.

- **Conditional ablation as a reusable tool.** The four-way conditional ablation (Section 6.2) based on signs of x_gate and x_in is a clean, generalizable analytical technique that goes beyond standard zero/mean ablation and allows attributing behavioral effects to specific activation regimes.

---

## Weaknesses

### Fatal
None.

### Major

- **Behavioral experiments confined to a single model while universality is claimed.** The paper's title and abstract advertise "outsize influence" as a universal property of weakening neurons. Yet all ablation experiments — including the core entropy and attribute-rate results in Figure 3, the conditional ablation in Section 6.2, and the case studies in Section 8 — are performed exclusively on OLMo-7B. The paper explicitly acknowledges this ("to save resources, we focus on a single model"), but the framing throughout the abstract and introduction strongly implies the behavioral claim generalizes. If the entropy-sharpening effect or the negative-gate-value mechanism is idiosyncratic to OLMo-7B — plausible given OLMo-7B's specific SwiGLU implementation and Dolma training corpus — the headline behavioral claim does not follow from the evidence. At minimum, replicating the key Figure 3 ablation on one additional model (e.g., Llama-3.2-3B, used throughout Section 5) would substantially increase the credibility of the behavioral findings. As written, "universal" structural patterns and "outsize influence" are presented in the same breath, but the evidence base for the latter is a single case study.

- **Activation-frequency confound in the ablation comparisons.** Section 7 (Figure 4) establishes that weakening neurons have far higher activation frequency (gate-positive probability) than conditional strengthening neurons — with an almost linear negative r ≥ −0.71. The ablation baseline (same number of random neurons from the same layers) will include many high-sparsity strengthening neurons with near-zero effective contribution at any given token. Under zero-ablation of a dormant neuron, the counterfactual difference approaches zero. The paper acknowledges this concern only briefly: "Note however that activation frequencies do not fully explain their effect, since we found that even their negative gate values are influential (section 6)." But this note does not rule out the possibility that the bulk of the ablation gap in Figure 3(a) (attribute rate) is explained by activation frequency rather than RW class identity. To support the claim that *class membership* drives the disproportionate influence, a frequency-matched baseline (comparing weakening neurons against non-weakening neurons sampled from the same layers *and* same activation-frequency stratum) is needed, or at minimum a scatter of per-neuron ablation effect vs. activation frequency.

### Minor

- **The "surprise" framing of the entropy result is not well-motivated.** Section 6.1 states: "We would expect the opposite: removing information from the residual stream should make it less informative and therefore flatten the output distribution." Weakening neurons, by construction, subtract directions most prominent in the residual stream. If those directions include the logits of plausible alternative tokens, ablating weakening neurons would amplify those alternatives, naturally increasing entropy. This alternative account does not require any special mechanism. The framing of "surprise" is used to motivate the conditional ablation analysis in Section 6.2, but the prior expectation cited is not well-justified. The conditional ablation finding itself (negative gate values drive entropy sharpening) remains novel and valuable regardless of this motivational framing.

- **Two-neuron case study (Section 8) is thin support for general interpretability claims.** The paper presents two neurons (28.4737 and 31.9634) and honestly notes that the weakening neuron "is much harder to interpret." Two neurons cannot establish that weakening neurons form a functionally coherent class. If "harder to interpret" is a systematic property of weakening neurons, it would weaken the paper's broader interpretability narrative. A brief examination of, say, 10-20 neurons per class — even as a summary statistic — would strengthen the claim that the taxonomy captures functionally meaningful categories.

### Trivial

- The paper's Figure 1 (abstract count) inconsistency — the text in Section 5 names 12 models while the abstract caption of Figure 1(a) says "nine larger models" — should be reconciled. This appears to reflect that Figure 1(a) uses only the 9 larger models but all 12 are discussed in Table/text in Section 5.

---

## Nice-to-Haves

- A sensitivity analysis over the classification threshold τ (currently fixed at ±0.5) showing how the fraction of neurons in each class changes as τ varies from 0.3 to 0.7 would establish whether reported proportions are robust to the threshold choice.
- Extension to instruction-tuned or RLHF'd models would be a natural and practically relevant follow-up, though it is outside the current scope.
- A scatter plot of per-neuron activation frequency vs. ablation-induced Δentropy would directly visualize whether the "outsize influence" finding is independent of activation frequency.
- The paper notes that mean-ablation results are similar to zero-ablation but defers them to Appendix F.4. Including at least one mean-ablation curve in the main text alongside the zero-ablation results would allow readers to judge this without consulting the appendix.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic, Section 3.2 preprocessing classification artifact:** The critic argues preprocessing could alter atypical category proportions in Figure 1(b). While technically possible, the main result (Figure 1(a), median cos(w_in, w_out) pattern) is explicitly unaffected by this step (both w_in and w_out get the same sign scalar so the cosine is unchanged). The effect on the minor "atypical" subcategory proportions is a very marginal concern.

- **Harsh Critic, Table 1 formatting ambiguity:** This is a PDF parser artifact per the review rules; the original submission does not have this issue.

- **Harsh Critic, mean-ablation results in main text only:** The paper is clear that mean ablation results are in Appendix F.4 with similar findings — this is a minor presentation preference, not a methodological flaw.

- **Harsh Critic, Figure 3 shows only zero-ablation in main text as "primary figure":** The paper does present mean ablation in the appendix and explicitly reports they are similar. This is not a substantive weakness.

- **Strength Finder, "Code availability":** Generic strength, not specific to this paper's contribution quality.

- **Harsh Critic, "Section 5: orthogonal output classification conservative" concern:** The paper explicitly discusses this (neurons outside the strict threshold often exceed the significance threshold). The paper treats this as a finding, not a flaw.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the combination of two findings: (1) weakening neurons are characterized by high activation frequency (near-universal activation), and (2) despite this, a disproportionate share of their behavioral influence — particularly entropy sharpening — comes from the rare case where gate values are *negative*, which inverts the neuron's role from weakening to effectively strengthening the residual stream direction. This duality — high-frequency activation in the positive-gate regime, but functional importance concentrated in the low-frequency negative-gate regime — is not a prior hypothesis in the mechanistic interpretability literature and may reflect a more general design principle about how transformers handle prediction sharpening near the final layers.

---

## Suggestions

1. **Add a frequency-matched ablation baseline.** Sample a set of non-weakening neurons from the same late layers as the weakening neurons, matched on activation frequency, and compare ablation effects. If the gap persists, it directly demonstrates that RW class membership drives the "outsize influence," not just activation frequency.

2. **Replicate the core Figure 3 ablation on one additional model.** Even qualitative replication on Llama-3.2-3B (already used for all weight-based figures) would substantially strengthen the universality framing of the behavioral findings.

3. **Reframe behavioral claims to clearly scope them to OLMo-7B** unless/until additional replication is provided. E.g., "We show in OLMo-7B that weakening neurons have outsize influence; we hypothesize this generalizes based on the universal structural finding."

4. **Clarify the 9 vs. 12 model discrepancy** between the abstract/captions and Section 5 text.

---

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/AwyxtyMwaG.md` — *Function Vectors in LLMs*, avg 6.00 (6,6,6,6). Closest thematic analog: simple mechanism discovered in LLM internals, demonstrated across multiple models and tasks, clean causal evidence. This paper's cross-model structural evidence is broader (12 models), but behavioral evidence is narrower (1 model vs. multiple tasks in FV).
- `/home/wg25r/review_agent/human_reviews/rLX7Vyyzus.md` — *Systematic Outliers in LLMs*, avg 6.00 (5,6,3,8,8). Cross-model outlier taxonomy with behavioral analysis; similar profile of strong structural finding + somewhat mixed behavioral evidence. 
- `/home/wg25r/review_agent/human_reviews/d63a4AM4hb.md` — *Not All LM Features Are One-Dimensionally Linear*, avg 7.00 (8,6,8,6). Multi-model interpretability paper with more carefully controlled experiments — higher bar than this paper.
- `/home/wg25r/review_agent/human_reviews/6NNA0MxhCH.md` — *Answer, Assemble, Ace*, avg 7.50. High-quality MI paper with multi-model causal evidence — the bar this paper does not quite reach on the behavioral side.
- `/home/wg25r/review_agent/human_reviews/fM1ETm3ssl.md` — *Towards Meta-Models for Automated Interpretability*, avg 3.00. Low-quality MI paper with no concrete evidence — clearly below this paper.
- `/home/wg25r/review_agent/human_reviews/9L9j5bQPIY.md` — *Metanetwork*, avg 2.50. Rejected MI paper with unclear contributions — clearly below this paper.

**Positioning:** The paper is clearly above the low anchors (3.00, 2.50) — it has a concrete, principled method with robust cross-model structural evidence and a novel mechanism. It sits at or slightly below the cluster of accepted MI papers averaging 6.0 (Function Vectors, Systematic Outliers) primarily because the behavioral "outsize influence" claim has an unaddressed activation-frequency confound and is demonstrated on only one model despite being framed as universal. These are major weaknesses by the standards of comparable accepted papers but not fatal ones. The weight-based universality finding and the negative-gate mechanism are genuine contributions. The paper lands at **6.0** — marginally above acceptance threshold, comparable to Function Vectors and Systematic Outliers, which received the same score with similar profiles of clean structural findings and limited behavioral generalization.

**Originality:** Good — the RW taxonomy for gated networks and the negative-gate mechanism are novel.  
**Importance of research question:** Solid — SwiGLU/GEGLU analysis has been underserved in MI.  
**Claims well-supported:** Partially — structural claims are well-supported; behavioral claims are overclaimed relative to single-model evidence.  
**Soundness of experiments:** Adequate for the structural finding; limited for behavioral claims.  
**Clarity of writing:** Good overall; minor inconsistencies in model counts.  
**Value to research community:** The conditional ablation method and negative-gate mechanism are concrete, reusable contributions.

**Decision: Accept (borderline), contingent on authors scoping behavioral claims more carefully and either providing a frequency-matched baseline or acknowledging the confound explicitly.**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>