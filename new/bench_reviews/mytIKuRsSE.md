Now I have all the information needed to write the final review. Let me compile my findings.

## Summary

The paper identifies and addresses Dual-level Noisy Correspondence (DNC) in Multi-Modal Entity Alignment (MMEA), where both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences are noisy. The proposed RULE framework estimates correspondence reliability via a two-fold principle (uncertainty via evidential deep learning / Dempster-Shafer theory, and consensus via similarity to estimated correct matches), then uses this reliability to robustify both inter-graph alignment (Dually Robust Learning, DRL) and intra-entity attribute fusion (Dually Robust Fusion, DRF). A test-time correspondence reasoning module (TTR) leveraging a 72B-parameter MLLM with Chain-of-Thought reasoning provides additional gains during inference.

## Strengths

- **Well-identified and empirically grounded problem.** The paper convincingly demonstrates that real-world MMEA benchmarks contain substantial inherent noise (e.g., >50% in ICEWS benchmarks per Appendix B), and that existing methods degrade significantly under noisy correspondence (Tables 1–2, Fig. 1(b)). The "Inherent DNC" setting shows improvements even without injected noise, confirming the practical relevance of the problem.

- **Principled two-fold reliability estimation.** The combination of uncertainty (via Dempster-Shafer Theory, Eq. 3) and consensus (Eq. 5) is theoretically motivated by Theorem 1, which proves that low uncertainty alone does not guarantee correct correspondence identification. The ablation in Table 3 confirms both principles are complementary: "Only Unc." (53.5 H@1) and "Only Cons." (48.3 H@1) each underperform their combination (58.2 H@1) on ICEWS-WIKI 50% DNC Non-name.

- **Strong and consistent empirical improvements from the training-time components.** Even without TTR, RULE (56.5 H@1 on ICEWS-WIKI 50% DNC Non-name) outperforms the best baseline HHEA (43.9) by a large margin. The DRL component is critical: removing it drops H@1 from 58.2→31.6 (Non-name) and 97.7→82.3 (All-attributes). The improvement is consistent across all five benchmarks and all three noise settings.

- **Effective reliability visualization.** Fig. 3(b) shows clean and noisy pairs are well-separated by the reliability measure; Fig. 5 confirms that correctly associated attributes receive high reliability scores while noisy attributes are suppressed during fusion.

- **Reproducibility.** Code is publicly available; key hyperparameters (λ=1e-4, β=0.3, τ=0.07, γ=0.5) are fixed across all experiments, and the same CLIP backbone is used for all methods.

## Weaknesses

### Fatal
None.

### Major

- **The TTR module inflates headline comparisons by leveraging a 72B-parameter MLLM unavailable to baselines.** Tables 1–2 include TTR in the reported results, but no baseline has access to a comparable test-time model. The ablation (Table 3) reveals a telling pattern: in the All-attributes setting (where entity names are available), "MLLM Enhance" alone achieves 97.6 vs. "w/o TTR" at 94.0 (+3.6 H@1), while in the Non-name setting, "MLLM Enhance" gives only 56.6 vs. 56.5 (+0.1). This strongly suggests the MLLM's value comes primarily from reading entity names rather than from the proposed CoT reasoning design uncovering "underlying attribute-attribute connections" as claimed. The paper does not report inference latency, GPU memory, or cost for TTR, making it impossible to assess practical value. While the core DRL/DRF contribution stands without TTR, the headline tables conflate algorithmic innovation with raw model scale, obscuring the actual contribution.

- **No noise-type-isolating experiments prevent attribution of robustness to specific components.** The noise injection simultaneously corrupts entity-entity, entity-attribute, and attribute-attribute correspondences (Section 3.1), but no experiment isolates a single noise type. The ablation in Table 3 removes *modules* (DRL, DRF, TTR) but does not test under *single-type noise*. Without this, it is impossible to determine whether DRL primarily addresses inter-graph noise, DRF primarily addresses intra-entity noise, or whether both are needed for each type. The claim that RULE handles "dual-level" noise cannot be decomposed into verifiable sub-claims about each level.

### Minor

- **Assumption 1 (Δ≥0 for correct attributes, Δ<0 for irrelevant ones) is strong and not empirically validated.** A generic but correctly associated attribute (e.g., a common image shared by many entities) might not improve the value function, violating the assumption. No empirical measurement of false positive/negative rates for this criterion is provided, even though such validation would be straightforward on a held-out clean subset.

- **No CoT reasoning examples or qualitative analysis of TTR outputs.** The paper claims CoT "enables the MLLM to leverage prior results and detailed steps for reasoning" (Section 2.5) but provides zero examples of what the CoT outputs look like, whether they are semantically meaningful, or how they differ from vanilla prompting. This makes it difficult to evaluate the TTR contribution beyond the numerical improvement.

- **The formal definition makes attribute-attribute correspondence derivative of entity-entity and entity-attribute correspondence** (Section 2.1: $y^m_{ij}=1$ iff $h^m_i=1$ & $\tilde{h}^m_j=1$ & $y_{ij}=1$), while the noise injection creates independent attribute-attribute perturbation (Gaussian noise for visual, character replacement for textual). This creates a slight inconsistency between the theoretical framing ("dual-level" with three noise types) and the formal definition (where attribute-attribute noise is derivative of the other two). The framing of three independent noise types is somewhat misleading given the definition.

- **The circular dependency between reliability estimation and representation learning is not discussed.** Both uncertainty (Eq. 2–3) and consensus (Eq. 5) depend on current entity representations $z_i$, which are trained using reliability-weighted losses (Eqs. 11, 14). The paper does not discuss initialization, warmup, or any mechanism to break potential error reinforcement. While this bootstrapping pattern is common in noisy label learning, acknowledging it and describing how it is handled would strengthen the methodological description.

### Trivial
None.

## Nice-to-Haves

- Report results (or at least ablation) with a smaller MLLM (e.g., 7B) for the TTR module to isolate the effect of model scale vs. CoT reasoning design.
- Provide single-noise-type experiments to decompose which components address which noise type.
- Report inference time and GPU memory with and without TTR.
- Validate Assumption 1 empirically on a held-out clean subset.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The paper claims DNC is 'new' but it extends existing ideas."** The paper positions DNC as a new problem formulation for MMEA specifically, not as a wholly new concept. The contribution is extending noisy correspondence to the multi-level structure of MMEA, which is valid. The framing is standard for problem-driven papers.

- **"DRF is just weighted concatenation, overclaimed as 'robust attribute fusion'."** The claim is about achieving robustness through reliability-weighted fusion, not about sophisticated interaction. Downweighting noisy attributes by their estimated reliability is precisely the mechanism for robustness, and the ablation confirms its effectiveness (50.4→58.2 without vs. with DRF). This is not overclaiming.

- **"Hyperparameters fixed across all experiments is suspicious."** This is a strength (reproducibility, generalizability) rather than a weakness. The sensitivity analysis is referenced in Appendix G.10.

- **"Baselines' numbers differ from original papers due to backbone change."** Using the same backbone for all methods is good practice for fair comparison. The paper explicitly states this choice.

- **"Missing comparison with noise-robust baselines adapted to MMEA."** The paper compares against seven state-of-the-art MMEA methods under noisy conditions. Demanding adaptation of methods from other domains is a nice-to-have, not a requirement.

- **"The Dirichlet integral implementation details are missing."** This is standard ICLR practice to defer computation details to the appendix. The paper cites relevant prior work for the Dirichlet loss formulation.

- **"Circular threshold in pair division uses noisy labels."** This is inherent to the problem of identifying clean samples from noisy labels and is a standard approach. The threshold β acts as a safeguard.

## Novel Insights

The ablation data reveals a striking asymmetry: the MLLM's contribution is almost entirely dependent on entity name availability (3.6 H@1 gain in All-attributes vs. 0.1 in Non-name), yet the full TTR design (combining MLLM scores with prior similarity) provides meaningful gains in both settings (1.7 in Non-name). This suggests that the value of TTR lies not in the MLLM's "deep reasoning" per se, but in the complementary combination of MLLM outputs with the learned similarity space—a design insight that the paper does not highlight and that could have been more explicitly analyzed.

## Suggestions

- Report headline results (Tables 1–2) both with and without TTR, or at minimum clearly flag in the table captions that TTR uses a 72B inference-time model. This would cleanly separate the training-time algorithmic contribution from the test-time compute contribution.
- Add a single-noise-type ablation (e.g., only entity-entity NC, only entity-attribute NC) to decompose which modules address which noise type, directly supporting the "dual-level" claim.

## Calibration

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| CorreGen (Noisy correspondence in multi-view clustering) | /home/wg25r/review_agent/human_reviews_2026/a4S1nQay3b.md | 7.0 (Oral) | Similar topic (noisy correspondence), principled framework, strong results without fairness concerns about inference-time models. RULE is weaker due to TTR fairness concern and missing decomposition, but has comparable problem formulation quality. |
| MASK (Multimodal aligned semantic knowledge) | /home/wg25r/review_agent/human_reviews_2026/d3CISVVO6v.md | 6.67 (Oral) | Similar area (multimodal alignment), concerns about OOD overestimation. Comparable strength level; RULE has larger empirical margins but TTR concern. |
| DiffNCL (Diffusion-based noisy correspondence learning) | /home/wg25r/review_agent/human_reviews_2026/6xQfjJxija.md | 5.0 (Reject) | Same problem area (noisy correspondence), flagged for overclaiming and missing baselines. RULE is clearly stronger—larger improvements, more comprehensive evaluation, more baselines. |
| ALMEA (Active learning for MMEA) | /home/wg25r/review_agent/human_reviews_2026/iitxXWqODX.md | 5.0 (Reject) | Same task (MMEA), limited baselines and overclaimed robustness. RULE is significantly stronger in problem formulation, evaluation breadth, and empirical gains. |
| Subjective NN (Bayesian + subjective logic) | /home/wg25r/review_agent/human_reviews_2026/RyQ25bGKDs.md | 2.0 (Reject) | Weak motivation and experiments. RULE is far above this level. |
| Meta-learning for noisy LLM alignment | /home/wg25r/review_agent/human_reviews_2026/oIAUP1K5Dq.md | 5.5 (Poster) | Meta-learning for noisy preference data. Comparable methodological depth but different domain. |

RULE's core contribution (DRL+DRF) is substantial and well-validated, clearly above the medium-scoring papers (ALMEA, DiffNCL at 5.0). However, the TTR module inflates headline comparisons with a 72B model no baseline can match, and the missing noise-type decomposition prevents full verification of the "dual-level" claim. This places it below CorreGen (7.0) which has comparable depth without similar fairness concerns. A score of 6.0 reflects the solid training-time contribution tempered by the TTR fairness issue and incomplete decomposition analysis.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>