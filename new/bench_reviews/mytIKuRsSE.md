Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper introduces RULE, a framework for robust multi-modal entity alignment (MMEA) against Dual-level Noisy Correspondence (DNC) — noise in both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences. RULE estimates reliability via a two-fold principle (uncertainty + consensus), divides pairs into clean/uncertain/noisy subsets with tailored loss treatments, fuses attributes with reliability-weighted modulation, and employs a test-time correspondence reasoning (TTR) module using a 72B-parameter MLLM to uncover latent attribute connections.

## Strengths

- **Well-motivated and novel problem formulation.** The identification that MMEA suffers from noise at both the intra-entity and inter-graph levels is genuinely practical. The examples in Figure 1(a) (e.g., Elvis Tsui's image assigned to Jason Momoa) are concrete and compelling, and Figure 1(b) empirically demonstrates that existing fusion and alignment methods degrade under both types of noise.

- **Principled and effective dually robust loss (DRL).** Separating pairs into $\mathcal{S}_U$ (excluded), $\mathcal{S}_I$ (refined via soft label mixing, Eq. 12), and $\mathcal{S}_C$ (used directly) with tailored treatments is well-justified. The ablation in Table 3 confirms this component is critical — removing DRL causes a 26.6-point drop on Non-name (58.2 → 31.6 H@1), and both uncertainty-only and consensus-only variants partially recover performance.

- **Consistent and substantial empirical gains across benchmarks and noise levels.** On the meaningful Non-name setting (Table 1), RULE achieves 73.8 avg H@1 under inherent DNC vs. 68.6 for the best baseline PMF, and 64.3 under 50% DNC vs. 54.0 for MEAformer — margins of 5.2 and 10.3 points respectively. Figure 3(a) further shows RULE degrades much more slowly with increasing DNC ratios.

- **Reliability distribution visualization supports the mechanism.** Figure 3(b) shows clean and noisy pairs are well-separated by the proposed reliability metric, and Figure 4 confirms uncertainty and consensus effectively partition pairs into the designed subsets.

## Weaknesses

### Fatal
None.

### Major

- **Headline comparisons include a 72B-parameter MLLM at test time that no baseline has access to, and main tables lack TTR-free results.** The TTR module (Section 2.5) uses Qwen2.5-VL-72B-Instruct at inference, providing an asymmetric advantage. While the ablation (Table 3, ICEWS-WIKI 50% DNC) shows TTR contributes ~1.7 H@1 on Non-name and ~3.7 on All-attributes, these ablation figures are only reported for a single dataset and noise level. Critically, on All-attributes, "MLLM Enhance" alone achieves 97.6 vs. the full method's 97.7, meaning the 72B model nearly matches the entire pipeline when entity names are available. The main comparison tables (Tables 1–2) should include TTR-free results so readers can attribute gains to the training-time framework versus the external model. Note: the training-time components likely provide the bulk of the improvement on Non-name (even subtracting ~1.7 points, RULE still significantly outperforms baselines), but this needs to be demonstrated in the main tables, not just a single-dataset ablation.

- **A-A noise injection conflates feature corruption with correspondence errors, undermining clean validation of the "dual-level noisy correspondence" claim.** The paper's central concept is noisy *correspondence* — wrong pairings between entities and attributes or across graphs (e.g., an image swapped to a different entity). Yet the A-A noise injection (Section 3.1) adds Gaussian noise to visual features and random character replacement to text — these are feature-level corruptions, not correspondence errors. A model robust to Gaussian blur is not necessarily robust to having its image swapped with that of a different entity, which is the semantically meaningful noise the paper motivates in the Introduction. While E-E and E-A injections do test actual correspondence errors, and the inherent DNC setting tests real-world noise, the A-A component conflates two distinct failure modes, and the overall DNC evaluation mixes them without disambiguation.

### Minor

- **Circular dependency in using inter-graph reliability to identify intra-entity noise.** Section 2.4 states "for correctly paired entities, the attribute-attribute correspondence is incorrect, iff, the corresponding entity-attribute correspondence is wrongly established" and uses inter-graph reliability $w_i^m$ to weight intra-entity attributes. This logic holds only for correctly paired entities. When the entity-entity pair itself is noisy (the core problem), correctly associated attributes will also receive low $w_i^m$ because they are compared against attributes of the wrong entity. The DRF module would then downweight correct attributes for noisily paired entities. This is partially mitigated by DRL (noisy entity pairs tend to fall into $\mathcal{S}_U$/$\mathcal{S}_I$ and receive reduced loss weight), but the paper does not acknowledge or analyze this interaction.

- **Theorem 1 is trivially true and Assumption 1 is strong and unverified.** Theorem 1 ("low uncertainty does not necessarily imply the highest belief is assigned to the annotated correspondence") is an expected property of any multi-class belief distribution and does not constitute a meaningful theoretical contribution. Assumption 1 (correctly associated attributes always yield $\Delta \geq 0$) underpins the greedy test-time correspondence estimation but has no empirical validation of how often it is violated and with what downstream impact.

- **Ablation only on one dataset/noise level; missing key ablation variant.** Table 3 only reports ablations on ICEWS-WIKI at 50% DNC. A variant with random or uniform reliability weights would clarify whether the principled reliability estimation matters or whether any soft weighting scheme suffices. TTR contribution is also unknown for other datasets/noise levels.

- **No variance reported across runs.** When claims rest on margins of a few points, standard deviations would strengthen confidence in the reported improvements.

### Trivial
None.

## Nice-to-Haves

- Baselines augmented with equivalent MLLM access to isolate the contribution of the RULE framework itself.
- Per-modality error analysis under DNC showing which modalities' noisy correspondences are most damaging and which are most effectively corrected.
- Discussion of computational cost and inference time of TTR with a 72B model, and whether a smaller model variant is feasible.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"50%+ noisy correspondences claim is deferred to Appendix"**: The harsh critic questioned the "over 50% in ICEWS benchmarks" claim because it's in Appendix B. Per the rules, missing appendix content is a parser artifact — the appendix exists in the original submission. Additionally, the claim is referenced in the main text (line 45: "According to the statistics in Appendix B, real-world benchmarks always contain numerous NC (e.g., over 50% in ICEWS benchmarks)"), which is sufficient for the reader to know where to verify it.

- **"Baseline performance too high for 50%+ noise"**: The critic argued that if 50%+ correspondences are noisy, baseline performance should be worse than observed. This is speculative — noise concentrated in easy-to-ignore pairs or in less important modalities could explain high baseline performance on some metrics. Not a verified weakness.

- **"Uncertainty dominated by $\tilde{N}$"**: The critic claimed uncertainty $u_i = \tilde{N}/Q_i$ is dominated by the number of candidates. In practice, $Q_i$ grows with both $\tilde{N}$ and the concentration of evidence, and $u_i$ is driven by how peaked the evidence distribution is, not just $\tilde{N}$. This concern is partially valid but overstated.

- **"Bootstrap problem in pair division"**: The thresholds depend on $\mathcal{S}^{TP}$. While this creates a bootstrapping concern, the $\beta$ parameter provides a floor/ceiling, and early in training, having most pairs classified as uncertain ($\mathcal{S}_U$) is actually conservative and reasonable behavior.

- **"TTR module not formalized / hard to reproduce"**: The prompt structure and MLLM call details are deferred to Appendix F.5 and I as stated in the text (line 193-194). Per the rules, missing appendix content is a parser artifact. The paper does reference the relevant appendix sections.

- **"All-attributes setting is nearly trivial"**: While true that entity names make alignment easy, the paper follows established evaluation protocols in the field (line 281: "Following (Chen et al., 2023a; Huang et al., 2024a; Xu et al., 2023)"). Criticizing the standard evaluation protocol is scope creep.

- **Strength finder's claim that "TTR is a novel and distinctive contribution not present in prior MMEA methods"**: While technically true, this strength is weakened by the fairness concern — the novelty of using a 72B MLLM is less compelling when it creates an asymmetric comparison advantage. Moved to removed.

## Novel Insights

The circular dependency between inter-graph and intra-entity reliability estimation is an underappreciated design tension in dual-noise frameworks: when you need one type of reliability to estimate the other, and vice versa, the estimation quality for both degrades precisely in the high-noise regime where accurate estimation matters most. The paper's pair division into $\mathcal{S}_U$/$\mathcal{S}_I$/$\mathcal{S}_C$ provides an implicit partial solution (noisy entity pairs get reduced loss weight regardless of their intra-entity reliability), but explicitly modeling or analyzing this interaction would strengthen the framework and is a direction worth pursuing in future work on multi-level noisy correspondence.

## Suggestions

- **Report TTR-free results in the main comparison tables (Tables 1–2)** at least for the Non-name setting. This is the single most impactful change that would address the fairness concern and allow proper attribution of gains to the training-time framework.
- **Add an ablation with random/uniform reliability weights** to establish whether the principled reliability estimation in Section 2.2 matters or whether any soft weighting achieves similar gains.
- **Replace A-A feature corruption with actual attribute-attribute correspondence swaps** (e.g., swapping visual attributes between entity pairs across graphs) to cleanly validate the method against the correspondence noise it claims to address.

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Norton (noisy correspondence, video-language) | `/home/wg25r/review_agent/human_reviews/9Cu8MRmhq2.md` | 8.0 | More principled and cleaner evaluation; RULE is below this due to TTR fairness and A-A noise conflation |
| Test-time Adaptation for Cross-modal Retrieval | `/home/wg25r/review_agent/human_reviews/BmG88rONaU.md` | 7.5 | Similar test-time augmentation concept but with cleaner comparison setup; RULE is below |
| Cocoon (uncertainty-aware multi-modal fusion) | `/home/wg25r/review_agent/human_reviews/DKgAFfCs5F.md` | 6.0 | Similar uncertainty-aware fusion topic; RULE is comparable but has the TTR fairness concern Cocoon lacks |
| NeuSymEA (neuro-symbolic entity alignment) | `/home/wg25r/review_agent/human_reviews/NNUiUwQWx6.md` | 5.75 | Entity alignment with novel framework but limited datasets; RULE is stronger due to comprehensive evaluation and consistent gains |
| Align-VL (modality alignment, overconfidence) | `/home/wg25r/review_agent/human_reviews/HhP9bgCugr.md` | 4.75 | Limited novelty and single dataset; RULE is clearly above this |
| Vision-free grammar induction (unfair LLM comparison) | `/home/wg25r/review_agent/human_reviews/63r6HyqyRm.md` | 2.33 | Similar unfair comparison concern but far more extreme (all gains from LLM); RULE's TTR adds only ~1.7 points on the main metric |
| Multiple2Vec (multi-modal entity representation) | `/home/wg25r/review_agent/human_reviews/a4O528mek9.md` | 3.0 | Confusing presentation, unclear methodology; RULE is clearly above |

RULE sits above the medium-rejected anchors (NeuSymEA at 5.75, Align-VL at 4.75) due to its comprehensive evaluation and significant improvements, but below the accepted anchors (Cocoon at 6.0, Norton at 8.0) due to the TTR fairness concern and A-A noise conflation. The closest anchor is Cocoon at 6.0 — RULE has stronger empirical results but Cocoon lacks the fairness concern. I place RULE at 5.5, just below the borderline, because the TTR fairness issue in the main tables is a substantive gap that prevents clean attribution of the method's contribution.

---

**Originality:** The DNC problem formulation is novel and well-motivated. The two-fold reliability estimation (uncertainty + consensus) is a reasonable extension of existing uncertainty quantification methods. The TTR module is less novel (applying an MLLM at test time).

**Importance of research question:** High — noisy correspondences are pervasive in real-world MMEA, and the paper convincingly demonstrates that existing methods are severely affected.

**Claims support:** Partially supported. The training-time framework's contribution is strong (ablation shows DRL is critical), but the headline comparisons are muddied by TTR's asymmetric advantage, and the A-A noise injection does not cleanly test the correspondence noise the paper defines.

**Soundness of experiments:** Generally sound but with gaps — ablation limited to one dataset/noise level, A-A noise conflation, no variance reported, missing TTR-free main-table results.

**Clarity:** Good — the paper is well-organized with clear motivation, formal definitions, and illustrative figures.

**Value to community:** Moderate to high — the DNC problem is important and the proposed framework offers a practical solution, but the TTR fairness concern and A-A noise conflation limit the clarity of the contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>