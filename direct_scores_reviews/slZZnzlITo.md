## Summary

This paper proposes Multimodal Open Set Recognition (MMOSR), a new task extending OSR to multimodal settings. The authors identify "fusion degradation" — a phenomenon where OSR regularization over-compresses fused representations and suppresses modality-specific information — and propose the Multimodal Representation Reactivation Network (MRN), which uses bidirectional cross-attention (mutually enhanced fusion) and a Mixture-of-Experts head (adaptive fusion) to recover richer representations. Experiments on four datasets spanning image-text, audio-visual, and RGB-depth settings show that MRN generally outperforms existing multimodal fusion methods, with gains up to +5.23 OSCR on Flower-102.

---

## Strengths

- **First systematic study of MMOSR across genuinely diverse modality pairings.** The experimental scope covers image-text (Food-101, Flower-102), audio-visual (CREMA-D), and RGB-depth (SUN RGB-D) — a breadth of modality types that most multimodal papers in this space do not achieve, and which meaningfully supports the claim that the challenge is not dataset-specific.

- **MRN as a fusion backbone consistently improves OSR methods across all datasets.** Table 2 shows that ARPL-MRN and CSRR-MRN outperform ARPL/CSRR with every other fusion strategy (ADD, CAT, GQA) on all four datasets, including CREMA-D where standalone MRN does not win. This asymmetry — the backbone always helps even when the standalone method does not dominate — provides a more robust signal of architectural value than any single headline number.

- **Fusion degradation identification is empirically grounded.** Table 1 shows a clear pattern: combining additive fusion with OSR regularization (Fusion-OSR) causes AUROC to drop below either single-modal OSR or plain fusion across 5/10/20-class splits on Food-101. This observation is genuine and non-obvious, even if the mechanism is not formally characterized.

- **Competitive with large pretrained models without pretraining.** Table 3 demonstrates that scratch-trained MRN outperforms zero-shot CLIP and 16-shot CoOp/MaPLe across all class-ratio settings on Food-101, despite having no access to the large-scale pretraining used by those models. Since any unfairness in this comparison favors the pretrained baselines, the MRN result is a stronger empirical claim by construction.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing the most natural baseline: per-modality OSR score ensemble.** The most straightforward approach to multimodal OSR is to run OSR independently on each modality and combine the rejection scores (e.g., by averaging MSP or taking the maximum). Without this baseline, it is impossible to determine whether MRN's gains arise from fusion architecture improvements or simply from aggregating two OSR signals. This is a critical omission for a paper whose central claim is that multimodal-specific fusion design is necessary.

- **No OSR-specific training objective.** Section 4.2.3 shows that MRN is trained with only a standard classification loss and a load-balancing loss. The fusion degradation narrative is about OSR regularization harming representations — yet MRN itself does not apply any OSR regularization at training time. Unknown rejection is then performed via standard MSP thresholding. The method is therefore a better multimodal classifier evaluated with post-hoc OOD scoring, not an OSR method in the usual sense. This mismatch between the problem framing and the actual training objective should be explicitly acknowledged. It is not fatal, but it significantly narrows the methodological contribution relative to what is claimed.

- **Inconsistent performance undermines "consistent superiority" claim.** On CREMA-D (Table 2), standalone MRN achieves 66.78 AUROC and 57.32 OSCR, which is worse than MLA's 67.83 AUROC and 57.50 OSCR. The paper's text in Section 5.2 states MRN "consistently demonstrates exceptional MMOSR performance," which is inaccurate. For a paper proposing a method for MMOSR broadly, this inconsistency requires either a more honest characterization or an analysis of why CREMA-D's audio-visual structure is a distinct failure mode.

- **No variance estimates for marginal gains.** Open-set class splits are randomly sampled. Several reported improvements are very small (SUN RGB-D: +0.37 AUROC, +0.01 OSCR; Food-101: +0.72 AUROC, +1.38 OSCR). Without standard deviations across multiple random splits, there is no basis to distinguish these from split-to-split variance. This is particularly important for ICLR where marginal gains without significance are insufficient support for broad empirical claims.

- **Ablation does not isolate the MoE module.** Table 4 ablates only the cross-attention directions (C1 and C2), with the first row defined as "encoders + adaptive fusion (MoE)." There is no comparison of MoE vs. a single MLP head, or MoE vs. simple concatenation + linear, which would be necessary to evaluate whether the MoE component provides benefit beyond a larger/deeper prediction head. Since adaptive fusion is presented as a co-equal contribution, this omission is significant.

### Minor

- **Threshold selection protocol underspecified.** Section 4.3 states the threshold τ is set to "ensure 95% of the known samples are correctly classified" but does not specify on which data split. If this uses test-known samples, it would contaminate evaluation. The paper should explicitly state that the threshold is tuned on a held-out validation set.

- **Fusion degradation diagnosis is restricted to one dataset and one baseline.** The necessity analysis in Section 3.2 is based entirely on Food-101 with a simple additive Fusion-OSR. While Section 5 broadens the comparison, the motivating failure demonstration would be substantially stronger if it included the stronger multimodal baselines (TMC, MLA) and additional datasets, especially given that the central argument relies on this failure mode being general.

- **No computational cost reporting.** MRN introduces 15 experts with top-4 gating plus bidirectional cross-attention on top of encoders. Parameter counts, FLOPs, or inference time relative to simpler baselines are not reported, making practical trade-off assessment impossible.

- **Cross-attention equation (Eq. 1) is notationaly unclear.** As written, `Softmax(W_Q z1 · z2 W_K / sqrt(d)) (W_V z2)` mixes matrix-vector conventions ambiguously. The shapes of z1 and z2 (pooled vectors vs. token sequences) are never stated, making the exact computation difficult to reproduce for different modality structures (e.g., text token sequences vs. pooled audio embeddings).

### Tiny

- **t-SNE and Grad-CAM visualizations are suggestive but not rigorous.** Figures 6 and 7 are illustrative, but t-SNE can create apparent cluster separation that does not reflect actual decision boundaries, and Grad-CAM examples may be cherry-picked. Quantitative proxies (e.g., intra/inter-class distance ratios, feature effective rank) would support the "reactivation" narrative more convincingly.

- **The ablation asymmetry in cross-attention direction** (C2 > C1 in Table 4) is noted but the explanation that "images serve as queries, leveraging richer visual information" is speculative and not tested. It would be more informative to check whether this direction asymmetry holds across modalities where text vs. image dominance differs.

---

## Nice-to-Haves

- **Pretrained encoders as MRN backbone.** Training from scratch with ResNet34 is a standard setup for controlled comparison, but evaluating MRN with CLIP's pretrained encoders (or similar) would demonstrate practical relevance at modern capability levels and directly answer whether the architectural contributions hold in the pretrained regime.

- **Quantitative measure of fusion degradation.** Adding metrics such as representation effective rank, class separation ratio, or cosine similarity between known clusters before/after OSR training would make the fusion degradation claim rigorous rather than relying on t-SNE plots.

- **Expert specialization visualization.** Showing which experts activate for which modality combinations or class types (e.g., per-expert routing frequency by modality) would validate whether MoE actually captures diverse, complementary representations or simply functions as a wider MLP.

- **Missing-modality robustness evaluation.** Since the practical motivation includes sensor-failure scenarios (robots, unmanned systems), even a brief analysis of MRN behavior when one modality is zeroed out or replaced with noise would inform practical deployment considerations.

- **Extension to more than two modalities.** The paper notes pairwise cross-attention can be extended but does not demonstrate this. An experiment with three modalities (or even a discussion of the quadratic scaling trade-off) would substantiate the generality claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Missing related works (multimodal OOD detection, selective prediction, open vocabulary multimodal recognition).** Per review policy, criticisms about missing citations are removed since we cannot confirm existence from external sources. The related work section covers the directly relevant OSR and multimodal fusion literature adequately.

- **GQA is not a canonical multimodal fusion baseline.** GQA (Ainslie et al., 2023) is described in the paper as an approach for multimodal learning efficiency and is included as a fusion baseline used in their setting. Since the paper cites and uses it, criticism of its relevance as a citation is not warranted.

- **Training all models from scratch unfairly disadvantages strong multimodal baselines.** This comparison is intentionally asymmetric in favor of pretrained baselines (CLIP, CoOp, MaPLe), not against them. Per review rules, comparisons where unfairness benefits the baseline are not legitimate weaknesses — they actually strengthen the claim when MRN still wins. REMOVED.

- **Lack of formal theoretical justification (theorems, proofs) for fusion degradation.** This is an empirical systems paper at ICLR; demanding theoretical proof of an empirically observed phenomenon imposes non-standard rigor requirements for this setting. REMOVED.

- **Requesting confidence intervals for all large-scale benchmark results.** Single-run reporting is the norm for large-scale benchmarks. The variance concern is legitimately kept only for the marginal-gain cases in Table 2. The broad demand for confidence intervals everywhere is REMOVED as a blanket expectation.

- **Style criticisms** (abstract phrasing strength, conclusion tone): REMOVED as pure framing/style issues.

---

## Novel Insights

The most substantive novel observation across the three reviews — one not directly made explicit in the paper itself — is the asymmetric evidence structure in Tables 2–3: standalone MRN is not universally best (it loses on CREMA-D), but MRN-as-backbone (ARPL-MRN, CSRR-MRN) consistently and universally improves over alternative fusion strategies including on CREMA-D. This suggests the paper's true contribution may be more precisely characterized as a robust multimodal feature extractor that reduces sensitivity to OSR regularization, rather than a complete MMOSR system. This reframing actually makes the contribution cleaner: the architecture reactivates suppressed representations regardless of the downstream OSR objective, which is a more defensible and reproducible claim than "MRN is the best MMOSR method."

---

## Suggestions

1. **Add a per-modality OSR score ensemble baseline** (e.g., average MSP from independent unimodal OSR models) as the simplest multimodal OSR competitor. This single baseline would either validate or substantially weaken the necessity of fusion-level design.

2. **Restate the main claim more precisely:** distinguish between (a) MRN as a standalone method and (b) MRN as a fusion backbone for existing OSR objectives. The fusion-backbone framing is more consistently supported and would align better with the asymmetric evidence in Table 2.

3. **Add variance across random class splits.** Run each configuration across at least 5 random known/unknown splits and report mean ± std. Focus this effort on the datasets where gains are smallest (SUN RGB-D, CREMA-D) to determine whether the results are statistically meaningful.

4. **Specify the threshold protocol.** Confirm explicitly that τ is chosen on a held-out validation set of known classes, not on the test distribution, and describe the split construction.

5. **Add one ablation row: MoE replaced by single MLP** with equivalent parameter count. This is the minimum needed to attribute any gain to the expert-diversity mechanism vs. simply more parameters.

6. **Broaden the Section 3.2 motivation** to include at least one non-food dataset and one stronger fusion baseline (MLA or GQA), to show fusion degradation is not specific to Food-101 + additive fusion.

---

## Evaluation on Key Axes

- **Originality**: Moderate. The MMOSR task formulation and fusion degradation observation are genuinely novel framing contributions. The MRN architecture composes known elements (bidirectional cross-attention, sparse MoE) without introducing new mechanisms; the novelty lies in their application to a new problem rather than the components themselves.

- **Importance of research question**: High. Multimodal OSR is a practically critical and underexplored problem. The paper is timely.

- **Claims well-supported**: Partially. The strongest claims (MRN as fusion backbone consistently helps) are well-supported. The "consistent superiority" of standalone MRN is not supported (CREMA-D). The fusion degradation mechanism is plausible and empirically motivated but not rigorously characterized.

- **Soundness of experiments**: Moderate. Four datasets with diverse modalities and many baselines are a genuine strength. Missing the per-modality ensemble baseline, absence of variance estimates, and incomplete MoE ablation are meaningful gaps.

- **Clarity of writing**: Adequate. The high-level idea is well-communicated. Technical under-specification (attention tensor shapes, threshold protocol) limits reproducibility.

- **Value to research community**: Moderate-to-good. Establishing the MMOSR benchmark and demonstrating that naive combination fails provides a useful starting point. The contribution would be considerably higher if the per-modality baseline were included and the evaluation protocol tightened.

- **Contextualized relative to prior work**: Adequate. The paper covers the directly relevant OSR and multimodal fusion literature and does not overclaim novelty relative to them.