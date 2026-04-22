Now I have all the information I need. Let me write the final consolidated review.

## Summary

The paper proposes Influence-Guided Diffusion (IGD), a training-free guided sampling framework for dataset distillation that uses trajectory influence functions to steer diffusion models toward generating training-effective data. By connecting the dataset distillation objective to trajectory influence (which approximates how training data impacts validation loss), the authors derive gradient-based guidance signals that promote both influence (training effectiveness) and diversity, achieving state-of-the-art results on ImageNet-1K (60.3% at IPC=50) without retraining diffusion models.

## Strengths

- **Novel conceptual contribution: using influence functions as diffusion guidance signals.** Connecting trajectory influence (which quantifies training impact) to guided diffusion generation is genuinely novel—this is the first work to use influence functions as a guidance mechanism for controlled diffusion, converting an abstract "training-effectiveness" condition into a computable gradient objective (Eq. 6–7). This is fundamentally different from prior guided diffusion that relies on explicit content specifications like class labels or text prompts.

- **Strong empirical results consistently outperforming baselines.** DiT-IGD (pretrained DiT + IGD, no fine-tuning) outperforms Minimax (which requires expensive fine-tuning) in most ImageNette/ImageWoof settings (e.g., 61.9% vs. 58.2% on ImageNette IPC=10, ConvNet-6 in Table 1). Minimax-IGD achieves 60.3% at IPC=50 on ImageNet-1K (Table 2), surpassing the previous SOTA RDED (56.5%) by 4.0 percentage points. This demonstrates that guided sampling is both more effective and more efficient than diffusion model retraining for this task.

- **Comprehensive cross-architecture generalization analysis.** Table 3 shows consistent improvements over RDED across four unseen architectures at IPC=50—ResNet-101 (+5.6%), MobileNet-V2 (+5.2%), EfficientNet-B0 (+4.2%), Swin Transformer (+5.0%) for Minimax-IGD. Table 4 further demonstrates robustness to surrogate architecture choice (ConvNet-6, ResNetAP-10, ResNet-18), with differences typically within 1–2%.

- **Clean ablation design demonstrating complementarity.** Table 5 isolates influence and deviation guidance contributions, showing they address distinct issues and their combination (81.0%/84.4% for DiT-IGD) is consistently better than either alone.

- **Efficient checkpoint selection via gradient similarity.** The proposed adaptive checkpoint filtering retains only 4 checkpoints at threshold 0.7 yet outperforms regular-interval selection with 10 checkpoints (82.0% vs. 81.1%, Table 6), providing both efficiency and effectiveness gains.

## Weaknesses

### Fatal
None.

### Major

- **The theoretical justification for replacing θ_e^S with θ_e^T in Eq. 6→7 has a circularity problem.** Section 3.2 states that replacing synthetic-data checkpoints with real-data checkpoints is "an optimally equivalent target" because "these two targets converge to the same optimal solution when z can provide the same training dynamics as T_c." This equivalence condition (matching training dynamics at all epochs) is essentially the goal of dataset distillation itself—achieving it means the method has already succeeded. At any non-optimal point (where the method actually operates), the substitution introduces uncharacterized approximation error. The paper then further relaxes to full-dataset checkpoints θ_e^T (rather than per-class θ_e^{T_c}) and switches from dot products to cosine similarity. Each step is motivated heuristically, and the cumulative deviation from the original objective is unanalyzed. The theory thus provides post-hoc motivation rather than a principled derivation of why IGD works—this is a meaningful gap between the claimed "correlation" and what is actually proven.

- **The relative importance of influence vs. diversity guidance varies by backbone in unexplained ways, undermining the narrative that influence is the primary mechanism.** Table 5 shows that for DiT at IPC=50, deviation guidance alone (78.2%) outperforms influence guidance alone (76.5%), while for Minimax, influence guidance (81.5%) substantially outperforms deviation (78.5%). The paper frames influence guidance as the core contribution, but the data suggests the mechanism of action is backbone-dependent. Section 4.4 notes this descriptively but does not explain why influence is more critical for Minimax (which already addresses diversity via fine-tuning) while diversity is more critical for DiT (which lacks diversity by default). Without understanding when and why influence guidance helps versus when diversity alone suffices, the paper's "understanding" contribution is incomplete.

### Minor

- **Key hyperparameters are dataset-dependent with no general principles for selection.** The guided range [A,B], influence scale k, deviation scale γ_t, and checkpoint similarity threshold are all tuned per-dataset (Appendix A.10). Figure 2c shows performance collapses when k≥10 for entire guidance. The early-stage guidance range [30,45] out of 50 steps is justified only by "empirical observations in diffusion generation" (Section 3.2). While the paper demonstrates sensitivity to k, it does not report sensitivity to [A,B] selection specifically. For a method claiming to be a general framework, the lack of principled setting guidelines is a weakness—though common for first works and partially mitigated by the ablations that do exist.

- **Compute cost accounting is incomplete.** The abstract correctly states "without the need to retrain diffusion models," and Section 4.1 claims "all results can be obtained on a single RTX 4090." However, the method requires training a ConvNet-6 surrogate on the full dataset for 50 epochs and computing/storing averaged gradients across all retained checkpoints. No wall-clock time or memory breakdown is provided for these steps, making it difficult to compare the true cost against competitors like RDED (which doesn't require a surrogate model). This is an important practical consideration that the paper should address.

- **The surrogate model capacity gap is untested for ImageNet-1K.** The ConvNet-6 surrogate used for ImageNet-1K is a very small model compared to the ResNet-18 and larger architectures used for evaluation. Table 4 tests cross-architecture surrogate robustness only on ImageNette/Woof, where the gap between surrogate and evaluation model is smaller. While the ImageNet-1K cross-architecture results (Table 3) implicitly validate the surrogate's usefulness, an explicit ablation on ImageNet-1K with a stronger surrogate would strengthen confidence.

### Trivial
None.

## Nice-to-Haves

- A principled heuristic for setting [A,B] and k (e.g., based on signal-to-noise ratio of the predicted clean image or variance of the influence gradient) would substantially strengthen the generality claim.
- Analysis of approximation error from using real-data checkpoints vs. synthetic-data checkpoints, even on a small-scale dataset, would quantify the gap introduced by the key theoretical shortcut.
- Failure cases showing when IGD produces worse results than vanilla (e.g., for rare classes or poorly chosen k) would reveal operational boundaries.
- Analysis of how the order of generation affects results under deviation guidance (since it pushes away from only the single most-similar previously generated sample).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that the abstract is misleading about soft labels.** The harsh critic argues the 60.3% headline depends on soft labels that are "not part of IGD itself." However, Section 4.2 clearly states: "Following the evaluation protocol of the RDED, we employ a ResNet-18 model, trained on the original dataset, to generate soft labels for synthetic images." This is a shared evaluation protocol, not a method-specific trick—and IGD's gains over DiT (52.9→59.8%) and Minimax (58.6→60.3%) at IPC=50 are computed under the same protocol, so relative improvements are attributable to IGD.

- **Claim that the comparison with baselines on ImageNet-1K may be unfair due to soft labels.** The paper explicitly states it follows RDED's evaluation protocol, implying the same soft-label setup. Whether SRe²L and G-VBSM use soft labels is a question about those methods' own protocols, not a fairness critique of IGD.

- **Claim that Table 6 comparison with 10 regular checkpoints is not apples-to-apples.** The paper frames this as an efficiency argument (4 smartly-selected checkpoints beat 10 regular ones), and the text makes this clear. It is fairly presented.

- **Claim that the ∝ in Eq. 5 is problematic.** This proportional relationship is inherited from the standard trajectory influence formulation (Pruthi et al., 2020) and is a well-known first-order approximation. Questioning the approximation's validity for short, high-learning-rate regimes is reasonable but is a known limitation of trajectory influence methods broadly, not specific to this paper.

- **Formatting and presentation nitpicks** (citation format, capitalization, etc.) are removed per the rules.

- **Missing experiments about surrogate model quality on ImageNet-1K** is moved to minor weakness above; the claim that this is a "critical" gap is overstated since the ImageNet-1K cross-architecture results already provide implicit validation.

## Novel Insights

The most striking finding from the reviews is the asymmetric contribution of influence vs. diversity guidance across backbones: for DiT (which lacks diversity), deviation guidance alone closes most of the gap with Minimax; for Minimax (which already handles diversity), influence guidance is the dominant factor. This suggests the two guidance components are not simply complementary filters but address fundamentally different failure modes that vary by backbone—diversity guidance compensates for the diffusion model's inherent mode concentration, while influence guidance compensates for suboptimal training-effectiveness of randomly sampled data. This asymmetry, if properly analyzed, could inform a more principled design where guidance allocation adapts to the backbone's known weaknesses.

## Suggestions

- Add a brief discussion or analysis section explicitly examining why influence vs. diversity guidance matter differently for DiT vs. Minimax, connecting this to each backbone's known limitations. This would transform the currently descriptive ablation into an explanatory one.
- Provide wall-clock time and memory comparisons for the full IGD pipeline (surrogate training + checkpoint collection + guided generation) vs. RDED to enable proper practical assessment.
- Add sensitivity analysis for the [A,B] guidance range, which is currently untested despite being critical to the method's success.

## Evaluation

**Originality:** The core idea of using trajectory influence functions as diffusion guidance signals is genuinely novel and a meaningful departure from prior guided diffusion work. The framing of dataset distillation as guided diffusion generation is also creative. The theoretical derivation, while having gaps, provides useful motivation.

**Importance:** Dataset distillation for high-resolution, large-scale datasets (especially ImageNet-1K) is an important problem with practical impact. Achieving SOTA results without retraining diffusion models is a significant practical advance.

**Claims support:** Empirical claims are well-supported by extensive experiments across datasets, architectures, and configurations. The theoretical claim of "correlating" DD with trajectory influence is partially supported but weakened by the unanalyzed approximations.

**Experimental soundness:** Experiments are thorough—multiple datasets, IPC settings, architectures, ablations, and visualizations. The main gaps are incomplete compute cost reporting and missing surrogate quality ablation on ImageNet-1K.

**Clarity:** The paper is generally well-written and the method is clearly described. Algorithm 1 helps implementation clarity. The theoretical section could be clearer about the status of its claims (motivational vs. rigorous).

**Community value:** This work opens a new direction (influence-guided diffusion for DD) and shows strong practical results. The code release further enhances value.

## Calibration

| Anchor Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| MGD³ (mode-guided diffusion for DD, training-free) | /home/wg25r/review_agent/human_reviews/vKJ8YH0iNp.md | 5.0 | IGD is clearly stronger: more novel guidance mechanism (influence vs. mode clustering), better results (60.3% vs. modest gains), more thorough evaluation, and genuine theoretical motivation even if imperfect. MGD³ was rejected partly because influence on DD was unclear; IGD directly addresses DD effectiveness. |
| DATM (difficulty-aligned trajectory matching for DD) | /home/wg25r/review_agent/human_reviews/rTBL8OhdhH.md | 7.0 | DATM has similarly strong empirical contributions and clearer practical impact (lossless DD). IGD has comparable empirical strength but weaker theoretical grounding than DATM. |
| Adversarial robustness via SNNs (overclaiming, avg 2.0) | /home/wg25r/review_agent/human_reviews/KncRpAnprQ.md | 2.0 | IGD is much stronger: real novelty, solid empirical results, and no fabricated claims—just theoretical gaps. This anchor establishes the low end. |
| Momentum-driven guided sampling (theoretical gaps, avg 5.33) | /home/wg25r/review_agent/human_reviews/i8bdPSmOwk.md | 5.33 | Similar profile: novel guidance mechanism with theoretical gaps but practical value. IGD is somewhat stronger due to better empirical results and a more significant problem setting. |
| Hyperparameter sensitivity paper (avg 4.33) | /home/wg25r/review_agent/human_reviews/8FxELTdwJR.md | 4.33 | IGD's HP sensitivity is real but less severe—k sensitivity is documented, and the method works across varied settings. |

IGD falls above MGD³ (5.0) and the momentum-guided paper (5.33) due to stronger empirical results and more novel contribution, but below DATM (7.0) due to weaker theoretical grounding. The paper makes a genuine and significant practical contribution with a novel idea, and the theoretical gaps—while real—are motivational rather than invalidating.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>