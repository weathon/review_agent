---

## Summary

This paper proposes the *Dynamic Sparsity Corruption Robustness* (DSCR) Hypothesis: Dynamic Sparse Training (DST) at low sparsity levels (10%–50%) can consistently outperform Dense Training in robustness against image corruption, without increasing resource cost. The claim is empirically validated across nine scenarios spanning image/video benchmarks (CIFAR-C, TinyImageNet-C, ImageNet-C/3DCC, corrupted UCF101), multiple DST algorithms (SET, RigL, MEST, GraNet), and architectures (VGG, ResNet, EfficientNet, DeiT, I3D). The paper additionally provides mechanistic analysis from spatial and spectral perspectives, arguing that DST introduces implicit regularization by reducing attention to high-frequency features.

---

## Strengths

- **Genuinely novel empirical discovery in a literature gap.** Corruption robustness has been studied for pruning/LTH-based sparse methods (Diffenderfer et al. 2021, Liebenwein et al. 2021), but systematic evaluation of *dynamic* sparse-to-sparse training (SET, RigL, MEST, GraNet) on corruption benchmarks is entirely absent from prior work. This paper fills that gap with a clear hypothesis.
- **Exceptional empirical breadth.** Nine experimental scenarios across images and videos, five DST algorithms, six architectures spanning CNNs and Transformers, and multiple corruption benchmarks (19+ corruption types, five severity levels) constitute one of the more thorough evaluations of its kind. The consistency of the finding across diverse settings — including 3D convnets, transformers, and video data — strengthens credibility.
- **Substantive gains in multiple settings.** Improvements are not uniformly marginal: CIFAR100-C shows +4 pp (51.6 → 55.6 for ResNet34), UCF101 shows +2–4 pp (51.14 → 53.57 for 3D ResNet50, 53.58 → 56.30 for I3D), and DeiT-base gains 2.6+ pp over its dense counterpart. These are practically meaningful, not noise-level differences.
- **Novel and mechanistically coherent spectral analysis.** The Radius-Accuracy (RA) curve framework (Figure 7) showing that high-frequency attenuation degrades dense models significantly more than DST models, while low-frequency attenuation affects both equally, is a clean diagnostic that directly connects DST's training dynamics to the corruption types where it excels. This is grounded in prior frequency aliasing literature (Grabinski et al., Li et al.) and represents a genuine explanatory contribution beyond pure benchmarking.
- **Validation extends to Transformers and video architectures.** Demonstrating that the effect transfers to DeiT-base and 3D ConvNets (I3D) substantially broadens the hypothesis beyond CNNs, an important generalization check.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing static sparse training baseline — the "dynamic" claim is unverified.** The paper's central claim is specifically about *Dynamic* Sparse Training. However, no comparison against *static* sparse training (same density, fixed topology) is presented. Without this control, one cannot conclude that the remove-and-regrow dynamics are responsible for robustness gains, as opposed to simple capacity reduction or magnitude-based regularization that would occur in any sparse model. This is the most critical ablation absent from the paper, and its omission means the DSCR Hypothesis as formulated — about *dynamic* sparsity — is not fully justified by the experiments.

- **Clean accuracy is never reported.** The paper focuses entirely on corruption robustness accuracy and never systematically reports clean (uncorrupted) accuracy. The robustness-accuracy tradeoff is well-documented, and DST could be implicitly sacrificing clean accuracy to gain robustness (a known failure mode of implicit regularization methods). Without a clean accuracy table, practitioners cannot assess whether DST is a genuine improvement or simply a tradeoff rebalancing. This is essential information for any robustness paper.

- **The sole comparison baseline — vanilla Dense Training — sets a low bar.** The paper compares DST only to vanilla dense training, with no robustness-enhancing baselines (e.g., Dense + AugMix, Dense + CutMix, or Dense + stronger weight decay). To be clear, the paper *explicitly acknowledges* this in Section 6 ("moving beyond the basic fact revealed in this paper, i.e., vanilla DST outperforms vanilla Dense Training") and positions the DSCR hypothesis as a starting point, not the final word. This scoping is reasonable, but it limits the practical significance of the current contribution: it is unknown whether DST provides any advantage over robustly trained dense models, which is what practitioners actually use.

- **ImageNet gains are very small without statistical support.** On the most widely benchmarked and practically important dataset (ImageNet-C), the improvements over Dense Training are 0.32 pp (38.38 → 38.70 for RigL) and 0.34 pp (→ 38.72 for GraNet). These are on single-run evaluations with no variance estimates. While single-run evaluation is standard practice at ImageNet scale, gains of this magnitude — below half a percentage point — could plausibly be training run variance, and no multi-seed or confidence analysis is offered. The case made for ImageNet specifically is notably weaker than for CIFAR100-C and UCF101.

### Minor

- **Table 2 has significant labeling issues that undermine reproducibility.** The "Reg." and "MixNets" column headers are not defined in the caption or the main text (the caption states the table presents gradient-based regrow DST methods but does not map these two labels to specific algorithms). Additionally, "ImageNet-38C" (row 6) appears to be a typo for ImageNet-3DCC, "DST" in the Model column (row 7) should be DeiT-base, and "VID" (row 9) should be I3D. These errors make Table 2 — the primary summary of all 9 scenarios — difficult to interpret.

- **Unexplained inconsistency in sparsity ranges across experiments.** CIFAR10-C, CIFAR100-C, and TinyImageNet-C experiments sweep sparsity ratios from 0.3 to 0.7, but ImageNet, UCF101, and DeiT experiments use only sparsity = 0.1. The DSCR Hypothesis claims benefits at "low sparsity levels (10%–50%)," but 0.1 (90% of parameters retained) is barely sparse and is near the boundary of what the hypothesis covers. Why ImageNet experiments were conducted at such low sparsity is not explained, and the lack of a sparsity sweep on ImageNet leaves the hypothesis less well-validated for this key benchmark.

- **The spectral analysis establishes correlation, not causation.** Section 5.2 demonstrates that (1) DST models are more robust to high-frequency corruptions, and (2) DST models are less sensitive to high-frequency attenuation. These two observations are consistent with a causal story but are also consistent with a common cause (e.g., reduced capacity). The paper does not provide a causal intervention — e.g., training a dense model with an explicit high-frequency suppression inductive bias to see whether it matches DST's robustness profile. The mechanistic explanation remains a well-motivated hypothesis rather than an established finding.

- **DeiT and ImageNet results evaluated at a single sparsity point.** For DeiT-base, only sparsity = 0.1 is tested. This provides no information about the sparsity-robustness curve for Transformers, making it impossible to know whether the behavior mirrors CNNs. A sweep would substantially strengthen the generalizability claim.

- **The "25% improvement" claim in Section 4.2 is not contextualized.** The paper reports "nearly a 25% improvement" for MEST_g at severity level 5 for impulse noise. This is a *relative* gain for a *single corruption type* at the *highest severity level* — arguably the most favorable slice of the data. The claim is accurate but should be contextualized against the mean improvement across all corruption types and severities (which on ImageNet-C is 0.32 pp absolute).

### Tiny

- **The abstract's claim "without adding (or even reducing) resource cost" is only qualified in a footnote.** Footnote 4 acknowledges that binary masking is used to simulate sparsity, and actual wall-clock savings depend on sparse hardware support. This important caveat should be in the main text alongside the claim.
- **Figure 2 (CIFAR10-C) shows non-monotonic behavior** that is not analyzed — DST does not uniformly outperform at all sparsity ratios for this dataset, and the paper's discussion of this case is less thorough than for CIFAR100-C/TinyImageNet-C.

---

## Nice-to-Haves

- **Compare DST combined with existing robustness methods** (AugMix, Mixup is already in the appendix — elevating this to a main-text result would strengthen the practical relevance argument). If DST is complementary to augmentation, showing this combination is important for practitioners.
- **DST component ablation:** Isolate whether robustness gains come from the pruning criterion, the regrowth strategy (random vs. gradient), or the topology update frequency. Section 6 mentions this as future work, but even a partial ablation would deepen the mechanistic claim.
- **Wall-clock training time comparison:** Rather than FLOPs estimates, actual training time measurements would clarify the "no extra resource cost" claim in practical terms.
- **Training dynamics visualization:** Plotting robustness accuracy during training (not just at convergence) for DST vs. Dense would reveal whether DST converges to a qualitatively different solution or merely regularizes throughout optimization.
- **Failure mode characterization:** A systematic characterization of which corruption types and severity levels where DST *underperforms* dense training (as visible in parts of Figure 2 for CIFAR10-C) would strengthen both honesty and utility.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Consistently outperform" framing in the abstract is too strong** *(Harsh Critic)*: The DSCR Hypothesis itself is explicitly qualified by "at low sparsity levels," and Figure 2 shows the behavior is non-monotonic. However, the hypothesis is properly bounded — the reviewer's concern partially reflects a mismatch between the abstract's phrasing and the body text rather than a factual error. The weaker phrasing in Section 7 ("at specific sparsity ratios") is more accurate. This is a minor wording issue, not a substantive flaw.

- **No confidence intervals / multi-seed statistics** *(Harsh Critic, Spark Finder)*: At ImageNet scale, single-run evaluation is the established norm in the field. Requiring multi-seed statistics here would be a non-standard rigor demand. The concern is partially valid for CIFAR-scale (absorbed into the "ImageNet gains" Major weakness), but removing it as a standalone point is appropriate.

- **Table 2 "9-0" framing is misleading because it's a snapshot** *(Harsh Critic)*: The paper explicitly states "The table takes a snapshot at a particular sparsity level" in the caption. Criticizing the authors for misrepresenting this is unfair given the disclosure, though the snapshot is indeed at favorable sparsity levels. This is absorbed into the Minor point on Table 2 labeling.

- **LTH robustness literature connection is insufficiently distinguished** *(Harsh Critic)*: The paper does cite and discuss Diffenderfer et al. 2021 and Liebenwein et al. 2021 in Section 2.1. The distinction between post-training pruning/LTH and sparse-to-sparse DST is real and noted. This is not a genuine gap.

- **Missing SAM/flat minima literature** *(Harsh Critic)*: Per review instructions, missing related works are not raised when external sources cannot be confirmed.

- **The training pipeline parity concern** *(Harsh Critic)*: The paper uses the same hyperparameter tuning for dense and DST baselines, with sparsity ratio as the sole additional variable. This is standard practice in DST evaluation and does not constitute an unfair comparison.

- **Unfair robustness comparison (DST vs. vanilla dense is asymmetrically favorable to the baseline)** *(Spark Finder)*: The comparison is intentionally "vanilla vs. vanilla" — DST is not augmented while Dense Training is not augmented either. This is symmetric, not asymmetric. The concern that it's a "low bar" is already captured as a Major weakness about baseline strength.

---

## Novel Insights

The paper's most novel analytical insight — beyond the empirical finding itself — is the Radius-Accuracy (RA) curve analysis showing that DST models are structurally *less reliant on high-frequency features* compared to their dense counterparts, while their reliance on low-frequency features is indistinguishable. This directly explains the pattern in Figure 3, where DST's largest advantages arise on corruption types rich in high-frequency content (impulse noise, Gaussian noise, shot noise, Perlin/plasma/blue noise). The connection to frequency aliasing at downsampling operations (Grabinski et al., Li et al.) provides a principled theoretical anchor: DST's implicit sparsity pressure naturally suppresses high-frequency representations that would otherwise get aliased during pooling, offering a mechanistic link between topology dynamics and robustness that is genuinely new in the corruption robustness literature.

---

## Suggestions

- **Add a static sparse training baseline at matched densities in at least one scenario.** Even a single dataset (e.g., CIFAR100-C with ResNet34 at sparsity 0.3/0.5) comparing SET or RigL to a fixed-mask network of identical density would directly address whether the *dynamic* aspect of DST is the source of robustness gain. This is the single highest-priority revision.
- **Report clean accuracy alongside robustness accuracy for all main experiments**, ideally as a Pareto plot (clean vs. robust accuracy) to allow readers to assess the tradeoff.
- **Fix Table 2:** Define "Reg." and "MixNets" columns explicitly (specifying exactly which DST algorithms they refer to), correct "ImageNet-38C" to "ImageNet-3DCC," replace "DST" in the Model column with "DeiT-base," and replace "VID" with "I3D."
- **Explain the sparsity range choice for ImageNet experiments (0.1 only) in the main text.** If hardware constraints or compute budget motivated this choice, say so. If 0.1 was chosen because higher sparsity failed to show gains, that is itself an important finding to report.
- **Explicitly contextualize the 25% relative gain claim** in Section 4.2 by also reporting the mean relative gain across all corruption types and severities so readers can calibrate the magnitude of DST's advantage at the typical case versus the best case.

---

**Evaluation summary:** The paper is a **moderately strong** empirical contribution that opens a genuinely new direction — corruption robustness as a benefit of dynamic sparse training — with impressive experimental breadth and a mechanistically coherent spectral analysis. Novelty is solid. Empirical support is broad but uneven (strong for CIFAR100-C and video, weak for ImageNet). Technical soundness is undermined by the absence of the static sparse training ablation and clean accuracy data, which together leave the central causal claim partially ungrounded. Clarity is good in prose but suffers from Table 2 labeling errors. Significance is meaningful but currently bounded by the vanilla-vs-vanilla comparison scope the authors themselves acknowledge. The paper is publishable with major revisions addressing the static sparse baseline and clean accuracy reporting.