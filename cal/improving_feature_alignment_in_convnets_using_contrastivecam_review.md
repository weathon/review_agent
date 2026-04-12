=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary
This paper argues that HiResCAM explanations are only defined at the logit level and are therefore ambiguous with respect to softmax-equivalent shifts, then proposes ContrastiveCAMs to remove that redundancy by using class-vs-class differences. Building on this representation, it introduces Core-Focused Cross-Entropy (and KL-regularized variants) to suppress non-core regions during training using spatial masks, and reports improved saliency-mask overlap on Hard-ImageNet, Oxford-IIIT Pets, and Pascal VOC, often with some loss in standard classification accuracy.

## Strengths
- The paper makes a specific and potentially useful connection between interpretability and training: the proposed losses are written directly in terms of spatial attribution maps, not merely used for post-hoc analysis. This is more ambitious than standard “explain then inspect” workflows and is concretely instantiated in Definitions 4.5/4.7 and the multilabel extension in Appendix B.
- The class-vs-class formulation is a real conceptual contribution even if one is skeptical of the framing around HiResCAM. The pairwise ContrastiveCAMs in Definition 3.3 expose which regions distinguish the target from each competing class, and Figure 2 supports the claim that different class comparisons can rely on different image regions.
- The empirical story on alignment is nontrivial and dataset-dependent rather than cherry-picked to only show wins on clean accuracy. On Hard-ImageNet, CFCE substantially reduces performance when core regions are ablated (e.g., gray mask and box masking in Table 2), which is consistent with the intended goal of making predictions depend more on core regions; on Pets and VOC, the method also shows large IoU gains.
- The paper does not rely entirely on perfect masks: Section 5.2 includes experiments with GT masks, SAM masks, and bounding boxes, and the SAM/BBOX results suggest the approach can retain some of the alignment benefit under weaker supervision.
- The paper is unusually explicit about the architectural conditions needed for its faithfulness claims. Appendix C.1 explains why the final downsampling, bias, and final BN/ReLU are removed, rather than quietly assuming the CAM derivations hold for an unmodified architecture.

## Weaknesses
### Fatal
- None.

### Major:
- **The central theoretical framing overstates what has been shown about HiResCAM.** Theorem 3.2 proves a non-identifiability statement from *probability predictions* back to classwise HiResCAMs because softmax is invariant to adding a constant to all logits. But for a fixed trained network and input, HiResCAM itself is still a deterministic quantity computed from that network’s activations and gradients. The paper repeatedly phrases this as a limitation of HiResCAM explanations themselves (“fail to guarantee a faithful interpretation”), which is too strong given what is actually proved. What is established is an ambiguity in explanations at the probability level under softmax-equivalent logit shifts, not that the computed HiResCAM for a given model/input is non-unique. This distinction matters because it weakens the claimed diagnosis that HiResCAM is fundamentally flawed.
- **The empirical alignment comparisons are partially confounded by using different explainers for baselines and the proposed method.** In Table 2, the paper states: “IoU for this benchmark was computed using GradCAMs only for consistency with baselines… We thus include additional evaluations using ContrastiveCAMs for core-focused models.” This means the key alignment comparison is not on a common explanation metric across methods. Since the paper’s core claim is that ContrastiveCAM is more faithful than standard CAMs, mixing GradCAM IoU for baselines with ContrastiveCAM IoU for the proposed models makes the magnitude of the reported alignment gain hard to interpret. A fair comparison should report the same attribution/evaluation pipeline for all methods, even if additional legacy metrics are also included.
- **The method’s gains come with a substantial clean-accuracy tradeoff that is not deeply analyzed.** On Hard-ImageNet, the standard accuracy drops from 94.25 for CE to 90.53 for CFCE and 90.35 for CFCE+KL. That may be acceptable for an alignment method, but the paper mostly presents this as a modest cost (“at the cost of some un-ablated performance”) without really analyzing the Pareto frontier or when the tradeoff is worthwhile. Since the method is explicitly designed to suppress contextual evidence, understanding when it removes harmful shortcuts versus genuinely useful context is central to judging significance.
- **The proposed training objective depends on CAM-derived quantities, but the paper gives little practical analysis of optimization behavior.** From Eq. 15/18 and the HiResCAM definition, training must differentiate through gradient-based attribution terms; this is more complex than standard CE training. The paper provides hyperparameters, but does not quantify computational overhead, stability, gradient norms, sensitivity to the loss weights, or whether the method is materially harder to optimize than CE. Given the unusual objective, this omission leaves practical viability insufficiently characterized.
- **The architectural changes are a meaningful confound and are not fully isolated.** Appendix C.1 modifies the backbone/classifier by removing final downsampling, final bias, and final BN/ReLU. The paper does include a “CE w/ Arch” baseline, which is good and addresses part of this concern, but there is still no ablation disentangling which of these modifications matter most and how much of the observed effect is due to altered architecture/feature resolution versus the proposed loss itself. Since faithfulness and IoU can be very sensitive to spatial resolution and final-layer structure, this matters for attributing gains correctly.

### Minor
- **The dependence on spatial supervision limits applicability.** The main CFCE formulation requires core-region masks \(H\). The paper does make a reasonable effort to mitigate this with SAM and bounding boxes, so this is not a fatal scope issue, but the method is still less broadly applicable than standard classification training and its effectiveness will depend on the quality of available masks.
- **Some of the theoretical claims are stronger than the assumptions justify.** For example, Theorem 4.6 relies on a realizability-style argument and is framed as consistency/classification calibration. As presented, this offers limited practical insight into behavior on realistic deep networks and noisy datasets; it is best viewed as a surrogate-risk sanity result rather than strong evidence for real-world optimization or generalization.
- **Variance in some IoU results suggests sensitivity that deserves discussion.** In Pets, CE w/ Arch has very large IoU variance (e.g., around ±17), and the paper could do more to explain whether this reflects instability in attribution thresholding, data split effects, or sensitivity to mask quality and initialization.
- **The paper would benefit from stronger robustness validation beyond mask-based alignment metrics.** Hard-ImageNet ablations are relevant, but stronger evidence for “feature alignment” would include OOD or distribution-shift tests showing that the enforced focus on core regions improves generalization rather than merely matching annotated masks.

### Trivial
- **The handling of the absolute value in Eq. 15 is not explained.** In practice, autodiff systems will use a subgradient almost everywhere, but the paper should explicitly state this and note behavior at zero.
- **Mask preprocessing details are under-specified.** Since losses and IoU are computed at the spatial resolution of the final feature map, the exact downsampling/interpolation strategy for masks should be stated because it can affect both training and reported overlap.

## Nice-to-Haves
- Report a common-explainer evaluation table for all methods: e.g., IoU using GradCAM, HiResCAM, and ContrastiveCAM for both baselines and proposed models.
- Add a compute/stability analysis: training time or memory overhead relative to CE, plus sensitivity to the KL/loss weights.
- Provide a Pareto analysis of clean accuracy vs. alignment/ablation robustness across different regularization strengths.
- Include failure cases where suppressing non-core regions hurts because context is genuinely informative.
- Add OOD/distribution-shift evaluations to test whether improved alignment translates into better robustness.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Eq. (3) does not hold for the standard architectures it initially claims to cover.”**  
  This is too strong as stated. The paper explicitly restricts the exact derivation to a simplified single-layer classifier setting and later modifies the architecture in Appendix C.1 precisely to recover the needed faithfulness property. This is better treated as a confound/ablation issue, not a factual error invalidating the derivation.

- **“The paper lacks a fair control with standard CE under the same architecture.”**  
  Removed because this is factually incorrect: the paper does include “CE w/ Arch” baselines in Tables 2 and 3, which directly address part of that concern.

- **“The model collapses under core-region ablation, so the method is broken.”**  
  This is a misinterpretation. For an alignment method designed to make predictions rely on core regions, worse accuracy after *removing core regions* is actually directionally consistent with the objective, not evidence of failure by itself. The real issue is the clean-accuracy tradeoff and whether the improved dependence on core regions justifies it.

- **Criticisms about code/model/dataset availability or release status.**  
  Not applicable here and removed by instruction.

- **Requests for additional related work.**  
  Omitted per instruction.

## Novel Insights
The most useful synthesis is that the paper is stronger as an interpretability-guided training paper than as a diagnosis of a fatal flaw in HiResCAM. The pairwise class-contrast view appears genuinely informative, and the training objective does seem to steer models toward mask-aligned evidence. However, the paper currently over-anchors its contribution on a softmax non-identifiability argument that does not by itself show ordinary HiResCAM outputs are non-unique for a fixed model. Reframing the contribution around “probability-level contrastive explanations and mask-guided alignment” would make the work both more precise and more compelling.

## Suggestions
- Sharpen the theory section to distinguish clearly between:
  1. non-identifiability of spatial explanations from class probabilities under softmax, and  
  2. deterministic computation of HiResCAM for a fixed model/input.  
  This would remove the biggest conceptual overclaim.
- Rework the experimental presentation so that all methods are evaluated with the same attribution metric in the main comparison table.
- Add an explicit compute/stability subsection for CFCE/RCFCE training, including memory/time overhead and any optimization tricks needed in practice.
- Provide an ablation over the architectural changes in Appendix C.1, especially final stride and bias removal, to isolate how much each change contributes.
- Analyze the clean-accuracy/alignment tradeoff more directly, ideally with a hyperparameter sweep and a Pareto plot.
- Expand robustness evaluation beyond saliency overlap and core ablation to include distribution shift or OOD benchmarks.


# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
