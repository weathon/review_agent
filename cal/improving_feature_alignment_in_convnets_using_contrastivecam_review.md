=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary

This paper identifies a theoretical limitation of HiResCAMs—non-uniqueness due to softmax shift invariance, where an arbitrary matrix M can be added to all class-level explanations without changing predictions—and proposes ContrastiveCAMs as an invariant alternative that additionally provides class-versus-class explanations. Leveraging the decomposability of ContrastiveCAMs, the authors derive Core-Focused Cross-Entropy (CFCE), a loss that penalizes contributions from non-core regions to improve feature alignment. Experiments on Hard-ImageNet, Oxford-IIIT Pets, and PASCAL VOC demonstrate improved alignment with core regions compared to standard cross-entropy training.

## Strengths

- **Formal theoretical grounding of an interpretability flaw:** Theorem 3.2 provides a clean mathematical proof that HiResCAMs admit arbitrary spurious shifts due to softmax invariance, and Theorem 3.5 proves ContrastiveCAMs eliminate this ambiguity. This goes beyond informal critique and offers a principled resolution, including Proposition 4.1 showing probabilities are directly expressible via ContrastiveCAMs.
- **Bridging interpretability and optimization:** Rather than using CAMs solely for post-hoc inspection, the paper embeds the interpretability signal into the training objective via CFCE. The derivation from ContrastiveCAMs to the loss (Proposition 4.2 → Definition 4.5 → Theorem 4.6 showing consistency with constrained risk minimization) is a coherent pipeline from analysis to intervention.
- **Evaluation focused on alignment, not just accuracy:** The paper uses core-region ablation accuracy, IoU with core masks, and Relative Foreground Sensitivity (RFS), directly measuring whether models attend to the right regions. On Hard-ImageNet, CFCE drops from 90.5% to 41.8% accuracy under gray-mask ablation (vs. 94.3%→75.9% for CE), confirming greater core-region dependence.
- **Testing with approximate masks (Section 5.2):** The paper evaluates CFCE with SAM-generated and bounding-box masks rather than only ground truth, demonstrating practical viability. CFCE+KL with SAM masks achieves 83.95% IoU on Oxford-IIIT Pets (valid), compared to 92.72% with GT masks—competitive given the weaker supervision.

## Weaknesses

- **Architectural modifications confound the loss function contribution.** Appendix C.1 describes removing the final downsampling stride, bias, BatchNorm, and ReLU from ResNet-50. While "CE w/ Arch" is included as a baseline, the paper never tests CFCE on an unmodified ResNet-50. The Table 2 CE w/ Arch row shows that architecture changes alone *reduce* alignment (IoU 16.25% vs. 18.44% for standard CE), which partially isolates the loss contribution—but it remains unclear whether CFCE would work at all without these architectural constraints, since the theory (Proposition 4.1 onward) assumes a bias-free classifier and the loss formulation relies on the direct HiResCAM-logit correspondence. The necessity of these modifications should be explicitly analyzed.

- **No comparison to simpler region-masking baselines.** A natural alternative to CFCE is simply masking or zeroing non-core regions during training input, without involving ContrastiveCAMs in the loss. Without this baseline, it is unclear whether the added complexity of computing CAMs during every training step and differentiating through them is justified. The paper's contribution could be significantly undermined if simple input masking achieves comparable alignment.

- **Computational overhead of training with ContrastiveCAMs is unreported.** CFCE requires computing ContrastiveCAMs (which involve gradients of logits w.r.t. feature maps) at every training step and differentiating through them (higher-order gradients). The paper states 150-epoch training with batch size 768 but provides no comparison of training time or GPU memory versus standard cross-entropy. For ICLR, this gap makes it impossible to assess practical feasibility.

- **Fundamental dependency on core-region masks H with no systematic noise analysis.** The method requires binary masks delineating core vs. non-core regions. While SAM and bounding-box approximations are tested, there is no controlled analysis of how mask quality (e.g., IoU between H and ground truth) affects CFCE performance. If H incorrectly labels relevant regions as non-core, CFCE will actively suppress useful features. Table 3 shows very high variance for CE w/ Arch IoU (38.58 ± 16.95), and CFCE IoU in the binary setting (82.92 ± 1.18) is more stable, but the sensitivity to mask errors is not characterized.

- **Evaluation metric inconsistency.** In Table 2, IoU is computed using GradCAMs "for consistency with baselines," despite the paper's central argument that GradCAM produces unfaithful explanations (citing Draelos and Carin, 2020). The additional ContrastiveCAM IoU column (93.39% for CFCE+KL) partially addresses this, but the main comparison uses a metric the paper itself argues is unreliable, creating a tension in the evaluation.

- **The non-uniqueness result reflects a general property of additive logit decompositions under softmax, not a flaw specific to HiResCAM.** Theorem 3.2 follows from the well-known softmax shift invariance (Proposition 3.1) applied to the logit decomposition (Eq. 3). While the paper correctly identifies the consequence for HiResCAM faithfulness, framing this as a "limitation of HiResCAM" overstates the case—it would apply to *any* attribution method that decomposes logits additively. The contribution is in identifying the consequence and proposing the resolution, not in discovering a novel flaw in HiResCAM specifically.

## Nice-to-Haves

- **Out-of-distribution robustness evaluation:** Feature alignment should, in principle, improve OOD generalization. Testing on domain shift benchmarks (e.g., ImageNet-A/R) would substantiate the broader robustness claim beyond core-region ablation.
- **Ablation on KL regularization hyperparameters:** The λ values {50, 10³, 10} are stated in Appendix C without sensitivity analysis. Given that the KL term applies softmax with temperature λ₃ to CAMs (which can be negative), the behavior is sensitive to these choices.
- **Per-class IoU breakdown:** Aggregate IoU may mask classes where CFCE fails to improve alignment. A per-class analysis would reveal whether certain object categories or mask qualities cause failures.
- **Evaluation on Vision Transformers:** The method explicitly targets ConvNets; testing generalizability to ViT architectures would strengthen claims about the broader utility of ContrastiveCAMs.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Absolute value operator non-differentiability at zero (Harsh Critic).** In Definition 4.5, |CAM| is used in the loss. Sub-gradient methods (standard in deep learning, e.g., ReLU) handle this trivially. This is an implementation detail, not a substantive flaw.
- **Weakness: Missing related work comparisons (IRM, attention-based methods, Gao et al. 2024, Weber et al. 2023).** Per hard rules, I cannot confirm the existence or relevance of specific uncited works. The paper does cite Weber et al. (2023) and Gao et al. (2024) in related work as surveys, not as methods to compare against.
- **Weakness: Number of random seeds not in main text.** Standard deviations are reported in all tables. Demanding explicit seed counts in the main text is a formatting nitpick.
- **Weakness: Table 2 interpretation confusion (Harsh Critic).** The critic misread the ↓ arrows on "Gray Mask" columns. Lower accuracy under core-region ablation is *better* (it means the model relies on core regions), which is exactly what CFCE achieves and what the paper claims. The critic's confusion was due to formatting, not an error in the paper.
- **Weakness: ContrastiveCAMs require C² pairwise comparisons (Harsh Critic).** For the *loss* (Definition 4.5), the summation is over C classes, not C² pairs. The C² concern applies only to full pairwise explanation visualization, which is used for analysis, not during training.
- **Weakness: "philosophical stance on attribution" (Harsh Critic).** The claim that HiResCAM non-uniqueness is merely philosophical rather than mathematical is incorrect—the paper proves that the same prediction maps to infinitely many explanations, which is a mathematical fact about the faithfulness of the mapping, not a philosophical preference.
- **Weakness: Large artifacts impractical to include (complete training logs).** Per hard rules, this is a nitpick.

## Novel Insights

The paper reveals an underappreciated structural connection: the same softmax property that makes neural network predictions well-calibrated (shift invariance) is exactly what makes logit-space attributions unfaithful. ContrastiveCAMs resolve this by operating in the *difference* space, which is the natural domain of softmax-based classification. A deeper observation from Table 1 is that the redundancy ratio γ varies substantially across datasets (0.201 for Hard-ImageNet vs. 0.367 for Pets), suggesting that the practical impact of the spurious shift M depends on dataset structure—datasets where non-core regions are more predictive (Hard-ImageNet) have lower redundancy ratios, meaning the shift M captures less of the total explanation. This hints that the non-uniqueness problem may be most severe precisely when it matters least (well-aligned models), and least severe when alignment is worst—a paradox worth investigating.

## Suggestions

- Run CFCE on a standard (unmodified) ResNet-50 to determine whether the architectural modifications are necessary for the method to function, and report the results even if performance degrades. This directly addresses the most significant confound in the experimental design.
- Add a simple input-masking baseline (zero out non-core regions of the input image before feeding to the network, train with standard CE) to determine whether ContrastiveCAM-based loss provides benefits over this trivial approach.
- Report training time and peak GPU memory for CFCE vs. standard CE to establish practical feasibility.
- Conduct a controlled mask-noise experiment: corrupt ground-truth masks at known error rates (e.g., flip 10%, 20%, 30% of mask pixels) and measure CFCE performance degradation, quantifying the method's robustness to imperfect supervision.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
