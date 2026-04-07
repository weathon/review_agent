=== CALIBRATION EXAMPLE 20 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately conveys the two main contributions: ContrastiveCAMs and Core-Focused Cross-Entropy (CFCE). The abstract's claims are mostly substantiated by the experiments. One concern: the abstract claims ContrastiveCAMs provide "more faithful attention maps," but the notion of faithfulness employed here is specifically faithfulness with respect to probability predictions rather than to some ground-truth attribution. This distinction matters and is only partially clarified in the paper body. The claim that HiResCAMs "fail to guarantee a faithful interpretation" (Section 3) should be more precisely scoped in the abstract — for a fixed trained model on a fixed input, HiResCAMs are deterministically computed; the non-uniqueness is a property of the equivalence class of logit decompositions, not of the computation itself.

---

### Introduction & Motivation

The motivation is clear and the problem is genuinely important. The paper correctly situates itself at the intersection of interpretability and feature alignment. The contributions are enumerated clearly.

**Key concern — scope of the M-ambiguity argument:** The paper claims that "HiResCAMs are not uniquely determined, allowing an arbitrary spurious shift by a common matrix M." This is mathematically correct as stated (Theorem 3.2), but the practical framing is potentially misleading. For a specific trained model evaluated on a specific input, the gradient ∇_{A_j} f_c is uniquely determined by the network weights and the input; there is no operational ambiguity in computing the HiResCAM. The ambiguity arises when asking: "among all models in the equivalence class that produce the same prediction, which attribution is correct?" The paper conflates this philosophical ambiguity with a practical failure of HiResCAM. Figure 1 illustrates two different matrices M yielding different explanations, but it does not clarify that these would correspond to two *different* models (or two different parameterizations). This conceptual imprecision weakens the theoretical motivation somewhat.

---

### Section 2: Preliminaries

The setup is clean and the notation is well-defined. The assumption that the classifier h is a single linear layer (Eq. 1) is correctly flagged as a restriction. However, the paper would benefit from noting earlier that the architecture modifications described in Appendix C (removing BatchNorm, ReLU, downsampling) are *required* for the HiResCAM equality (Eq. 3) to hold exactly. This is a non-trivial constraint.

---

### Section 3: ContrastiveCAMs

**Proposition 3.1** and **Theorem 3.2** are correct and straightforward. The proofs in Appendix A are complete.

**Theorem 3.5** (M-invariance of ContrastiveCAMs) is also correct and the proof is clean. The fix — taking differences across classes — is natural and elegant.

**Definition 3.3 vs. 3.4:** Definition 3.3 gives C−1 pairwise maps (one for each non-target class), while Definition 3.4 reconstructs a single map. It is not made explicit how the pairwise maps in Definition 3.3 are displayed (which pairs? all of them?), nor how they are selected in Figure 2. The paper would benefit from clarifying the procedure for selecting visualizations in practice.

**Table 1:** The redundancy metric γ = ‖R‖_F / ‖CAM^{HiRes}_{ct}‖_F is reported, but R is described as "−1/C · ΣC_{c=1} CAM^{HiRes}_c." This is the mean HiResCAM, not a redundancy in the usual sense. The connection to the M-shift is not explained clearly in the text. Furthermore, the PASCAL VOC entry shows γ = "−1," which appears to be a parsing/formatting artifact and should be addressed.

**Class-versus-class explanations:** The qualitative analysis in Figure 2 is interesting but remains qualitative. The claim that "ContrastiveCAMs reveal circumstances wherein regions that contribute towards prediction are hidden by HiResCAMs" would benefit from a quantitative evaluation (e.g., localization metrics on a dataset with known ground-truth regions).

---

### Section 4: Learning with ContrastiveCAMs

**Proposition 4.1** (Correctness of ContrastiveCAMs) is well-proven. The reformulation of softmax as a direct function of ContrastiveCAMs and bias (Eq. 11) is clean and provides a useful computational basis for what follows.

**The bias-free assumption:** The paper requires setting b=0 in the linear classifier h to enable the precise disassociation of core and non-core contributions (Eq. 12–13). This is a significant constraint: removing the bias reduces model capacity. The ablation "CE w/ Arch" in Table 2 partially accounts for this, but the specific effect of removing the bias (separate from removing BatchNorm, ReLU, and downsampling) is not isolated. This matters because the bias removal is theoretically motivated, not empirically demonstrated as harmless.

**Proposition 4.2:** The decomposition of cross-entropy into core and non-core contributions (Eq. 12) is correct under the bias-free assumption, and the insight that CE "does not inherently favor using the core or non-core regions" is a genuine theoretical contribution.

**Definition 4.5 (CFCE):** The loss replaces the non-core contribution CAM^{Cntrst}_{(ct,c)} with its absolute value |CAM^{Cntrst}_{(ct,c)}| for non-core regions. This penalizes any signal (positive or negative) from non-core regions. The justification for using |·| rather than, e.g., a squared penalty or a direct clipping to zero is not discussed. This design choice may have significant effects on training dynamics.

**Theorem 4.6 (Consistency of CFCE):** This is the most important theoretical claim and deserves more scrutiny.

- The proof relies on "sufficiently expressive F" and "realizability of R*_{CCRM}" (stated below Eq. 58). These are strong assumptions that are not verified for the ResNet-50 architecture used in experiments.
- The key step in Eq. (58)→(59): the bound shown is a lower bound on the inf of the ratio, not an exact characterization of when R*_{CFCE} is achieved. The implication "R_{CFCE}(f_n) → R*_{CFCE} ⟺ inf_f exp(...)... ∀c" (Eq. 59) is stated without justification of when the bound is tight.
- The argument about the denominator (Eqs. 61–63): "By convexity of exp, f* ≥ sup_f exp(H ⊙ CAM^{Cntrst}_{(ct,c)})" (Eq. 62) is not a standard use of convexity and requires further justification.
- Ultimately, the proof establishes that in the realizable setting, a minimizer of R_{CFCE} achieves zero CCRM risk — which essentially means "if CFCE is minimized perfectly, non-core contributions go to zero." This is largely tautological given the loss design, and the "proof" of Theorem 4.6 is not rigorous enough to constitute a formal guarantee.

**Definition 4.7 (Regularized CFCE):** The KL-divergence regularization is motivated but the hyperparameters λ1=50, λ2=10³, λ3=10 (from Appendix C) are very large and highly sensitive. No sensitivity analysis is provided. The claim that these were chosen to "mitigate reward-hacking" is opaque.

---

### Section 5: Experiments

**Architecture modifications (Appendix C):** The paper makes three changes to ResNet-50: (i) removing the final downsampling (stride 2→1), (ii) removing the bias in h, (iii) removing BatchNorm & ReLU in the final block. These are non-trivial modifications that increase feature map resolution and change training dynamics. The "CE w/ Arch" baseline correctly controls for these, but the interaction between modifications is not ablated. For example, does CFCE provide gains on the standard unmodified ResNet-50? This would establish whether the loss itself (vs. the arch changes) is responsible for improvements.

**Hard-ImageNet (Table 2):**
- The CFCE model drops 3.72 pp in accuracy vs. baseline CE (94.25% → 90.53%). This is non-trivial. The paper mentions this as "at the cost of some un-ablated performance" but does not investigate the accuracy-alignment trade-off curve. Is there a value of the non-core suppression that preserves accuracy while improving alignment?
- The ContrastiveCAM IoU metric (89.22% for CFCE, 93.39% for CFCE+KL) is only reported for core-focused models, not for baselines. This makes it impossible to assess whether the ContrastiveCAM visualization itself (vs. the CFCE training) is responsible for the apparent improvements. GradCAM IoU is 18.44% for CE and 18.88% for CFCE (negligible improvement) — why is the ContrastiveCAM IoU so dramatically different and only reported for CFCE models?
- CORM, DFR, and CORM+DFR baselines do not show ContrastiveCAM IoU. This asymmetry in evaluation weakens the comparison.

**Oxford IIIT-Pets (Table 3):**
- CFCE with GT masks achieves strong IoU improvement (83% → 94% valid binary IoU with CFCE+KL), with minimal accuracy cost (99.40% → 99.32%). This is the most compelling result.
- However, the class imbalance (4978 dogs vs. 2371 cats) is acknowledged but not addressed. The binary accuracy of ~99% is near-ceiling, making it hard to detect accuracy degradation.
- The SAM-generated masks achieve 84% IoU vs. 93% with GT masks. The gap (9 pp) could be significant in practice; what are the failure modes of SAM masks on this dataset?

**Pascal VOC (multilabel, Section 5.3):**
- CFBCE achieves a Pareto improvement (+1.07 pp AP, +37.57 pp IoU) over CE. This is a strong result.
- However, the downstream segmentation results are presented as a bar chart (Figure referenced but not fully reproduced in the parsed text), which limits detailed assessment.

**Computational cost:** Computing ContrastiveCAMs during training requires backpropagating through the network per class pair. For Hard-ImageNet with many classes, or for scaling to ImageNet-scale, this could be prohibitively expensive. The paper does not report training time, GPU memory, or any computational comparison with baselines.

**Missing baseline:** Ismail et al. (2021), "Improving deep learning interpretability by saliency guided training," is cited in the related work but not included as a baseline. This is the most directly comparable prior work on interpretability-guided training.

---

### Writing & Clarity

The paper's structure is somewhat non-linear in the parsed version (Sections appear out of order: e.g., Section 5.1 appears before Section 4.2). This is likely a parser artifact. Setting aside formatting artifacts:

- The paper does not provide pseudocode or an algorithm box for CFCE training, which would significantly aid reproducibility.
- The phrase "mitigate reward-hacking" (Appendix C) to describe hyperparameter selection is unusual and unexplained.
- The PASCAL VOC γ value of "−1" in Table 1 appears to be a parser artifact or a genuine error that should be addressed.

---

### Limitations & Broader Impact

The paper acknowledges scope limitations (ConvNets only, single linear classifier) in the title and introduction. However, several limitations are not discussed:

1. **Mask availability:** The method requires per-sample segmentation masks (or at least bounding boxes). This is a strong supervision requirement that limits applicability to datasets without such annotations.
2. **Computational overhead:** Not addressed.
3. **Accuracy-alignment trade-off:** A 3-4% accuracy drop may be unacceptable in safety-critical applications (the very applications cited in the introduction — medical imaging, self-driving).
4. **Potential for misuse of "non-core suppression":** If core/non-core masks are imperfectly specified, CFCE may suppress genuinely informative features.

---

### Overall Assessment

The paper makes a genuine theoretical contribution in identifying and fixing the M-ambiguity in HiResCAMs, and proposes CFCE as a principled extension of cross-entropy for feature alignment. The experimental results on Hard-ImageNet and Oxford Pets are encouraging, with dramatically improved saliency alignment. However, several significant concerns temper enthusiasm: (1) the practical significance of the M-ambiguity is overclaimed — for a fixed model, HiResCAMs are deterministic; the ambiguity is about equivalence classes of models, not about the map itself; (2) Theorem 4.6's proof contains non-rigorous steps and relies on strong realizability assumptions; (3) the evaluation is asymmetric — ContrastiveCAM IoU is only reported for CFCE models, making it impossible to disentangle the contribution of the visualization method vs. the training loss; (4) the consistent 3-4% accuracy drop on Hard-ImageNet is not adequately analyzed; (5) computational cost is not reported; and (6) the most comparable prior method (Ismail et al. 2021) is not included as a baseline. The work is interesting and directionally sound, but in its current form falls short of the bar for ICLR acceptance without significant revisions addressing the theoretical rigor of Theorem 4.6, the evaluation asymmetry, and a more careful empirical analysis of accuracy-alignment trade-offs.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a theoretical limitation in HiResCAM, proving that class activation maps are not uniquely determined due to spurious shifts allowed by the softmax operation. To address this, the authors propose ContrastiveCAMs, which remove this redundancy, and introduce Core-Focused Cross-Entropy (CFCE) to optimize feature alignment by penalizing reliance on non-core regions. Experimental results on Hard-ImageNet and Oxford-IIIT Pets suggest that this method produces more faithful attention maps and reduces shortcut learning compared to standard cross-entropy.

### Strengths
1.  **Theoretical Insight into Attribution Ambiguity:** The paper provides a rigorous mathematical proof (Theorem 3.2) demonstrating that HiResCAMs admit arbitrary spurious shifts that do not affect probability predictions. This clarifies a subtle but significant issue in interpretability literature where explanations might be mathematically valid but semantically misleading.
2.  **Alignment-Driven Training Objective:** The core contribution linking interpretability metrics to the training loss (Definition 4.5, CFCE) is conceptually sound. By explicitly penalizing non-core regions using ContrastiveCAMs, the method directly targets the problem of shortcut learning (spurious correlations) rather than treating interpretability as purely a post-hoc analysis tool.
3.  **Targeted Evaluation on Hard-ImageNet:** The use of Hard-ImageNet (Moayeri et al., 2022) is highly appropriate for the proposed method. This dataset specifically isolates classes known to rely on spurious features, allowing the authors to demonstrate the method's efficacy in improving robustness and feature alignment where standard models typically fail (Table 2).

### Weaknesses
1.  **Architectural Constraints Limit Generalizability:** The method requires specific modifications to standard architectures to satisfy the HiResCAM assumptions, such as removing the bias vector, changing the final downsampling stride to $(1,1)$, and neutralizing BatchNorm/ReLU in the final block (Appendix C.1). Standard pre-trained backbones (like ResNet-50) cannot be used "off-the-shelf" without these changes, which restricts the practical applicability of the proposed loss function.
2.  **Dependence on High-Quality Core Masks:** While the paper demonstrates resilience to approximate masks (SAM, BBOX), the theoretical formulation and consistency (Theorem 4.6) rely on a binary mask $H$ defining core regions. In real-world scenarios where ground-truth core regions are unavailable or coarse, the performance of CFCE relies heavily on the quality of segmentation proxies, which are not always perfect.
3.  **Accuracy vs. Alignment Trade-off:** Table 2 indicates a noticeable drop in raw accuracy for CFCE on Hard-ImageNet (90.53%) compared to standard Cross-Entropy (94.25%), despite better performance under core-region ablation. The paper does not sufficiently discuss whether this accuracy trade-off is an acceptable cost for improved alignment in safety-critical applications without further benchmarking against modern robust learning methods.

### Novelty & Significance
The paper offers moderate novelty; while using saliency maps to guide training has precedent, the specific theoretical derivation connecting HiResCAM ambiguity to the construction of a contrastive loss is a novel contribution. It bridges the gap between post-hoc explanations and training objectives. The significance lies in its potential to mitigate shortcut learning, a persistent issue in deep vision. However, the requirement for significant architectural modification to the final layers prevents it from immediately replacing standard training regimes in practical deployment contexts.

### Suggestions for Improvement
1.  **Abandon Architecture Hacks:** Investigate whether the proposed loss function and theoretical guarantees can hold for standard ResNet architectures with their bias and BatchNorm layers intact. Alternatively, frame the architectural changes as a distinct "interpretability-friendly" architectural design alongside the loss function, rather than a strict requirement.
2.  **Expand Baseline Comparisons:** Include comparisons with more recent robustness techniques (e.g., IRM, SGLD, or other shortcut-learning corrections specifically designed for classification) rather than only CAM-based baselines, to contextualize the accuracy-robustness trade-off more accurately.
3.  **Clarify Mask Sensitivity:** Provide a more detailed analysis of how the performance of CFCE degrades as the quality of the core-region masks ($H$) decreases (e.g., adding noise to masks or using looser bounding boxes) to better inform users about the method's robustness to supervision quality.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **OOD Generalization Benchmarks:** Evaluate on standard spurious correlation datasets (e.g., Waterbirds, CelebA) to verify if alignment translates to robustness, as IoU on In-Distribution data is insufficient for ICLR robustness claims.
2. **Architectural Ablations:** Isolate the impact of removing bias, BN, and downsampling layers to ensure improvements stem from CFCE loss rather than architectural inductive biases or capacity reduction.
3. **Scalability & Overhead:** Measure training time and memory costs of computing pairwise ContrastiveCAMs within the loss loop for large-scale datasets (e.g., full ImageNet) compared to standard cross-entropy.
4. **Mask-Free Training:** Demonstrate performance when core masks are unavailable or generated purely from model saliency (bootstrapping), as relying on GT masks limits practical utility and conflates segmentation with classification.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical Validity of HiResCAM Ambiguity:** Clarify if the "shift M" affects the gradient-based CAM definition itself or only the logit decomposition, as gradients are typically bias-invariant for a fixed model.
2. **Accuracy-Alignment Trade-off:** Provide a Pareto analysis showing the cost of alignment on standard accuracy across diverse datasets, quantifying the performance drop observed in Table 2.
3. **Mask Noise Sensitivity:** Analyze how performance degrades when training masks contain errors or are coarse, quantifying robustness to supervision noise which is inevitable in real-world scenarios.
4. **Gradient Flow Verification:** Analyze gradient magnitudes in core vs. non-core regions during training to empirically confirm the loss actively suppresses non-core gradients as claimed.

### Visualizations & Case Studies
1. **Failure Modes on Mask Errors:** Visualize cases where incorrect mask supervision leads the model to ignore discriminative features, exposing the risk of over-constraint.
2. **OOD Generalization Examples:** Show side-by-side predictions on spurious correlation test sets (e.g., background changed) to validate robustness claims visually.
3. **HiResCAM vs. ContrastiveCAM on Fixed Model:** Show that for a *single trained model*, the claimed ambiguity exists or is merely a theoretical artifact of equivalent model classes.

### Obvious Next Steps
1. **Decouple from GT Masks:** Develop a mechanism to generate core masks dynamically during training without oracle supervision to enable unsupervised feature alignment.
2. **Extend to Vision Transformers:** Validate if the method and theoretical claims hold for attention-based architectures, not just ConvNets, given the prevalence of ViTs.
3. **Downstream Task Validation:** Evaluate features on detection or retrieval tasks to prove learned representations are more transferable, not just aligned.

# Final Consolidated Review
## Summary

This paper identifies a theoretical limitation of HiResCAM explanations: they are not uniquely determined because softmax invariance allows arbitrary matrix shifts that preserve predictions. The authors propose ContrastiveCAMs, which remove this redundancy by computing class-versus-class differences, and introduce Core-Focused Cross-Entropy (CFCE), a loss modification that penalizes contributions from non-core regions to improve feature alignment. Experiments on Hard-ImageNet and Oxford-IIIT Pets demonstrate improved attention alignment with core regions under ablation tests.

## Strengths

- **Theoretical identification of HiResCAM ambiguity:** Theorem 3.2 correctly proves that HiResCAMs admit arbitrary spurious shifts by a common matrix M while producing identical prediction probabilities. This is a valid theoretical observation—multiple CAM explanations can correspond to the same softmax output, and taking class-versus-class differences (ContrastiveCAM) elegantly eliminates this ambiguity.

- **Principled connection between interpretability and training:** The paper derives a training objective (CFCE) directly from the theoretical properties of ContrastiveCAMs, establishing that cross-entropy does not inherently favor core regions. Proposition 4.2's decomposition of cross-entropy into core and non-core contributions is a genuine insight that motivates the proposed loss.

- **Strong empirical alignment gains:** On Hard-ImageNet, CFCE models show dramatic improvements in core-region IoU (from ~18% GradCAM IoU to 51.52% ContrastiveCAM IoU with KL regularization) and substantially better performance under core-region ablation (accuracy drops to 41.78% vs. 75.94% under gray masking, indicating the model relies more on core features).

- **Demonstration of mask flexibility:** Table 3 shows that SAM-generated masks and even bounding boxes achieve competitive alignment (IoU ~83-84% binary, comparable to ground-truth masks at 94%), suggesting the method is robust to approximate supervision.

## Weaknesses

- **Asymmetric evaluation of ContrastiveCAM:** Table 2 reports ContrastiveCAM IoU only for CFCE-trained models, not for baseline methods (Cross-Entropy, CORM, DFR). This makes it impossible to determine whether the improved alignment scores reflect the training method or inherent properties of ContrastiveCAM visualizations. GradCAM IoU shows negligible improvement (18.44% → 18.88%), so readers cannot assess whether ContrastiveCAM itself provides better alignment or merely produces different numerical values.

- **Accuracy-alignment trade-off insufficiently analyzed:** On Hard-ImageNet, CFCE drops accuracy by 3.72 percentage points (94.25% → 90.53%) compared to cross-entropy. While the paper notes this as "cost of some un-ablated performance," there is no Pareto analysis of the accuracy-alignment frontier, no investigation of whether partial suppression achieves better trade-offs, and no comparison to robust learning methods that might achieve alignment without accuracy loss.

- **Architecture modifications create practical barriers:** The method requires removing the final downsampling layer, final bias, and BatchNorm/ReLU from the final convolutional block (Appendix C.1). These changes mean standard pre-trained weights cannot be used directly, limiting applicability. While "CE w/ Arch" controls for these changes, the specific contribution of each modification (bias removal vs. BatchNorm removal vs. stride change) is not ablated.

- **Computational overhead unreported:** ContrastiveCAM requires computing gradients for each class pair within the training loop. For datasets with many classes, this could be prohibitive. The paper provides no training time, memory footprint, or scaling analysis.

- **Dependence on ground-truth core masks:** Despite showing robustness to approximate masks (SAM, BBOX), the method fundamentally requires per-sample region annotations. Many datasets lack such supervision, and the paper does not evaluate scenarios where masks are unavailable or contain systematic errors (e.g., mislabeled regions).

## Nice-to-Haves

- **Evaluation on OOD spurious correlation benchmarks:** Testing on Waterbirds or CelebA would validate whether improved alignment translates to robustness against spurious correlations in standard benchmark settings.

- **Mask noise sensitivity analysis:** A systematic study of how performance degrades with noisy or partially incorrect masks would inform practical deployment.

- **Comparison to saliency-guided training baselines:** Ismail et al. (2021) is cited in related work but not included as a baseline, despite being directly comparable prior work on interpretability-guided training.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"HiResCAM is deterministic for fixed models":** The harsh critic argues that for a fixed trained model, the gradient ∇_{A_j} f_c is uniquely determined, so HiResCAM is deterministic. This misunderstands the paper's theoretical point: the ambiguity is about which CAM correctly *explains* the prediction, not about computational determinism. Multiple CAMs differing by M produce identical softmax outputs; ContrastiveCAM correctly identifies that only differences between class CAMs matter for explanation. The theoretical contribution is valid.

- **"Theorem 4.6 proof is tautological":** The critic claims the consistency proof is "largely tautological given the loss design." While the proof could be more rigorous in its realizability assumptions, it correctly establishes that CFCE minimization drives non-core contributions toward zero, consistent with the CCRM objective. The proof structure is standard for surrogate loss consistency.

- **"PASCAL VOC γ = −1 is an error":** This appears to be a parsing artifact. The paper is properly focused on three main datasets, and this formatting issue does not affect the core contributions.

- **"Missing pseudocode for reproducibility":** The paper provides complete mathematical definitions and specifies hyperparameters in Appendix C. While pseudocode would be helpful, the method is sufficiently specified for reproduction.

## Novel Insights

The paper reveals an elegant symmetry: softmax's invariance to uniform shifts, normally considered a harmless property, creates a fundamental ambiguity in gradient-based attribution methods. The ContrastiveCAM fix (taking class differences) is mathematically natural and has the additional benefit of revealing class-versus-class attribution patterns that single-class CAMs cannot provide. This dual-purpose—both fixing theoretical ambiguity and enabling granular multi-class explanations—is a genuine conceptual advance that could inform future work on attribution methods beyond ConvNets.

## Suggestions

1. **Report ContrastiveCAM IoU for all baselines:** Add this column to Table 2 to enable fair comparison of alignment metrics across training methods.

2. **Add accuracy-alignment Pareto analysis:** Vary the suppression strength (e.g., through a scaling coefficient on the non-core penalty term) and plot the accuracy vs. IoU frontier to show users how to trade off these objectives.

3. **Isolate architecture modification effects:** Run CFCE on standard ResNet-50 (with bias, BatchNorm, and original stride) to quantify how much performance comes from the loss versus the architectural changes.

4. **Report training time and memory overhead:** For Hard-ImageNet training, provide wall-clock time and peak GPU memory for CFCE versus cross-entropy to inform practitioners about computational costs.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
