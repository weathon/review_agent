=== CALIBRATION EXAMPLE 9 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is accurate. The abstract correctly identifies the two main contributions: ContrastiveCAMs (invariance to spurious HiResCAM shifts) and Core-Focused Cross-Entropy (CFCE). However, the abstract frames Theorem 3.2's result more dramatically than warranted — the non-uniqueness of HiResCAMs due to softmax invariance is essentially a consequence of a classical property of the softmax function (shift-invariance), and this is only "new" in the sense of being applied specifically to HiResCAM. The abstract should be more tempered in language: "completely corrupt HiResCAM explanations" is an extreme claim.

The abstract mentions Hard-ImageNet and Oxford-IIIT Pets but omits PASCAL VOC, which is also reported in the paper.

---

### Introduction & Motivation

The motivation is generally well-stated, but the framing of Theorem 3.2 as a significant theoretical discovery warrants scrutiny. The softmax shift-invariance property (Proposition 3.1) is standard textbook material. The extension to spatial maps (HiResCAMs) in Theorem 3.2 is a natural application of this property, not a deep result. The introduction somewhat oversells this observation as a theoretical limitation "we prove."

The contributions list is clear and the work on connecting interpretability to training (CFCE) is the more genuinely novel piece. The introduction would benefit from spending more space motivating *why* contrastive class-versus-class explanations are specifically useful, rather than primarily dwelling on the non-uniqueness observation.

---

### Preliminaries / Setup (Section 2)

The restriction to single-layer classifiers (Eq. 1) is presented as a natural consequence of modern architecture design, citing VGG, ResNet, ViT, etc. This is reasonable but glosses over architectural nuances: ViT's CLS token processing involves multi-layer MLP heads in many implementations, and GAP + single linear layer is only one variant. The scope restriction is acceptable but should be stated as a restriction, not a universally valid assumption.

---

### ContrastiveCAMs (Section 3)

**Theorem 3.2 — The main theoretical claim:**
The proof (Appendix A) is correct but mathematically shallow. The core insight is simply: because softmax is invariant to adding a constant to all logits, and because adding a constant to each spatial map M adds a constant to each class logit (via summation), the CAMs are non-unique up to M. This is a straightforward two-step argument. The "infinitely many HiResCAMs" framing is technically correct but somewhat misleading in practice — most naturally trained models would not have a large M shift, so calling this a "failure to guarantee faithful interpretation" needs more empirical grounding.

**Definition 3.3/3.4 (ContrastiveCAM and Reconstructed ContrastiveCAM):**
ContrastiveCAMs are essentially *pairwise differences* of class-specific activation maps. This is a natural and elegant construction but not entirely novel — the idea of contrastive explanations ("why class A rather than class B?") has been explored in the XAI literature. The paper does not adequately survey or distinguish itself from prior contrastive explanation methods (e.g., work on contrastive saliency, TCAV, or why-not explanations). This gap in related work weakens the originality claim.

**Theorem 3.5 (M-invariance):**
The proof is correct and straightforward (M cancels in pairwise differences). This is a nice but unsurprising property of subtraction.

**Redundancy metric γ (Table 1):**
Table 1 shows γ values (0.201 for Hard-ImageNet, 0.367 for Pets) but provides no interpretation of what constitutes a "large" redundancy. Is γ = 0.2 worrisome or negligible? The paper does not provide a theoretical or empirical baseline for this quantity, making Table 1's purpose primarily illustrative rather than diagnostic.

**Figure 2:**
Only a single illustrative image is shown for the class-versus-class comparison. This is qualitative and anecdotal. The paper should provide more systematic evidence (e.g., averaged localization metrics across the test set) that ContrastiveCAMs expose hidden regions consistently, not just in cherry-picked examples.

---

### Learning with ContrastiveCAMs (Section 4)

**Proposition 4.1 and bias-free assumption:**
A crucial dependency appears here: results in Section 4 require zeroing out the bias vector of the classifier h. The paper treats this as a minor note ("By zero-ing the final bias vector... for h only"), but this is a genuine constraint that reduces model expressiveness (bias terms help with class-frequency imbalance and calibration). The theoretical results technically only hold under this restriction, yet the experiments appear to train models with this constraint applied. The practical impact of removing the bias in h needs careful discussion — in imbalanced settings (e.g., Oxford-IIIT Pets has 4978 dogs vs. 2371 cats, acknowledged in Section 5.2), this could be particularly consequential.

**Definition 4.5 (Core-Focused Cross-Entropy):**
The key modification is replacing `(1−H) ⊙ CAM_c` with `(1−H) ⊙ |CAM_c|` in the loss. This penalizes absolute magnitude of non-core contributions, which is the intuitive idea. However, computing CFCE requires backpropagating through the CAM computation (gradients of activations with respect to inputs), i.e., second-order gradients. This is computationally significantly more expensive than standard cross-entropy. The paper **never mentions** the computational overhead of this approach — neither training time comparisons nor memory requirements are reported. For ICLR, this omission is substantial.

**Theorem 4.6 (Consistency of CFCE):**
The proof proceeds by showing that minimizing R_CFCE to its optimum implies satisfying the CCRM constraint. The argument relies on the realizability assumption ("sufficiently expressive F") and convexity of exp. The jump from Eq. (58) to Eq. (59) implicitly assumes that infimum and supremum over *different* functions can be achieved simultaneously — this deserves more careful treatment. The realizability assumption is also very strong and unrealistic in practice. The theorem is useful as a theoretical sanity check but should not be presented as a guarantee for practical finite-dimensional models.

**Remark 4.3 and Proposition 4.2:**
These are clear and helpful for decomposing CE loss into core/non-core contributions. The observation that CE "does not inherently favor using the core or non-core regions" is the key negative result motivating CFCE — this is well-stated.

**Divergence Regularization (Definition 4.7):**
Three additional hyperparameters (λ1, λ2, λ3) are introduced. No ablation over these values is provided anywhere in the paper, making it unclear how sensitive the method is to their choice or how they were set in experiments.

---

### Experiments & Results (Section 5)

**General Setup:**
All experiments use ResNet-50 with ImageNet-pretrained weights. Testing on a single architecture is a significant limitation for ICLR — it is unclear whether the findings generalize to other modern backbones (ConvNeXt, ViT, EfficientNet), especially since the theoretical framework explicitly mentions these architectures as motivating examples.

**Hard-ImageNet (Table 2):**
- The CFCE results show striking improvements in accuracy under core-region ablation (41.78% vs 75.94% for CE, Gray Mask setting), correctly interpreted as the model relying more on core features. This is the paper's strongest result.
- **Baseline fairness concern**: The IoU values for baseline methods (CE, CORM, DFR) are reported using GradCAM, while core-focused models report *both* GradCAM IoU and ContrastiveCAM IoU. This is explicitly noted as being done "for consistency," but it introduces a systematic advantage for the proposed models. The paper acknowledges GradCAM provides "unfaithful explanations," so baselines should either be re-evaluated with a fairer metric or this disparity prominently flagged.
- CORM and DFR baselines show minimal improvement over plain CE (20.43% vs 18.44% IoU), suggesting the competition is not strong. More recent feature alignment methods (e.g., Ismail et al., 2021, which is cited) are not compared against, weakening the experimental section.
- No standard deviations are reported for the (non-"w/ Arch") baselines, only for the proposed methods. This asymmetry prevents statistical significance assessment.
- The ~4% accuracy drop (94.25% → 90.53%) for CFCE vs CE under full-image classification is non-trivial and its practical significance in deployment scenarios is not discussed.

**Oxford IIIT-Pets (Table 3):**
- Classification accuracy is near ceiling (>99% for most models), making these numbers essentially uninformative. All the meaningful signal is in IoU metrics.
- KL regularization (CFCE+KL with GT masks) achieves 92.72% validation IoU vs 78.37% for CE — a very large improvement. This is impressive.
- The SAM-based mask results (83.95% IoU) approach the GT-mask performance (82.92% without KL), which effectively demonstrates that coarse auto-generated masks are nearly as good. This is a valuable practical result.
- KL regularization is explicitly noted as inappropriate for bounding box masks, which is a useful practical caveat, but no ablation shows what happens when it *is* applied (to quantify the degradation).
- The class imbalance (4978 dogs vs 2371 cats) is noted but no correction is applied. It's unclear whether the bias-free classifier constraint worsens this imbalance issue.

**PASCAL VOC (Section 5.3):**
- The classification results table shows a modest improvement (88.39% AP for CFCE vs 87.32% for CE) alongside large IoU gains (82.07% vs 44.50%). This Pareto improvement claim is valid.
- **Critical omission**: The segmentation results figure appears truncated in the parsed paper (showing only axis labels "0, 20, 40, 60, 80" with no actual curve descriptions beyond a raw axis dump). Even accounting for PDF parsing artifacts, the discussion of "improvements in IoU performance of core-focused backbones on downstream segmentation" is insufficiently detailed — specific numbers for segmentation IoU are absent from the main text.
- The multilabel extension (CFBCE, Proposition B.1) is relegated to the appendix without full description in the main text, making the PASCAL VOC results harder to interpret.

---

### Writing & Clarity

The paper has significant flow issues due to apparent PDF parsing artifacts causing section order to be scrambled (Section 4.2 appears after the Hard-ImageNet results section at line 617, Section 4 begins at line 429 but continues after Section 5.1). Even accounting for parsing, some cross-references (e.g., "supplemental formulations deferred to Appendix B") make the main narrative feel incomplete. The discussion of scale-sensitivity (Section 4.1) would benefit from a worked example or figure rather than purely verbal description.

The notation section (Section 2) is well-organized. The definitions are clearly numbered and stated.

---

### Limitations & Broader Impact

The paper contains no formal limitations section. Key limitations that should be acknowledged:

1. **Mask supervision requirement**: The method requires binary segmentation masks (or approximate proxies) at training time. This is substantially stronger supervision than standard classification. The paper frames SAM-generated masks as "approximate" supervision, but using a strong foundation model (SAM) for pseudo-labeling is still a form of privileged information that competitors (CE, CORM, DFR) do not use.

2. **Computational cost**: Training with CFCE requires computing gradients through CAM computations (second-order gradients). This is not discussed.

3. **Single architecture**: Generalization beyond ResNet-50 is unvalidated.

4. **Bias-free classifier constraint**: The practical impact of removing bias terms in the classification head is uncharacterized.

5. **No discussion of robustness to mask quality degradation**: Only GT, SAM, and BBOX masks are tested. Systematic degradation experiments are absent.

---

### Overall Assessment

This paper makes a genuine contribution at the intersection of interpretability and feature alignment in CNNs. The theoretical framework is clean: the non-uniqueness of HiResCAMs follows naturally from softmax shift-invariance, and ContrastiveCAMs provide an elegant invariant alternative with the bonus of class-versus-class granularity. The Core-Focused Cross-Entropy is the more substantive contribution, with reasonable theoretical guarantees under strong assumptions. The experimental results on Hard-ImageNet are convincing, with dramatic improvements in core-region IoU and ablation sensitivity. However, the work has several weaknesses that must be addressed for ICLR acceptance: (1) the computational cost of CFCE (second-order gradients) is never discussed or measured; (2) all experiments use a single architecture (ResNet-50); (3) baseline comparisons use different evaluation metrics (GradCAM IoU for baselines vs. ContrastiveCAM IoU for proposed methods), which is not fair; (4) the hyperparameter sensitivity of Regularized CFCE (three λ parameters) is not ablated; (5) the PASCAL VOC segmentation results are incompletely reported; and (6) the consistency theorem (Th. 4.6) has proof gaps and relies on unrealistically strong realizability assumptions. The paper sits at the borderline: the core ideas are interesting and empirically demonstrated, but the experimental rigor and breadth fall below what ICLR typically requires for acceptance. A thorough revision addressing computational cost, multi-architecture evaluation, and fairer baseline comparisons would substantially strengthen the paper.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a theoretical limitation in HiResCAM, demonstrating that its spatial attributions are non-unique due to the shift invariance of the softmax function, which can introduce arbitrary spurious attribution patterns. To address this, the authors introduce ContrastiveCAMs, which are mathematically invariant to these shifts and provide class-versus-class explanations that reveal a tendency of convnets to rely on non-core (background) regions. Building on this insight, the paper proposes Core-Focused Cross-Entropy (CFCE), a modified training objective that uses binary core-region masks to suppress non-core activations, thereby improving feature alignment and robustness to spurious correlations.

### Strengths
1. **Clear Theoretical Foundation:** The identification of softmax-induced non-uniqueness in HiResCAM (Theorem 3.2) and the subsequent derivation of ContrastiveCAMs (Definition 3.3/3.5) are mathematically sound and well-motivated. The proofs are concise, and the $M$-invariance property provides a principled correction to a known limitation in gradient-based attribution methods.
2. **Direct Bridge Between Post-Hoc XAI and Representation Learning:** Rather than treating interpretability solely as a post-training diagnostic, the paper leverages ContrastiveCAMs to design a novel training loss (CFCE, Definition 4.5). This aligns closely with ICLR's interest in mechanistic interpretability and shortcut mitigation, offering a concrete method to enforce feature alignment during optimization.
3. **Rigorous and Multi-Faceted Empirical Evaluation:** The experiments span three diverse datasets (Hard-ImageNet, Oxford-IIIT Pets, PASCAL VOC) and employ multiple alignment/robustness metrics (IoU, RFS, core ablation, downstream segmentation). The inclusion of Table 1 (quantifying non-core contributions) and Table 2 (showing consistent robustness gains under ablation) demonstrates that the method effectively reduces background reliance.

### Weaknesses
1. **Confounding Architectural Modifications:** Section C.1 details significant backbone changes (removing final bias, BatchNorm, ReLU, and altering the downsampling stride to 1,1) to guarantee CAM faithfulness. However, the experimental results do not disentangle the impact of these architectural changes from the proposed CFCE loss. Without an ablation isolating the loss function from the architecture, it is unclear which component primarily drives the reported alignment improvements.
2. **Strong Dependency on High-Quality Masks Limits Practicality:** CFCE requires explicit binary masks $H$ during training. While the paper tests auto-generated (SAM) and bounding-box masks (Table 3), performance degrades and variance increases compared to ground-truth masks. The method lacks a clear scalability strategy for datasets where precise pixel-wise masks are unavailable, which constrains real-world applicability.
3. **Unquantified Accuracy-Alignment Trade-off:** The method intentionally sacrifices in-distribution accuracy (e.g., Hard-ImageNet drops from ~94.3% to ~90.5% in Table 2) to improve feature alignment. While this is a common robustness trade-off, the paper does not provide failure case analysis, confusion matrices, or OOD generalization benchmarks to justify whether the dropped predictions correspond to genuinely hard examples or if the model over-penalizes useful contextual features.
4. **Limited Empirical Validation of the Theoretical Shift Problem:** While Theorem 3.2 proves that an arbitrary shift $M$ is possible, the paper does not empirically quantify how frequently or severely this shift actually corrupts standard HiResCAM maps in trained ResNet-50 models. Demonstrating the practical magnitude of this issue compared to more common attribution artifacts (e.g., gradient saturation or noise) would strengthen the motivation for ContrastiveCAMs.

### Novelty & Significance
- **Novelty:** Moderate-High. The formal proof of softmax-induced non-uniqueness in CAM methods and the contrastive reformulation are novel theoretical contributions. CFCE extends existing saliency-guided training and region-suppression techniques by grounding them in a mathematically invariant attribution scheme, offering a fresh perspective on loss design for alignment.
- **Clarity:** High. The progression from theoretical limitation (Sec 3) to methodological proposal (Sec 4) and empirical validation (Sec 5) is logical. Notation is consistent, and the derivations in the appendices are thorough. The paper is well-structured for an audience familiar with CAMs and robust optimization.
- **Reproducibility:** High. The paper provides explicit training hyperparameters, optimizer settings, learning rate schedules, and detailed architectural modifications (Appendix C). Code and weights are promised under a permissive license, and the methodology is fully specified, enabling straightforward reproduction.
- **Significance:** High for ICLR's interpretability, robustness, and representation learning tracks. The paper directly addresses shortcut learning and proposes a principled, theoretically grounded method to enforce core-feature reliance. If the architectural confounder is resolved and mask dependency is better addressed, CFCE could become a standard reference for XAI-guided training in convnets.

### Suggestions for Improvement
1. **Conduct Architecture vs. Loss Ablations:** Run experiments comparing (a) Standard ResNet + Cross-Entropy, (b) Modified ResNet (Sec C.1) + Cross-Entropy, (c) Standard ResNet + CFCE, and (d) Modified ResNet + CFCE. This will isolate the true contribution of the loss function and clarify how much architectural faithfulness adjustments influence alignment metrics.
2. **Quantify the Practical Impact of Shift $M$:** Provide a metric or visual benchmark across the test set showing the divergence between original HiResCAM and ContrastiveCAM attributions. Reporting attribution stability scores or correlation metrics would empirically validate how often the spurious shift meaningfully misleads practitioners.
3. **Analyze the Accuracy Drop Rigorously:** Perform error analysis on the samples lost in-distribution when training with CFCE. Show confusion matrices or feature-space visualizations (e.g., UMAP/t-SNE) to demonstrate whether the model loses performance on genuinely ambiguous cases or unfairly penalizes useful semantic context.
4. **Expand Baseline Comparisons:** While CORM and DFR are included, they are primarily evaluation/post-training methods. Compare CFCE against modern training-time alignment/shortcuts mitigators such as LISA, GroupDRO, or feature-decoupling methods. This will contextualize CFCE's sample efficiency, training stability, and computational overhead.
5. **Provide Practical Guidelines for Weak Supervision:** Add a discussion on the reliability thresholds for $H$ masks. If SAM or bounding boxes are used, quantify the IoU or mask-fidelity requirements under which CFCE remains effective, and propose a fallback or adaptive weighting scheme to handle noisy mask approximations.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **ContrastiveCAM vs. HiResCAM in Loss:** Train using the proposed CFCE loss but substitute ContrastiveCAM with standard HiResCAM. Without this, the claim that ContrastiveCAM is necessary for stable optimization (due to shift invariance) is unsupported.
2. **Standard Architecture Performance:** Evaluate CFCE on unmodified ResNet-50 (with bias, BN, ReLU). The current method requires architectural surgery (Appendix C.1) to maintain theoretical guarantees; claiming general "ConvNet" improvement without testing standard architectures is misleading.
3. **Auxiliary Segmentation Baseline:** Compare against a standard multi-task learning baseline (Classification + Segmentation loss) using the same masks. This isolates whether the gain comes from the specific CFCE formulation or simply from supervising core regions via any loss.

### Deeper Analysis Needed (top 3-5 only)
1. **Empirical Evidence of Shift Ambiguity:** Quantify the variance in HiResCAM explanations across models with identical softmax outputs (e.g., via different seeds or logits shifts). Without showing this ambiguity manifests in practice, the theoretical motivation is weak.
2. **Clean Accuracy vs. Robustness Trade-off:** Analyze the significant drop in clean accuracy (e.g., Table 2: 94.25% -> 90.53%). Provide a Pareto analysis to determine if the robustness gain justifies the performance cost for general applications.
3. **Computational Overhead:** Report training time and memory costs. Calculating CAMs and their gradients during every training step introduces significant overhead compared to standard Cross-Entropy, which is unaddressed.

### Visualizations & Case Studies
1. **Failure Cases with Imperfect Masks:** Visualize cases where SAM/BBOX masks are inaccurate and CFCE suppresses relevant features. This exposes the method's sensitivity to mask quality.
2. **Shift Ambiguity Visualization:** Show two models with identical predictions but visibly different HiResCAMs due to the theoretical shift $M$, validating the core theoretical claim.
3. **Downstream Segmentation Qualitative Results:** Show side-by-side segmentation masks generated by CFCE vs. CE backbones to visually verify the claimed "feature alignment" improvement.

### Obvious Next Steps
1. **Architecture-Agnostic Formulation:** Adapt the method to work without removing bias/BN layers, as these modifications limit deployment in pre-trained models.
2. **Vision Transformer (ViT) Evaluation:** Extend experiments to ViTs, as the introduction cites them as relevant but experiments are restricted to ResNets.
3. **Mask-Free Weak Supervision:** Develop a mechanism to generate core masks dynamically during training to remove the dependency on external mask annotations (SAM/BBOX).

# Final Consolidated Review
## Summary

This paper identifies a theoretical limitation in HiResCAM: due to softmax shift-invariance, HiResCAM explanations are not uniquely determined and can admit arbitrary spatial shifts while preserving the same predictions. The authors propose ContrastiveCAMs, which achieve invariance to these spurious shifts by computing pairwise differences between class activation maps. Using the class-versus-class granularity of ContrastiveCAMs, they reveal that models often rely on non-core (background) regions, and propose Core-Focused Cross-Entropy (CFCE), a training objective that suppresses contributions from non-core regions to improve feature alignment.

## Strengths

- **Principled theoretical foundation:** Theorem 3.2 correctly identifies that HiResCAM explanations can be shifted by an arbitrary matrix M without changing predictions. While the underlying softmax shift-invariance property is well-known, its specific implication for HiResCAM—and the derivation of ContrastiveCAMs as an invariant alternative—is a clean theoretical contribution. The M-invariance proof (Theorem 3.5) is straightforward but valid.

- **Direct connection between interpretability and training:** The paper goes beyond post-hoc explanation by designing CFCE, a training loss grounded in ContrastiveCAM's theoretical properties. This bridges XAI and representation learning in a principled way, aligning with growing interest in shortcut mitigation and feature alignment.

- **Strong alignment improvements:** On Hard-ImageNet, CFCE dramatically improves accuracy under core-region ablation (e.g., 41.78% vs. 75.94% for cross-entropy under gray mask ablation), demonstrating that models genuinely shift reliance toward core features. IoU between predicted attention and ground-truth core regions improves substantially across datasets.

- **Practical applicability with approximate masks:** The finding that SAM-generated masks achieve comparable alignment to ground-truth masks (83.95% vs. 82.92% validation IoU on Oxford-IIIT Pets without KL) is practically valuable, showing the method can work without expensive pixel-level annotations.

## Weaknesses

- **Architectural modifications confound the loss contribution:** Appendix C.1 describes three significant backbone changes (removing final bias, BatchNorm, ReLU, and changing downsampling stride) required to maintain CAM faithfulness guarantees. All experiments apply these modifications alongside CFCE, making it impossible to isolate whether alignment improvements stem from the proposed loss or the architectural changes. The paper claims improvements for "ConvNets" broadly but validates only this specific modified architecture.

- **Computational overhead unreported:** CFCE requires computing gradients through the CAM computation at every training step, which involves second-order gradients through the feature maps. The paper provides no analysis of training time, memory requirements, or convergence behavior compared to standard cross-entropy. For a training-time intervention, this is a significant omission.

- **Limited empirical validation of the theoretical shift problem:** Theorem 3.2 establishes that an arbitrary shift M is mathematically possible, but the paper does not quantify how frequently or severely this ambiguity manifests in practice. A study showing divergence between HiResCAM and ContrastiveCAM attributions across trained models—or between models with identical softmax outputs but different logits—would substantiate that this is not merely a theoretical curiosity.

- **Accuracy-alignment trade-off undercharacterized:** Hard-ImageNet accuracy drops from 94.25% to 90.53% with CFCE. While the paper frames this as an expected trade-off for robustness, there is no analysis of which examples are misclassified or whether useful contextual features are unfairly penalized. Confusion matrices or failure-case analysis would clarify whether the sacrificed accuracy represents genuine shortcut removal or harmful over-constraint.

- **Incomplete baseline comparisons:** Table 2 evaluates baselines (CE, CORM, DFR) using GradCAM IoU but evaluates proposed methods using both GradCAM and ContrastiveCAM IoU. Since the paper argues GradCAM provides "unfaithful explanations," this asymmetry makes direct comparison difficult. Additionally, more recent training-time alignment methods (beyond CORM and DFR) are not included.

- **Single architecture tested:** All experiments use modified ResNet-50. The theoretical framework mentions ConvNeXt, ViT, and EfficientNet as motivating examples, yet no validation on these architectures is provided.

- **Missing related work on contrastive explanations:** The ContrastiveCAM formulation computes pairwise differences between class activation maps, which shares motivation with prior contrastive explanation methods. The paper does not survey or distinguish itself from this literature.

## Nice-to-Haves

- **Ablation testing CFCE with standard HiResCAM:** To validate that ContrastiveCAM's M-invariance is necessary for stable training (rather than just a theoretical curiosity), train with CFCE using standard HiResCAM and compare results.

- **Multi-task learning baseline:** Compare CFCE against a standard multi-task setup (classification + segmentation loss using the same masks) to isolate whether gains come from the specific CFCE formulation versus simply supervising core regions.

- **ViT or ConvNeXt evaluation:** Given the theoretical framework's scope, validation on at least one additional architecture would strengthen generalization claims.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Theorem 3.2 is just textbook softmax shift-invariance":** While the softmax property is indeed standard, the paper's specific application to HiResCAM spatial maps—and the derivation that this leads to non-unique explanations with arbitrary M shifts—is novel. The criticism that this is "mathematically shallow" misunderstands the contribution: the insight is the application, not the mathematical complexity.

- **"Abstract omits PASCAL VOC":** This is a minor omission that does not affect the paper's core contributions or validity.

- **"PASCAL VOC segmentation figure truncated":** While the parsed PDF shows formatting issues, the segmentation results are discussed in Section 5.3 with numerical improvements reported. The figure presentation is suboptimal but the results are present.

## Novel Insights

The redundancy ratio γ reported in Table 1 (~0.2 for Hard-ImageNet, ~0.37 for Pets) suggests that non-core contributions are not merely present but constitute a meaningful fraction of total attribution magnitude. This quantifies a previously qualitative observation about shortcut learning. Additionally, the finding that SAM-generated masks achieve comparable performance to ground-truth masks suggests that the method's sensitivity to mask quality has reasonable tolerance—useful for practitioners without annotation budgets.

## Suggestions

- **Add an ablation study:** Compare (a) standard ResNet-50 + cross-entropy, (b) modified architecture + cross-entropy, (c) standard ResNet-50 + CFCE, and (d) modified architecture + CFCE to isolate contributions.

- **Report computational costs:** Include training time per epoch and peak GPU memory for CFCE versus cross-entropy baseline.

- **Analyze the accuracy drop:** Provide per-class accuracy changes or confusion matrices to characterize what the model loses when shifting to core features.

- **Quantify the shift problem empirically:** Show examples or aggregate statistics comparing HiResCAM and ContrastiveCAM attributions across models to demonstrate the practical significance of M-invariance.

- **Add related work on contrastive explanations:** Briefly discuss how ContrastiveCAM relates to prior contrastive XAI methods to clarify novelty.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
