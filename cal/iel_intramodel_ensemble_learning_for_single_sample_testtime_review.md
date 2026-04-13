=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
## Summary
The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method that adapts an ensemble of pre-trained models using unlabeled data one sample at a time. IEL selects the ensemble member with highest confidence for the majority-voted class as a dynamic "teacher" and minimizes the cross-entropy between this soft target and all other members via backpropagation. Experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C demonstrate accuracy improvements over static ensemble baselines across most corruption types.

## Strengths
- **Practical single-sample setting:** The method operates with batch size 1, which is genuinely restrictive and relevant for real-world applications where batch statistics cannot be reliably estimated. The authors explicitly freeze BN layers to ensure adaptation comes from weight updates rather than batch statistics exploitation.
- **Heterogeneous ensemble support:** IEL does not require identical architectures for ensemble members, demonstrating flexibility with combinations of ResNet, VGG, MobileNet, ShuffleNet, and RepVGG models (Tables 1-2).
- **Dual improvement mechanism:** Unlike standard ensemble methods that only improve aggregate predictions, IEL shows that individual member models can improve during inference (Figure 2, Table 1), with some models gaining 15-20% accuracy on specific corruptions.
- **Comprehensive corruption coverage:** Evaluation across 15 corruption types on three datasets (CIFAR-10-C, CIFAR-100-C, ImageNet-C) provides a thorough assessment of the method's behavior across diverse distribution shifts.

## Weaknesses
- **No comparison to existing TTA methods:** The paper compares only against a static (unadapted) ensemble baseline. TENT, EATA, CoTTA, ROID, and TTT are all discussed in related work but none appear in experimental comparisons. Without comparing against single-model TTA methods configured for batch-size 1, it is impossible to determine whether IEL's gains exceed what existing methods would achieve under the same conditions. This is a critical omission for establishing contribution.

- **Cherry-picking by reporting best accuracy across epochs:** Tables 1-3 report "highest accuracy improvements over all epochs" rather than accuracy at a principled stopping point. The paper acknowledges accuracy can degrade below the static baseline at later epochs (Section 3.1, Figure 3), yet no termination criterion is provided. In practice, one cannot select the best epoch post-hoc without access to labels, making the reported numbers optimistic and not reproducible under realistic deployment.

- **Multi-epoch adaptation contradicts single-sample framing:** The method is motivated as fitting the "single sample" setting, yet experiments pass through 9,000+ samples per corruption type for up to 5 epochs (45,000 ordered samples total). This is fundamentally online mini-batch adaptation with batch size 1, not the restrictive "only one sample" setting described. The contrast with batch-based methods is therefore overstated.

- **Systematic failures on noise corruptions unexplained:** IEL produces negative results (catastrophic forgetting) on Gaussian Noise, Shot Noise, and Impulse Noise in CIFAR-10-C and CIFAR-100-C (Tables 1-2). The paper notes these failures but provides no analysis of why noise corruptions are problematic. Understanding failure modes is essential for practitioners to know when IEL should not be applied.

- **Core design choice unvalidated:** The central design decision—selecting the highest-confidence model for the majority class as teacher—receives no ablation. Why not use the average ensemble softmax as the target? Why not the model with minimum entropy? This choice is asserted but not empirically justified.

- **Gradient flow through teacher unclear:** The loss function uses H(x) as the soft target, but the paper does not specify whether gradients flow through H(x) (the selected model's output) or if stop-gradient is applied. This distinction fundamentally affects what is being optimized and should be clarified.

- **No ablation studies:** Key questions remain unanswered: How does performance scale with ensemble size M? Does architectural diversity matter (ImageNet uses only ResNet variants)? What is the effect of learning rate? Without ablations, the method's sensitivity and requirements cannot be assessed.

- **Small evaluation set for ImageNet:** The 90/10 split on a 7,000-sample subset yields only ~700 evaluation samples for 1,000 classes—fewer than one sample per class on average. This introduces substantial variance into accuracy estimates.

## Nice-to-Haves
- Analysis of how often each model acts as the teacher (to verify collaboration vs. domination)
- Calibration assessment (reliability diagrams) to check whether entropy minimization causes overconfidence
- Computational efficiency metrics (latency, FLOPs, memory) comparing IEL to static inference and other TTA methods

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Title "intra-model" misleading:** The naming is a reasonable stylistic choice given the paper's framing of models learning "from each other" within the ensemble. The criticism is semantic and not substantively problematic.

- **"Diversity as new optimization signal" contradiction:** While the reviewer notes the method minimizes diversity rather than using it as a signal, the paper itself acknowledges this explicitly ("by minimizing it we force members to agree"). The framing could be clearer but is not factually incorrect.

- **"Solid step forward in understanding human collaboration" claim:** This is vague marketing language in the contributions section, not a substantive technical claim. While unnecessary, it does not constitute a core weakness.

- **Human collaboration analogy consuming space:** The analogy is a reasonable pedagogical device. Criticizing its presence is a stylistic preference rather than a weakness.

- **Regularization constant α = 10e⁻¹¹ unexplained:** The paper states this "effectively makes our learning rate even smaller." While unusual, this is a minor hyperparameter detail, not a core flaw. The more important unexplained design choice is the teacher selection mechanism.

- **Freezing BN not ablated:** While ablations would strengthen the paper, this design choice is reasonable and explicitly motivated (ensuring adaptation comes from weight updates). Requesting ablations for every design decision is scope creep.

## Novel Insights
Beyond the paper's own contributions, the results reveal an interesting asymmetry in TTA difficulty: IEL shows substantial gains on blur-type corruptions (Defocus, Glass, Motion, Zoom) but systematic failures on noise-type corruptions. This pattern—where ensemble-based pseudo-labeling succeeds on structured corruptions but fails on additive noise—suggests that noise corruptions may produce confidently incorrect predictions that propagate through the ensemble, an observation that could inform future TTA method design. Additionally, the paper's finding that heterogeneous ensembles (different architectures) can mutually improve during inference opens an interesting direction: rather than treating ensemble diversity as a static property, actively leveraging it as an adaptation signal.

## Suggestions
- **Critical:** Add comparisons against at least one single-model TTA baseline (e.g., EATA or ROID configured for batch-size 1) to establish whether IEL's ensemble approach provides benefits beyond existing adaptation strategies.
- **Critical:** Report accuracy at a fixed epoch or provide a principled termination criterion rather than maximum accuracy across all epochs. At minimum, report results with and without oracle epoch selection to quantify the gap.
- Analyze failure modes on noise corruptions: Are the majority-voted predictions systematically wrong? Do confident incorrect predictions cause error amplification?
- Clarify gradient flow: explicitly state whether stop-gradient is applied to H(x) or whether gradients flow through the selected teacher model.
- Add one ablation study: comparing the current teacher selection against using the average ensemble softmax as the target would validate the core design choice.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 1.0]
Average score: 2.5
Binary outcome: Reject
