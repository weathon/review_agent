# Adversarial Attacks Already Tell the Answer: Directional Bias-Guided Test-time Defense for Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Vision-Language Models (VLMs), such as CLIP, have shown strong zero-shot generalization but remain highly vulnerable to adversarial perturbations, posing serious risks in real-world applications. Test-time defenses for VLMs have recently emerged as a promising and efficient approach to defend against adversarial attacks without requiring costly large-scale retraining. In this work, we uncover a surprising phenomenon: under diverse input transformations, adversarial images in CLIP’s feature space consistently shift along a dominant direction, in contrast to the dispersed patterns of clean images. We hypothesize that this dominant shift, termed the Defense Direction, opposes the adversarial shift, pointing features back toward their correct class centers. Building on this insight, we propose Directional Bias-guided Defense (DBD), a test-time framework that estimates the Defense Direction and employs a DB-score–based two-stream reconstruction strategy to recover robust representations. Experiments on 15 datasets demonstrate that DBD not only achieves SOTA adversarial robustness while preserving clean accuracy, but also reveals the counterintuitive result that adversarial accuracy can even surpass clean accuracy. This demonstrates that adversarial perturbations inherently encode directional priors about the true decision boundary.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an interesting observation that adversarial features exhibit a "directional bias" under input transformations. Based on this, it proposes DBD, a simple and training-free test-time defense that estimates and reverses this bias. The method achieves state-of-the-art robustness, with reported robust accuracy **even surpassing** clean accuracy under certain conditions.

### Strengths
- The core observation of a "Defense Direction" for adversarial examples is novel and insightful.

- The proposed method is simple, intuitive, and computationally efficient as a test-time defense.

### Weaknesses
- Unrealistic threat model: The paper's most impressive results are predicated on a PGD attack that requires access to ground-truth labels. This is a well-known weak and impractical threat model that leaks information about the correct class, which the defense is perfectly tailored to exploit. The validity of the core claims hinges on this flawed setup. In other words, the white box + cross-entropy loss configuration was originally established within the community for stress testing (previously the strongest achievable attack effect). If a model performs well under this configuration, there is reason to expect it will also perform well under other configurations. However, this method alters that situation by meticulously designing the approach specifically for this stress-testing scenario, rather than for more general scenarios.

- The headline claim that "robust accuracy can surpass clean accuracy" is an artifact of the unrealistic threat model. The more realistic pseudo-label evaluation, where this phenomenon vanishes and performance drops significantly, is not presented as the main result, which is misleading.

- The "directional bias" hypothesis is well-motivated for gradient-based attacks. However, the paper lacks a convincing analysis of why this mechanism is also effective against AutoAttack, potentially weakening the claimed technical contribution.

### Questions
- Could the authors justify centering the main evaluation on a threat model that requires access to ground-truth labels? To what extent does the high robustness achieved in this setting transfer to more realistic attack scenarios where such information is unavailable to the adversary?

- If the results from the more realistic pseudo-label attack evaluation (Table 3) were presented as the paper's primary findings, how would the authors reframe the central claims and contributions of this work?

- The "Defense Direction" hypothesis is well-motivated for speicific gradient-based attacks. Could you explain why this is also effective against the gradient-free components of AutoAttack, beyond a general robustness improvement from ensembling multiple input transformations?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identifies a dominant feature-space direction for adversarial images under diverse transformations, quantified via a DB-score, and proposes DBD to reconstruct features by shifting along a “Defense Direction” (high-DB) or averaging (low-DB). Evaluations on 15 datasets and multiple attacks (including AA on a subset) report high robust accuracies, sometimes exceeding clean accuracy. The approach is training-free, simple, and efficient relative to some prompt-tuning defenses.

### Strengths
- Clear empirical phenomenon and DB-score metric with intuitive visualizations
- Simple, training-free mechanism with precise equations
- Inclusion of all CLIP backbone tests and AutoAttack, as well as efficiency comparisons
- $\lambda$/$\tau$ ablations are helpful
- most hyperparameters are included for reproducibility, but the exact AA configs, seeds, and adaptive scripts are not provided even in the appendix, thus reproducibility concerns remain

### Weaknesses
- No adaptive white-box evaluation (BPDA/EOT) against the full defense despite stochastic and non-smooth components
- AutoAttack scope is limited and apparently non-adaptive to DBD; targeted AA not analyzed
- Robust > clean claims require stronger calibration evidence to rule out masking
- Causal validation of “Defense Direction” via gradient alignment is missing
- Timing numbers seem optimistic for 31 transforms; measurement details needed
- Prompt-set sensitivity not ablated

### Questions
- Can you add BPDA+EOT attacks with AA, differentiating through expectation over transforms, entropy filtering, and reconstruction, plus ε/step monotonicity curves? 
- Which specific AA modes and losses were used? was EOT applied? any targeted AA results? 
- Can you provide resource costs including (batching, precision, forward counts) and FLOPs? 
- Can you ablate prompt sets and template counts (hand-crafted vs CuPL) and report multi-seed confidence intervals?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DBD, a test-time defense method for vision-language models such as CLIP.
The key idea is that when applying multiple transformations to an input, clean images produce scattered features in the latent space, while adversarial images produce features that move in a consistent direction. This directional bias is quantified through a DB-score, which measures the alignment of feature shifts across transformations. The defense then routes the input based on this score.
The approach is training-free and evaluated on multiple datasets, showing improved robust accuracy compared to standard CLIP.

### Strengths
. The paper is well motivated and touches on a very active area. 

. The DB score is an interesting metric to characterize how adversarial perturbations behave across transformations.

. The method is simple to plug into existing models, requiring only forward passes under different transformations.

. The paper includes multiple datasets and compares against several baselines.

### Weaknesses
. My main concern is that the paper’s central claim, which is analyzing how and why transformations mitigate adversarial effects, is not actually supported by the experiments. The analysis is mostly descriptive, as they visualise CLIP features and observe that clean samples scatter, while adversarial ones align along a single direction. However, this remains a qualitative observation, rather than a real explanation. The proposed Defense Direction is repeatedly described as being “anti-parallel to the adversarial perturbation” and “pointing back toward the correct class center,” but this is never verified. A simple geometric check: cosine similarity between the Defense Direction and (a) the true class text embedding, (b) the nearest clean class centroid, or (c) the adversarial gradient, would make the claim much more credible.
The overall pipeline (multiple transformations + reconstruction) is not new. They follow the same pattern as many prior test-time defenses (transform, measure consistency, and reconstruct), but reinterpret it as a “directional bias”.  So while the framing is novel, the underlying idea is not fundamentally different, and the claimed analysis does not deepen our understanding of why these transformations work.

.  My other main concern is that the paper never tests adaptive or defense-aware attacks. All adversarial images are generated against the CLIP model and then passed through the defense, so the attacker is completely unaware of DBD. But DBD is a deterministic and differentiable post-processing pipeline, you can easily backpropagate through the transformations, entropy filter, and DB-score using BPDA or EOT (arxiv 1802.00420). If an attacker actually optimizes through the whole pipeline, the “directional bias” pattern could disappear completely. 

. The authors do include an ablation on τ showing that 0.8 gives the best mean accuracy across datasets. However, this analysis only reports overall accuracy averages. Since the routing decision between “clean” and “adversarial” streams fully depends on this threshold, we need to know how reliable that decision is. A proper analysis would include AUROC or false positive/negative curves for detecting adversarial inputs, and report how performance changes when τ or λ vary slightly around the chosen value. Also, they say τ is “estimated from clean images,” but it doesn’t explain how many, from which dataset, or whether this tuning leaks evaluation data.

. They claim robust accuracy sometimes exceeds clean accuracy. This is suspicious because the defense applies multiple transformations and filtering, which might remove noise or sharpen decisions even for clean samples. So part of the gain might come from the pipeline itself, not the defense logic. A proper ablation (transforms-only vs transforms + DB shift) is missing, and without it, we can’t tell what really improves robustness.

.

### Questions
. Could you verify that the Defense Direction truly points toward the correct class? A simple geometric check.

. How does DBD perform under adaptive, defense-aware attacks? Since the pipeline is differentiable, please evaluate with BPDA or EOT to see if the “directional bias” still holds.

. The method depends heavily on τ and λ. How were these calibrated, and do they generalize across datasets?
A small sensitivity or AUROC plot for the DB score would help confirm stability.

. For the reconstruction step, what is the reasoning behind the specific scaling rule in Eq. (8)?

. The paper reports robust > clean accuracy in several cases. Could you show where this gain comes from? I mean, does it persist if you use the same transformation pipeline without the DB shift?

. How sensitive is robustness to the number of transformations used? It would be useful to know how the defense performs if the set is smaller.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DBD (Directional Bias-guided Defense), a training-free, test-time defense framework that enhances the adversarial robustness of vision-language models such as CLIP. The authors identify an interesting and consistent phenomenon: adversarial perturbations in CLIP’s feature space tend to produce a dominant shift direction, whereas clean samples exhibit more dispersed feature transformations. Based on this observation, the authors design DBD to compute a Directional Bias (DB) score and reconstruct image features along a “Defense Direction” that counteracts adversarial perturbations. For inputs with high DB scores (likely adversarial), DBD shifts features along the Defense Direction; for low-DB inputs (likely clean), it averages multiple transformed views. Extensive experiments on 15 datasets demonstrate that DBD achieves SOTA while maintaining or even exceeding clean accuracy.

### Strengths
- Novel empirical discovery:  The finding that adversarial perturbations exhibit a strong, consistent directional bias in the feature space is insightful and clearly visualized (Fig. 1). This observation provides an interpretable foundation for designing test-time defenses.
- Excellent algorithmic performance:  DBD is training-free and efficient, operating entirely at test time without requiring adversarial training or prompt optimization. It achieves strong generalization and robustness across 15 datasets and multiple attack types, and shows remarkably high robustness accuracy under PGD attacks.
- Comprehensive analysis: The experimental analysis is thorough, particularly the Analysis under PGD attack with pseudo-label experiment, which convincingly demonstrates that Directional Bias-guided Defense can effectively recover information about the pre-attack image features.

### Weaknesses
- Lack of theoretical explanation: The paper does not provide a clear theoretical justification for the observed behavior, especially in the counterintuitive case where robust accuracy surpasses clean accuracy. A more rigorous analysis is needed to support this surprising result.
- Limited discussion of attack-specific behavior:  The results suggest that under PGD attacks, DBD almost perfectly reconstructs pre-attack image features, while this phenomenon does not hold for other attack methods. What causes this discrepancy? Would DBD still achieve robustness higher than clean accuracy under non-PGD attacks? The paper should provide a more detailed analysis and comparison across different attack types, rather than emphasizing its exceptional performance only under PGD.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
