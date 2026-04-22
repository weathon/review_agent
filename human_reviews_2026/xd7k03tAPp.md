# EviMix: Evidential Deep Learning with Latent-Space Mixing for Uncertainty Quantification and OOD Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Reliable uncertainty quantification (UQ) is essential for deploying deep neural networks in safety-critical domains such as autonomous driving and medical imaging. Evidential Deep Learning (EDL) provides a computationally efficient framework for estimating epistemic and aleatoric uncertainty through Dirichlet evidence assignment, enabling real-time uncertainty estimation. However, recent studies raise concerns about robustness, including conflation of uncertainty types, persistent epistemic uncertainty under abundant data, and sensitivity to training dynamics. The interaction between EDL and modern augmentation strategies remains poorly understood.

This work introduces three contributions: (1) analysis of how pixel-space mix-based augmentations affect EDL uncertainty and OOD metrics, (2) EviMix, a feature-space evidence-mixing framework performing cross-depth latent interpolation with layer-wise severities that decay with depth-early layers emphasize stronger low-level perturbations; late layers emphasize semantic interpolation-and (3) coupling these severities into the EDL objective to regulate aleatoric and epistemic evidence during training.

Experiments demonstrate that EviMix improves OOD detection, enhances functional specialization among uncertainty components, and yields stronger calibration gains relative to pixel-space and single-layer feature mixing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LatentMix for Evidential Deep Learning (EDL), a feature-space augmentation framework that interpolates latent representations across multiple network depths using both in-batch cross-class mixing and an external mixing set. The approach enhances EDL on out-of-distribution (OOD) detection and corruption robustness. Experiments on multiple benchmarks demonstrate significant improvements compared to previous methods.

### Strengths
- The problem of studying data augmentation's effect on uncertainty quantification is of great significance.
- The experiments cover multiple datasets and previous mixup-based augmentation methods.
- The proposed approach achieves superior performance in all metrics compared to the selected baselines.

### Weaknesses
- The baseline mixup methods selected in the paper are relatively outdated, with the latest one being PixMix from 2022. Mixup-based augmentations have many more recent works up to 2025, including not only pixel-space but also feature-space approaches. However, the paper does not sufficiently discuss them or justify their exclusion. The feature-space methods are stronger baselines then the pixel-space ones, which the paper should definitely include for comparison.
  - Cao, C., Zhou, F., Dai, Y., Wang, J., & Zhang, K. (2024). A survey of mix-based data augmentation: Taxonomy, methods, applications, and explainability. ACM Computing Surveys, 57(2), 1-38.
  - Jin, X., Zhu, H., Li, S., Wang, Z., Liu, Z., Tian, J., ... & Li, S. Z. (2024). A survey on mixup augmentations and beyond. arXiv preprint arXiv:2409.05202.
  - Ahmad, H. M., Morle, D., & Rahimi, A. (2025). LayerMix: Enhanced Data Augmentation for Robust Deep Learning. Pattern Recognition, 112332.

- Several key claims in the paper are largely heuristic and lack sufficient experimental validation. For example:
  - "Our key insight is that augmentation depth dictates the type of uncertainty probed: early-layer perturbations highlight aleatoric noise, while later-layer cross-class interpolation induces epistemic uncertainty by pushing representations off-manifold." This claim is not directly tested, and Table 6 only provides an indirect performance comparison of a few severity scheduling schemes.
  - "Larger s(l)aux inject more external variability, promoting aleatoric robustness, while larger s(l)cross pull features toward other classes, generating off-manifold representations that regularize epistemic uncertainty." This statement is only intuitively plausible, without any quantitative or visual evidence to substantiate it.

- Potentially unfair comparisons.
  - "Baseline methods converge within 100 epochs - typically plateauing after ≈ 75 epochs - whereas LatentMix is trained for 150 epochs to compensate for the reduced in-distribution updates introduced by cross-class mixing." First, why does LatentMix have reduced in-distribution updates, since each iteration should have even more ID data from other classes. Second, how is convergence defined, by training loss or test UQ metrics? Showing the plots that training other methods by more epochs would not improve their test results while training LatentMix would can justify this early stopping for other methods.
  - Section 3.4 introduces a 3-stage dynamic mixing cap for training LatentMix, where the mixing is disabled in the first and third stage, but it is not stated in the paper whether the same scheme is applied to other baseline methods. If not, whether the improvement of LatentMix benefits from this dynamic training scheme needs to be clarified.

- The method appears to rely on an external set of auxiliary samples, potentially derived from prior approaches such as PixMix (as implied by Figure 1). However, the paper does not clearly specify the source or composition of these auxiliary samples used in the experiments. If LatentMix indeed depends on such externally provided data, this dependency should be explicitly discussed, as it affects both reproducibility and the fairness of comparisons.

- The severity sampling distributions are defined as uniform for s(l)aux and Bernoulli for s(l)cross, but the rationale for these specific selections is not discussed, nor is any ablation presented to assess their influence on the model’s behavior or uncertainty estimates.

- The paper uses "pixel-space augmentations" multiple times to refer to their baselines (e.g., PixMix), which overstates the scope. Pixel-space augmentation is a much broader category that should include far more augmentation types such as photometric augmentations (color jitter, brightness, contrast, etc.). In fact, the main comparison here is only against some pixel-space mixup methods.

- The computational overhead due to feature mixing at multiple depths is not analyzed at all in the paper.

### Questions
Authors can refer to weaknesses for the rebuttal.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes changes to Evidential Deep Learning for uncertainty quantification in image classification. In particular, they propose to mix in latent space instead of in input space, and to mix in two ways — with external images (for aleatoric noise) and with other classes (for epistemic noise). The results show a solid outperformance against EDL baselines and other uncertainty methods.

### Strengths
1. The proposed changes to EDL are well-explained and I appreciate the discussion of hyperparameters in the appendix, for both the proposed method and all baselines. (Releasing code would be even more appreciated)
2. The paper tests against a nice range of baseline methods both from EDL and general UQ. The models (VGG-16, ImageNet-18/50) and datasets (CIFAR, ImageNet) are sufficient and relatively standard in UQ literature. 
3. The recorded performance gains are consistent and have a decent delta across all models and datasets.

### Weaknesses
Weaknesses, in order of magnitude:
1. There is one point that I think might make the comparison unfair: The proposed method is trained for 150 epochs, whereas all baselines use 75-100. The authors note that this is because “they plateau”, and because their method sees less clean examples due to the training. However, from a generic FLOP perspective, the comparison is still unfair and most results can be explained by longer training. It would be nice to see model curves throughout training (baselines + proposed), and to compare proposed model @ 75 epochs, see questions. 
2. The same holds for the data the proposed method trains on. It is the only method that trains on perturbations of CIFAR-10C / ImageNet-C style, which we also test on. See questions. 
3. A larger issue I see is that the paper claims boldly and multiple times that the proposed training disentangles aleatoric and epistemic uncertainty, even criticizing prior works for this. However, this is never tested. I would recommend to run cross-tests, so whether the proposed aleatoric component can predict epistemic tasks like OOD detection and vice-versa. There are benchmarks for this purpose, see https://openreview.net/forum?id=x8RgF2xQTj and https://arxiv.org/html/2408.12175v1 . It would strengthen the argumentation to evaluate on them.
4. The paper introduces many changes to default EDL. Curriculum learning and mixing scheduler are ablated, but the extra regularization terms (and the longer training) are not. It would be nice to ablate these. 
5. It would be nice to add naive baselines: Simple CE training, either in one model or as a deep ensemble (n=10).
6. It would be nice to discuss the computational tradeoff. Naively, it looks like the mixing with auxiliary images should double forward time, which, assuming a 1-to-2 time mixture with backpropagation, might increase train time per batch by 25%.

### Smaller things that would improve the revised version but don’t influence my rating and don’t need rebuttal

* Consider numbering all equations, whether “important” or not. That simplifies referencing to them (for example in the questions below)
* I think in Section 4.2.1 you used \paragraph{} for the first sentence, but somehow the formatting is off. I suggest to fix it to increase readability
* Confidence intervals would be appreciated, but not giving a worse score here since confidence intervals on ImageNet tend to be small thanks to the large eval set
* In Section 4.2.2, you are falling into past tense. Consider using present tense in the whole paper. 
* Section 4.2.2 does not discuss or reference Tables 4 and 5. Please add a discussion.
* Table 8 has two bottomrules
* There is the point of impact. I personally do not weigh this into the final rating because I think reviewers are horribly bad in estimating future impact. However, just in case some other reviewers mention it or the AC lays value on it, the benchmarks in this work, image classification, are a bit dated. Modern Vision-Language models would be interesting. However, this setup is standard for EDL, so I personally do not see this as a problem.

## Justification for overall score

I appreciate the robust results, and they seem reasonable in the light of the current progress of EDL methods for image classification. However, I am sceptical of two points: 1) the claimed improvement over baselines (because of potentially unfair comparisons, see above and below), and 2) the claims of disentangling aleatoric and epistemic uncertainty, which are untested. I am looking forward to data on those points in the rebuttal period and am willing to increase my score if these points get addressed.

### Questions
1. In step 1, why is the uniform random number multiplied by 0.5? Couldn’t this just be added into the uniform upper limit?
2. In step 3, I assume you use only one class j != i to mix with? If yes, you could add the sampling of that class into the algorithm to resolve ambiguity.
3. In line 184 you write that “late layers apply cross-class mixing to induce calibrated epistemic uncertainty”. Why is this calibrated, i.e., according to which definition? 
4. You train your method for double the epoch that baseline methods are trained for (lines 308-310). Can you provide train curves to check when your method and when baselines plateau, and can you provide results for what happens when you train baselines for 150 epochs (or your method for 75 epochs)?
5. Do I understand correctly that your method is the only one that is trained on corrupted inputs similar to CIFAR10-C (lines 195-197), whereas all baselines only use overlaying augmentations like in Mixup/Cutmix/...? Doesn’t that make the comparison in Figure 3 a bit unfair? (Because obviously a method trained on corrupted data will perform better on corrupted data)
6. What exactly is compared in Table 3? Whether the total uncertainty metric can capture wrong outputs? 
7. In Table 6, external early seems to drive most performance. Can you reach the same performance by just having external early, without cross-class, to simplify your method? (If you can match performance here, this would make your method simpler and thus more impactful; I will not reject the paper because you can make a simpler version of it that performs the same way)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study uncertainty estimation and OOD detection via evidential deep learning. They propose LatentMix, a feature-space augmentation framework inspired by approaches like PixMix and AugMix, but that performs interpolation on intermediate network layers.

They apply LatentMix to Resnet models and evaluate calibration and OOD detection performance on CIFAR10 and Imagenet dataset variants, mainly comparing against MixUp and other pixel-based augmentation methods.

### Strengths
- The proposed LatentMix method makes sense overall.
- LatentMix improves performance in terms of calibration and OOD detection of standard evidential deep learning, as well as pixel-based augmentation methods such as MixUp and PixMix.

### Weaknesses
- The paper could definitely be more well written. In particular, Section 4 is quite difficult to follow and the results are somewhat sloppily presented. For example, Table 4 and 5 are not referenced in the text and the results are not described anywhere.
- The experimental evaluation is quite limited, only pure image classification, only relatively small models.
- The technical contribution/novelty is somewhat limited.

### Questions
Questions/suggestions:
- Is the proposed method only applicable to classification problems, or could it be extended to regression as well?
- Could the proposed method be applied to e.g. ViT models as well?
- Table 2 should contain results for both Resnet-18 and VGG16? Is the top part for Resnet-18 and the bottom part for VGG16? But what is meant by the "Near-OOD Average" and "Far-OOD Average" headings then?
- Line 186: Do you have B examples both in the batch and in the set of auxiliary samples, or are these of different sizes? 





Minor things:
- The title has "EviMix" but then across the paper the proposed method is called "LatentMix".
- Figure 1 caption: Why is this text italicized?
- Figure 2 caption: "Refer to 4 regarding OOD datasets" --> "Refer to Section 4 regarding OOD datasets".
- "Evidential Deep Learning (EDL)" is defined multiple times, probably unnecessary.
- Figure 2 is not referenced in the text.
- Line 348: "Figure 3 and table 1" --> "Figure 3 and Table 1".
- Line 363: "Tables 2" --> "Table 2".
- In Section 4.2.1, "Calibration under CIFAR-10-C", "OOD Detection on ResNet-18 and VGG16" and "Comparison with Prior Methods" are not formatted correctly, they should be paragraph headings like in Section 4.3 or 5.
- Line 351: "For full results see A" --> "For full results see Appendix A.".
- Table 2 caption, "CIFAR-10 OOD datasets refer to section 4.1 4", don't know what you mean here.
- Table 4 and 5 are not referenced in the text.
- Line 446, "proposed multi-value assignment B.7".
- Line 448, "and then vanishes 1."
- I would consider reformatting the equation at the end of Section 3.3 a bit.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses robustness issues in Evidential Deep Learning (EDL) for uncertainty quantification by investigating the impact of data augmentation strategies. The authors systematically study how pixel-space augmentations affect EDL's uncertainty estimates and propose LatentMix, a novel feature-space augmentation that interpolates latent representations across multiple network depths with layer-specific mixing severities. These severities directly regulate the EDL loss to shape aleatoric and epistemic uncertainty. Experiments show LatentMix improves out-of-distribution detection, better separates uncertainty types, and enhances calibration compared to pixel-space and single-layer mixing approaches.

### Strengths
- Overall, the paper is well written and easy to follow. 
- The paper tackles an important problem. 
- The proposed method is technically sound. 
- The proposed method outperforms pixel-space and single-layer alternatives

### Weaknesses
- The method involves multiple components according to section 3.2 and 3.3. How sensitive is the proposed method with respect to these hyper-parameters?
- In general, the proposed method seems quite ad-hoc, most of which lacking theoretical justifications. It is not clear how well it would work outside the scenarios tested. Does this work across different architectures and domains?
- The authors of the paper use the OpenOOD benchmark. How does the method compare with other methods that use the OpenOOD benchmark? A thorough comparison against other methods in addition to EDL and mixup variants might be needed to demonstrated the effectiveness of the proposed method. 
- In light of the fact that doing mixup in the latent space is studied extensively previously, it is not immediately clear to me if the contribution is sufficient from algorithmic perspective.

### Questions
- How does LatentMix compare to existing feature-space augmentations (MixFeat, ManifoldMixup)[1, 2]?
- How are mixing severities sampled? What's the distribution and hyperparameter sensitivity?

[1] "MixFeat: Mix Feature in Latent Space Learns Discriminative Space"
[2] "Manifold Mixup: Better Representations by Interpolating Hidden States"

### Soundness
3

### Presentation
3

### Contribution
2
