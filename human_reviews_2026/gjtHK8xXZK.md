# Depth in Motion: Robust Self-Supervised Learning via Representation-Optimization-Supervision Synergy

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Self-supervised monocular depth estimation recovers scene geometry from unlabeled monocular videos, yet its reliance on photometric constancy tends to cause failures in dynamic scenes: motion and occlusion corrupt correspondences, bias optimization toward texture-sparse regions, and drive residuals into heavy-tailed distributions that undermine supervision. To address these challenges, we propose a Representation–Optimization–Supervision Synergy Network (ROSS-Net), which establishes a holistic defense by restructuring the entire estimation flow to mitigate interlinked failure modes. At the representation level, the Spatio-Temporal Epipolar Calibrator (STEC) validates correspondences across appearance, feature, and temporal cues to filter motion-induced mismatches while preserving dynamic evidence. At the optimization level, the Entropy-Guided Spectral Integrator (EGSI) calibrates depth-axis spectra to counter low-frequency optimization bias while adding no inference-time overhead. At the supervision level, the Order-Statistic Consensus Operator (OSCO) trims and reweights outlier residuals, converting noisy reprojections into robust supervision. Experiments on KITTI and NYUv2 show that ROSS-Net significantly outperforms prior methods under motion and occlusion, and generalizes strongly to unseen domains such as Make3D and ScanNet.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Traditional self-supervised monocular video depth estimation methods rely on photometric constancy, leading to failures in motion and occlusion scenes because:
1. edge misalignment, where appearance edges do not align with depth boundaries.
2. gradient misalignment, where the image and depth gradients diverge.
3. tailed photometric blurries, caused by motion, occlusion, or illumination changes.

Correspondingly, the authors proposed a three-level framework called ROSS-Net across representation, optimization, and supervision:
1. STEC validates correspondences across appearance, feature, and temporal cues to produce calibrated features and cost volumes.
2. EGSI applies frequency-domain weighting using spectral entropy derived from FFT to guide optimization.
3. OSCO estimates pixel-wise confidence through trimmed quantile statistics and reweights supervision accordingly.

On the KITTI (outdoor) and NYUv2 (indoor) datasets, the author claimed that ROSS-Net outperforms existing methods (such as ManyDepth2, EDS-Depth, etc.) in all error and accuracy metrics.

### Strengths
1. The motivation from "tailed photometric blurries" is intuitive. When I'm training depth estimation models, the reviewer also usually observed such phenomenon that sometimes the tailed photometric residuals yields illed supervision.
2. The focus on "photometric constancy" is valuable in self-supervised depth estimation.
3. The appendix is rich. The statements of ethics and reproducibility is also provided, which are optional.

### Weaknesses
1. [Motivation/Idea level] The motivation from "edge misalignment" and "gradient misalignment" is not solid enough. First, they are basically the same thing. Image can be considered as the combination of so many dense factors, depth is only one of them. No matter in image to depth or video to depth, these two misalignments is not enough to be viewed as "limitations" of existing works, but something more essential and fundamental. Second, for common videos, the lighting change between continuous frames is ignorable, perhaps the "photometric constancy" should be better studied given videos with large temporal gaps. Thus, the reviewer doesn't consider these "limitations" valuable. The motivation from "tailed photometric residuals" is intuitive to the reviewer, and solving it via a re-weighting is also intuitive. But though it may produce some outliers during training, in the reviewer's experience, it has nearly zero influence in the final performance. Also, only this point is not valuable enough for a research paper.
2. [Method level] The method design of three modules is difficult and the training pipeline is complex. All three modules have multiple hyperparameters (gating thresholds, entropy smoothing terms, and quantile ratios), which are very expensive to tune. Thus, it's imaginable that it's very difficult to analyze the independent sources of performance improvement, and some small difference in settings can lead to large performance variances. Also for the efficiency, although the authors claim there is no additional overhead during inference, during training the reviewer think it will be more computationally intensive.
3. [Experiments] The table is quite large, but the author just simply summarizes it with the phrase "outperforms prior methods," without discussing why certain metrics improve more significantly, which will be more insightful than pure numbers.
4. [Writing] The correspondences between your motivation and method are not clear, the reviewer feels you were attempting to combine (all) related formulas and networks, without clearly and directly targeting the problems you want to solve and the critical bottleneck in the field. Also, the presentation of your method lacks of intuitive explanation, which requires hours to roughly understand, putting a heavy burden on the readers to understand. Moreover, the naming of your modules is weird. It takes the reviewer several minutes to separate those three names. A simpler and more understandable name would be better.

### Questions
Please refer to the "Weaknesses".

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of self-supervised monocular depth estimation in dynamic scenes, where traditional photometric consistency assumptions fail. The authors propose a holistic framework called ROSS-Net (Representation-Optimization-Supervision Synergy Network) that introduces a three-level defense mechanism to improve robustness by addressing failures at the representation, optimization, and supervision levels. The paper demonstrates state-of-the-art performance on KITTI and strong generalization capabilities.

This paper presents an ambitious, tripartite framework to solve the problem of self-supervised depth estimation in dynamic scenes. Its motivation is clear, and the experimental results on existing benchmarks are impressive. However, the paper has some major potential flaws, particularly concerning the design rationale of its core module (STEC) and the experimental setup (validating a dynamic model on the static NYUv2 dataset).

**My recommendation is therefore conditional (just because of the impressive results). If the authors can provide a convincing explanation and clarification in their rebuttal to the weaknesses and questions raised, I will maintain my score.** If the authors cannot adequately address these major concerns, I will consider `lowering` my score.

### Strengths
- Ambitious Technical Approach: The paper is technically ambitious, combining ideas from geometric validation (STEC), frequency-domain analysis (EGSI), and robust statistics (OSCO). This comprehensive approach attempts to build a multi-faceted defense against motion-induced artifacts.
- Strong Empirical Results on KITTI: The method achieves state-of-the-art results on the challenging KITTI benchmark, demonstrating its effectiveness in outdoor driving scenarios where dynamic objects are prevalent. The quantitative and qualitative results show clear improvements over prior methods in handling object boundaries and thin structures.

### Weaknesses
- Questionable Soundness of the STEC Module: The claim that STEC "preserves dynamic evidence while mitigating motion-induced mismatches" is not well-supported, and its underlying mechanism appears flawed. The module seems to rely on inter-frame photometric or feature differences to identify motion. This heuristic is incorrect for general motion patterns. For example, in the case of two cars moving towards each other, the difference will be large, but this large difference does not provide a correct supervisory signal for a rigid-world assumption. This issue has been discussed in detail in previous works.

- Illogical Experimental Setup on NYUv2: A major weakness is the decision to train and evaluate on the NYUv2 dataset. This benchmark consists almost exclusively of static indoor scenes with virtually no dynamic objects. Using this dataset to validate a method explicitly designed to handle motion and dynamic scenes is inappropriate and undermines the credibility of the results presented for indoor environments.

- Insufficient Scalability Validation: The paper lacks validation on other standard and challenging driving datasets like Cityscapes and DrivingStereo. It is common practice in the literature to evaluate on these benchmarks to demonstrate robustness and generalization. Many prior works have done so, including MonoViT [1], RobustDepth [2], D4RD[3], Yan et al [4], and Jasmine [5]. This omission limits the assessment of the method's applicability. Please also cite these papers because they are very important papers in this area!

- Unsupported Core Motivation: The paper states that motion and occlusion "bias optimization toward texture-invariant, low-frequency cues." This is a strong claim that serves as a key motivation for the work, but it is presented without any citation or empirical proof.

- Complex and Obfuscating Terminology: The paper uses a very specific and complex naming scheme for its components. This makes the paper difficult to read and feels like it may be masking the true, and potentially simple (or flawed, in the case of STEC), underlying ideas.
```
[1]Zhao, Chaoqiang, et al. "Monovit: Self-supervised monocular depth estimation with a vision transformer." 2022 international conference on 3D vision (3DV). IEEE, 2022.
[2]Saunders K, Vogiatzis G, Manso L J. Self-supervised monocular depth estimation: Let's talk about the weather[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 8907-8917.
[3]Wang, Jiyuan, et al. "Digging into contrastive learning for robust depth estimation with diffusion models." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.
[4]Yan W, Li M, Li H, et al. Synthetic-to-real self-supervised robust depth estimation via learning with motion and structure priors[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 21880-21890.
[5]Wang J, Lin C, Guan C, et al. Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation[C]. Advances in neural information processing systems, 2025, 30.

### Questions
- Justification for Core Claim: In the abstract and introduction, you claim that motion and occlusion bias optimization toward low-frequency cues. What is the source of this claim? Is it based on established findings in prior work, or is it an observation from your own analysis? If the former, please provide the necessary citations.

- Rationale for STEC's Design: Given that simple frame differencing is known to be an unreliable cue for handling dynamic objects under general motion, could you provide a more detailed justification for the design of the STEC module? How does it handle common scenarios like opposing traffic or objects moving parallel to the camera at different speeds?

- Rationale for using NYUv2: Could you explain the reasoning behind training and evaluating a method designed for dynamic scenes on the static NYUv2 dataset?

- Clarification of Notation:
  - In Equation (6), what does the time index $t$ represent? Is it the frame index in a video sequence or the training iteration?
  - In Equation (9), what is the index $m$ for the spectral bin? Please provide a clear definition.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles self-supervised monocular depth in the wild by filtering and reweighting pixels so that training focuses on trustworthy evidence. First, STEC screens pixels with appearance/feature checks plus short-term temporal smoothing, then builds a cleaner cost volume so each pixel’s ``depth vs. likelihood” curve is sharper and less noisy.  Next, EGSI looks at those curves, uses a simple 1-D FFT to judge their uncertainty (learning which shapes are reliable), and, only during training, uses that uncertainty to blend multi-view depth with a monocular prior; inference cost stays the same. Finally, OSCO applies a soft, robust loss with multi-view consensus that down-weights outliers (occlusions, independently moving objects) instead of hard-dropping everything dynamic. Overall, the idea is not mysterious: it’s a noise-filtering and weighting pipeline that keeps dense predictions but gives low influence to unreliable pixels; the design is sensible and engineering-oriented, with modest novelty mainly in the spectral uncertainty step rather than brand-new building blocks.

### Strengths
The paper tackles an important practical problem, robust self-supervised depth in dynamic scenes, using a clear three-layer pipeline that is technically sound and easy to integrate with no extra inference cost. Its spectral (FFT-based) uncertainty is the most original element and plausibly improves where depth likelihoods are ambiguous, with ablations and qualitative results that generally back the claims. Overall, the contribution is useful, likely valuable to practitioners even if the conceptual novelty is moderate.

### Weaknesses
Novelty is incremental, and experimental coverage is narrow. Please add seed variance and confidence intervals, as well as sensitivity curves for STEC thresholds, EMA, depth sampling choices, and OSCO’s penalty and consensus settings. In addition, clarity suffers: the figures, especially the method diagram, are cluttered, mix several subplots without explicit inputs or outputs, and make the flow hard to follow.

### Questions
(1) EGSI vs. simpler uncertainty: If you replace EGSI with (a) a learned confidence head without FFT and (b) plain Shannon entropy on the depth-likelihood, under the same backbone,  how much performance do you lose? Please provide a controlled ablation to justify the spectral step.

(2) Clarity/IO flow: Can you provide a single input→STEC-gated cost volume→EGSI (train-only) →OSCO loss→outputs diagram with explicit inputs/outputs and a few lines of pseudocode per module? This would resolve much of the current ambiguity.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a self-supervised depth estimation framework designed for both single-view and multi-frame inputs, trained on video sequences. The authors address three key challenges—edge misalignment, gradient misalignment, and heavy-tailed photometric errors—at three corresponding levels: representation, optimization, and supervision. The proposed approach is novel and demonstrates superior performance across four benchmark datasets compared to existing methods.

### Strengths
1. The paper is well-written and technically solid.

2. The motivation is clear: the authors systematically identify three core challenges in self-supervised depth estimation and propose corresponding solutions at different levels to address them.

3. The method is thoroughly evaluated on four datasets—two outdoor and two indoor—showing consistent performance improvements over previous approaches.

4. Comprehensive ablation studies and qualitative comparisons effectively validate the proposed contributions and demonstrate the method’s efficacy.

### Weaknesses
1. The model architecture is not clearly described. Is it based on a ResNet, a Transformer, or another backbone?

2. Although the authors discuss challenges related to dynamic scene reconstruction, the proposed method is not evaluated on datasets containing significant dynamic motion, such as the BONN RGB-D or TUM-Dynamic datasets.

3. In the indoor scene evaluation, the paper “Auto-Rectify Network for Unsupervised Indoor Depth Estimation” (TPAMI 2022) should be included for a more complete comparison.

4. The proposed method still requires separate training for different datasets. It would be valuable to explore whether a single unified model, trained on a mixed dataset containing both indoor and outdoor scenes, could generalize across diverse environments.

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
4
