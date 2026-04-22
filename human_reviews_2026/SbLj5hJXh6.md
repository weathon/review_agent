# Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
This work analyzes the training dynamics of Image Restoration (IR) Transformers and uncovers a critical yet overlooked issue: conventional LayerNorm (LN) drives feature magnitudes to diverge to a million scale and collapses channel-wise entropy. We analyze this in the perspective of networks attempting to bypass LayerNorm’s constraints, which conflict with IR tasks. Accordingly, we address two misalignments: 1) per-token normalization that disrupts spatial correlations, and 2) input-independent scaling that discards input-specific statistics. To address this, we propose Image Restoration Transformer Tailored Layer Normalization (i-LN), a simple drop-in replacement that normalizes features holistically and adaptively rescales them per input. We provide theoretical insights and empirical evidence that this design effectively captures important spatial correlations and better preserves input-specific statistics throughout the network. Experimental results verify that the proposed i-LN consistently outperforms vanilla LN on various IR tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper analyzes the training dynamics of the image restoration Transformer, identifies the limitations of layer normalization, and proposes a global version of layer normalization that normalizes features holistically. Through theoretical analysis and experimental verification, its effectiveness in various tasks of image restoration is demonstrated.

### Strengths
1. The paper is well-written and the theoretical analysis is solid.
2. The method is very simple and has been verified for its effectiveness in various IR tasks. I highly appreciate its motivation of specifically designing methods for low-level visual tasks. This is more impressive compared to simply transplanting methods from other fields directly.
3.This method is plug-and-play and may become the standard approach for designing network structures in the field of low-level vision.

### Weaknesses
No obvious weaknesses are identified.

### Questions
1. Could you provide the pseudo-code of the i-LN you proposed? This would be helpful for us to quickly get started with the implementation and accurately reproduce the results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates why Image Restoration (IR) Transformers often show unstable training behavior when using standard Layer Normalization (LN). The authors find that LN causes feature magnitudes to diverge and channel entropy to collapse, indicating a fundamental mismatch between LN and dense prediction tasks.
To address this, they propose i-LN, a simple drop-in replacement that applies spatially holistic normalization and input-adaptive rescaling. i-LN stabilizes training and improves performance across multiple IR tasks (super-resolution, denoising, deblurring, and artifact removal) and Transformer architectures.

### Strengths
1. Novel insight into a fundamental issue – The paper is the first to thoroughly diagnose the instability and abnormal training dynamics of IR Transformers caused by LN. The observations of feature divergence and entropy collapse are well-supported and enlightening.

2. Strong theoretical grounding – The authors formalize the notion of structure-preserving transformations and mathematically show that LN fails to maintain spatial isomorphism, whereas the proposed i-LN satisfies these properties.

3. Simple yet impactful method – i-LN is conceptually straightforward, easily implemented, and compatible with existing architectures without any retraining overhead.

### Weaknesses
1. The theoretical analysis is insightful but somewhat idealized — it assumes a simplified setting without the learnable affine parameters (β, γ) and does not account for the nonlinear activations that follow normalization. While this abstraction helps clarify the core argument, it may limit the generality of the conclusions in practical Transformer architectures.
2. Qualitative evaluation could be expanded – The visual examples are convincing, but adding perceptual metrics (e.g., LPIPS, FID) or user studies would make the improvements more tangible.

### Questions
see the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies the effect of Layer Normalization (LN) in Transformer-based image restoration networks, focusing on the observed instability and feature magnitude divergence during training. The authors analyze how the per-token LN operation may disturb spatial correlations and feature statistics, and propose a variant called i-LN, which normalizes across both spatial and channel dimensions with an input-dependent scaling term. Experiments across several image restoration tasks (SR, DN, DR, CAR) show modest but consistent improvements in stability and quantitative performance. Overall, the work offers a careful empirical examination of normalization behavior in low-level Transformer models, though the methodological change is relatively minor.

### Strengths
1. The paper focuses on a clear and well-defined problem — the instability of Layer Normalization in image restoration Transformers — and provides intuitive analysis to support it.

2. The proposed i-LN module is conceptually simple and easy to implement, making it practical for existing Transformer-based IR frameworks.

3. Experiments across several restoration tasks show consistent but moderate improvements, indicating the method’s general usefulness.

4. The paper is clearly written and well-organized, making the motivation and results easy to follow.

### Weaknesses
### Main Concerns:
1. Since the proposed i-LN is designed to stabilize feature statistics across varying inputs, it would be informative to evaluate its effectiveness in an All-in-One restoration setting, where a single model must generalize across multiple degradation types. This could further validate the claimed robustness and statistical consistency advantages.

2. While the proposed i-LN improves feature stability by normalizing across spatial and channel dimensions, this inevitably increases the computational scope. The paper claims negligible overhead but provides no quantitative analysis to support this; it would be helpful to report the actual runtime or FLOPs to assess the trade-off between stability and efficiency.

3. For a low-level vision paper, the visual comparisons are too limited and small in scale. Given that image restoration quality is best evaluated through perceptual inspection, more visual comparison samples are expected to be included in at least the appendix or supplementary materials, but this is missing.

4. While the proposed i-LN is claimed to improve feature stability and robustness, the paper does not include any evaluation on real-world degradations, which are common and practically relevant in image restoration. Since the method is designed to better preserve low-level statistics, testing it on real-world datasets would be important to demonstrate its practical effectiveness and generalization beyond synthetic benchmarks.

5. Despite the solid analysis, the performance gain is modest, which may limit perceived impact.

6. The method is incremental in implementation—essentially a normalization variant—though well justified.

7. Computational overhead or efficiency impact of holistic normalization is not deeply discussed.

8. Could include more discussion on generalization beyond image restoration tasks (e.g., classification, segmentation)?.

### Minor Concerns:
1. The computation resources are expected, for example, the authors only mentioned in Line 236 via A6000 GPUs, but how many for each experiment?

2. The code of the proposed method is expected to be released.

3. The term “IR Transformer” is not clearly defined — since the architecture is largely identical to standard vision Transformers, the authors should clarify whether the analysis truly targets image restoration–specific behavior or applies more broadly to general Transformer models.

### Questions
1. Could the authors provide quantitative evidence of efficiency, such as runtime, FLOPs, or memory cost, to support the claim that the proposed holistic normalization incurs negligible computational overhead?

2. Since i-LN aims to stabilize feature statistics, have the authors considered evaluating it under an All-in-One or real-world degradation setting to demonstrate robustness across diverse degradation types?

3. The paper focuses on Transformer-based IR models — do the authors expect i-LN to generalize to non-Transformer or hybrid CNN–Transformer architectures? If so, could preliminary results or insights be shared?

4. The analysis emphasizes improved stability and entropy preservation, but could the authors quantify this improvement (e.g., through variance reduction, entropy scores, or training convergence curves) to better support these claims?

5. Given that the proposed method is relatively simple, could the authors discuss whether i-LN interacts with other normalization techniques (e.g., InstanceNorm, GroupNorm) or whether similar stability effects could be achieved by alternative normalization formulations?

Please also refer to the **Weaknesses**.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper address the image reconstruction task in vision domain. 
The authors found that image reconstruction using transformers with vanilla layer norm lead to feature magnitudes divergence and  channel entropy collapses.
The authors further analyze the patterns by relaxing the constraints of layer norm, i.e., per-token normalization and input-independent scaling. Specifically, vanilla layer norm breaks spatial correlation due to the per-token normalization and discard input-specific statistics due to the input-independent scaling, which are critical for image restoration tasks. To handle these issues, the authors propose i-LN, which normalizes holistically across space and channels and re-scales each block by the input’s own standard deviation to recover global scale. Experimental results qualitatively and quantitatively show that i-LN obtain better PSNR and SSIM on super-resolution, deraining, denoising, and JPEG artifact removal than vanilla LN on various benchmarks.

### Strengths
* Easy to read
* Simple but intuitive claims
* Extensive experiments

### Weaknesses
* Major concern
    * While the authors demonstrated the strengths of the proposed i-LN over vanilla layer norm (LN) and spatially holistic layer norm (LN*), the experiments only show the comparisons with vanilla LN and other normalization approaches. If the provided claims are valid, I guess the superiority among LN, LN*, and i-LN on the image restoration problem should be LN < LN* < i-LN. Could you further validate this superiority, not only over specific benchmarks but over various benchmarks?
    * While the authors provided channel entropy collapse comparison among LN, LN*, and i-LN in Fig. 8, it seems not sufficient since the volume for this comparative analysis is too light than the overall volume of the experiments. And still there's no qualitative or quantitative performance comparison among LN, LN* and i-LN, while both are necessary.
    * Suggestion
        * I recommend to supplement these experimental results. Since the results for LN and i-LN are already secured, only results for LN* are necessary for the following comparisons:
            * relative position embedding comparison (Fig. 9) among LN, LN* and i-LN
            * Qualitative results comparisons (Fig. 6) among LN, LN* and i-LN
            * Quantitative comparisons (Table 1, 2) for at least two datasets (e.g., Set14 and BSD100) among LN, LN* and i-LN
    * I'll raise my rating if this concern is resolved
* Explanation for the derivation of feature divergence in Figure 4 and other experimental results in some place (e.g., the paragraph in lines 260-267) would be helpful

### Questions
* Why were three type of model implementations necessary?
* I recommend to reorganize the location of Table 5 in appendix. There would be better position than the top of the first page of Appendix

### Soundness
3

### Presentation
3

### Contribution
2
