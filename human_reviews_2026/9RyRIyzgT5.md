# Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Classifier-free Guidance (CFG) is a widely used technique in modern diffusion models for generating high-quality samples. However, through an empirical analysis on both Gaussian mixture models with closed-form solutions and real-world data distributions, we observe a discrepancy between the suboptimal results produced by CFG and the ground truth. The model's excessive reliance on these suboptimal predictions often leads to low fidelity and semantic incoherence. To address this issue, we first empirically demonstrate that the model's suboptimal predictions can be effectively refined using sub-networks of the model itself, without requiring additional training or the integration of external modules. Building on this insight, we propose **$S^2$-Guidance ($S$tochastic $S$elf-Guidance)**, a novel method that leverages stochastic block-dropping during the denoising process to construct sub-networks. This approach effectively guides the model away from potential low-quality predictions, thereby improving sample quality. Extensive qualitative and quantitative experiments across multiple standard benchmarks for text-to-image and text-to-video generation tasks demonstrate that **$S^2$-Guidance** delivers superior performance, consistently surpassing CFG and other advanced guidance strategies. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a general and elegant strategy to enhance generative performance by mitigating issues of low fidelity and semantic incoherence. Specifically, the authors empirically demonstrate that suboptimal predictions of diffusion models can be effectively rectified using the model’s own sub-networks, without requiring additional training or the integration of external modules. Building upon this observation, the paper introduces S²-Guidance, a novel method that leverages stochastic block-dropping during the denoising process to activate sub-networks for self-guidance. The experiments conducted on three tasks show the great performance of the proposed method.

### Strengths
1. Clear and Insightful Motivation. The paper identifies a fundamental limitation in classifier-free guidance (CFG)—its tendency to over-amplify conditional signals and distort the underlying data distribution. The authors’ analysis of this phenomenon using Gaussian mixture toy examples is both illustrative and rigorous. This strong diagnostic foundation naturally motivates the development of S²-Guidance as a self-correcting mechanism.

2. Elegant and Minimalist Design. The proposed method achieves a notable balance between simplicity and effectiveness. By leveraging stochastic block dropping within the model itself, S²-Guidance creates intrinsic “weak sub-networks” that serve as self-guiding components. This design eliminates the need for externally trained weak models or additional data, while maintaining competitive performance and extremely low computational overhead. The minimal intervention required for implementation highlights the method’s engineering practicality and reproducibility.

3. Strong Empirical Validation and Robust Generalization. The experiments convincingly demonstrate that S²-Guidance enhances both sample fidelity and distributional consistency across diverse models and tasks. The method consistently improves over CFG and performs on par with, or better than, external weak-model guidance approaches—all without additional training or tuning. The results underscore its robustness and broad applicability to various diffusion architectures.

4. Well-Written and Structured Presentation.

### Weaknesses
1. Although the proposed method is simple yet efficient, its novelty remains limited. The authors introduce a more general form of the “weak” model; however, it still primarily targets essential problem-solving within existing frameworks rather than presenting a fundamentally new paradigm.

2. While the inclusion of the additional sub-optional model contributes to performance improvement, it also introduces extra computational cost and resource consumption.

3. The improvements in the text-to-video experiments are primarily reflected in enhanced visual quality. However, as shown in Table 4, the temporal consistency of the generated videos slightly degrades—particularly in terms of subject and background consistency. This suggests that the proposed method may offer limited benefits for video generation while demonstrating stronger advantages in image-level synthesis.

### Questions
None

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
This paper extends the idea of autoguidance, which generates negative predictions via stochastic dropout and improves sampling through extrapolation, by introducing block dropout to form negative predictions and formulate guidance. While selectively dropping certain blocks improved performance, it introduced a block selection problem, determining which blocks to drop. The authors address this issue by applying stochastic dropout during the sampling process, proposing S2-Guidance, and demonstrate improved performance on both GMM toy examples and real datasets.

### Strengths
- The comparison and visualization of different guidance methods on GMM data are clear and well-presented.
- Using block dropout instead of parameter dropout contributes to inference efficiency, akin to structured pruning, compared to conventional dropout.
- The observation that the specific choice of dropout block has limited impact on performance is important.
- Replacing manual block selection with stochastic dropout simplifies usage while maintaining or even improving general performance, making the approach more practical.

### Weaknesses
There are several concerns regarding the use of block dropout. Dropping entire model blocks could severely degrade prediction quality if applied to critical blocks, potentially corrupting the model’s behavior. While AutoGuidance adopts parameter-level dropout, this paper employs the stronger perturbation of block-level dropout without providing sufficient motivation or references for this design choice. 

Indeed, as mentioned around L250-L254, applying naive S2-Guidance to key blocks drastically reduces model performance, suggesting that the block dropout mechanism’s impact is under-analyzed. Providing a visualization of how predictions change when each block is dropped would substantially strengthen the motivation and understanding of this method. 

Furthermore, in L161-, the paper claims to offer a more general solution compared to previous weak model guidance methods, which relied on hand-crafted designs. However, since the proposed method still requires optimizing the dropout probability and potentially identifying key blocks to avoid, it may also involve a certain level of manual tuning, making the claim of being fully general somewhat overstated. 

Although S2-Guidance removes explicit block selection by introducing stochastic dropout per timestep, the paper does not discuss whether dropping key blocks during random selection leads to performance degradation or whether the method produces consistent performance across different runs. A discussion or variance analysis on this point would be valuable. In addition, since this method perturbs model weights, the paper should further clarify the advantages and motivation of block-level dropout over the stochastic parameter dropout used in AutoGuidance. 

Finally, the approach resembles PAG (Ahn et al., 2024), which applies attention masking to generate degraded samples and extrapolate them for guidance. It would be beneficial to include a comparison with PAG, as such a baseline could reinforce the paper’s empirical claims.

Minor

- In L158, the description of Hong et al. (2023) is inaccurate: the method performs attention-guided blurring of predicted samples, not blurring of attention maps themselves.

### Questions
- In Eq. (3), the guidance is formulated by subtracting the ω-scaled negative prediction, differing from the standard extrapolation-based CFG formulation.
- What is the rationale behind this design? Would applying similar ω-scaled subtraction to other bad-version guidance methods yield comparable effects?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces S²-Guidance, a training-free modification to classifier-free guidance (CFG) for diffusion models. At each denoising step, the method forms a stochastic “weak” sub-network by randomly dropping some transformer blocks and subtracts its prediction (scaled by ω) from the standard CFG result. The goal is to mitigate CFG’s over-confident or semantically inconsistent generations while preserving distributional fidelity. The paper reports consistent improvements on several text-to-image and text-to-video benchmarks with transformer-based backbones.

### Strengths
- Training-free inference procedure that does not require auxiliary models.
- Broad experimental coverage across multiple transformer-based diffusion backbones (images and video) with clear qualitative examples.
- Ablation studies on several design choices (ω, drop ratio) and comparisons against a range of CFG variants.
- The idea is easy to understand and implement for transformer architectures, and the empirical results are consistently positive.

### Weaknesses
- Computational overhead: per denoising step the method requires an additional denoiser call (three evaluations vs. two for standard CFG). The efficiency claim is mainly relative to a heavier “naive” multi-sample variant rather than to CFG. The phrasing of the term "efficient" in the main text is therefore slightly misleading.
- Architecture dependence: all demonstrations are on transformer-based diffusion models and the mechanism relies on dropping entire residual blocks with fixed shapes. It is unclear how to apply the same stochastic sub-network idea to UNet/CNN backbones without nontrivial changes.
- Theoretical framing is largely interpretive. The Bayesian idea discussed in the appendix is suggestive but not shown to tighten score estimation or likelihood in a formal sense.
- Scope of validation: no compute/runtime comparisons versus CFG are provided, and the variance introduced by the stochastic dropping is not discussed.

### Questions
1. Applicability to UNet/CNN architectures: please explain concretely how the stochastic sub-network would be constructed for UNet-style diffusion models (e.g., which modules are droppable without breaking shapes/skip connections) and provide some experiments in this direction to support the generality claim.
2. Compute efficiency: please report runtime or FLOP ratios relative to standard CFG for a typical sampling trajectory, and include wall-clock time where possible.
3. Theory: in the Bayesian derivation, why should “repulsion from the posterior mean” improve sample fidelity or score accuracy? 
5. Output variance: does the stochastic mask introduce noticeable variance across runs with identical seeds?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces $S^2$-Guidance (Stochastic Self-Guidance), a training-free guidance algorithm for diffusion models. The authors have empirically shown that Classifier-Free Guidance (CFG) can produce suboptimal results. To address this issue, they proposed activating subnetworks of the same model through stochastic block dropping during denoising to obtain a “weak” prediction that corrects the CFG direction. The proposed $S^2$-Guidance improves fidelity across class-dependent ImageNet, text-to-image, and text-to-video generation, outperforming CFG and current guidance algorithms.

### Strengths
- The paper is well-written and easy to understand.
-  Activating an internal “weak” predictor by randomly omitting blocks during inference is simple but effective. Unlike distillation-based methods and external weak models, this algorithm is training-free and plug-and-play. It can therefore be applied directly to the generation process, which is currently based on CFG.
- An additional stochastic forward pass per step keeps complexity low compared to many other alternatives with guidance or self-ensemble, making the ratio between quality and computational effort efficient in practice.

### Weaknesses
- In Appendix A, the authors provides an analysis of $S^2$-Guidance and Naive $S^2$-Guidance. The authros "posit" that $S^2$-Guidance is an approximately unbiased estimator of the "expected guidance" $G_{Naive}$. However, the "unbiasedness" of $G_{S^2-Guidance}$ is assumed, not shown. In addition, 
$\mu_{post}$ denotes a Bayesian posterior mean, but the algorithm actually only defines an empirical average of the predictions of the sub-network obtained by randomly dropping modules, which cannot follw eq. (25). 
- The author must argue that the variance of $G_{S^2-Guidance}$ is small, or specify concentration bounds so that one guidance per step is reliable.
- A randomly drawn sub-network leads to a deviation in the trajectory. Sensitivity and stability under randomness have not yet been sufficiently investigated in the paper.
- For videos, stochastic guidance can lead to temporal flickering or shifts in object identity. Therefore, please add temporal stability metrics to verify the quality of the generated video.
- The proposed $S^2$-Guidance requires an extra forward pass per step. Thus, FLOPs and peak memory requirements must be reported.

### Questions
- How large in the variance of the single stochastic block-dropping operation?
- How dose $S^2$-Guidance affect long composition prompts, text rendering, or high-frequency textures?

### Soundness
3

### Presentation
3

### Contribution
2
