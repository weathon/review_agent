# KinemaDiff: Towards Diffusion for Coherent and Physically Plausible Human Motion Prediction

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Stochastic Human Motion Prediction (HMP) has become an essential task for the realm of computer vision, for its capacity to anticipate accurate and diverse future human trajectories. Current diffusion-based techniques typically enforce skeletal consistency by encoding structural priors into network architectures. Although effective in promoting plausible kinematics, this approach provides only indirect control over the generative process and often fails to guarantee strict physical constraint satisfaction. In this work, we propose a structure-aligned and joint-aware diffusion framework that enforces physical constraints by embedding skeletal topology and joint-specific dynamics directly into the diffusion process. Specifically, our framework consists of two key modules, the Joint-Adaptive Noise Generator and the Structure-Aligned Constraint Enforcer. The former component, Joint-Adaptive Noise Generator, infers joint-specific dynamics and injects
heterogeneous, instance-aware noise per joint and sample to capture spatial variability and enhance motion diversity. The latter component, Structure-Aligned Constraint Enforcer, encodes skeletal topology by modeling joint connectivity and bone lengths from historical motions, and it constrains each denoising step to preserve anatomical consistency. Through their synergistic operation, these modules grant KinemaDiff direct control over physical realism and motion diversity, addressing the common limitations of indirect structural priors and uniform noise application. Extensive experiments on multiple benchmarks demonstrate the effectiveness of our method, attributable to tailoring the diffusion process through structural alignment and joint-adaptive noise modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method to address stochastic human motion prediction, particularly by integrating anatomical consistency into the generative diffusion process. The paper proposes: (a) a Joint-Adaptive Noise Generator, which learns noise scheduler specifically tailored for different join, to take into account its characteristics in the human dynamic, (b) a Structure-Aligned Constraint Enforcer, which integrates the bone length information into the diffusion process by a training loss, enforcing that the avarage length of the bones in the prediction aligns with the prediction in the observed past. The paper compares with relevant state-of-the-art methods, showing promising results on AMASS and H3.6M.

### Strengths
- The idea of learning noise schedulers for different individual joints is interesting, and as far as I know, not yet explored
- The presentation is clear, and the paper is easy to follow.

### Weaknesses
- The "Structure-Aligned Constraint Enforcer" seems to be a simple loss that penalizes the error in the bones' length between the observed past and the prediction. Hence, its contribution seems a bit overstated in the paper, as well as the name is a bit deceptive. Also, in the ablation, it is not reported the case where Encoder and J-Noise are activated, but the Enforcer is not. This would be important to actually assess the contribution of this component.

- Although the paper introduces a loss specifically to take into account bone length, there are no metrics to measure this aspect.  This is surprising, as Skeleton Diffusion actually proposed to measure limb stretch and jittering across the prediction, since it has been observed that more diverse results correlate with unrealistic deformation of the skeleton. 

- The paper does not include an analysis of what is learned by the Joint-Adaptive Noise Generator. I would find it interesting, as it can provide insights and interpretability in the proposed framework.

MINOR:
The work does not discuss/compare with [1,2]

[1]: Xu, S., Wang, Y. X., & Gui, L. Y. (2022, October). Diverse human motion prediction guided by multi-level spatial-temporal anchors. In European Conference on Computer Vision (pp. 251-269). Cham: Springer Nature Switzerland.

[2]: Xu, G., Tao, J., Li, W., & Duan, L. (2024, September). Learning semantic latent directions for accurate and controllable human motion prediction. In European Conference on Computer Vision (pp. 56-73). Cham: Springer Nature Switzerland.

### Questions
Currently, I slightly lean toward rejection: I find the proposed modification interesting, but it is also quite limited as a contribution, and some evidence is still missing to fully establish its validity. For the rebuttal, I'd like to see the following questions addressed:

1) Is it possible to clarify the "Structure-Aligned Constraint Enforcer" contribution, in light of the weaknesses above?
2) Would it be possible to see the metrics about limbs jittering and scratching for your comparison setting?
3) Would it be possible to include the ablation where only the Enforcer is removed?
4) A minor question lies in the nature of the choice of not relying on a latent diffusion, though the J-Noise module could also be applied there. Is there any reason or intuition behind this choice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces KinemaDiff, a novel diffusion-based framework for stochastic human motion prediction designed to improve physical plausibility and motion diversity. Conventional methods often suffer from anatomical inconsistencies and ignore heterogeneous joint dynamics. KinemaDiff addresses this by integrating two core modules directly into the diffusion process: (1) a Joint-Adaptive Noise Generator, which learns to inject instance-aware, heterogeneous noise for each joint based on its specific dynamics and motion history, and (2) a Structure-Aligned Constraint Enforcer, which preserves anatomical consistency by embedding skeletal topology (i.e., bone lengths) as an alignment loss during the denoising process. Experiments on the Human3.6M and AMASS datasets demonstrate that KinemaDiff achieves state-of-the-art performance, particularly in prediction accuracy (ADE/FDE) and realism (FID/CMD).

### Strengths
The paper's primary strength lies in its fundamental adaptation of the diffusion process itself, rather than merely modifying the denoiser's network architecture. It embeds kinematic and anatomical priors directly into the noising and denoising steps.

The Joint-Adaptive Noise Generator is a significant innovation. Moving beyond uniform or static anisotropic noise, this module learns a noise schedule conditioned on both the joint index and its specific motion history. This is more physically grounded, as different joints (e.g., wrist vs. hip) and different motions (e.g., walking vs. jumping) inherently have different degrees of freedom and stochasticity.

The Structure-Aligned Constraint Enforcer directly tackles a major failure mode of generative models, physical implausibility (e.g., stretching bones). By defining an explicit alignment loss based on bone lengths from observed motion, the model is strongly guided to produce anatomically consistent skeletons.

The method achieves state-of-the-art results on two standard benchmarks (Human3.6M and AMASS). The improvements in accuracy metrics (ADE/FDE) and, most notably, realism metrics (FID/CMD) are substantial, validating the effectiveness of the proposed components.

The ablation in Table 3 clearly demonstrates the contribution of each module. The Structure-Aligned module ("Align") provides the most significant boost in accuracy (ADE/FDE) and realism (FID), while the Joint-Adaptive Noise ("J-Noise") further improves these metrics, confirming their synergistic value.

### Weaknesses
The name "Structure-Aligned Constraint Enforcer" is a misnomer. The method described is a soft constraint or regularizer implemented via a loss function. It "encourages" anatomical consistency during training but does not "enforce" it, meaning it does not mathematically guarantee that the final output will be physically plausible.

The alignment loss is defined as the discrepancy between the average bone length over the observed history and the average bone length over the entire predicted future sequence. This is a very weak constraint. It allows for physically impossible motions (e.g., a bone shrinking in one frame and stretching in another) as long as the average length over the sequence is correct. A true physical constraint would require bone lengths to be (nearly) constant at every predicted frame.

Section 3.5 is confusing. It states, "at each timestep, after the initial encoder, we apply the same operation on $y_0$ to ensure that the human skeleton structure remains consistent". This implies the alignment loss is applied at every step $t$ of the denoising process. However, the overall loss function and the definition of the alignment loss based on the final prediction suggest it is only applied at the end of the training step. This ambiguity is critical for reproducibility.

The results (e.g., Table 1) show that while KinemaDiff excels in accuracy and realism, its diversity score (APD) is lower than several baselines. While this is a common trade-off, the paper could benefit from a deeper discussion on whether this is a fundamental consequence of enforcing stricter physical realism or a limitation of the current noise generation module.

### Questions
Regarding the "Constraint Enforcer": Did you experiment with a stronger, frame-wise alignment loss? For example, penalizing the L1 or L2 deviation from the observed bone length at each predicted future frame $f \in [1, F]$, rather than just penalizing the deviation of the sequence-level average? If so, how did this impact training stability and the final metrics?

Could you please clarify the exact mechanism described in Section 3.5? Is the alignment loss computed and backpropagated at every denoising timestep $t$ during training? Or is it only applied once to the final prediction $\hat{y}_0$, as suggested by Equation 12?

For the Joint-Adaptive Noise Generator, the function $f_{\theta}$ maps the joint index $j$ and its full history $x_j^{(1:H)}$ to a scaling factor $s_j$. Given that the paper describes this as "a few linear layers", how is the variable-length temporal history $x_j^{(1:H)}$ aggregated into a fixed-size vector to be processed by these layers?

The results show a trade-off between realism/accuracy and diversity (APD). In your view, is this lower diversity a necessary consequence of enforcing strict anatomical constraints (i.e., the model correctly "prunes" unrealistic but diverse motions), or do you see potential in the Joint-Adaptive Noise module to further enhance diversity while maintaining the high level of realism?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents KinemaDiff, a diffusion-based framework for stochastic human motion prediction. The primary idea is to embed kinematic heterogeneity and skeletal consistency directly within the denoising process. Two components are introduced:

**Joint-Adaptive Noise Generator** — assigns heterogeneous Gaussian noise scales to each joint and dynamically modulates them using temporal features derived from the observed motion history.

**Structure-Aligned Constraint Enforcer** — enforces bone-length invariance at each denoising step based on topology extracted from past motion sequences.

Experiments on Human3.6M and AMASS demonstrate superior performance over recent baselines in accuracy and realism, supported by ablation studies and qualitative visualizations.

### Strengths
Strong state-of-the-art results on Human3.6M and AMASS in terms of accuracy and realism.

Effective integration of heterogeneous noise and skeletal constraint within the generation process, improving physical plausibility.

Comprehensive experiments, including ablations, quantitative benchmarks, and qualitative visualization, with consistent performance gains.

Process-level enforcement of anatomical constraints mitigates unrealistic poses that prior methods often handle via post-processing.

### Weaknesses
1. The Joint-Adaptive Noise Generator is quite similar to SkeletonDiffusion[1]’s anisotropic noise, with the main addition being a temporal feature to adjust the scale per joint.

2. The Structure-Aligned Constraint Enforcer uses the common bone-length consistency idea seen in prior motion generation work (e.g. [2]), mainly changing it to be applied during each diffusion step.

3. The gains in quantitative metrics are modest, and the visual results are only compared with CoMusion, which limits the strength of the realism claim.

[1]Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction, CVPR25
[2]InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions, IJCV2024

### Questions
1. I’d be curious to see a clearer ablation of the J‑Noise design — for instance, what’s the gain if you just use per‑joint independent noise instead of uniform noise, and then how much extra does adding the temporal information actually help?

2. Can you explain a bit more how your Structure‑Aligned Constraint Enforcer is different from the usual bone‑length consistency losses used in motion generation? It might also be worth running a simple ablation to show its specific impact.

3. Would it be possible to add visual comparisons with a couple more baselines, like SkeletonDiffusion and BeLFusion, so the realism claim is backed by broader evidence?

### Soundness
2

### Presentation
3

### Contribution
2
