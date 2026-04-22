# Score Distillation Beyond Acceleration: Generative Modeling from Corrupted Data

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Learning generative models directly from corrupted observations is a long-standing challenge across natural and scientific domains. We introduce **Restoration Score Distillation (RSD)**, a unified framework for learning high-fidelity, one-step generative models using **only** degraded data of the form
$
y = \mathcal{A}(x) + \sigma \varepsilon, x\sim p_X,\ \varepsilon\sim \mathcal{N}(0,I_m),
$
where the mapping $\mathcal{A}$ may be the identity or a non-invertible corruption operator (e.g., blur, masking, subsampling, Fourier acquisition). RSD first pretrains a corruption-aware diffusion teacher on the observed measurements, then *distills* it into an efficient one-step generator whose samples are statistically closer to the clean distribution $p_X$. The framework subsumes identity corruption (denoising task) as a special case of our general formulation.

Empirically, RSD consistently reduces Fr\'echet Inception Distance (FID) relative to corruption-aware diffusion teachers across noisy generation (CIFAR-10, FFHQ, CelebA-HQ, AFHQ-v2), image restoration (Gaussian deblurring, random inpainting, super-resolution, and mixtures with additive noise), and multi-coil MRI—*without access to any clean images*. The distilled generator inherits one-step sampling efficiency, yielding up to $30\times$ speedups over multi-step diffusion while surpassing the teachers after substantially fewer training iterations. These results establish score distillation as a practical tool for generative modeling from corrupted data, *not merely for acceleration*. We provide theoretical support for the use of distillation in enhancing generation quality in the Appendix. The code is available at https://github.com/TianyuCodings/RSD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a unified and powerful framework, Distillation from Corrupted Data (DCD), for a long-standing and critical challenge: training high-fidelity generative models using only corrupted or degraded data. The core idea is a two-phase approach: first, a "corruption-aware" diffusion model (teacher) is trained directly on the imperfect observations; second, this slow, multi-step teacher is distilled into a highly efficient one-step generator (student). The central and most significant finding is that this distillation process does not merely accelerate sampling but consistently and substantially *improves* the generative quality, often surpassing the teacher by a large margin. The authors provide an exceptionally thorough empirical validation across a wide array of tasks—including denoising, deblurring, inpainting, super-resolution, and real-world multi-coil MRI reconstruction—demonstrating state-of-the-art performance in the challenging "zero-shot" setting where no clean data is ever used.

### Strengths
1. The paper tackles a problem of immense practical importance. The discovery that score distillation can serve as a powerful tool for *improving* generation quality from corrupted data, rather than just for acceleration, is a significant conceptual contribution that could shift how we approach learning from imperfect sources.

2. The experimental results are comprehensive and consistently impressive. DCD achieves remarkable FID improvements over its teachers across all tested corruption types, datasets, and severity levels. Its ability to outperform specialized "few-shot" methods without access to any clean data is a testament to the framework's power and robustness.

3. The two-phase design is clean, intuitive, and highly flexible. By decoupling the corruption-aware pretraining from the distillation, the framework can readily incorporate new and future advances in diffusion modeling for inverse problems. This modularity makes DCD a highly practical and future-proof solution.

4. The successful application to the multi-coil MRI problem, a challenging scientific domain with complex-valued data and real acquisition constraints, strongly underscores the real-world viability and potential of this work.

### Weaknesses
1. While the paper mentions theoretical support in the appendix, the main text could benefit from a more intuitive explanation of *why* distillation provides such a significant quality boost in the corrupted data regime. Is it a form of regularization? Does it find a more stable generator mode? A brief, high-level discussion would further strengthen the paper's narrative.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces **Distillation from Corrupted Data (DCD)**, a two-stage framework for training one-step generative models directly from corrupted observations without access to clean data. The method first trains a corruption-aware diffusion teacher on corrupted data (e.g., noisy, blurred, masked, or undersampled measurements), then distills it into a one-step generator using a score-matching objective that respects the corruption operator. The authors demonstrate state-of-the-art performance across a variety of corruption types (denoising, deblurring, inpainting, super-resolution, MRI reconstruction) on multiple datasets (CIFAR-10, CelebA-HQ, FFHQ, AFHQ-v2, FastMRI), consistently improving FID over teacher models while achieving up to 30× speedup in inference.

### Strengths
- **Originality:** Combines corruption-aware diffusion training with score distillation in a unified framework, with a focus on real-world scenarios where clean data is inaccessible.
- **Quality:** Extensive experiments across multiple tasks, datasets, and noise levels demonstrate robust and consistent improvements.
- **Clarity:** The paper is well-written, with clear explanations of both methodology and experimental design.
- **Significance:** Addresses a critical challenge in generative modeling for scientific and medical imaging, with practical speedups and performance gains.

### Weaknesses
- **Limited Comparison to GANs:** While diffusion-based methods are the focus, a comparison with GAN-based approaches trained on corrupted data could provide a broader perspective.
- **Complexity of Theoretical Analysis:** The theoretical section is dense and may be challenging for readers without a strong background in divergence metrics and Gaussian analysis.
- **Hyperparameter Sensitivity:** The performance of distillation losses (e.g., SiD vs. DMD) may depend on hyperparameter tuning, which was not fully explored.

### Questions
1. **Generalization to Non-Linear Operators:** Can DCD handle non-linear corruption operators (e.g., non-linear blur, sensor saturation)? If not, what are the limitations?
2. **Robustness to Mismatched Corruptions:** How does DCD perform if the corruption operator during distillation differs from that during pretraining?
3. **Scalability to Higher Resolutions:** Have you tested DCD on higher-resolution datasets (e.g., ImageNet, 256×256)? Are there any architectural or training adjustments needed?
4. **Comparison with GANs:** Have you considered comparing with GAN-based methods that also train on corrupted data?
5. **Real-World Deployment:** Are there any latency or memory constraints when deploying the one-step generator in real-time applications (e.g., MRI reconstruction)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a two-stage framework named DCD for learning a one-step generative model from corrupted observations. Stage one involves training a diffusion model teacher on the corrupted data using a suitable corruption-aware objective. Stage two distills this teacher into a one-step generator, where the key step is to apply the known corruption operator to the generator's output during the distillation loss computation. Extensive experiments show that the second stage not only makes the model fast but also significantly improves sample quality (FID) over the teacher model across various restoration tasks and MRI reconstruction.

### Strengths
1. The paper addresses an important practical problem. Learning from corrupted data is particularly useful in scientific areas like medical imaging. A notable contribution of this paper is that it shows that score distillation can improve generation quality from corrupted data in addition to accelerating sampling.

2. The method is empirically strong. The proposed method is tested on multiple datasets, diverse corruption types, and a real-world application. The gains over the teacher models are consistent and significant.

3. The two-phase design is practical to allow leveraging the rapidly growing literature on corruption-aware diffusion models for the teacher pretraining stage. This makes the framework adaptable.

### Weaknesses
1. The primary novelty of framework lies in specific composition of established techniques and the empirical validation that distillation is particularly effective in this setting. It leverages corruption-aware diffusion training and score distillation as constituent elements but is limited in inventing new components. The manuscript would be strengthened by more precise positioning of its contribution and clarifies its novelty and contribution in synthesizing existing methods to solve a new problem, rather than the introduction of fundamentally new techniques.

2. The central claim that distillation improves quality is empirical. A deeper and more intuitive analysis of the underlying mechanism is necessary to clarify the claim. In addition, the crucial question on the reason that enforcing self-consistency on a corrupted output space results in a more faithful representation of the clean data manifold remains unaddressed in the main text. Existing explanation for this phenomenon is cursory, and the core theoretical justification is deferred to the appendix.

### Questions
Please refer to the section of Weaknesses. I suggest that the authors provide a precise positioning of their contribution and a more intuitive analysis on the underlying mechanism to explain why distillation improves quality.

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
4

### Summary
This paper proposes a framework named Distillation from Corrupted Data (DCD) for learning high-fidelity, one-step generative models from corrupted data. The framework is implemented in two stages. A diffusion model is first pretrained on the corrupted data as a teacher model, and the knowledge of the teacher model is then distilled into an efficient one-step generator. The contribution of this manuscript lies in applying score distillation technology to learning generative models from corrupted data, providing a new solution for the relevant field.

### Strengths
1. The paper introduces DCD, a novel framework for learning generative models from corrupted data without needing clean data. This approach is original in its unified treatment of diverse corruptions.

2. The research is of high quality, with a rigorous two-phase framework combining corruption-aware diffusion pretraining and score distillation. Theoretical analysis supports the method, and comprehensive experiments validate its effectiveness across various datasets and tasks.

3. The proposed DCD framework is potential to improve generative modeling in scenarios with limited clean data.

### Weaknesses
1. The DCD framework extends score distillation to corrupted data but lacks novel methodological contributions. Score distillation has already been widely used for learning generative models from clean data, and is extended to the scenario of corrupted data in this paper. However, it seems that DCD relies on existing diffusion models and score distillation techniques and offers limited innovations and does not provide unique paradigm for learning from corrupted data. 

2. In the distillation phase, the authors used an auxiliary fake diffusion model to approximate the distribution induced by the generator. The use of an auxiliary fake diffusion model during distillation introduces potential approximation errors. This method may not accurately capture the distribution induced by the generator and affect the overall performance of the framework.

3. Experimental results are not sufficient to validate the effectiveness of DCD.

i) Evaluations are performed on small, single-type datasets like CIFAR-10 and CelebA-HQ. The results on these small, homogeneous datasets cannot demonstrate the generalization of DCD on different image sizes, diverse classes, or various corruption types/levels/schedules.

ii) In most experiments, FID is adopted as the main evaluation metric. There lacks of a comprehensive assessment of the performance of DCD under varying metrics.

iii) Ablation studies on the fake diffusion surrogate, shared initialization, and training stability are missing. Both the fake diffusion model and the generator are initialized from the teacher model. Since the teacher model itself may learn incorrect features, this could cause staying in local optima during training. 

4. In the mathematical derivation of score distillation, the authors assume that the score fields between the teacher model and the generator could be aligned, but do not explain the rationality of the assumption or provide supporting evidence.

5. Minor issue. The names of the horizontal and vertical coordinates are missing in Figure 12.

### Questions
1. What are the innovative points of the DCD framework in terms of methodology and how do these innovative points give it a unique advantage in learning generative models from corrupted data?

2. What is the additional error introduced by approximation with auxiliary fake diffusion model?

3. How do the results change with resolution and class diversity?

4. What is the performance of DCD under more metrics like KID, Precision/Recall, and Density/Coverage?

5. Could the authors provide ablation studies on the fake diffusion surrogate, shared initialization, and training stability?

6. What is the rationality of the assumption that the score fields between the teacher model and the generator could be aligned?

### Soundness
2

### Presentation
3

### Contribution
2
