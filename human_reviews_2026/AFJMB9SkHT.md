# FideDiff: Efficient Diffusion Model for High-Fidelity Image Motion Deblurring

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Recent advancements in image motion deblurring, driven by CNNs and transformers, have made significant progress. Large-scale pre-trained diffusion models, which are rich in real-world modeling, have shown great promise for high-quality image restoration tasks such as deblurring, demonstrating stronger generative capabilities than CNN and transformer-based methods. However, challenges such as unbearable inference time and compromised fidelity still limit the full potential of the diffusion models. To address this, we introduce FideDiff, a novel single-step diffusion model designed for high-fidelity deblurring. We reformulate motion deblurring as a diffusion-like process where each timestep represents a progressively blurred image, and we train a consistency model that aligns all timesteps to the same clean image. By reconstructing training data with matched blur trajectories, the model learns temporal consistency, enabling accurate one-step deblurring. We further enhance model performance by integrating Kernel ControlNet for blur kernel estimation and introducing adaptive timestep prediction. Our model achieves superior performance on full-reference metrics, surpassing previous diffusion-based methods and matching the performance of other state-of-the-art models. FideDiff offers a new direction for applying pre-trained diffusion models to high-fidelity image restoration tasks, establishing a robust baseline for further advancing diffusion models in real-world industrial applications. Our dataset and code will be available at https://github.com/xyLiu339/FideDiff.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a single-step, diffusion-based deblurring model that improves the pixel and perceptual fidelity of deblurring results. Motion deblurring can be formulated as a diffusion-like process, with each timestep representing an increasingly blurred image. A kernel control net is proposed for kernel estimation and adaptive timestep prediction. This approach outperforms existing diffusion-based methods and achieves superior performance on reference-based metrics.

### Strengths
The proposed method is novel, and the manuscript is well organised and written. FideDiff significantly improves the fidelity of deblurring results, outperforming other diffusion-based models. It may also encourage other researchers to explore diffusion-based, high-fidelity methods for other restoration tasks.

### Weaknesses
1.	In Section 3.4, the mapping function between the number of averaging frames and the timestep is t = g(n) = (n - 1) * 20. Why was this set?
2.	The symbol * is used many times in the manuscript, but it has different meanings in each instance. For example, in Eq. (4), it represents a convolution operation. However, in the mapping function t = g(n), it may represent a multiplication operation. Please clarify this.
3.	The proposed framework uses a learnable text embedding as a condition for prediction. However, this is lacking in the framework architecture shown in Fig. 4. Moreover, what are the text descriptions used in the proposed method? Is the CLIP model used in the proposed framework?
4.	Figure 4 lacks the symbol k_t, which represents a blur kernel. Furthermore, Z_(in1) = Conv(Z_t) should be Z_(in1) = Conv(Z_(LQ)(Z_t)). Where is k_(in) in Fig. 4? What do the orange and blue blocks represent? In the filter-like module F, these lines lack arrows. Does * represents the element-wise multiplication? Why are there two lines with arrows showing the output of the DMs?
5.	What is the EA-LPIPS function? How does it influence the performance of image deblurring? Please conduct experiments to evaluate this.
6.	What is the meaning of * in Eq.(12)?
7.	I think the proposed method could be extended to other low-level tasks, such as super-resolution and deraining. However, it is unclear how it can be extended to image compression, given that this task does not involve modelling physical degradation.
8.	In the implementation details, I see that you perform 2x upsampling before inference and then resize the outputs back to their original spatial size. What interpolation operation is used here, and might it cause detail loss?

### Questions
Please See the weaknesses part.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a novel motion deblurring architecture based on a pre-trained stable diffusion model, successfully distil the multi-step diffusion model into a single-step model.

### Strengths
1. The performance shown in the paper is promising. Even though the model is trained on synthetic data, it shows better performance on all datasets compared with other diffusion-based methods.
2. Solid mathematical analysis. Although the markov prediction of kt from kt-1 is infeasible to calculate, following the training assumption of diffusion model, every training step share the same sharp ground truth, which makes the single-step possible.
3. The paper proposed an efficient single-step diffusion model that avoid the loss of diffusion inductive bias as other one-step methods did, which reduce the sensitivity of the method to spatial-variant and severe blurs. The key thing here is the proposed time-consistency training. 
4. High-fidelity results, which surpass both conventional and diffusion based methods.
5. Properly modify the controlnet to better inject the blur kernel as an extra condition.

### Weaknesses
1. How to generate z_0 with input Z_LQ is unclear. Needs further clarification.
2. The training data compatible of this architecture are limited, which reduces the generalization of the proposed method.
3. How to concatenate latent feature z with the estimated blur kernel?

### Questions
1. What is the blue block in Figure 4?
2. Why can't the model be trained end-to-end instead of using three stages.
3. Any evaluation of time prediction? I am not sure how accurate it is.
4. The data generation process in supp is unclear to me. What is the meaning of "For the 13-frame average sample, using the middle sharp frame as a reference, we synthesize blurry images by averaging 15 and 11 frames" and "For the 11-frame average sample, we employ a probabilistic augmentation method. First, we synthesize all 13-frame combinations. Then, we generate blurry images by averaging 9 and 15 frames based on a probability distribution of [0.3, 0.7]."

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FideDiff, a single-step diffusion model that reformulates motion deblurring as a diffusion-like process where each timestep corresponds to a progressively blurred image rather than Gaussian noise levels.
The core technical contributions include a consistency training paradigm where the model learns to map all blur levels along a trajectory to the same clean image and Kernel ControlNet, which integrates blur kernel estimation into the diffusion model via a filter-like module
The model is built upon Stable Diffusion 2.1 and evaluated on GoPro, HIDE, and RealBlur datasets.

### Strengths
1. **Problem Formulation Is Reasonable**: The paper clearly discusses the limitations of existing diffusion-based deblurring methods (significant inference time, fidelity-perception tradeoff) and provides concrete evidence (Figure 2) showing the tradeoff between steps and fidelity in super-resolution as motivation.

2. **Consistency Training Framework**: The reformulation of deblurring as a diffusion-like process where timesteps correspond to blur severity rather than noise levels is reasonable. The insight that temporal consistency can enable single-step inference is valuable.

3. **Comprehensive Experimental Evaluation**: The paper includes extensive comparisons across multiple datasets (GoPro, HIDE, RealBlur-J, RealBlur-R) and metrics (PSNR, SSIM, LPIPS, DISTS), with both transformer-based and diffusion-based baselines. Tables 4-7 and Figures 7-8 systematically evaluate the contribution of each component (foundation model design choices, Kernel ControlNet variants, consistency training, timestep prediction).

### Weaknesses
1. **Conceptual Novelty and Practical Advantages** Single-step diffusion with Kernel ControlNet and GAN discriminator is conceptually similar to transformer-based models with discriminators. The main distinction is initialization from pre-trained Stable Diffusion weights. However, this does not translate to practical advantages, the performance (both speed and metrics on academic benchmarks) is still a bit inferior or similar to transformer-based methods 

2. **Dataset Scale**
The training relies on the enlarged GoPro dataset containing only ~7,877 image pairs, which might be too small to efficiently fine-tune large-scale generative models, which partially can explain the performance gap.

Minor:
- The filter-like module F (Section 4.2) is presented as novel, but it's a relatively standard way to inject spatial information
- The improvement over vanilla ControlNet is small: 0.06 PSNR (28.73 → 28.79) showing that theoretically correct kernel fusion doesnt have signifact impact on model performance.
- Table 5 also shows "motion align" module performs similarly (28.75 PSNR), suggesting the specific design choice is not critical
- Line 013: "true-world modeling" should be "real-world modeling"

### Questions
- What is the actual benefit of using pre-trained diffusion models if both performance and speed are inferior to specialized transformer models? This brings an interesting discussion. The authors claim that generative models produce higher image quality due ot their ability to generate/hallucinate scene details. Yet in most examples in the supplementary, the result is nearly identical across all state-of-the-art models. In addition, the evaluation still focuses on standard image similarity metrics: PSNR / SSIM / LPIPS, favoring noisier solutions that hallucinate less (transformer-based models). To demonstrate the effectiveness of the method, authors should consider computing image quality metrics (FID and others) or conducting a user preference study.
- Do you think the current approach might require orders of magnitude more data than transformer methods to achieve superior performance to transformer-based methods?

### Soundness
3

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
3

### Summary
The paper proposes applying a diffusion-model to the single-image deblurring task. The authors claim that by diffusion process, they achieve high-fidelity sharp image restoration on datasets such as GoPro and RealBlur‑R.

### Strengths
The idea of using a diffusion-model for deblurring is interesting in the deblurring community.

The paper presents a comprehensive experimental evaluation with multiple datasets, which shows the authors’ effort to validate the method across settings.

The writing is clearly structured, and the method is described with reasonably good clarity in terms of pipeline, training and inference details.

### Weaknesses
1. Numbers inconsistent with public SOTA.

On GoPro, the paper reports PSNR 28.79 dB, far below recent results, e.g., 34.09 dB LoFormer; 34.60 dB AdaRevD. On RealBlur-R, it reports 36.01 dB, while widely reported numbers are 38.6 dB. This raises concerns about evaluation setup and fairness.

2.Perceptual evaluation is insufficient.

For a diffusion model, perceptual quality should be central. The paper underweights CLIP-IQA/LPIPS/DISTS and provides limited qualitative/video evidence. No user study or side-by-side videos under realistic motion is shown.

3. Cost–benefit trade-off is unclear.

The method adds diffusion + kernel control modules, yet runtime, memory, and step count are not thoroughly compared to strong CNN/Transformer baselines under the same hardware. Efficiency claims are therefore hard to judge.

4. Ablations not fully isolating gains.

The paper does not cleanly separate the contributions of the foundation diffusion, Kernel ControlNet, and t-prediction across datasets. Stronger ablations and cross-dataset tests are needed to establish robustness.

### Questions
Could you provide more visual materials such as videos?

### Soundness
2

### Presentation
3

### Contribution
3
