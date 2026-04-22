# Training-Free Distribution Adaptation for Diffusion Models via Maximum Mean Discrepancy Guidance

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Pre-trained diffusion models have emerged as powerful generative priors for both unconditional and conditional sample generation, yet their outputs often deviate from the characteristics of user-specific target data. Such mismatches are especially problematic in domain adaptation tasks, where only a few reference examples are available and retraining the diffusion model is infeasible. Existing inference-time guidance methods can adjust sampling trajectories, but they typically optimize surrogate objectives such as classifier likelihoods rather than directly aligning with the target distribution. We propose *MMD Guidance*, a training-free mechanism that augments the reverse diffusion process with gradients of the *Maximum Mean Discrepancy (MMD)* between generated samples and a reference dataset. MMD provides reliable distributional estimates from limited data, exhibits low variance in practice, and is efficiently differentiable, which makes it particularly well-suited for the guidance task. Our framework naturally extends to prompt-aware adaptation in conditional generation models via product kernels. Also, it can be applied with computational efficiency in latent diffusion models (LDMs), since guidance is applied in the latent space of the LDM. Experiments on synthetic and real-world benchmarks demonstrate that MMD Guidance can achieve distributional alignment while preserving sample fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a method to guide a pretrained (possibly conditional) diffusion model using reference samples. Guidance is introduced by subtracting the gradient of a maximum mean discrepancy term (computed between model samples and reference samples) during the sampling process. The authors motivate the use of MMD, providing both theoretical justification and intuitive insight into the effect of this gradient correction. The approach is evaluated on synthetic Gaussian mixture experiments as well as on image generation tasks, in both unconditional and prompt-aware settings, showing clear improvements in sample alignment with the reference data.

### Strengths
* The paper is clearly written and easy to follow.
* The proposed idea is simple, elegant, well-motivated, and computationally efficient.
* The experiments convincingly demonstrate the method’s effectiveness in well-controlled synthetic settings using Gaussian mixtures.

### Weaknesses
* The paper provides no formal proof that the kernel in Eq. (5) converges to, or accurately approximates, the kernel corresponding to the denoising of the reference distribution.

### Questions
- I have questions on the MMD considered itself. Why not denoting the reference distribution $\hat{Q} = N^{-1} \sum_{j = 1}^N \delta_{\tilde{z}^{(j)}}$ where the dataset of the latent reference samples is $(\tilde{z}^{(j)})_{j=1}^N$ ? It seems more consistent with the notation on $P_t$. Moreover, why computing the MMD between noisy generated samples and clean reference samples (both in latent space) ? Wouldn't it be more correct to compute the distance between the noisy generated samples and the noisy reference samples ? Or maybe between the samples obtained by the denoiser at sampling step $t$ and the clean reference samples ?
- In section 4.2, you show that what is actually doing the job is the action $z \leftarrow z - \lambda_t \nabla \widehat{\operatorname{MMD}^2}$ but thats not what Eq. 5 implements. Why not actually **composing** the classic sampling steps with MMD gradient descent steps ? It would also raise the question of doing multiple MMD GD steps after a single sampling step.
- Have you investigated the impact of the number of reference samples ?
- Coming back to the synthetic MoG experiment, it would be interesting to see if, given a diffusion model trained on a MoG with equal weights on each mode, you can use reference samples with different weighting of each mode to change the proportions. It seems like a difficult task.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduced the MMD Guidance method to help diffusion models create images that match a user's style without needing retraining. This method adds a new component during the sampling step that changes the direction of image generation.
This approach uses a new gradient component (MMD^2). It shifts the generation process toward the user's data by comparing the current distribution of model samples with the user's distribution. This gradient also improves the diversity of the generated samples.
The paper looks at the results and usefulness of this method in two cases: unconditional generation and prompt-aware diffusion. It provides detailed results and comparisons using synthetic data (GMM), face datasets (FFHQ and CelebA-HQ), and models like SDXL and PixArt.

### Strengths
a) No retraining is required
b) It is easy to implement and understand
c) It has low computational costs
d) It is based on a strong theoretical foundation
e) The method is clearly described
f) It produces very good visual and quantitative results (FD, KD, Coverage)
g) The experiments are reliable, with results averaged from five random tests

### Weaknesses
a) It cannot learn new styles that the diffusion model doesn't know
b) Hyperparameter values ​​(guidance strength, kernel bandwidth) are selected empirically, which can hinder the correct generation of images
c) The experiments mainly focus on simple prompts, like "person with sunglasses," so its effectiveness for more complex prompts is not clear.

### Questions
1. How does the computational cost of MMD Guidance scale with the number of reference samples? 
2. How sensitive is the method to the choice of guidance strength and kernel bandwidth? 
3. Could adaptive or data-driven tuning of these hyperparameters improve robustness? 
4. Can MMD Guidance handle complex or compositional prompts beyond simple cases? 
5. Does the method provide convergence guarantees for distribution alignment during sampling? 
6. Have the authors explored alternative kernels that may better capture perceptual similarity? 
7. How does the proposed approach compare to other training-free guidance methods like:
- Classifier-free Guidance with Adaptive Scaling 
- Learn to Guide Your Diffusion Model
- CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models
8. Could this method be extended to other modalities such as audio, video, or 3D data?

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
5

### Summary
This paper proposes MMD Guidance, a training-free adaptation method that directly guides the reverse process of a diffusion model using the gradient of the Maximum Mean Discrepancy (MMD) between generated samples and a reference dataset. The authors argue that MMD provides reliable distributional estimates from limited data, exhibits low variance in practice, and is efficiently differentiable, making it well suited for guidance. The experiments show that MMD Guidance can achieve distributional alignment while preserving sample fidelity.

### Strengths
* MMD Guidance is intuitive: it leverages MMD to guide transfer tasks in a training-free manner.
* Theoretical derivations and toy examples are sensible and support the effectiveness of MMD Guidance.
* The experiments show that MMD Guidance outperforms the CG and fine-tuning baselines on the FFHQ benchmarks.

### Weaknesses
* An important reference is missing [1], which also presents a guidance-based framework for adapting a pre-trained diffusion model. Reference [1] can be regarded as the classifier-free variant of the DomainCG baseline in Table 1, and is also related to the fine-tuning baseline in Table 5. The authors should carefully discuss [1] and include it as a baseline.
* As a training-free approach, MMD Guidance may struggle to adapt to domains that are quite different from the pre-training domain. Could the authors provide adaptation tasks with a large domain gap so we can fairly assess adaptation capability in boundary scenarios?
* MMD Guidance relies on accurate computation of the gradient of the distance metric. Could the authors discuss in more detail how the size of the reference dataset influences performance, and analyze the computation–precision trade-off as the number of reference samples increases?
* Unreliable guidance signals on noisy data points are a well-known obstacle in CG. How do reference datasets at different noise levels influence the MMD calculation?
* Guidance approaches that rely on explicit gradients often suffer from "gradient hacking." How does MMD Guidance avoid the reverse process exploiting the MMD loss and producing biased samples?

[1] Domain Guidance: A Simple Transfer Approach for a Pre-trained Diffusion Model. ICLR 2025.

### Questions
See weakness 2,3,4,5.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies the mismatch between generated samples from diffusion models and target data. To address this issue, the authors augment the generative process with gradients of the maximum mean discrepancy (MMD) between generated samples and a reference dataset. The MMD-based method is practical since it doesn't suffer the curse of dimensionality. Furthermore, this method extends to prompt-aware distribution matching and latent diffusion models. Experiments are provided to verify the distribution matching performance.

### Strengths
- The authors use the maximum mean discrepancy (MMD) between generated samples and a reference dataset to address the distribution mismatching problem. This is a natural approach to guide generation toward our desired distribution.

- The MMD is a practical metric that measures the discrepancy between generation and reference distributions, since it avoids the curse of dimensionality from other metrics. 

- The authors provide an explicit evaluation of MMD gradient and illustrate the necessity of cross term for encouraging distribution matching. 

- The MMD-based method extends to prompt-aware distribution matching, which is a widely used application: text-image generation. Experiments are also provided to demonstrate the distribution matching performance.

### Weaknesses
- The maximum mean discrepancy is used to measure the distribution mismatching at every step. This might not be the best way. The reason is that  initially distribution mismatching should be large, and gradually reduces as generation ends. We only need to ensure the final  few steps have small distribution mismatching. 

- As shown in Theorems 1 and 2, it takes a large number of reference data to ensure that the empirical cross term is close to the ideal one. Since this is required for all time steps, it might require a large reference data set. 

- The MMD-based method is not analyzed, theoretically. It is not characterized how MMD-based guidance push the generative process to a target distribution. 

- Experiments show the distribution matching performance is attained by sacrificing image background quality.

### Questions
See comments in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
