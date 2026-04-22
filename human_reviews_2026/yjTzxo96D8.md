# Revisiting Spectral Representations in Generative Diffusion Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Diffusion models have shown remarkable performance on diverse generation tasks. Recent work finds that imposing representation alignment on the hidden states of diffusion networks can both facilitate training convergence and enhance sampling quality, yet the mechanism driving this synergy remains insufficiently understood. In this paper, we investigate the connection between self-supervised spectral representation learning and diffusion generative models through a shared perspective on perturbation kernels. On the diffusion side, samples (e.g., images, videos) are produced by reversing a stochastic noise-injection process specified by Gaussian kernels; on the spectral representation side, spectral embeddings emerge from contrasting positive and negative relations induced by random perturbation kernels. Motivated by this, we propose a self-supervised spectral representation alignment method to facilitate diffusion model training. In addition, we clarify how joint spectral learning can benefit diffusion training from a geometric perspective. Furthermore, we find that the optimization of the spectral alignment objective is in an equivalent form of diffusion score distillation in the representation space. Building on these findings, we integrate a spectral regularizer into diffusion training objectives to improve the performance of diffusion models on multiple datasets. Experiments across images and 3D point clouds show consistent gains in generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a spectral regularizer for diffusion models with a projection head trained to approximate the top-K eigenfunctions of the forward-diffusion kernel. The authors link the resulting objective to score-distillation in representation space and report gains on several visual and point cloud datasets.

### Strengths
The method proposed is teacher-free, and the authors connect it to the forward-diffusion kernel, which is good.
The implementation should be fairly efficient, with a time-conditioned head. 
The data ablations considered are reasonable for synthetic data, images, and point clouds. 
The results seem to yield mild but consistent gains, especially on point clouds or low res images.

### Weaknesses
The reported results are severely lacking in statistical significance. For all trend figures and tables the authors should (at the very least) report average and std/err over three or more seeds. Because of the mild improvements in some cases, and the lack of multi-run statistics, it is hard to judge significance.

The results are limited to mostly low-resolution experiments, and there seems to be a negative correlation in Table 1 between FID improvements and scale.

### Questions
1. It is hard to judge the significance of the reported results due to the lack of reported statistics over multiple runs. Can you report mean+std error for each of the metric figure/tables in the paper?

2. It would be worthwhile to understand the reason behind the diminishing FID improvements in table 1. Are these due to scale, or perhaps working with latents on ImageNet? Statistics over multiple runs could also help here, as well as additional latent diffusion experiments time allowing.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the intrinsic link between self-supervised Spectral Representation Learning (SRL) and Diffusion Models through the shared lens of time-dependent perturbation kernels. The authors propose a novel self-supervised spectral regularization loss to align the intermediate representations of the diffusion network, successfully avoiding the need for external pre-trained encoders. Empirical results consistently demonstrate that the proposed method significantly enhances the performance across both image and 3D point cloud synthesis tasks.

### Strengths
- The method offers a novel and versatile self-supervised regularization, requiring no external pre-trained encoder.
- The proposed method is applicable across multiple modalities and resolutions.
- Solid theoretical analysis connects the spectral loss to the mode-seeking dynamics of score distillation.

### Weaknesses
- Sensitivity analysis for the critical regularization hyperparameter $\lambda$ is insufficient.
- There lacks a discussion of the specific intermediate hidden layer for alignment.
- The improvements in FID scores, while consistent, are relatively modest.

### Questions
1. **Ablation on simplest contrastive loss.** Despite sufficient motivation explanations and theoretical analysis, practically, I believe the proposed SRL factually acts like a consistent loss across different perturbation levels. A similar line of this work is to use the simplest contrastive loss (w/ or w/o negative samples) to align the intermediate features across different noises. Could the authors provide an ablation study on this simplest contrastive loss to compare with the proposed SRL, to better justify the effectiveness of SRL?

2. **The improvement is limited.** Despite without using any external pre-trained encoder, the improvement on FID seems to be insufficient to demonstrate the effectiveness of the proposed method, especially that on CelebA. Could the authors provide more comprehensive experiments including:
   - combining SRL with REPA and reporting the comparision
   - similar to REPA, showing how many times can SRL boost training

3. **Training curve.** Could the authors explain why the performance of SRL is worse than the baseline at the beginning of training on ImageNet-64 in Figure 2?

4. **Sensitivity of $\lambda$.** The regularization weight $\lambda$ is critical to the final performance. Could the authors provide a sensitivity analysis on $\lambda$ to show how it affects the final performance?  

5. **Ablation on the hidden layer.** Could the authors provide an ablation study on which intermediate hidden layer to apply the proposed SRL? How does it affect the final performance?

### Soundness
3

### Presentation
2

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
This paper adds an additional self supervised alignment loss to the standard denoising score matching objective, so that similar samples in the representation space are expected to be clustered while different samples are pushed apart.

### Strengths
Adding extra self supervised regularization to the representation space for training diffusion models is interesting.

### Weaknesses
1. The motivation of the proposed alignment method is unclear. What is rationale behind minimizing the KL divergence term in (16)? Why do we want the score of negative samples to match the score of positive samples? Why is this beneficial?

2. The performance gain from the proposed algorithm is marginal. In particular, in table 1, very often, the metrics only marginally improved. From my experience, such marginal improvement won't affect the perceptual quality of the generated samples, and I cannot find any qualitative results for image generation in the paper.

### Questions
1. I don't understand the reasoning in line 305-310. Why minimizing the KL divergence leads to mode seeking behavior, clustering similar data while pushing apart dissimilar ones? Please explain. 

2. The performance gain is marginal, does your method improve image quality at all? 

3. Have you tried CFG? How does the model performance change when you apply CFG? Please plot the FID v.s. Inception curve for varying guidance strength for both the base model and your model.

### Soundness
2

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
3

### Summary
The paper proposes a method that applies spectral representation learning to diffusion models. To do this, additional MLP projection head is attached for learning spectral embedding and a spectral loss is added to train the embedding and this works as a regularizer to encourage representation alignment between perturbed samples derived from the same clean data. This can be seen as replacing the perturbation kernel used in traditional spectral representation learning with the diffusion noise kernel. Without any external model like REPA, the model could achieve better performance than baseline models on both ImageNet image generation and 3D point-cloud generation tasks.

### Strengths
- The paper proposes a novel connection between spectral representation learning and diffusion models, providing new insight by viewing them under a unified framework.
- The paper is well-written, with clear explanations of the preliminary background and methodology, which enhances readability.
- The method appears relatively easy to implement and improves internal representations without relying on any external models.
- The approach demonstrates noticeable performance improvements across very different domains, including ImageNet and 3D point cloud generation.

### Weaknesses
- The comparison experiments are somewhat limited. Although REPA is included, comparisons with other feature regularization methods such as Dispersive Loss are missing.
- While the method does not require external models, its generation quality appears lower than REPA, raising questions about its practical usefulness in real applications.
- In Figure 2, unlike REPA, the method does not seem to converge faster or show any clear trend in training behavior. To confirm that the improvement is not due to random fluctuations, multiple runs with statistical analysis would be necessary.

### Questions
- How much additional computational cost is introduced by adding the spectral branch?
- Are there any visualization comparisons for the image generation results?
- Is f in L133 a typo of h?

### Soundness
3

### Presentation
4

### Contribution
3
