# What Secrets Do Your Manifolds Hold? Understanding the Local Geometry of Generative Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Deep Generative Models are frequently used to learn continuous representations of complex data distributions by training on a finite number of samples. For any generative model, including pre-trained foundation models with Diffusion or Transformer architectures, generation performance can significantly vary across the learned data manifold. In this paper, we study the local geometry of the learned manifold and its relationship to generation outcomes for a wide range of generative models, including DDPM, Diffusion Transformer (DiT), and Stable Diffusion 1.4. Building on the theory of continuous piecewise-linear (CPWL) generators, we characterize the local geometry in terms of three geometric descriptors - scaling ($\psi$), rank ($\nu$), and complexity/un-smoothness ($\delta$). We provide quantitative and qualitative evidence showing that for a given latent vector, the local descriptors are indicative of post-generation aesthetics, generation diversity, and memorization by the generative model. Finally, we demonstrate that by training a reward model on the 'local scaling' for Stable Diffusion, we can self-improve both generation aesthetics and diversity using geometry sensitive guidance during denoising. Website: https://imtiazhumayun.github.io/generative_geometry.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Building on the continuous piecewise-linear hypothesis, this paper introduces three local geometric descriptors for generative models at specific stages of generation: scaling, rank, and smoothness. Using these descriptors, the paper empirically examines the local geometry of both a toy dataset and Stable Diffusion, drawing several conclusions about the learned image manifold. Finally, leveraging these descriptors, the paper proposes a novel reward model that guides the generation in Stable Diffusion, allowing control over specific local geometric properties.

### Strengths
This paper represents a valuable empirical contribution to understanding the local geometric properties of the image manifold learned by diffusion models. Based on the hypothesis that the output space of neural networks exhibits a continuous piecewise-linear structure, the learned data manifold can be approximated as a union of affine transformations across local regions. This foundation makes the proposed local geometric descriptors both intuitive and well-motivated. In empirical experiments, these descriptors align closely with human visual evaluation, underscoring their effectiveness. I believe this paper will significantly impact the study of image manifold geometry.

### Weaknesses
1. This paper lacks discussion with related works [1, 2, 3, 4]. The local dimensionality of the diffusion model-based image manifold has also been studied in those papers. However, as concurrent work, this won't significantly weaken the contribution of this paper. Better discussion about the relation and differences with those works could highlight the contribution of this paper.
2. The proposed method is not very strong. The authors provide only qualitative results without any quantitative evaluation or comparison to related works. 
3. Some parts of the writing are unclear. I suggest including the numerical method used to approximate the calculation of local complexity (Definition 3) in this paper. Although this method was introduced in previous work, including it here would make the paper more complete and easier to understand.






[1] Stanczuk, Jan Pawel, Georgios Batzolis, Teo Deveney, and Carola-Bibiane Schönlieb. "Diffusion Models Encode the Intrinsic Dimension of Data Manifolds." In Forty-first International Conference on Machine Learning.

[2] Kamkari, Hamidreza, Brendan Leigh Ross, Rasa Hosseinzadeh, Jesse C. Cresswell, and Gabriel Loaiza-Ganem. "A Geometric View of Data Complexity: Efficient Local Intrinsic Dimension Estimation with Diffusion Models." arXiv preprint arXiv:2406.03537 (2024).

[3] Wang, Peng, Huijie Zhang, Zekai Zhang, Siyi Chen, Yi Ma, and Qing Qu. "Diffusion models learn low-dimensional distributions via subspace clustering." arXiv preprint arXiv:2409.02426 (2024).

[4] Chen, Siyi, Huijie Zhang, Minzhe Guo, Yifu Lu, Peng Wang, and Qing Qu. "Exploring low-dimensional subspaces in diffusion models for controllable image editing." arXiv preprint arXiv:2409.02374 (2024).

### Questions
1. In line 115, the paper discusses unrolling the sampling process of the diffusion model. Does this mean that when calculating the descriptor at a large timestep $t$, you need to compute the Jacobian iteratively over the neural network multiple times because the sampling steps are large? From my experiments, computing the Jacobian over the UNet in Stable Diffusion takes hours, even without unrolling. What is the computational cost of your implementation?

2. Are these three proposed descriptors redundant? From Figures 5, 12, and 13, it appears that as local scaling, local rank, and local complexity increase, the visual complexity of the images all increases. Seems we only need one descriptor for the image visual complexity. What specific, distinct properties do each of these three descriptors capture?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the local geometry of the generative model and its relationship to the quality of the generation. Built upon the theory of continuous piecewise-linear generators, the authors propose to use three metrics: (1) local rank; (2) local scaling; and (3) local complexity to quantify the geometry of the generative model. Leveraging these descriptors, they try to understand how the local geometry relates to the generation process. Specifically, they show that local scaling is related to the complexity of the generated scenes, and can be used as a tool to detect whether the data is on the input (training) manifold. In addition, they propose to train a reward model to approximate the local scaling descriptor and present results showing more objects and complex backgrounds while increasing the value of the local scaling.

### Strengths
The idea is novel and the authors conduct extensive experiments in multiple scenarios. It is very interesting to understand the geometry of the model and its role in the generation process.

### Weaknesses
1. My biggest concern regarding this paper is the approachability to the audience. The three local geometric metrics discussed in the paper are very interesting. However, in terms of the method presentation, particularly the reward model built upon the local scaling descriptor, the current presentation lacks detailed descriptions, which makes me concerned about the reproducibility of the results.

2. The claim that "local geometry trajectories are discriminative of memorization" is relatively weak for the local complexity metric when the trend of the metrics is almost un-distinguishable when increasing the guidance scale.


Minors:
Line 323: I think it should be capitalized DDPM instead of "ddpm".

### Questions
1. Eqn.2: Should the singular values here be normalized?

2. Line 164: What's the definition of $S$, why the local scaling is proportional to the NLL of the generative model when the relationship between $\psi$ and $S$ is injective?

3. Lines 262-269: Where are the corresponding results and visualizations? Why do results at 0.17T show the minimized discrepancy between the on-manifold and off-manifold input space? And why does the minimized discrepancy indicate ``local geometry indicators have the highest distinction geometrically between on and off manifold vectors from the input space'' (should it be the opposite)?

4. Figure 3: What does $[-6, 6]^2$ mean? What's the x-axis of the plot on the right, is it the noise scale?

5. Figure 3 and Figure 4: What are the axes of the local generator? Does each pixel correspond to one input image? How is this being calculated?

6. Figure 4: What images correspond to the pixels in the center of the geometric desciptors?

7. Section 5: How is the reward function defined? Why the calculation of the Hessian matrix is required in the first place? The paper describes three local geometric descriptors, but why does the reward model only approximate the local scaling value?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to study the local geometry of pretrained diffusion models and explore how different local geometry indicates different properties. It proposes three descriptors scaling (ψ), rank (ν), and smoothness (δ) to characterize the local geometry in diffusion models based on the continuous piecewise-linear (CPWL) assumption. The paper uses these descriptors to study the local geometry in classical DDPM and stable diffusion models. The paper conducts various experiments to show the following statements: 1. Local geometry of off manifold is different from on manifold. 2. Increasing local scaling leads to the generation of more complex image content. 3. Local geometry of Stable Diffusion is sensitive to image corruptions. 4. Local geometry of Stable Diffusion is sensitive to memorization of text prompts. Lastly, the paper proposes to guide generation with gradients of the descriptors (specifically local scaling). To avoid heavy computation of Hessian, they actually train a reward model to approximate local scaling. With this geometric guidance, the paper can control content complexity in the generated images.

### Strengths
1. The paper is studying the local geometry from a comprehensive perspective, modeling scaling (ψ), rank (ν), and smoothness (δ).

2. The paper shows experiment results from several perspectives including off and on manifold, content complexity, image corruption, and text prompt memorization.

3. The paper utilizes local scaling to control the generated image complexity.

### Weaknesses
1. Lack of literature review. There exists prior work [1] related to out-of-domain detection with local geometry, and [2] related to uncertainty quantification with local geometry. Even though the metrics are not exactly the same, it is worth discussing these previous works, as well as other ones related to the manifold of diffusion models.

2. Lack of clarity and technical details. See the questions part.

3. Lack of Justification of the continuous piecewise-linear (CPWL) assumption: The paper does not provide justification that CPWL can be a good approximation for DDPM and stable diffusion. In contrast, recent work [3] has experimentally verified such local linearity.

4. Redundancy of metrics: Though all three metrics are interesting, it seems rank (ν), and smoothness (δ) are redundant since most observations and applications can be correlated only with scaling.

[1] A Geometric Explanation of the Likelihood OOD Detection Paradox. Hamidreza Kamkari et al.

[2] On the Posterior Distribution in Denoising: Application to Uncertainty Quantification. Hila Manor et al.

[3] Exploring Low-Dimensional Subspaces in Diffusion Models for Controllable Image Editing. Siyi Chen et al.

### Questions
(a)  Where does equation (3) come from?

(b) In equation (7), how is B defined? (i. How to choose number of dimensions P, is it the same aound any latent $z$? ii. How to choose columns in B? iii. Why a constant projection B can be used for all neighbor around the specific $z$?)

(c) What explicitly are the input space vectors for DDPM and Stable Diffusions? Are they the inputs to the denoising UNet, or could they be features in the middle layers of the UNet?

(d) In section 3.1 L250, "input vectors within 0.05 units of the training data", how is 0.05 chosen and why it is a reasonable value?

(e) In Figure 4, how are the three anchors (representing latent from 3 text prompts) labeled in the latent space and what are other points?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper, "What Secrets Do Your Manifolds Hold? Understanding the Local Geometry of Generative Models", explores the role of local geometry in deep generative models and its effect on generation quality, diversity, and memorization. The authors propose three geometric descriptors—local scaling ($\psi$), rank ($v$), and complexity ($\delta$)—to characterize the latent space of generative models, focusing on models with Continuous Piecewise-Linear (CPWL) mappings. They empirically demonstrate that these descriptors are correlated with various aspects of generative performance and propose a reward model trained on these descriptors to guide sample generation in diffusion models, particularly Stable Diffusion.

### Strengths
**1. Interesting Indicators of Geometry**: This paper introduces an interesting use of geometric descriptors to understand and control generative models, providing new insights into latent space structure.

**2. Comprehensive Experiments**: The paper includes extensive experiments across various models, from toy models to Stable Diffusion.

**3. Practical Framework for Control**: The reward model offers a practical way to influence generation characteristics, such as diversity and aesthetic quality, by guiding sampling in the latent space based on geometry based on the trained scalar model of the geometry descriptor.

### Weaknesses
**1. Computational Demands**: The method relies on calculating Jacobians on each linear piece of model manifold. This may be computationally intensive for large models, limiting its practical application. Such a limitation potentially indicates the approach is less scalable. Besides, since the computation of each geometric descriptor is computationally demanding, generating data for training latent space reward models is also computationally demanding and therefore possibly not scalable. 

**2. Descriptor Interpretation**: Some descriptors, particularly local complexity ($\delta$), lack intuitive interpretation when applied to high-dimensional latent spaces, which could be further clarified. This makes the paper more like a pure empirical computation of each descriptor instead of an in-depth study of the latent geometry of generative models.  

**3. Weak Evaluations**: A major drawback of this paper is the weak lack of quantitative evaluations. For instance, the paper should provide quantitative metrics such as FIDs or Human preference scores to evaluate the performances of the proposed geometry descriptor guided sampling instead of merely qualitative plots.

**4. Poor Compatibility for Networks with Smooth Activation Functions**: If I do not misunderstand, I think the approach is only properly defined for models with piece-wise linear neural networks instead of those networks using smooth activation functions such as SiLU, GELU or SwiGLU. This might limit the broader usage of the proposed approach and the impacts of the study.

### Questions
**1.** I am curious about the computational costs such as GPU hours per 1k samples to calculate each descriptor value, as well as the data preparation of reward models. I think this would give readers a more comprehensive understanding of the empirical impact of the proposed approach.

**2.** I think it would be good if authors could provide more studies on DiT or MM-DiT-based text-to-image diffusion models. The UNet-based Stable Diffusion 1.4 model is kind of weak when compared with current strong DiT-based models such as Stable Diffusion 3 or Flux. This study will give readers an in-depth understanding of how compatible the approach is across different neural network architectures. 

**3.** It would be good if authors could provide more intuitive discussions on each of the three descriptors. This will help readers understand the intuitions behind them. 

**4.** The author should provide quantitative metrics such as FID or human preference scores when using reward guidance sampling.

### Soundness
3

### Presentation
2

### Contribution
3
