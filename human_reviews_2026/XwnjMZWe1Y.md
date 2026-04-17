# LUMA: Low-Dimension Unified Motion Alignment with Dual-Path Anchoring for Text-to-Motion Diffusion Model

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
While current diffusion-based models, typically built on U-Net architectures, have shown promising results on the text-to-motion generation task, they still suffer from semantic misalignment and kinematic artifacts. Through analysis, we identify severe gradient attenuation in the deep layers of the network as a key bottleneck, leading to insufficient learning of high-level features. To address this issue, we propose \textbf{LUMA} (\textit{\textbf{L}ow-dimension \textbf{U}nified \textbf{M}otion \textbf{A}lignment}), a text-to-motion diffusion model that incorporates dual-path anchoring to enhance semantic alignment. The first path incorporates a lightweight MoCLIP model trained via contrastive learning without relying on external data, offering semantic supervision in the temporal domain. The second path introduces complementary alignment signals in the frequency domain, extracted from low-frequency DCT components known for their rich semantic content. These two anchors are adaptively fused through a temporal modulation mechanism, allowing the model to progressively transition from coarse alignment to fine-grained semantic refinement throughout the denoising process. Experimental results on HumanML3D and KIT-ML demonstrate that LUMA achieves state-of-the-art performance, with FID scores of 0.035 and 0.123, respectively. Furthermore, LUMA accelerates convergence by 1.4$\times$ compared to the baseline, making it an efficient and scalable solution for high-fidelity text-to-motion generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes LUMA, a unified motion alignment with dual-path anchoring for text-to-motion diffusion model. Authors introduce the gradient vanishing problem and design the frequency and temporal modules to enhance the semantic alignment, which improves the generation performance in HumanML3D and KIT-ML datasets.

### Strengths
1.Authors point out the gradient vanishing problem and try to solve that. Through some visualized results, they prove that this phenomenon exists.
2.This work enhances the performance of motion generation and improve the convergence speed, which reduces the time cost during training.

### Weaknesses
1. I believe the vanishing gradient problem mentioned by the authors does exist, but the method proposed in the paper does not seem to be specifically designed to address this issue. In my understanding, introducing additional loss functions and certain heuristic tricks during training can also alleviate the vanishing gradient problem. The authors should further elaborate on the necessity of temporal and frequency alignment as well as other advantages it brings.  

2. For tasks like motion generation, the absence of video results is quite unusual. The paper only presents three static visualization figures; more cases and video demos are needed to demonstrate the effectiveness of the method.  

3. The baseline lacks comparisons with more methods, such as LaMP and ReMoDiffuse.  

4. The authors state that this work does not use a pre-trained foundation model as an alignment supervisor. However, they trained a MoCLIP from scratch and aligned it with the CLIP text encoder. I believe this is inconsistent with the authors' statement.  

5. The font size in some tables of the paper is excessively large, which affects readability and aesthetics (e.g., Table 8). Additionally, the font formatting in the pseudocode of the algorithm appears somewhat abrupt, and the framework diagrams are relatively unclear. There are also inconsistencies in certain expressions—for instance, "downsampling" is used in Line 75, while "down-sampling" is used in Line 103.

### Questions
The main issues are listed in the "Weakness" section. Here, I have an additional question: I hope the authors can specify what the frequency information of motions specifically represents, and clearly state the exact role of such alignment.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper tries to improve the UNet-based diffusion models for the text-to-motion generation task. It first identifies the gradient attenuation issue in the deep layers of U-Net, and proposes Low-dimension Unified Motion Alignment (LUMA) for better representation learning via extra supervision signals to the hidden feature. LUMA provides hidden feature supervision signals with features from a motion encoder and DCT. By adopting a better text encoder, "deep supervision" for specific hidden features, the proposed model shows improved performance compared to the baseline on which the model is built. Ablation experiments also quantitatively show the effectiveness of the proposed designs.

### Strengths
1. The overall presentation is nice and clear, which is easy to follow.
2. The proposed method shows better performance compared to the baselines, according to the quantitative results.
3. The ablation experiments effectively show that removing any proposed component will lead to performance degradation.
4. The paper reveals a relatively new issue on text2motion generation, with respect to the architecture design.
5. The code for reproduction and the detailed appendix are greatly appreciated.

### Weaknesses
1. The scope of this work/experiment is relatively narrow, which focuses on text2motion with diffusion UNet, specifically. 
2. The discussion in the introduction section is not reasonable. a) It mentions that "increasingly sophisticated architectures" lead to performance improvement, yet "diminishing efficiency and limited practical improvement." This is not logical, and the proposed method has nothing to do with efficiency and still relies on deeper and larger networks. b) Diffusion UNet "are limited by their high computational cost during training". The proposed method also requires the same or extra computational cost because of the use of additional latent encoders for "deep supervision".
3. "Deep supervision" or representation learning for diffusion models is a relatively well-explored topic. 
4. DCT for "low-frequency components, recognized for their rich semantic content," I'm a bit skeptical that the DCT feature will help semantic representation learning. According to Table 2, removing L_fre does not affect the R@3, which is related to semantic alignments.
5. Adaptive FiLM modulation is a good design choice, but is a common design for the diffusion model, which is also used in the Adaptive Normalization. 
6. Overall, the method is relatively incremental, and the performance gain is not signicant. For example, it worse than the SOTA reported from a work from ICLR last year [1].
7. There's no video result to assess the performance qualitatively, which is crucial for animation tasks.
8. According to Table 3 and 4. It seems that removing FiLm or the deep supversion signal can achieve better performance than inject the supervsion into improper position, e.g. Bottleneck. This emprical result make the overall design becomes a bit tricky.

[1] Li, Zhe, et al. "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning." The Thirteenth International Conference on Learning Representations.

### Questions
1. If the gradient vanishing is a severe issue, will the proposed method also generalize to improve the performance of e.g. text-to-image Diffusion UNet?

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
To address semantic misalignment and kinematic artifacts in text-to-motion generation, The authors propose LUMA, a diffusion-based text-to-motion model that mitigates semantic misalignment and kinematic artifacts through dual-path anchoring: a lightweight MoCLIP for temporal semantic guidance and low-frequency DCT features for complementary frequency-domain alignment.

### Strengths
- The authors conducted extensive experiments to support their method, although the improvements achieved are quite limited.
- The paper is clearly written and the language is generally fluent and coherent.

### Weaknesses
- The motivation is vague. The authors claim that experimental analysis reveals gradient shifts in the deeper layers as the root cause of the problem, but it is unclear what specific issue this refers to.Is it low semantic fidelity, high computational cost, or something else?

- Figure 2 is not clearly presented, making it difficult for readers to understand the overall structure and flow of the proposed method.

- The Method and Preliminaries sections are poorly written. They do not clearly explain how the proposed approach addresses the problem.

- The concept of Low-Dimensional Unified Motion Alignment is insufficiently explained. In addition, the Method section lacks clarity and coherence, which makes the proposed approach hard to comprehend.

- The relationship between gradients and high-level feature extraction is not clearly explained. The paper does not clarify why gradient behavior would affect the learning of high-level features.

- In Table 1, there are several errors in the bold and underline markings. For example, in the fifth column, the best result should be 9.724 rather than 9.466. Moreover, compared with StableMoFusion, the performance improvement appears to be quite limited.

- Table 2 shows that, except for a slightly improved FID score, the proposed modules contribute only marginal gains. In particular, rows 3–5 perform worse than the full LUMA configuration in the last row.

- The paper lacks visual results to demonstrate or analyze the semantic alignment between text and high-level features. 

- The authors claim that previous methods suffer from high computational cost; however, the paper does not provide any analysis or comparison to demonstrate that the proposed method is computationally efficient.

### Questions
- Since deeper layers are expected to capture high-level abstract representations, the loss of fine-grained details should be attributed primarily to the downsampling operations rather than the upsampling stages. This is somewhat inconsistent with the authors’ stated motivation.

- While the authors assert that gradients in the downsampling and bottleneck layers are considerably weaker or even vanish compared to those in the upsampling layers, the paper lacks concrete experimental evidence to substantiate this observation.

### Soundness
1

### Presentation
1

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
This paper analyze the gradient descent in downsampling and bottleneck layers in UNet. And it proposes a dual-path semantic information injection mechanism: one path uses mo-clip to extract semantic information of motion frames, another path utilizes the low-frequency part of spectrum after DCT.

### Strengths
1. It analyzes the gradient descent phenomenon in UNet. It points out that the small gradient norm in the deep layers of UET affects the learning of high-level semantics, and verifies that the proposed method can improve this phenomenon.
2. Compared to related works such as Repa, this work does not require a pre-trained visual model. Thus it solves the problem in the text-to-motion domain where the lack of large-scale pre-trained understanding models prevents the use of Repa.
3. This work achieves state-of-the-art performance while accelerating convergence.

### Weaknesses
1. Lack of comparison with alternative gradient enhancement techniques: The paper focuses on dual anchors but does not compare with other methods for mitigating gradient attenuation in diffusion models (e.g., residual connections tailored for U-Net deep layers, adaptive learning rate scheduling for bottleneck layers). 
2. The dual-path method will introduce additional cost while training and inference. The paper does not have a detailed quantitative discussion.
3. Recently, diffusion models are always using DiT as backbone. How does this method perform on DiT?

### Questions
1. How does MoCLIP’s text-motion alignment performance compare to other SoTA encoders?
2. For motions with distinct frequency characteristics (e.g., fast, jerky actions vs. slow, smooth actions), does the optimal k (number of DCT coefficients) change? If so, how might the framework be adapted to dynamically adjust k based on input text or motion type?
3. What is the training cost of MoClip and how does it compare to other similar models?

### Soundness
3

### Presentation
3

### Contribution
3
