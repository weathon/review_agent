# LapFlow: Laplacian Multi-scale Flow Matching for Generative Modeling

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
In this paper, we present Laplacian multiscale flow matching (LapFlow), a novel framework that enhances flow matching by leveraging multi-scale representations for image generative modeling. Our approach decomposes images into Laplacian pyramid residuals and processes different scales in parallel through a mixture-of-transformers (MoT) architecture with causal attention mechanisms. Unlike previous cascaded approaches that require explicit renoising between scales, our model generates multi-scale representations in parallel, eliminating the need for bridging processes. The proposed multi-scale architecture not only improves generation quality but also accelerates the sampling process and promotes scaling flow matching methods. Through extensive experimentation on CelebA-HQ and ImageNet, we demonstrate that our method achieves superior sample quality with fewer GFLOPs and faster inference compared to single-scale and multi-scale flow matching baselines. The proposed model scales effectively to high-resolution generation (up to 1024×1024) while maintaining lower computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LapFlow, a novel Laplacian multiscale flow matching. The proposed model follows a coarse-to-fine generation strategy across k scales in a Laplacian pyramid. It splits the time steps into k segments with k-1 critical points. In the early time steps, the model learns the lowest image scale. After passing each critical point, the Laplacian residual for the next level is added to be learned along with the previous scales. To learn the scales in parallel, the paper employs a mixture-of-transformers (MoT) architecture with causal attention mechanisms. LapFlow outperforms the baseline LFM and the recent multiscale approaches on CelebA-HQ and ImageNet datasets.

### Strengths
- The proposed model follows a coarse-to-fine generation strategy, which is theoretically more efficient and easier to extend to higher resolutions compared with the standard approach.
- To learn the scales in parallel, the paper employs a mixture-of-transformers (MoT) architecture with causal attention mechanisms, which is quite recent and novel.
- LapFlow outperforms the baseline LFM and the recent multiscale approaches in performance and efficiency.

### Weaknesses
- Beside inference speed and computation complexitity, memory consumption and training time comparison are recommended to add. 
- Some denotations defined in L210 were used in L140. The authors should define the denotations before using.
- L430: The term "temporal segment value" was used without definition. How to use that value to construct critical time-step points?
- It is interesting to see how advanced training techniques like REPA can be integrated in LapFlow to boost training convergence and output model quality.

### Questions
- Beside inference speed and computation complexitity, memory consumption and training time comparison are recommended to add. 
- L430: The term "temporal segment value" was used without definition. How to use that value to construct critical time-step points?
- It is interesting to see how advanced training techniques like REPA can be integrated in LapFlow to boost training convergence and output model quality.

### Soundness
3

### Presentation
3

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
The paper proposes multi-scale flow matching training which offers efficiency inference and enhance the model performance across resolution compared to Edify, LFM and Pyramidal Flow. To enable the multi-scale training, the author proposes to use laplacian representation, jointly training algorithm for multiple scale and inference algorithm.

### Strengths
1. Paper is well-written and easy to understand
2. Jointly training multiscale removes the complexity of rescaling noise technique when shifting to different scale like in Relay Diffusion and Pyramidal Flow.
3. The ablation and experiment is conducted extensively.

### Weaknesses
1. The fig.2 is not clear enough for reader. It would make reader misunderstand that MoT block has no FFN layers.
2. According to table 2g, two scale representation seems to be the best one, while more scale seems lag behind which not gaining much efficiency during the sampling.
3. The author should include more comparison with relay diffusion in the table and I would suggest to also compare your method with Pixelflow [1] work.
4. The results for imagenet is under-training which is hard for reviewer to judge the training convergence on this dataset.
5. How training framework can be adapted to different architecture like Unet (not transformer) ?

Minor: line 47 is not correct, pyramidal flow is not finetuning technique, it is training technique.

[1] PixelFlow: Pixel‑Space Generative Models with Flow (Chen et al., 2025)

### Questions
1. Why the face image in resolution 1024 seems much better than 512 and 256 but the FID is lower. The quality of 1024 is close to the real training image.

### Soundness
2

### Presentation
2

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
This paper introduces Laplacian Multi-scale Flow Matching (LapFlow), a novel framework designed to improve the efficiency and scalability of generative modeling. To tackle the high computational cost of generating high-resolution images, LapFlow moves away from single-scale generation. The core contribution is a method that decomposes images into multi-scale representations using a Laplacian pyramid. The model is then trained to generate these Laplacian residuals in parallel, which are finally combined using the pyramid reconstruction process to create the full-resolution image.The LapFlow architecture is based on a unified Mixture-of-Transformers (MoT) model, which processes all scales simultaneously. The model employs a progressive multi-stage training and sampling process, generating the coarsest scale first and subsequently adding finer residuals. Experiments conducted on CelebA-HQ and ImageNet demonstrate that LapFlow achieves superior sample quality (lower FID) and scalability, reaching resolutions up to $1024 \times 1024$. Furthermore, it surpasses existing flow matching baselines in computational efficiency, requiring fewer GFLOPs and achieving faster inference speeds.

### Strengths
+ LapFlow achieves superior FID scores across all tested resolutions, particularly at high resolutions ($1024\times1024$) where the performance gap is pronounced (e.g., $5.51$ FID vs. $8.12$ for LFM). The method simultaneously requires fewer function evaluations (NFE) and reduced computational resources (GFLOPs and time) compared to baselines, validating its efficiency hypothesis.

+ The use of a unified Mixture-of-Transformers (MoT) model with a causal attention mask to process multiple Laplacian residuals in parallel is an effective solution. This design eliminates the limitations of cascaded models (separate networks) while enforcing hierarchical dependencies.

+ The ablation studies confirm the necessity of key components like the MoT architecture for computational efficiency (GFLOPs reduced from $38.9$ to $16.5$) and the optimality of causal masking over other strategies (FID $3.53$ vs. $5.19$ for self-attention).

### Weaknesses
The progressive training strategy relies on empirically selected critical time points $T_{1}$ and $T_{2}$ to segment the training time range $[T_{k+1}, 1]$ for each scale $k$. While an ablation is provided for a single critical point $T$ in a two-scale setup (Table 2d), a justification for the chosen three-scale points $T_{2}=0.33$ and $T_{1}=0.67$ at higher resolutions is needed. The complexity analysis (GFLOPs and time) reported in the tables is based on the average inference cost. Providing a formal theoretical complexity analysis (similar to the total quadratic cost in tokens in standard DiT) to prove the superior scaling property of the MoT over standard DiT as a function of output resolution would strengthen the theoretical contribution. The two-scale representation achieves optimal performance at $256\times256$ (Table 2g), which seems counter-intuitive given the general goal of multi-scale modeling. A discussion on why increasing to three or four scales is detrimental (or offers no further gain) in this specific setting is warranted.

### Questions
- The core acceleration benefit stems from avoiding the cost of processing higher resolution scales for the entire duration of the flow matching process. Can the authors provide a formal theoretical derivation of the computational complexity (GFLOPs) of the multi-scale MoT attention mechanism and compare it to the complexity of a single-scale transformer (like DiT) as a function of the total number of tokens across scales? 

- For the ablation study on the number of scales (Table 2g), why does the performance degrade when moving from two scales to three and four scales at $256\times256$ resolution? Does this suggest a fundamental limitation of the Laplacian pyramid decomposition at this resolution, or is it purely a model optimization challenge? 

- Can the authors provide an intuition or ablation on how the fixed coefficients $\sigma_{t}^{(k)}=1-t$ and $\alpha_{t}^{(k)}=\frac{t-T_{k+1}}{1-T_{k+1}}$ for the linear path were chosen, particularly the decision to use $1-t$ for the noise decay, which is typically $\sigma_{t}$ (or $1-\alpha_{t}$) in flow matching, but here it is explicitly set to $1-t$ regardless of the non-linear $\alpha_{t}^{(k)}$ term?

### Soundness
2

### Presentation
2

### Contribution
3
