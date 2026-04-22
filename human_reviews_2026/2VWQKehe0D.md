# Z-Cache: Accelerating Diffusion Transformers via  Self-Reflection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Diffusion transformers have become the most powerful models for visual generation, but still suffer from massive computation costs. To solve this problem, feature caching has been proposed to cache the features of diffusion models in the previous computation steps and then reuse them in the following caching steps, which brings significant acceleration but also degradation in generation quality. To address this problem, this paper proposes Z-cache as a feature caching method that can maintain high-quality generation through self-reflection. Concretely, we observe that the error from feature caching tends to be sharply reduced after each full computation. Based on this observation, Z-Cache is designed to first predict the features in the future caching steps and then perform a full computation. After that, Z-Cache returns to the caching steps and re-predicts them based on the previous and the current computation steps, which brings correction in features.  Experiments demonstrate that with Z-Cache, diffusion transformers achieve comparable generation quality to the original model  but with faster inference speed, for instance, 5.53$\times$ acceleration on FLUX-dev for text-to-image generation. Our codes have been released in the supplementary materials and will be released on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes z-cache, a solution to reduce the computational cost during the denoising process of diffusion models by caching features, thereby improving inference speed. The difference from previous cache-based acceleration methods is that this paper finds the error accumulation caused by feature caching can be alleviated after a full computation, thus considering that diffusion models can self-correct. The paper suggests using the feature after a future full computation to correct the previous cached feature, mitigating the accumulated errors. This paper conducts extensive comparisons and experiments with various baseline methods on tasks such as c2i, t2i, and t2v.

### Strengths
1. The motivation of the paper is clear, and fig.2 demonstrates the effect of error accumulation and the corrective impact brought by full computation.
2. The experimental results are abundant, with extensive comparisons made across various tasks such as t2i, c2i, t2v, and different baseline solutions.
3. The paper is well-written and easy to read.

### Weaknesses
1. There are no quantitative metrics for image quality and text consistency in t2i tasks, only some low-level metrics like PSNR reflect similarity to the original image. If fair comparisons with other methods are difficult to reproduce, at least a comparison with the direct baseline method TaylorSeer is necessary.
2. The error introduced by feature caching may have little impact on image generation tasks without reference images, but for some image editing tasks that require consistency in non-edited regions and personalized generation that maintains IP consistency, could the error from feature caching potentially harm critical visual information needed to discern the identity of the original subject or background?
3. I have some doubts about the motivation of the article. It is reasonable that the paper observes that full computation can correct the cumulative errors caused by feature caching at the current time step, but why is the feature at future time steps, when weighted with cached feature from previous time steps, closer to the ground truth feature of the midstep time steps?
4. The novelty of the paper is limited, and the technical contribution is incremental. Using the feature computed fully at future time steps to correct the cached representation at previous time steps seems merely a reverse thinking of TaylorSeer, which uses representations from past time steps to correct future feature, this paper offering limited technical contributions.

### Questions
Why is the weighting function $ w( \cdot ) $ designed in the form of a monotonically decreasing logarithmic function, and what is the motivation behind it?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents Z-Cache, a training-free acceleration framework for diffusion transformers that revisits and refines cached features within the sampling trajectory. By introducing a self-reflection mechanism, Z-Cache slashes cumulative error. Extensive evaluations on ImageNet-1K, PartiPrompts and VBench show that DiT, FLUX and HunyuanVideo all enjoy up to 5.53x-6.22× speed-up while retaining top-tier FID, IS, LPIPS and VBench score

### Strengths
1.	It is interesting to observe that the feature error drops sharply exactly at the point where full computation steps are completed.
2.	The manuscript is well-structured, clearly written, and easy to follow.

### Weaknesses
1.	What operation does the circle symbol in lines 172 and 173 represent?
2.	It would be great if the proposed method could further improve the sampling speed of the distilled models.
3.	It's better to provide a user study to verify, through human evaluation, whether the generative performance of the method is close to the baseline.
4.	The layout of the paper is somewhat inconsistent. Most table captions are placed above the tables, while the caption of Table 4 is below the table.

### Questions
The use of future features to update the cache is an interesting idea, and the method demonstrates its ability to accelerate multi-step sampling. It would be valuable to further examine whether the approach is effective for few-step distilled mode

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
4

### Summary
To solve the issue of large computation costs during generation with diffusion transformers, this paper proposes a new approach named “Z-cache”. The Z-cache is designed to predict the features in the future caching steps and then perform a full computation. Experiments show the faster inference speed for t2i generation.

### Strengths
1.	The proposed method belongs to training-free paradigm, which makes it easy to use.
2.	The results of FID, SSIM, LPIPS demonstrate the effectiveness of the proposed method.
3.	The writing is easy to understand, and the painting is well-drawn.

### Weaknesses
1.	In the paper, authors indicate that “the diffusion model can correct itself from a wrong generation trajectory to some extent” and this is the key idea to make full use of self-correction ability. However, there is no further theoretical analysis or explanation to answer why the self-reflection mechanism reduces error. The mechanism behind the observation is important for deep research of DiTs and the robustness of the model.
2.	The Z-cache only repairs the hidden states in the network, instead of the semantic consistency, so, if model generates some wrong semantic contents in i-th step, it is still maintained in future steps.
3.	Whether the proposed Z-cache is still useful under the “residual connection” structures?
4.	Limited experiments:
a)	The paper primarily compares Z-cache with other caching-based methods, however, it lacks a comprehensive comparison with non-caching acceleration techniques, including quantization, pruning, or knowledge distillation.
b)	The experiments are mainly tested on the DiT-based methods, I wonder whether the method is also effective on U-Net-based frameworks?
c)	Hyperparameters: although the paper claims the logarithmic averaging weight eliminates hyperparameter tuning, the choice of base caching strategy (e.g., TaylorSeer vs. TeaCache) still impacts performance.
d)	In Fig. 5, some cases do not support the superiority of the proposed method compared with other models, such as TaylorSeer. This includes the cases in the first, second, third, fourth, and fifth rows.
5.	The failure cases and more analyses should be discussed.

### Questions
See above.

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
4

### Summary
Z-Cache proposes a “cache-then-check” framework to accelerate diffusion Transformers while preserving generation quality. Starting from any existing feature-caching baseline, the method first caches or extrapolates hidden states for the next N timesteps, performs one full forward pass at timestep t+N+1, then rolls back to revisit the N cached steps. The newly computed “future” features are blended with the original predictions through a distance-dependent logarithmic weight, yielding corrected hidden states before sampling continues. Experiments on DiT-XL/2, FLUX-dev and HunyuanVideo show 4.5–6.2× speed-ups.

### Strengths
1.	Achieves 4–6× inference acceleration across image and video generation tasks. 

2.	Method is lightweight and model-agnostic: it requires no retraining of the backbone.

### Weaknesses
1.	The novelty is Limited novelty. The contribution is an incremental refinement over existing cache-and-forecast strategies rather than a fundamentally new acceleration paradigm. 

2.	Noticeable quality drops in challenging regions, e.g., facial consistency in Figure 1.

3.	Figures in the paper are not sufficiently high-resolution, making it hard to inspect the visual differences.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
