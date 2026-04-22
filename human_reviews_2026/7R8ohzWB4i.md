# ReDDiT: Rehashing Noise for Discrete Visual Generation

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
In the visual generative area, discrete diffusion models are gaining traction for their efficiency and compatibility. However, pioneered attempts still fall behind their continuous counterparts, which we attribute to noise (absorbing state) design
and sampling heuristics. In this study, we propose a rehashing noise approach for discrete diffusion transformer (termed **ReDDiT**), with the aim to extend absorbing states and improve expressive capacity of discrete diffusion models. ReDDiT enriches the potential paths that latent variables traverse during training with randomized multi-index corruption. The derived rehash sampler, which reverses the randomized absorbing paths, guarantees high diversity and low discrepancy of the generation process. These reformulations lead to more consistent and competitive generation quality, mitigating the need for heavily tuned randomness. Experiments show that ReDDiT significantly outperforms the baseline model (reducing gFID from 6.18 to **1.61**) and is on par with the continuous counterparts. The code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles a central problem of discrete diffusion models, especially for the masked visual token models (MVTMs), which have relied on a single absorbing state to perturb the input discrete visual token and Gumbel softmax with relatively high sampling temperature. Existing sampling strategies introduce instability and degradation as vocabulary size grows. By introducing masking capacity, ReDDiT effectively improves the capabilities of MVTMs. The experimental results are promising.

### Strengths
- **Well-Motivated and Simple-yet-General Solution**: While discrete diffusion models (DDMs) mimic the continuous one, a single absorbing state is a fundamental restriction of the MVTMs, which limits the diversity of samples and enforces MVTMs to utilize high sampling temperature, which compromises sample quality.  By simply extending the absorbing state design, ReDDiT achieves promising results, paving the way for improving DDMs' generative capabilities.

- **Clear Presentation**: The visualization in Figs 1 and 3 is good. The paper is well-structured, facilitating a clear understanding of the background and contributions of the paper. Simple notation with a theoretically grounded explanation ensures the readability and accessibility of the paper.

- **Generalizability and Significance**: The proposed rehashing masking can be generally adopted by the masked generative modeling paradigm, bringing a significant advancement to the field.

### Weaknesses
- **Controlling the Stochasticity**: Controlling the stochasticity of the sampling process is critical, as some tasks, such as drug design, require exact outputs at the expense of diversity, while in some cases, like artistic painting, require high variance. However, the proposed method does not utilize sampling temperature, lacks such control. This may significantly limit the applicability of the method in various fields. I think the proposed method could also adopt a sampling temperature. In some research, sampling with high temperature equipped with an appropriate guidance scale could show superior sample quality and diversity simultaneously [1]. I think it is more beneficial if the stochastic sampling can also be adopted for the ReDDiT sampling pipeline. 

- **Justification for Exact Sampling in Large Vocabulary**: The authors argue that ReDDiT captures the distribution better with a large vocabulary size. However, I can't find any theoretical justification for this statement. I agree that the noise rehashing can express better diversity without heuristic temperature scaling. However, I wonder why it can capture the exact distribution under large vocabulary settings.

- **Scalability to Sampling Steps**: In Table 1, the optimal sampling steps increase as the models' capacity grows. MaskGIT has shown the "sweet spot" of sampling steps, where large sampling can degrade the sample quality, conversely. The scalability analysis of the ReDDiT sampler with respect to the number of sampling steps is insufficient. The plot in Figure 3, showing gFID versus sampling steps, is hard to interpret, does not indicate the sampling steps, and fails to demonstrate a clear trend.

[1]  Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance, NeurIPS 2024

### Questions
- Low sampling temperature is known to incur a multi-modality problem in non-autoregressive (NAR) sampling since NAR samples multiple tokens at a time, and their relationship is ignored. Since ReDDiT does not utilize sampling temperature and uses multinomial sampling, I wonder if ReDDiT also suffers from such a limitation, especially for the small or large sampling steps.

- The rehash operation (Algorithm 2, line 6) is mentioned briefly. This step randomly shuffles tokens that are already noise tokens at the start of each sampling step. What is the theoretical and empirical justification for this? What happens to performance (gFID, diversity) if this line is removed? (i.e., what happens if the sampling becomes deterministic to the initial noise?)

### Soundness
4

### Presentation
4

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
The paper proposed a new rehashing noise approach for discrete diffusion transformer, which guarantees high diversity  and low discrepancy of the generation, outperforming baseline model by a large margin. Supported by solid theory and extensive experiments, the effectiveness and advantages of the method have been validated. The overall quality is satisfactory, while more explanation and interpretation in methodology are needed for clear presentation.

### Strengths
1. The paper proposed a new sampling methods that facilitate efficient and diverse generation for discrete diffusion, and it has a good structure with clear motivation, distinctive contributions, and solid theory.
2. The proposed method is well-supported by authors’ experiment results, either in tables other than figures.

### Weaknesses
1. The high diversity of the generation can be seen from qualitative examples, while what does low discrepancy mean in the proposed sampler? 
2. The methodology part is somewhat confusing. For equation 6, the definition of \mathbf{m}_j is not clear. Does \mathbf{m}_j indicate a absorbing token at the j-th position of the vector? Plus, I=0 and j=0 does mot make sense for index starting from 1. For equation 8, after rewriting equation 1, why does the transition kernel become \frac{1}{m}? If it is due to m states of  \mathcal{M}, then this problem comes back again to the definition of \mathbf{m}\in\\mathbb{R}^{m}: does \mathbf{m}_{j} indicate m token at position j and elsewhere all 0? Is there only one absorbing state, which is m? Moreover, for equation 9, the symbol usage is not consistent, sometime it is $x_{t}^{I}$,  while sometimes it is $x_{t}$.
3. More interpretation needed for figures. For fig 2, it is not clear how the learned distributions is better than ordinal ones. Is it because of a hyperplane between visual vocabulary and the rehashing noise?
4. The author tested on well-known ImageNet dataset, while the performance generalization to other (more challenging) datasets is not clear.

### Questions
1. Can authors analyze the complexity of ReDDiT sampler? Does the efficiency come from less operation needed?
2. The discussion of the limitations of the proposed sampler  and how to improve it would be beneficial for future work of researchers.

### Soundness
2

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
This paper introduces ReDDiT (Rehashing Noise for Discrete Diffusion Transformer), a novel discrete diffusion framework for high-quality visual generation. The authors identify two fundamental limitations in existing
masked visual token models (MVTMs): (1) the use of a
single absorbing (mask) token, which restricts the expressivity of the noise process, and (2) reliance on
heuristic, Gumbel-max–based sampling, which is unstable and requires heavy tuning—especially with large codebooks. To address these, ReDDiT proposes
rehashing noise, which expands the absorbing state into
multiple randomized indices, thereby enriching the diffusion trajectory space. It further introduces a
principled rehash sampler grounded in discrete diffusion theory, which uses multinomial sampling over softmax probabilities to ensure high diversity and low discrepancy without ad-hoc randomness. On ImageNet 256×256, ReDDiT
achieves a gFID of 1.61, significantly outperforming prior discrete models (e.g., MaskGIT: 6.18) and approaching the performance of top continuous diffusion models, while maintaining superior inference efficiency.

### Strengths
1.Significant empirical improvement: ReDDiT delivers a substantial leap in generation quality for discrete models, closing much of the gap with continuous diffusion while preserving the efficiency advantages of discrete token-based generation.
2. Insightful problem diagnosis and elegant solution: The paper clearly articulates the shortcomings of single-mask noise and Gumbel-based sampling, and the proposed rehashing noise mechanism is both theoretically motivated and practically effective.
3. Clear presentation and thorough evaluation: The writing is accessible, and the paper includes comprehensive ablations, visualizations (e.g., Figure 1, 3), and comparisons that convincingly demonstrate the contribution of each component.

### Weaknesses
1.Minor performance gap with best continuous models: While ReDDiT achieves gFID = 1.61, the best continuous models (e.g., MDTv2) report gFID ≈ 1.58 under similar settings. Although the efficiency advantage is compelling, the paper could more explicitly acknowledge this small but notable gap as a limitation or future direction.
2. Tokenizer-dependent hyperparameter tuning: The optimal noise capacity m varies with the tokenizer (e.g., m=128 for LlamaGen-f8 vs. m=1024 for IBQ), requiring empirical tuning. This slightly undermines the method’s plug-and-play appeal and generalizability across tokenization schemes.

### Questions
1.The paper successfully adapts Representation Alignment (RepA) from continuous to discrete diffusion. It would strengthen the work to include a brief discussion (even in the appendix) on why aligning discrete token embeddings with continuous DINOv2 features is effective—e.g., whether it stabilizes training by providing semantic priors in the absence of gradient flow through the tokenizer.
2. In Figure 4(a), the x-axis should be labeled “Training Steps” (or similar) for clarity.

### Soundness
4

### Presentation
3

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
This paper presents ReDDiT, a novel discrete visual generative model that improves upon traditional discrete diffusion models by introducing rehashing noise. This innovation addresses limitations in absorbing states and sampling diversity, expanding latent variable paths to enhance both diversity and image quality. The model outperforms existing approaches, including MaskGIT, with significant gains in gFID and Inception Score.

### Strengths
1. The introduction of rehashing noise provides a novel way to enrich latent variable traversal, offering improved diversity and higher quality generation in discrete diffusion models.

2. ReDDiT outperforms the baseline models (MaskGIT and DDM) on critical metrics like gFID and IS, with competitive efficiency when compared to continuous models.

3. The model works effectively with large vocabulary codebooks (up to 16,384 entries), demonstrating its robustness even when scaled.

### Weaknesses
1. Although ReDDiT reports strong numbers on ImageNet‑1K, its effectiveness on more complex or diverse datasets is not discussed. Moreover, for a generation paper, the visualization evidence is quite limited, making it difficult to fully assess the qualitative improvements or appreciate the contribution beyond the single benchmark.

2. The paper does not discuss potential limitations or failure cases of rehashing noise, especially under large‑vocabulary tokenizers or more difficult semantic distributions. Without such analysis, the robustness and stability of the approach remain unclear.

3. How does the rehashing noise behave when scaling to far larger category spaces or higher‑complexity images? Is there a regime where the increased latent path diversity starts to hurt performance (e.g., over‑randomization), and if so, how is this controlled or prevented?

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
