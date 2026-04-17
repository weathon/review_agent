# Group Critical-token Policy Optimization for Autoregressive Image Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Recent studies have extended Reinforcement Learning with Verifiable Rewards (RLVR) to autoregressive (AR) visual generation and achieved promising progress.
However, existing methods typically apply uniform optimization across all image tokens, while the varying contributions of different image tokens for RLVR's training remain unexplored. 
In fact, the key obstacle lies in how to identify more critical image tokens during AR generation and implement effective token-wise optimization for them. 
To tackle this challenge, we propose $\textbf{G}$roup $\textbf{C}$ritical-token $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{GCPO}$), which facilitates effective policy optimization on critical tokens. 
We identify the critical tokens in RLVR-based AR generation from three perspectives, specifically: 
$\textbf{(1)}$ Causal dependency: early tokens fundamentally determine the later tokens and final image effect due to unidirectional dependency;
$\textbf{(2)}$ Entropy-induced spatial structure: tokens with high entropy gradients correspond to image structure and bridges distinct visual regions;
$\textbf{(3)}$ RLVR-focused token diversity: tokens with low visual similarity across a group of sampled images contribute to richer token-level diversity. 
For these identified critical tokens, we further introduce a dynamic token-wise advantage weight to encourage exploration, based on confidence divergence between the policy model and reference model.
By leveraging 30\% of the image tokens, GCPO achieves better performance than GRPO with full tokens.
Extensive experiments on multiple text-to-image benchmarks for both AR models and unified multimodal models demonstrate the effectiveness of GCPO for AR visual generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This papers proposes Group Critical-token Policy Optimization for autoregressive image generation models drawing insights from Reinforcement Learning with Verifiable Rewards in the context of LLMs and Chain of Thought reasoning.

### Strengths
- The paper identifies a more efficient and insightful method to improve RL post-training for AR models.
- The paper shows good empirical performance
- The paper carries out extensive experiments

### Weaknesses
The paper does not discuss the recent Visual Autoregressive VAR paradigm: https://arxiv.org/pdf/2404.02905
There are works that have used Group Relative Policy Optimization for VAR such as 'Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization': https://arxiv.org/abs/2505.23331

These should be discussed in the text, and extensions of GCPO to VAR should be mentioned as a future research direction. The problem with standard AR is that they impose a raster-scan inductive bias on image generation which does not align with the multi-scale nature of images: there is no reason to believe that the top left corner should condition the bottom right corner and not the other way around. As the authors mention 'early tokens serve as global priors and structural guides', which more closely aligns with a multi-scale approach.

### Questions
The paper relies on heuristics which may make the method brittle and sensitive to hyper-parameters:

- The assumption is that the first K tokens (e.g., 10% of the sequence length ) are critical because their initial generation decisions propagate and determine the overall image structure (why this particular number?)
- The 2D gradient of the token entropy map. The authors observed empirically that tokens with high entropy gradients consistently correspond to important image structures and region-bridging elements. This is interesting, any theoretical insights or links to grad-cam and interpretability methods?
- Token diversity within a group of generated images. The criterion selects tokens with the lowest average pairwise cosine similarity of their embeddings across a batch of images. The rationale is that low-similarity tokens are associated with more complex regional structures that contribute richer information for policy optimization, while high-similarity tokens are mundane background or texture. Any further experiments to check this? Perhaps and appendix with more examples and images.

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
This paper introduces Group Critical-token Policy Optimization (GCPO), a novel RLVR framework that focuses on optimizing critical tokens, regarding to causal dependency, structure importance, and token diversity, for autoregressive image generation, achieving significant improvements in efficiency and performance over GRPO across benchmarks.

### Strengths
* Novelty: Introduces a critical-token selection strategy based on causal dependency, entropy gradients, and token diversity.
* Exploration: Provides intriguing insights into the role of structure in token entropy during text-to-image generation.
* Comprehensive Validation: Demonstrates improvements across Geneval, T2I-CompBench and DrawBench benchmarks.

### Weaknesses
* Limited Gains: Improvements over GRPO are minor, especially for Geneval on 1B models and T2I-CompBench on 7B models, despite added complexity.

### Questions
1. Given the limited performance gains, could the authors clarify additional advantages, such as the impact on RL training time due to token filtering?
2. How would GCPO perform on non-spatial vision tokenizers like VAR’s application on T2I [1] and DDT [2] for vision token extraction?

[1] Infinity∞: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

[2] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens

### Soundness
3

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
4

### Summary
Current RL methodologies typically assign similar weights to all image tokens, but different tokens are expected to play different roles. To distinguish among tokens, the authors identify three types of critical tokens based on the causal nature of autoregressive models, token entropy, and token diversity. In addition, they apply differentiated weights by comparing each identified token with the initialization model.

### Strengths
- Considering that different image tokens play different roles in image generation, it is reasonable to focus on optimizing critical image tokens using RL.
- The paper provides the valuable insight that image tokens with high entropy gradients convey spatial structural information of the generated images.
- The proposed RL method demonstrates better performance than GRPO on both Janus-Pro and LLaMaGen models.

### Weaknesses
- Since critical tokens can change during training, they should be re-identified at each training step, which may introduce substantial additional computation. However, the paper lacks a comparison of training cost/consumption.
- The performance of the 1B model drops on the GenEval benchmark in Table 1. What is the reason?
- Figure 7(a) only shows the effect of the selection ratio for each token type, while the analysis does not address the choice of the total selection proportion (e.g., 30%).

### Questions
Since critical tokens may overlap across the defined categories, does this situation require additional processing?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes GCPO, which selects critical tokens through multiple criteria to improve GRPO for T2I generation. The authors identify the critical tokens as initial tokens, tokens with high entropy gradients, and tokens with low average similarities. A dynamic advantage weight is also proposed to balance exploration and stabilization.

### Strengths
1. The idea is intuitive and well-motivated.
2. Section 4 provides solid experimental observations, which support the proposed criteria for selecting critical tokens.
3. The proposed method is simple and appears to be effective in improving generation performance.

### Weaknesses
1. Figure 7 shows that “30% Critical Tokens with DAW” performs better than the “100% baseline,” while “30% Critical Tokens” performs worse. How does the model perform when DAW is applied to the “100% baseline”? It is important to determine whether DAW or the critical tokens contribute more to the overall performance.
2. To my understanding, $Z_{struct}$ is selected within a single image, while $Z_{sim}$ is selected across all images. Should $Z_{struct}$ therefore come from different positions for each image, and $Z_{sim}$ from the same positions across images? If so, why does $Z_{sim}$ come from different positions as shown in Figure 5?
3. The selection ratio (along with the ratio of each token type) appears to be an important hyperparameter. The current setting is 10%:10%:10%. It would be better if the authors could provide results for additional selection ratio settings.

### Questions
1. How does the group size G affect performance? Does it have a strong positive correlation?
2. Is the local averaging of the entropy map very important? It is intuitively reasonable, but it would be better to include an ablation study to verify its effect.

### Soundness
2

### Presentation
2

### Contribution
2
