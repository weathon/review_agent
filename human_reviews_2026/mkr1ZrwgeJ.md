# MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Essential to visual generation is efficient modeling of visual data priors. Conventional next-token prediction methods define the process as learning the conditional probability distribution of successive tokens. Recently, next-scale prediction methods redefine the process to learn the distribution over multi-scale representations, significantly reducing generation latency. However, these methods condition each scale on all previous scales and require each token to consider all preceding tokens, exhibiting scale and spatial redundancy. To better model the distribution by mitigating redundancy, we propose Markovian Visual AutoRegressive modeling (MVAR), a novel autoregressive framework that introduces scale and spatial Markov assumptions to reduce the complexity of conditional probability modeling. Specifically, we introduce a scale-Markov trajectory that only takes as input the features of adjacent preceding scale for next-scale prediction, enabling the adoption of a parallel training strategy that significantly reduces GPU memory consumption. Furthermore, we propose spatial-Markov attention, which restricts the attention of each token to a localized neighborhood of size (k) at corresponding positions on adjacent scales, rather than attending to every token across these scales, for the pursuit of reduced modeling complexity. Building on these improvements, we reduce the computational complexity of attention calculation from (\mathcal{O}(N^{2})) to (\mathcal{O}(N k)), enabling training with just eight NVIDIA RTX 4090 GPUs and eliminating the need for KV cache during inference. Extensive experiments on ImageNet demonstrate that MVAR achieves comparable or superior performance with both small model trained from scratch and large fine-tuned models, while reducing the average GPU memory footprint by  3.0x.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Markovian Visual AutoRegressive modeling (MVAR). Unlike conventional next-scale prediction methods that condition each scale on all previous scales and require each token to attend to all preceding tokens, MVAR introduces scale and spatial Markovian assumptions to reduce redundancy and computational complexity. Specifically, the model only conditions on adjacent scales (scale-Markov) and restricts attention to local neighborhoods (spatial-Markov), significantly lowering GPU memory usage. Experiments on ImageNet demonstrate that MVAR achieves comparable or superior performance to existing methods while reducing memory consumption by up to 3-4$\times$.

### Strengths
- This work introduces Markovian properties into VAR generation process, making the proposed MVAR achieves more efficient training and eliminates the need for a KV cache.

- The paper is well-written and easy to follow.

### Weaknesses
- In the proposed spatial Markov property, it seems that the authors restrict each pixel $i$ to interact only with its neighboring pixels within the same scale. Although the term “Markov” is used, performing attention within local neighborhoods has already been extensively explored in previous works.

- In the visualization of attention maps in Figure 3, the behavior of the attention maps produced by this paper appears quite different from those of prior works on VAR like FastVAR[1]. In previous work, the visualized attention maps exhibit strong cross-scale interactions in their attention visualizations. I would like to know how the authors generated Figure 3.

- In Figure 7(b), the attention mask may be incorrect, the current figure seems to consider only intra-scale token interactions while ignoring tokens from the previous scale.

- The paper lacks comparisons with other recent AR-based approaches, such as MAR[2]. Including these comparisons would better situate the proposed MVAR within the broader context of image generation methods.

> [1]FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning. ICCV25

> [2]Autoregressive Image Generation without Vector Quantization. NIPS24

### Questions
- How was the visualization in Figure 2 obtained? Which dataset and how many samples were used?

- Have the authors designed CUDA kernels for the spatial Markov attention? Does the current implementation affect inference efficiency?

### Soundness
2

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
This paper proposes a novel visual autoregressive framework, MVAR, which introduces scale-Markov (depending only on adjacent scales) and spatial-Markov (focusing only on a local neighborhood) assumptions. This approach addresses the significant computational and memory redundancy in existing "next-scale prediction" models, reducing attention complexity from $O(N^2)$ to $O(Nk)$, cutting down GPU memory usage, and enabling an inference process that is entirely free of a KV cache.

### Strengths
1. The paper accurately identifies and solves a critical bottleneck in existing VAR models related to memory and computation (especially the KV cache). This is a practical problem that has hindered scaling these models to larger sizes and higher resolutions.
2. MVAR's two core components (scale-Markov trajectory and spatial-Markov attention) are conceptually simple but highly effective. The "KV-cache-free" and "parallel training" properties drastically reduce the training cost.

### Weaknesses
1. While the scale-Markov assumption proves effective in the current setup, it might be an approximation. In more complex scenarios requiring strong long-range dependencies (e.g., images with complex global structures or multi-object interactions), discarding all information from $r_1$ to $r_{l-2}$ could become a performance bottleneck.
2. The paper mentions that training for $r_1$ to $r_8$ is parallel, but $r_9$ and $r_{10}$ (which account for 60% of the tokens) seem to be handled separately using single-scale conditional modeling. The complexities of this mixed training strategy and its impact on final performance consistency are not sufficiently discussed in the main text.

### Questions
1. The ablation study (Table 3) shows that method (d) achieves a better FID than using all prior scales. Does this imply that the original VAR design (relying on all scales) is not just redundant, but potentially a **harmful inductive bias**? Does it force the model to process irrelevant historical information, thereby interfering with the generation?
2. MVAR uses a fixed $k \times k$ neighborhood. Have you considered using dynamic or deformable neighborhoods? For example, using a larger $k$ in smooth areas of an image and a smaller $k$ in complex, textured regions?
3. Appendix B mentions that the causal mask is disabled for the last two scales. Does this mean that within these scales, token generation is parallel (i.e., non-autoregressive)? If so, does this contradict the model's definition as an "autoregressive" framework? Please clarify how $p(r_l | \eta_k(r_{l-1}))$ is modeled for these final two scales.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Markovian Visual AutoRegressive Modeling (MVAR), an efficient framework for visual autoregressive generation. The method introduces Markovian assumptions in both scale and spatial dimensions: each scale depends only on its immediately preceding scale, and each token attends only to a local neighborhood. This design effectively removes redundant dependencies in next-scale prediction and reduces attention complexity from $O(N^2)$ to $O(Nk)$. Experiments on ImageNet show that MVAR achieves similar or slightly better generation quality compared to vanilla VAR while using significantly less memory and allowing parallel training.

### Strengths
- The motivation is clear and well supported by empirical observations of redundancy in next-scale prediction.
- The Markovian formulation is conceptually simple yet brings strong computational benefits.
- The method achieves 3× memory reduction without degrading generation quality.
- Results on ImageNet demonstrate good efficiency–performance trade-offs, making the approach practical for large-scale settings.
- The paper is overall well written and easy to follow.

### Weaknesses
- The experimental comparison is limited. The paper does not include results against related methods such as Randomized Autoregressive Visual Generation (RAVG, 2024), which also targets efficiency improvement in visual AR models.
- Table 1 reports results compared with VAR-d16, but it is unclear whether the MVAR model also uses 16 decoder layers. The discrepancy between Table 1 and Table 2 results (different FID/IS scores) suggests inconsistent model settings.
- Several experiments in the appendix only report FID without IS or precision/recall, making the evaluation incomplete.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MVAR, which exploits scale and spatial redundancy based upon Visual Autoregressive Modeling (VAR). Specifically, MVAR only uses the preceding scale for next-scale prediction, and restricts the attention map of each token to a localized neighborhood. These two changes significantly reduces the computational complexity from O(N^2) to O(NK), and eliminate the need for KV cache during inference.

### Strengths
1. This paper is overall well written and rather easy to follow. The figures look nice and captures the core idea in a glimpse.

2. The scale and spatial redundancy in original VAR work makes intuitive sense to me, and the proposed solution is both straightforward and effective as shown in experiments.

3. MVAR shows both performance advantages over vanilla VAR, as well as reduced memory footprint and accelerated inference speed.

### Weaknesses
1. Only model size of 300M is studied in this paper. It is unclear if MVAR shows good scalability or not. The authors are encouraged to show the scaling trend following VAR (up to 2B model).

2. Beyonds ImageNet unconditional generation, The authors are encouraged to try the image in-painting and out-painting task and class-conditional image editing task as in the zero-shot setup in VAR paper.

### Questions
1. Based on Table 1, the introduction of markovian scale prediction and localized attention leads to even better performance (FID 3.09 vs 3.55, IS 285.5 vs 280.4)? Would the authors care to explain thsi phenomenon?

2. And how much performance gain does scale markovian conditioning along brings?

3. Why is it called "spatial Markovian conditioning"? Localized attention is a common operation, dating back to the days of Swin-Transformer. I do not see any markovian part in localized attention.

### Soundness
4

### Presentation
3

### Contribution
3
