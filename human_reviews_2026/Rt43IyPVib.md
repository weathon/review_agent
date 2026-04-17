# Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Large language models have achieved remarkable success in recent years, primarily due to the implementation of self-attention mechanisms. However, traditional Softmax attention suffers from numerical instability and reduced performance as the number of inference tokens increases. This paper addresses these issues by proposing a new design principle for attention, viewing it as a two-stage process. We first decompose the Softmax operation into a non-linear positivity transformation and an $l_1$-normalisation step, identifying the latter as essential for maintaining model performance. Our first proposal is to replace the standard exponential function with the more numerically stable Softplus activation and introduce a dynamic scale factor based on invariance entropy, creating a novel attention mechanism that outperforms conventional Softmax attention. Our second proposal is to introduce a re-weighting mechanism that sharpens the attention distribution, amplifying significant weights while diminishing weaker ones. This enables the model to concentrate more effectively on relevant tokens, mitigate the attention sink phenomenon, and fundamentally improves length extrapolation. When combined, these changes ensures numerical stability and dramatically improves length extrapolation, maintaining a nearly constant validation loss at 16$\times$ the training length while achieving superior results on challenging long-context retrieval tasks and standard downstream benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a two-stage attention mechanism designed to address numerical instability and performance degradation in traditional Softmax attention, particularly for long sequences. The proposed LSSA component replaces the exponential function with Softplus and incorporates a dynamic length scale, while the re-weighting mechanism sharpens the attention distribution. The method shows promising results in length extrapolation and passkey retrieval tasks. However, the evaluation could be strengthened by more comprehensive comparisons and deeper analysis to better establish its novelty and practical utility.

### Strengths
- The paper provides a clear decomposition of the Softmax operation, identifying the \(l_1\)-normalization as a critical component, which offers a fresh perspective on attention mechanism design.  
- The proposed LSSA and re-weighting modules are intuitively motivated and empirically shown to improve length extrapolation, especially in challenging tasks like passkey retrieval.  
- The visualizations of attention maps (Appendix A.1) offer qualitative support for the method’s ability to mitigate attention sink and produce sharper attention distributions.

### Weaknesses
1. **Limited and Weak Baselines**  
   The paper compares LSSAR primarily against standard Softmax and a few Softmax-free variants (e.g., ReLU-based and Sigmoid-based attention). However, many recent works have explored efficient attention mechanisms, linear attention, and advanced positional encoding strategies to improve length extrapolation. The absence of comparisons with state-of-the-art methods such as Linear Attention, Performer, or recent long-context transformers (e.g., LongFormer, Mamba) makes it difficult to assess the true contribution of LSSAR. A stronger and more diverse set of baselines is necessary to position this work within the existing literature.

2. **Insufficient Ablation Study**  
   The paper claims that both the normalization stage (LSSA) and the re-weighting stage contribute to the improved performance. However, the ablation study does not clearly isolate the individual impact of each component on length extrapolation. For instance, it remains unclear how much of the improvement comes from the dynamic scaling factor (\(\log \mathbf{N}\)) versus the Softplus activation, or how the re-weighting mechanism performs when applied to other attention forms. A more detailed ablation is needed to validate the design choices and support the two-stage formulation.

3. **Marginal Performance Gains**  
   The reported improvements in perplexity and downstream tasks are relatively modest. For example, the validation loss reductions are small, and the gains on standard benchmarks (e.g., MMLU, ARC) are minimal. Given the additional complexity introduced by LSSAR, the performance benefits do not clearly justify the computational overhead. The paper would benefit from a more thorough analysis of the trade-off between performance and efficiency.

4. **Computational Efficiency**  
   The proposed method introduces several non-standard operations, including Softplus, dynamic scaling, and power-based re-weighting, which are likely to increase both memory usage and inference time. However, no computational analysis or latency comparisons are provided. In practice, such overhead could limit the applicability of LSSAR, especially in resource-constrained settings. A discussion of efficiency and potential optimizations (e.g., kernel fusion) is strongly recommended.

5. **Novelty and Conceptual Contribution**  
   The decomposition of Softmax into a non-linear transformation and \(l_1\)-normalization is not new, and the use of re-weighting to sharpen attention distributions has been explored in prior work. While the combination of Softplus and a length-aware scaling factor is novel, the paper does not sufficiently differentiate its contributions from existing approaches. The conceptual framework—though well-motivated—does not significantly advance the community's understanding of attention mechanisms beyond established knowledge.

### Questions
See the weaknesses section for details

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
Large language models have achieved remarkable success in recent years, primarily due to the implementation of self-attention mechanisms. However, traditional Softmax attention suffers from numerical instability and reduced performance as the number of inference tokens increases. This paper addresses these issues by proposing a new design principle for attention, viewing it as a two-stage process. This paper decompose the Softmax operation into a non-linear positivity transformation and an 
-normalisation step, identifying the latter as essential for maintaining model performance. The author first replace the standard exponential function with the more numerically stable Softplus activation and introduce a dynamic scale factor based on invariance entropy, creating a novel attention mechanism that outperforms conventional Softmax attention. The second proposal is to introduce a re-weighting mechanism that sharpens the attention distribution, amplifying significant weights while diminishing weaker ones. This enables the model to concentrate more effectively on relevant tokens, mitigate the attention sink phenomenon, and fundamentally improves length extrapolation. When combined, these changes ensures numerical stability and dramatically improves length extrapolation, maintaining a nearly constant validation loss at 16x the training length while achieving superior results on challenging long-context retrieval tasks and standard downstream benchmarks.

### Strengths
This paper is clearly written. The authors first decompose Softmax into two steps, then argue that the second step (normalization) is more important. The paper focuses on discussing the first step and improves the activation function (exp), resulting in LSSA and its re-weighting version LSSAR.

### Weaknesses
1. The paper evaluates models of limited scale. The authors only discuss a single model size, but should examine larger models. Based on my experience, a token consumption of approximately 10 billion should be feasible on the authors' GPU infrastructure.
2. The paper lacks discussion on efficient kernel implementations. I am not requesting the authors to implement Triton/CUDA-level kernels, but rather expect a discussion of the mathematical derivation of efficient algorithms. For example, algorithmic descriptions or PyTorch code implementations would be appreciated. Additionally, the paper should analyze the theoretical computational overhead comparison with Flash Attention.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes LSSAR, an attention mechanism that claims to achieve better length extrapolation compared to the standard Softmax attention.

Specifically, the authors first replace the exponential operation in the Softmax function with a scaled Softplus function, where the scaling factor depends on the sequence length and model dimension. They further introduce a re-weighting mechanism that sharpens the attention distribution via a shifted ReLU-p transformation, followed by an L1 normalization to ensure the attention weights sum to one.

A series of ablation studies demonstrate that LSSAR outperforms both the Softmax baseline and other Softmax-free variants in terms of validation loss and passkey retrieval accuracy, indicating stronger extrapolation capability.

### Strengths
1. The paper presents a clear motivation, addressing the length extrapolation limitations of Softmax attention. The authors demonstrate a solid understanding of the key factors affecting extrapolation, such as maintaining entropy invariance, and mitigating attention over-smoothing and distraction.

2. The proposed method, LSSAR, is conceptually simple and easy to implement, while being supported by a reasonable set of experiments.

3. The proposed method exhibits strong extrapolation performance across the evaluated benchmarks, consistently outperforming other softmax-free variants.

### Weaknesses
1. **Baseline implementation details are insufficient.**
In current mainstream LLM implementations (e.g., Qwen3, Gemma3, etc.), Softmax attention typically incorporates QK-Norm, and employs NTK/Yarn or length-scaling factors ($\alpha\log{L}$) in out-of-context settings—techniques that have been widely validated to enhance extrapolation. It is unclear whether these enhancements were applied to the Softmax baselines in this paper, while the proposed LSSAR explicitly includes both QK-Norm and two length-dependent scaling operations.

2. **Several design choices introduce uncertainty.**
The ReLU-p hyperparam $p$ requires careful tuning; as shown in Tab.2, when $p=3$, LSSAR performs worse than Softmax on extrapolation. Moreover, the method involves multiple scaling ($N$ in Eq. 4 and .5) and offset parameter $O$ (in Eq. 5) with specific stability-sensitive configurations. Such fine-grained parameter adjustments could equally enhance Softmax attention, especially regarding scaling factor, making it difficult to isolate the true source of improvement.

3. **Experimental scale is too limited to support strong conclusions.**
The experiments (120M + 10B) represent only early-stage validation and cannot reliably indicate final converged performance. The observed gains may reflect faster convergence due to added nonlinearity or param-tuning complexity, rather than a genuine increase in asymptotic capability. In addition, many benchmark scores in Tab.5 remain noisy (even below random guess levels), which weakens the empirical support for the claims. A more convincing evaluation would include moderately larger model scales (not necessarily billion-level) and report scaling law slopes to quantify robustness.

4. **The paper’s organization and presentation could be improved.**
(a) The experiments involve too many variables, leading to scattered conclusions. Key findings could be better structured—e.g., enumerated or highlighted for clarity.
(b) The full LSSAR formulation should be presented more explicitly. For instance, it is unclear whether Eq.4 and Eq.5 are applied sequentially, and whether the L1 normalization is executed once or twice.

### Questions
1. Did LSSA/LSSAR employ RoPE? If so, was NTK-based extrapolation used? 

2. Are both of the length-related scaling factors in the method necessary ($\log{N}$ in Eq.4 and $N$ Eq.5), and have any ablation studies been conducted to justify their inclusion?

2. For a fair comparison, was the Softmax baseline equipped with a scaling factor (as in the sigmoid baseline) and QK-Norm?

3. In Table 3, the ReLU-p variant performs poorly, which the authors attribute to the suppression of negative values and emphasis on positive activations. However, in the re-weighting stage, this property seems beneficial. Does this imply that only the two-stage Softplus–then–ReLU-p combination is effective? Have you tested other nonlinear formulations for the second stage, such as Softplus–then–exp?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a Length Scaled Softplus Attention (LSSA) combined with an re-weighting scheme, aiming to replace the standard Softmax attention mechanism used in transformers. The proposed model (LSSAR) excels in length extrapolation while ensuring numerical stability, as demonstrated in the long-context passkey retrieval tasks.

### Strengths
1. The decomposition of Softmax operation into none-linear positive transformation and l1-norm is conceptually direct and simple to understand. The re-interpretation helps unify multiple Softmax-free attention variants under a coherent framework.

2. Supported by both quantitative results and visualization analyses, the proposed method shows stronger abilities in improving length extrapolation and reducing attention sink, suggesting that the two-stage normalization and re-weighting design offers a practical and effective improvement over conventional Softmax.

### Weaknesses
1. While the proposed method (LSSA/LSSAR) is conceptually solid and supported by several illustrative experiments, evaluation across a broader set of backbones is necessary to demonstrate robustness and effectiveness beyond a single model family. More baselines should be added to long-context tasks and downstream evaluation.

2. Since efficiency is central to practical deployment, the measurements of runtime or memory profiling relative to optimized kernels are critical for completeness.

### Questions
1. Could the authors provide a theoretical justification or mathematical analysis to support why normalization plays such a fundamental role in Softmax attention? 

2. The paper states that N is an L×L matrix where each element in row i is equal to i, however, it remains unclear whether N is strictly lower-triangular (reflecting causal masking) or a dense matrix? Furthermore, how does logN behave for the first few tokens where Ni = 1?

3. Section 2.2 briefly mentions cosine similarity attention when discussing scaling factors, but the relationship between this mechanism and the proposed LSSA formulation remains unclear. 

4. Why ReLU is chosen for the re-weighting mechanism instead of smoothing functions such as Sigmoid/GELU?

5. Considering that Softplus and Sigmoid are mathematically related through their derivatives, and the results in Table 4 show that the performance of Sigmoid Attention (Ramapuram et al., 2024) is consistently close to that of the proposed LSSAR, could the authors analyze the relationship between the two methods?

6. Can LSSAR be directly incorporated into efficient attention frameworks (e.g., FlashAttention, Mamba-2) without loss of parallelism?

### Soundness
3

### Presentation
2

### Contribution
2
