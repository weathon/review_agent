# LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error

- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Deploying large language models (LLMs) on platforms with insufficient computational resources remains a key challenge. Weight pruning is an efficient model compression technique that can reduce model size without retraining LLMs. However, due to the massive number of parameters, it is infeasible to estimate the importance of weights globally, and most prior studies assign a uniform sparsity ratio across all layers. Recent findings reveal that layers contribute unevenly to LLM performance, making it necessary to investigate Layer-wise importance. Existing Layer-wise sparsity allocation methods, such as OWL and DLP, rely on weight scoring and carefully designed score proxies to estimate Layer-wise importance and sparsity ratios, while enforcing identical sparsity to blocks and projection weights within a layer to avoid performance degradation. In this work, we propose Layer-wise Sparsity Allocation (LSA) for LLM pruning, which quantifies Layer-wise importance by evaluating the minimal linear reconstruction error (LSE) of each transformer layer under the assumption that 50\% of its least important weights are removed. Moreover, our method supports non-uniform sparsity allocation at block- or projection-level granularity within layers, without incurring catastrophic performance degradation. Experimental results demonstrate that LSA maintains high performance at high sparsity levels. At an overall sparsity ratio of 70\%, LSA surpasses state-of-the-art methods across language modeling tasks and seven zero-shot tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Layer-wise Sparsity Allocation (LSA), a new method for pruning large language models by measuring layer importance via minimal linear reconstruction error (LRE) rather than relying on heuristic weight scores such as those used in Wanda, OWL, or DLP. The method supports finer-grained sparsity allocation (block- and projection-level) without catastrophic performance degradation. Extensive experiments on multiple LLMs (LLaMA1/2/3, Mistral, Qwen) show that LSA achieves lower perplexity and higher zero-shot accuracy than prior methods at high sparsity (70%), while also providing measurable inference speedup.

### Strengths
1. The paper introduces a new way to measure layer importance directly through reconstruction error, avoiding arbitrary scoring or reduction functions.
2. LSA is conceptually simple and model-agnostic, applicable to unstructured pruning and extensible to block/projection levels.
3. Extensive experiments on multiple LLMs (LLaMA1/2/3, Mistral, Qwen) show that LSA achieves lower perplexity than prior methods at high sparsity (70%), while also providing measurable inference speedup.

### Weaknesses
1. While the proposed method shows clear gains in perplexity, the improvements on the seven zero-shot benchmarks appear relatively marginal compared to OWL and DLP. The authors are encouraged to provide further discussion on this part.
2. The notation in the Method section ( e.g. $S_w$, $S_x$ ) is not clearly defined, which makes the mathematical formulation somewhat difficult to follow. Additionally, the method description is overly verbose and could be streamlined for better readability and comprehension.

### Questions
1. The proposed LRE offers a novel perspective for measuring layer importance. However, the paper lacks a thorough comparison or discussion of how LRE relates to previously established metrics, such as Hessian-based saliency or gradient magnitude. It would strengthen the work to analyze potential correlations between LRE and these existing indicators, clarifying whether LRE captures complementary or more comprehensive information. Furthermore, while the empirical performance of LRE-based pruning is impressive, the underlying intuition and theoretical motivation behind why LRE serves as a better proxy for layer importance remain insufficiently explained.

2. It is great to see that the paper includes experiments on structured pruning and N:M sparsity. However, these results would be more convincing if they also included comparisons with existing layer-wise sparsity allocation methods such as OWL and DLP, to better contextualize the advantages of LSA in structured settings.

3. The authors claim that the sparsity ratio $p$ used to compute the linear reconstruction error is not a sensitive hyperparameter. However, the provided evidence is limited to models of similar size and within the same family. A more convincing analysis would involve evaluating across different model families and scales to demonstrate broader generalization.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates layer-wise Sparsity Allocation (LSA) for LLM pruning, which quantifies layer-wise importance by evaluating the minimal linear reconstruction error (LSE) of each transformer layer under the assumption that 50% of its least important weights are removed.

### Strengths
The method supports non-uniform sparsity allocation at block- or projection-level granularity within layers, without incurring catastrophic performance degradation. It enables the assignment of distinct sparsity ratios to different projections within the same
Transformer layer, providing a finer-grained allocation that can lead to performance improvements.

### Weaknesses
The novelty of the proposed method may be limited. It follows traditional methods to compute the layer importance first and then allocate the sparsity for each layer according to the importance. The framework is very similar to previous works such as DLP, including the importance score and the sparsity allocation range between [pr - $\beta$, pr + $\beta$].  Some detailed part may be different, but the framework and other parts are similar to provious works. The technical contribution may be limited. 

The experimental results with detailed baseline comparsion mainly focus on 70% sparsity. The results of other sparsity are very rare, and other sparsity does not have baseline results. It is better to compare with baselines under different sparsity ratios to demonstrate the general performance, not only under 70% sparsity. 

It mentions to be more efficient. It is better to provide more detailed analysis about the efficiency such as complexity comparison. It is hard to see why it is more efficient, more discussions with detailed complexity demonstration can enhance this part. 

In the experiemnts, the improvements seem to be marginal. The performance is very close to DLP with a very small gap. Some results are different from the original DLP paper. For example, in table 5, for sparsegpt on Llama2 7B, it reports LSA with PPL 18.63 and DLP with 18.68 PPL.  The gap is very small. And in the original DLP paper, the PPL is 18.58 under the same configuration, which is better than 18.63.  Different runs may lead to different results with small variantions. The current gap is minor, and another run may change the result leading to a better baseline. The appendix also shows that in many cases, the baselines can outperform the proposed method. The  experimental improvements  seem to be marginal, and it does not seem to be significantly better than baselines. 

There are some other methods which also investigate the sparsity allocation for different layers, such as [R1,R2]. It is better to discuss the comparison with more baselines. 

SparseGPT and wanda are basic llm pruning methods. There are more advanced pruning methods such as [R3], which can outperform SparseGPT with large margins under uniform sparsity. It is better to combine the proposed method with more advanced pruning methods to demonstrate the advantages. 

[R1] Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models

[R2] Adaptive Layer Sparsity for Large Language Models via Activation Correlation Assessment

[R3] Fast and Effective Weight Update for Pruned Large Language Models

### Questions
See the weakness.

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
This paper proposes LSA (Layer-wise Sparsity Allocation), a novel method for pruning large language models (LLMs) that quantifies layer-wise importance by evaluating the minimal linear reconstruction error (LRE) under the assumption of removing 50% of the least important weights. LSA avoids weight scoring and empirical reduce functions, enabling non-uniform sparsity allocation at finer granularities (block- or projection-level) without performance degradation. Experiments show LSA outperforms state-of-the-art methods like OWL and DLP at high sparsity levels (e.g., 70%), achieving better performance on language modeling and zero-shot tasks while supporting efficient inference and fine-tuning.

### Strengths
1.LSA introduces minimal linear reconstruction error as a direct measure of layer importance, eliminating the need for weight scoring or manual reduce function design.

2.LSA consistently outperforms OWL and DLP across multiple models (e.g., LLaMA, Vicuna) and tasks (e.g., WikiText perplexity, zero-shot accuracy) at high sparsity levels (70%).

### Weaknesses
1.The authors claim that LSA is the first method to achieve projection-level sparsity allocation, but TRIM[1] had previously implemented even finer-grained allocation: assigning sparsity rates to rows and columns of matrices.

2.These sparsity allocation methods cannot be applied to the currently most commonly used 2:4 semi-structured pruning, and are limited to unstructured pruning only. However, unstructured pruning cannot be accelerated by GPUs, which limits the practical applications of such methods.

3.As shown in Table 6, LSA only shows slight improvements over DLP in most scenarios. Why didn't the authors conduct zero-shot task performance comparisons on Llama3? On more powerful models, the advantages of LSA over DLP might be more prominent.

[1]Beck, Florentin, William Rudman, and Carsten Eickhoff. "TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning." arXiv preprint arXiv:2505.16743 (2025).

### Questions
As shown in Tables 1 and 2, Layer-wise allocation outperforms Block-wise in most cases on LLaMA1-7B and LLaMA2-7B. However, on LLaMA3-8B, Block-wise allocation significantly surpasses Layer-wise. Do the authors provide any insights into this discrepancy?

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
4

### Summary
This paper introduces LSA, a pruning method for large language models that measures each layer’s importance using *minimal linear reconstruction error* instead of weight-based scoring. By directly quantifying how much information each layer loses when half of its least important weights are removed, LSA assigns non-uniform sparsity ratios across layers, blocks, and even projection levels, achieving fine-grained pruning without performance collapse. Extensive experiments on LLaMA, Vicuna, Mistral, and Qwen models show that LSA consistently surpasses state-of-the-art methods like OWL and DLP in perplexity, zero-shot accuracy, and inference speed, demonstrating its robustness and generalization across architectures.

### Strengths
1. The proposed minimum linear reconstruction error is a meaningful and insightful contribution for both the application and analysis of pruning.

2. The proposed LSA method achieves strong performance in high-sparsity pruning on the LLaMA-1/2 model series, and the experiments are solid and convincing.

3. The paper is clearly written and easy to follow.

### Weaknesses
1. The LLMs used in this paper are outdated and do not reflect state-of-the-art LLMs. I think it is necessary to conduct experiments on the LLaMA‑3 family or the Qwen 2.5/3 series.

2. The manuscript omits several relevant baselines and sparsity-allocation references (e.g., EvoPress, DSA).

[1] Sieberling, O., Kuznedelev, D., Kurtic, E. & Alistarh, D. (2025). EvoPress: Accurate Dynamic Model Compression via Evolutionary Search. ICML. 

[2] Li, L., Dong, P., Tang, Z., Liu, X., Wang, Q., Luo, W., Xue, W., Liu, Q., Chu, X., & Guo, Y. (2024). Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models. NeurIPS.

### Questions
It is necessary to conduct experiments on the LLaMA‑3 family or the Qwen 2.5/3 series.

### Soundness
3

### Presentation
3

### Contribution
3
