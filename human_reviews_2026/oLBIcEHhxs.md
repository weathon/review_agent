# CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose $\textbf{CoSpaDi}$ ($\textbf{Co}$mpression via $\textbf{Spa}$rse $\textbf{Di}$ctionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, $\textbf{CoSpaDi}$ leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse–dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. 
We evaluate $\textbf{CoSpaDi}$ across multiple Llama and Qwen models under per-layer and per-group settings at $20-50$\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. 
Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel training-free LLM compression framework called CoSpaDi, which leverages Sparse Dictionary Learning combined with calibration data to compress the weight matrices of large language models. This method decomposes the weight matrix into a dense dictionary and a column-sparse coefficient matrix, providing a more flexible joint subspace representation compared to traditional low-rank approximation. Through experiments conducted on Llama and Qwen models, the paper demonstrates that the proposed framework achieves superior performance over existing methods, especially at compression ratios of 20-50\%.

### Strengths
1. The paper is well-written with a sound structure. The methodology section includes detailed derivations and pseudocode, and the figures and tables are easy to understand.
2. The method requires no fine-tuning and only a small amount of calibration data, which makes it highly attractive for deployment scenarios with limited resources.
3. Comprehensive evaluations were conducted on multiple models (Llama-3.2 1B, Llama-3 8B, Qwen-3 0.6B/8B, Llama-2 7B) and multiple datasets (8 zero-shot datasets, WikiText, and LAMBADA-OpenAI datasets), covering both layer-wise and cross-layer settings. The performance of the proposed method outperforms the SVD and SVD-LLM baselines.

### Weaknesses
1. Although the authors provided the inference speedup results for each projection layer of the model in Figures 4–6, they did not present the end-to-end inference speedup results of the model.  
2. The performance degradation is excessive. When using the CoSpaDi method with a compression ratio of 0.2, the average zero-shot accuracy of the Llama-3 8B model drops by 8.6%; at a compression ratio of 0.5, the accuracy drops by 41.99%. For the smaller Llama-3.2 1B model, the performance degradation is even more severe. This significantly undermines the practical deployment applicability of the compressed models obtained via the CoSpaDi method, as models with such severe performance degradation will perform poorly in real-world scenarios.  
3. The model scale tested is limited. The paper primarily conducts tests on models ranging from 1B to 8B parameters and does not include larger models (e.g., 13B/70B). It is recommended to supplement experiments on larger models to verify scalability.  
4. There is a lack of comparison with Dobi-SVD [1], a more advanced SVD-based method.  
5. There is a lack of comparison with unstructured pruning methods, such as SparseGPT [2] and Wanda [3]. Although the authors compared with LLM-Pruner (a structured pruning method), they failed to compare with unstructured pruning methods—which are compression approaches with superior performance. Specifically, SparseGPT and Wanda can achieve lossless zero-shot accuracy compression for models at a 40% pruning ratio, which is clearly superior to the CoSpaDi method.  

[1] Qinsi W, Ke J, Tomizuka M, et al. Dobi-svd: Differentiable svd for llm compression and some new perspectives [C]. The Thirteenth International Conference on Learning Representations.

[2] Frantar E, Alistarh D. Sparsegpt: Massive language models can be accurately pruned in one-shot [C]. International conference on machine learning. PMLR, 2023: 10323-10337.

[3] Sun M, Liu Z, Bair A, et al. A Simple and Effective Pruning Approach for Large Language Models [C]. The Twelfth International Conference on Learning Representations.

### Questions
1. What is the end-to-end overhead of the proposed method when used to compress models with different parameter scales?  
2. Why is K-SVD chosen as the dictionary learning algorithm? What are its advantages in LLM compression?  
3. How do the compressed LLMs obtained via the CoSpaDi method perform in answering real-world questions? Can corresponding case studies be provided?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an LLM compression method that replaces low-rank approximation with sparse factorization (each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix) optimized with alternating steps of K-SVD. The authors show that this representation is more expressive and leads to a smaller approximation error.

### Strengths
1. **Novelty.** The authors apply dictionary learning, a technique known from signal and image processing, to improve on top of standard low-rank approximation while compressing LLM weights. They evaluate their method on a variety of models and benchmarks.
2. **Significance.** The paper addresses the problem of reducing the size of LLM parameters, which is crucial for deploying these models.
3. **Clarity.** The paper is well-written and describes the proposed method in a clear way.

### Weaknesses
1. **Compression time.** While L468 mentions that the current implementation is not optimal, it would be helpful to know how the compression time scales with model dimensions (now and with a potentially faster implementation), as well as how long it takes to compress the models used for the experiments.
3. **Only small models are evaluated.** The paper only mentions experiments on 1-8B models. If possible, it might be useful to evaluate larger models to make sure that the method still outperforms the baselines at scale.
2. **Optimal $\rho = k/s$ ratio.** Figure 2 doesn't clearly show that $\rho = 2$ is optimal (instead, $\rho = 1.6$ or $1.8$ seem the best), though I acknowledge that choosing a better $\rho$ would only make the proposed method stronger. Still, this figure is generally hard to read, and it might be useful to split it into three figures (one for each compression ratio) to highlight the accuracy/perplexity differences.
4. **Typos.** The paper often uses \cite{} in place of \citep{}.

### Questions
1. How does the compression time scale with model dimensions? How long does it take to compress the models used for the experiments?

### Soundness
3

### Presentation
4

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
This paper proposes CoSpaDi, a training-free compression framework for large language models that employs sparse dictionary learning as an alternative to conventional low-rank decomposition methods. The key innovation lies in representing each weight matrix as a product of a dense dictionary and a column-sparse coefficient matrix, enabling a union-of-subspaces representation where different weight columns can be approximated in distinct subspaces. The method incorporates calibration-guided optimization to minimize functional reconstruction error in the activation space rather than direct weight approximation. The authors provide comprehensive theoretical derivations and demonstrate superior performance over state-of-the-art low-rank baselines (SVD-LLM, ASVD, Basis Sharing) on Llama and Qwen models at 20-50% compression ratios, evaluated on standard language understanding benchmarks. The framework supports both per-layer and cross-layer compression scenarios and is compatible with post-training quantization.

### Strengths
1. **Strong theoretical foundation:** The paper provides rigorous mathematical derivations connecting low-rank approximation to PCA (Appendix A.1), deriving the data-aware optimization objective (Appendix A.2), and establishing the compression ratio formulation (Eq. 9-10). The theoretical treatment is comprehensive and well-presented, clearly distinguishing the proposed union-of-subspaces representation from traditional single-subspace low-rank methods.

2. **State-of-the-art model selection:** The experimental evaluation employs very recent and relevant models including Llama-3.2, Llama-3, Qwen-3, and Llama-2 across different scales (0.6B to 8B parameters), demonstrating the method's applicability to modern LLM architectures.

3. **Consistent performance gains:** CoSpaDi demonstrates substantial improvements over SVD-based baselines across multiple compression ratios, with particularly strong results at moderate compression levels (20-30%), showing both better accuracy and lower perplexity.

### Weaknesses
1. **Limited evaluation benchmarks:** The evaluation relies heavily on relatively outdated tasks (PIQA, HellaSwag, ARC, SciQ from 2017-2019, MMLU from 2021). For a paper targeting state-of-the-art models like Llama-3 and Qwen-3, the benchmark suite should include more recent and practically relevant evaluations:

2. **Poor presentation quality and readability:**
- Figure 1 is particularly problematic: The use of gradient colors is visually unappealing and does not effectively convey the conceptual differences between low-rank decomposition and dictionary learning. The color scheme is distracting rather than informative.
- Inconsistent figure quality: Figures 2-7 vary significantly in style and quality, lacking a unified visual language
- Table formatting: Tables 2-6 are dense and difficult to parse quickly; better use of spacing, grouping, and highlighting would improve readability
- Complex notation overload: The main text introduces substantial notation (W, D, S, L, X, Y, etc.) without sufficient visual aids or intuitive explanations for readers less familiar with dictionary learning

### Questions
1. **Benchmark selection rationale:** [Key Concerns] Can the authors explain why more recent and practical benchmarks were not included? For instance, technical reports for Llama-3, 3.2 and Qwen-3 emphasize code generation, mathematical reasoning, and domain-specific capabilities. Would including these tasks change the relative performance of CoSpaDi vs. baselines?

2. **Dictionary interpretability:** Do the learned dictionary atoms have any semantic interpretation? For example, do certain atoms consistently activate for specific linguistic phenomena or token types?

3. **Coefficient quantization:** Table 1 shows mantissa truncation results. Have the authors explored other quantization schemes (e.g., k-means quantization, learned quantization) that might preserve more information in the sparse coefficients?

### Soundness
3

### Presentation
2

### Contribution
3
