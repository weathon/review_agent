# Learning Semi-Structured Sparsity for LLMs via Shared and Context-Aware Hypernetwork

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Large Language Models (LLMs) achieve state-of-the-art performance but are costly to deploy in resource-constrained environments. Pruning with $n:m$ semi-structured sparsity reduces computation and enables hardware acceleration, yet existing methods face a trade-off: one-shot approaches are efficient but heuristic, while optimization-based methods are accurate but expensive.  
We introduce \textbf{HyperPrune}, a resource-efficient framework that directly optimizes $n:m$ sparsity. A lightweight hypernetwork, shared across layers and conditioned on learnable embeddings, generates structured masks in a one-shot, layer-wise manner. \textit{Continual pruning} preserves cross-layer knowledge, and \textit{feature outlier regularization} retains critical activations, unifying the strengths of heuristic and optimization-based methods.  
Experiments on LLaMA-7B to 70B show state-of-the-art accuracy–sparsity trade-offs on a single A100 GPU, achieving higher efficiency, accuracy, and scalability than prior approaches. HyperPrune offers a practical, scalable, and hardware-friendly solution for structured LLM pruning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces HyperPrune, a new framework for n:m semi-structured pruning of large language models (LLMs). Unlike traditional one-shot heuristics (e.g., SparseGPT, Wanda) or heavy optimization-based methods (e.g., MaskLLM, MaskPro), HyperPrune uses a lightweight shared hypernetwork conditioned on layer and component embeddings to generate pruning masks in a context-aware, layer-wise fashion.

### Strengths
1. The shared hypernetwork with conditional embeddings is a novel solution to the scalability problem in structured pruning, enabling n:m sparsity without massive parameter overhead.
2. The mutual information theorem provides an interpretable bridge between discrete mask learning and differentiable optimization.
3. Extensive experiment results on the LlaMA-2 family demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The experiments focus primarily on LLaMA-2. It’s unclear how HyperPrune performs on other architectures (e.g., LLaMA-3, Qwen, Mistral, OPT)
2. The hypernetwork setting and training detail is not sufficiently illustrated in the paper, including the network structure, parameters, and training settings.
3. Some equations and notation (e.g., in Section 4.1) are dense and difficult to follow. Simplifying or visually decomposing them would improve readability.

### Questions
1. While the paper highlights HyperPrune’s scalability to 70B models on a single A100 GPU, it lacks a detailed breakdown of training or optimization costs (e.g., wall-clock time, computational budget) compared to other optimization-based pruning methods such as MaskPro or MaskLLM. A more explicit cost–performance analysis would strengthen the paper’s practicality claim.

2. The experimental evaluation primarily focuses on the LLaMA-2 family. It remains unclear how well HyperPrune generalizes to other architectures.

3. The current ablation section briefly evaluates the effects of embeddings and regularizations but does not isolate the impact of the proposed mutual information relaxation. Including such an analysis would better justify its theoretical contribution and clarify how much it contributes to the overall improvement.

4. Although the paper compares HyperPrune against strong baselines like SparseGPT, Wanda, and MaskPro, adding more recent state-of-the-art pruning methods[1][2][3] would further contextualize its advantage.

[1]Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models (ICML 2024) 

[2]AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved Layer-wise Pruning of Large Language Models (NeurIPS 2024)  

[3]Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity (ICML2024)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper relates to pruning LLMs under the constraint of n:m sparsity. This aims at leveraging, e.g. 2:4 sparsity that some computers support in hardware, potentially increasing math throughput by 2x.

This paper sets itself in the context of:
* heuristic methods: fast, but approximate, and unable to support structured sparsity
* optimization methods: accurate but computationally expensive, as they usually operate on the full model for a joint optimization.

The paper proposes an intermediate solution which applies a block-wise optimization, thus only one block (of e.g. 4 parameters) at a time needs to be loaded in memory for the optimization to take place. This is done via a shared hypernetwork (an MLP). Each weight is partitioned into sets of m parameters, and each set is provided as input into the hypernetwork, which outputs logits for mask selection (in the case of 2:4 sparsity, there are 6 possible masks, thus we need 6 logits). 
In order to further improve the efficiency of the hypernetwork, the type of weight (q,k,v, o for attention, u,v,d for FFN) is provided as an input by ways of a contextual embedding.
Under the formulation, the hypernetwork would ignore cross-layer dependencies, thus a regularization loss is applied to encourage the hypernetwork to remain consistent with its pruning of previous layers.
Additionally, the authors observe that LLMs often exhibit intermediate features with very large magnitude. Given that these features are thought to be critically important, a regularization term is used to encourage the hypernetwork to preserve these features. 

The method is validated with experiments that show competitive results against a set of heuristic and one optimization baselines (though MaskLLM is not included in the comparison due to the computational complexity). Results are shown for both perplexity (next token prediction) and accuracy (common benchmarks).

### Strengths
The paper is very well written and proposes an interesting intermediate solution between heuristic and optimization methods for n:m sparsity pruning.

The use of a shared hypernetwork to predict n:m masks is, from what I can tell, novel.

The theoretical (Theorem 1) provides a principled information-theoretic justification for using the reconstruction loss as a surrogate objective under n:m sparsity constraints.

Ablation studies provide justification for each component of the method.

### Weaknesses
No major weakness.

### Questions
On line 237, did you mean "producing *6* logits from m=4"?

What is the embedding dimension $d$ for the layer-specific embeddings $e_\ell$ and component embeddings $t$, and how sensitive is the method's performance to different values of $d$?

What temperature annealing schedule is used for the Gumbel-Softmax relaxation during hypernetwork training, including initial/final values and decay strategy?

From algorithm 1, I understand you train the hypernetwork for each layer until convergence (the "for layer ℓ = 1 to L do" is the outer scope). Did you try a round-robin approach (i.e. with the loop over calibration data being the outer scope?)

### Soundness
4

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
3

### Summary
This paper proposes HyperPrune, a post-training framework for learning n:m semi-structured sparsity in LLMs. The method employs a lightweight shared hypernetwork conditioned on learnable layer and component embeddings to generate structured masks in a layer-wise manner. Two key regularizations are introduced: (1) continual pruning regularization to preserve cross-layer knowledge during sequential pruning, and (2) feature outlier regularization to retain critical activations. The authors provide a theoretical justification by showing that minimizing reconstruction loss is equivalent to maximizing mutual information between dense and pruned models under n:m constraints (Theorem 1). Experiments on LLaMA-2 models (7B/13B/70B) demonstrate competitive performance compared to both heuristic methods (SparseGPT, Wanda) and optimization-based methods (MaskPro) on a single A100 GPU.

### Strengths
1.	Well-motivated approach: The use of a shared hypernetwork with context-aware embeddings provides an efficient solution to the memory scalability challenge of optimization-based pruning methods, enabling layer-wise mask generation without requiring massive GPU resources.
2.	Comprehensive experimental validation: The paper provides extensive evaluation across multiple model sizes (7B/13B/70B), diverse benchmarks (perplexity and zero-shot tasks), and thorough ablation studies analyzing the contribution of each component.
3.	Theoretical foundation: Theorem 1 provides an information-theoretic justification connecting mutual information maximization with reconstruction loss minimization, offering a principled view of the relaxation.

### Weaknesses
1.	Inferior perplexity on large models: On LLaMA-2-70B, HyperPrune achieves worse Wikitext perplexity (5.13) compared to Wanda (5.21), and significantly worse than Pruner-Zero (4.87). This challenges the claimed scalability advantage and suggests that the method's benefits diminish or even reverse at larger scales.
2.	Insufficient analysis of design choices: The hypernetwork architecture (4-layer FFN with hidden size 256/512) appears arbitrary with no justification or ablation on architecture depth/width.

### Questions
1.	The proof of Theorem1 assumes linear activation, but transformers use SwiGLU (LLaMA). How does this affect the theoretical justification in practice? Can you provide empirical validation that the Gaussian assumptions hold for actual LLM activations?
2.	For hypernetwork, why is a 4-layer FFN chosen?  The hidden dimension is either 256 or 512. Is there a systematic way to choose this, or was it selected via trial and error?
3.	All experiments use LLaMA-2. How well does HyperPrune transfer to other model families?
4.	Section C introduces prior-based initialization but provides no empirical evaluation. How much does this improve convergence speed or final performance? Is the method sensitive to prior quality?

### Soundness
3

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
HyperPrune introduces a hypernetwork-based framework for n:m semi-structured pruning of LLMs. A shared, context-aware hypernetwork generates block-wise masks conditioned on layer embeddings, combined with continual-pruning and feature-outlier regularizers. The method provides an information-theoretic view linking reconstruction loss to mutual-information preservation, and achieves competitive performance on LLaMA-2 (7B/13B/70B) with up to 1.43× end-to-end speedup (2:4 sparsity) on A100 GPUs.

### Strengths
1. The idea of targeting n:m semi-structured sparsity with a shared, context-aware hypernetwork is interesting and novel to me.

2. Results across 7B/13B/70B with comparisons to magnitude, SparseGPT, Wanda, Pruner-Zero, and MaskPro; includes data-scaling ablation

3. An MI-based justification shows the reconstruction objective as a surrogate for information preservation under n:m constraints (Theorem 1)

### Weaknesses
1. All experiments are conducted exclusively on LLaMA-2, an outdated architecture. Given the architectural and activation differences in modern models (LLaMA-3, Qwen3, Gemma-3), it is unclear whether HyperPrune’s sparsity patterns or latency gains generalize to contemporary LLMs.

2. The appendix reports per-layer GEMM latency improvements (≈1.6×) against the dense baseline on A100 using CUTLASS kernels, but provides no end-to-end inference benchmarks on realistic engines (vLLM, TensorRT-LLM) or comparisons with other 2:4 pruning methods. Because Ampere tensor cores inherently yield ~1.5× speedups for any valid 2:4 pattern, these numbers mostly reflect hardware properties rather than advantages of HyperPrune’s mask design. Without full throughput measurements (tokens /s, batching, attention caching), the claimed 1.43× overall speedup remains a proxy rather than a demonstrated system-level gain.

3. Although the shared hypernetwork reduces parameter count, it brings typical drawbacks of hypernetwork-based methods—training instability, sensitivity to learning-rate and regularization hyperparameters, and potential overfitting to limited calibration data. Because the hypernetwork must generate masks for all layers jointly, cross-layer interference and catastrophic forgetting remain possible despite the continual-pruning regularizer. These issues make the approach potentially fragile and difficult to reproduce without careful hyperparameter tuning and large-scale validation.

### Questions
1. Provide GPU-hour and wall-clock costs per model (7B/13B/70B) for mask training; how does this compare to MaskPro under identical calibration sizes? MaskLLM is excluded for cost; while MaskPro is compared, a tighter apples-to-apples system study (same calibration sizes, wall-clock, memory footprint, failure cases) would clarify where HyperPrune truly wins.

### Soundness
3

### Presentation
2

### Contribution
3
