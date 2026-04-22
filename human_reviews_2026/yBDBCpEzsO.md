# BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 6, 6

## Abstract
Binary quantization represents the most extreme form of large language model (LLM) compression, reducing weights to $\pm$1 for maximal memory and computational efficiency. While recent sparsity-aware binarization methods achieve sub-1-bit compression by pruning redundant binary weights, they suffer from three critical challenges: performance deterioration, computational complexity from sparse mask management, and limited hardware compatibility.  In this paper, we present BTC-LLM, a novel sub-1-bit LLM quantization framework that leverages weight transformation and binary pattern clustering to overcome these limitations, delivering both superior accuracy and efficiency. Our approach incorporates two key innovations: 
(1) a Flash and Accurate Binary Codebook that identifies recurring binary vector clusters, compressing them into compact indices with tailored distance metrics and sign-based centroid updates; (2) a Learnable Transformation that optimizes invertible scaling and rotation matrices to align binarized weights with full-precision distributions, enabling incoherence processing to enhance layer-wise representation quality.
This eliminates the need for sparse masks, enabling efficient inference on standard hardware. Extensive evaluations across LLaMA-1/2/3, Qwen-2.5/3, and FBI-LLM families demonstrate that BTC-LLM establishes a new state-of-the-art for extreme LLM compression at 1.11$\sim$0.7 bits. Notably, our BTC-LLM delivers strong performance under extreme compression settings, with just a 3.1\% accuracy drop on LLaMA-2-13B at 0.8 bits in zero-shot benchmarks while achieving a 1.6$\times$ speedup over FP16. Code is in the Appendix.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper tries to do sub-1-bit quantization.
However, it includes orthogonal proprocessing, which is not accounted for in the memory costs, and after accounting for storing orthogonal matrices, we would have the same storage costs as the original model.

### Strengths
Perplexity looks good.

### Weaknesses
The paper uses a matrix $R$ of size $n \times m$ for preprocessing before binarization (note that previous works used Hadamard matrices, which are much smaller).

I assume that each weight matrix $W$ has its own orthogonal preprocessing $R$ matrix. For simplicity, I will ignore diagonal factors.
After calculating $R$, we compute $R^TW$ and binarize that and get $B = ARB(R^TW)$ (Algorithm 1).
But for computation of the layer output we need to calculate $y = xRB$, thus we need to store $R$ somehow, but $R$ is as big as the original matrix and thus we would not have any reduction of storage costs.

### Questions
How is matrix R stored, and how much space does it take?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new method for sub-1-bit quantization, built on top of 1-bit quantization of the weights. Matrices of 1-bit quantized weights are approximated using a coodebook of K-centroids over sub-vectors of rows. To reduce outliers and improve quantization, the authors further propose to learn an invertible transformation for the weight matrices, with an auxiliary loss guiding toward an improved representation.

### Strengths
- The paper provides extensive details about the codebook assignment method.
- The proposed method improves scores across all tasks compared to STBLLM on LLaMA-1 and LLaMA-2.
- The proposed method achieved low perplexity across all models, even at very low bitrates.

### Weaknesses
- The major weakness is the lack of comparisons with existing methods on the more challenging settings. Application on downstream task is only superficially experimented in the main paper, with comparison only with STBLLM and FP16, on old models (LLaMA-1 and LLaMA-2), while table 6 shows clearly that LLaMA-3 is the most challenging compression setting. Both this lack of experimental settings and comparison with baselines severely hinders the results of this work.
- More effort is put into comparing perplexity, but this only acts as a proxy for actual model performance on tasks.
- It is unclear what is the novelty of the outlier elimination part compared to existing methods.
- The paper is hard to follow. Although the overall structure is well organized, each part is unclear, lack details and clarity. Some sentences are not complete, and the paper overall it lacks polishing.
- In the preliminaries, explanations are unclear, especially for the “Codebook Compression” part. This makes it difficult to understand later how the codebook quantization is applied. There is also no reference to existing literature in this section, which would help understand more easily this part.
- The paper is not clear on how the orthogonal transformation is merged into the weight matrix. Figure 4 seems to hint that some transformations are shared by multiple weight matrices, while this is not explicit in the paper, and there is no clear explanation on how the merging of the inverse matrix occurs, especially when combined to a block with shared orthogonal transformations.

### Questions
- Why introduce both $R$ and $D_{\pm}$, which are both orthogonal matrices? What does the second term adds to the expressivity?
- How does your model compare on downstream tasks compared to other quantization methods on LLama-3 or Qwen3?
- Could you clarify your contribution and method?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a system-oriented method that compresses LLM weights to sub-1-bit while remaining fast and deployable on hardware. It applies a binary codebook-based vector quantization. To reduce distribution mismatch between real-valued weights and the binary codebook space, the method incorporates a learnable transformation. It uses an EM-style training loop alternating E-step index updates and M-step codebook/transform updates.  The empirical evaluation is strong, reporting real memory and latency gains with solid ablations across bitwidths and components.

### Strengths
1. The performance gain is consistent across models and tasks. 
2. The paper not only shows perplexity gains, and reductions in memory and measured speedups on hardwares. 
3. The ablation studies are comprehensive.

### Weaknesses
1. Section 4.2 is hard to follow. Is this transformation entirely a new design, or does it draw on prior lines of work?

### Questions
1. Could the author compare the quantization and mean accuracy under different setup with other VQ-based baselines?
2. Could the author include AQLM as a baseline?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BTC-LLM, a sub-1-bit quantization method for LLMs that combines a binary codebook and a learnable transformation to improve both compression and accuracy. The approach avoids sparse masks and instead clusters binary weight patterns while using a scaling–rotation transform to align binarized weights with full-precision ones. Experiments on Llama and Qwen models show strong results with minimal accuracy loss and efficiency gains.

### Strengths
- The combination of a binary codebook and learnable transformation is smart. It tackles both redundancy and activation outliers which are two core issues in binary quantization.
- The paper provides detailed formulations, including the codebook optimization (binary K-Means style) and orthogonal transformation training (Cayley SGD). The authors explain computational trade-offs and hardware friendliness.
- The discussion on implementing the codebook via XNOR-POPCNT operations and shared-memory caching is practical. The demonstrated latency and memory reductions sound good.

### Weaknesses
Areas for improvement:

- Some sections (especially the methodology) are quite dense, with long formulae and mixed pseudo-code, which may hinder readability. Visual aids for algorithm flow could help.
- The authors briefly mention the lack of KV-cache compression and the overhead of the learnable transform only in the appendix. A more upfront treatment of these trade-offs would make the contribution sound more balanced.
- While the results are extensive, there’s limited reporting on real downstream application tasks (e.g., instruction following or reasoning benchmarks). Zero-shot QA tasks are useful but don’t fully stress model robustness under binary compression.
- Although speedups are claimed (1.6× over FP16), there is little granular profiling (e.g., GPU utilization, kernel fusion comparisons). More system-level measurements would strengthen the argument for hardware efficiency.

### Questions
- How much extra compute or memory does the learnable transformation add during quantization or inference? A short breakdown would help gauge practicality.
- Have you tried BTC-LLM on instruction-following or reasoning tasks like MMLU or GSM8K? It’d show how well it generalizes beyond zero-shot QA.
- You mention sharing a codebook across layers. how much accuracy does that save or cost? Would per-layer codebooks improve results?
- The 1.6× speedup is great, but could you share a bit more detail, e.g., GPU utilization or whether any custom kernels were used?
- How does BTC-LLM stack up against recent 2-bit or ternary methods in terms of both accuracy and energy efficiency?

### Soundness
3

### Presentation
2

### Contribution
3
