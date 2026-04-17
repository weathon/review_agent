# PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Mixture-of-Experts (MoE) models have shown strong potential in scaling language models efficiently by activating only a small subset of experts per input. However, their widespread deployment remains limited due to the high memory overhead associated with storing all expert parameters, particularly as the number of experts increases. To address this challenge, prior works have explored expert dropping and merging strategies, yet they often suffer from performance drop at high compression ratios. In this paper, we introduce \name, a training-free MoE compression method that achieves both high accuracy and efficient inference through two key innovations: First, \name performs sparse expert merging by identifying element-wise weight redundancy and specialization. It uses a dual-mask to capture both shared and expert-specific parameters. Second, to avoid the overhead of storing binary masks and signs, \name introduces a bit-packed encoding scheme that reuses underutilized exponent bits, enabling efficient MoE inference on GPUs. Extensive experiments demonstrate that \name can compress MoE models by up to 50\% while maintaining accuracy across various tasks. Specifically, it outperforms prior MoE compression methods by up to 16.7\% on MMLU at 50\% compression ratio, and achieves up to 1.28$\times$ inference speedup.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes **PuzzleMoE**, a training-free compression method for Mixture-of-Experts (MoE) LLMs that (i) merges experts element-wise using complementary similarity (magnitude-based) and saliency (activation-aware) masks to preserve both shared and expert-specific parameters, and (ii) achieves metadata-free inference by bit-packing per-expert masks/sign bits into under-utilized BF16 exponent bits with a custom CUDA decoding kernel. Across multiple MoE models and seven benchmarks, PuzzleMoE targets 25–50% expert reduction and reports competitive or better accuracy (e.g., improvements on MMLU at 50% compression) and up to ~1.2–1.3× inference speedups, with compression completed in minutes.

### Strengths
**1. Originality:** Element-wise, pairwise expert merging with dual masks is a fine-grained alternative to coarse expert dropping/averaging; **the bit-packed inference** path is a neat systems idea that removes separate metadata.

**2. Quality:** Broad zero-shot evaluation and ablations (e.g., mask thresholds, grouping strategies) demonstrate robustness at 25–50% reductions; compression is training-free and fast.

**3. Clarity:** The method and masks are clearly defined with equations/figures, and the pipeline (merge → pack → inference) is easy to follow.

**4. Significance:** Addresses a practical deployment bottleneck (expert memory and multi-GPU inference) with a simple, training-free path that shows favorable accuracy–efficiency trade-offs.

**5. Breadth of model coverage:** Experiments span **Mixture, DeepSeek-MoE, and Qwen-MoE** variants, giving evidence the approach generalizes across popular MoE families and sizes.

### Weaknesses
**1. Expert-pair selection metric at 25% compression is unspecified/under-motivated.**

For the 25% setting (“reduce experts to 75% of original”), PuzzleMoE’s pairwise scheme implies only a subset of experts are merged and the rest remain unmerged. The paper does not clearly define the criterion for choosing which experts to pair (random vs. similarity-aware vs. routing-aware). Without a transparent policy and sensitivity analysis, results at 25% may depend on pairing luck.

**2. Pairwise merging caps the unquantized/pruning-free compression at ~50%.**

Because each pair produces one merged expert, the theoretical floor with pairwise merging is half the experts (50% reduction) unless you move to higher-order grouping (3-to-1, 4-to-1) or combine with other techniques.

**3. Bit-budget and quantization interaction is underexplored.**

The packing scheme uses slack in BF16’s exponent to store per-pair ${\mathbf{mask}_i, \mathbf{mask}_j, \mathbf{sign}_i, \mathbf{sign}_j}$. With quantization, additional “freed bits” or altered formats (e.g., FP8/INTx + side scales) might change what can be packed, potentially enabling merging more experts per tensor (e.g., multi-expert packing) or richer masks.

**4. Speedup mechanism is not fully justified beyond single-GPU fit.**

The paper reports latency gains, but merging does not change activation parameters of MoE; if the baseline already fits on one GPU (or uses efficient tensor-parallel inference), why does PuzzleMoE deliver 1.2–1.3× speed up? Is it HBM traffic reduction (fewer expert weight loads), better cache locality, or kernel fusion/packing benefits?

### Questions
See weakness.

If these issues are addressed, I would be happy to raise my score.

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
5

### Summary
This paper proposes PuzzleMoE, a training-free compression framework for Mixture-of-Experts (MoE) models. It combines fine-grained sparse expert merging with a bit-packed inference mechanism that embeds binary masks into unused exponent bits of BFloat16 weights. A custom CUDA kernel enables on-the-fly decoding. Experiments on Mixtral, Qwen, and DeepSeek models show around 50% compression with minimal accuracy drop and small speedup (~1.2×).

### Strengths
- The proposed dual-mask merging mechanism is technically coherent and effectively preserves both shared and expert-specific information. 

- The work includes comprehensive experiments across several modern MoE architectures and diverse tasks, including reasoning benchmarks like GSM8K.

- The writing is organized and reproducible, with detailed algorithms, ablations, and implementation notes that enhance transparency.

### Weaknesses
- The proposed design mainly integrates ideas from prior expert merging (e.g., HC-SMoE, Sub-MoE) and bit-level quantization methods rather than introducing a fundamentally new approach.

- The reported inference acceleration (~1.2×) is relatively small given the 50% compression ratio; the main benefit appears to be memory reduction rather than compute efficiency. 

- Pairwise merging may not scale efficiently to larger expert counts, and the method’s complexity for >128 experts or hierarchical routing is not analyzed.

- Although GSM8K is included, the paper lacks results on more generation-intensive or long-context tasks, which would better demonstrate generalization.

### Questions
- How much real latency reduction is achieved in multi-GPU inference settings?

- Can the bit-packing approach be extended to FP8 or INT8 deployment?

- What is the time and memory cost of the pairwise merging stage itself?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a novel and intuitive expert merging-and-unmerging strategy that provides a structured approach to MoE compression. The core idea of dynamically composing experts from a smaller set of merged "puzzle pieces" is well-motivated by the observation of functional redundancy among experts (Fig. 1; Sec. 3.1; p.3). The experimental results are strong, showing that for Mixtral-8×7B, PuzzleMoE achieves 73.2% and 72.6% average accuracy at 25% and 50% sparsity, respectively, which is competitive with or superior to several baselines (Table 2; Sec. 4.2; p.7). However, the unmerging mechanism introduces additional complexity and parameters, and the scalability of the similarity computation to models with a very large number of experts is not fully explored.

### Strengths
- **Novel and intuitive merging/unmerging framework**
  - The "puzzle" analogy, where experts are assembled from shared pieces, provides a clear and compelling conceptual model for structured expert compression (Fig. 2; Sec. 3.2; p.4). This enhances the clarity and impact of the proposed method.
  - The framework allows for flexible, data-driven expert reconstruction during inference, which is more sophisticated than static merging or pruning techniques (Algorithm 1; Sec. 3.2.2; p.6).
  - The approach naturally preserves knowledge by forcing experts within a group to share a common representation, reducing redundancy while maintaining functional diversity through the unmerging coefficients.
- **Strong empirical results with comprehensive comparisons**
  - On Mixtral-8×7B, PuzzleMoE demonstrates strong performance. At 25% sparsity, it achieves a 73.2% average accuracy, outperforming baselines like Sub-MoE (72.1%) and HC-SMoE (70.2%). At 50% sparsity, it achieves 72.6% accuracy, significantly better than Sub-MoE (69.8%) and HC-SMoE (63.8%) (Table 2; Sec. 4.2; p.7). This shows the method's effectiveness.
  - The method shows consistent gains across various benchmarks, including commonsense reasoning (PIQA, SIQA, HellaSwag) and world knowledge (MMLU, TriviaQA), indicating its robustness (Table 2; Sec. 4.2; p.7).
  - Ablation studies confirm the effectiveness of the learnable unmerging mechanism compared to static or random unmerging, and validate the choice of similarity metrics for the merging process (Table 7; Sec. 5; p.9).
- **Efficient implementation with practical performance gains**
  - By reducing the number of active experts, PuzzleMoE directly translates to lower FLOPs and potentially faster inference, which is a key practical advantage for deploying large MoE models (Table 1; Sec. 4.1; p.6).
  - The paper provides a clear analysis of the trade-offs between the number of merged groups and performance, offering practical guidance for applying the method under different compression constraints (Fig. 4; Sec. 4.3; p.8).
  - The proposed method is compatible with existing MoE architectures and can be integrated with relatively minor modifications, facilitating its adoption (Sec. 3.2; p.4).

### Weaknesses
- **Complexity and overhead of the unmerging mechanism**
  - The learnable unmerging mechanism introduces additional parameters (the unmerging coefficients) and computational steps, which could offset some of the gains from expert merging. The overhead is not fully quantified in terms of memory and latency (Sec. 3.2.2; p.6).
  - The process of learning the unmerging coefficients requires a separate optimization step, which adds complexity to the training pipeline. The sensitivity to the hyperparameters of this learning process is not explored in detail.
  - It is unclear how the unmerging mechanism scales to models with a very large number of experts (e.g., >128), as the number of coefficients to learn could become substantial.
- **Scalability of the similarity computation**
  - The expert merging strategy relies on computing a similarity matrix between all pairs of experts within a layer. For models with a large number of experts, this O(N^2) computation could become a bottleneck (Sec. 3.2.1; p.5). The paper does not provide a complexity analysis or discuss potential optimizations for this step.
  - The choice of similarity metric (e.g., cosine similarity on weights) may not fully capture the functional relationships between experts. The paper does not explore more sophisticated or learned similarity metrics.
- **Limited evaluation on diverse model architectures and tasks**
  - The experiments are primarily focused on the Mixtral architecture. While comprehensive, evaluation on a wider range of MoE models (e.g., with different gating mechanisms or expert specializations) would strengthen the generalizability claims (Sec. 4; p.7-9).
  - The evaluation is centered on language modeling and commonsense reasoning. Demonstrating the method's effectiveness on other tasks, such as code generation or multilingual applications, would be beneficial.

### Questions
- **Quantify the overhead of the unmerging mechanism**
  - Provide a detailed analysis of the memory and computational overhead introduced by the learnable unmerging coefficients. Report the increase in parameter count and the extra latency incurred during inference for different numbers of merged groups.
  - Conduct a sensitivity analysis of the hyperparameters used for learning the unmerging coefficients and provide practical guidelines for their selection.
  - Discuss the scalability of the unmerging mechanism and propose potential simplifications or approximations for models with a very large number of experts.
- **Address the scalability of the similarity computation**
  - Provide a formal complexity analysis of the similarity computation step and measure its actual runtime for models with different numbers of experts. 
  - Explore and evaluate more efficient, approximate methods for clustering experts, such as locality-sensitive hashing (LSH) or other fast clustering algorithms, to mitigate the O(N^2) bottleneck.
  - Investigate the use of learned or task-aware similarity metrics to better capture the functional relationships between experts, potentially leading to more effective merging strategies.
- **Broaden the experimental evaluation**
  - Evaluate PuzzleMoE on a more diverse set of MoE architectures, including models with different gating mechanisms (e.g., noisy top-k) or expert designs, to demonstrate the method's robustness and generalizability.
  - Extend the evaluation to a wider range of tasks, such as code generation (HumanEval), mathematical reasoning (GSM8K), or multilingual benchmarks, to showcase the method's applicability beyond the tested domains.

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
4

### Summary
The paper proposes PuzzleMoE, a training-free method to compress Mixture-of-Experts (MoE) models by addressing their large memory footprint. The method relies on two core innovations. First, it introduces a fine-grained, pairwise sparse expert merging algorithm. This algorithm uses a dual-mask system to create a merged expert: (1) a similarity-based mask identifies and averages shared, redundant weights between two experts, and (2) an activation-based saliency mask preserves critical, specialized weights from the more salient expert. Second, to avoid the significant storage overhead of these fine-grained masks and sign bits, it introduces a bit-packing scheme. This scheme exploits the narrow distribution of Bfloat16 exponent values, freeing 3 bits to store the 4 bits of required metadata (two masks, two signs) within the existing 16-bit weight representation. Inference is performed by a custom CUDA kernel that decodes this information on the fly.

### Strengths
* The method is training-free and exceptionally fast. Compressing Mixtral-8x7B takes only 2 minutes, drastically outperforming the 90 minutes required for search-based NAEE or 55 minutes for SVD-based D2.
* The dual-mask merging strategy is highly effective, demonstrating state-of-the-art accuracy. At 50% compression, it shows minimal degradation, while prior methods suffer catastrophic accuracy drops. On Mixtral-8x7B MMLU, PuzzleMoE achieves 65.7%, whereas NAEE and HC-SMOE collapse to 47.3% and 49.0%, respectively.
* The bit-packing of metadata into underutilized Bfloat16 exponent bits is a clever solution to the overhead problem that typically makes fine-grained sparsity impractical. This co-design enables practical, efficient inference, delivering up to 1.28x speedup on Mixtral.
* The method appears robust, showing low sensitivity to the choice of calibration dataset (C4 vs. Math) and the expert grouping strategy (random vs. searched), which simplifies its practical deployment.

### Weaknesses
* The bit-packing scheme is the method's primary strength but also its critical weakness. It is rigidly tied to 2-to-1 pairwise merging. As the ablation in Table 6 confirms, merging 3+ experts is infeasible because the required metadata (5+ bits) exceeds the 4 bits available (1 sign + 3 freed exponent). This fundamentally limits PuzzleMoE to a fixed 50% expert compression ratio (or 75% at 25% sparsity), lacking the flexibility of other pruning methods.
* The paper's claim of compatibility with quantization is not well-supported by its own data. In Table 9, the proposed PuzzleMoE (50%) + 3-bit Quant (resulting in 3.35 avg bits) achieves lower MMLU accuracy (72.4) on Mixtral-8x7B than the simpler 3-bit AWQ baseline alone (73.0 accuracy at 3.25 avg bits). This suggests that at this compression level, standard quantization is more effective, and the sparse merging adds complexity for a net performance loss.
* The comparison to Wanda (2:4) is inappropriate. PuzzleMoE implements fine-grained, unstructured sparsity that requires a custom kernel, whereas Wanda (2:4) is a structured sparsity pattern designed for native hardware acceleration. A 50% unstructured magnitude or Wanda pruning baseline is conspicuously missing. Without this comparison, it is impossible to know if the complex dual-mask merging algorithm is actually superior to simpler fine-grained pruning criteria when run with a similar custom kernel.
* The method relies on a saliency metric ($|W| \cdot ||X||_2$) directly borrowed from Wanda19. The ablation in Table 10 only compares this against an even simpler magnitude-only metric20. It fails to investigate any MoE-specific saliency signals, such as router gating scores, which might be more effective at identifying and preserving expert specialization.

### Questions
1) The proposed bit-packing scheme, which frees 3 bits by shifting the 8-bit exponent, is a key innovation. However, this seems to rigidly tie the method to a 2-to-1 merge (requiring 4 metadata bits, packed into the 1 sign + 3 freed bits). How can PuzzleMoE be adapted to other compression ratios (e.g., 3-to-1 for 67% sparsity) without abandoning this core bit-packing efficiency?
2) Table 9 indicates that for Mixtral-8x7B, 3-bit AWQ (a baseline) achieves 73.0 MMLU, while PuzzleMoE + 3-bit (the proposed method) achieves 72.4 MMLU. This suggests that at this low precision, the proposed sparse merging is actually detrimental. Could you clarify why this performance drop occurs and justify the combined approach?
3) The paper's baseline for 50% sparsity is 2:4 structured Wanda4. A more direct comparison would be 50% unstructured magnitude pruning (using the Wanda metric $|W| \cdot ||X||_2$) implemented with the same custom bit-decoding CUDA kernel. Was this baseline tested? It is essential for isolating the gains of the dual-mask merging algorithm from the gains of the custom kernel itself.
4) The saliency mask uses an activation-based metric from Wanda. Did the authors experiment with MoE-specific signals, such as the router's gating frequencies or weights, to define expert specialization? This seems like a more direct method for identifying critical, expert-specific parameters than a generic saliency score.

### Soundness
3

### Presentation
3

### Contribution
2
