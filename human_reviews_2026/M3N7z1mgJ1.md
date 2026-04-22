# NeuroSlice: Forward Selection-Based LLM Pruning via Neuron Contribution Decomposition

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Large language models (LLMs) have dramatically advanced natural language processing, but their deployment is often hindered by exorbitant computational and memory demands. LLM pruning offers a promising pathway to efficiency, yet most pruning methods rely on the layer output as the signal for parameter importance estimation. In this work, we revisit this issue and demonstrate that the layer output is not an atomic unit. Leveraging matrix identity transformations, we decompose each layer's output into an additive summation of individual neuron contributions, thereby reshaping the original token-by-feature tensor into a more granular token-by-feature-by-neuron representation. This decomposition yields much richer pruning signals by explicitly quantifying each neuron's individual contribution to the original layer output, enabling us to convert structured pruning as an neuron subset selection problem. To further optimize pruning ratio allocation, we introduce a layer-adaptive sparsity assignment method that dynamically allocate the global pruning ratio based on empirical reconstruction gains across layers. Our empirical analysis uncovers intriguing insights into depth-wise and module-wise redundancy patterns, offering actionable insights for future LLM pruning designs. Through comprehensive experiments on diverse LLM benchmarks, we show that our proposed NeuroSlice method consistently surpasses state-of-the-art structured pruning baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes NeuroSlice, a structured pruning method for LLMs that first decomposes each layer’s output into additive per-neuron contributions across both FFN and attention modules, then casts pruning as a neuron subset selection problem solved by forward selection. A ranked list of neurons is produced by iteratively adding the neuron most correlated with the current residual, with an efficient residual-update scheme based on a precomputed Gram matrix to avoid repeated passes. The method also allocates layer-wise pruning ratios by distributing neurons in proportion to their marginal information gains recorded during selection.

### Strengths
1. The paper introduces a decomposition that reframes layer outputs as sums of per-neuron contributions for both FFN and attention at finer granularity than heads, yielding a consistent pruning unit and a richer importance signal.

2. Marginal information gains recorded during selection drive cross-layer pruning ratios, acknowledging that some layers are more critical than others.

3. On multiple LLaMA variants, NeuroSlice matches or outperforms baselines on zero-shot accuracy while achieving substantially lower perplexity, especially at 20 percent and 50 percent sparsity.

### Weaknesses
1. The paper studies structured pruning but does not report end-to-end acceleration on GPUs at different sparsity levels. It would be important to show latency, throughput, and energy measurements on real hardware across a range of sparsity targets.
2. The method’s sensitivity to the choice of calibration dataset is underexplored. A systematic analysis across domains, sequence lengths, and calibration budgets would clarify how robust the neuron ranking is to distribution shift.
3. The novelty may be limited because the importance signal is derived from reconstruction of activations. The approach reads as a finer-grained activation-based criterion rather than a fundamentally new importance measure, so the conceptual advance should be articulated more clearly.

### Questions
1. The current evaluation only covers the LLaMA family. Please include additional model families such as Qwen and Mixtral to demonstrate robustness across architectures and training recipes.

2. The strategy for determining layer-wise sparsity lacks comparison to OWL (outlier-based) and similar methods, as well as deeper insights about where sparsity should be allocated. Additional analysis could reveal layer trends and help explain when and why certain layers are pruned more aggressively. 

3. The approach stores a score for every neuron. Please report the memory overhead of these scores, describe how it scales with model size and sequence length, and provide experiments on larger models (for example, beyond 30B parameters) to validate scalability in practice.

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
This paper introduces a neuron pruning method called NeuroSlice which involves neuron contribution decomposition (NeuCoDe), neuron subset selection, and flexible one shot pruning. Two importance metrics are introduced, neuron energy and neuron correlation. These metrics demonstrate improved selection criteria over baselines of magnitude selection and random selection. Neuron subset selection is performed via forward selection, which involves an efficient iterative selection of neurons that greedily minimizes the reconstruction error. This algorithm ranks all neurons and therefore can be used to achieve any sparsity target without additional computation.This method outperfoms other structured pruning approaches at comparable efficiency. Ablation studies and analysis validate the findings and highlight interesting trends.

### Strengths
The paper is very well written. The method development is easy to follow and experiments validate the main points. The method is simple, interesting, and effective. The results are strong and demonstrate the efficacy of the method.

### Weaknesses
Some minor typos. See questions.

### Questions
1. What are the comparable pruning times for other methods? 
2. Have neuron energy and neuron correlation been used in other pruning methods? Or are the authors the first to use these metrics?
3. How does the method perform on non-Llama models? I understand if these results cannot be obtained during the rebuttal period but it would be interesting to see results on a different model family.

### Soundness
4

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
3

### Summary
NeuroSlice is a structured pruning method that treats pruning as neuron subset selection. It introduces Neuron Contribution Decomposition to measure each neuron’s additive impact on layer outputs, then applies forward selection to choose neurons that best preserve reconstruction error and determine adaptive sparsity per layer. The approach achieves comparable or slightly better perplexity and zero-shot performance on LLaMA-7B/2-7B/3-8B models than structured pruning baselines like SliceGPT, FLAP, SlimLLM, and CFSP.

### Strengths
1. The neuron-level decomposition is clearly formulated, linking pruning to feature selection with a mathematically explicit view of neuron contribution.

2. The same contribution formulation applies to both FFN and attention modules, offering a consistent pruning unit across submodules.

### Weaknesses
1. While the paper presents a clean formulation, the main idea—ranking neurons via contribution correlation and selecting them greedily—is a direct adaptation of feature-selection and orthogonal matching pursuit methods. Similar per-channel or per-head selection ideas appear in earlier works like SliceGPT, DISP-LLM, and even SparseGPT’s layerwise reconstruction; NeuroSlice mostly adds finer-grained bookkeeping and a new interpretation layer.

2. The results report perplexity and zero-shot accuracy but omit any measurement of latency, throughput, memory, or FLOPs. Without showing real-world acceleration (e.g., on vLLM, TensorRT-LLM, or FlashAttention kernels), it is unclear whether the structured pruning translates into practical efficiency.

3. Forward selection, even with Gram-matrix caching, still requires multiple matrix–vector correlations per layer. The paper claims it’s “3–4× faster” than naive selection, but provides no absolute runtime or GPU budget. For multi-billion-parameter LLMs, this cost could still be prohibitive relative to one-shot pruning or magnitude baselines.

### Questions
1. How does the computational cost (wall-clock time, GPU hours) of NeuroSlice compare to SliceGPT or FLAP on the same models?

2. Do you observe real inference acceleration (tokens/sec, latency) after pruning on actual inference frameworks (e.g., vLLM)?

3. I would suggest adding more open-source models (now only LlaMA family) to enhance the empirical studies. Some great candidates include Qwen3, Gemma, GPT OSS, etc.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a neuron-level structured pruning framework for large language models. It reformulates layer outputs into per-neuron contributions and applies a forward selection strategy for pruning. Experiments on LLaMA models show modest but consistent improvements over existing baselines.

### Strengths
1. The paper addresses a relevant and timely topic in LLM efficiency, focusing on structured pruning with an intuitive neuron-level formulation. The proposed method is simple and implementation-friendly, requiring no fine-tuning.

2. Empirical results across multiple reasoning benchmarks demonstrate that the approach maintains stable accuracy under moderate sparsity.

### Weaknesses
1. The experiments are conducted only on the LLaMA family (7B–8B models), which are relatively small by current LLM standards. The lack of evaluation on larger or more diverse architectures limits the generality and credibility of the empirical claims.

2. The idea of neuron-level decomposition has been discussed in prior pruning works such as WANDA and related studies on structured sparsity. The proposed “neuron contribution decomposition” appears to be a reformulation rather than a fundamentally new perspective or mechanism.

3. The paper provides no illustrative figure or schematic to clarify the proposed framework. Without visual explanation or pseudocode, it is difficult for readers to follow the main algorithmic pipeline and intuition. Also, it is not standard to put results into appendix and discuss them in the main paper for the entire section 4.4.

4. Although the method shows slightly better accuracy and perplexity than some baselines, the gains are modest and may fall within expected experimental variance. The improvements do not convincingly demonstrate a clear advance over existing structured pruning techniques.

Part of review is revised with LLM assistance.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
