# Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2

## Abstract
Deep spiking neural networks (SNNs) hold immense promise for low-power event-driven computing, but their direct training via backpropagation through time (BPTT) incurs prohibitive memory cost, which limits their scalability. Existing memory-saving approaches, such as online learning, BPTT-to-BP, and reversible networks, compromise accuracy, training speed, or applicability. In this work, we propose a novel and broadly applicable pipeline for memory-efficient SNN training that preserves BPTT's accuracy. Our pipeline integrates layer-wise gradient checkpointing with lossless spike compression to eliminate internal state storage and reduce the memory cost of per-layer input spikes. We also introduce a multi-stage checkpoint adjustment strategy that adaptively refines checkpoint placement based on profiling results to further optimize memory usage and improve training speed. Wrapped in an optimization pass, the pipeline automatically restructures the computation flow before training with minimal user effort. Extensive experiments on diverse architectures and tasks demonstrate up to $8\times$ memory efficiency gains with $\le 20\%$ speed reduction and no accuracy loss. Our method provides a practical solution for efficient and scalable SNN training. Code is available at https://github.com/AllenYolk/snn-gradient-checkpointing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes to use gradient checkpoint (GC), possibly combined with compression, to reduce the peak memory required for training of SNN. The advantage of the proposed methods is that they do not introduce mathematical discrepancies (except for possible numerical ones), thus maintaining the training accuracy. The paper introduces a simple yet effective heuristic to minimize the peak memory by combining spatial and temporal GC segment partitioning. The paper is well written and easy to read.

Overall, the training performance penalty remains very limited (>0.9x on average) compared to the gain in peak memory (<0.4x on average), making the proposed method relevant and applicable to real world models training. However, the novelty and innovation of the paper remains limited: GC is a well-known technic, as well as compression.

### Strengths
- The paper is well written and easy to read;
- The peak memory reduction technic and heuristic proposed in the paper is relatively generic and applicable to many SNN models;
- The is a mathematical equivalency with the original model (despite possible numerical discrepancies);
- The impact on learning performance is very limited, compared to the gain on peak memory.

### Weaknesses
- The scope and impact of the paper remains limited;
- The novelty limited: GC optimization and compression are well-known technics;
- It is not clear how the proposed method will benefit the community: while some code snippets are provided in the supplementary materials, the possible release or diffusion of the source code is not mentioned.

### Questions
- Do the authors intent to release their method source code?
- It could be interesting for the authors to bring more insight about the different compression methods explored? In particular, by providing some figures on how they compare in terms of performances and peak memory?

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
This paper presents an automatic, lossless memory optimization pipeline for training spiking neural networks (SNNs) using backpropagation through time (BPTT). The core idea is to combine layer-wise gradient checkpointing with lossless spike compression, targeting the two major memory bottlenecks in SNN training—internal state storage and per-layer spike activations.

### Strengths
Originality
1. Introduces a lossless and automatic memory-saving approach for SNNs, avoiding the accuracy compromises typical in online learning or reversible architectures.
2. The combination of gradient checkpointing with binary spike compression is novel and elegantly leverages SNN characteristics (binary activations).

Quality
1. Provides comprehensive theoretical analysis (Equations 2–5) for memory cost and correctness.
2. Extensive empirical validation on multiple architectures and datasets.
3. Clear comparisons with other efficiency-oriented methods (online learning, BPTT-to-BP, reversible networks).

Clarity
Well-structured paper with good logical flow. Figures illustrate the differences between BPTT, checkpointing, and the proposed compression. The language is clear and technically rigorous.

Significance
1. Addresses one of the most critical barriers in SNN research: high memory cost during training.
2. Enables scaling SNNs to large architectures and long sequences, potentially democratizing access to neuromorphic research on commodity GPUs.

### Weaknesses
1. Limited exploration of trade-offs: Although the paper claims ≤20% slowdown, more fine-grained runtime profiling across levels would strengthen the argument for scalability. The additional computational cost of spike compression/decompression could be analyzed more quantitatively.
2. Sparse ablation: The adaptive checkpoint adjustment (spatial vs. temporal) is key but not deeply evaluated in isolation; Ablations showing how each component (spatial partitioning, temporal partitioning, greedy restoration) contributes to performance would enhance interpretability.

### Questions
1. Checkpoint adaptation: How sensitive is the memory efficiency to the chosen “level” parameter (O1–O4)? Could adaptive tuning be integrated dynamically during training? How does the system decide spatial vs. temporal split thresholds? Could these be learned or auto-tuned?
2. The paper reports ≤20% slowdown, but could the authors provide more detailed runtime decomposition? Does the time overhead scale linearly with the number of checkpoints or show nonlinear interactions with spatial/temporal splits?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Training spiking neural networks (SNNs) is highly memory-intensive, typically requiring O(LT) memory. This paper addresses the issue by taking advantage of the binary nature of spikes and adopting a checkpointing strategy. During training, the computational graph is not constructed; instead, the input data is passed through the network while intermediate activations, if they are spikes, are compressed to reduce memory usage. Once the loss at the final layer is computed, a local computational graph is reconstructed at each layer, starting from the final layer and proceeding backward to the first layer, using the stored intermediate representations as inputs. This strategy significantly reduces the memory footprint during training.

### Strengths
Paper is easy to follow. 

The proposed method can be easily adapted to train SNN models in a memory-efficient way.

### Weaknesses
I guess it's not very efficient if the SNN models are trained online, here it is assumed that the entire spiking data is available.

### Questions
1. Is it correct that the main source of memory efficiency arises from the fact that spike activations can be compressed, rather than being stored as 32-bit floating-point representations?

2. This method appears to rely on having access to the entire sequence of T input spikes during training. In an online setting, where input spikes arrive sequentially, how would the model behave? Could the authors comment on its applicability and performance in such scenarios?

3. The checkpointing strategy has also been successfully applied to train Neural ODEs [1]. Given the conceptual similarities between SNNs and NODEs from the ODE perspective, the authors may consider citing this related work for completeness.

4. Could the authors provide results obtained after training the models for the full number of epochs (e.g., on DVS-CIFAR10 or other datasets)? In the paper, only partial training results are presented to demonstrate the memory advantage. It would be valuable to also show that the accuracy remains unaffected when the model is trained to convergence.

[1] https://proceedings.mlr.press/v119/zhuang20a/zhuang20a.pdf

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates memory compression mechanisms in large language models (LLMs), focusing on achieving lossless or near-lossless memory efficiency. It systematically analyzes the relationship between attention redundancy, representation compression, and retrieval fidelity, and proposes a new compression architecture intended to retain semantic fidelity while reducing activation and KV cache memory.

### Strengths
The paper addresses an important issue in modern LLM systems — the scalability of context memory and cache compression. This is a crucial bottleneck for both open-source and proprietary systems. The authors provide a solid overview of various strategies (quantization, pruning, clustering, key-value cache compression, etc.), framing them under a unified memory-entropy perspective. This synthesis is useful for the community. The experiments, particularly those on retrieval fidelity and downstream reasoning, demonstrate that small amounts of information loss can lead to disproportionate accuracy degradation, supporting the motivation for “lossless” compression.

### Weaknesses
1. The “lossless” definition used in the paper lacks a rigorous mathematical formulation. The text refers vaguely to “semantic equivalence under compression,” but no formal mapping (e.g., bijection between original and compressed latent space preserving mutual information) is provided. This makes it hard to assess the conceptual contribution.
2. While the paper cites several existing compression methods (e.g., KV cache quantization, low-rank adaptation, memory token merging), it is unclear what aspect is fundamentally new. The proposed framework seems to integrate known techniques rather than introduce a distinctly novel algorithm.
3. The experiments appear to be conducted on small- to medium-scale models and limited benchmarks. It’s unclear how the proposed method scales to models beyond 7B parameters or generalizes to instruction-following and multimodal tasks. In addition, most results lack variance reporting and ablation analysis.
4. The metrics primarily evaluate perplexity and accuracy but not information preservation. A more appropriate evaluation would involve mutual information, reconstruction error, or similarity metrics on attention distributions.
5. The claim that “compression can improve reasoning stability by removing noise” is speculative and not well supported by experiments. Correlation is shown, but no causal analysis or controlled experiments are presented.
6. The paper repeatedly claims that “redundant activations lead to inefficiency” and “compression improves generalization,” but it only shows correlations between compression ratio and performance metrics (e.g., perplexity).
There is no controlled experiment isolating causal factors — for instance, whether observed gains arise from reduced redundancy, implicit regularization, or simple noise filtering. Without causal validation, these claims remain speculative.
7. Memory compression may cause unstable behavior, especially in autoregressive decoding where small perturbations can cascade.
The paper does not evaluate output stability under repeated sampling, sensitivity to compression noise, or robustness across different temperature settings. Without such analyses, it is unclear whether the approach is reliable for real-world use.

### Questions
1. How exactly is “lossless” defined in your experiments? Is it equivalent to maintaining identical outputs, or bounded semantic deviation?
2. Have you measured mutual information or entropy changes before and after compression?
3. How does your method compare to dynamic memory eviction or entropy-based key selection methods?
4. Does the compression framework support online adaptation — e.g., varying compression ratio during inference?
5. Are the proposed memory structures compatible with GPU-efficient implementations (e.g., FlashAttention or PagedAttention)?

### Soundness
1

### Presentation
2

### Contribution
1
