# UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6

## Abstract
Deploying large language models (LLMs) on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by on the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework, with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to cater to diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting that speeds up the computation by 20×, quantization-aware singular value decomposition (SVD) decompositions to minimize the quantization errors, state-aware weight sorting for SSMs, and a fused rotary embedding (RoPE) kernel for the pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a one-shot fashion, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models offer a memory reduction of 4×–5.7× and a token throughput improvement of 2.7×–3.4×, maintaining accuracy within 5% of the original models at 15% pruning rates across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models will be released at: https://github.com/enyac-group/UniQL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets dynamic, resource-constrained edge deployment of LLMs. It proposes a one-shot, post-training pipeline that produces a single INT4 model which can be adaptively pruned on-device at different rates depending on runtime constraints. Overall, the paper is a well-executed systems integration that addresses a timely deployment problem and demonstrates tangible engineering wins. However, for ICLR, the lack of clear algorithmic novelty and the absence of a rigorous case for “why combine pruning with quantization instead of simply going lower-bit” (especially W3) are significant gaps. Clarifying the scope of Hadamard/rotation, aligning it with pruning, and adding stronger/fairer baselines and ablations would strengthen the submission.

### Strengths
1. Practicality and scope: A unified, one-shot post-training pipeline that supports Transformers, SSMs, and hybrids, and enables on-device adaptive pruning without retraining is very relevant for edge deployment.
2. Solid engineering: the fused RoPE kernel, quantization-aware SVD, and avoidance of pseudo-inverses show careful engineering; the export path quantizes even embeddings/LM head to 4-bit, reducing footprint beyond common W4A16 libraries.

### Weaknesses
1. Most core techniques (importance-driven structured pruning, Hadamard/rotation-based quantization smoothing, W4 PTQ) are known. The main novelty is system integration and some engineering refinements (QSVD, fused RoPE, SSM-aware sorting). This level of originality feels marginal for ICLR unless paired with stronger conceptual/theoretical advances or broader evidence.

2. No justification for “quantization + pruning” vs “lower-bit quantization alone”: The paper does not compare W4 + k% pruning to a lower-bit alternative (e.g., W3) under matched memory/latency. Prior results (e.g., QuaRot with 3-bit) suggest that W3 may match or exceed the accuracy of W4 + 25% pruning at similar or lower budgets. Without this key comparison, the necessity of combining pruning with quantization remains unproven.

3. Weak PTQ baselines : PTQ comparisons are largely against framework built-ins (TRT-AWQ, TAO-HQQ) and a basic GPTQ variant; missing stronger recent baselines such as AWQ/QServe, Quarot/SpinQuant, etc.

4. The scope of rotations is not clearly specified: If rotations are applied on input channel axes of layers whose channels will later be pruned (e.g., O_proj or MLP down-proj input channels), this mismatch can harm pruning efficacy. There is no ablation on restricting rotations to non-pruned axes or aligning pruning boundaries with quantization/rotation groups.

### Questions
1. Please provide matched-budget comparisons of W4 + {25}% pruning versus W3 (with Hadamard/rotation, e.g., QuaRot-style) on the same models, reporting accuracy–memory Pareto curves. Under what budgets does “W4 + pruning” strictly dominate “W3 alone”?

2. Which layers/axes are rotated (Q/K/V/O proj in attention; MLP up/gate/down; SSM B/C/Z/X/O)? Are rotations applied to channel dimensions that will be pruned on-device?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes UniQL, a unified post-training quantization and low-rank compression framework designed to support adaptive deployment of large language models on resource-constrained edge devices. UniQL achieves integrated compression for Transformer, SSM, and hybrid models through structured weight ranking, quantization-aware SVD decomposition, state-aware SSM compression, and the integration of RoPE kernels. The framework performs one-time compression in the cloud and supports dynamic on-device pruning (up to 35%) based on current load. Experiments demonstrate that UniQL achieves significant memory reduction and inference acceleration across multiple models and hardware platforms, with manageable accuracy loss.

### Strengths
Wide Model Architecture Coverage: This approach systematically supports post-training quantization and structured pruning for Transformer, SSM, and hybrid models for the first time.

Strong On-Device Adaptability: This approach dynamically adjusts model size based on memory and compute resources after deployment to adapt to the dynamic load of edge devices.

High Compression Efficiency: This approach significantly accelerates the compression process (up to 22x faster than MoDeGPT) by avoiding pseudo-inverses and optimizing SVD decomposition.

System-Level Optimization: This approach integrates RoPE kernels and quantization-aware decomposition to significantly improve inference speed and reduce quantization error.

Extensive Experimentation: This approach demonstrates its effectiveness and versatility across a variety of models (e.g., Llama, Qwen, Mamba) and hardware (e.g., A6000, Nano 8G).

### Weaknesses
Accuracy-compression tradeoff not yet optimal: At high pruning rates (e.g., 35%), the accuracy of some models (e.g., Mamba2) drops significantly (down to 57.7%).

Sensitive to calibration data: Structured ordering and quantization depend on the calibration set; their robustness to different data distributions is not analyzed.

Lack of comparison with unstructured methods: No comparison with popular unstructured pruning methods (e.g., SparseGPT) or hybrid sparse methods is provided.

Device-side pruning overhead not quantified: While device-side pruning is supported, its runtime overhead (e.g., memory rearrangement, index lookup) is not analyzed.

Limited interpretability: No visualization or interpretable analysis is provided for the "state-aware" or "quantization-aware" mechanisms.

### Questions
Calibration Data Sensitivity:
How sensitive is UniQL to the choice of calibration data? Have you tested its robustness across domains (e.g., code, math, dialogue) or with out-of-distribution samples?

Comparison with Unstructured Pruning:
How does UniQL compare with unstructured pruning methods like SparseGPT, especially in terms of accuracy-efficiency trade-offs and hardware friendliness?

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
4

### Summary
This paper proposes a unified quantization and on-device structural pruning method for edge LLMs, including Transformers, State Space Models (SSMs), and hybrid models. To enable the on-device structural pruning, weight sorting is designed for different model blocks, and a LoRA-based recovery fine-tuning (FT) is conducted on the sorted model. 4-bit quantization and pruning with different rates are applied to the model. SVD decomposition is used to reduce quantization error. The evaluation presents different pruning rates that offer a memory reduction of 4x–5.7×.

### Strengths
1. The adaptive LLM memory problem discussed in this paper is important and interesting.
2. The proposed methods are evaluated on different model structures.

### Weaknesses
1. The contribution is limited, and the proposed methods are very incremental.
a) Methods applied to different model structures appear more as a systematic engineering effort than a novel algorithmic advancement.
b) The quantization, pruning combination has been explored.

2. Insufficient Empirical Evaluation.
a) The paper claims adaptive deployment of LLMs on the edge, but there is no real deployment with different workloads. Crucially, the paper does not address the core systems challenge: how the memory footprint can be dynamically managed at runtime without first loading the entire unpruned model.
b) Unignored performance drop. The reported performance drop is substantial, especially at lower pruning rates (e.g., a 6% drop for Llama2-7B at only 35% pruning).
c) Lack of comparison with related works: Quantization or pruning methods for SSM and hybrid models. Structure pruning methods, such as SliceGPT [1] .

3. The paper writing should be improved.
a) Explain why you do it before the detailed introduction. For example, the reason for weight sorting should be given in the introduction. 
b) Too many mathematical symbols make the methods part hard to follow. The algorithms in the Appendix are better for understanding.
c) All of the components should be included in the overview Figure.

[1] Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, James Hensman. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. ICLR 2024

### Questions
1. The finetune is applied after the weight sorter. Why does it apply after quantization? Have you experimented with different orders of the applied methods?
2. The pruning method is similar to SliceGPT [1]. What’s the key difference between SliceGPT and the proposed sorting method?

[1] Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, James Hensman. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. ICLR 2024

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces UniQL, a unified post-training compression framework designed to efficiently deploy large language models (LLMs) on edge devices. It integrates structured pruning and quantization in a single-shot pipeline that supports adaptive on-device compression based on real-time resource availability.

### Strengths
- Unified Framework: UniQL supports Transformers, State Space Models (SSMs), and hybrid architectures, addressing a wide range of LLM structures.

- On-device Adaptive Pruning: Enables users to prune the model at inference time based on the current device memory state.

### Weaknesses
- While the results are generally strong, some inconsistencies exist in the evaluation setup:
In Tables 1 and 2, the latency results for different models and methods are evaluated on different hardware platforms (Llama-3.1-8B and Nemotron-H-8B on A6000; Qwen-2.5-7B and Mamba2-8B on Nano 8G). Additionally, Table 2 lacks baseline comparisons such as TRT-AWQ for some models. This raises concerns about the consistency and comparability of latency evaluations across models and methods. Can the authors clarify why all models and methods are not evaluated uniformly across both platforms, and whether such comparisons are fair and meaningful under these mixed settings?

- UniQL is compared to SVD-LLM both with and without fine-tuning, which is helpful. However, the comparison with MoDeGPT is conducted only without fine-tuning, despite UniQL including fine-tuning in its best-performing configuration. Furthermore, all comparisons are conducted at only one sparsity level (15%), which limits the ability to assess how robust each method is across different compression regimes (e.g., 25%, 35%).

- In Table 7, UniQL is evaluated under single-pass adaptive pruning across multiple pruning rates and compared only with SVD-LLM. However, MoDeGPT, another key baseline used in Table 5 and throughout the paper, is not included.

### Questions
- With masked fine-tuning (FT), UniQL remains faster (6h 59 m) than both MoDeGPT (7h 03 m) and SVD-LLM (15h 57 m). I do not understand why masked fine-tuning would be faster — could the authors clarify the reason behind this behavior?

I am open to discussing this further during the rebuttal and will be happy to increase my score if my concerns are addressed.

### Soundness
3

### Presentation
2

### Contribution
4
