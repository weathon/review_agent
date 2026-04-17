# AutoHete: An Automatic and Efficient Heterogeneous Training System for LLMs

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Transformer-based large language models (LLMs) have demonstrated exceptional capabilities in sequence modeling and text generation, with improvements scaling proportionally with model size. However, the limitations of GPU memory have restricted LLM training accessibility for many researchers. Existing heterogeneous training methods significantly expand the scale of trainable models but introduce substantial communication overheads and CPU workloads. In this work, we propose AutoHete, an automatic and efficient heterogeneous training system compatible with both single-GPU and multi-GPU environments.  AutoHete dynamically adjusts activation checkpointing, parameter offloading, and optimizer offloading based on the specific hardware configuration and LLM training needs.
Additionally, we design a priority-based scheduling mechanism that maximizes the overlap between operations across training iterations, enhancing throughput. Compared to state-of-the-art heterogeneous training systems, AutoHete delivers a 1.32x~1.91x throughput improvement across various model sizes and training configurations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes AutoHete, a framework that optimizes LLM training and reduces GPU memory pressure using activation checkpointing, parameter offloading, and optimizer offloading. The problem is formulated as an integer linear program (ILP) to decide which layers should apply each of the three techniques. AutoHete comprises three components: a computation/memory profiler, an ILP solver, and a priority-based scheduler. The profiler first captures the computation graph and infers each node’s output shape using a fake input tensor. The ILP objective is built from empirical insights and a cost model that accounts for overlap between computation and communication. The priority-based scheduler favors earlier blocks for gradient offloading and CPU-side optimizer updates to reduce idle periods. With these systematic optimizations, AutoHete improves throughput by up to 1.91× on a single GPU and up to 1.84× on multiple GPUs compared to PatrickStar.

### Strengths
- This work targets an important, commonly encountered problem: GPU memory pressure when training large language models.

- It formulates the configuration of activation checkpointing, parameter offloading, and optimizer offloading as an integer linear program and finds solutions within a tractable search space.

- It demonstrates strong throughput improvements in both single-GPU and multi-GPU settings, and includes multi-node experiments showing effectiveness.

- The pipeline diagram and figures are clear and helpful for understanding the proposed method.

- The proposed method does not rely entirely on heuristics and is based on a derived cost model.

### Weaknesses
- The paper does not consider other parallelization techniques—such as tensor parallelism and pipeline parallelism—that also reduce GPU memory pressure. Experiments combining the proposed method with these techniques are needed to demonstrate effectiveness.

- The reported insights are rather trivial; developing them into formal theory with proofs would strengthen the claims.

- Numerous system-level details hurt readability. I am also unsure this paper fits ICLR well; it may be a better fit for a systems venue that values low-level optimization details.

- The flow and section structure are unclear. For example, I don’t see why Section 3.2 is titled “Solver” when it derives the cost model and formulates the problem as an integer linear program.

- The experimental setup for multi-GPU runs is unclear; the parallelism strategy used is not specified.

- The evaluation covers only a GPT-like architecture. Including recent strong models (e.g., Qwen, DeepSeek) or different architectures (e.g., MoE) would better demonstrate generalizability.

- The work relies on many fine-grained system optimizations, raising reproducibility concerns unless the authors open-source the implementation.

### Questions
- Parallelization coverage: Can you evaluate the proposed method in combination with tensor parallelism and pipeline parallelism to demonstrate effectiveness under common mixed-parallelism setups?

- Theoretical grounding: Can you formalize the key insights (e.g., as theorems/lemmas) and provide proofs to strengthen the claims?

- Venue fit & readability: Which machine-learning contributions make this work suitable for ICLR? Can you streamline low-level systems details or move them to an appendix to improve readability?

- Organization clarity: Why is Section 3.2 titled “Solver” if it derives the cost model and formulates the ILP? Would you clarify or restructure/retitle this section?

- Multi-GPU setup: What parallelism strategy was used in multi-GPU experiments (e.g., DP/TP/PP, ZeRO stage)? Please specify all relevant settings.

- Model coverage: Can you add results on recent strong models (e.g., Qwen, DeepSeek) and different architectures (e.g., MoE) to demonstrate generalizability?

- Reproducibility: Will you release code, configs, and scripts (including system-level optimizations) to ensure reproducibility?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work aims to make large language model (LLM) more accessible with limited GPU memory. Particularly, it introduces a training system named AutoHete, whose goal is to make use of both GPU and CPU memory by automatically deriving the configurations of three memory-saving techniques (activation checkpointing, parameter offloading, and optimizer offloading). AutoHete formulates the searching for the three techniques as an integer linear programing (ILP) problem. Additionally, it designs a priority-based scheduling mechanism that prioritizes the (off)loading of earlies layers so that the training of the next iteration would not be blocked. Evaluations show that AutoHete achieves substantial throughput improvement compared to ZeRO-Offload, PatrickStar, and StrongHold.

### Strengths
S1) A holistic system that jointly optimizes three memory-saving techniques (activation checkpointing, parameter offloading, and optimizer offloading) is developed.

S2) The priority-based scheduling to avoid blocking the next iteration’s training is a good solution.

S3) The empirical results are positive.

### Weaknesses
W1) The novelty is overall limited as all these memory-saving techniques have been investigated for long. 

W2) The ILP formulation requires the model to have identical layers. When this is not met, for example, multi-modal models or hybrid (full/sparse/linear) attention models, the proposed method may not apply.

W3) The baselines are outdated (in years 2021-2022).

### Questions
Q1) Can AutoHete outperform mainstream LLM training frameworks, such as newly updated versions of Megatron and TorchTitan.

Q2) Please consider conducting a detailed ablation study to assess the impact of each memory-saving techniques. I believe this can help readers understand why they should be combined together. 

Q3) I would also suggest doing a few case studies to show how each memory-saving technique is employed and how they are integrated.

Q4) Can AutoHete be applied to models with non-identical layers?

### Soundness
3

### Presentation
3

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
The paper introduces AutoHete, a heterogeneous training system that dynamically integrates activation checkpointing, parameter offloading, and optimizer offloading to enhance training efficiency. The system employs a priority-based scheduling mechanism to maximize operation overlap across training iterations.

### Strengths
1. The paper is well-explained and easy to follow.
2. AutoHete’s performance is thoroughly evaluated across various model sizes and hardware configurations, demonstrating significant throughput improvements.

### Weaknesses
1. The paper does not provide sufficient validation of the accuracy of the performance model used to predict GPU memory consumption and execution time.
2. There is a lack of ablation studies to verify the effectiveness of individual components in AutoHete (e.g., the contribution of the priority-based scheduling to overall performance).
3. The framework diagrams—particularly the overview of AutoHete—are poorly designed and may hinder comprehension; they should be revised for clarity.
4. While I highly appreciate the paper’s contribution from an efficiency standpoint, its novelty appears limited, as it largely combines existing techniques with engineering-level optimizations.

### Questions
see weakness above.

### Soundness
3

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
AutoHete formulates 3 memory saving technics: activation checkpointing, parameter offloading, and optimizer offloading. It constructs a cost model that accurately captures GPU peak memory usage and execution time. It proposed ILP to search optimal decisions for which layers to checkpoint, offload, or retain on GPU. AutoHete includes a profiler that extracts operation-level compute and memory characteristics using torch.fx. AutoHete achieves 1.32×–1.91× higher throughput than existing state-of-the-art heterogeneous training systems such as ZeRO-Offload, PatrickStar, and StrongHold

### Strengths
fair originality
The paper presents a novel integration of activation checkpointing, parameter offloading, and optimizer offloading into a single unified optimization framework. Each of the technics are well understood, but how to schedule them to find the best trade off remains challenging.  formulation into an ILP problem is principled approach. 

good quality
The cost model can model memory/computation holistically, and transform it a well-defined optimization formulation. AutoHete is fully implemnted with requirements for model code changes. Experimental evaluation covered multiple model scales, batch sizes, GPU counts, and memory budgets. I appreicate the ablation studies

good clarify
system overview and scheduling examples are extremely helpful to illustrate operation overlap and memory allocation. The writing is effectively communicating both high-level ideas and low-level technical details, like scheduling dependencies and cost modeling.

Fair significance
AutoHete addresses the peak GPU memory contraints make training easier for researchers with limited hardware. The automatic optimization framework can speed up iteration and avoid GPU OOMs according to the hareware spec. The work represents a meaningful step toward democratizing LLM training

### Weaknesses
Need more originality
The three core mechanisms (activation checkpointing, parameter offloading, and optimizer offloading) have been extensively studied in prior works such as ZeRO-Offload, PatrickStar, and StrongHold. AutoHete primarily combines these methods rather than introducing a fundamentally new training technique.

The ILP-based optimization formulation, though systematic, is a straightforward formalization of the decision process (which layers to offload or checkpoint). Given the small search space (three integer variables), this formulation is conceptually simple and unlikely to be viewed as a theoretical innovation.

The priority-based scheduling strategy, while effective in practice, builds on standard ideas of operation overlap and stream prioritization used in heterogeneous training and distributed systems. Its novelty lies in application rather than mechanism.

### Questions
How does AutoHete’s integration of checkpointing and offloading differ fundamentally from earlier hybrid memory-management approaches? Could you argue that this integration enables qualitatively new optimization behaviors?

### Soundness
3

### Presentation
3

### Contribution
2
