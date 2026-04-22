# PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Continual Instruction Tuning (CIT) enables Large Multimodal Models (LMMs) to rapidly adapt to new tasks without retraining, but it suffers from the catastrophic forgetting problem. By adding new branches, model extension provides a great idea to accommodate novel knowledge while causing huge memory consumption. To jointly address forgetting and memory explosion, we propose the Compression–Integration–Learning (CIL) pipeline, which draws on the memory consolidation processes during human sleep. Compression streamlines old parameters to release capacity. Integration merges knowledge from similar tasks to restore the performance loss due to compression. For example, based on LLaVA-7B, the forgetting is reduced from 11.29 to 5.09. Learning reallocates released capacity for new task-relevant parameters. Next, based on the characteristics of LMMs at different learning stages, we establish the progressive learning process, further reducing forgetting from 5.09 to 3.39. Moreover, to adapt this process, we decompose LoRA into a set of rank vectors and introduce an extremely fine-grained architecture, LoRA Rank Pool (LRP), with the goal of flexible knowledge employment and editing. Finally, we combine all components, and yield **P**rogressively **C**ompressed **L**o**R**A (PCLR). Extensive experiments demonstrate that PCLR owns a memory budget close to non-extension methods while outperforming extension methods in performance. The implementation code is available at https://github.com/SII-HITclearlove777/PCLR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses catastrophic forgetting and unbounded memory growth in Multimodal Continual Instruction Tuning (CIT) for Large Multimodal Models (LMMs) by proposing Progressively Compressed LoRA (PCLR). Through integrating the fine-grained LoRA Rank Pool (LRP) architecture and Compression–Integration–Learning (CIL) pipeline (inspired by human memory consolidation), the authors demonstrate that PCLR balances stability, plasticity, and memory efficiency: LRP decomposes LoRA into rank vectors with learnable keys for flexible knowledge reuse, while CIL prunes redundant experts, fuses knowledge via distillation, and trains new experts in released memory. Experiments demonstrate that PCLR outperforms state-of-the-art baseline methods in terms of average accuracy and forgetting rate, and its variant PCLR-LwF further enhances the performance of long-term CIT.

### Strengths
1. It innovatively proposes the LRP architecture, decomposing LoRA into fine-grained rank vector experts with shared components and dynamic gating, which enables flexible knowledge reuse and reduces redundancy, addressing the memory explosion issue of traditional extension methods.
2. The CIL pipeline, inspired by human memory consolidation, balances catastrophic forgetting and memory efficiency through compression-pruning redundant experts, integration-compensating performance loss via distillation, and learning-allocating capacity for new tasks, achieving a stable-plastic-memory balance.

### Weaknesses
1. The paper lacks sufficient analysis on why the progressive learning process adjusts the number of new ranks and compression retention rate in specific ways (e.g., 75% to 87.5%), with no clear theoretical or experimental justification for these hyperparameter choices.
2. It fails to compare PCLR with more multimodal continual instruction tuning methods published in 2024 and 2025, such as SEFE[1], HiDe-LLaVA[2], ProgLoRA[3].

[1] SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning, ICML 2025

[2] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model, ACL 2025

[3] Progressive LoRA for Multimodal Continual Instruction Tuning, ACL 2025

### Questions
1. In Section 4.4, Progressive Learning mentions that "as the number of tasks increases, fewer new ranks are allocated and a lighter, high-retention compression scheme is adopted", but the quantitative criteria for adjustment are not clearly defined. Is "a certain degree of task increase" based on a task quantity threshold or a model knowledge density indicator? What is the basis for reducing the rank from 64 to 32 in the Experimental Setup Section?
2. When comparing with the LoRA baseline—where LoRA uses a rank of 128, while PCLR adopts a "total rank of 256 and activated rank of 64"—the authors claim to "balance parameter efficiency," yet fail to align the "effective trainable parameter scale" between the two methods. Since PCLR involves fewer parameters actually participating in computation, a critical question arises: Does the "shared expert reusing historical knowledge" implicitly increase the effective parameter count? It is necessary to supplement direct comparisons of parameter scale and computational cost between the two methods.
3. In the experiments, all tasks were trained with 1 epoch, and the impact of the number of epochs on PCLR was not verified. If a new task is highly complex and requires multi-epoch training, should the compression-integration frequency of CIL be adjusted? Will multi-epoch training exacerbate the forgetting of old tasks, and will PCLR's anti-forgetting ability still be superior to that of baseline methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Compression–Integration–Learning (CIL) pipeline and a LoRA Rank Pool (LRP) architecture for multimodal continual instruction tuning. The method releases model capacity by compressing redundant rank vectors, integrates old knowledge through knowledge distillation, and incrementally learns new tasks, thereby mitigating catastrophic forgetting in multi-task continual learning. Experiments demonstrate that PCLR significantly reduces forgetting on LLaVA and Qwen-VL models, outperforming existing methods on two benchmarks.

### Strengths
- The performance is impressive, achieving state-of-the-art results on both datasets.
- The CIL process exhibits a high degree of innovation.

### Weaknesses
- The capacity limit of the expert pool may constrain the method’s effectiveness in ultra-long continual learning sequences. When experts from earlier tasks are excessively compressed, complete prevention of forgetting becomes difficult. Therefore, the method may not be well-suited for extremely long task sequences.
- The Integration stage requires additional training, which could lead to longer overall training time compared to existing methods.

### Questions
- What is the retention rate used during the Compression stage? What is the rationale behind this choice?
- For the first task, are all experts trained, and is compression applied only after the second task begins?

### Soundness
3

### Presentation
3

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
This paper proposes a method named PCLR for CIT of LMMs, introducing the LRP and a CIL pipeline, and demonstrates superior performance over existing approaches on the CoIN and Continual-NExT benchmarks. Despite the appealing experimental results, the work suffers from significant weaknesses in theoretical grounding and systematic efficiency analysis, which undermine its scientific rigor and practical deployment value.

### Strengths
1. The writing is fluent, the figures are clear, and the content is easy to understand.
2. PCLR introduces a progressive strategy that dynamically adjusts compression and learning strategies according to the learning stage, thereby optimizing long-term learning performance.

### Weaknesses
1. The theoretical foundation heavily relies on intuitive analogies and lacks formal justification. The paper extensively borrows the neuroscience concept of “memory consolidation during human sleep” and maps it onto the three-stage CIL pipeline. However, this analogy remains purely heuristic, with no accompanying verifiable computational model or theoretical guarantees.
2. Moreover, the paper lacks critical comparisons of efficiency and parameter counts, making it impossible to assess the true validity of its claimed memory advantages. Particularly given its emphasis on “solving memory explosion” the absence of parameter statistics and GPU memory usage curves constitutes a fatal flaw.

### Questions
1. Could you provide the theoretical basis for decomposing and obtaining the LoRA rank pool? The current method lacks relevant theoretical support. Although LoRA contains "linearly dependent redundant rank vectors," linear dependence does not equate to redundant knowledge: even if two rank vectors are mathematically linearly dependent, they may carry different information at the semantic or task representation level.
2. Why is the rank level the optimal granularity? Why not coarser or finer? It is recommended to include related ablation experiments.
3. Since this paper's method is based on AdaLoRA, should AdaLoRA be included as a baseline in the experiments?
4. In Table 2, the parameter count for LLaVA-1.5-hf is not reported—does this refer to LLaVA-1.5-7b-hf?

### Soundness
3

### Presentation
3

### Contribution
3
