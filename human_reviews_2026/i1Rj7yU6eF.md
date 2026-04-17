# Merge before Forget: A Single LoRA Continual Learning via Continual Merging

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Parameter-efficient continual learning has emerged as a promising approach for large language models (LLMs) to mitigate catastrophic forgetting while enabling adaptation to new tasks. Current Low-Rank Adaptation (LoRA) continual learning techniques often retain and freeze previously learned LoRAs or generate data representations to overcome forgetting, typically utilizing these to support new LoRAs learn new tasks. However, these methods not only ignore growing computational memory with tasks and limited storage space but also suffer from potential task interference due to the lack of effective LoRA merging mechanisms. In this paper, we propose a novel continual learning method that orthogonally initializes and sequentially merges LoRAs updates into a single unified LoRA. Our method leverages orthogonal basis extraction from previously learned LoRA to initialize the learning of new tasks, further exploits the intrinsic asymmetry property of LoRA components by using a time-aware scaling mechanism to balance new and old knowledge during continual merging. Our approach maintains constant memory complexity with respect to the number of tasks, minimizes interference between past and new tasks via orthogonal basis initialization, and improves performance over asymmetric LoRA merging via adaptive scaling. We provide theoretical analysis to justify our design and conduct extensive experiments across diverse continual learning benchmarks using various LLMs, demonstrating the effectiveness and efficiency of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript addresses the linear growth in memory and computational cost in LoRA-based continual learning (CL), where prior methods store a separate adapter per task. The authors propose SLAO, a novel method that maintains only a single LoRA adapter, achieving constant memory complexity. SLAO frames CL as a sequential merging problem. For each new task, it uses an orthogonal basis from the prior adapter for initialization, fine-tunes, and then performs an asymmetric merge: it replaces the task-invariant A matrix and applies a time-aware weighted average to the task-specific B matrix. Experiments on several benchmarks and Llama models show SLAO outperforms other data-free, constant-memory CL baselines.

### Strengths
1. The core insight to treat LoRA's A and B matrices asymmetrically is clever, well-supported by empirical analysis (Fig. 1), and validated in ablations (Table 3), proving its critical role in the method's success.

2. The method is comprehensively evaluated against the relevant baselines (adapter-isolation, merging, etc.) and is clearly shown to be the superior data-free approach. The claims are further supported by thorough ablations for initialization, merging strategy.

### Weaknesses
1. The merge for the A matrix is a full replacement ($A_{merge}^{i} = A_{ft,i}$) by the newest task, which risks forgetting the optimal basis for older tasks. This "winner-takes-all" approach needs more discussion.

2. A significant performance gap remains between SLAO and data-rehearsal methods, especially on complex benchmarks. A brief discussion on hybridizing SLAO's constant-memory model with a constant-sized buffer would add context.

3. The theoretical analysis provides good motivation for the design, but the final algorithm isn't a strict derivation. For instance, the scaling factor $\lambda(i) = 1/\sqrt{i}$ is adopted from prior work, not derived from the paper's own analysis.

### Questions
Refer to Weaknesses for related questions.

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
3

### Summary
This paper introduces SLAO, a novel continual learning (CL) method for large language models (LLMs) that leverages Low-Rank Adaptation (LoRA) and continual merging to maintain a single shared LoRA across tasks. Unlike existing LoRA-based CL methods that retain multiple task-specific LoRAs or generate pseudo-data, SLAO sequentially merges new task updates into a unified LoRA via orthogonal initialization and time-aware scaling. The approach ensures constant memory usage, reduces task interference, and improves generalization. Theoretical analysis and extensive experiments on multiple benchmarks and model scales (e.g., Llama-2, Llama-3) demonstrate its effectiveness and efficiency.

### Strengths
+ Provides a formal analysis of forgetting and intransigence in the NTK regime and motivates the design via LoRA asymmetry.
+ The concept of continual merging into a single LoRA is novel and addresses key limitations of existing LoRA-based CL methods.

### Weaknesses
+ The method is designed for llm but evaluated on models that are not state-of-the-art, as well as tasks that are easy for current llms. Models such as qwen2.5/3 series and tasks such as aime, livecodebench or at the same difficulty level are needed. At lease, the reviewer think the tasks should be more diverse.
+ Could the orthogonal initialization strategy be combined with other PEFT methods?

### Questions
See the weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SLAO (Single LoRA Continual Learning with Orthogonal Initialization via Continual Merging), a framework that maintains a single shared LoRA across all tasks by sequentially merging new task updates, eliminating the need for task-specific LoRA storage.  Experiments on Llama-2-7B-chat, Llama-2-13B-chat, and Llama-3-3B demonstrate SLAO’s effectiveness.

### Strengths
1. SLAO is the first to enable CL with a single shared LoRA via sequential merging.  

2. SLAO is robust to hyperparameters (e.g., LoRA rank, learning rate) and model scales. In particular, performance improves with larger models.

### Weaknesses
1. While SLAO’s training overhead is low, QR decomposition for orthogonal basis extraction adds a one-time cost per task. The paper does not quantify this cost for long sequences (e.g., 50+ tasks) or analyze whether approximate orthogonal methods (e.g., randomized SVD) could reduce the cost without performance loss.

### Questions
1. How does SLAO perform when tasks have extreme similarity (e.g., multiple sentiment analysis tasks) or dissimilarity (e.g., sentiment vs. QA)? Does orthogonal initialization overly restrict knowledge transfer for similar tasks, and can the orthogonality constraint be relaxed dynamically?

### Soundness
3

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
This paper proposes SLAO, an algorithm that can avoid forgetting while maintaining a single shared LoRA across all tasks, avoiding the linear memory growth of storing multiple LoRAs. The authors provide NTK-based theoretical analysis to motivate orthogonality and asymmetry handling, and show consistent improvements over LoRA-based CL methods.

### Strengths
+ The motivation of the intrinsic asymmetry property of LoRA is clearly presented in Figure 1.
+ The paper provides theoretical analyses grounded in NTK theory.
+ The writing is generally clear and well-organized.

### Weaknesses
+ Unclear mechanism for avoiding forgetting. 1)While I can understand how InfLoRA prevents forgetting by projecting updates into the null space of old task features or directly using old task samples, I find it difficult to see how the proposed orthogonal initialization in this paper achieves the same effect. For the merging step, many prior works in multi-task learning compute task vectors as ΔW = B·A. In your formulation, however, A is replaced, and only B is merged. It remains unclear why merging B alone can effectively prevent forgetting. 2) Another concern is, the definition of task vector here is different from that in MTL, where the fine-tuned parameters subtract the same original weights instead of the merged ones. 3) It would be helpful to explore the effect of the scaling factor $\lambda(i)$.
+ The paper claims that the proposed method avoids linear memory growth by maintaining only the merged LoRA and the current task’s LoRA, without storing all task-specific adapters. However, in methods such as O-LoRA and InfLoRA, it is also possible to directly merge different task-specific LoRAs while achieving comparable performance. I would appreciate clarification on how Figure 2 was computed in this regard, and a more detailed explanation of how the proposed approach fundamentally differs from these existing merging strategies in terms of scalability.
+ While analytically clean, this setting does not fully capture nonlinear behavior in LLM fine-tuning. The link between the NTK bounds and empirical performance could be better validated with empirical NTK measurements.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
