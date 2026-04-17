# ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Incremental Learning (IL) for Open-ended Image-to-Text Generation (OpenITG) enables models to continuously generate accurate, contextually relevant text for new images while preserving previously acquired knowledge. Unlike prior studies, this paper addresses a more practical scenario in which the predominant category of visual data shifts over time as environments evolve. In this context, we introduce a new notion of continual alignment, which incrementally adapts the alignment module within pre-trained VLMs to preserve high-quality cross-modal representations. Based on this idea, we propose **E**fficient **C**ontinual **A**lignment (ECA), a novel exemplar-free IL approach for OpenITG. The key challenge is enabling the model to acquire new, task-specific features while minimizing interference with the established alignment without accessing raw data from previous tasks. To address this, ECA employs three core mechanisms: a **M**ixture **o**f **Q**uery (MoQ) module that adapts task-specific query tokens, a **F**ish**e**r **D**ynamic **Ex**pansion (FeDEx) that dynamically expands model structure based on a Fisher Information Matrix (FIM)-based metric, and an embedding dictionary with **D**ictionary **R**eplay (DR) to retain past knowledge. To evaluate ECA's performance, we construct four new IL OpenITG benchmarks that better reflect real-world scenarios. Experimental results demonstrate that ECA significantly mitigates catastrophic forgetting and improves IL performance compared to baseline methods. Benchmarks are available at <https://anonymous.4open.science/r/ECA-ToS-Benchmarks-FB17>.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of Incremental Learning (IL) for Open-ended Image-to-Text Generation (OpenITG), such as VQA and captioning. 

Experiments on the four new benchmarks show that ECA significantly outperforms strong baselines (including regularization and prompt-based IL methods) and achieves performance very close to the joint-training upper bound, demonstrating its ability to mitigate catastrophic forgetting while remaining parameter-efficient.

### Strengths
1. The "main topic" shift scenario, which incorporates semantic overlap, is a significant and more realistic formulation for IL benchmarks compared to standard disjoint-category setups. The four new ToS benchmarks are a strong contribution.
2. The FIM-based metric for deciding when to expand with a new parallel adapter is the paper's strongest technical novelty. It is theoretically motivated (Theorem 1) and empirically validated (Fig. 4), providing a non-heuristic way to balance positive transfer (reusing adapters) and mitigating interference (adding new adapters).
3. The combination of MoQ and DR (Dictionary Replay) is a highly effective *exemplar-free* approach. The DR mechanism, using a learned sparse dictionary of embeddings, is a clever way to perform rehearsal without storing raw data, addressing privacy and storage concerns.

### Weaknesses
1. The final ECA method combines three distinct modules (MoQ, FeDEx, DR), each with its own logic and (minor) hyper-parameters (e.g., $\lambda$ for $\mathcal{L}_{DR}$, dictionary size $m$). This is inherently more complex than a simpler baseline like a single PA or a prompt-based method.
2. The paper does not explicitly quantify the computational cost of the FeDEx module. Calculating the FIM-based metric $S(\omega_t)$ requires additional gradient and FIM diagonal computations at the end of each task to decide whether to expand. While likely manageable (as it's not per-batch), a brief analysis of this overhead would be beneficial.
3. The method is instantiated on BLIP-2's Q-Former. While the *principles* (adapting the alignment module) are general, it's not immediately clear how MoQ or FeDEx would be applied to different VLM architectures, such as projector-based models (e.g., LLaVA), which lack a Q-Former. A brief discussion on this potential for generalization would improve the paper.

### Questions
1. Could you please quantify the computational overhead of the FeDEx module? Specifically, what is the cost of calculating the FIM-based metric $S(\omega_t)$ at the end of each task (relative to the task's training time)?
2. The Dictionary Replay (DR) module relies on a dictionary of a fixed size $m=5 \times d_v$. How does this fixed-size dictionary scale as the number of tasks $T$ grows very large? Do you foresee this becoming a bottleneck, and would a dynamic-sized dictionary (e.g., adding new atoms per task) be beneficial?
3. Could you elaborate on how the core ideas of ECA, particularly FeDEx and MoQ, could be adapted to other popular VLM architectures that do not use a Q-Former, such as those with simple MLP projectors?

### Soundness
4

### Presentation
4

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
This work proposes Efficient Continual Alignment (ECA), an exemplar-free incremental learning framework for open-ended image-to-text generation (OpenITG) that adapts pre-trained vision-language models to evolving visual domains. ECA introduces continual alignment, ensuring cross-modal consistency while learning new tasks without accessing prior data. It achieves this through three key components: a Mixture of Query (MoQ) module for task-specific query adaptation, a Fisher Dynamic Expansion (FeDEx) mechanism that expands model capacity using FIM-based metrics, and a Dictionary Replay (DR) strategy to preserve past knowledge. Together, these techniques effectively mitigate catastrophic forgetting and enhance continual generation performance in dynamic visual environments.

### Strengths
1. This paper presents the proposed setting and methodology through clear introductions and illustrations, along with detailed definitions.
2. This paper proposes a novel IL Benchmarks for OpenITG framework that addresses the issue of semantic overlap in image categories or background scenes across different tasks.

### Weaknesses
1. Given the continual changes in visual semantic themes, why only fine-tune the alignment module? In real-world scenarios, can this solution still perform well when encountering scenes or categories that the visual extractor has never seen before?
2. This paper employs BLIP-2 for experimentation. Has consideration been given to validating the method's effectiveness on more novel models? In particular, consider other forms of multimodal alignment such as Linear Projector/MLP.
3. How effective is the Fisher metrics screening in Fisher Dynamic Expansion? The lack of experimental demonstration shows how much unnecessary expansion Fisher metrics reduce during dynamic expansion. Also, the increased inference costs resulting from dynamic expansion have not been taken into account.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces Efficient Continual Alignment (ECA), a novel exemplar-free incremental learning approach for open-ended image-to-text generation that addresses the practical scenario where visual data categories shift over time. ECA enables vision-language models to continuously adapt to new images while preserving previously learned knowledge through three core mechanisms: a Mixture of Query module for task-specific adaptation, Fisher Dynamic Expansion for strategic model growth based on Fisher Information Matrix metrics, and Dictionary Replay using an embedding dictionary to maintain past knowledge without storing raw historical data. The authors construct four new benchmarks reflecting real-world conditions and demonstrate that ECA significantly reduces catastrophic forgetting and outperforms baseline methods in incremental learning scenarios where the alignment module must continuously evolve without access to previous task data.

### Strengths
1. This paper addresses a more realistic scenario where visual data distributions shift over time as environments evolve, unlike previous static assumptions. 
2. This paper introduces an exemplar-free approach that preserves knowledge without storing raw historical data, making it more practical and privacy-preserving. 
3. This paper proposes a novel continual alignment framework with dynamic model expansion that efficiently adapts to new tasks while minimizing interference with established cross-modal representations.

### Weaknesses
1. Based on Figure 2, the proposed FeDEx and Q-Former appear to be the same component, which contradicts the paper's claimed contributions.
2. The manuscript does not include a limitations section. The reviewer requests that the authors provide a comprehensive discussion of the method's limitations in their rebuttal response.
3. The experimental comparisons are limited to earlier image captioning approaches and lack benchmarking against recent large-scale vision-language models (e.g., Qwen-VL and LLaVA)

### Questions
Please refer to above weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes Efficient Continual Alignment (ECA), an exemplar-free incremental learning (IL) framework for open-ended image-to-text generation (OpenITG), which addresses catastrophic forgetting by adapting only the alignment module of pre-trained vision-language models (VLMs) (e.g., BLIP-2’s Q-Former) while freezing visual encoders and large language models (LLMs). ECA integrates three key components (Mixture of Query, Fisher Dynamic Expansion, Dictionary Replay) to preserve cross-modal alignment, and the authors construct four realistic IL benchmarks (ToS-COCO Caption, ToS-VQAv2, etc.) split by image main topics, with experiments showing ECA outperforms SOTA exemplar-free baselines in average performance, forward/backward transfer.

### Strengths
A key strength is the introduction of "continual alignment" as a novel concept for multi-modal IL, marking the first work to explicitly target preserving the cross-modal alignment of VLM alignment modules in exemplar-free OpenITG—filling a gap in existing works that either rely on raw exemplars or ignore alignment stability.

ECA’s component design is highly motivated and parameter-efficient: Fisher Dynamic Expansion uses FIM-based metrics to avoid unnecessary adapter expansion, and Dictionary Replay replaces raw exemplars with a sparse embedding dictionary (solving privacy/memory issues), while the self-constructed benchmarks (capturing real-world semantic overlap) ensure rigorous evaluation.

### Weaknesses
1. It is recommended that the authors conduct a comparative analysis of the computational complexity and memory costs between the proposed ECA method and the compared baselines.

2. Only MoE-LoRA is included as a multi-modal IL baseline. Recent works in 2025 tailored for incremental vision-language task are omitted, making it difficult to fully assess ECA’s standing in the multi-modal IL landscape.

3. It is suggested that the authors discuss existing MLLM-based continual learning methods [1] [2] that have covered VQA and image captioning tasks, and explicitly elaborate on the significance of the proposed ECA method in comparison to these works within this paper.

[1] MCITlib: Multimodal Continual Instruction Tuning Library and Benchmark

[2] Continual LLaVA: Continual Instruction Tuning in Large Vision-Language Models

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
