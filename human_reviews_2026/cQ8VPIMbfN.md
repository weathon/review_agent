# Fourier Minds, Forget Less: Discrete Fourier Transform for Fast and Robust Continual Learning in LLMs

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
Continual learning (CL) for large language models (LLMs) is challenged by both catastrophic forgetting and efficiency constraints when facing long sequential tasks. While low-rank adaptation in LoRA-based approaches reduces per-task trainable parameters, the cumulative parameter budget grows with stream length and can be substantial. This limits their applicability in lifelong learning scenarios, especially under strict resource constraints. In this work, we explore the potential of the parameter-efficient Sparse Fourier Transform (SFT) in the context of continual learning. Our preliminary experiments reveal that directly applying SFT in CL settings leads to temporal instability and forgetting.  Motivated by this finding, we propose Discrete Fourier Continual Learning (DF-CL), which leverages a spectral decomposition strategy to disentangle shared and task-specific knowledge components, facilitating more stable continual learning. By leveraging the orthogonality properties inherent to the SFT bases, DF-CL ensures that task-specific knowledge is encoded within its own dedicated parameter space, minimizing interference between tasks. Furthermore, we introduce a max-magnitude task-weight merging strategy, which enables efficient knowledge consolidation and transfer across sequential tasks. Extensive experiments on both T5-Large and LLaMA2-7B demonstrate the scalability, efficiency, and effectiveness of DF-CL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to address the cumulative parameter budget growing with the number of sequential tasks for continual learning in LLMs. Based on the instability and forgetting issue of directly applying Sparse Fourier Transform in continual learning, the authors propose Discrete Fourier Continual Learning (DF-CL), which decouples general and task-specific knowledge during training and utilizes a max-magnitude merging strategy in the post-training phase, to maintain the previous knowledge as well as adapt to new tasks. Experimental results show that DF-CL achieves superior performance and reduces the trainable parameters than other methods.

### Strengths
1. The problem this paper aims to address is important for continual learning in LLMs, since existing methods almost initialize a newly added LoRA for a new task, which heavily increase the cost with the growth of the sequential tasks.

2. The proposed DF-CL method can reduce the number of trainable parameters for sequential tasks.

### Weaknesses
1. The paper does not provide any theoretical analysis to support the principle of the proposed DF-CL, especially for the max-magnitude merging method. The paper does not clearly explain why DF-CL chooses this merging method theoretically.

2. The hypothesis of DF-CL proposed in this paper lacks support. In line 226, it claims the motivation and reason “we first hypothesize that the observed stability gap stems from the use of a shared coefficient vector x across all tasks” for utilizing task-specific branch in DF-CL, but the paper directly makes such statements without any analysis or related work. This makes the process of proposing task-specific coefficients not convincing.

3. The experiments only use classification tasks, which are relatively simple to evaluate the performance of the proposed DF-CL in LLMs. NLP Generation tasks like SuperNI benchmark have been utilized in parameter-efficient continual learning, but this paper does not conduct experiments on such benchmarks. 

4. To show the performance of DF-CL, the paper tried to show the accuracy flow of the first task across sequential tasks, such as Figure 1(c). But this result cannot fully evaluate DF-CL, since the metric average accuracy reflects not only previous tasks' accuracy but also new task accuracy. If only showing the accuracy of the first task, especially in an order of four tasks, it cannot explain how the proposed method achieves the trade-off between maintaining previous knowledge and adapting to new tasks.

5. The evaluation metric is only the average accuracy. There are other common metrics, such as OPD [1], to better show the performance of the method.

[1] Scalable and Order-robust Continual Learning with Additive Parameter Decomposition, ICLR2020.

6. The experimental tables in this paper are not presented in an organized way, and also do not explain well for each table. For example, Figure 3 does not define the meaning of x-axis.

### Questions
1. During training, since DF-CL still freezes previous tasks' task-specific coefficient matrices, can we have the conclusion that DF-CL does not save computational memory during training? Since in the forward, previous task task-specific coefficient matrices still join the computation for the output. 

2. DF-CL involves merging strategy after all tasks training. But the problem is, after addressing current sequential tasks, when another task comes, how does DF-CL balance this new task and the previously merged task-specific coefficient matrices? Why not change DF-CL to merge each task directly after its training? Why does DF-CL choose to merge all tasks coefficient matrices after all tasks training?

3. Can authors compare the experimental performance of DF-CL with that of SAPT [1] and InfLoRA [2]? Also, can authors use SuperNI benchmark to evaluate DF-CL?

[1] SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models, ACL2024.

[2] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning, CVPR2024.

4. Figure 1(c) only shows the performance of the first task drops down may not be solid enough to show the effectiveness of DF-CL, and that’s why the common metric is average accuracy.

5. Figure 3 does not define the meaning of x-axis. For readers who are not familiar with continual learning, it’s a little hard to understand. Figure 3 only lists the performance of the first task to support the claim “instability” of directly applying SFT, which is not solid.

6. There is no clear definition of “sensitivity of model weights to the tasks” and “instability of task performance” in the paper, which makes it confusing about how to distinguish them.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a discrete Fourier continual learning algorithm, a continual learning framework for large language models based on the sparse Fourier Transform. The proposed method decomposes model updates into spectral components, separating shared and task-specific knowledge while enforcing orthogonality among tasks. It introduces a max-magnitude weight merging strategy to consolidate task-specific adaptations into the global model. Experiments on T5-Large model and LLaMA2-7B model across standard and long continual learning benchmarks show improvements over LoRA-based baselines with only 1–3% of trainable parameters.

### Strengths
1. This paper is well structured, provides sufficient implementation details, and includes explicit training configurations and baselines, supporting reproducibility of the proposed method.
2. As the authors claimed in Table 1, the proposed method greatly reduces trainable parameters without sacrificing accuracy, which is significant for scaling continual learning with large language models.

### Weaknesses
1. The idea of studying sparse Fourier Transform in continual learning is only superficially motivated. The authors claim that frequency decomposition separates “shared” and “task-specific” knowledge, but this analogy is not well justified. No analysis is given on what the frequency components represent in the model weights or how this relates to forgetting dynamics. The method appears as an arbitrary parameter reparameterization rather than a principled CL framework. To conclude, this is rather heuristic, while lack of theoretical gurantees.
2. The proposed approach mainly combines existing ideas of FourierFT parameterization and the LoRA-like task decomposition, without introducing a fundamentally new mechanism for mitigating forgetting in continual learning. The “max-magnitude merging” is a simple heuristic lacking theoretical analysis or ablation explaining why it should preserve knowledge.
3. The experiments focus exclusively on text classification datasets. No reasoning, generation, or open-ended tasks are evaluated, which makes the results less impactful for LLM continual learning. 
4. Baselines are limited and not fairly tuned. Classical regularization-based methods are either omitted or weakly configured. And the reported improvements are small (often within 1–2%), and there is no statistical analysis to assess significance.
5. The paper is a little bit verbose.

### Questions
1. Will the method be instabile if tasks are highly correlated, e.g., sequential fine-tuning on similar domains?
2. Is the merging operation performed once at the end or progressively after each task?

### Soundness
3

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
4

### Summary
This paper proposes Discrete Fourier Continual Learning (DF-CL) for continual learning, which leverages Sparse Fourier Transform (SFT) for parameter efficiency. Experiments on T5 and LLaMA show DF-CL outperforms baselines.

### Strengths
1. This is the first work to apply SFT to CL, moving beyond LoRA/prompts to decouple knowledge via spectral properties.

2. Extreme parameter efficiency: DF-CL uses far fewer parameters (e.g., 120k vs. 11.8M for O-LoRA on T5-Large with 15 tasks), with the gap widening for more tasks/larger models.

### Weaknesses
1. Only evaluates classification tasks (sentiment, NLI, QA), not complex tasks like reasoning or open-ended generation.

2. Claims Fourier bases’ orthogonality reduces interference but lacks formal proof or quantitative overlap analysis.

3. Focuses on parameter efficiency but not IDFT’s computational cost for large weight matrices (e.g., LLaMA2-7B’s 4096×4096).

### Questions
1. Will DF-CL work for non-classification tasks (e.g., generation)?

2. How is Fourier bases’ orthogonality preserved during training (e.g., no global coefficient leakage into task subspaces)?

3. What’s DF-CL’s training latency vs. O-LoRA, especially IDFT’s cost?

### Soundness
2

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
This paper proposes Discrete Fourier Continual Learning (DF-CL), a parameter-efficient method using the Sparse Fourier Transform (SFT) to separate shared and task-specific knowledge in continual learning. By leveraging Fourier orthogonality and a max-magnitude merging strategy, DF-CL reduces interference and forgetting, achieving strong performance on T5-Large and LLaMA2-7B with only 1–3% trainable parameters.

### Strengths
1. The paper is well-organized and easy to follow.

2. The investigated problem of continual learning using low-rank adaptation is important.

### Weaknesses
1. The paper appears to overlook several relevant studies on continual learning (or continual fine-tuning) of large language models, such as LoRA-MoE and TreeLoRA. Providing a discussion on how the proposed approach relates to or differs from these methods would help clarify its novelty and contribution.

   *LoRA-MoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin*

   *TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree*

2. It would be appreciated if the authors could further elaborate on the main contributions of the paper. In particular, a clearer explanation of the fundamental challenge in integrating the Discrete Fourier Transformation with LoRA would strengthen the clarity and significance of the proposed method.

3. In the experimental section, it is unclear why only accuracy results are reported. Including the forgetting metrics and the standard deviation of results would provide a more comprehensive and reliable evaluation.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
