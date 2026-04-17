# Developmental Federated Tuning: A Cognitive-Inspired Paradigm for Efficient LLM Adaptation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Federated fine-tuning enables Large Language Models (LLMs) to adapt to downstream tasks while preserving data privacy, but its resource-intensive nature severely limits deployment on edge devices. In this paper, we introduce Developmental Federated Tuning (DevFT), a resource-efficient approach inspired by cognitive development that progressively builds a powerful LLM from a compact foundation. DevFT decomposes the fine-tuning process into developmental stages, each optimizing a submodel with increasing parameter capacity. Knowledge acquired in earlier stages is transferred to subsequent submodels, providing optimized initialization parameters that prevent convergence to local minima and accelerate training. This paradigm mirrors human learning, gradually constructing a comprehensive knowledge structure while refining existing skills. To efficiently build stage-specific submodels, DevFT introduces deconfliction-guided layer grouping and differential-based layer fusion to distill essential information and construct representative layers. Evaluations across multiple benchmarks demonstrate that DevFT significantly outperforms state-of-the-art methods, achieving up to $\textbf{4.59$\times$}$ faster convergence, $\textbf{10.67$\times$}$ reduction in communication overhead, and $\textbf{9.07}$\% average performance improvement, while maintaining compatibility with existing federated fine-tuning approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DEVFT, a federated fine-tuning framework designed to reduce the resource demands of large language model adaptation through cognitive developmental training. DEVFT progressively fine-tunes models across multiple developmental stages, each expanding parameter capacity. It employs a deconfliction-guided layer grouping mechanism and a differential-based layer fusion strategy to construct stage-specific submodels efficiently.

### Strengths
1.	The addressed problem (how to design stage-specific submodels that facilitate progressive knowledge transfer while optimizing overall performance) is both practical and relevant to real-world federated learning scenarios.

2.	The study is supported by comprehensive experimental evaluation across diverse benchmarks.

### Weaknesses
1.	The motivation could be strengthened. The framework assumes limited device resources only in the initial training phase, yet it remains unclear why smaller submodels are necessary if participating devices can accommodate full-model fine-tuning in the end. Furthermore, while the approach draws inspiration from human cognitive development, its applicability to heterogeneous resource settings warrants deeper justification—especially in cases where client resources evolve over time. I.e., for the cases that there are some devices who have more compute/memory resources available in the initial phase of the training and some clients who cannot accommodate larger models for the later half of the training.

2.	The connection between LoRA/PEFT's limitations and federated-specific constraints is not clearly articulated, raising the question of whether the proposed solution addresses a genuinely federated challenge or a more general fine-tuning issue.

3.	The layer grouping mechanism lacks empirical evidence for claims such as "opposite signs neutralizing each other’s unique contributions," and the conceptual clarity of this section could be improved.

4.	The assertion that “redundant layers limit representational diversity” conflicts with the subsequent need for a larger model, suggesting an inconsistency in the motivation for model expansion.

### Questions
1.	Does the framework assume identical or distinct data distributions across developmental stages, and how does this affect convergence?

2.	Is there reference to federated-specific parameter-efficient fine-tuning (pEFT) literature in Section 2.1, or are the cited works limited to centralized settings?

3.	Does each developmental stage correspond to a single global model update, or are multiple federated communication rounds performed within a stage?

4.	In the differential-based layer fusion process, how are non-linear interactions between layers preserved when performing linear subtraction-based fusion?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses the resource overhead in federated fine-tuning for LLMs at the edge devices and proposes a method called developmental federated tuning. This method decomposes the fine-tuning process into multiple developmental stages, with each stage optimizing a sub-model with increasing parameter capacity. Experimental results show that the proposed method outperforms several existing baselines in terms of communication overhead and convergence speed.

### Strengths
+ This works aims at practical issues in federated instruction tuning on the edge.
+ Figures are organized in a good shape.

### Weaknesses
1. Lack of investigation on existing work: Gradually introducing knowledge has already been demonstrated multiple times in research related to continual learning, such as in [1] (although [1] adjusts the model's capacity, which is not exactly the same as the technical approach in the current work). Additionally, the manuscript highlights the communication efficiency of this method in the introduction but does not discuss some existing works that optimize communication efficiency, such as [2] and [3].
2. Lack of baseline: The currently referenced resource-aware methods are either not tailored for LLMs or have not been formally published. It is recommended to include highly relevant methods from top conferences or journals in the past two years as baselines. 
3. The relationship between this developmental tuning method and FL is unclear. The current method design may yield certain effects at the edge and seems to remain valid even without the distributed training architecture of FL. 
4. This work uses a scenario with only 20 clients, which is too few for the cross-device scenario targeted by this work. It is recommended to conduct evaluations in scenarios with a larger number of clients and greater data heterogeneity. 
5. The paper lacks theoretical analysis of convergence and does not provide a comparison of convergence curves, raising concerns about the method's convergence performance.

[1] Compacting, Picking and Growing for Unforgetting Continual Learning. NeurIPS 2019. 

[2] Federated full-parameter tuning of billion-sized language models with communication cost under 18 kilobytes. ICML 2024.

[3] FwdLLM: Efficient Federated Finetuning of Large Language Models with Perturbed Inferences. ATC 2024.

### Questions
1. The paper introduces its method design by drawing an analogy to human growth. However, the final method design mainly focuses on progressively adjusting the scale of trainable model parameters. In human growth, different stages may involve learning knowledge of varying complexity. Could the method be further optimized by perhaps adjusting the learning difficulty or task complexity at different stages? 
2. Please refer to the Weaknesses.

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
This paper introduces DEVFT, a framework designed for resource-efficient federated fine-tuning of LLMs. DEVFT decomposes training into progressive stages of increasing model capacity, starting from a compact submodel. The method relies on two novel components: (1) **Deconfliction-Guided Layer Grouping (DGLG)**, which uses spectral clustering to group layers based on parameter similarity, and (2) **Differential-Based Layer Fusion (DBLF)**, which creates a representative layer for each group by fusing an anchor layer with parameter differentials. Knowledge is transferred between stages via LoRA parameters.

### Strengths
- **Novelty:** Introducing a developmental training framework to federated LLM tuning is creative and aligns with cognitive learning principles.
- **Technical soundness:** The DGLG and DBLF modules are mathematically clear and empirically validated through detailed ablations.
- **Practical benefits:** The approach significantly reduces communication and compute costs while improving accuracy on standard instruction-tuning tasks.
- **Compatibility:** The method can be combined with existing frameworks such as FedIT and FedSA-LoRA, further improving efficiency.
- **Reproducibility:** The authors provide source code, facilitating replication.

### Weaknesses
+ **Non-IID simulation:** Experiments are performed on datasets without explicit modeling of data heterogeneity, which is a core challenge in FL.
+ **Resource accounting scope:** The reported communication overhead accounts only for uplink transmission of LoRA parameters, while downlink costs (e.g., transferring dense submodels between stages) and server-side overheads for clustering and layer fusion are not explicitly quantified.
+ **Theory scope:** The convergence proof inherits classical FedAvg assumptions but does not analyze the approximation bias introduced by representative-layer fusion.
+ **Hyperparameter sensitivity:** The method depends on several hyperparameters ($\beta$, number of stages, initial capacity), yet sensitivity analysis is missing.

### Questions
1. **Layer adjacency constraint:** In Appendix B, each group's layers appear contiguous in the original model, but the clustering algorithm does not inherently enforce adjacency. How is this achieved or post-processed in implementation?
2. **Fusion weighting:** When constructing representative layers, have the authors considered similarity-weighted averaging (e.g., weighted by inter-layer cosine similarity) instead of uniform $\beta$-weighting? Would this further improve fusion fidelity?

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
5

### Summary
This paper presents DEVFT, a federated fine-tuning framework that reduces LLM adaptation costs through cognitive developmental training by progressively expanding submodel capacities across stages. Leveraging deconfliction-guided layer grouping and differential-based layer fusion, DEVFT achieves efficient and effective fine-tuning across multiple benchmarks.

### Strengths
The paper’s main strength lies in its effective reduction of resource consumption while maintaining strong performance across benchmarks.

### Weaknesses
1. The intuition behind the use case of this developmental process is unclear. While the method seems to optimize overall efficiency, it lacks discussion on peak efficiency (e.g., throughput, maximum GPU memory limits), which is often more critical in practical federated learning scenarios. To improve fairness and applicability, results should also be evaluated under varying resource constraints.

2. The paper lacks a clear system-level formulation of the FL process. Specifically, the grouping of parameters across clients is not well explained. In Equation (5), the meaning of θ is ambiguous and should be explicitly defined.

3. The current setup appears to assume that all clients have devices capable of hosting the full model. Can the method be extended to cases where not all layers are trainable due to device limitations?

4. It is also unclear how the method adapts to heterogeneous resources across clients. What strategies are employed for aggregation when clients train different subsets of layers?

5. Please consider including loss and accuracy curves. I am uncertain about the number of local and global training rounds, as downstream fine-tuning of LLMs typically requires fewer rounds. Further justification of the experimental settings would be helpful.

6. The choice of LoRA rank (e.g., 32) seems arbitrary. Since the optimal rank often depends on the downstream task, a fixed configuration might not generalize well. An ablation or explanation would strengthen this aspect.

7. The anchor layer is set to the first layer by default. Is there a particular reason for this choice? Would selecting a different anchor (e.g., middle or final layers) affect the performance?

8. The training rounds for each stage are fixed. Inspired by cognitive paradigms, one might argue that different stages (akin to learning phases) should have adaptive durations. For instance, adults may learn certain tasks faster than children and vice versa. Is the stage duration setting task-specific or task-agnostic?

9. If “capacity” refers to the number of unfrozen layers, what does the stage-wise scaling actually control? For example, the assumption of linear memory increase may not hold. As shown in Fed-Pilot [1], memory cost tends to grow non-linearly due to activation reuse and other factors. The effect of the scaling setting on the resource is worth discussing.

10. The implementation details of FedSA-LoRA as a baseline are unclear. In their original work, only the A matrix is shared across clients, and the server lacks a complete B matrix for global evaluation. How is this adapted in your framework for comparison?

11. Important prior works are missing from the related work section. For example, Fed-Pilot [1] provides a memory-aware LoRA allocation strategy; Fed-HeLLo [2] introduces heuristic and Fisher Information-based layer selection; FlexLoRA [3] and HETLoRA [4] address task and resource heterogeneity. These works should be discussed and compared if necessary.

12. Any generalizable findings from your experiments on the layer partition?

13. Lastly, Equation (5) appears to be a straightforward weighted average, as in standard FedAvg.

[1] Fed-pilot: Optimizing LoRA Allocation for Efficient Federated Fine-Tuning with Heterogeneous Clients. ArXiv 2024.

[2] Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation. IEEE TNNLS 2025.

[3] Federated fine-tuning of large language models under heterogeneous tasks and client resources. NeurIPS 2024.

[4] Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models. EMNLP 2024.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2
