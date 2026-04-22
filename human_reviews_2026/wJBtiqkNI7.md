# Fine-Grained Mixture of Experts for Medical Multimodal Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Fine-Grained Mixture of Experts is a powerful architecture for scaling large models, yet its application in specialized domains like medicine remains underexplored. In this work, we conduct the first systematic study of expert granularity in a medical multimodal VQA context. Our findings reveal a fundamental trade-off: while increasing granularity significantly enhances out-of-distribution (OOD) generalization and robustness, it slightly degrades in-distribution (ID) fitting and, notably, amplifies expert co-occurrence and functional similarity, indicating stronger collaborative tendencies among experts. We argue that these intensified co-occurrence patterns place additional computational pressure on the routing mechanism, yet also reveal exploitable structure in how experts are jointly activated. To address this, we introduce Adaptive Expert Grouping (AEG), a novel, end-to-end learnable mechanism that leverages these collaborative patterns by dynamically clustering frequently co-activated, functionally related experts. By shifting routing decisions from the individual expert level to the group level, AEG substantially reduces computational overhead and improves model sparsity, while preserving the generalization benefits of the fine-grained architecture. We further observe similar co-occurrence phenomena beyond the medical domain, suggesting that our findings and AEG are broadly applicable. Our work offers a new path towards building more efficient and robust MoE models for specialized domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first systematic study of Fine-Grained Mixture of Experts (FGMoE) for medical multimodal learning. The authors investigate how expert granularity affects model performance. Key findings include that increasing granularity significantly improves out-of-distribution generalization and robustness but creates substantial functional redundancy among experts, adding unnecessary computational burden to routing. Main contributions include a shared expert that preserves pre-trained knowledge while enabling fine-grained specialization, and the Adaptive Expert Grouping (AEG) framework that dynamically clusters functionally similar experts into groups, shifting routing from expert-level to group-level to reduce computational overhead. 
Results show their finest-grained model (96 experts) achieves best overall performance, while AEG maintains comparable performance with substantially fewer activated experts.

### Strengths
1. The paper reveals the dual effect of granularity, which is a fundamental trade-off that hasn't been systematically studied. It provides practical guidance that identifies G=16 as an inflection point where benefits plateau.

2. Empirical evaluation is thorough, including a systematic granularity study, different types of baselines, and evaluation paradigms, including zero-shot, fine-tuned, and robustness. 

3. The proposed AEG mechanism can be learned jointly with task objectives and enables adaptive computation by allowing empty group selection.

### Weaknesses
1. The motivation is somewhat unclear; the paper claims that it focuses on medical applications, but the overall design of the approach is generic, rather than addressing a particular medical need. While the evaluation focused mostly on the medical datasets, it lacked detailed clinical insights into interpreting the results. For example, what does the routing modality distribution across layers reveal about the actual clinical patterns learned by the model?

2. Also on medical multimodal evaluation, it seems that the experiments are narrowly focused on VQA. While the model itself is not limited to certain tasks, the authors could consider extending this to a broader range of clinical tasks such as diagnostic tasks, report generation, bounding-box reasoning, etc.

3. AEG is presented as the solution to routing overhead, but no comparison with other efficiency approaches is provided. For example, expert pruning, hierarchical routing etc.

### Questions
1. In Figure 2, it is unclear where the matrices $A$, $E_g$ and $E_e$ are located.

2. In line 292, intuitively, as granularity increases, the router should tend to route tokens to different experts due to extra choice; but why does it turn out they are sent to the same expert pairs where they constantly process the same input?

### Soundness
3

### Presentation
2

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
This paper systematically studies fine-grained MoE in the field of medical multimodal learning, analyzing how expert granularity influences both in-domain and out-of-distribution performance. The authors empirically show that finer granularity substantially improves OOD generalization and robustness, while slightly reduces ID fitting and induces significant functional redundancy among experts, increasing routing overhead. To address this, they propose a grouping strategy called Adaptive Expert Grouping (AEG), which shifts routing decisions from individual experts to learned expert groups, to reduce computational cost and routing sparsity while preserving the generalization benefits of fine-grained specialization. Overall, the paper’s contributions include both novel empirical findings and a targeted methodological innovation.

### Strengths
- The observations provided by this paper on fine-grained MoE are interesting and original, which I believe could bring inspirations to future researches.
- The method proposed in this paper is explanable and well presented, and is effective according to the given experiment results.
- The experiments of this paper are comprehensive with results on multiple models and datasets.

### Weaknesses
- The domain focus of this paper is unclear. While the paper claims to focus on **medical multimodal learning**, the methodological core and analyses remain largely domain-agnostic. Most components are general techniques applicable to any multimodal or language model. The work does not convincingly justify why the medical setting is essential for either the problem formulation or the proposed method, beyond serving as an evaluation domain.
- The depth in interpreting the findings is limited. The empirical results are well presented, but the explanatory analysis lacks depth. For example, the authors note that finer expert granularity improves OOD generalization but degrades in-domain fitting, yet they provide little mechanistic insight into *why* this occurs. Similarly, the relationship between expert co-occurrence, redundancy, and degraded in-domain performance is asserted rather than rigorously examined (e.g., through probing, correlation, or ablation analyses). This limits the paper’s scientific contribution beyond empirical observation.
- The novelty of the proposed method is limited. Although the proposed AEG mechanism is a thoughtful extension, the idea of grouping or clustering experts based on similarity has been explored in several prior MoE variants (e.g. [[1]](https://arxiv.org/abs/2207.09094), [[2]](https://arxiv.org/abs/2504.09265)). The paper’s contribution lies more in recontextualizing this idea for fine-grained setups rather than introducing a substantially new principle. As such, the methodological novelty appears incremental relative to the growing body of literature on adaptive routing and expert aggregation.

### Questions
- Could the authors provide stronger evidence that the findings (e.g. optimal granularity, expert grouping behavior) are intrinsically tied to **medical multimodal tasks** rather than arising from general multimodal or MoE settings? How well do the proposed methods and conclusions transfer to non‐medical multimodal tasks (e.g. general vision+language or other non‐clinical domains)?
- There is a claim that with the same total activated parameter budget / number of active parameters, using finer granularity of experts leads to worse in-domain fitting. Could the authors explain what mechanism leads to that degradation? For instance, is it because with finer granularity, each expert receives fewer data points so they overfit less or under-specialize?

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
3

### Summary
Summary:

This paper proposes a Fine-Grained MoE for medical multimodal learning. An interesting point is that they have empirically demonstrated a clear trade-off: finer granularity gives better OOD generalization and robustness, but slightly worse ID fit and strong expert redundancy. To handle this redundancy, they propose Adaptive Expert Grouping (AEG), which learns to cluster similar experts and route at group level to cut routing cost while keeping gains. They also provide a fast proxy  for exploration. Experiments validate the effectiveness of the proposed method, and comprehensive analysis studies including the OOD-IID tradeoff, have been conducted.

### Strengths
Strength:

1. This paper systematically explored expert granularity and redundancy in leveraging MoE for multi-modal learning; and pointed out the resulting OOD-ID tradeoff. These are value findings that can guide MoE-based infra designs.
2. The model backbone is very updated, mostly are build on recent large VLMs or MoE backbones, and both main experiments and analysis studies are very comprehensive. Detailed results and experiemnt settings are provided in the appendix.
3. Strong results. The proposed method achieved SOTA or second highest points in most cases.
4. The proposed method is simple yet efficient, I think it's not complex to implement and can be easily plugged into many MoE infras.

### Weaknesses
Weakness & Questions:

1. My first question is that - why not pre-defined a group of shared experts, and several groups of sliced experts? Has your design been motivated by any evidence from training dynamics or other phenomena you observed?
2. For the $\mathcal{L}_{intra}$, is there any insight on doing the separation by considering the distances of experts' parameters? How about operating on the embeddings projected by the experts?
3. A potential solution to remove the redundancy while remain the high generalization ability, would be utilizing soft-MoE instead of sparse-MoE with gating networks. Because each expert in soft-MoE has been optimized join the training, and they are utilized in the inference time. Could you also discuss this type of works [1-4] in the related works?
4.  I noticed that you also used the histopathology images, how did you process them? If you use higher resolution pathology images, how do we change the design of shared experts and other sliced experts?
5. Is there any experiment results or expected phenomena about the scaling ability, to larger num_of_experts, parameters? 
6. Also, does the choice of vision encoder (like pretrained in medical images or not) affect the experiment results? 



[1] Puigcerver J, Ruiz C R, Mustafa B, et al. From Sparse to Soft Mixtures of Experts[C]//The Twelfth International Conference on Learning Representations.

[2] Wu C, Shuai Z, Tang Z, et al. Dynamic modeling of patients, modalities and tasks via multi-modal multi-task mixture of experts[C]//The Thirteenth International Conference on Learning Representations. 2025.

[3] Shen L, Chen G, Shao R, et al. Mome: Mixture of multimodal experts for generalist multimodal large language models[J]. Advances in neural information processing systems, 2024, 37: 42048-42070.

[4] Li Y, Jiang S, Hu B, et al. Uni-moe: Scaling unified multimodal llms with mixture of experts[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

### Questions
see weakness

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
3

### Summary
This paper studies how the granularity of experts in Mixture-of-Experts (MoE) architectures affects model performance in medical multimodal learning. The authors start from Qwen2-VL-2B and systematically vary the number of sub-experts per layer to observe the trade-off between in-distribution fitting and out-of-distribution (OOD) generalization. They find that finer granularity improves OOD performance and robustness but also increases redundancy among experts. To address this, they propose Adaptive Expert Grouping (AEG), a differentiable grouping mechanism that clusters functionally similar experts during training. The method relies on a Gumbel-Softmax assignment between expert and group embeddings, with a separation loss that encourages cohesive but distinct groups. Experiments on medical VQA datasets (SLAKE, PATH-VQA, OMNI-MINI) show that fine-grained MoE improves OOD generalization and that AEG achieves similar performance while activating fewer experts and reducing routing cost.

### Strengths
The experiments are extensive and well-documented. The authors carefully analyze the dual effect of granularity, showing improved OOD robustness but increased redundancy. The AEG mechanism is a practical way to exploit that redundancy. The results show clear quantitative improvements and the efficiency gains are convincing. The inclusion of robustness and co-occurrence analysis adds depth. Overall, the paper is technically strong and addresses a relevant open question in scaling MoE for specialized domains.

### Weaknesses
### W1 Limited novelty relative to adaptive gating work.
AEG is essentially a learned clustering mechanism that reduces routing granularity, which overlaps with ideas in XMoE [1] and DeepSeek-MoE [2]. These prior works already explored adaptive expert selection, specialization, and redundancy reduction. The paper would be stronger with direct conceptual and empirical comparisons showing how AEG differs beyond adding a learnable expert-to-group assignment.
### W2 Narrow experimental scope.
All experiments are on medical VQA, which is a limited slice of multimodal reasoning. Without additional tasks (e.g., retrieval, report generation, or non-medical datasets like ScienceQA), it is unclear whether the observed trade-offs and grouping effects generalize to other domains or modalities.
### W3 DenseMaskMoE proxy may distort efficiency claims.
The proxy simplifies training by replacing sparse dispatch with dense masking, but that also changes the real compute and communication pattern of MoE routing. Efficiency results therefore may not reflect actual runtime gains under realistic distributed setups such as DeepSpeed-MoE or Expert Parallelism.
### W4 Missing ablations to isolate the effect of grouping.
There is no comparison against static or heuristic grouping (e.g., k-means on expert weights or co-activation graphs), nor sensitivity analysis on group number, Gumbel temperature, or the separation loss coefficient. Without these, it is hard to tell whether AEG's benefit comes from learned grouping itself or simply from reduced routing depth.
### W5 Lack of interpretability of expert groups.
While the paper visualizes group sizes and sparsity, it never analyzes what these groups represent--whether they align with modalities, anatomical regions, or question types. Simple probes or correlation analyses could make the claimed "functional grouping" more convincing.
### W6 No statistical or significance reporting.
Results are single-seed, with small differences (around 0.5–1.5 points) that may fall within variance. Reporting mean and standard deviation across runs, or simple bootstrap tests, would make the comparisons more credible and show whether the improvements are statistically meaningful.

[1] XMoE: Sparse Models with Fine-grained and Adaptive Expert Selection. https://arxiv.org/abs/2403.18926

[2] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Model. https://arxiv.org/abs/2401.06066

### Questions
1. How sensitive is AEG to the number of groups and the Gumbel-Softmax temperature?
2. Could static clustering (for example k-means on expert weights) achieve similar results without end-to-end learning?
3. How much real compute saving does DenseMaskMoE provide compared to dispatch-based implementations?
4. Are the learned expert groups stable across random seeds or highly variable?
5. Have you tried applying the same setup to a non-medical benchmark to test whether the observed redundancy patterns are domain-specific?

### Soundness
2

### Presentation
2

### Contribution
2
