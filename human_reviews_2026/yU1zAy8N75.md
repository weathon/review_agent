# Query Routing over Multimodal Knowledge Bases for Retrieval-Augmented Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Multimodal Retrieval-Augmented Generation (MRAG) has shown promise in mitigating hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge during generation. Existing MRAG methods typically adopt a static retrieval pipeline that fetches relevant information from multiple Knowledge Bases (KBs), followed by a refinement step. However, these approaches overlook the reasoning and planning capabilities of MLLMs to dynamically determine how to interact with different KBs during the reasoning process.
To address this limitation, we propose R1-Router, a novel MRAG framework that learns to decide ***when*** and ***where*** to retrieve knowledge based on the evolving reasoning state. Specifically, R1-Router can generate follow-up queries according to the current reasoning step, routing these intermediate queries to the most suitable KB, and integrating external knowledge into a coherent reasoning trajectory to answer the original query. Furthermore, we introduce Step-wise Group Relative Policy Optimization (Step-GRPO), a tailored reinforcement learning algorithm that assigns step-specific rewards to optimize the reasoning behavior of MLLMs.
Experimental results on various open-domain QA benchmarks across multiple modalities demonstrate that R1-Router outperforms baseline models by over 7\%. Further analysis shows that R1-Router can adaptively and effectively leverage diverse KBs, reducing unnecessary retrievals and improving efficiency and accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces R1-Router, a framework that performs step-by-step retrieval and routing over multimodal knowledge bases (including text, images, and tables). By applying Step-GRPO, it provides fine-grained rewards at each intermediate reasoning step. The authors claim that the method achieves an average improvement of around 7% over various RAG and routing baselines on multiple QA benchmarks, and further present analyses on routing preferences as well as comparisons of reasoning depth and computational cost.

### Strengths
* The paper proposes a new perspective on multimodal RAG, introducing a step-by-step routing and retrieval framework that enhances reasoning over text, images, and tables.

* The training objective is well designed — by integrating step-wise advantage estimation and two types of rewards (query/routing and answer) on top of GRPO, the method effectively mitigates sparse reward issues.

* The empirical evaluation is comprehensive, covering multiple QA scenarios (Text / Visual / Table) and providing clear analyses of routing behavior evolution and computational cost curves, with an average performance gain of around 7% over strong baselines.

### Weaknesses
* Novelty: The proposed method builds on an iterative plan–retrieve–reason paradigm and agent-based subproblem decomposition with multimodal routing, which have already been explored in prior work such as OmniSearch[1]. The core difference mainly lies in the step-wise reward and advantage normalization of Step-GRPO, but this appears more like an engineering refinement of GRPO (e.g., grouped normalization, format reward, step-level query/routing reward) rather than a substantial innovation with theoretical support.

* Single evaluation metric: he experiments rely solely on F1-Recall (token-overlap) across all tasks, which does not fully capture the objectives of multimodal, tabular, or extraction-based QA. The lack of comparisons with more appropriate metrics, such as Rouge-L, Retrieval Hit Rate@k, or other task-specific measures.

* Supervision for multiple valid routes: When multiple gold-standard routing paths exist for the same question, it is unclear how r_route handles supervision. Using a single label may be insufficient, and a set-based matching or multi-path reward strategy would be more appropriate.

### Questions
Please refer to the weakness section. There are two more questions:

- How do group size,  $\epsilon$, and in Step-GRPO affect performance and stability? Have the authors examined whether reward collapse or variance explosion occurs under different settings?

- How was the upper bound of $n \leq 3$ reasoning steps determined? How does the model perform on datasets that require longer reasoning chains?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes R1-Router, a framework for Multimodal Retrieval-Augmented Generation (MRAG) that dynamically decides when and where to retrieve external knowledge during reasoning. Unlike prior static or heuristic retrieval pipelines, R1-Router employs a reasoning-driven query routing mechanism, combined with a reinforcement learning algorithm called Stepwise Group Relative Policy Optimization (Step-GRPO). The introduction motivates this need by arguing that current MRAG systems treat retrieval as a one-shot process, neglecting dynamic reasoning and adaptive querying across heterogeneous multimodal knowledge bases (text, image, table).

### Strengths
- Propose R1-Router and Step-GRPO, a reasoning-aware MRAG controller that adaptively queries multiple modalities.
- Extensive experiments across six benchmarks demonstrate +7% improvement and robust generalization.

### Weaknesses
- From the method perspective, Step-GRPO relies heavily on ground-truth reasoning trajectories that are constructed using large teacher models (specifically R1-Distill-Qwen-32B and Qwen2.5-VL-7B). While this provides a strong supervisory signal during training, it also introduces potential distillation bias. Since the teacher-generated trajectories already embed the reasoning style and retrieval preferences of these large models, the student (R1-Router) may simply imitate the teachers’ patterns rather than discovering genuinely new reasoning behaviors. This dependency limits the generality and autonomy of Step-GRPO.

- In the method design, the Step-GRPO and R1-router introduce lots of hyperparameters, such as different reward designs. A thorough analysis or ablation of these parameters is missing. 

- Related work on advanced agentic search is missing.

### Questions
Could retrieval stopping be learned instead of fixed n≤3?

Is retrieval redundancy or cost considered in the reward function?

Does Step-GRPO scale to larger KBs or unseen modalities?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes R1-Router, a Multimodal Retrieval-Augmented Generation (MRAG) framework enabling Large Multimodal Language Models (MLLMs) to dynamically decide when and where to retrieve knowledge during reasoning. The model introduces Stepwise Group Relative Policy Optimization (Step-GRPO), i.e., a reinforcement-learning variant assigning step-specific rewards to intermediate reasoning stages. Experiments across 6 QA datasets show promising performance.

### Strengths
- The motivation is clear. Dynamic query routing over heterogeneous multimodal KBs is necessary to break through the limit of language only.

- Step-GRPO extends GRPO with fine-grained stepwise rewards for multi-stage reasoning.

- Dataset selection is extensive, which covers text, vision, and tables.

- The paper is well-organized with appendix details on retrievers and training configs.

### Weaknesses
- Figure 1 is not very self-illustrative; it is not very easy to distinguish the main breaking novelty or uniqueness of the proposed method.

- The theoretical contribution of the paper is not extensive. Step-GRPO is currently empirical; a convergence or variance analysis versus standard GRPO would add credibility and novelty to the paper.

- Training cost is not very clearly shown in the paper, which may be a concern. 

- The design of Eq (9) can be extended, especially the relationship between 'ask' and 'route', the need for balancing, whether they are contributing or competing with each other.

### Questions
Beyond weakness, there are also some questions below.

- How sensitive is Step-GRPO to $\alpha$ and $\beta$?

- What percentage of the generated reasoning trajectories are filtered out as incorrect during training (Appendix A.3)?

- How would R1-Router perform if KB modalities contain noisy or partially missing entries (e.g., imperfect captions)?

### Soundness
2

### Presentation
3

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
I found the core contribution, the Step-GRPO algorithm, to be a significant strength and a clear leap beyond static RAG. One of my issue is the model's dependence on golden trajectories from a teacher, which limits it to imitation rather than true discovery. I also question the claim of improved efficiency, as the accuracy gains appear to come at a substantial cost in inference time. Finally, the reported performance for some of the baseline is drastically different from its original paper, which undermines my confidence in the evaluation.

### Strengths
The Step-GRPO algorithm is a major highlight. By designing fine-grained rewards for each decision point in the reasoning process (sub-question generation, KB routing, intermediate answer generation), it effectively addresses the reward sparsity and credit assignment challenges faced by traditional RL in long-sequence, multi-action tasks. The paper is clearly written and well-structured. The method decides not only "what" to retrieve but also "when" and "from where," representing a significant leap beyond the traditional static RAG paradigm.

### Weaknesses
1. Dependence on "Golden Reasoning Trajectories": The Training relies on "golden reasoning trajectories" generated by a more powerful teacher model. This means the model's performance ceiling might be constrained by the quality of this synthetic data. The model may be biased towards imitating the teacher's specific reasoning patterns rather than discovering novel, potentially superior, paths.
2. As stated in the end of the abstract “…enhance both efficiency and accuracy”, improved accuracy can be seen in Sec. 5, however, the improved efficiency lacks detailed illustration. Why does “2s additional per-step latency align well with performance-computational-cost-trade-off” (Ling 410)? E.g., compared to CogPlanner, the F1-Recall increase around 25% while the inference time increases around 60%.

### Questions
1. As reported in [OmniSearch paper](https://arxiv.org/pdf/2411.02937) Table 4, OmniSearch obtains 41.20 on Dyn-VQA. However, in this paper, OmniSearch obtains 18.94 (Table 1). What causes this significant gap? (Did I miss something?) If KB routing degrades the model, what is the performance of OmniSearch (or other baselines) w/o KB routing?
2. Why does random routing with Step-GRPO obtain comparable performance (Table 2, 52.50 & 55.93)? Could you please elaborate on “random routing”? E.g., we may adopt Text Retrieval (selected by random) even if the router predicts to use Text-Image Retrieval?
3. Figure 2c: how many steps do other vanilla baseline methods use (E.g., CogPlanner and OmniSearch)?

### Soundness
3

### Presentation
2

### Contribution
3
