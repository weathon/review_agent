# On the Efficiency of Structured Pruning in Small Language Model Pretraining

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Recent advancements in generative language models have intensified the need for efficient and deployable models within limited inference budgets, while companies possess enormous computational resources for training.
This scenario opens a new regime and presents a fundamental question: given sufficient training resources but strict inference constraints, what is the most effective approach to obtain the best possible small generative language model?
One solution is to utilize structured pruning to compress a large model to a small model.
However, while structured pruning has shown promise compared to training a target-size model from scratch as shown in existing works, the overall efficiency becomes unclear when incorporating the cost of pretraining the large model that serves only as an intermediate step in our new scenario.
In this paper, we first study the question of whether it is worth pretraining the large model even if it is never deployed. 
Our results show that once the pretraining cost of the large model is taken into account, existing pruning methods are less token-efficient than training the target-size model from scratch.
Therefore, we further investigate how to improve the efficiency of the entire pipeline for producing small models.
To this end, we propose an integrated enlarge-and-prune pipeline, which combines enlarged model training, pruning, and recovery under a single cosine annealing learning rate schedule. 
This approach is further complemented by an iterative structured pruning method for the gradual removal of parameters.
We conduct comprehensive experiments on compressing 2.8B models to 1.3B with up to 2T tokens in pretraining. 
Our results demonstrate that the integrated approach not only provides insights into the token efficiency of structured pruning but also achieves superior performance of pruned models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes IDEA Prune, an integrated enlarge-and-prune pipeline that unifies model training, pruning, and recovery under a single learning rate schedule. The method shows improvements over naive pipelines and reveals important training dynamics

### Strengths
1. The paper is well written and targets the practical case of "abundant training resources but strict deployment constraints" and makes a convincing case that training dynamics impact pruning results beyond just the pruning method itself. 
2. Well-Designed Ablations with Interesting Findings： Section 5 provides a thorough examination of initialization, learning rate schedules, and model size factors. The paper identifies 2.6× FFN width as optimal, demonstrating robustness across a wide 1.3×-5.3× range and demonstrating an interesting synergy where pruning combined with knowledge distillation outperforms direct KD training from scratch.

### Weaknesses
1. All main experiments use only 2.8B compression, which cannot validate whether findings generalize to larger models. Does IDEA Prune work for 7B/3B compression? Would the 2.6× optimal enlarged size also hold for larger models, or does it scale with model size? Do intermediate checkpoints remain superior at larger scales?
2. Missing Ablations： No ablation study is provided to compare different token allocation strategies and iterative pruning step frequency.
3. The paper overlooks several critical factors established in prior work: **Learning Rate**: Prior research [1] shows that peak and end learning rates significantly influence pruning outcomes, yet this paper does not examine their effects. **Optimizer**: All experiments use only Adam without testing whether findings generalize to other optimizers. **Learning Rate Schedule**: The paper only compares cosine annealing while overlooking Warmup-Stable-Decay (WSD) [2], which outperforms cosine for small LLM training (and board using for current LLMs). Including WSD would strengthen the paper's claims.

[1] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration

[2] MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

### Questions
The key questions raised in the Weaknesses section above

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
The paper explores how to most effectively train small generative language models (SLMs) when training resources are abundant but inference constraints, such as memory and latency, are strict. To address these issues, the paper introduces IDEA Prune (IntegrateD Enlarge-And-Prune), a unified training framework that integrates model enlargement, pruning, and recovery into a single pretraining schedule.

### Strengths
1. The paper introduces IDEA Prune (IntegrateD Enlarge-And-Prune), a new, unified training pipeline that merges model enlargement, pruning, and recovery within a single cosine annealing schedule. This integration avoids the discontinuities and knowledge forgetting observed in conventional two-stage pipelines. Moreover, the method smoothly transitions between pretraining and pruning without resetting optimization states. 

2. The analysis section provides several counterintuitive yet insightful findings, such as: intermediate checkpoints, though weaker in absolute performance, yield better pruning outcomes than final checkpoints due to better learning rate alignment. Moreover, the learning rate schedule affects performance more than initialization quality.

### Weaknesses
1. The core components of IDEA Prune, iterative structured pruning, importance scoring via sensitivity, and cosine-annealed learning rate schedule are well-established in prior literature [1]. The paper’s main contribution is integrating existing techniques into a unified pipeline rather than proposing fundamentally new algorithms or pruning criteria. All technical differences are only a recombination of known ideas under a specific training regime.

2. The study exclusively focuses on pruning FFN width and excludes attention layers and hidden size, which are all included in Minitron and ShearedLLaMA paper.

3. How to decide the optimal hyperparameter configuration in the main experiments? The authors explicitly state that the token budget allocation (50% for enlarged pretraining, 50% for pruning + recovery) and learning rate settings in the main experiments were not tuned for peak performance but chosen to enable fair comparison with naive baselines.

[1] Sreenivas S T, Muralidharan S, Joshi R, et al. Llm pruning and distillation in practice: The minitron approach[J]. arXiv preprint arXiv:2408.11796, 2024.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the pretraining of small LMs using structured pruning. The authors argue that, when accounting for the pretraining cost of the large source model, the conventional pipeline (which prunes an existing large model and retrains it for performance recovery) becomes inefficient. To address this, they propose an enlarge-and-prune pipeline and compare it with both conventional pruning pipelines and training small LMs from scratch under an equivalent training token budget. Their analysis shows that the integrated approach, which adopts iterative structured pruning and a single cosine learning rate schedule, can produce more effective small LMs.

### Strengths
- This work revisits existing pruning–retraining pipelines and provides a controlled analysis comparing multiple scenarios for obtaining small LLMs.
- The proposed method, including its learning rate scheduling and FFN neuron pruning strategy, is simple and easily adaptable.
- The paper is clearly written and easy to follow.

### Weaknesses
- While the paper's flow and presentation are clear, the main motivation "Is it worth pretraining a large model even if it is never deployed?" seems somewhat unrealistic. In practice, a common way to obtain small LMs is by pruning or distilling already existing large models, rather than by pretraining a new large model solely for the purpose of pruning. The experimental setup does not include comparisons with existing pretrained large models (e.g., LLaMA-3, Qwen-3) pruned to comparable small model sizes to those studied in this work under an equivalent training budget, which would better reflect practical real-world settings.
- The study focuses solely on FFN pruning, possibly due to the ease of applying it in an enlarge-prune setting. Nevertheless, prior research (e.g., https://arxiv.org/abs/2405.18218) has shown that attention layers can often be more prunable than FFN layers, and many recent LLM pruning approaches jointly prune MHA and FFN, or even remove entire layers or transformer blocks. Could the authors also consider MHA and/or layer pruning, which are widely adopted structured pruning strategies?
- While the proposed method shows improvements over the conventional pruning-retraining pipelines in Table 1, the performance gains appear relatively modest and not clearly substantial compared to strong baselines such as Sheared LLaMA and Minitron. Moreover, Table 1 reports results only for a single target scale (1.3B), which limits the generality of the findings. It would be valuable to examine whether the observed trends hold for other small model sizes (e.g., 0.7B, 2B). In addition, the study considers only a 2.8B enlarged source model; it remains unclear how the method would behave when pruning from larger source models, which could provide further insights into scaling behavior.
- The analysis on the impact of training token budget is informative. However, since the proposed method focuses on structured pruning, it would also be valuable to provide an analysis of the practical inference-time speedup and memory footprint reduction. Such results would strengthen the empirical evaluation by demonstrating the real-world efficiency benefits.
- Although the paper claims to adopt iterative pruning, the main text does not clearly illustrate the prune-retrain iteration mechanism. If Algorithm 1 (currently in Appendix D) were included in the main paper, readers could more easily understand that pruning and retraining are alternated within each training step, making the 'iterative' nature of the method explicit.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether structured pruning remains efficient when accounting for the full training cost of LLMs. It finds that traditional enlarge-and-prune pipelines lose token efficiency once the large model’s pretraining cost is included, motivating a new IDEA Prune method that integrates enlargement, pruning, and recovery under a single learning-rate schedule. Experiments show that IDEA Prune achieves smoother knowledge retention and superior performance compared to naive pipelines or training small models from scratch, offering practical guidance on building compact yet powerful language models.

### Strengths
1. The paper presents an inspiring idea by firstly considering token efficiency during the pre-training stage of LLMs, offering a new perspective on the practicality of structured pruning.

2. It proposes a novel enlarge-and-prune pipeline that effectively eliminates the loss gap during training. The method is simple yet effective.

### Weaknesses
1. The underlying assumption of this paper seems somewhat unrealistic. In most practical scenarios, users do not pre-train an LLM from scratch that is never deployed; instead, they typically rely on open-source LLMs. As a result, users neither consider the training cost nor have access to the learning rate. Therefore, I have reservations about the paper’s practical applicability.


2. The experimental validation is also not sufficiently solid, as it only evaluates a single setting where a 2.8B model is compressed to 1.3B. Hence, I remain skeptical about whether the proposed method would remain effective for larger LLMs or under higher compression ratios.

### Questions
1. Do we need to consider the pre-training cost?

2. Does the proposed method remain effective for larger LLMs or under higher compression ratios?

### Soundness
3

### Presentation
3

### Contribution
2
