# HieraSuite: A Holistic Toolkit for Building Versatile System-User Instruction Hierarchy

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Instruction Hierarchy (IH), the structured prioritization of system prompts over user prompts, has emerged as a key security mechanism for language models (LMs). Despite its importance for flexible steering and robust safety control, current LMs offer limited support and often fail to enforce system-level specifications when these conflict with user instructions. In this work, we introduce HieraSuite, a full-stack toolkit for building steerable and secure system-user IH for LMs. HieraSuite encompasses four key components: (1) HieraInstruct, a large-scale and diverse collection of 221K system–user instruction pairs spanning four real-world application domains (system constraints, privacy and security, steerability, and task execution); (2) HieraConsReasoner, an effective and compact reasoner model, paired with training data, that elicits contextualized rubrics to specify what constitutes valid responses under hierarchical instructions; (3) HieraCRO, an iterative response optimization approach, grounded in constitutional rubrics, that enhances LM compliance with instruction hierarchy; and (4) HieraBench, a unified benchmark that integrates ten tasks to assess controllability, steerability, customizability, and security of system-user instruction hierarchy. Together, these components form an end-to-end solution that yields consistent gains across model families and scales, including up to 66.9% improvements on HieraBench tasks and over 306.3% gains in overriding conflicting user instructions. Systematic testing of alignment recipes further identifies design choices that balance user instruction-following, system instruction-override, and general capabilities. This work provides a principled framework and practical toolkit for LM user-system instruction hierarchy, laying the foundation for future studies on “instruction un-following” and advancing steerability and security in LM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work enhances the LLMs' ability to follow the system-user instruction hierarchy by a systematic process: (1) they construct a dataset of system-user instruction pairs that include conflicting instructions, (2) they distil the GPT-4.1 model's responses for these instructions into a small 7B model, and then (3) they train a model using DPO and SFT to optimize for generations that are refined by the distilled model in the loop. Finally, experiments on a benchmark show significant improvements for different language models.

### Strengths
- The paper is mostly well-written with a good writing structure, minimal typos, and a documented Appendix. 
- Experiments are comprehensive and cover a large space of hyperparameters and ablations. 
- Curated tasks are well-aligned with the objective, and the trained model could also be helpful when released. 
- The proposed approach improves the instruction hierarchy ability of language models across different settings. 
- Appendix includes sufficient examples for the dataset.

### Weaknesses
- There is very limited originality, and the paper is just a technical report of a standard sequence of implementations over the gaps identified in Zhang et al., 2025c.
- There is a lack of transparency regarding the synthetic data creation, and it is not clear why it is needed at all, given that a lot of datasets are already available. If the contributions are mainly the new synthetic dataset, then this should be clearly written and the whole process of creation explained clearly. 
- Variance in Figure 3 should be included. GPT-4.1 is used as the judge model and also the teacher model, which might lead to biased evaluations. 
- The sampling strategy for HCReasoner training data is not clearly presented.
- There is a potential overlap between the HieraInstruct dataset and the benchmark datasets, which again leads to a biased evaluation. This also limits the generalization capability of the instruction hierarchy, which should also be studied in out-of-distribution tasks in addition to the generic tasks of MMLU.
- Ablation on the datasets (prompts for the preferences) used to train the HieraCRO models should be considered. Currently, it is only with HieraInstruct and HieraInstruct+HelpSteer but it is important to study what is the least amount of data (number of preferences and their category-wide diversity) that would be enough to reach a decent performance. 
- The choice of using HCReasoner instead of self-improvement is not well-demonstrated in the experiments, as the performance gains are not quite enough to arguably motivate training an extra model and make the contributions even more obscure. 
- Minor:
  - Figure 1 and 2 are both very hard to follow given the color choices and the density of text.

### Questions
- Why not just train a model with IHEval? Just full fine-tuning or preferential DPO following the strategy defined here.
- Why do we not need a clever sampling strategy (either oversampling or such) to handle conflicting pairs as opposed to non-conflicting?
- What happens if the user prompt conflicts with GPT itself? In that case, won't the distillation become a bottleneck?
- See above weaknesses

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework to evaluate and improve the LM's ability to follow the instruction hierarchy. The author first created HieraInstruct to collect system-user instruction pairs, and use them to train response reasoners to help improve the instruction hierarchy following ability of the model. The improvement is done by iterative refinement of the model response and using contrastive pairs to finetune the model with DPO. The improvement was shown on diverse instruction following datasets.

### Strengths
The paper did a comprehensive analysis and data collection to help improve the instruction hierarchy following ability. 
The data collection serves a good evaluation and training framework for future LLM design.
The ablation study could support the experiment design well.

### Weaknesses
The main weakness is the over-reliance on the LLM evaluation over the whole pipeline. This is understandable as large-scale data collection requires a scalable approach for evaluation. However, it would be great if there is any human evaluation involved, or at least more focus on the LLM-as-a-Judge analysis. 

Currently, the evaluation part has little information on how the evaluation was done.

### Questions
see weakness above

### Soundness
3

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
3

### Summary
The paper presents a large-scale framework for constructing and evaluating instruction hierarchices in LLMs. The is a very important question that poses safety concern, so the authors propose HieraSuite, a holistic toolkit consisting of dataset, reasoning model, an iterative response optimization framework and a suite of evaluation benchmarks and demonstrated that the proposed model achieved larged gains on their benchmark while maintaining general performance.

### Strengths
1. Quality: The paper is technically sound, with very extensive and comprehensive empirical experiments and a consistent methodology across diverse model backbones. Finishing this project requires lots of computing resources.
2. Clarity: the paper is well-written and visually organized. Although I feel Figure 1 is a bit too dense but doesn’t hurt the overall readability. Details are provided so that I believe the results are reproducible.
3. Significance: given that IH is a very important topic to improve LLM safety, this paper contributes a very valuable infrastructure for future research.

### Weaknesses
1. Originality: the contribution is primarily empirical and integrative, not methodological contributions. For example, it reuses existing frameworks including DPO-style preference optimization, constitution guided scoring [1], iterative refinement [2] with minimal algorithmic innovation. Besides, for the HierBench, most of the benchmarks are reused public benchmarks.
2. I’m worried that much of the reported gain seems to come from scale and data curation, rather than a new training principle.
3. It would be helpful if authors could clarify how evaluation benchmark datasets are selected and to what extent they overlap with the training or optimization data used in HierCRO (especially since you feed synthetic data which are harder to control). Given that both the method and the benchmarks emphasize hierarchical obedience, it will be nice to provide evidence or analysis demonstrating that the reported improvements are not primarily due to benchmark-specific tuning. 

[1] Bai, Yuntao, et al. "Constitutional ai: Harmlessness from ai feedback." arXiv preprint arXiv:2212.08073 (2022).

[2] Madaan, Aman, et al. "Self-refine: Iterative refinement with self-feedback." Advances in Neural Information Processing Systems 36 (2023): 46534-46594.

### Questions
In line 175, when authors mentioned: “For practical use, LMs must (i) override conflicting user instructions, (ii) integrate supplementary non-conflicting system constraints, and (iii) perform robustly on user-only inputs.” I wonder if the model were trained in this way, how does the model perform on user system conflict detection, and people could prefer an LLM that explictly acknowledge the conflict first without continuing directly to override user instructions. Otherwise, you could imagine a user will get confused because they feel the model is not following provided instructions.

### Soundness
3

### Presentation
3

### Contribution
2
