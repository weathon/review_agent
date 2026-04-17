# BaseReward: A Strong Baseline for Multimodal Reward Model

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The rapid advancement of Multimodal Large Language Models (MLLMs) has made aligning them with human preferences a critical challenge. Reward Models (RMs) are a core technology for achieving this goal, but a systematic guide for building state-of-the-art Multimodal Reward Models (MRMs) is currently lacking in both academia and industry. Through exhaustive experimental analysis, this paper aims to provide a clear “recipe” for constructing high-performance MRMs. We systematically investigate every crucial component in the MRM development pipeline, including reward modeling paradigms (e.g., Naive-RM, Critic-based RM, and Generative RM), reward head architecture, training strategies, data curation (covering over ten multimodal and text-only preference datasets), backbone model and model scale, and ensemble methods.

Based on these experimental insights, we introduce BaseReward, a powerful and efficient baseline for multimodal reward modeling. BaseReward adopts a simple yet effective architecture, built upon a Qwen2.5-VL backbone, featuring an optimized two-layer reward head, and is trained on a carefully curated mixture of high-quality multimodal and text-only preference data. Our results show that BaseReward establishes a new state-of-the-art (SOTA) on major benchmarks such as MM-RLHF-Reward Bench, VL-Reward Bench, and Multimodal Reward Bench, outperforming previous open-source and proprietary models. Furthermore, to validate its practical utility beyond static benchmarks, we integrate BaseReward into a real-world reinforcement learning pipeline, successfully enhancing an MLLM’s performance across various perception, reasoning, and conversational tasks. This work not only delivers a top-tier MRM but, more importantly, provides the community with a clear, empirically backed guide for developing robust reward models for the next generation of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on conducting a series of ablation studies on multimodal reward models and provides a comprehensive recipe for building high-quality multimodal reward models.

The main aspects explored include modeling paradigms (e.g., Naive-RM vs. Generative-RM), architectural design, training regularization, data configuration, initialization model selection, and integration strategies.

Based on the insights gained from these ablation studies, the authors trained a state-of-the-art reward model on multimodal RM benchmarks, named BaseReward, and further applied it to reinforcement learning (RL) pipelines, achieving noticeable improvements in policy performance.

### Strengths
- The paper conducts a thorough ablation analysis on multimodal reward modeling, with a clear experimental methodology and rigorous control of ablation variables, providing valuable insights and references for the research community.
- It presents several interesting experimental observations, such as the finding that incorporating textual data can improve the performance of multimodal reward models.
- The proposed BaseReward model demonstrates strong performance on multimodal RM benchmarks, particularly on VL-RewardBench.

### Weaknesses
- Essentially, the paper is more of an experimental report rather than a work proposing new technical innovations.
- Some of the claims made in the paper appear somewhat one-sided and lack convincing justification — for example, the preference for Naive RM over LongCoT-GRM, and the conclusion that scaling up model parameters brings limited performance gains (although the difference may be small on RM benchmarks, larger RMs could exhibit better robustness during real-world RL processes, as discussed in arXiv:2210.10760).
- The paper does not provide sufficient in-depth analysis or compelling theoretical explanations for some of its observations — such as the finding that incorporating textual data improves multimodal reward model performance.
- Although the proposed BaseReward performs well on VL-RewardBench, its improvements in actual RL training appear modest, and it shows no solid gains on more comprehensive benchmarks like MMBench or MMStar.

### Questions
- Regarding the conclusion on the “diminishing returns of model scaling,” could you provide a more comprehensive scaling curve? Could you also try increasing the training data size to examine whether larger models show clearer advantages with more data? Additionally, it would be valuable to compare the performance of small and large reward models in actual RL training.
- Concerning the choice of Naive RM over LongCoT-GRM, could you offer more complete evidence? From Table 1, it appears that after RL training, LongCoT-GRM achieves better average performance than Naive RM.
- For the observed phenomena (e.g., that incorporating textual data improves multimodal reward model performance), could you provide a deeper analysis or interpretation?
- Are there any potential technical innovations in multimodal reward modeling that could be further discussed or explored?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Through a comprehensive and large-scale empirical study, this paper provides a clear recipe for building high-performance Multimodal Reward Models (MRMs) and introduces BaseReward, a model that achieves strong results across multiple mainstream benchmarks. The work rigorously evaluates every component from reward modeling paradigms to data curation, revealing a key insight that high-quality textual data can significantly enhance the judgment ability of MRMs.

However, the paper’s claim of achieving “optimal performance” is not fully supported by its own experimental results; instead, it highlights the strong task dependence of the optimal architecture. Therefore, the true contribution of this work lies not in identifying a universal “best” configuration, but in providing the community with a powerful general-purpose baseline and a valuable experimental framework that clearly exposes the complex trade-offs between architectural and data design choices.

### Strengths
- This paper not only introduces a multimodal reward model (BaseReward) but, more importantly, presents a methodological framework for building high-performance Multimodal Reward Models (MRMs).
- It rigorously evaluates multiple key components — from reward modeling paradigms to data curation — providing researchers in the MRM field with a clear and valuable roadmap.
- The BaseReward model achieves state-of-the-art or highly competitive results across several mainstream benchmarks and demonstrates its effectiveness through real-world reinforcement learning pipelines.

### Weaknesses
- The paper aims to address questions such as “Which reward model architecture delivers optimal performance?” and “What constitutes the most effective architectural design for reward models?” However, the experimental results clearly show that no single architecture consistently performs best across all settings. These findings strongly indicate that “optimality” is relative and task-dependent. Therefore, positioning the paper’s goal as the pursuit of a universal optimal architecture is inappropriate and leads to a misalignment between the conclusions and the evidence presented. The authors should instead frame the work as establishing a strong general-purpose baseline and systematically revealing task-dependent architectural trade-offs, which would make the paper’s contributions more consistent with its empirical findings.
  - Table 1: The Generative RM (GRM) paradigm outperforms the paper’s chosen Naive-RM on both encoding and safety tasks.
  - Table 5: Different backbone models (e.g., Intern-VL and Qwen-VL) exhibit clear performance trade-offs between text-only and multimodal tasks.
- One of the most valuable findings in the paper is that “Incorporating text-only data improves multimodal RM performance.” However, the paper does not include any ablation studies on the mixing ratio between multimodal and text-only preference data. It would greatly strengthen the analysis if the authors examined, under a fixed total data budget, how varying this ratio affects the model’s performance across multiple dimensions (e.g., general VQA, mathematics, safety).
- Appendix B contains logical inconsistencies and structural confusion. Specifically, lines 796–797 state that “we compute normalized weights using three distinct methods,” but only two methods are listed immediately afterward. In addition, the hierarchical relationship between “category” and “method” in Appendix B is ambiguously described, which reduces overall clarity.

### Questions
- Since different backbone models can be integrated to complement each other’s strengths and weaknesses, is there also potential for ensemble learning across different types of reward models (e.g., Naive-RM and GRM)? For instance, could a dynamic routing or gating mechanism be designed to adaptively select or weight the outputs of Naive-RM and GRM based on task characteristics—such as determining whether the input involves code-related or safety-related queries?
- When constructing BaseReward, you mention that seven datasets were mixed. Could you specify the exact mixing ratios used among these datasets? Have you conducted any experiments exploring different mixing ratios? If so, do the results shed light on what balance between text and multimodal data yields the best synergistic effect—particularly across different downstream task categories (e.g., reasoning vs. safety)?

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
4

### Summary
This paper addresses the lack of systematic guidance for building Multimodal Reward Models (MRMs) by conducting exhaustive experimental analysis across all crucial components of the MRM development pipeline. Based on these insights, the authors introduce BaseReward, a baseline that uses a Qwen2.5-VL backbone with an optimized two-layer reward head and is trained on a curated mixture of multimodal and text-only preference data. BaseReward outperforms both open-source and proprietary models on major benchmarks.

### Strengths
1. This work provides the systematic analysis of all crucial MRM components including paradigms, architecture, training strategies, and data curation across over ten datasets. The rigorous experimental design offers comprehensive insights into multimodal reward modeling.
2. The paper reveals some interesting findings with notable novelty, e.g., text-only preference data significantly enhances multimodal RM performance, particularly in safety and mathematics dimensions. This challenges conventional wisdom and provides valuable insights for efficient data curation strategies.
3. The work establishes BaseReward as a strong SOTA baseline that outperforms existing models while providing actionable guidelines for the community to develop robust reward models.

### Weaknesses
1. As an empirical analysis paper, the work needs to improve how it presents conclusions. Each subsection lacks clear highlighting of main findings and actionable takeaways, making it difficult to quickly grasp the key insights. The paper would benefit from summarizing the most important conclusions at the end of each subsection.
2. The paper would benefit from adding statistical significance analysis to the performance comparisons, making it unclear whether observed differences are statistically meaningful or due to random variation.
3. While the paper provides extensive "what works" conclusions, it lacks sufficient analysis of "why it works," offering limited insights into the underlying mechanisms behind key findings.

### Questions
1. Why does GRM show advantages in coding and safety tasks while Naive-RM performs comparably or better in VQA, general, and hallucination tasks?

2. What are the underlying differences between activation functions and why does SiLU outperform Tanh and others?

3. Have you explored other length normalization methods beyond log-based normalization, such as data resampling to balance length bias proportions (e.g., https://arxiv.org/pdf/2502.17173)?

4. The ineffectiveness of zero-coefficient regularization is surprising. Could you analyze the potential reasons and what are the chosen/rejected reward distributions when not using this regularization?

5. Why can text-only data enhance multimodal RM performance while multimodal data cannot improve pure-text tasks? Is this related to MLLM model architecture?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BaseReward, a simple yet effect design for multimodal reward models. The authors systematically study different reward model paradigms, architecture design, training and data strategies, and introduce a 'recipe' for building strong multimodal rewards. They further propose to apply text-only preference data to improve multimodal rewards. The proposed methods achieves state-of-the-art results on various reward benchmarks. The authors also integrate BaseReward into a RL pipeline (GRPO) and show consistent gains in downstream multimodal perception, reasoning, and dialogue tasks.

### Strengths
- The study thoroughly ablates major components of reward modeling, offering practical guidance for future MRM development.
- Solid experiments and strong results. The proposed methods demonstrate clearimprovements on various reward benchmarks over previous reward models and even frontier models. The authors also conduct
- Extensive ablation studies are conducted to validate each individual components, enhancing the empirical soundness.

### Weaknesses
While the empirical study is solid and exhaustive, the proposed “recipe” remains empirical and lacks deeper theoretical understanding or technical insights.

- e.g. The paper shows that the performance of MRMs varies significantly with the choice of backbone, but does not offer clear analysis behind these variations.
- e.g. The analysis of pure-text versus multimodal reward models could be further elaborated—what are their distinct strengths, failure modes, or domain preferences?

### Questions
- Could the paper include qualitative examples of reward model outputs or RL training cases using BaseReward?
- What might be the underlying reasons that generative RMs does not perform better in multimodal domains?

### Soundness
2

### Presentation
3

### Contribution
3
