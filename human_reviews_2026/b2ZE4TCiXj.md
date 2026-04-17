# Controllable and Interpretable Multi-Value Alignment For Large Language Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Large Language Models (LLMs) are increasingly expected to embody human values in socially consequential contexts, but current alignment methods often lack interpretability, controllability and value diversity. 
We propose **V**alue-aligned **C**onstitutional **AI** (VCAI), a novel framework for fine-grained value alignment based on Schwartz’s Basic Value. 
Through VCAI we construct **ML-Values**, a multi-level dataset generated through role-playing, value decomposition, and iterative rewriting, allowing precise control over alignment intensity. 
ML-Values captures rich, context-aware expressions of values and supports multi-value alignment. 
Besides, by reformulating traditional value questionnaires into generative formats, we can obtain more accurate values assessment results.
Experimental results demonstrate that models trained with ML-Values present enhanced controllability and generalization across moral, psychological, and cultural dimensions. 
Moreover, alignment influences not only local response fidelity but also global value structures of LLMs, promoting coherent moral reasoning and structured preference expression. 
Our work offers a robust and interpretable foundation for building trustworthy, human-centered AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Value-aligned Constitutional AI (VCAI), a framework for constructing the ML-Values dataset to achieve fine-grained, controllable multi-value alignment in large language models. Based on Schwartz's Basic Value Theory, the method generates multi-level alignment data through role-playing, value decomposition, and iterative rewriting mechanisms, and explores two multi-value alignment strategies: mixed-dataset fine-tuning and expert model fusion.

### Strengths
- This paper addresses critical gaps in existing value alignment methods (lack of interpretability, controllability, and diversity) with a systematic solution that is both timely and practically significant.
- Comprehensive evaluation across psychological (Schwartz values), cultural (VSM13), and moral (MFT23) frameworks demonstrates cross-framework transferability.
- Insightful visualizations (e.g., MDS analysis) effectively illustrate the impact of alignment on model value structures, providing intuitive empirical support.

### Weaknesses
-  The choice of Linear and Karcher fusion methods lacks sufficient theoretical justification. Why are these two methods suitable for value fusion? Were other fusion strategies considered (e.g., attention-based dynamic fusion)?

- Experiments only on Qwen2.5-7B, lacking cross-model validation (e.g., Llama, Mistral). Generalizability is questionable.

- Visualizations in Figures 4&5 lack statistical significance testing. Cannot determine if observed differences are statistically meaningful.

- Iterative rewriting requires multiple LLM calls, up to 500 iterations. What is the computational cost? Are there efficiency optimization strategies?

### Questions
- Is value decomposition done manually or automatically? If manual, what is the composition of the expert team?

- How are failed samples handled (cases that don't meet standards after maximum iterations)?

- In mixed dataset training, how are data proportions across value dimensions determined? Was imbalanced sampling attempted?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a VCAI framework for achieving multi-value alignment in large language models. The framework constructs the ML-Values dataset, generating multi-dimensional value data.

### Strengths
The paper introduces a framework to construct a fine-grained human-value dataset based on psychological value theories, such as Schwartz’s Basic Human Values.

### Weaknesses
1. Novelty: The core contribution lies in applying the more fine-grained value dimensions from Schwartz’s Basic Value Survey to construct a dataset, followed by SFT on this dataset and reporting some alignment observations. However, the experimental results are not particularly inspiring and seem largely predictable. 
2. The introduction contains several overstated or conceptually inaccurate claims.
Example: Line 41 — the paper states that “existing datasets are limited in interpretability, controllability, and diversity.”It is questionable whether a dataset itself can possess interpretability or controllability. I don’t know if the authors understand the difference between interpretability and explanability. Even if the authors mean explainability, the claim that constructing a fine-grained dataset improves it is weak. Similarly, it is unclear how SFT with such a dataset directly leads to controllable behavior. These claims appear to overstate the paper’s actual technical contribution.
3. Dispersed and Unfocused Writing：The writing is highly scattered, making it difficult for readers to grasp the central insight. The paper frequently lists adjectives without clear conceptual grounding—for instance: Line 43: “broader, structured, and culturally varied”; Line 48: “fine-grained, controllable and interpretable datasets”; Lines 53–54:”VCAI uses role-playing...play evaluations”.

### Questions
1. Line 319: How exactly are the “correlation matrices of value dimensions” computed? Please specify the underlying metric.
2. The claimed contribution centers on multi-value alignment. However, there are no experiments comparing multi-value alignment with single-core-value alignment to demonstrate its advantage. Does it improve control success rate, fluency?
3. As shown in [1], SFT alone often fails to achieve strong control success rates. Therefore, the significance of using simple SFT methods for alignment remains questionable.

[1] Internal Value Alignment in Large Language Models through Controlled Value Vector Activation. Haoran Jin, et.al. ACL 2025.

### Soundness
3

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
There are many challenges faced by aligning LLMs with human values, such as aligning LLMs with multiple potential values and ensuring stability under value trade-offs. To support further research in this filed, this paper aims to construct a fine-grained, controllable and interpretable value dataset for alignment and evaluation. It proposes Value Contitutional AI, a framework with multiple role-playing and value decomposition, and an iterative rewriting mechanism to build the multi-level, controllable dataset ML-Value. With this dataset, they also explore multi-level alignment via both mixed-dataset fine-tuning and expert model fusion.

### Strengths
1. An automatic framework VCAL is constructed to build a fine-grained, controllable dataset ML-Values.
2. Methods for single value alignment and multiple value alignment are explored on the dataset.
3. Many experiments to verify and analysis

### Weaknesses
1. The significance of setting multiple role-playing for value evaluation in Sect 3.3 is unclear. How these roles for value evaluation distinguish from each other? Does each of them evaluate all sub-values and aggregate all their responses to get the final responses? Or does each of them only consider a separate sub-value as an expert? If all of them evaluate all sub-values, can I think that this step only benefit from majority voting?
2. The whole VCAI framework for value data construct lacks manual validation for the effectiveness. How about the accuracy of value evaluation in Step 3.3 and how about the quality of the final dataset? 
3. Several points in the writing and methodology need clarification.

- In line 203, the alignment level a_i \in [0,1], why does is inconsistent with the multiple level scores defined in Sec 3 (-2,-1,0,1,2). In this case, how do you use the above dataset for SFT?
- What data are used for supervised finetuning in Sec 4? Is the data synthesized in Sec 3?
- How to set the parameters lambda_i in Line 241?
- What are the relationship between the value assignment method in Line 261 and the value evaluation method in Sec 3.3? Can the method in Sec 3.3 be directly used for this goal?
4. More explanations are required about the baselines you selected. What are Hybrid strategy, Strongly Agree (St_a) and Strongly Disagree (St_d)? Moreover, the inclusion of more relevant and competitive baselines would strengthen the evaluation.
5. What do these experiments in Sec 7.1 mean about value alignment? How do they indicate the alignment degree towards different value profiles as the target? What value profiles, i.e. the value vectors, you set as the target to align with?

### Questions
1. In Sec 3.1, how to you decompose core values into sub-values, automatically, manually or just directly use the sub-values from the original theory?
2. Does the constructed ML-Values dataset only contain single-value data?
3. The readability of Figures 3, 4, and 9 is low due to small font sizes. Please improve figure clarity.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the alignment of LLMs with human values, grounded in the Schwartz Theory of Basic Human Values from social science, and proposes a data generation framework VCAI. VCAI utilizes a role-play based LLM data synthesis pipeline, which consists of two modules: the role-paly eval and re generation. The Role-Play Eval assesses whether the generated response reflect the target value and degree, while the Re-Generation re-generates a new response based on the assessment. Using VCAI, the authors created an alignment dataset, called ML-Values, and then conducted several analysis experiments, deriving findings: 1) alignment based on ML-values can change LLMs’ value patterns into a more modular, interpretable one; 2) the designed degree based alignment can successfully control LLMs’ values and bring cross-framework effects, influencing LLMs’ behaviors on culture, moral and cognition; 3) alignment may also produce some impacts on downstream tasks, especially on safety and security.

### Strengths
1. This paper focuses on an important direction, aligning LLMs with human values, which is increasingly more essential with the integration of LLMs with human life.

2. This work conducts a lot of experiments and analysis. Particularly, the analysis of alignment’s influence on other psychological frameworks and downstream tasks are interesting.

### Weaknesses
The topic of this paper is quite interesting, but there are several fundamental problems:

1. The biggest problem is the lack of baselines and comparisons. Besides the analysis experiments, the core contribution of this work lies in the alignment data generation pipeline VCAI. Considering there are already i) previous value alignment datasets rooted in Schwartz Theory [1][2]; 2) existing methods/pipelines to automatically create alignment data [3][4]. Without comparing with existing work, it’s hard to support the effectiveness of the proposed VCAI.

2. Another problem is that the quality of ML-Values is not verified. Since the whole construction pipeline frequently uses LLMs (generation of LLM-as-a-jude), the quality, including value relevance, degree accuracy, data coverage, diversity and fluency, of the generated data is unknown, which may significantly influence subsequent analysis. I noticed that there is a subsection about data quality in Appendix B (though it’s not mentioned anywhere in the main body). However, the key information of the human evaluation is not presented, e.g., evaluation protocol, number and background of human annotators, compensation, inter-annotator agreement and so on, which makes it hard to judge the quality of generated data.

3. The evaluation method is also questionable. In Sec.5, most analysis relies on the modified PVQ questionnaire. This brings two major problems: i) the size of questions is too small (only 57), which may fail to reveal the values of LLMs considering models’ Inherent bias and randomness; ii) Since PVQ is an old and famous questionnaire, which might have been included in LLMs’ training data, causing the data contamination problem [5]. Besides, in Sec.7.3, for other frameworks, I guess the authors also use questionnaires for measurement (since Appendix. E.4 is empty, it’s unknown. Correct me if I’m wrong). If so, this may cause another problem, as questionnaires/multi-choice questions have been proven to be inappropriate for LLMs [6][7].

4. There is a lot of missing information throughout the paper. In detail:

    (a) Appendix D.3, E.4 are empty.
    
    (b) The whole pipeline frequently uses LLMs (e.g., Sec.3. and line 264) for both generation and judgement, but there is no information about these LLMs.

    (c) There is not sufficient information about Linear, Hybrid, and Karcher.

5. Some experimental results and analysis are not clear or even confusing. More explanation and description are needed. For example,

    (a) In Fig.6, the clusters seem indistinguishable, but the authors claimed “dispersed, individualized value positions” (line 366)

    (b) In both Fig.7 and Fig.9, why does alignment cause the increase of both safety rate and attack success? These figures are confusing.

Generally, I like this paper’s idea and topic, but there are many essential problems and small issues.

Reference:

[1] Yao et al., Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Values. 2023.

[2] Qiu et al, VALUENET: A New Dataset for Human Value Driven Dialogue System. 2022.

[3] Li et al., CulturePark: Boosting Cross-cultural Understanding in Large Language Models. 2024.

[4] Kang et al., From Values to Opinions: Predicting Human Behaviors and Stances Using Value-Injected Large Language Models. 2023.

[5] Balloccu et al., Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs. 2024.

[6] Zheng et al., Large Language Models Are Not Robust Multiple Choice Selectors. 2024.

[7] Myrzakhan et al., Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena. 2024

### Questions
1. In Fig. 3, the modular, interpretable pattern induced by St_a and St_d alignment does not appear to fully conform to the Schwartz Value Structure. Could you clarify why?

Suggestions: (1) The caption/text at the bottom of Fig. 3 appears misaligned. (2) It would be better to reorder the figures so that each one appears close to the section where it is discussed.

### Soundness
2

### Presentation
1

### Contribution
2
