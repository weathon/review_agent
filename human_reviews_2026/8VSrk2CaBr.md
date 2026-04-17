# Unveiling the Cognitive Compass: Theory-of-Mind–Guided Multimodal Emotion Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Despite rapid progress in multimodal large language models (MLLMs), their capability for deep emotional understanding remains limited. We argue that genuine affective intelligence requires explicit modeling of Theory of Mind (ToM), the cognitive substrate from which emotions arise. To this end, we introduce HitEmotion, a ToM-grounded hierarchical benchmark that diagnoses capability breakpoints across increasing levels of cognitive depth. Second, we propose a ToM-guided reasoning chain that tracks mental states and calibrates cross-modal evidence to achieve faithful emotional reasoning. We further introduce TMPO, a reinforcement learning method that uses intermediate mental states as process-level supervision to guide and strengthen model reasoning. Extensive experiments show that HitEmotion exposes deep emotional reasoning deficits in state-of-the-art models, especially on cognitively demanding tasks. In evaluation, the ToM-guided reasoning chain and TMPO improve end-task accuracy and yield more faithful, more coherent rationales. In conclusion, our work provides the research community with a practical toolkit for evaluating and enhancing the cognition-based emotional understanding capabilities of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces HitEmotion, a three-tiered evaluation benchmark, and TMPO, a novel preference optimization method. The benchmark is constructed from 24 tasks across 22 datasets with standardized prompts to assess MLLMs' affective capabilities at progressive cognitive depths. The proposed TMPO method is designed to significantly enhance model performance on this benchmark.

### Strengths
The study features clear figures and a comprehensive set of experiments. The HitEmotion benchmark establishes itself as a systematic framework for the objective evaluation of MLLMs across the three stages of affective understanding.

### Weaknesses
The methodologies employed in TMPO are relatively common, indicating a potential lack of methodological novelty. The HitEmotion benchmark primarily focuses on evaluating model capabilities at isolated stages—for instance, assessing Stage 1 competence through Input A and Stage 2 through Input B. However, it overlooks the evaluation of multi-stage reasoning capabilities based on a single integrated input. A comprehensive emotional understanding of an event often requires holistic multi-stage reasoning, which the current benchmark design fails to adequately capture. Additionally, there is an issue with duplicated labeling in Figures 24 and 25.

### Questions
1. Does the HitEmotion benchmark evaluate the model's capability in processing inputs that may require multi-stage affective analysis? For instance, when provided with Input A (assuming A necessitates the simultaneous application of all three stages for a complete analysis), can the benchmark effectively assess the model's holistic competence across all three stages?

2. If a specific data instance from the original 22 datasets qualifies for multiple tasks among the 24 defined tasks, how is this data instance allocated? What is the principle or methodology for partitioning such multi-qualifying data to avoid data leakage or ensure unambiguous evaluation?

3. Regarding the reward function R, what are the precise computational formulas for each component on the right-hand side of the equation (e.g., R_structure, R_content, R_process, R_consistency)? A detailed description of their calculation is requested.

4. Concerning the parameters (µ1, µ2, µ3, µ4) in the reward function, does the current parameter selection demonstrably represent the optimal configuration? What evidence or ablation studies justify that this particular set of weights is the most effective for the optimization objective?

### Soundness
3

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
3

### Summary
This work addresses the limited emotional understanding of multimodal large language models (MLLMs) by emphasizing the need for explicit Theory of Mind (ToM) modeling. The authors introduce HitEmotion, a hierarchical ToM-grounded benchmark, and TMPO, a reinforcement learning method that leverages intermediate mental states for process-level supervision. Experiments demonstrate that HitEmotion reveals deep emotional reasoning deficits in current models, while the proposed ToM-guided reasoning and TMPO significantly enhance emotional reasoning accuracy and coherence.

### Strengths
1. It provides a large-scale datasets on emotion reasoning and a method for constructing ToM reasoning chains.
2. The experimental results show that the benchmark is useful and valuable, and the ToM reasoning chains can improve the model performance.

### Weaknesses
1. The paper models cognitive abilities into three levels. But it didn't analyze the relation among the three levels on model performance.
2.  The paper lack some analysis and discussions on the benchmark, e.g., data distribution, data quantity for each task.

### Questions
1. Could you analyze the relation among the three levels: EPR, EUA and ECR on model performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper enhances emotional understanding in multimodal language models by incorporating Theory of Mind (ToM) into reasoning. The authors introduce HitEmotion, a benchmark of emotion-related tasks organized by cognitive depth, and propose ToM-guided reasoning chains that trace beliefs, intents, and feelings before answering. They develop TMPO, a reinforcement learning framework that aligns model reasoning with intermediate mental state sequences using custom rewards. Experiments show that ToM-tuned models outperform both open and proprietary baselines on high-level emotional reasoning tasks, yielding more coherent and human-like rationales. This work contributes a new diagnostic toolkit and a training strategy for cognitively aligned affective reasoning in AI.

### Strengths
- HitEmotion Benchmark - The benchmark’s hierarchical ToM-based structure is unique, filling a gap in existing evaluations by mapping tasks to cognitive reasoning levels (first-order, second-order ToM, etc.). This will be valuable for the community.
- Ground Truth Reasoning - The reasoning chain curation pipeline shown in Figure 3 is also valuable, where the authors generate intermediate reasoning chains with an LLM and then refine them with human review. This yields high-quality supervision for the model’s thought process, which is a robust approach that enhances the credibility of the results.
 - TMPO Framework - The reward function in TMPO is particularly well thought out – it combines four complementary objectives (Structure Reward, Content Reward, Process Reward and the Consistency Reward) which are shown via ablation to work in synergy.
- Comprehensive Quantitative Results - The experiments are extensive, covering 24 diverse tasks and comparing a wide range of models. Such evaluation adds confidence that the improvements are real and not cherry-picked.

### Weaknesses
- Scope - The work's focus is confined only to the domain of emotion understanding. It’s unclear how well the proposed ToM-guided reasoning and TMPO training would generalize to other domains (e.g., logical reasoning puzzles, mathematical problem solving, or non-emotional tasks). The paper would be stronger if it discussed or demonstrated applicability beyond emotion-centric scenarios.
- Scalability to Larger Models - The experiments only use a 7B parameter model (Qwen2.5-Omni-7B). What is the rationale behind choosing Qwen2.5-Omni-7B as the base model? Does the approach scale to larger models as well, or are there diminishing returns because larger models might already internalize some ToM-like patterns, as already shown in Tables 3-5?
- Base Model Dependency - As acknowledged by the authors, for direct perception driven tasks (e.g. recognizing facial expressions from images), the model lags, primarily due to inherent limitation in the base model. Extending on the previous point about scalability, the reasoning would be more effective/convincing if other better base models are also used. 
- Computational Cost - How efficient is TMPO, both in training and during inference? How does it compare to other baselines? Since the authors choose only a small baseline model (7B parameters) and compare against larger open and closed source models, it would make sense to compare efficiency as well, and even claim it as an advantage if the numbers reflect that.    
- Reward Component Ablation Study - Although Table 6 indicates that the different reward components are effective and synergistic, it is not exhaustive, e.g. how does the model perform if R_structure is removed? Does it help or degrade performance for other combinations?
- Comparison with open source models - The authors state that "Emotion-LLaMA-7B attains 34.18 on the MQE task, outperforming most untuned baselines" (L420-421). Are there any open-sourced models in Tables 3-5 which are untuned? Emotion-LLaMA-7B only outperforms some open source models, and none of the zero-shot (untuned) closed-sourced models. Comparing against untuned open-source baselines would not be fair.

### Questions
- How generalizable is the ToM-guided reasoning approach to domains outside of emotion cognition?
- Do the authors anticipate even better performance if the approach is applied to a larger backbone (e.g., 13B or 70B model)? Or are there diminishing returns because larger models might already internalize some ToM-like patterns?
- How efficient is the TMPO approach?

### Soundness
3

### Presentation
2

### Contribution
3
