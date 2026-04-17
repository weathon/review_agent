# Omni-RRM: Automatic Preference & Reasoning Construction Advances Multimodal Reward Modeling

- Decision: Reject
- Scores: 6, 6, 2, 6, 6

## Abstract
Multimodal large language models (MLLMs) have shown remarkable capabilities, but their safe deployment is hindered by alignment failures. A critical bottleneck is the lack of effective reward models (RMs), which are typically limited to vision, provide opaque scalar scores, and depend on costly human annotations. To address these challenges, we introduce Omni-RRM, the first open-source, reasoning-driven reward model that provides explainable preference judgments across text, image, video, and audio. At the core of our approach is Omni-Preference, a novel, large-scale dataset constructed through a fully automated pipeline: we generate preference pairs by contrasting models of varying capabilities and then enrich them with multi-criteria, chain-of-thought rationales from powerful teacher models, completely eliminating the need for human labeling. Omni-RRM is trained in a two-stage process: supervised fine-tuning to instill the ability to generate structured rationales, followed by reinforcement learning to sharpen its judgment on difficult, low-contrast examples. Comprehensive evaluations show that Omni-RRM achieves state-of-the-art accuracy on video (80.2% on ShareGPT-V) and audio (66.8% on Audio-HH) benchmarks, while significantly outperforming existing open-source RMs on image tasks, with an overall improvement of 17.7% over its base model. Furthermore, Omni-RRM demonstrates strong generalization, boosting downstream task performance via Best-of-N selection and even improving accuracy on text-only preference benchmarks. Our data, code and models are available at https://anonymous.4open.science/r/Omni-RRM-CC08

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Omni-RRM proposes a unified multimodal reward model that judges image, video, audio, and text responses through structured reasoning instead of scalar scoring. Using the Omni-Preference dataset (~41K auto-labeled pairs) and a two-stage training scheme, it achieves state-of-the-art accuracy on preference benchmarks and improves downstream generation.

### Strengths
1. It redefines the notion of a reward model as a structured, reasoning judge rather than a scalar scorer.
2. It proposes a unified reward model.
3. It introduces Omni-Preference, a scalable automatic preference dataset with structured multi-criterion justifications.

### Weaknesses
1. The Omni-Preference dataset labels the "strong model" output as preferred and the "weak model" output as rejected. 
2. The current data scale is limited.
3. The design of the format reward is not clear. "If any field is missing, malformed, or includes illegal characters, the output
is assigned a strong negative reward." should be illustrated more clearly. Besides, would the structured JSON-based reward encourage the critic to optimize for format conformity rather than genuine comparative reasoning.
4. Could the Omni-RRM be extended to a single-response evaluation setting? The performance should be reported.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
3

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
This paper introduces Omni-RRM, a reasoning-driven multimodal reward model (RRM) that provides explainable preference judgments across text, image, video, and audio modalities.

### Strengths
1. Reformulating reward modeling as a generation task with interpretable reasoning output (rather than scalar regression) is an important conceptual advance

2. The methodology is sound and clearly detailed: the dataset creation, two-stage training, and reward design are carefully explained, with mathematical formulations (Eq. 1–7) and implementation specifics (Appendix A).

3.The zero-human-annotation pipeline has major practical significance for future MLLM alignment, offering a potentially general framework for automated preference modeling.

### Weaknesses
1. The “capability-gap” configuration in Table 1 primarily contrasts 7B vs 3B models within the same family (e.g., Qwen2.5-VL-7B vs 3B). According to the Qwen2.5-VL technical report, the performance gap between these variants is relatively modest.  While the Stage-1 assumption that “the stronger model tends to produce better answers” is reasonable on average, such a narrow capability difference may not reliably hold at the sample level. Consequently, label noise is likely introduced when the weaker model occasionally provides correct or even superior outputs, pushing the burden of correction entirely to the subsequent teacher-filtering stage.

2.The paper does not report how often weak models outperform strong ones, how many “tie” cases the teacher found, or what fraction of Stage-1 pairs were discarded after Stage-2 reconciliation. These statistics are essential to assess the actual reliability of the automatically generated preference labels.

3.If both strong and weak models generate incorrect answers, the comparison still forces a relative choice. Without an external correctness reference, this may introduce relative bias—the model learns which incorrect answer the teacher preferred rather than what a correct answer should look like. The authors should clarify how such cases are detected or mitigated, for example through teacher scoring thresholds or reference-answer checks.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces Omni-RRM, a reward model that provides explainable preference judgments across image, video, and audio. It is trained on the newly constructed OmniPreference dataset. The training process consists of two stages: SFT and GRPO. Omni-RRM demonstrates modest improvements in comparison experiments with other baseline reward models and in the best-of-N sample experiments.

### Strengths
1. Compared to existing baselines, this work introduces audio understanding reward ability.

2. A new preference dataset, OmniPreference, is proposed, providing a new resource for training reward model.

### Weaknesses
1. The technical road of this work is similar to previous methods like UnifiedReward-Think [1]. The Previous work distilled 5k image generation reward data for cold-start and achieved unified multimodal CoT reasoning through rejection sampling and GRPO. However, this work performs large-scale distillation of closed-source models across all tasks for SFT before applying GRPO.
Could the authors provide a deeper discussion on the advantages of the training method compared to previous approaches? 

2. The proposed model, Omni-RRM, includes reward capabilities for image, video, and audio understanding, while its text-only reward capabilities are primarily based on generalization. However, other baseline models, such as R1-Reward, likely also possess similar generalization capabilities for text. Why wasn't a comparison made between Omni-RRM and these models specifically for their text-based reward performance?

3. In my opinion, simply having reward capabilities for understanding is not sufficient to claim it as "omni." Compared to previous work [1], which already includes multimodal reward capabilities, Omni-RRM adds audio understanding but lacks reward capabilities for image/video generation. Therefore, I don't believe this constitutes a significant contribution.

4. In the comparison experiments of reward models, this work lacks a comparison with the latest models, such as UnifiedReward-Think[1] and IXC-Reward[2].

5. From the experimental results, it appears that the SFT version of Omni-RRM-7b performs significantly worse than UnifiedReward, which was also trained solely through SFT, particularly on MMRewardBench (-10.8 points) and ShareGPT-Video (-9.1 points). Could the authors provide a deeper analysis to explain this performance gap?

6. Based on Table 1, the data sources for the Omni-Preference dataset are quite limited. For example, in the video task, pairs are constructed using only Qwen2.5-VL-7b/3b, which results in the reward model being trained on a very narrow output distribution of these models. When applying the reward model to assess other models with largely different output distributions from those used in the dataset construction, there may be potential issues with inaccurate judgments. Could the authors provide more analysis of this limitation?

7. Based on Figure 2, the improvement in best-of-N using Omni-RRM is not significant. For instance, on Qwen2.5-Omni-7B, the image and video tasks show only a 0.6 and 0.9 point improvement, respectively.

8. The paper does not explain the motivation behind incorporating image, video, and audio understanding reward training together. Can image and video understanding enhance the performance of audio tasks? If so, could the authors provide a comparison between training with audio data alone and the unified training of all tasks?

[1] Wang, Yibin, et al. "Unified multimodal chain-of-thought reward model through reinforcement fine-tuning." NeurIPS, 2025.


[2] Zang, Yuhang, et al. "Internlm-xcomposer2.5-reward: A simple yet effective multi-modal reward model." ACL (Finding), 2025.

### Questions
Please see Weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Omni-RRM, an open-source, reasoning-driven reward model designed to provide interpretable and multimodal preference judgments across text, image, video, and audio domains. Central to the approach is OmniPreference, a large-scale dataset generated through a fully automated pipeline that contrasts model outputs of different capabilities and augments them with multi-criteria, chain-of-thought rationales from powerful teacher models.

The model is trained in two stages: (1) Supervised fine-tuning (SFT) to teach rationale generation and structure-aware preference reasoning, and (2) Reinforcement learning (RL) to refine judgment accuracy on ambiguous or low-contrast preference pairs. Experimental results show that Omni-RRM achieves state-of-the-art results on video (ShareGPT-V, 80.2%) and audio (Audio-HH, 66.8%) benchmarks, while outperforming existing open-source RMs on image tasks by a 17.7% margin.  Omni-RRM shows improved downstream performance via Best-of-N (BoN) sampling and also improvements on text only benchmarks.

### Strengths
1. Omni preference consturciton framework, specifically strucutred rationale annotation and two stage training pipeline with custom reward score function, and application of this on to a omni modla dataet with text,audio or visal inputa and text outputs. (overally okais novelty in the alorithm but novely in the problem domain of being omni).

2. Rgiorusou expriments, compraiosn woith baselines, analysis, use in BoN setup dodntresm, impact on text only benchmarks, comapriosn with public dataset (only the image compkent)

3. Paper is largely clearlt written apart from some places which ned more clarificaiton

### Weaknesses
1 Is it Novel enough?: The framework’s structured reward formulation and rubric-based scoring is similar to prior works such as the delta learning hypothesis [1], which contrasts strong vs. weak models, and the GRPO pipeline with minor reward adjustments. Beyond the structured reasoning pipeline and rubric grounding, most elements feel incremental, with the key novelty being the application to the Omni multimodal setup.

2. Lack of Human Validation There is no explicit human evaluation or quality control of the reconciled teacher rationales. Without even a small-scale human validation, it’s difficult to confirm that the 41K annotated samples maintain consistent reasoning quality. While downstream benchmark gains (Table 1) indirectly reflect data quality, a more direct validation, human or validated metric-based, would strengthen confidence in the dataset and pipeline.

3. Missing Baseline: A comparison against an RM trained only with SFT (without structured rationales or two-stage fine-tuning) is missing. This would help isolate the contribution of structured rationale and the second training stage, especially for both 3B and 7B variants, like Omini-RM-3B or 7B on SFT only

4. Unconvincing BoN Results:
a)  Improvements in the Best-of-N (BoN) experiments appear marginal. Statistical significance tests (e.g., bootstrap sampling, confidence intervals) would help establish whether the observed gains are meaningful.
b) Since Omni-RRM requires generating rationales and reward scores, the approach likely incurs higher latency. A comparison of runtime or inference cost versus performance improvement would be valuable to gauge practical adoption trade-offs.

5. Generalization Across Model Families
It’s unclear whether the proposed framework generalizes to models with different reasoning priors. Prior work shows that models like Qwen2-VL already exhibit self-reflection and backtracking abilities, whereas LLaMA-based ones often require fine-tuning to develop them [2]. It would be interesting to see if all model families benefit equally from the structured rationale training, or if some already internalize this behavior.

[1] Geng, Scott, et al. "The delta learning hypothesis: Preference tuning on weak data can yield strong gains." arXiv preprint arXiv:2507.06187 (2025).
[2]  Gandhi, Kanishk, et al. "Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars." arXiv preprint arXiv:2503.01307 (2025).

### Questions
1. Rationale Annotation and Filtering

a) Which teacher model(s) are used to generate the rationales?
b) What exact reconciliation and filtering algorithms are applied to merge annotations?
c) When merging diagram or multimodal rationales, averaging scores alone may not ensure correctness. How do you validate merged rationales to prevent propagation of incorrect reasoning from one teacher?

2. Reward Score Design

a) In Appendix A.1, the reward score is defined as 1 – 2*(…)/20. What is the intuition behind this formulation and the constants? Are they empirically derived or theoretically motivated?
b) How is the rubric reward computed to capture both dimension coverage and comparative reasoning?
c) How are weights across components (context, rubric, reasoning quality) determined when forming a single composite score?
d) Why are the five evaluation dimensions set specifically to - Fluency & Coherence, Relevance to the Question and Modality, Accuracy & Completeness, Reasoning Quality, Safety & Ethical Alignment - are these empirically validated or adopted from prior rubrics?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Omni-RRM, an open-source, reasoning-driven multimodal reward model (RM) designed to address the alignment challenges of modern Multimodal Large Language Models (MLLMs) across text, image, video, and audio. Its core contributions are twofold: the Omni-Preference dataset, constructed via a fully automated pipeline that circumvents human annotation by contrasting model pairs and enriching them with detailed, multi-criteria rationales from teacher models; and a progressive training strategy that combines Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) to instill and refine the model's reasoning capabilities. 

Empirical results demonstrate that Omni-RRM achieves state-of-the-art or highly competitive performance on several benchmarks, with substantial gains over its base model and other specialized open-source baselines.

### Strengths
1. Automated, Reasoning-Augmented Data Construction: The fully automated pipeline for the Omni-Preference dataset, which leverages a "capability-gap" approach and teacher model rationales, effectively bypasses the human labeling bottleneck, addressing a critical scalability challenge in creating RLHF-style alignment data (Section 4.1, Table 1).
2. Explicit Reasoning and Transparency: The model moves beyond opaque scalar rewards to generate rich, interpretable, chain-of-thought rationales for its preferences (Section 3). This design significantly enhances trust, interpretability, and debuggability.
3. Strong Empirical Performance: Omni-RRM establishes a new state-of-the-art on key video (80.2% on ShareGPT-V) and audio (66.8% on Audio-HH) benchmarks.
4. Generalization to Text-Only Tasks: The model demonstrates a positive transfer effect, where its multimodally trained reasoning capabilities enhance its performance on standard text-only preference benchmarks, as shown in Figure 3.

### Weaknesses
1. Limited Evaluation on Difficult Cases: Although the data is categorized into "hard" and "easy" pairs (Table 1), the experimental results are not disaggregated by difficulty. This makes it difficult to assess the model's robustness and its ability to adjudicate nuanced, low-contrast comparisons.
2. Gaps in Related Work and Positioning: The paper fails to discuss or benchmark against several highly relevant, recent works on reasoning-augmented reward models, such as "Unified Multimodal Chain-of-Thought Reward Model," "RM-R1: Reward Modeling as Reasoning," and "VR-Thinker." This omission undermines the paper's claimed novelty and fails to properly situate its contributions within the current state of the art.
3.

### Questions
1. How robust is the data generation process to systematic biases or stylistic artifacts from the teacher models? Have you conducted any experiments to measure or mitigate the propagation of such biases?
2. Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning" and "RM-R1: Reward Modeling as Reasoning" propose and analyze reasoning-driven multimodal reward models, but they lack relevant discussion and necessary comparison.

### Soundness
2

### Presentation
3

### Contribution
2
