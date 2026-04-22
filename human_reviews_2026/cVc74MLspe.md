# Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: A Comprehensive Evaluation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated remarkable multimodal perception capabilities, garnering significant attention. While numerous evaluation studies have emerged, assessing LVLMs both holistically and on specialized tasks, fine-grained image tasks—fundamental to computer vision—remain largely unexplored. To fill this gap, we introduce a comprehensive fine-grained evaluation benchmark, i.e., FG-BMK, comprising 1.01 million questions and 0.28 million images. Our evaluation systematically examines LVLMs from both human-oriented and machine-oriented perspectives, focusing on their semantic recognition and fine-grained feature representation capabilities. Through extensive experiments on twelve representative LVLMs/VLMs, we uncover key findings regarding the influence of training paradigms, modality alignment, perturbation susceptibility, and fine-grained category reasoning on task performance. This work provides critical insights into the limitations of current LVLMs and offers guidance for future data construction and model design in the development of more advanced LVLMs. Our code is open-source and available at https://github.com/SEU-VIPGroup/FG-BMK.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FG-BMK, a comprehensive benchmark designed to evaluate Large Vision-Language Models (LVLMs) on fine-grained image tasks, which remain underexplored compared to generic visual reasoning or captioning tasks. FG-BMK comprises 0.28 million images and 1.01 million automatically generated questions sourced from 12 established fine-grained datasets such as CUB, Stanford Cars, and FGVC Aircraft. The benchmark features two complementary paradigms:

1. Human-oriented evaluation: measures LVLMs’ ability to answer fine-grained visual questions in conversational form (attribute recognition, hierarchical granularity reasoning, and knowledge bias estimation).  
2. Machine-oriented evaluation: measures the discriminability and robustness of visual representations via retrieval and classification tasks.

The study evaluates 12 representative models (GPT-4o, Gemini-2.0-flash, Qwen2.5-VL, InternVL3, LLaVA, DINOv2, etc.) and provides systematic analyses. Main findings include:  
- Contrastive training (e.g., CLIP, DINOv2) yields better fine-grained discriminability than generative or reconstruction paradigms.  
- Text-image alignment can harm fine-grained distinctions, especially when text granularity mismatches image detail.  
- LVLMs are more vulnerable to perturbations in fine-grained domains than generic vision tasks.  
- Scaling model size or data volume brings limited gain compared to data quality and alignment design.  
- LVLMs perform well on appearance description but lag behind domain-specific fine-grained models.

### Strengths
- Proposes a large, reproducible benchmark that systematically covers multiple aspects of fine-grained vision-language understanding.  
- Provides clear empirical findings with actionable insights, notably about alignment granularity and contrastive pretraining.  
- Includes both human-style and feature-based evaluations.
- Reveals interesting findings for LVLM's behaviour in fine-grained tasks.
- Gives insights for how to design and train vision representation encoders in the context of improving fine-grained perception of LVLM

### Weaknesses
- Primarily an evaluation paper; lacks methodological novelty or a new model proposal. I agree that a benchmarking and analysis paper can bring valuable insights to the field. However, since this paper already proposes several small fixes (fine-tuning on balanced fine-grained datasets, improved alignment strategies), could these be integrated into a more systematic approach that contributes to improving MLLMs?

- Another concern is that automatic question generation may limit linguistic diversity and realism compared to human-written prompts. There are only a few chat templates shown in Appendix A.2. Could the use of more natural prompts affect performance?

- Lack of a systematic comparison. Although this paper evaluates many mainstream models, it lacks a clear comparison of their results on FG-BMK (for example, an average of all sub-benchmarks on FG-BMK). A clear comparison would be important for readers to understand the performance and relative ranking of different LVLMs.

- The benchmark is sourced from existing public datasets, which raises concerns that different LVLMs may have been trained on overlapping portions of these datasets. This could make fair performance comparison difficult. Can the authors provide an analysis of this potential train-test overlap issue?

### Questions
See Weakness.

- Additionally, can the authors provide more details on the PGD attack setup for feature perturbation? L∞ PGD requires specifying an upper bound for the perturbation. Do all encoders use the same upper bound?

Overall, this paper delivers insightful analyses of LVLMs on fine-grained visual tasks, with solid and detailed evaluations from both semantic and feature-based perspectives. I recommend acceptance and would consider raising the score if the above concerns are addressed.

### Soundness
3

### Presentation
2

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
This paper introduces FG-BMK, a large-scale benchmark designed to provide a fine-grained evaluation of Large Vision-Language Models (LVLMs) from both human-oriented and machine-oriented perspectives. The benchmark contains 1.01 million questions paired with 0.28 million images, covering 12 fine-grained visual datasets. The human-oriented evaluation focuses on assessing the model’s ability to understand and respond to fine-grained visual queries in conversational contexts. The machine-oriented evaluation measures model performance on two standard fine-grained vision tasks: image retrieval and image recognition.

### Strengths
1. The paper is well structured and easy to follow.

2. This paper attempts to address an important topic, the evaluation of Large Vision-Language Models (LVLMs) on fine-grained visual tasks. The authors propose a benchmark, FG-BMK, by curating questions over 12 existing fine-grained datasets.

3. It provides a large set of pair of questions-images to comprehensely evaluate LVLMs in different relevant tasks, like Attribute Recognition, Hierarchical Granularity Recognition, etc.

### Weaknesses
1. Limited novelty: While the topic is relevant, the paper’s novelty is limited. Several recent benchmarks, such as MERLIM (Villa et al., 2023), HallusionBench (Guan et al., 2023), MMVP (Tong et al., 2024), MMBench (Liu et al., 2024), and AMBER (Wang et al., 2023) have already evaluated LVLMs across similar tasks. The paper does not clearly explain how FG-BMK differs from or improves upon these existing benchmarks.

2. Limited model evaluation: The set of evaluated models is small and outdated. The study omits many recent (like LLaVA-OneVision, LLaVA More, EAGLE 2.5) and commercially competitive LVLMs (like, GPT-4o and Gemini).

### Questions
1. Novelty and differentiation: As mentioned in the first weakness, there are already numerous LVLM benchmarks covering fine-grained evaluation. Could you elaborate on the specific advantages or unique contributions of FG-BMK compared to existing benchmarks such as MERLIM (Villa et al., 2023), HallusionBench (Guan et al., 2023), MMVP (Tong et al., 2024), MMBench (Liu et al., 2024), and AMBER (Wang et al., 2023)?

### Soundness
2

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
This paper introduces FG-BMK, a new, comprehensive benchmark for evaluating Large Vision-Language Models on fine-grained image tasks, an area the authors claim is currently underexplored. The benchmark consists of 1.01 million questions and 0.28 million images, which are sourced from 12 existing, well-established fine-grained datasets. The evaluation is structured into two paradigms, a human-oriented evaluation and a machine-oriented evaluation. Based on extensive experiments on 12 representative LVLMs, the paper reports several key findings. Critically, the authors find that contrastive training paradigms produce superior fine-grained features, and that the vision-language alignment process itself can impair fine-grained discriminability.

### Strengths
1. The paper's greatest strength is the FG-BMK benchmark itself. The dual-paradigm (human-oriented and machine-oriented) evaluation is effective, allowing for a holistic assessment of both conversational understanding and raw feature quality.
2. The paper delivers more than just a leaderboard. It uncovers several novel and important insights.
3. The paper is exceptionally well-written, with clear figures and tables that make the complex results easy to digest.

### Weaknesses
1. The "machine-oriented" evaluation provides the paper's most novel and important insights. However, this evaluation requires access to internal model features, which is impossible for closed-source models. As a result, this entire, crucial part of the benchmark cannot be applied to SOTA models like GPT-4o and Gemini.
2. The paper contains a clear factual error. The Ethics Statement explicitly claims the benchmark "does not cover sensitive areas like medical imaging," but Table 6 in the appendix clearly lists the "SkinCon" dataset, which it explicitly labels as "Dermatology".
3. The abstract's claim of "comprising... 0.28 million images"  could be mildly misleading, implying a new image collection. The core contribution is the 1.01 million questions and the evaluation framework, which are built on 12 existing public datasets.

### Questions
Please respond to the weaknesses I mentioned above.

### Soundness
3

### Presentation
4

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
This paper presents FG-BMK, a large-scale benchmark containing 3.49 million questions and 3.32 million images that systematically evaluates VLMs on fine-grained visual tasks from both human-oriented dialogue and machine-oriented representation perspectives. Across twelve domain-specific datasets, the authors demonstrate that contrastively trained encoders achieve superior subordinate-level discriminability, while generative or reconstruction-based paradigms underperform; that vision–language alignment can degrade fine-grained accuracy when image–text granularity is mismatched; and that current VLMs remain more vulnerable to adversarial perturbations and less effective at attribute-based reasoning than specialised fine-grained models, thereby identifying critical directions for future multimodal model development.

### Strengths
1. Introduction of FG-BMK, the first million-scale benchmark dedicated to fine-grained image understanding, offering paired human-style Q&A and machine-centric retrieval/classification protocols for comprehensive VLM assessment.

2. Extensive empirical evidence that training objectives, modality-alignment strategies, and encoder scaling choices exert measurable, task-specific effects on subordinate category recognition and robustness, with contrastive losses consistently outperforming generative or reconstruction losses.

3.Systematic disclosure of knowledge bias, granularity sensitivity, and perturbation fragility in existing VLMs, coupled with quantitative gaps relative to fine-grained expert models, providing actionable guidance for data curation and architectural refinements toward stronger fine-grained visual reasoning.

### Weaknesses
1. The ground-truth answers of FG-BMK are directly inherited from the original coarse-grained annotations and are paired with template-generated questions; no domain expert relabelling was performed. Consequently, a non-negligible proportion of species-level items contain incorrect negative labels or ambiguous attribute descriptions. A model may therefore be penalized for a prediction that is in fact consistent with expert knowledge, undermining the statistical reliability of the reported accuracy rankings. (e.g., in the Spotlight study of Chihuahuas, the term "Japanese spaniel" is not common knowledge, and different participants gave different answers.)

2. The machine-oriented protocol exclusively reports Top-1 accuracy and mAP, omitting recently proposed robustness and fairness measures. This conservative choice restricts the diagnostic value of the benchmark and may conceal performance disparities that become evident under more lenient or distributional metrics.

### Questions
It's recommended to conduct experiment below to improve the limitations in the weakness session.

1. Expert Relabelling: Commission ornithologists, botanists or other taxonomic specialists to manually verify every species-level true/false and multiple-choice pair, ensuring that labels reflect up-to-date consensus and that linguistic ambiguities are resolved.

2. Expanded Metric Suite: Complement Top-1 accuracy with Top-5 and Top-10 scores, and integrate robustness indicators such as corruption-accuracy curves or fairness-aware metrics (e.g., worst-group accuracy) to provide a richer, more nuanced assessment of model behaviour.

### Soundness
3

### Presentation
3

### Contribution
3
