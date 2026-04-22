# AutoDavis: Automatic and Dynamic Evaluation Protocol of Large Vision-Language Models on Visual Question-Answering

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Large Vision-Language Models (LVLMs) have become essential for advancing the integration of visual and linguistic information. While existing benchmarks have laid a solid foundation for evaluation, they are often static, resource-intensive to build, and limited in adaptability. In comparison, automatic evaluation has shown promise in the textual domain, but the visual modality remains far less explored. To advance this frontier, in this work, we introduce AutoDavis, a first-of-its-kind automatic and dynamic evaluation protocol that enables on-demand benchmarking of LVLMs across specific capability dimensions. AutoDavis leverages text-to-image models to generate relevant image samples and then utilizes LVLMs to orchestrate visual question-answering (VQA) tasks, completing the evaluation process efficiently and flexibly. To ensure data diversity, our framework employs a hierarchical aspect-driven generation process enhanced with semantic graph-based constraints. To safeguard reliability, the framework incorporates a self-validation mechanism to detect and correct errors, along with an error-driven adjustment module to mitigate potential bias. Through an extensive evaluation of 11 popular LVLMs across five demanded user inputs (i.e., evaluation capabilities), the framework shows effectiveness and reliability, offering a new paradigm for dynamic benchmarking of multimodal intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AutoDavis, an automatic and dynamically regenerable evaluation protocol for assessing LVLMs on VQA. The framework allows users to specify capability dimensions and difficulty levels. AutoDavis then employs a LVLM-based examiner coupled with a T2I generator to automatically produce question–image pairs, followed by self-consistency validation and error-driven distractor generation. The system integrates image-free controls and answer position balancing to mitigate textual shortcuts and positional bias.
Experiments cover 5 capability categories × 3 difficulty levels × 11 LVLMs, showing clear performance degradation with increasing difficulty. The paper further demonstrates that multi-examiner ensembles yield more stable rankings, and dynamic regeneration mitigates data leakage from repeated exposure. The paper also provides a theoretical sample–validation bound for controlling evaluation error and explores data re-use for training, showing minor improvements on external benchmarks.

### Strengths
1.	Systematic and operational definition of “dynamic evaluation”
AutoDavis formalizes three key criteria—flexibility, anti-leakage, and visual grounding—and instantiates them with concrete mechanisms. The framework transforms regenerable evaluation from a conceptual goal into a practical, verifiable protocol.
2.	Hierarchical task decomposition with controlled diversity
The authors decompose abilities hierarchically, ability → sub-skill → constrained diversity, and use semantic-graph–constrained description generation to ensure topical coverage and reduce redundancy. Quantitative diversity analysis supports the method’s effectiveness.
3.	Comprehensive experiments with clear separation effects
The benchmark includes 11 LVLMs and reports fine-grained breakdowns by ability, difficulty, and visual evidence. The results convincingly show increasing difficulty correlates with performance degradation, and image-free variants sharply reduce accuracy—confirming reliance on visual reasoning.

### Weaknesses
1.	Evaluator family dependence and circularity risk
The entire loop—question generation, VQA validation, distractor rewriting, and scoring—relies on strong LVLMs from the same or related model families. Although the authors introduce “multi-examiner” diversity, they do not quantify cross-family variance or bias. Without cross-judge robustness, AutoDavis risks overfitting to shared linguistic priors or visual biases.
2.	Reliability of T2I and self-verification
The faithfulness of generated images is only indirectly ensured by the self-check VQA threshold (ζ). Yet VQA self-checkers may share textual priors with the examiner, producing false positives for visually inconsistent images. The paper lacks manual inspection results or sensitivity studies under varied T2I or checker models.
3.	Leakage and regeneration analysis remains limited
The anti-leakage experiment is small-scale and defines “leakage” narrowly. Broader experiments would provide stronger evidence for robustness against contamination.
4.	External validity limited to MMMU correlation
While a Spearman ρ = 0.817 with MMMU is encouraging, validation against additional human-annotated benchmarks (e.g., MMBench, SEED-Bench 2, MMMU-Pro) and fine-grained per-ability consistency would better support generalizability.
5.	Potential ambiguity in error-driven distractor generation
Forcing “plausible wrong” alternatives by relabeling the correct answer can introduce ambiguous or semi-correct options. The paper lacks quantitative measures of question unambiguity or human adjudication consistency after augmentation.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces AUTODAVIS, a dynamic and automated evaluation protocol for Large Vision-Language Models (LVLMs). It addresses limitations of static benchmarks by enabling on-demand test generation, preventing data leakage through dynamic regeneration, and ensuring visual grounding via error-driven option adjustment. Key contributions include: (1) the AUTODAVIS framework with modules for user-oriented aspect generation, guided description creation, self-validated image generation, and test case evaluation; (2) comprehensive experiments demonstrating its effectiveness in generating diverse, reliable assessments across dimensions like basic, spatial, and reasoning understanding, with high human alignment and correlation to existing benchmarks; (3) insights into model capabilities, revealing performance drops with increasing difficulty and systemic weaknesses in fine-grained visual reasoning; and (4) evidence that AUTODAVIS-generated data can enhance model training and generalization. The protocol offers a scalable, cost-effective supplement to static benchmarks, promoting trustworthy LVLM evaluation.

### Strengths
Originality: This work moves beyond creating another static benchmark by proposing a dynamic, on-demand generation system. This effectively addresses known limitations of static benchmarks, such as data leakage and rapid obsolescence

Significance: The work's importance is significant for the sustainable and trustworthy evaluation of LVLMs. It provides a practical solution to critical issues of benchmark staleness and data contamination that plague static datasets.

Clarity: The exposition is clear and well-structured.

### Weaknesses
1. The authors designate GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet as examiner models; however, these models are simultaneously included among the representative models being evaluated. Could this dual role introduce examiner bias and potentially compromise the fairness of the evaluation?

2. Could multiple trials be conducted with humans serving as examiners for the same questions, to observe whether human scoring is consistent or inconsistent with automated scoring?

### Questions
1. Figure 3 on page 7 is missing an overall caption. Figure 5 is not cited or mentioned in the body of the text. In Figure 3(b), the term 'relative change' is not clearly defined.

2. Please provide statistical information about the dynamic benchmark, for example, the average length of the generated questions and the average length of the answer options. Multiple experiments may be conducted to determine either a range or an average."

3. Within the AUTODAVIS pipeline, does generating images at different resolutions have an impact on the evaluation performance?

4. What is the actual visual complexity?

### Soundness
2

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
4

### Summary
The paper targets the limits of static LVLM benchmarks and proposes AUTODAVIS, a system that builds capability-focused tests on demand. It synthesizes images with a text-to-image model and then probes models via VQA prompts, using simple structural rules to keep variety. A self-check and error-driven update step aim to catch mistakes and curb bias. The authors report trials on 11 LVLMs across five user-specified skill areas, showing the setup is practical and fairly stable. The idea is appealing for scale and flexibility, but it leans heavily on generative components, so bias and domain transfer remain concerns; clearer cost and robustness analyses would strengthen the case.

### Strengths
- Proposes an interesting evaluation idea that treats LVLMs as both examiners and test-set generators, yielding a largely model-driven and potentially more scalable pipeline.

- The framework appears solid, producing diverse and accurate images conditioned on user queries, as also the question desination.

- Ablation studies are informative, especially the reported performance gains after integrating the framework.

### Weaknesses
- The motivation is not clear enough. This paper point out that static benchmarks are insufficient at first, it sounds important, but, why must use LVLM itself to solve this question. In other words, what the necessary to use this pipeline to solve this problem? As the description is the introduction, the Dynamic-ME [1] also can solve it.

- The evaluation set of models is dated. Recently released open-source LVLMs available before the ICLR deadline, such as InternVL-3 and Qwen2.5-VL, should be included, but your works only test InternVL-2.5 and Qwen2-VL.

- The analysis is shallow, more explanation should be obtained, such as the deep reason on the bad performance on spatial reasoning.

- While the ablation study indicates that the proposed design influences model outcomes, the paper does not further investigate the underlying causes or compare the effects of different improvement directions. More comprehensive analysis is needed.

[1] Yang, Yue, et al. "Dynamic multimodal evaluation with flexible complexity by vision-language bootstrapping." arXiv preprint arXiv:2410.08695 (2024).

### Questions
- This paper points that these static benchmarks are inefficient, but you conduct the experiment and obtain that high correlation with existing benchmark, this make me confuse. If your works obtain a similar conclusion, what the necessary of your work?

- This work is proposed based that existing static benchmarks are inefficient, so I want to know the difference of performance with the traditional benchmarks and yours, like, if there is one model which is good at others but bad at yours, or contrast.

- Please report results for current LVLMs available (e.g., Intern3-VL, Qwen3-VL).

- This pipeline point that ‘leakage and self-enhancement bias when the same family of models writes and takes the exam’, so I want to know how about the performance with the same family such as Qwen3-series.

### Soundness
2

### Presentation
2

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
This paper proposes AutoDavis, an automatic and dynamic evaluation protocol for large vision-language models (LVLMs). Unlike traditional static benchmarks, AutoDavis can dynamically generate evaluation datasets on-demand using text-to-image synthesis and LVLM-based question–answer generation. The pipeline includes hierarchical aspect generation, semantic graph–guided prompt diversification, self-validation for image–text alignment, and multi-examiner evaluation to reduce bias. It supports five core evaluation aspects—basic, spatial, semantic, reasoning, and atmospheric understanding—and shows strong correlation (Spearman ρ = 0.817) with human-curated benchmarks such as MMMU, suggesting reliability.

### Strengths
-  Benchmarking LVLMs dynamically is an important and emerging challenge as static datasets become saturated and prone to leakage.

- Systematic design: The modular pipeline (aspect generation, semantic graph, self-validation, option adjustment) is thoughtfully structured, and the authors provide theoretical guarantees for diversity and alignment.

- Strong empirical study: Evaluates 11 LVLMs with extensive analysis across difficulty levels, examiner configurations, and bias controls.

- Correlations and validations: Human studies and comparison with MMMU substantiate reliability.

- Practical implications: Demonstrates that AutoDavis can both evaluate and generate synthetic data useful for fine-tuning LVLMs.

### Weaknesses
- Lack of discussion why such new protocol will be adopted by community.

- Weak novelty in methodology: Many core mechanisms (semantic graph diversity, self-validation, error-driven adjustment) are adapted from prior works like TIFA and AutoBencher, with limited algorithmic innovation.

- Lack of quantitative clarity: While AutoDavis claims dynamic flexibility, there is no quantitative analysis comparing generation diversity or cost-efficiency with prior dynamic benchmarks (e.g., MME-Unify, LENS).

- Evaluation reliability: Since LVLMs serve as both examiners and subjects, it remains unclear how much bias persists even with the multi-examiner setup.

- Limited impact of results: The performance gap among models largely mirrors existing benchmarks, suggesting AutoDavis may not yet reveal qualitatively new insights about LVLM capabilities.

- Presentation issue: The paper is lengthy and reads more like a system report; tighter focus on unique contributions would help.

### Questions
How do you ensure examiner–evaluatee independence when many models share similar training corpora?

### Soundness
3

### Presentation
2

### Contribution
3
