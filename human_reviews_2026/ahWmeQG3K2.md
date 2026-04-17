# EmotionHallucer: Evaluating Emotion Hallucinations in Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6, 4

## Abstract
Emotion understanding is a critical yet challenging task. 
Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced their capabilities in this area. However, MLLMs often suffer from ``hallucinations'', generating irrelevant or nonsensical content.
To the best of our knowledge, and despite the importance of this issue, there has been no dedicated effort to evaluate emotion-related hallucinations in MLLMs.
In this work, we introduce \textbf{EmotionHallucer}, the first benchmark for detecting and analyzing emotion hallucinations in MLLMs. 
Unlike humans, whose emotion understanding stems from the interplay of biology and social learning, MLLMs rely solely on data-driven learning and lack innate emotional instincts. 
Fortunately, emotion psychology provides a solid foundation of knowledge about human emotions.
Building on this knowledge, we assess emotion hallucinations from two perspectives: emotion psychology knowledge and realworld multimodal perception. 
To support robust evaluation, we utilize an adversarial binary question–answer (QA) framework, which employs carefully crafted basic and hallucinated pairs to assess the emotion hallucination tendencies of MLLMs.
By evaluating 41 LLMs and MLLMs on EmotionHallucer, we find that:
(1) most current models exhibit substantial issues with emotion hallucinations;
(2) closed-source models outperform open-source models in detecting emotion hallucinations, and reasoning capability provides additional advantages;
and (3) existing models perform better in emotion psychology knowledge than in multimodal emotion perception.
As a byproduct, these findings inspire us to propose the \textbf{PEP-MEK} framework, which yields an average improvement of 9.90\% in emotion hallucination detection across selected models.
Resources will be available on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces EmotionHallucer, the first benchmark designed to evaluate emotion hallucinations in MLLMs. The benchmark assesses two complementary aspects, emotion psychology knowledge and multimodal emotion perception. Evaluating 41 models using an adversarial binary QA framework, the paper reports three main findings and propose PEP-MEK, a framework that integrates modality-specific and emotional reasoning to mitigate emotion hallucination. PEP-MEK achieves an average 9.9% improvement across selected models. Therefore, EmotionHallucer provides a benchmark and valuable insights for advancing emotionally reliable MLLMs.

### Strengths
1.	The paper introduces the first study of emotion hallucinations in MLLMs, addressing a crucial yet previously unexplored aspect of emotion understanding.

2.	EmotionHallucer is designed based on emotion psychology and real-world emotion perception, spanning four modalities and multiple evaluation settings. The adversarial QA framework provides a controlled protocolto assess emotional reasoning errors.

3.	The benchmark evaluates a large number of models, providing comprehensive insights into the current state of MLLMs. 

4.	PEP-MEK is compatible with both open- and closed-source models through standard APIs, offering a plug-and-play approach for emotion hallucination mitigation.

5.	The paper is well-structured, easy to follow.

### Weaknesses
1.	While the adversarial binary QA framework provides strong objectivity and control, and the authors have additionally performed consistency checks between binary and open-ended results, this setting may still not fully capture open-ended emotion hallucinations in real-world generative scenarios. It would be valuable to further discuss how this limitation could be addressed, and what directions the authors plan to explore in future work.

2.	The benchmark primarily focuses on English-language. Although some data sources contain diverse cultural content, explicitly incorporating cross-cultural emotion understanding tasks would further enhance its generalizability.

3.	While this trade-off is reasonable given the improvement in reliability, a more detailed discussion on the efficiency–accuracy balance and potential optimization strategies would further strengthen the work.

### Questions
See Weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces EmotionHallucer, a novel benchmark designed to evaluate emotion-related hallucinations in multimodal large language models (MLLMs). The benchmark spans four modalities (text, image, audio, video) and is organized around two main dimensions: emotion psychology knowledge and multimodal emotion perception. The authors also propose PEP-MEK, a reasoning-enhanced framework aimed at mitigating emotion hallucinations. Extensive experiments on 41 LLMs and MLLMs reveal widespread emotion hallucination issues, particularly in multimodal perception tasks, and demonstrate the effectiveness of PEP-MEK in improving model robustness.

### Strengths
- Novel Benchmark: EmotionHallucer is the first comprehensive benchmark targeting emotion hallucinations, with a well-designed adversarial evaluation protocol.
- Multimodal Coverage: The benchmark spans text, image, audio, and video, enabling a holistic assessment of MLLMs’ emotion understanding capabilities.
- Large-Scale Evaluation: Experiments on 41 models offer broad insights into current limitations and trends.
- Practical Mitigation Framework: PEP-MEK demonstrates consistent improvements across models and modalities, offering a simple yet effective approach to reducing emotion hallucinations.

### Weaknesses
- Limited Cross-Lingual and Cultural Analysis: The benchmark is limited to English, and there is no discussion of how cultural or linguistic differences might affect emotion hallucination patterns. This limits the generalizability of the findings.
- Superficial Error Analysis: While the paper reports performance drops in open-set and multimodal settings, it does not deeply investigate the root causes of failures (e.g., which types of cues are most often misinterpreted).
- Benchmark Design Limitations: The binary QA format, while useful for controlled evaluation, may not fully capture the complexity of open-ended emotion understanding in real-world scenarios.
- Lack of Human Baseline: The absence of human performance comparison makes it difficult to gauge the practical significance of the model results.

### Questions
- Have the authors considered extending EmotionHallucer to include non-English or cross-cultural emotional expressions? If so, what challenges do they anticipate?
- Could the authors provide more detailed analysis or examples of cases where PEP-MEK fails? Understanding its limitations could help guide future improvements.
- How might the benchmark be adapted to support more open-ended emotion generation or reasoning tasks, beyond binary QA?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EmotionHallucer introduces the first systematic benchmark for evaluating emotion-related hallucinations in multimodal large language models (MLLMs). The benchmark is grounded in emotion psychology and real-world multimodal perception and uses adversarial binary QA pairs (basic vs. hallucinated versions) across modalities (text, image, audio, video). The authors evaluate 41 MLLMs (open- and closed-source) on multiple subtests (e.g., perception-level, psychology/knowledge-level, reasoning-result), report quantitative metrics (including bias/FP measures and separate accuracy for basic vs. hallucinated items), and analyze model behaviors. They find that many models remain vulnerable to emotion hallucinations, that closed-source models often outperform open-source ones, and that models are typically better at explicit emotion-knowledge tasks than at grounded multimodal perception and inference. To improve detection, the paper proposes PEP-MEK, a multimodal+emotion-knowledge augmentation framework that boosts hallucination detection and is evaluated via ablations. The paper provides dataset construction, annotation procedures, limitations, and plans to release resources on GitHub.

### Strengths
### A novel benchmark (EmotionHallucer) specifically targeting emotion hallucinations in MLLMs:
- Covers multiple modalities and multiple diagnostic levels (perception, emotion knowledge, reasoning results).
- Uses adversarially constructed basic vs. hallucinated QA pairs to probe hallucination propensity.
### Large-scale empirical evaluation and analysis:
- Systematic evaluation of 41 MLLMs (both open- and closed-source) with detailed metrics (Pct. Diff, FP Ratio, separate Basic vs. Hallucinated accuracy, overall scores).
- Insights showing systematic weaknesses (e.g., multimodal perception and reasoning produce more hallucinations; closed-source models typically fare better).
### Proposed mitigation/analysis method (PEP-MEK) and ablation studies:
- PEP-MEK integrates psychology-grounded emotion knowledge and perceptual cues to improve hallucination detection.
- Demonstrated consistent improvements across models and includes ablation studies showing the contribution of emotion knowledge and other components.

### Weaknesses
### Annotation noise and scope limited to English and certain datasets:
- The benchmark relies on human annotation (e.g., creating hallucinated variants), admitting annotation noise.
- The dataset is English-only and does not address cross-lingual or cultural variability in emotional expression.
### Partial exploration of root causes:
- While the paper documents hallucination phenomena and correlates them with modality and model class, it does not deeply investigate underlying causes (e.g., pretraining biases, modality misalignment, lack of emotion-specific supervision) or provide mechanistic explanations.
### Separation of evaluation axes and incomplete real-world integration:
- Emotion understanding and hallucination detection are treated separately, whereas practical systems need integrated capabilities (joint perception, inference, and hallucination-awareness).
- Temporal and long-form audio/video integration remain challenging and less explored; the benchmark and methods may not fully capture these complex real-world scenarios.

### Questions
SEE WEAKNESS

### Soundness
2

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
The paper introduces EmotionHallucer, a benchmark to detect and analyze emotion-related hallucinations in multimodal LLMs. It evaluates two axes: 
- Emotion psychology knowledge (theory, definitions, empirical findings) 
- Real-world multimodal perception (category, intensity, reasoning cues/results) across text, image, audio, and video. 

The benchmark uses adversarial binary QA pairs—a “basic” item and a matched “hallucinated” item—and counts a prediction as correct only if the model answers both correctly, reducing prompt/length confounds seen in captioning metrics and self-evaluation bias.

### Strengths
This is the first benchmark that is dedicated to emotion hallucinations, spanning both psychology knowledge and multimodal perception; prior hallucination suites are general-purpose. The seven subcategories (theory/definition/finding; category/intensity/reasoning cue/reasoning result) make the construct very concrete. And, the adversarial paired QA design (basic vs hallucinated) is what I call a neat, low-variance way to test detection of hallucination, beyond typical caption/LLM-judge setups. 

The paper is well structured, with clear logic, a well-defined task taxonomy, examples, and an easy-to-follow pipeline; the appendices document the collection/annotation and PEP-MEK details; ethics and reproducibility statements are included. 

Non-trivial scale and coverage, with broad evaluation and clear metrics, which is very nice.

### Weaknesses
1. Adversarial pair construction & QA artifacts. The process risks introducing superficial cues between the basic and “hallucinated” versions. Report inter-annotator agreement, pair-level quality controls, and checks against annotation artifacts (e.g., spurious lexical markers).

2. Latency/compute overhead and failure cases are not quantified. A wall-clock and token-cost-wise comparison is needed here, along with ablations for each PEP-MEK component and per-subcategory gains.

### Questions
1. Again, I would like to know a wall-clock and token-cost-wise comparison, along with ablations for each PEP-MEK component and per-subcategory gains.

2. Bias balancing details. The authors stated that the yes/no is balanced. Can you share the exact balance per subcategory and modality, and how you prevented position or wording biases between paired items? 

3. Open-ended evaluation. Beyond the pilot LLM-judge setup, do you plan a human-rated open-ended benchmark slice to validate the binary proxy and reduce judge-model bias?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EmotionHallucer, the first benchmark designed to evaluate emotional hallucinations in multimodal large language models (MLLMs). The findings indicate that most current models exhibit significant issues with emotional understanding, particularly concerning hallucinations. By introducing the PEP-MEK framework, the authors demonstrate an average performance improvement of 9.90% in emotion hallucination detection. The study draws from emotion psychology knowledge and real-world multimodal perception, providing a comprehensive evaluation perspective. Overall, the paper contributes valuable tools and directions for future research in the field of emotional understanding.

### Strengths
This study fills a critical gap in the evaluation of emotional hallucinations, offering the first benchmark tailored for MLLM emotional understanding. The introduction of the PEP-MEK framework shows significant effectiveness, enhancing model performance in hallucination detection. The authors provide robust experimental data and statistical evidence to support their conclusions, increasing the paper's credibility. The research methodology integrates insights from emotion psychology, ensuring the scientific validity of the assessments. Additionally, the use of diverse multimodal data sources adds practical relevance to the findings.

### Weaknesses
Language Limitation: The study is restricted to English, failing to account for cross-linguistic and cross-cultural variations in emotional expression.
Complex Definitions: The definitions and classifications of emotional hallucinations may be overly intricate, potentially leading to ambiguity in the evaluation process.
Suboptimal Performance: Model performance in processing multimodal data, particularly in audio and video emotional understanding, remains inadequate.
Result Stability: The stability and reliability of certain experimental results need further validation to ensure consistency.
Real-World Reflection: The assessment of existing models may not accurately reflect their performance variations in practical applications.

### Questions
In defining emotional hallucinations, how can researchers effectively balance scientific rigor with interpretability?

Is the current benchmark adaptable to different types of multimodal models to ensure consistency in evaluation?

Can the PEP-MEK framework be further optimized for additional emotional understanding tasks beyond those currently evaluated?

### Soundness
2

### Presentation
2

### Contribution
2
