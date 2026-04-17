# M3Kang: Evaluating Multilingual Multimodal Mathematical Reasoning in Vision-Language Models

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Despite state-of-the-art vision-language models (VLMs) have demonstrated strong reasoning capabilities, their performance in multilingual mathematical reasoning remains underexplored, particularly when compared to human performance. To bridge this gap, we introduce M3Kang, the first massively multilingual, multimodal mathematical reasoning dataset for VLMs. It is derived from the Kangaroo Math Competition, the world’s largest mathematics contest, which annually engages over six million participants under the age of 18 across more than 90 countries. M3Kang includes 1,747 unique multiple-choice problems organized by grade-level difficulty, with translations into 108 culturally diverse languages, some of them including diagrams essential for solving them. Using this dataset, we conduct extensive benchmarking on both closed- and open-source SOTA models. We observe that, despite recent advances, models still struggle with basic math and diagram-based reasoning, with performance scaling with language presence and model size, but not with grade level. We also find that multilingual techniques can be effectively extended to the multimodal setting, resulting in significant improvements over baseline approaches. Our analysis also incorporates performance data from over 68,000 students, enabling direct comparison with human performance. We are open-sourcing M3Kang, including the English-only subset M2Kang, along with the framework and codebase used to construct the dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes M3Kang, the first highly multilingual, graphically-based mathematical reasoning assessment dataset. This dataset draws 1,789 questions from the Kangaroo competition and expands to 15 languages, supported by an open automated translation and quality control pipeline. Benchmark results show that models significantly underperform text-only problems on questions containing diagrams. Cross-lingual performance is strongly correlated with internet coverage (more pronounced for smaller models). A simple "text + English translation parallel prompt" (MTR) approach is most effective in multimodal scenarios, significantly narrowing language gaps.

### Strengths
1. This paper introduces a multilingual, multimodal math reasoning benchmark across 15 languages (with an English subset), enabling fair cross-lingual comparison on a unified problem pool.
2. This paper provides an open, rigorous translation and quality-control pipeline (reference-free backtranslation metrics, LLM-as-judge) that preserves layout and is easily reusable.
3. This paper delivers comprehensive evaluations of open/closed VLMs and multilingual techniques, revealing a strong visual reasoning gap and establishing MTR as the most effective method; it also includes human comparisons with 68k students.

### Weaknesses
1. Dataset and code links are inaccessible
2. The dataset's statistical metrics are unclear. How many multimodal questions are there and how many are plain text?
3. Can you provide a comparison with existing multilingual and multimodal datasets, perhaps in a table format?
4. Regarding **sec. 4.4 Text-Only vs. FIGURE PROBLEMS**. The article observes that the accuracy rate for plain text questions is higher than for multimodal questions. Could this be because multimodal questions are inherently more difficult? I suggest adding a "comparative experiment at the same difficulty level" to make the argument more convincing.

### Questions
see weaknesses

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
Build a highly multilingual, multimodal benchmark (M3Kang) from Kangaroo Math problems; create an automated, quality-controlled translation pipeline; benchmark SOTA VLMs; test multilingual inference techniques in multimodal settings; compare to human performance.

### Strengths
- A multilingual and multimodal math benchmark with a reproducible pipeline.
- This work offers a scalable data translation pipeline.
- Comprehensive benchmarking across open and closed models.

### Weaknesses
- Reliance on backtranslation may systematically disadvantage low-resource languages
- Cross-language fairness relies on filtered subsets, limited statistical testing of comparability.
- Some models (Gemma) perform below chance; analysis of why (prompting, vision adapters) is shallow.

### Questions
- Provide exact prompts per language and the LLM-as-judge criteria and models used.
- Clarify licensing and permissions for dataset redistribution.
- Why not include Chinese?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces M3Kang, a multilingual and multimodal mathematical reasoning benchmark. The dataset includes both text-only and diagram-based questions organized by grade-level difficulty. The authors developed an automated translation pipeline with backtranslation-based quality control. 
Extensive benchmarking of open- and closed-source VLMs reveals key findings: models struggle with basic math and diagram-based reasoning, performance correlates with language Internet presence, Gemini-2.5-Pro leads among closed models. Additionally, direct comparison with student participants shows no significant correlation between VLM and human reasoning patterns, highlighting fundamental differences in problem-solving approaches.

### Strengths
1. M3Kang units multilingual, multimodal, and mathematical reasoning, enabling rigorous evaluation of VLMs.
2. The study benchmarks a diverse set of models, compares text-only vs. diagram-based performance, tests multilingual techniques, and includes human baselines, offering insights into VLM capabilities and limitations.
3. By leveraging real-world competition data and student performance, the benchmark has direct implications for educational AI development and multilingual model optimization.

### Weaknesses
1. The automated translation pipeline may introduce uneven quality across languages, particularly low-resource ones, and human translation (though resource-intensive) is not explored as a refinement.
2.  Without detailed classification of problem types (e.g., geometry, arithmetic, logical reasoning), it is difficult to pinpoint specific reasoning components where VLMs fail most frequently.

### Questions
1. Given the translation quality disparities across resource levels, have you analyzed specific error types in low-resource languages, and how might these errors confound model performance evaluations?
2. The study finds no significant correlation between VLM and human reasoning—do you hypothesize that this stems from differences in visual processing, mathematical intuition, or other factors, and how might future benchmarks better align with human problem-solving contexts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents M3Kang, a multilingual and multimodal mathematical reasoning benchmark derived from the Kangaroo Math Competition. The dataset contains 1,789 unique multiple-choice problems across 15 languages, spanning both text-only and figure-based questions. It provides a unified evaluation framework for Vision-Language Models (VLMs) in multilingual mathematical reasoning, with comparisons to over human participants. The authors design an automated three-stage pipeline: (1) extracting and cleaning Catalan problems, (2) translating them to English (M2Kang), and (3) extending to 15 languages with backtranslation-based quality filtering. Experiments benchmark major open and closed models.

### Strengths
(1) M3Kang fills an important gap at the intersection of multilingual, multimodal, and mathematical reasoning. Prior datasets have addressed these dimensions separately; this benchmark enables joint evaluation.

(2) The authors test about 10 VLMs, analyze correlations between accuracy and language Internet presence, compare text-only vs. figure-based questions, and benchmark multilingual reasoning methods. The analysis is thorough and supported by clear figures.

(3) Using performance data from 68,000 students allows a rare and informative human–AI comparison.

### Weaknesses
(1) Section 2 omits key multilingual multimodal datasets such as EXAMS-V (ACL 2024), M4U (2024), and M3Exam (NeurIPS 2024), mentioning only M5 (Schneider & Sitaram 2024) without detailed comparison. A table contrasting coverage, modality, and translation strategy would strengthen the contribution claim.

(2) The dataset originates from Catalan, chosen for data availability rather than linguistic suitability. The paper does not analyze potential bias or LLM performance limits in Catalan processing.

(3) Only the Catalan→English stage includes manual correction; multilingual translations rely solely on automatic quality metrics. A small human audit (even 100 samples) would help substantiate reliability.

### Questions
In the experiments comparing text-only and figure-based problems, I did not receive the full methodological details. Such comparisons should ideally be conducted on the same set of questions, where the figures in the original problems are converted into equivalent textual descriptions for the text-only version. Please clarify the full details.

### Soundness
3

### Presentation
3

### Contribution
2
