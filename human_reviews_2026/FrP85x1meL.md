# HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024–2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with multiple golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight the performance gap between open-source models and top students, the strong reasoning abilities of closed-source models, and the remaining room for improvement. HiPhO, a human-aligned Olympiad benchmark for multimodal physical reasoning, is open-source at https://anonymous.4open.science/r/HiPhO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces HiPhO, a benchmark for evaluating MLLMs on recent high-school physics Olympiad exams. It emphasizes: (1) up-to-date coverage of 13 Olympiad exams; (2) human-aligned evaluation that combines answer-level checks with step-level scoring using official marking schemes; and (3) direct comparison with human contestants via medal thresholds. Large-scale experiments on 30 models show closed-source reasoning MLLMs frequently reach gold-level scores but still trail top human contestants, while open-source MLLMs generally remain bronze-level; strong open-source LLMs (text-only) can earn multiple golds on text-dominant tasks.

### Strengths
1. This benchmark covers 13 Olympiads, 5 different fields, and 4 modality types. This is the most comprehensive benchmark for physics olympic competition.

2. Human-aligned evaluation with official marking schemes and step-level score is more faithful than answer-only accuracy.

3. This paper demonstrates insightful conclusions, including modality difficulty, field-wise gaps, and visual challenges.

4. The writing is good and easy to follow.

### Weaknesses
1. This benchmark does not evaluate the training data contamination of the models. It is highly likely that closed-source MLLMs have already been trained on these competition problems, since they are publicly available. Providing contamination results would better support the credibility and validity of this benchmark.

2. Although the authors emphasize the modality/field gaps, they could further include fine-grained failure modes (e.g., dimensional analysis lapses, algebraic slips, diagram parsing errors, data-reading inaccuracies) with concrete examples and per-type confusion patterns to better guide method development. If a subset of data were sampled for manual error analysis, it could more deeply reveal the challenges specific to this domain.

### Questions
No

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HiPhO, a new benchmark designed to evaluate the physical reasoning capabilities of large language models (LLMs) and multimodal large language models (MLLMs) using high school physics Olympiad problems. The benchmark's main contributions are threefold: (1) a comprehensive and up-to-date dataset of 13 international and regional Olympiad exams from 2024–2025, featuring mixed-modality problems; (2) a professional, human-aligned evaluation methodology that uses official marking schemes for both final-answer and step-level grading; and (3) a direct comparison with human contestants by mapping model scores to official gold, silver, and bronze medal thresholds. The authors evaluate 30 state-of-the-art models, finding that top closed-source models achieve gold-level performance in many exams but still fall short of the absolute top human scores. The results also highlight a significant performance gap between leading models and the broader field of open-source models, particularly open-source MLLMs.

### Strengths
*   **Originality & Significance:** The most significant contribution of this work is its holistic, human-aligned evaluation framework. While other physics benchmarks exist, HiPhO is the first to systematically benchmark models against real-world human performance standards (medal cutoffs) from recent competitions. 

*   **Quality:** The quality of the benchmark and the evaluation is high. 

*   **Clarity:** The paper is written with clarity. The motivation is well-established, the contributions are clearly articulated, and the structure is logical and easy to follow.

### Weaknesses
*   **Reliance on a Proprietary Grader Model:** The step-level evaluation, a key contribution, relies entirely on API calls to Gemini-2.5-Flash. While the authors compellingly argue for its superiority over GPT-4o, this introduces a dependency on a closed-source model that is subject to change, deprecation, or cost fluctuations. This presents a long-term risk to the benchmark's reproducibility and accessibility. While a difficult problem to solve, the authors could strengthen the paper by discussing this limitation and suggesting future work toward creating a fully open-source, fine-tuned grader to accompany the benchmark.

*   **Anecdotal Evidence for Grader Choice:** Table 7, which provides crucial evidence for the choice of grader, is based on a "single inference run example." For a claim as central as the quality of the evaluation metric, this is anecdotal. A more robust analysis would compare the grader models' outputs to human expert scores across a larger sample of problems and multiple runs to provide statistical evidence (e.g., correlation, mean absolute error) of the grader's reliability.

### Questions
1.  **On the Impact of Step-Level Grading:** Your evaluation metric is `max(answer-level score, step-level score)`. This is a sensible choice. Could you provide a brief quantitative analysis of how frequently the step-level score was the decisive factor (i.e., when a model got the final answer wrong but still received partial credit for correct steps)? This would provide powerful evidence for the necessity of your more fine-grained evaluation method.

2.  **On Medal Threshold Determination:** For Olympiads with both theory and experimental parts (IPhO, EuPhO), you set the medal threshold based on the "lowest theoretical exam score" of human medalists. This seems reasonable, but could it be skewed by a contestant who scored extremely high on the experimental exam and just high enough on the theory exam to secure a medal? How sensitive are the medal thresholds to such outliers, and does this methodology create a robust and representative performance bar?

3.  **On Future Directions for Experimental Physics:** You correctly identify the inability to perform experiments as a key limitation. Beyond simply stating this, could you elaborate on what you believe are the most critical research steps needed to bridge this "embodiment" gap? For example, would this involve creating simulated physics environments, developing agents capable of active experimentation, or something else entirely? A more detailed perspective would add valuable depth to the future work discussion.

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
4

### Summary
This paper introduces HiPhO, a benchmark for evaluating LLMs and MLLMs on recent high-school physics Olympiad problems. It compiles 13 exams from 2024–2025, covering both text and diagram-based questions, and applies official marking schemes for fine-grained, step-level grading. Model scores are further mapped to Olympiad medal thresholds, enabling direct comparison with human contestants. Experiments on 30 state-of-the-art models show that closed-source reasoning models reach near–gold-level performance, while most open-source MLLMs remain around bronze. The benchmark provides a systematic and human-aligned testbed for multimodal physical reasoning, though its contribution is primarily evaluative rather than methodological.

### Strengths
1. The benchmark includes recent Olympiad exams from 2024–2025, ensuring up-to-date coverage.
2. The dataset construction is technically thorough, with careful extraction, verification, and alignment to official marking schemes.
3. The multimodal categorization is well defined, distinguishing text-only, illustration, variable-based, and data-figure problems for fine-grained evaluation.

### Weaknesses
1. The paper primarily serves as a benchmark contribution, offering extensive evaluations but limited analytical depth. While the dataset and results are well organized, the work does not provide deeper investigation into model behavior or new insights into the reasoning mechanisms of LLMs, resulting in limited conceptual originality.
2. The introduction of the Mean Normalized Score (MNS) lacks sufficient explanation. The paper briefly presents the metric without clarifying its motivation, interpretation, or comparison to other normalization choices. As a result, it remains unclear how MNS meaningfully captures modality-specific reasoning performance beyond simple score rescaling.
3. While the paper acknowledges that experimental and embodied reasoning tasks are excluded, it still frames HiPhO as a benchmark for “human-level physical reasoning.” This phrasing may overstate the benchmark’s scope, since the evaluated skills are limited to theoretical problem solving rather than full-spectrum physical reasoning.
4. Although the paper introduces step-level scoring based on official marking schemes, it does not include any corresponding step-level analysis or discussion. Without examining how models perform across different reasoning steps, the potential advantages of this fine-grained evaluation are underutilized.

### Questions
1. What is the intended motivation behind introducing the Mean Normalized Score (MNS), and how should it be interpreted relative to standard scoring metrics?
2. Given that experimental and embodied reasoning tasks are excluded, would the authors consider refining how “human-level physical reasoning” is framed in the paper?
3. Have the authors analyzed model performance at the step level to identify specific reasoning weaknesses or error patterns? If not, could such analysis provide deeper insights into how models approach multi-step physics problems?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces HiPhO, a human-aligned benchmark for high-school Physics Olympiads that aggregates 13 up-to-date 2024–2025 exams across international and regional competitions, evaluates with answer- and step-level scoring aligned to official marking schemes, and enables direct model–human comparison via exam scores and medal thresholds. A large-scale study over 30 SOTA (M)LLMs shows closed-source reasoning MLLMs frequently reach gold level yet still trail the best human contestants; open-source MLLMs mostly sit at bronze–silver, while several open-source LLMs (text-only) achieve multiple golds. Analyses reveal substantial drops on diagram-heavy items (variable/data figures) and consistent difficulty in Optics relative to other fields.

### Strengths
- There is a clear data construction and verification pipeline 
- Large-scale evaluation across 30 (M)LLMs with per-modality and field analysis
- The paper is easy to follow. Dataset statistics, thresholds, and score distributions are explicit.
- The mixed modality taxonomy exposes where (M)LLMs reasoning falters.

### Weaknesses
- Given the recency (2024–2025 exams), many models might have trained on or memorized some content. The paper mentions up-to-date coverage but does not provide data leakage / contamination checks.
- Taking the maximum of answer- and step-level scores can inflate credit if the grader is permissive at either level. Consider reporting both components and their correlation, plus a conservative aggregate (e.g., min or weighted combination) as a sensitivity check.

### Questions
- The paper reports model performance in terms of Olympiad medal tiers (gold/silver/bronze), but there isn't a clear justification for **why** this human-referenced scale is necessary or preferable to standard metrics. Could the authors explain the decision-making or evaluation scenarios where medal-level reporting adds value beyond accuracy scores? Do the medal tiers correlate with deployment-relevant reliability? 
- The paper claims that HiPhO is “the first benchmark to compare model performance with medal cutoffs and contestant scores” and introduces a medal-table ranking “just like an Olympiad.” However, this paper (https://arxiv.org/pdf/2406.16772) already proposed an Olympic medal table approach to rank AI models across subjects in mid-2024. Could the authors clarify how HiPhO’s contribution differs from OlympicArena’s earlier medal-table methodology?
- What is the inter-rater reliability between human experts and the automatic step-scorer across multiple exams? How did the authors ensure that using a strong judge model would not impose errors?

### Soundness
3

### Presentation
3

### Contribution
2
