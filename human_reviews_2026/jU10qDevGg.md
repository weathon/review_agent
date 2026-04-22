# U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Ultrasound is a widely-used imaging modality critical to global healthcare, yet its interpretation remains challenging due to its varying image quality on operators, noises, and anatomical structures. Although large vision-language models (LVLMs) have demonstrated impressive multimodal capabilities across natural and medical domains, their performance on ultrasound remains largely unexplored. We introduce U2-BENCH, the first comprehensive benchmark to evaluate LVLMs on ultrasound understanding across classification, detection, regression, and text generation tasks. U2-BENCH aggregates 7,241 cases spanning 15 anatomical regions and defines 8 clinically inspired tasks, such as diagnosis, view recognition, lesion localization, clinical value estimation, and report generation, across 50 ultrasound application scenarios. We evaluate 23 state-of-the-art LVLMs, both open- and closed-source, general-purpose and medical-specific. Our results reveal strong performance on image-level classification, but persistent challenges in spatial reasoning and clinical language generation. U2-BENCH establishes a rigorous and unified testbed to assess and accelerate LVLM research in the uniquely multimodal domain of medical ultrasound imaging.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces U2-BENCH, a benchmark for evaluating LVLMs on ultrasound. Its core contributions are a public dataset (7,241 cases spanning 15 anatomical regions), a systematic evaluation of 20 LVLMs, and important observations regarding model capabilities and limitations in this domain.

### Strengths
1. The main strength of this work lies in its effort to create and release U2-BENCH for a specialized domain, which may serve as a resource for the community.
2. The evaluation of 20 diverse LVLMs provides a preliminary landscape of current capabilities, offering a baseline for comparison.

### Weaknesses
1. The benchmark's scale and imbalance undermine its reliability. With only 7,241 cases thinly spread across 15 anatomies and 8 tasks, the effective sample size for many evaluations is small.
2. The design of certain tasks is problematic. For instance, reducing complex spatial tasks like KD / LL to a 9-class classification problem is a significant oversimplification.
3. The paper primarily presents a performance leaderboard but the key findings are largely intuitive. The analysis remains on the surface, which severely limits the paper's contribution beyond a mere performance report.

### Questions
1. I wonder the time cost for data collection and curation, and annotation. And how scalable is this process for future expansion?
2. The current findings are intuitive. Could you provide a deeper analysis to uncover the underlying reasons for LVLMs' failures, which would significantly elevate the contribution?
3. For KD/LL tasks, did you experiment with more precise evaluation methods (e.g., bounding box coordinates, point regression) to validate if the coarse-grained task is a reliable proxy for fine-grained spatial reasoning?
4. Beyond the U2-Score, what actionable guidance does the benchmark provide for model developers to improve performance on ultrasound tasks?

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
This manuscript presents U2-BENCH, a comprehensive benchmark for large VLM ultrasound understanding on eight clinically-inspired classification, detection, regression and text generation tasks, with 20 SOTA LVLMs evaluated on the benchmark.

### Strengths
- (Quality) Application of a large range of existing VLMs to the benchmark

 - (Clarity) Empirical justification for weighing of tasks

 - (Significance) Extensive coverage of ultrasound understanding datasets, and unification into a single comprehensive benchmark

### Weaknesses
- Segmentation task underrepresented due to unification of ground truth to bounding boxes and predefined spatial localization

 - Preprocessing of video data may limit analysis potential

### Questions
1. For Figure 2, the statement that "...The length of the bar reflects the number of samples for each anatomy region" may be slightly confusing, since it is not immediately clear that this refers to the blue bar. If so, this might be clarified in the caption.

2. In Figure 3, language unification is stated, but its implementation does not appear to be discussed. This appears relevant since the only dataset used for report generation (RG), the Chinese Ultrasound Report Dataset, appears to be in Mandarin.

3. In Section 3.2, format unification was stated to be performed on ultrasound scans by converting to a uniform image format. It might be briefly stated as to whether any conversion loss (either in image quality due to compression, or image dimensions due to cropping etc.) was involved.

4. In Section 3.2, it is stated that a small number of representative frames were sampled, for video sequences. It might be clarified as to how the sampling was performed, since this would appear to restrict video analysis scope.

5. In Section 3.2, it is stated that segmentation masks are converted to bounding boxes. However, this would appear to imply a loss of granularity (since the pixel-level annotations are simplified), and a potential loss of individual objects (since bounding boxes may overlap for originally-distinct irregular objects). This limitation might be considered if relevant.

6. In Section 3.2, the reference to the ablation study is missing.

7. In Section 4.1, it is stated that detection tasks were converted to position classification tasks, and utilized accuracy as the metric. This appears to discount segmentation models/tasks unnecessarily, despite their clinical utility.

8. In Section 4.2, individual performance metrics may be expanded from relevant literature - for example, while accuracy is the main metric for classification and detection tasks, AUROC etc. may be considered for inclusion.

9. In Section 4.3 (or the Appendix), the implementation for Random Guessing might be described.

10. In Appendix B, it is stated that all data was either publicly released with appropriate usage permissions or obtained through official licensing agreements (assumed for datasets with no license stated, in Appendix E). It might be clarified that these licensing agreements include redistribution and modification of the data.

11. In Appendix C.2, any particular VLM parameter settings (especially relating to temperature) might be stated.

12. In Appendix C.3, the normalization for $d_t$ (e.g. by assuming a normal distribution on the scores of the evaluated VLMs) might be stated.

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
The paper introduces U2-BENCH, a benchmark for evaluating large vision‑language models (LVLMs) on ultrasound understanding with 7,241 studies across 15 anatomies and 8 task types spanning classification, detection, regression, and text generation. It reports results for 20 LVLMs and proposes a composite “U2‑Score” that aggregates task metrics via sample‑count weights. Key findings show image‑level classification and some regression tasks are relatively strong, while spatial reasoning tasks (e.g., keypoint detection) and clinical text generation remain challenging; a prompt ablation indicates anatomy tokens boost diagnostic accuracy, and closed‑source models generally outperform open‑source ones.

### Strengths
The benchmark targets ultrasound, a clinically crucial yet under‑evaluated modality for LVLMs, with a broad task suite aligned to typical sonography workflows and clear task definitions and prompts per scenario.

The evaluation spans 20 modern LVLMs with standardized prompt formats and metrics, including most popular and current SOTA models, open- as well as closed-source.

The dataset curation aggregates many sources and applies multi‑stage QA with automated filtering plus manual review, and the authors release prompts, metrics, and an evaluation toolkit to facilitate reusability.

### Weaknesses
The composite U2‑Score’s task weights are proportional to sample counts, which conflates data availability with clinical importance and mixes heterogeneous metrics into a single scalar without uncertainty quantification, making ranking sensitivity high and potentially misaligned with clinician priorities.

No uncertainty metrics (e.g., confidence intervals, paired tests, bootstrap CIs) are reported for main tables, so small deltas across 20 models and many tasks may reflect sampling noise or prompt variance rather than true differences. This is especially problematic for small tasks like caption generation. Thus also not allowing a statistical significance test in order to determine the best models.

Error analysis is insufficient: failures are not stratified by artifact types, view standardization, anatomy, or acquisition protocol, limiting insight into whether errors stem from perception, spatial reasoning, or prompt sensitivity.


Dataset transparency gaps: heavy anatomical imbalance is acknowledged, but standardization, sampling, and QA procedures are described briefly without inter‑annotator agreement, rejection counts, or harmonization diagnostics across 40 sources.


Reproducibility is limited by heavy reliance on closed‑source models whose APIs evolve, and by omitted inference details (e.g., seeds, temperatures, multi‑pass voting) in the main leaderboard.


Minor issues: a missing section reference (Line 274), a typo (“Brest”→“Breast”), and Table 1 lacks per‑task sample counts and task‑abbreviation expansions in the caption despite being the primary results table.

### Questions
Can you report inter‑rater variability for the 10‑annotator review (e.g., per‑task agreement, adjudication rates) and quantify rejections during automated filtering to bound annotation noise ?

Please ablate prompt components beyond anatomy tokens on tasks where models struggle (e.g., KD, RG) to separate instruction‑following failures from perceptual limits.

Can you define minimal clinically acceptable thresholds for CVE, DD, and RG to contextualize utility and report the fraction of predictions meeting those thresholds ?

Please clarify and detail the “standardization, sampling, and quality checks” pipeline across 40 datasets, including per‑dataset preprocessing, label harmonization, leakage controls, and cross‑dataset consistency checks.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces U2-BENCH, the first comprehensive benchmark designed to systematically evaluate Large Vision-Language Models (LVLMs) on *ultrasound understanding* tasks. The benchmark includes 7,241 ultrasound cases across 15 anatomical regions and defines 8 clinically inspired tasks (disease diagnosis, view recognition, lesion localization, organ detection, keypoint detection, clinical value estimation, report generation, and caption generation) covering 50 ultrasound application scenarios. The authors benchmark 20 state-of-the-art LVLMs (open/closed-source, general/medical-specific) using standardized prompts and evaluation metrics. Results reveal that current LVLMs perform well on image-level classification but struggle with spatial reasoning and clinical language generation.

### Strengths
- This paper presents a well-motivated and comprehensive benchmark targeting an underexplored domain—ultrasound imaging. Its breadth (15 anatomies, 8 tasks) and evaluation rigor (20 models, standardized prompts) represent a good contribution.

- Twenty sota LVLMs are evaluated and compared, which makes the benchmark comprehensive. This benchmark would be beneficial for the medical multimodal community

### Weaknesses
- While I appreciate the substantial effort invested in constructing this benchmark and the meticulous annotation process, the paper currently lacks sufficient conceptual or analytical insights for the research community. As noted in Lines 132–134, prior work such as GMAI-MMBench has already included ultrasound-related evaluation scenarios. Although U2-BENCH expands the dataset scale and task diversity, the incremental novelty over existing benchmarks appears marginal, focusing primarily on scope rather than methodological innovation.
- Moreover, the experimental findings are largely expected: closed-source LVLMs outperform open-source and medical-specific models, reflecting well-known trends in multimodal evaluation. Given the proliferation of benchmark papers in the field, it would be more valuable if this work offered deeper analysis or actionable insights, such as identifying failure patterns or proposing improved solution, rather than solely reporting model performance.

### Questions
'??' at Line 274，missing reference

### Soundness
3

### Presentation
3

### Contribution
2
