# SpatiaLab: Can Vision–Language Models Perform Spatial Reasoning in the Wild?

- Decision: Accept (Poster)
- Scores: 8, 2, 2, 4

## Abstract
Spatial reasoning is a fundamental aspect of human cognition, yet it remains a major challenge for contemporary vision–language models (VLMs). Prior work largely relied on synthetic or LLM-generated environments with limited task designs and puzzle-like setups, failing to capture the real-world complexity, visual noise, and diverse spatial relationships that VLMs encounter. To address this, we introduce **_SpatiaLab_**, a comprehensive benchmark for evaluating VLMs’ spatial reasoning in realistic, unconstrained contexts.
**_SpatiaLab_** comprises 1,400 visual question–answer pairs across six major categories: *Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation,* and *3D Geometry*, each with five subcategories, yielding 30 distinct task types. Each subcategory contains at least 25 questions, and each main category includes at least 200 questions, supporting both multiple-choice and open-ended evaluation.
Experiments across diverse state-of-the-art VLMs, including open- and closed-source models, reasoning-focused, and specialized spatial reasoning models, reveal a substantial gap in spatial reasoning capabilities compared with humans. In the multiple-choice setup, InternVL3.5-72B achieves 54.93% accuracy versus 87.57% for humans. In the open-ended setting, all models show a performance drop of around 10–25%, with GPT-5-mini scoring highest at 40.93% versus 64.93% for humans. These results highlight key limitations in handling complex spatial relationships, depth perception, navigation, and 3D geometry.
By providing a diverse, real-world evaluation framework, **_SpatiaLab_** exposes critical challenges and opportunities for advancing VLMs’ spatial reasoning, offering a benchmark to guide future research toward robust, human-aligned spatial understanding. **_SpatiaLab_** is available at: https://spatialab-reasoning.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces SpatialLab, a benchmark designed to evaluate spatial reasoning in vision-language models (VLMs). It features real-world images and carefully annotated questions that span a diverse range of spatial reasoning tasks. Using this benchmark, the authors assess various off-the-shelf VLMs and further analyze performance improvements achieved through different enhancement strategies, including prompt-based, multi-agent-based, and supervised fine-tuning (SFT)-based approaches.

### Strengths
- Carefully curated set of images and questions covering major spatial reasoning types.
- Comprehensive analysis showing that even proprietary VLMs perform worse than humans across nearly all subtasks.
- Additional image complexity analysis provides insights into which visual domains require more attention in future training datasets.
- In-depth quantitative and qualitative error analysis.
- Evaluation of multiple strategies for improving spatial reasoning in VLMs (prompt-based, multi-agent-based, SFT-based, etc.) — particularly valuable in Section 5.4.

### Weaknesses
- The evaluation section (5.4) could be further strengthened by including reinforcement learning (RL)-based approaches for comparison, though this is not strictly necessary.

### Questions
- In L1132, the authors mention “we review prior benchmarks … analyze their limitations.” Could the authors elaborate on the specific limitations identified in each benchmark? A comparison table summarizing these would greatly aid readers and inform future benchmark design.
- [Suggestion] While page limits are understandable, brief descriptions of each improvement approach in Section 5.4 would improve clarity. For instance:
  - What self-reflection prompt was used?
  - Which dataset and dataset size were used for SFT fine-tuning?
  - What base VLM was used?
  - Could the authors provide a one-sentence description of SpatialXolver?
- [Suggestion] Adding a table of contents to the appendix would help readers navigate the paper more easily.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces SPATIALAB, a comprehensive benchmark for evaluating spatial reasoning in vision–language models (VLMs) under realistic, unconstrained visual conditions. It contains 1,400 visual question–answer pairs spanning six major categories and 30 subcategories, each testing aspects like depth, occlusion, orientation, and navigation.
The benchmark supports both multiple-choice and open-ended formats, enabling comparison between discriminative and generative reasoning.
Extensive experiments on 25+ models (open-source, proprietary, reasoning-tuned, and spatially specialized) reveal a substantial gap between human and model performance (e.g., 54.9% vs. 87.6% on MCQ; 40.9% vs. 64.9% on open-ended).
The paper provides error analysis, fine-tuning experiments, and attempts at improvement via CoT prompting, self-reflection, SFT, and multi-agent systems.

### Strengths
Comprehensive Evaluation
- Evaluates over 25 VLMs, including open-source and proprietary systems, and human baselines.
- Dual-format testing (MCQ + open-ended) is valuable, revealing a 20–25% accuracy gap between the two modes


Diagnostic and Actionable Insights
- The benchmark reveals concrete gaps (e.g., geometry-aware supervision, spatial chaining, embodied data) that can guide future research. The diagnostic perspective makes it a useful community tool even without conceptual novelty.

### Weaknesses
Limited Conceptual Novelty
- Many recent benchmarks (OmniSpatial, BLINK-Spatial, Spatial-MM, SpatialRGPT, EmbSpatial) already use real-world imagery, multiple spatial categories, and QA-based evaluation. SPATIALAB’s innovation lies mainly in breadth and integration, not in introducing new reasoning types or data modalities

Limited Guidance on Model Improvement
- Although weaknesses of current VLMs are carefully diagnosed, the paper offers little practical guidance or insight into how to overcome them. The discussion remains observational (what fails) rather than prescriptive (how to fix it), limiting its utility for researchers aiming to design better spatial reasoners.

Moderate Dataset Scale
- Despite 1,400 QA pairs sounding large, it is relatively small compared to existing multimodal datasets (often tens or hundreds of thousands). The modest size restricts its usefulness for training or fine-tuning and confines SPATIALAB to evaluation only.

### Questions
1. How exactly does SPATIALAB differ from OmniSpatial or Spatial-MM beyond taxonomy size and annotation detail?

2. Are there reasoning types uniquely represented here?

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
3

### Summary
This paper introduces SPATIALAB, a new benchmark designed to evaluate the spatial reasoning abilities of vision–language models (VLMs) in real-world, unconstrained settings.

The dataset comprises 1,400 visual question–answer pairs across six high-level categories (e.g., Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation, and 3D Geometry) and supports both multiple-choice and open-ended evaluations.

The authors benchmark over 25 state-of-the-art models (open- and closed-source) and provide quantitative comparisons against human baselines, together with error analyses and several reasoning interventions (e.g., Chain-of-Thought prompting, supervised fine-tuning, and multi-agent reasoning).

The work aims to reveal systematic weaknesses in current VLMs’ spatial reasoning and propose SPATIALAB as a comprehensive diagnostic framework.

### Strengths
**Comprehensive empirical evaluation.** The authors test a wide range of modern VLMs under multiple evaluation formats, which provides useful diagnostic data and an updated empirical snapshot of model limitations in spatial reasoning.

**Well-structured benchmark.** The dataset taxonomy (6 categories × 5 subcategories) is clearly defined and covers a broad set of spatial reasoning tasks beyond synthetic toy examples.

**Clear presentation.** The paper is readable and systematically organized, with detailed tables and qualitative examples that make the results easy to interpret.

**Reproducibility focus.** The authors discuss data collection, annotation, and quality control in detail and commit to open release, which is commendable for community benchmarking.

### Weaknesses
**Lack of methodological contribution.** The work’s novelty lies almost entirely in dataset construction and large-scale evaluation.
There is no new modeling approach, algorithm, or analytical framework proposed.
While benchmarks can be valuable, ICLR typically expects either new learning methodology, representation insights, or deeper diagnostic mechanisms beyond dataset release.

**Limited depth of analysis.** Despite extensive tables, the analysis remains descriptive rather than mechanistic.
The paper identifies “what fails” (e.g., navigation and occlusion tasks) but not “why” in terms of representation or model architecture.
There are no probing studies, attention analyses, or causal/intervention experiments that explain the underlying representational failure modes.
Many claims (e.g., “models lack geometric grounding”) are plausible but unsupported by direct evidence.

### Questions
Could the authors please justify their contributions in the "in-depth analysis" and provide key takeaways from it?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce SpatiaLab, a new benchmark dataset of 1,400 visual question-answer pairs designed to evaluate VLM spatial reasoning in "in-the-wild" contexts. The benchmark spans six categories (30 subcategories) and supports both multiple-choice and open-ended evaluation formats. The authors conduct a large-scale evaluation of over 25 VLMs, revealing a substantial performance gap between SOTA models (e.g., InternVL3.5-72B at 54.93%) and a human baseline (87.57%). The analysis is extended to improvement strategies (SFT, CoT, Agents), which are shown to provide limited or inconsistent gains, suggesting current models lack fundamental spatial grounding.

### Strengths
1. **Well-Motivated Problem**: The work correctly identifies a critical flaw in existing benchmarks: an over-reliance on synthetic, "puzzle-like" setups that fail to capture real-world visual complexity. The focus on cluttered, "in-the-wild" imagery is a necessary contribution.

2. **Dual-Format Analysis**: The direct comparison of MCQ and Open-ended formats is a key strength. It provides a quantitative basis for the intuition that MCQ overestimates model capabilities, highlighting a significant $\approx$23% average performance drop.

3. **Failure Analysis**: The demonstration that standard improvement techniques (CoT, SFT) provide marginal, inconsistent, or even negative gains is an important finding for the field, pointing to deeper representational deficits.

### Weaknesses
1. **Statistical Robustness of the Benchmark**: The primary methodological flaw is the dataset's scale. 1,400 items spread across 30 subcategories means each sub-task is evaluated with a small sample (as few as 25 items, averaging <50). This $n$ is insufficient to draw robust, fine-grained conclusions. The granular analysis in Tables 5-9, while interesting, risks being statistically noisy. 

2. **Perplexing SFT Dynamics**: The SFT analysis (Sec 5.4, Fig 4)  is central to the paper's insight, but the results are anomalous and underexplored. The sharp U-shaped curve in open-ended performance (dropping from 34.4% to 12.6% before recovering to 35.5%)  is highly non-trivial. The paper gestures at "catastrophic forgetting"  but provides no direct investigation. This dynamic must be rigorously explained (e.g., via representational analysis or multi-seed validation) to be a credible scientific finding rather than a training artifact.

3. **Marginal Novelty in a Crowded Field**: The paper's own related work (Table 1) demonstrates this is an extremely crowded and concurrent field (e.g., SpatialMM, OmniSpatial, BLINK, VLMAD). The claim to novelty rests on "real-world complexity" , yet several competitors also use "Internet" or "Mix" data sources. The authors must provide a much sharper justification for why SPATIALAB's 1,400 "manual" items provide fundamentally different insights than the (often larger) concurrent datasets.

4. The model set omits several state-of-the-art commercial VLMs (e.g., GPT o3/5, Google Gemini 2.5 pro), which weakens the headline claim about “current VLMs.” They have more robust and powerful ability.

### Questions
I am curious about the true value of the dataset, as I strongly suspect that it may merely overfit to its own format.

If a base model (e.g., Qwen-VL) could be fine-tuned on this dataset and subsequently demonstrate performance gains on other benchmarks (such as VSI-Bench, OmniSpatial and SPACE), I would be much more inclined to recognize the dataset’s contribution.

### Soundness
2

### Presentation
2

### Contribution
2
