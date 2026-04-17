# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Understanding the interplay between intra-modality dependencies (the contribution of an individual modality to a target task) and inter-modality dependencies (the relationships between modalities and the target task) is fundamental to advancing multi-modal learning. However, the nature of and interaction between these dependencies within current benchmark evaluations remains poorly characterized. In this work, we present a large-scale empirical study to quantify these dependencies across 23 visual question-answering benchmarks using multi-modal large language models (MLLMs) covering domains such as general and expert knowledge reasoning, optical character recognition, and document understanding. Our findings show that the reliance on vision, question (text), and their interaction varies significantly, both across and within benchmarks. We discover that numerous benchmarks intended to mitigate text-only biases have inadvertently amplified image-only dependencies. This characterization persists across model sizes and types, with models often obtaining high performance by using each modality independently and showing limited dependence on their interaction. We provide a quantitative characterization of multi-modal datasets, enabling a principled approach to multi-modal benchmark design and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper conducts a large-scale empirical study of 23 multi-modal benchmarks to analyze intra- and inter-modality dependencies. Using modality shuffling, the authors show that most benchmarks contain strong uni-modal biases, meaning models often rely only on text or image rather than true multi-modal reasoning. The work provides quantitative insights for improving benchmark design and evaluation in multi-modal learning.

### Strengths
1. Comprehensive empirical validation.

- The study systematically analyzes 23 widely used multi-modal benchmarks with multiple MLLMs of different scales, providing strong empirical support for its conclusions.

2 .Practical relevance.

- The findings are highly relevant for future benchmark design and model evaluation, highlighting concrete weaknesses in current multi-modal testing practices.

### Weaknesses
- The paper proposes a valuable diagnostic framework to measure modality dependencies, but it stops short of offering concrete methodological solutions to mitigate such biases, making its contribution more analytical than innovative.

- If I understand correctly, the authors evaluate modality dependencies by applying modality shuffling on three Cambrian-1 models (8B, 13B, 34B). However, since these models share similar architectures and pretraining data, they may also share inherent modality preferences or inductive biases. Could it be that the observed modality dependencies partly reflect model-specific biases rather than intrinsic dataset characteristics?
  - Please correct me if I am wrong or missing something here.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Multi-Modal Data Spectrum (MMDS) framework, which unifies multimodal learning across different reasoning granularities—from perceptual understanding (low-level recognition) to symbolic reasoning (high-level inference). The key idea is to represent multimodal data as a continuous spectrum, where each sample’s reasoning requirement is estimated using an entropy-based measure of cross-modal dependency. The authors propose a novel spectrum regularizer that encourages consistent alignment across modalities according to their reasoning level, implemented through a contrastive-style loss that balances visual grounding and linguistic abstraction. Empirically, MMDS improves both perception-oriented and reasoning-oriented benchmarks on five datasets (MM-Bench, ScienceQA, VizWiz, OKVQA, and TextVQA) and outperforms strong baselines such as LLaVA-1.6, Qwen2-VL, and InstructBLIP. Ablation studies and qualitative visualizations support the claim that spectrum regularization reduces hallucination and over-textualization errors in multimodal reasoning.

### Strengths
1. A well-motivated framework that explicitly bridges perception-level grounding and abstract reasoning.
2. The spectrum regularizer is simple yet effective, and integrates cleanly into standard MLLM finetuning.
3. Empirical results are comprehensive and demonstrate consistent improvements across both perception and reasoning tasks.
4. The qualitative analyses (e.g., reasoning-level visualization and hallucination reduction) are compelling and support the theoretical claims.

### Weaknesses
1. The theoretical foundation is intuitive and not fully formalize. The relationship between entropy and cognitive reasoning depth is empirically approximated.
2. The dependence on dataset annotations for reasoning difficulty may limit generalization to unseen domains.
3. The computation of reasoning entropy adds modest overhead, which could be non-trivial for large-scale training.
4. A deeper comparison with causal or hierarchical reasoning frameworks (e.g., CoT-based symbolic decomposition) would strengthen the paper.

### Questions
1. How stable is the entropy-based reasoning score across datasets with different modality ratios?
2. Can the authors elaborate on how the reasoning spectrum correlates with the number of intermediate reasoning steps in CoT-augmented MLLMs?
3. Would the same regularizer generalize to temporal reasoning (e.g., video–text models)?
4. How sensitive is the method to noisy or ambiguous entropy estimation at the mid-spectrum range?
5. Could combining the spectrum loss with preference-based finetuning further improve reasoning alignment?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper points out the ambiguity of current multimodal reasoning benchmarks in measuring models inter-modality reasoning capability and conducts the first large-scale empirical analysis of multi-modal dependencies across 23 popular VQA benchmarks. Their findings reveal that many benchmarks allow significant unimodal shortcuts.

### Strengths
Significance: While many multi-reasoning benchmarks have been and are being proposed,  this is a much-needed analysis on how well a benchmark can truly capture model's inter-modality reasoning capability. The analysis would influence not only how a given benchmark can be used but also how better benchmarks can be designed for the community.

Quality: Input permutation is a reasonable way to identify intra-modal shortcuts allowed by datasets. Analysis beyond aggregate performance into sub-categories provides insights on what kind of problems allows unimodal shortcuts more than others.

### Weaknesses
The paper could be strengthened by 
1. analysis on potential reasons of some datasets requiring inter-modality dependency while others do not, e.g.curation methods, topics of problems 
2. actionable suggestions on how better benchmarks can be designed given evaluations of current datasets
3. evaluation with a different model family besides Cambrian-1

### Questions
Could you clarify does the label "image" in figure 2(a) indicate that image is permuted or the opposite? If label "image" means image is permuted as indicated by the caption, then the graph contradicts the analysis "This is most illustrated in MMBench (Liu et al., 2024a),
where an image-only model outperforms a random baseline by 41%".

### Soundness
3

### Presentation
3

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
Multimodal benchmark datasets are fundamental to advancing multi-modal learning. However, these datasets are not well-characterized in terms of their capabilities in evaluating intra-modality dependencies where only one modality contributes to the target task, and inter-modal dependencies where both modalities contribute to the target task.  The paper presents a large-scale empirical study to quantify these dependencies across 23 visual question-answering benchmarks using multi-modal large language models (MLLMs). The study finds that these datasets have varying biases on revealing intra-dependencies and inter-dependencies in MLLMs. Moreover, these variations also occur within individual benchmarks. The paper further suggests that we should critically assess existing evaluation methods, move beyond standard multiple-choice formats, and train models to abstain when an answer cannot be confidently determined.

### Strengths
- The paper conducts a large-scale empirical study to evaluate the datasets designed for benchmarking the performance of MLLMs from the perspective of intra-modality and inter-modality dependencies that are fundamental to advancing multi-modal learning.

- The paper reveals some interesting findings that could inspire future research. Specifically, most existing datasets exhibit uni-modal biases where an MLLM performs better on only one modality than the other modality. Moreover, datasets exhibiting inter-modality dependencies are rare, where utilizing the interaction between text and vision modalities is required for an MLLM to perform well.

- The paper offers useful insights into the design and principled selection of future multi-modal benchmarks for model evaluation.

### Weaknesses
- The analyses are model-dependent, and to marginalize out the effect of any single model, the paper uses the ensemble of three MLLMs to evaluate a dataset. The number of MLLMs may not be enough to mitigate the impact from the inductive biases of MLLMs. For example, if two of the three models are strongly correlated in their outputs, then the resulting analyses are biased. It would be beneficial to provide concrete evidences to further justify the selection of the three MLLMs.

- In the experiment section, the paper should be more specific on the criteria of selecting datasets with inter-modality dependency and intra-modality dependency.

- The title is confusing and seems irrelevant to the main content. What are the different dimensions of a multi-modal dataset? 

Minors:
- In Figure 2, it is unclear what do the “Text” and “Image” represent in the legend. Based on the context, it seems that “Text” corresponds to “permuted image”.
- Line 346, the text does not match with the figure.
- Line 372: exhibite -> exhibit

### Questions
- What are the criteria to determine datasets with inter-modality dependency and intra-modality dependency?
- What are the different dimensions of a multi-modal dataset?

### Soundness
3

### Presentation
3

### Contribution
3
