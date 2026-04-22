# PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Understanding human behavior traits is central to applications in human-computer interaction, computational social science, and personalized AI systems. Such understanding often requires integrating multiple modalities to capture nuanced patterns and relationships. However, existing resources rarely provide datasets that combine behavioral descriptors with complementary modalities such as facial attributes and biographical information. To address this gap, we present PersonaX, a curated collection of multimodal datasets designed to enable comprehensive analysis of public traits across modalities. PersonaX consists of (1) CelebPersona, featuring 9444 public figures from diverse occupations, and (2) AthlePersona, covering 4181 professional athletes across 7 major sports leagues. Each dataset includes behavioral trait assessments inferred by three high-performing large language models, alongside facial imagery and structured biographical features.
We analyze PersonaX at two complementary levels. First, we abstract high-level trait scores from text descriptions and apply five statistical independence tests to examine their relationships with other modalities. Second, we introduce a novel causal representation learning (CRL) framework tailored to multimodal and multi-measurement data, providing theoretical identifiability guarantees. Experiments on both synthetic and real-world data demonstrate the effectiveness of our approach. By unifying structured and unstructured analysis, PersonaX  establishes a foundation for studying LLM-inferred behavioral traits in conjunction with visual and biographical attributes, advancing multimodal trait analysis and causal reasoning. 
The code is available at https://github.com/lokali/PersonaX.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PersonaX, a pair of multimodal datasets for studying LLM-inferred behavioral traits: CelebPersona and AthlePersona. Each example includes (i) LLM-generated Big Five trait descriptions/scores, (ii) facial information, and (iii) biographical fields. The authors analyze PersonaX at two levels: (1) applying five statistical independence tests between inferred traits and structured/visual attributes; and (2) proposing a Causal Representation Learning (CRL) framework for multi-modality & multi-measurement data with identifiability guarantees, validated on synthetic data, and then used it to learn latent representations and causal graphs on PersonaX.

### Strengths
1.	A New Multimodal Dataset.
The paper introduces PersonaX, a multimodal dataset that aligns two corpora and pairs LLM-inferred Big Five personality traits with facial imagery/embeddings and biographical attributes, enabling large-scale cross-modal analysis.
2.	Model and Prompt Selection Strategy
The authors conduct a detailed evaluation (Figure 2) of various models and prompt templates to select the most reliable combination for dataset construction. 
3.	Two-Level Analysis Pipeline.
The proposed analysis framework consists of two levels:
	* Level I applies five statistical independence tests to systematically characterize dependencies between Big Five scores and structured/visual features.
	* Level II introduces a causal representation learning framework with identifiability guarantees. This framework is first validated on synthetic data and then applied to the real multimodal embeddings, uncovering interpretable cross-modal structures.

### Weaknesses
1.	Lack of Human Validation for LLM-Inferred Personality Traits.
The Big Five personality traits inferred by LLMs have not undergone human validation. While prior work has shown that LLMs can somehow perform personality analysis tasks, the release of a dataset built on such inferences should be accompanied by human verification to ensure quality. Although the authors carefully selected the model and prompt to reduce variance and improve consistency, this does not guarantee alignment with human judgments.
2.	Limited Generalizability.
The AthlePersona subset is restricted to male athletes, and the CelebPersona subset primarily consists of public figures. As such, the conclusions drawn in this paper may not generalize well to broader or more diverse populations.

### Questions
1. Has any effort been made to assess the consistency between LLM-inferred and human-assessed Big Five personality traits?
2. Could the inclusion of league information in the prompt introduce or reinforce stereotypes?
3. How can the current analysis of athletes and public figures offer for generalizing to broader populations?

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
The paper presents PersonaX, a multimodal dataset framework designed to explore behavioral trait inference through large language models (LLMs). It introduces two datasets: CelebPersona, comprising 9,444 public figures, and AthlePersona, including 4,181 professional athletes from seven major leagues. Each record combines LLM-inferred Big Five personality traits, facial embeddings, and biographical metadata. The authors conduct a two-level analysis: structured statistical independence testing and a causal representation learning (CRL) framework with identifiability guarantees, evaluated on both synthetic and real data. The results reveal interpretable cohort-specific dependencies, and the dataset is released under privacy-preserving, non-commercial terms.

### Strengths
1. Comprehensive dataset linking visual and behavioral attributes through LLM inference.
2. Multi-level analysis combining structured statistical tests and causal representation learning.
3. Privacy-aware release strategy (embeddings only, non-commercial use).
4. Systematic evaluation of LLMs and prompts for reliability.
5. Interpretability and cross-domain insights (e.g., appearance importance for celebrities vs. organizational traits for athletes).
6. Well-documented, reproducible methodology.

### Weaknesses
1. No human-ground-truth validation for LLM-inferred Big Five scores. The inferred personality traits are never compared against human annotations or standardized psychometric assessments, making it difficult to evaluate their validity. A small-scale validation study would substantially improve confidence in the dataset’s behavioral realism.
2. Gender and domain bias (athletes are all male, celebrities are mostly Western). These sampling biases limit generalizability and may influence the discovered dependencies, as LLMs and facial embeddings could internalize social and cultural stereotypes from biased data sources.
3. Causal interpretations rely on untestable assumptions. The proposed causal representation learning (CRL) framework depends on theoretical assumptions such as independent measurement noise, injectivity of modality-specific measurement functions, and sufficient diversity across modalities to ensure identifiability. These assumptions are difficult to verify empirically in real-world multimodal datasets like PersonaX, where both LLM-inferred traits and facial embeddings are high-dimensional and potentially correlated. Without diagnostic checks or robustness analyses, the causal graphs may reflect associations rather than true underlying causal structure, making interpretability plausible but not necessarily causal.
4. Lack of downstream benchmark tasks to demonstrate the utility of learned latents.

### Questions
1. Have you performed any human validation or external benchmarking to assess the reliability of the LLM-inferred Big Five traits, even on a small subset of samples?
2. The causal representation learning (CRL) framework depends on assumptions such as measurement noise independence and modality injectivity. What empirical checks or sensitivity analyses did you perform to verify that these assumptions approximately hold for PersonaX?
3. How stable are the learned causal graphs across random seeds or when a modality (e.g., images or text) is removed? Do key dependency patterns persist?
4. Have you compared the CRL embeddings to any non-causal multimodal baselines (e.g., CLIP-style or multimodal autoencoders) to verify that the causal formulation provides additional structural or representational advantages?
5. Could you elaborate on the privacy-preserving mechanism applied to the released facial embeddings, specifically, whether the transformation is invertible or resistant to re-identification attacks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes PersonaX, a collection of multimodal datasets including LLM-inferred behavior trait assessments with facial imagery and structured biographical features, and the paper introduces a two-level analysis pipeline: (1) structured independence testing to discover trait-attribute dependencies (2) a multimodal, multi-measurement causal representation learning (CRL) framework with identifiability guarantees that learns shared and modality-specific latent factors. The authors provide dataset construction details, systematic LLM selection and prompt design, synthetic benchmarks, and real-world analyses on PersonaX.

### Strengths
1.Novel multimodal benchmark. PersonaX fills a clear gap by uniting LLM-derived personality assessments with facial and biographical modalities, enabling large-scale, cross-modal trait analysis not available in existing resources.
2.Rigorous LLM measurement pipeline. The authors comprehensively benchmark multiple LLMs, prompt templates, and scoring schemes, reporting consistency and missing-rate analyses that enhance transparency and reproducibility.
3.Integration of theory and practice. The proposed CRL framework offers identifiability guarantees and is validated through both synthetic benchmarks and real-world data, effectively bridging theoretical development and empirical evaluation.

### Weaknesses
1.Insufficient external validation of LLM-inferred traits. Despite internal consistency checks, the lack of human-annotated validation or behavioral benchmarks undermines confidence in the real-world accuracy of the inferred personality traits.
2.Limited demographic diversity and potential bias. The datasets primarily include male athletes and high-profile celebrities, introducing demographic and socioeconomic biases that constrain generalization and may lead to spurious correlations.
3.Under-specified theoretical assumptions in practice. The identifiability results rely on strong technical conditions whose practical validity for noisy, LLM-based embeddings is not empirically examined, limiting interpretability of real-world findings.

### Questions
1.Strengthen label validity. Incorporate human-annotated subsets or downstream behavioral prediction tasks to quantitatively verify the reliability and utility of the inferred traits.
2.Address cohort bias. Broaden demographic coverage or conduct sensitivity and subgroup analyses to isolate dataset-specific dependencies and clarify the scope of generalization.
3.Clarify theoretical grounding. Provide ablations and diagnostics verifying identifiability-related assumptions and assess the impact of the embedding obfuscation transformation on model performance.

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
3

### Summary
This paper introduces PersonaX, a set of curated multimodal datasets—CelebPersona and AthlePersona—incorporating LLM-inferred behavioral trait assessments (Big Five), facial embeddings, and structured biographical metadata for public figures and professional athletes. The authors present a two-level analytical framework: (1) statistical independence tests exploring modality-specific dependencies between behavioral traits and structured features, and (2) a novel causal representation learning (CRL) model with theoretical identifiability guarantees for multimodal, multi-measurement data. Empirical validations are performed using both synthetic and real-world datasets, demonstrating the proposed approach’s effectiveness in cross-modal trait analysis and causal discovery.

### Strengths
PersonaX provides a valuable resource that bridges LLM-inferred behavioral traits with biographical and visual data across thousands of public personas, addressing a clear gap in multimodal behavioral trait datasets. 
Methodological advances in causal representation learning: The paper proposes an end-to-end causal modeling framework tailored to multimodal, multi-measurement data, supported by reasonably rigorous identifiability theory.
Extensive experiments are conducted both on synthetic settings and real-world data, with multiple baselines including BetaVAE, MCL, and MMCRL.
The data pipeline and experimental protocols are well-illustrated, with detailed breakdowns in supplementary materials.

### Weaknesses
Limited diversity and representativeness in dataset sampling. CelebPersona and AthlePersona focus on public figures and male athlete, which are not demographically representative of broader populations. This constrains the generalizability of reported findings and may bias downstream analyses, as acknowledged but not addressed in either analysis or ablation.
Experiments lack real-world baseline comparisons for CRL. The effectiveness of the CRL component is shown mostly on synthetic MNIST and real-world data in isolation; it would be more persuasive to demonstrate improvements over state-of-the-art real-world baselines or alternative causal discovery methods.
Causal graph interpretability and stability concerns. The mapping between learned latents and original human-interpretable traits is indirect.
Presentation and clarity issues in tables/figures. Table 1 is visually dense; important metrics such as overall score should be more clearly emphasized.

### Questions
What is the impact of prompt selection and LLM choice on downstream causal graph structure and variable interpretability? Are the reported findings robust to these design choices?
How would the proposed method scale to or generalize with with populations beyond public celebrities and male athletes, and what steps are planned to mitigate cohort-specific biases?

### Soundness
3

### Presentation
3

### Contribution
3
