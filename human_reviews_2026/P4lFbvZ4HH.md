# ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 440 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ChartGalaxy, a large-scale dataset comprising million-scale synthetic and 61K real infographic charts paired with underlying data tables, aimed at advancing multimodal reasoning and generation in large vision-language models (LVLMs). The dataset is constructed through a two-stage process. The authors demonstrate the dataset's utility via three applications: fine-tuning LVLMs for improved infographic understanding, benchmarking 17 LVLMs on code generation for infographics (with Gemini-1.5-Pro topping), and an example-based generation method that outperforms GPT-4V in user studies on fidelity, aesthetics, and creativity.

### Strengths
- The paper addresses a clear and important gap on infographic data. The community has largely focused on plain charts, while complex infographics, which are common in media, business, and education, remain a major challenge for LVLMs and no large-scale high-quality data exists.

- The pipeline for generating synthetic data is a major strength. The authors incorporate D3, which offers much more visualization options than conventional approaches relying on matplotlib or seaborn. The use of template-based generation approach is quite clever and neat.

- The authors do not just present a dataset; they thoroughly demonstrate its utility. All three applications are well-conceived and executed through chart understanding, code generation, and chart generation.

### Weaknesses
- The authors did not discuss the performance of the object detection model (InternImage+DINO) used to parse real charts.


- While outperforming baselines in applications, direct comparisons to baselines are missing; e.g., no fine-tuning ablation using only prior datasets. The authors can compare relevant training datasets such as [1][2][3] as baselines. These additional experiments would enhance the soundness of the paper.


[1] Liu et al., MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning. NAACL 2024

[2] Huang et al., Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding. ACL 2025 Findings 

[3] Masry et al., ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild. COLING 2025 Industry Track

### Questions
In section 3.4, you first mention you use gemini-2.0-flash to select templates. However, in the next paragraphs about layout optimization, you mentioned you filter templates based on some criteria. These two parts seem conflicting. Could you elaborate more?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This manuscript introduces ChartGalaxy, a large-scale dataset designed for infographic chart understanding and generation. The dataset combines 61,833 real infographic charts collected from 18 publicly available sources with 1.7 million synthetic infographic charts generated through an inductive, template-based synthesis pipeline. Each chart is paired with a tabular dataset, enabling multimodal learning. The authors demonstrate ChartGalaxy’s utility through three applications: (1) fine-tuning LVLMs for infographic chart understanding, (2) benchmarking infographic chart code generation, and (3) example-based infographic chart generation evaluated via a user study.

### Strengths
S1: Comprehensive and large-scale dataset. The dataset includes over 1.76 million infographic charts paired with tabular data, exceeding the scale of previous datasets. This scope enables training LVLMs for realistic infographic scenarios, supporting broad generalization. The inclusion of 75 chart types, 440 chart variations, and 68 layout templates reflects high visual and structural diversity. The dual-source construction (real + synthetic) provides both authenticity and scalability, addressing prior dataset limitations.

S2: Well-defined inductive synthesis pipeline. The human-in-the-loop design for extracting and expanding layout templates demonstrates methodological soundness. The pipeline’s iterative detection and clustering stages ensure diversity. The layout optimization step formalizes the spatial packing problem, providing a principled way to ensure high visual quality. The combination of data-to-chart mapping rules, adaptive sampling for variation balance, and semantically resonant color selection enhances the realism of synthetic charts.

S3: Clear evidence of downstream utility. Fine-tuning LVLMs with ChartGalaxy yields large gains on InfographicVQA and ChartQAPro and even larger gains on the authors’ evaluation set, showing measurable improvement. The code generation benchmark evaluates 17 LVLMs with reproducible metrics (SVG-level and GPT-4o-based judgments), positioning the dataset as an evaluation standard. The example-based generation experiment includes a controlled user study showing statistically significant improvements in fidelity, aesthetics, and creativity.

### Weaknesses
W1: 
The manuscript would benefit from a clearer definition of what qualifies as an infographic chart and a more concrete explanation of how it differs from plain charts in terms of reasoning challenges. The current description in the Introduction could be made more specific, perhaps with brief examples or a sharper comparison to existing VQA or chart QA tasks to better highlight the unique difficulty of infographic chart understanding.

W2: 
Table 1 shows that models achieve higher accuracy on InfographicVQA than on ChartQAPro. It would be helpful to include a short discussion on potential factors contributing to this gap, such as differences in question design, dataset composition, or task difficulty. Clarifying this would better support the manuscript’s motivation and help readers interpret the reported results.

W3: 
The synthetic infographic chart generation pipeline is a valuable contribution. To further strengthen the work, it could be beneficial to include a small quantitative or qualitative assessment of the fidelity and usefulness of the generated charts beyond the downstream fine-tuning results. Even a brief ablation on key components, such as layout optimization, would provide additional insight.

W4: 
The dataset and benchmark are timely and impactful. Adding more analysis or reflections on model behavior, enabled by this dataset, could enhance the contribution for the ICLR audience, which often values methodological insight alongside dataset and system advances.

### Questions
Q1: Shouldn’t the original intention and practice of infographic charts, such as adding explanatory text, icons, and descriptive narrative structures as clues, be to make it easier to understand? Is there any empirical evidence that infographic charts are more difficult for VLMs to understand?

Q2: W2.

Q3: W3.

Q4: The setup of the GPT-Image-1 baseline in the example-based infographic chart generation experiment is insufficiently documented. How was GPT-Image-1 prompted? Any layout or structural guidance supplied?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper constructs a large-scale dataset for infographic charts.
While previous infographic datasets were limited in scale, this work builds a massive dataset of 1.7 million synthetic samples, combining web-collected real data with diverse layout templates.
Using the constructed dataset, the authors conduct experiments on three tasks: chart understanding, code generation, and example-based infographic chart generation.

### Strengths
- 1. Templates are carefully extracted from real data and utilized for synthetic data generation, resulting in a diverse set of samples.
- 2. A large number of evaluation experiments are conducted in a thorough and detailed manner.
- 3. This work provides a large-scale dataset in the infographic domain, where available data has been scarce until now.

### Weaknesses
There are some unclear points regarding the details and procedures of the experiments.
- 1. In Section 3.3, the paper describes the extraction of layout templates. Could you clarify the format in which these templates are stored? While Figure 3 presents visual examples of the template images, it would be helpful to know whether they also contain information such as bounding boxes or other structural annotations.
- 2. In Section 3.4, which discusses Element Generation, could you please clarify the procedure for chart generation? Specifically, are the charts produced using predefined D3.js code that is associated with each layout template, or are they dynamically generated through the use of LLMs or other generative models?
- 3. Does the proposed method described in Section 4.3 generate charts by specifically utilizing the Element Generation/Recommendation and Layout Optimization processes presented in Section 3.4?

### Questions
- 1. In Section 4.2 on the infographic chart generation task, only zero-shot evaluation is conducted. Is it also possible to fine-tune the model using the synthetic data?
- 2. In the same Section 4.2, why are the evaluation samples drawn only from the synthetic subset? Since the synthetic data itself was generated by combining chart types and variations using D3.js with LLM-based synthesis, wouldn’t using only LLM-generated data for evaluation introduce a potential bias that makes the generation task easier for the model?

### Soundness
2

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
3

### Summary
This paper presents ChartGalaxy, a dataset for chart understanding and generation. It contains over 1 million infographics, some of them sourced from the real world and some of them auto-generated. Authors show three use cases with the dataset: improving infographics QA, benchmarking infographics generation, and enabling example-based infographic generation.

### Strengths
+ The dataset size is very large, and it seems to span many chart types and designs.
+ The dataset quality seems to be high.
+ The use cases are in general convincing.
+ The supplemental materials are a very helpful addition for presentation.

### Weaknesses
- I am a bit lost in reading the curation of machine-generated synthetic infographics. I can see a lot of work went into this, but I feel like it is lacking a big picture. Please see my first question. The other parts of the paper all seem pretty easy to follow.
- Huge improvements on questions based on ChartGalaxy is probably not that unsurprising given the vast majority of infographics in the QA set are based on synthetic charts with fixed templates. Finetuning is expected to help a lot here. Also I believe the QA set is not yet published so we do cannot gauge QA set quality.
- The code generation metric seems a bit questionable. For example, GPT-4.1-mini ranks higher than o4-mini, o1, o3, and GPT-4o. It is better to have a validation process where humans independently score each chart without seeing the model generating it and establish some sort of inter-rater reliability with scores produced by your metric.

### Questions
- The authors say templates capture the spatial relation of chart elements, but how are templates represented? Is each template a D3 code snippet that can take in different data tables and images as input? Do you use a VLM to generate these templates? And how does layout optimization work with these templates?
- All Chit Chart charts seem to not wrong urls. They all have https://chitchart.com/ without further specification (e.g., 00060283, 00059496). Could the authors check if this is the case please?
- The authors say "Moreover, we construct an independent, human-verified evaluation set containing 2,176 synthetic charts with 4,975 question-answer pairs." Are these a sampled and verified subset of all VLM-generated questions?

### Soundness
3

### Presentation
2

### Contribution
3
