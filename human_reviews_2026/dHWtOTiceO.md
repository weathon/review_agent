# Text2Arch: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 0

## Abstract
Communicating complex system designs or scientific processes through text alone is inefficient and prone to ambiguity. A system that automatically generates scientific architecture diagrams from text with high semantic fidelity can be useful in multiple applications like enterprise architecture visualization, AI-driven software design, and educational content creation. Hence, in this paper, we focus on leveraging language models to perform semantic understanding of the input text description to generate intermediate code that can be processed to generate high-fidelity architecture diagrams. Unfortunately, no clean large-scale open-access dataset exists, implying lack of any effective open models for this task. Hence, we contribute a comprehensive dataset, \system, comprising scientific architecture images, their corresponding textual descriptions, and associated DOT code representations.  Leveraging this resource, we fine-tune a suite of small language models, and also perform in-context learning using GPT-4o. Through extensive experimentation, we show that \system{} models significantly outperform existing baseline models like DiagramAgent and perform at par with in-context learning based generations from GPT-4o. We have added code and data as Supplementary material, and will make them (and models) publicly available on acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TEXT2ARCH, a new task and large-scale dataset (75,127 text–DOT–image triplets; 60,519/7,565/7,043 split) for generating scientific architecture diagrams from natural-language descriptions via intermediate DOT code. The dataset is curated through a three-stage pipeline—(1) filtering architecture figures, (2) DOT extraction with OCR/object detection + GPT refinement (DOT1→DOT2→DOT3), and (3) description refinement—illustrated in Fig. 2, with distributions and statistics in Fig. 3. The authors propose graph-level metrics (node/edge precision/recall/F1, PR-AUC, Jaccard) in addition to NLG metrics.

### Strengths
- Scoped, high-impact dataset for architecture diagrams with aligned text–code–image triplets and complexity bucketing; useful beyond the paper’s models.
- Clear curation pipeline (classifier + OCR/detection + GPT refinement) with ablations across DOT variants (DOT3≫DOT1/DOT2).
- Consistent empirical gains from finetuning small models (DeepSeek‑7B best on both automatic and GPT-based judging).

### Weaknesses
- Baseline fairness: DiagramAgent (TikZ) → DOT via GPT may degrade/alter structure; results could change with a native TikZ-based evaluation.
- Label/eval circularity risk: GPT‑4o is used to generate/refine DOT labels (DOT1→DOT3) and to score outputs; this can bias comparisons and makes “ground truth” partly model-dependent. The human set (n=99) helps but is small.
- string-similarity matching (Hungarian with τ=0.5) ignores diagram layout and may over/under-match aliases; multi-edges and duplicates aren’t handled; layout attributes are ignored.
- no analysis of near-duplicate figures across splits or of overlap between description sources and DOT generation that could inflate performance.

### Questions
- Beyond Table 3, can you expand the human-labeled set (≥500) and report the same metrics to better calibrate DOT3 quality?
- Provide TikZ-native evaluation for DiagramAgent (and a DOT-native variant for your model) to avoid cross-format conversion via GPT.
- Did you de-duplicate near-identical figures/descriptions across splits? Please report a hash/similarity analysis.
- Show results across node-match thresholds and with alternative label normalization; report effects of handling multi-edges/duplicates.
- Quantify contributions of each curation stage (classifier, OCR/detection, GPT refinements) to final performance; show training with DOT1-only/DOT2-only.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
**TEXT2ARCH** introduces a 75K+ text–DOT–image dataset that fills a clear gap for text-to-architecture generation, compares **DOT1/2/3** variants, and adds **novel graph-level metrics**; fine-tuned 7B–8B models (notably DeepSeek-7B) beat DiagramAgent and few-shot baselines.

### Strengths
1. **TEXT2ARCH fills a clear data gap.** The dataset addresses a previously missing resource for text-to-architecture diagram generation with aligned text–code–image triplets.
2. **Comprehensive comparison of DOT variants.** Evaluating DOT1/DOT2/DOT3 provides a thorough view of how different curation/refinement stages affect quality.
3. **Novel, task-appropriate metrics.** The graph-level evaluation (node/edge F1, PR-AUC, Jaccard) is thoughtful and well-aligned with the problem’s structural nature.

### Weaknesses
1. **Compilation success rate is unreported.** The paper does not quantify the percentage of generated DOT that compiles successfully, which is critical for practical usability.
2. **Limited qualitative evidence.** Case studies mostly show prompts and code; they lack rich visual side-by-side outputs and analyses of both good and bad generations across compared methods. It would be stronger to show the rendered diagrams and juxtapose multiple model outputs with brief error analyses.
3. **Insufficient SFT details (around line 322).** SFT setup is under-specified, only the prompt is given. Training loss/objectives should be described to ensure reproducibility.
4. **Missing TikZ results despite related discussion.** While focusing on DOT is reasonable, the paper cites TikZ-based prior work; a small TikZ transfer study (even limited) would help position the approach and set expectations for broader applicability.

### Questions
See weaknesses; I’m open to revising the score if the rebuttal provides sufficient evidence.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a dataset for diagram generation, based on descriptions. The paper provides a dataset involving the DOT language. It then continues to explore closed-source models such as GPT4o and fine-tuned models, leveraging automatic metrics for evaluation.

### Strengths
- a new dataset

- insights on the performance of different models, e.g. fine-tuning helps

- discussion of a relevant new problem, even though there are already some existing works on similar tasks

### Weaknesses
- There is no human evaluation at all

- While competitor approaches such as Automatikz are mentioned, there is no comparison

- Some competitor models are missing, e.g. TikZero

- Arguably, just fine-tuning a model on the dataset could be considered a bit incremental in contribution

- I feel more sophisticated automatic metrics such as the ones proposed in TikZero or Automatikz should be explored

TikZero: https://iccv.thecvf.com/virtual/2025/poster/51

### Questions
l. 126: I don't understand the argument why you don't want to evaluate the image: comparing the ground-truth image to the generated image is very important from a user perspective

- Why did you exclude GPT5?

- In l.203ff: you do some human filter - where are the agreements? How reliable are the humans involved?

- l.310: so, few-shot prompting is meaingless?

- l.322ff: But SFT mostly leverages training on the training dataset, right?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces a dataset for training models to automatically convert natural-language descriptions into scientific architecture diagrams. The authors report that a compact model trained on the proposed data outperforms larger models (e.g., GPT-4o) on the benchmark tasks defined in the paper.

### Strengths
1. Framing NL to diagram generation for scientific architectures is a well-scoped and timely problem with practical value for documentation and education.
2. The paper provides a purpose-built dataset aligned with the task, which can catalyze further research and standardized evaluation.
3. Initial experiments indicate that a smaller, task-specialized model can surpass much larger general-purpose models, suggesting meaningful gains from domain-specific supervision.

### Weaknesses
1. The paper’s core innovations and their separation from prior art are not sufficiently explicit. 
2. The definition of scientific architecture, the selection of the 99 images in Fig. 3, and the complexity distribution of diagrams across the full dataset are unclear.
3. It is unclear whether a diagram has a single “correct” \texttt{DOT} representation, and how correctness is measured when multiple valid encodings exist. 
4. For the same digram, only one correct DOT exist? If not, how they meature the correctness?
5. If many diagrams resemble the simple patterns in Fig. 1, current VLMs may already perform well.

### Questions
1. Definition and coverage of “scientific architecture.”
2. How were the 99 images in Fig. 3 selected, and how does their complexity compare with the full dataset?
3. Is the target \texttt{DOT} constrained by a formal grammar during training/inference?
4. How robust is the model to paraphrase, long/underspecified descriptions, or OOD domains?

### Soundness
2

### Presentation
1

### Contribution
2
