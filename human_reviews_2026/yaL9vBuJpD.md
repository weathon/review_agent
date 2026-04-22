# A11YN: Aligning LLMs for Accessible Web UI Code Generation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) have recently demonstrated strong capabilities in generating functional and aesthetic web interfaces directly from instructions. However, these models often replicate accessibility flaws from their training data, resulting in interfaces that exclude users with diverse needs and contexts. To address this gap, we introduce A11yn, the first method that aligns code-generating LLMs to reliably produce accessibility-compliant web UIs. A11yn optimizes a novel reward function that penalizes violations of the Web Content Accessibility Guidelines (WCAG), with penalties scaled to the severity of each violation as identified by an accessibility testing engine. To support training, we construct UIReq-6.8K, a dataset of 6,800 diverse instructions for web UI generation. For evaluation, we introduce RealUIReq-300, a benchmark of 300 real-world web UI requests grounded and manually curated from public web pages, spanning a broad range of use cases. Empirical results show that A11yn significantly outperforms strong baselines, lowering the Inaccessibility Rate by 60% over the base model while preserving semantic fidelity and visual quality of generated UIs. These findings demonstrate that accessibility can be systematically optimized within LLMs, showing the feasibility of aligning code generation for accessibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces A11YN, an approach to align large language models for generating more accessible web UI code. The method integrates accessibility evaluation (via Axe-core) into a GRPO-based reinforcement learning framework and fine-tunes an open-source code model using synthetic accessibility-focused datasets (UIReq-6.8K and RealUIReq-300). Experiments show reduced violation rates and improved accessibility scores compared with baselines.

### Strengths
+ Socially meaningful objective. Addressing accessibility in code generation is an important and underexplored direction that broadens the scope of model alignment beyond general coding accuracy.

+ Clear pipeline design. The integration of automatic accessibility evaluation into a reinforcement learning loop is well described and straightforward to reproduce.

+ Comprehensive baseline coverage. The experiments include multiple strong models (e.g., GPT-4, CodeLlama, DeepSeek-Coder), providing a fair empirical comparison within the same evaluation setup.

### Weaknesses
- Limited methodological novelty.  The paper mainly applies an existing RL-based alignment framework (GRPO) to web accessibility without introducing new algorithmic ideas or training mechanisms. While the topic is valuable, the technical contribution is largely an adaptation rather than a conceptual advance.


- Synthetic and weakly validated data.  Both the training dataset (UIReq-6.8K) and the benchmark (RealUIReq-300) rely heavily on GPT-generated instructions and code. This raises concerns about data authenticity and generalization, as the method is essentially evaluated on distributions created by the same type of model.


- Evaluation lacks robustness.  The reported improvements mainly rely on Axe-core–based automatic scores. There is no human or cross-tool validation, nor analysis of code correctness or executability after alignment. The gains therefore reflect optimization toward the reward function rather than clear practical improvements in accessibility.

### Questions
1. How well does the proposed method generalize to real-world accessibility requests beyond GPT-generated synthetic data?


2. Can the authors provide human or cross-tool evaluations to confirm that the observed improvements go beyond optimizing for the Axe-core reward?

### Soundness
2

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
This work proposes to train website coding LLM with a reward function that comes directly from an automated WCAG auditor (axe‑core), which measures the violations of accessibility defects in HTML/CSS. For training prompts, the authors synthesize UI requests with GPT‑4o‑mini to form UIReq‑6.8K spanning 68 application categories. For evaluation, they collect screenshots of public webpages, extract structured metadata, and prompt another GPT model to generate user requests from the metadata. The LLM coder is GRPO‑tuned against the accessibility reward. The models are scored by WCAG violations and the appearance of the generated website. The results show that the proposed method improves accessibility without degrading visual/semantic quality.

### Strengths
1. This work presents a novel application of GRPO for improving accessibility of generated websites.

2. The newly curated dataset may be useful for future work on website coding.

3. The proposed approach is simple and effective.

### Weaknesses
1. The technical depth is limited: Although the application of GRPO for improving accessibility of website coding is straightforward but somewhat incremental, training LLMs with hand‑engineered rewards is already well explored.

2. Using only one testset leaves generalization to unseen domains and request styles, especially those not represented during training, unclear.

3. Both training and eval prompts are GPT‑generated, and it's unclear that the synthesized requests are realistic. As seen in Figure 4, the many remain high‑level and omit concrete functional elements (e.g., dropdowns, buttons).

### Questions
1. Whether the RL training decreases the diversity of website generation? The code LLM may learn a specific UI style, which is a safe solution.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new method to align code generation LLMs to produce accessibility-compliant web UIs. The approach augments existing LLMs with a new reinforcement learning reward. The reward takes into account violations of accessibility as negative rewards and trains the LLM using GRPO. To evaluate the performance, the authors also curated a new benchmark consisting 300 real-world web UIs. Results show that the propose method outperforms the baselines including Claude Sonnet 4, reducing the inaccessibility rate by 60% over the base model, while preserving semantic fidelity and visual quality of generated UIs.

### Strengths
1.	The paper concerns accessibility as a new metric for text-to-UI generation, which is an important research aspect. 
2.	The paper is generally well-structured. The motivation, methodology, and results are easy to follow. 
3.	The authors provide a new benchmark, RealUIReq-300, containing realistic web UI requests curated from 300 real-world web pages, which can be useful for future evaluation.

### Weaknesses
1.	The core methodological novelty is modest. The proposed method merely augments GRPO with a custom accessibility reward. While practically useful, this design does not introduce fundamentally new RL methodology or optimization mechanisms, making the novelty borderline for ICLR standards.
2.	The qualitative study in 6.2 is rather shallow. It provides only a single qualitative example (color contrast). A deeper case analysis, covering diverse violation types (e.g., ARIA roles, landmark semantics, keyboard navigation), would strengthen the narrative.
3.	The paper evaluates accessibility solely via automated tools (Axe-core). Given the human-centered nature of accessibility, even a small-scale user or expert study would greatly enhance credibility. 
4.	There is no ablation showing the contribution of different reward components or GRPO hyperparameters. It is unclear how sensitive the results are to the weighting scheme, the base score B, or the severity mapping.
5.	The work focuses narrowly on HTML-based UIs. The paper could discuss whether the proposed alignment strategy generalizes to other UI platforms (e.g., React, Flutter, mobile UIs) or broader code-generation tasks.
6.  There are also some recent related work on Web UI code generation, which can be discussed. For example:

UICopilot: Automating UI Synthesis via Hierarchical Code Generation from Webpage Designs, https://arxiv.org/abs/2505.09904

Unlocking the conversion of web screenshots into html code with the websight dataset. 2024. URL https://api.semanticscholar.org/CorpusID:268385510.

VISION2UI: A Real-World Dataset with Layout for Code Generation from UI Designs, https://arxiv.org/abs/2404.06369v1, April 2024.

### Questions
- Did you perform a user or expert study on accessibility? 
- Any ablation study showing the contribution of different reward components?

### Soundness
3

### Presentation
3

### Contribution
3
