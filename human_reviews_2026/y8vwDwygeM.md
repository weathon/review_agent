# Eliminating Agentic Workflow for Introduction Generation with Parametric Stage Tokens

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
In recent years, using predefined agentic workflows to guide large language models (LLMs) for literature classification and review has become a research focus. However, writing research introductions is more challenging. It requires rigorous logic, coherent structure, and abstract summarization. Existing workflows often suffer from long reasoning chains, error accumulation, and reduced textual coherence. To address these limitations, we propose eliminating external agentic workflows. Instead, we directly parameterize their logical structure into the LLM. This allows the generation of a complete introduction in a single inference. To this end, we introduce the Stage Token for Introduction Generation (STIG). STIG converts the multiple stages of the original workflow into explicit stage signals. These signals guide the model to follow different logical roles and functions during generation. Through instruction tuning, the model learns the mapping between stage tokens and text functions. It also learns the logical order and transition patterns between stages, encoding this knowledge into the model parameters. Experimental results show that STIG can generate multi-stage text in a single inference. It does not require explicit workflow calls. STIG outperforms traditional agentic workflows and other baselines on metrics of semantic similarity and sentence-level structural rationality. The code is provided in the Supplementary Materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes STIG (Stage Token for Introduction Generation), a model that eliminates external agentic workflows in scientific paper introduction generation by parameterizing writing logic into LLMs via 8 stage token pairs.

### Strengths
- Provides an incisive, task-specific analysis of key limitations in academic introduction generation ,such as cascading errors and high computational overhead of traditional agentic workflows.

### Weaknesses
- Methodological novelty is insufficient. STIG merely transforms agentic workflows into single-step inference via training, sacrificing flexibility and generalization without achieving significant improvements in generation quality.  
- SS and NQ fail to fully demonstrate the method’s advantages, raising doubts about the reliability of semantic similarity to the original text as a metric; GPT-2-derived perplexity for NQ lacks robustness, and sampling perplexity across multiple models plus human expert calibration are recommended.  
- Experimental models and datasets are limited. Only small open-source models are tested, with no verification of adaptability to models of different parameter sizes, limiting result generalizability.

### Questions
- Why is the no-citation constraint adopted in experiments? The rationale for this exclusion needs further explanation.
- For baselines like Pure Prompt/ELABORATE Prompt, why were their original paper model settings not adopted? Why not compare with baselines on closed-source/large-parameter models (e.g GPT-4), as small open-source models may not reflect real baseline performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a pipeline (workflow) that employs a parametric stage-token–based prompting strategy on a fine-tuned STIG LLM for generating academic paper introductions. Experiments are conducted on a new dataset derived from 3,800 ACL papers, annotated with multi-stage writing structures corresponding to different rhetorical functions. By integrating workflow logic into trainable stage tokens, STIG unifies agentic reasoning and fine-tuned modeling within one pipeline. According to the reported results, it achieves higher semantic and structural quality with greater token efficiency than prompt-based and multi-agent baselines.

### Strengths
S1. The paper introduces an original idea that embeds workflow logic directly into model parameters via parametric stage tokens.

S2. This approach reduces multi-agent dependency and improves inference efficiency in structured text generation.

S3. The dataset of 3,800 ACL papers is large and well-structured. It represents a meaningful contribution for future research in the field.

S4. Five multi-dimensional evaluation metrics comprehensively capture semantic, structural, and narrative quality.

S5. Figures and examples illustrate the pipeline clearly, and the paper is generally well written and easy to follow.

### Weaknesses
W1. The paper lacks clarity in distinguishing between the STIG framework and the STIG fine-tuned model.
While the conclusion claims STIG eliminates agentic workflows, the framework still depends on them for data construction and stage definition.

W2. Fine-tuning details are incomplete. No hyperparameter settings, training configurations, or sensitivity analyses are reported. 

W3. A hyperparameter study is essential to confirm the stability of stage-token fine-tuning. All five evaluation metrics are newly proposed, but no external validation or human correlation study is provided.

W4. The main table and ablation experiments include too few baseline models. This makes comparisons less comprehensive.

W5. It is unclear if the proposed approach can be generalized to other academic domains rather than ACL papers.

### Questions
Q1. Is it possible to include an experiment or visualization that analyzes the internal weighting or influence of each stage token?

Q2. Is it possible to add a hyperparameter study on the number of stages (e.g., four vs. eight) to understand whether performance depends on workflow granularity?

Q3. Is it possible to expand the literature review to include prior research on LLM agentic workflows for paper writing? Current discussion on “LLM agents” and “LLMs for writing” is a bit general. 

Q4. Is it possible to conduct experiments with closed-source models (e.g., GPT-4 or Claude) on the new dataset and metrics?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes STIG, a single-inference method that replaces multi-turn “agentic” pipelines for writing the Introduction section of CS papers. Instead of orchestrating separate agents sequentially, STIG embeds stage tokens directly into the LLM’s parameter space via supervised fine-tuning. An 8-stage token scheme forces the model to emit Background-outline, Background-content, …, Contributions-content in single decode. Trained on 3.8 k ACL papers and evaluated on 1.2 k ACL-2025 test papers.

### Strengths
1. STIG outperforms several training-free agentic baselines (AutoSurvey, Outline-Writing) in structural rationality, content coverage, while using fewer tokens.

2. First work to parameterise an entire writing workflow into stage tokens. 

3. Contribute a customized dataset tailored for training and testing introduction generation, derived from over 3,800 ACL main conference papers.

### Weaknesses
1. Trained only on ACL NLP papers; no CV, Theory or other domains. Claims “research introductions” but evidence is NLP-only (ACL).

2.  The eight stage tokens are defined for the four subsections that appear tailored to research-style papers; however, ACL also contains many dataset papers whose introductions do not necessarily follow the Background–Problem & Limitations–Method & Experimental Results &  Contributions structure. 

3. The 'SR' metric is aligned with STIG’s own staged structure, making the comparison appear unfair.

### Questions
1. Although stage-by-stage agentic workflows may suffer from error accumulation, they allow targeted evaluation at each stage, which can improve the final output. STIG's end-to-end generation seems to leave no room for intermediate refinement?

Other questions refer to weakness.

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
This paper proposes the Stage Token for Introduction Generation (STIG) to automatically write research introductions and eliminate the external agentic workflows. STIG converts multiple stages of the original workflow into explicit stage signals so that it can generate multi-stage text in a single inference. STIG uses the title, abstract, description of figures, description and table contents, and the abstracts of baseline references to guide structured output. To train the STIG model, the paper construct a dataset from ACL main conference papers. STIG outperforms traditional agentic workflow and other baselines on automatic evaluation metrics like semantic similarity and structural rationality.

### Strengths
- The paper proposes STIG, a method that can combine multi-stage agentic workflow generation of research writing into a single inference pass.
- The paper constructs a high-quality dataset from over 3,800 scientific papers from ACL main conferences, utilizing MinerU, GPT-4o, and the Semantic Scholar API.

### Weaknesses
The evaluation metrics are insufficient. The reliance on automated metrics without human validation means we don't actually know if STIG produces good introductions. We only know that it produces text that scores well on these specific metrics. Furthermore, for scientific writing, one of the most important metrics that you should consider evaluating on is the factual accuracy (whether the claims in the introduction is accurate, whether there is fabricated content, etc). Furthermore, I am not sure BERTScore and perplexity from GPT-2 models are suitable for evaluating this task, because BERTScore may not capture long-form coherence needed for introduction writing that well, and GPT-2 is a very outdated model. (I do appreciate the example generations of the STIG models and AutoSurvey models in the appendix.)

### Questions
- Figure 3 seems a bit abrupt at the context. If possible, I strongly recommend you put a good example of STIG model’s generation here, as the purpose for this paper is to promote STIG, not criticize AutoSurvey.
- Did you validate the quality of GPT-4o annotations?
- Can you provide human evaluation results? Even a small-scale study (e.g., 50 papers rated by domain experts) would significantly strengthen the claims about generation quality.

### Soundness
2

### Presentation
2

### Contribution
3
