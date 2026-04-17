# ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoning LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized chemical knowledge, ChemFG, annotating the presence of functional groups in molecules and the changes of functional groups during chemical reactions, to enhance the model’s understanding of the fundamental principles and internal logic of chemistry. Then, we propose a mix-sourced distillation method that integrates expertise in atomized knowledge with general reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the model's reliability, transparency, and practicality in real-world human-AI collaboration scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ChemDFM-R, a large language model for chemical reasoning that integrates atomized chemical knowledge with reasoning ability. The authors first build ChemFG, a 101B-token corpus that explicitly encodes functional–group–level information in molecules and reactions based on their developed toolkit for functional group identification for domain pretraining. Then they propose a mix-sourced distillation method combining chemical knowledge, pseudo-reasoning data, and rationales from general reasoning LLMs (Deepseek-R1, o3-mini), followed by domain-specific reinforcement learning to enhance interpretability and reasoning depth. Experiments on SciKnowEval and ChemEval show that ChemDFM-R significantly outperforms both general and chemical-domain baselines, producing accurate and transparent rationales that support reliable human-AI collaboration in chemistry.

### Strengths
- The paper demonstrates substantial technical and engineering effort, presenting a complete and well-executed pipeline that includes data construction, model design, and extensive evaluation.

- The idea of introducing atomized chemical knowledge from the perspective of functional groups is conceptually sound and offers a novel, fine-grained lens for enabling chemical reasoning.

- The proposed toolkit, ChemFG corpus, and the trained ChemDFM-R model are valuable community resources that can support future research in chemistry-focused LLMs and reasoning modeling.

- The model shows solid empirical performance, achieving strong results across multiple chemical benchmarks and demonstrating clear improvements over both domain-specific and general LLM baselines. Human studies are also provided for necessary experiments as justification.

### Weaknesses
- From a pure research perspective, while this work represents great engineering efforts, its research value may be limited. In my opinion, the research value of this work lies in the integration of functional groups as atomized chemical knowledge. There are not many insights or conclusions brought beyond that.
- I question the relevance of the inclusion of Figure 6. I understand that this section is used to support the claim of "human-AI collaboration". But I think this is a little hand-wavy by only presenting a case study. More statistical comparison against baselines would be more helpful. Otherwise, I would suggest removing this section to the Appendix and making the claim weaker in the paper.
- I think some experiments and additional analyses would further complement the paper. For example, the authors can consider running experiments on FGBench [1], which specifically targets functional group reasoning. Additionally, the claim made in lines 90-92 needs better support. I do not see from the current experiment results how the decomposed reasoning capability is improved. Experiment results for o3-mini would be beneficial since training data are distilled from it.

[1] Liu, Xuan, et al. "FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models." NeurIPS 2025 Datasets & Benchmark Track.

### Questions
- More details of the human annotation on quality control should be stated. For example, on what level are the three expert chemists, undergrads, grads, or researchers in the chemistry field? The exact accuracy could be clearly presented, with examples of error cases.
- Any intuition or reasons for choosing 70%, 22%, and 8% for data distribution in mix-sourced distillation?
- In Table 2, why is the performance improvement brought by reinforcement learning (ChemDFM-R) far less on SciKnowEval than ChemEval?

### Soundness
3

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
The authors present ChemDFM-R, a chemical-reasoning LLM centered on functional-group knowledge and trained in stages: domain pretraining, domain instruction tuning, mixed-source distillation, and RL; while explicitly preserving general language abilities. It retains performance on text-centric tasks and substantially outperforms its base model on molecule- and reaction-centric evaluations.

### Strengths
- S1: Strong concept: A chemical-reasoning LLM that handles a broad set of structure-level tasks (analysis, transformation, and property reasoning) is timely and useful.

- S2: Functional-group–centric reasoning: Framing reasoning around functional groups, rather than atoms or whole-molecule graphs, aligns with chemists’ mental models and can improve interpretability.

- S3 Capability retention: The effort to preserve general LM abilities (and to verify limited degradation on text tasks) is well-motivated; it justifies using an LLM rather than a narrow specialist and supports broader applicability.

### Weaknesses
- W1: Lack of uncertainty and significance.
The significance of results cannot be assessed without variability estimates. If these are Pass@1, please run ≥3 independent runs with different seeds and report mean ± s.d., plus ideally paired significance tests. Please do this at least for you model, to see if the others land within your error bars.

 - W2: Code (or full specs) unavailable. 
 Please release training and evaluation code (an anonymized repo is fine). If code release isn’t possible, provide a complete appendix with information such as exact prompts/templates and inference settings to ensure reproducibility.

 - W3: Even if near-contemporaneous, cite early chemical-reasoning LLM preprints (e.g., ether0) in Introduction/Related Work for completeness and positioning.

 - W4: No limitations/outlook.
Add a brief Limitations/Outlook section covering scope and failure modes and outline mitigation or future directions.

- W5: Compute and data budget missing.
Please report the compute footprint and the number of tokens used in each training stage.

### Questions
- Q1: What do you mean by "an average instruction-entry ratio of 1:50" on L197?

- Q2: Do the SciKnowEval / ChemEval suites include tasks absent from your training data? If yes, quantify: how many (or what fraction) are exact overlaps vs. novel tasks, and how does performance differ between seen and unseen task types? If no, please add an evaluation on tasks not explicitly trained on to support claims of generalization.

- Q3: Please use a common y-axis across subplots in Fig. 5; the current scaling overstates the impact of distillation+RL. Also, due you have any hypothesis for why the second phase yields much smaller gains than the first.

- Q4: What metric is shown in Tables 2,3 and 4?

- Q5: You currently group domain pretraining+instruction tuning and distillation+RL. It would be interesting to get the effect all four steps independently, especially disentangling distillation vs. RL effects.

- Q6: How confindent are you in the "reliability" of your chemical reasoning chatbot? How many of these multi-step discussion did you do and analyze?

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
4

### Summary
This paper presents ChemDFM-R, a LLM for chemistry, built by fine-tuning Qwen2.5-14B through a four-stage pipeline: (1) domain pretraining on a functional-group-centered chemistry corpus (ChemFG), (2) instruction tuning, (3) reasoning distillation with functional group rationales and teacher LLMs, and (4) domain-specific RL with a reward that promotes correct, well-structured reasoning. The model demonstrates strong performance on several chemistry benchmarks and introduces a novel dataset focused on functional group transformations.

### Strengths
1. The integration of functional group annotations in both training and evaluation is novel and impactful.

2. The paper performs careful evaluation across benchmarks and multiple tasks, and shows competitive or superior performance.

3. The RL finetuning with structured reward is well-aligned with reasoning goals.

### Weaknesses
1. Some implementation details, such as reward shaping specifics, filtering of rationale data, are not specified.

2. The current evaluation focuses mainly on answer correctness. More human evaluation of rationale quality could make it more complete.

3. The datasets used to train the model are not released, which can hinder the reproducibility.

### Questions
1. How does the model handle conflicting functional group cues when multiple transformations are possible?

2. Are there examples where the model hallucinates chemically invalid rationales despite producing correct answers?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a three-step approach to training chemical reasoning models, consisting of: a) pretraining on unstructured chemical data, b) supervised fine-tuning (SFT), and c) reinforcement learning (RL). For (a), the authors collect a pretraining dataset composed of general text corpora from the chemical domain, as well as functional-group annotated molecules and reactions. SFT is performed on three task categories: molecule-centered tasks, reaction-centered tasks, and general-knowledge tasks (e.g., QA questions derived from scientific publications). The authors demonstrate on two benchmarks that their model achieves competitive results, and that each training stage consistently improves performance when comparing models before and after the respective steps.

### Strengths
- **(S1 - relevance/novelty) - Training data as a valid contribution.** The authors provide both unstructured data for pretraining and SFT data within the chemical domain. This constitutes a potentially valuable contribution to the community. (However, the quality of the data cannot be assessed, as no supplementary material has been provided.)

- **(S2 - significance) - Results indicate the effect of the deployed training stages.** The results suggest that all three training stages (pretraining, SFT, and RL) contribute to improved performance (however, see (W3) and (W4)), resulting in a final model that is competitive with state-of-the-art generalist LLMs.

### Weaknesses
- **(W1 - relevance, novelty, quality) - Core research question has already been answered by prior work.** The main research question — whether models can be trained to reason about chemistry despite chemistry-specific challenges — has been addressed previously [1]. The authors completely ignore this, setting a misguided focus for their work and effectively re-answering an already resolved question. In fact, the authors present an alternative approach to developing a chemical reasoning model, where the main difference lies in the inclusion of pretraining on domain-specific data — a step that [1] does not include. A more interesting research question would have been whether the proposed approach constitutes a more efficient training procedure compared to [1], e.g., in terms of training FLOPs. However, the authors never explore this question and do not compare to [1], making (a) the manuscript low in novelty and (b) the proposed approach of unclear relevance.

- **(W2 - quality) - Claims not backed by insights or references.** Several claims appear overstated and are not substantiated by prior work. For example:
    * "The reasoning-before-answering pattern directly demonstrates how and why the LLM arrives at the answer" (l40f). While reasoning traces may indeed provide insights, this statement feels overly strong. Especially for long traces, it is highly unclear which tokens actually contribute to the final answer.
    * "The advanced domain knowledge is typically insufficient in general-purpose corpora" (l50f). This is the central claim on which the authors’ approach is based. However, it is not supported by a) literature references or b) empirical results in this manuscript. Regarding a), the training data of the base models are largely unspecified. Regarding b), since knowledge about the base models’ training data is missing, their performance can serve as a proxy. As shown in Table 2, the base model appears to include some chemistry knowledge. Notably, Chem-DFM-I outperforms the base model, so it is expected that RL training with Chem-DFM-I would be more efficient. However, it remains unclear whether this is truly the case and whether it justifies the additional pretraining step.

- **(W3 - significance) - Missing error bars and statistical tests.** All results are reported without error bars or statistical tests. Therefore, observed performance differences might have arisen by chance. While the proposed training procedure is computationally expensive and multiple training runs may not be feasible, error bars could at least have been reported across tasks or samples.

- **(W4 - relevance/quality) - Potential problem of data leakage.** Since the presented experiments aim to measure reasoning capability rather than memorization, the authors should ensure that no molecule or reaction appearing in the evaluation benchmarks is included in the pretraining or SFT data.

### References
* [1] Narayanan. *Training a Scientific Reasoning Model for Chemistry.*

### Minor comments
* Functional groups are not well defined. Although the Appendix provides more details, the main manuscript might incorrectly imply that the concept of “functional group” is unambiguous. The authors should clarify this.
* Details of the RL training step are insufficient. The authors mention a multi-task RL scheme, but its realization is not described. For example: were different tasks randomly sampled? Did the authors employ curriculum learning, and if so, did individual tasks require distinct curricula? The exact training data used also remain unspecified.

### Questions
- Annotating functional groups is known to be an NP-hard problem. Could the authors please provide more details on their annotation mechanism or model?

- The effectiveness of the functional-group annotated data remains unclear. What performance gain does pretraining only on the unstructured chemical text yield? Conversely, what happens if pretraining is performed solely on the functional-group annotated data?

- What is the completion length of the reasoning traces? I ask because all traces were reportedly checked by human experts, which may indicate that the traces are relatively short. Notably, recent work has shown a clear correlation between the reasoning budget and model performance.

### Soundness
2

### Presentation
2

### Contribution
2
