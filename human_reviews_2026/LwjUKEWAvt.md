# SafetyChat: Learning to Generate Physical Safety Warnings in Instructional Assistants

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
While large language models (LLMs) excel in language generation and conversational abilities, their broader utility hinges on meeting additional requirements to ensure reliability and safety. Recent research has explored areas such as minimizing hallucinations, grounding outputs in credible sources, and safeguarding user privacy. However, the critical aspect of physical safety has received limited attention—an oversight that becomes increasingly important as LLMs are integrated into multimodal voice assistants (e.g., smart glasses) that are capable of guiding users through complex, safety-critical tasks such as automotive repair. In this work, we investigate the limitations of current LLMs in generating effective and contextually appropriate safety warnings in the context of complex repair tasks. We introduce SafetyChat, a multi-domain dataset that can evaluate LLMs’ ability to model and prioritize safety awareness. We further enhance model alignment by post-training on this data, comparing the performance of various techniques. Through this process, we identify key challenges and establish robust baselines, paving the way for future research on integrating physical safety considerations into LLM-driven instructional systems. We will release data and code to reproduce our results on publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SAFETYCHAT, a real-world, multi-turn benchmark for physical safety warnings in repair tasks and shows that post-training on this data materially improves LLMs’ ability to anticipate and communicate relevant hazards, surpassing several baselines and setting foundations for safer instructional assistants.

### Strengths
Novel dataset. Proposes a multi-turn benchmark focused on physical safety in automotive and electronics repair scenarios.

Clear task formulation. Formulate the problem as warning classification and warning generation, enabling systematic evaluation.

Convincing empirical evidence. Shows that popular LLMs underperform on these tasks, while simple post-training such as SFT and DPO yields substantial performance gains.

### Weaknesses
Beyond the two works cited, there are several recent datasets/benchmarks for physical-scenario safety in embodied/agent settings (e.g., SafeAgentBench; Agent-SafetyBench; robot constitutions/semantic safety; task-planning safety frameworks)[1,2,3,4]. Even if tasks differ, the paper should clarify what is new or harder here and why the proposed two tasks are distinctly more important than existing benchmarks.

The dataset is text-only, yet the source corpora (iFixit-Auto, wikiHow, TSB, iFixit-Elec) and many related benchmarks are multimodal (include images). For repair scenarios, users often rely on photos to communicate context. The paper should justify the text-only choice, discuss what is lost without images, and consider a multimodal extension.

Parts of the dataset are generated/evaluated with GPT-4o. To avoid model-specific bias, the pipeline should incorporate multiple diverse LLMs (and/or human adjudication) and report agreement and robustness across models.

Missing baselines likely to be strong. (1) RAG baseline: Given the availability of procedural documents (iFixit/wikiHow/TSB), a retrieval-augmented approach (retrieve instructions → summarize applicable warnings) is natural and may perform well; if no relevant instruction is found, classify as safe. (2) Reasoning models: Since the tasks require deciding whether to warn and what to warn about, chain-of-thought / reasoning models (or reasoning-enabled decoding) are appropriate baselines.

The post-training relies on standard SFT/DPO; there is no algorithmic contribution. Consider framing physical safety as hard/soft constraints during training or decoding (e.g., constrained optimization, safety filters, or control-theoretic constraints) to increase novelty and rigor.

Evaluation clarity issues. (1) Table 4 caption: It says “GPT-4-as-judge,” but the table has two columns: “GPT-4o Judge” and “Claude-3.7 Judge.” Please reconcile the caption with the content and clearly describe how GPT-4 (vs. GPT-4o) is used (and where it is first introduced). (2) Formatting: In §4.2, spacing differs between Query and Resp; ensure consistent English spaces throughout formulas and text.

[1]Yin, Sheng, et al. "Safeagentbench: A benchmark for safe task planning of embodied llm agents." arXiv preprint arXiv:2412.13178 (2024). 
[2]Sermanet, Pierre, et al. "Generating robot constitutions & benchmarks for semantic safety." arXiv preprint arXiv:2503.08663 (2025). [3]Zhang, Zhexin, et al. "Agent-safetybench: Evaluating the safety of llm agents." arXiv preprint arXiv:2412.14470 (2024).
[4]Huang, Yuting, et al. "A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents." arXiv preprint arXiv:2504.14650 (2025).

### Questions
Did the authors encounter convergence or stability issues when training DPO given that preferred responses are human-authored while non-preferred responses are GPT-4o–generated (i.e., a clear source/distribution mismatch)? With a relatively small fine-tuning set, such shift can cause the model to learn source cues rather than preference signals and may hinder convergence.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SafetyChat, a novel, multi-domain dataset designed to enhance Large Language Models (LLMs) in generating accurate and contextually relevant physical safety warnings for complex, multi-step instructional tasks like automotive and electronics repair. The authors argue that existing LLMs often fail to adequately address real-world physical hazards during conversational guidance, an oversight that becomes critical as these models integrate into voice assistants. SafetyChat consists of conversational benchmarks grounded in authentic repair procedures, with human annotators providing gold-standard safety rewrites for LLM responses that missed necessary warnings. Experiments confirm that while off-the-shelf LLMs perform poorly in hazard identification and warning generation, post-training on SafetyChat significantly improves their safety awareness, demonstrating a clear path toward developing safer instructional AI assistants.

### Strengths
- The paper is well written and well structured
- The authors present a good analysis of related work and datasets, as well as a good understanding of the state-of-the-art
- The paper introduces a potentially useful dataset of considerable size (6,391 annotated turns across 528 repair procedures) for evaluating AI assistants that provide instructions with greater safety awareness
- The SafetyChat dataset includes 1077 human-authored rewrites to address cases of missing warnings from GPT-4o responses.

### Weaknesses
- The experimental procedure could eventually be improved with some simple baseline approaches (e.g., see question 1). Essentially, since the main contribution is a benchmark dataset, demonstrating the performance of a more considerable number of base models and techniques would make up for a much more informative experiment.
- Given the nature of the contribution of the paper, having access to the dataset/anonymized code repository would be quite useful. For this reason, I can't comment on reproducibility.
- The experimental setting evaluates the models on a hold-out set. However, in this setting, out-of-distribution performance can be quite important (i.e., how do models perform on safety-sensitive tasks unrelated to any of the categories included in the train/validation subsets?). Something as simple as using one of the categories as a hold-out set could give readers an idea of how generalizable fine-tuning a model with this dataset could be (or employing something like a k-fold validation, where a fold would be a different category, for example).

### Questions
1. How does a fine-tuned model on this dataset perform (safety-wise) on out-of-distribution tasks, comparatively to its base version? Similarly, how would a prompting approach (i.e., using a model with specific instructions to be extra conscious on safety procedures, which I believe was the authors' approach with GPT-4o?) perform, comparatively to the fine-tuning and the base model approach, without such instructions? How would a model perform if, instead of fine-tuning, one employs retrieval-augmented generation instead (using the train set partition you defined)?
2. How many annotators were used? What was the selection criteria? In my opinion, this type of information should be better detailed (if not in the main body, at least in the appendix). I couldn't find much information on the annotation phase other than the interface developed.

### Soundness
4

### Presentation
4

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
The paper introduces SafetyChat, a multi-domain, multi-turn benchmark for teaching/evaluating LLMs to insert physical safety warnings into instructional dialogues, mainly for automotive and electronics repair. It also shows that small open models fine-tuned on this data can beat prompted GPT-4o on warning identification and on generating safety-aware rewrites.

### Strengths
- The paper introduces an interesting dataset in a space that is still underexplored - task-oriented, physical safety in instructional dialogues (automotive/electronics). This is a useful complement to the more common policy/content-safety benchmarks.

- Using iFixit/wikiHow/TSBs plus AR-style chat simulation keeps the dialogues realistic (images, long steps, workshop vs DIY differences). The Ford TSBs in particular justify the “high-stakes” framing; few safety datasets actually contain OEM technical bulletins.

- The task formulation is good: Separating (i) safety-warning identification from (ii) safety-aware response generation mirrors how production systems are actually built.

### Weaknesses
- Everything is repair-like: automotive, electronics, all from 3 sources. It’s plausible that the model is just learning “car-repair warning priors” (always tell them to park & cool down) and “electronics warning priors” (unplug, discharge capacitor) rather than contextual reasoning. The paper claims “realistic, multi-turn” but never shows cross-domain or out-of-domain transfer to kitchen, DIY home improvement. A cross-domain evaluation would make the “first step toward physically safe assistants” claim much stronger.

- The paper does not clearly describe the safety/technical expertise of the annotators (e.g., whether they had automotive/electrical background or were trained annotators following a rubric). Since the task is about physical risk, clarifying annotator qualification and quality control is important.

- On the modeling side, there is limited methodological novelty - mainly SFT and DPO on the collected data. This is fine for a dataset paper, but the work would be stronger with richer baselines (e.g. retrieval-augmented warning injection). The authors optionally can compare the approach of this paper: AURA: Affordance-Understanding and Risk-aware Alignment Technique for Large Language Models. https://arxiv.org/abs/2508.06124

- The evaluation relies heavily on LLM-as-a-judge. For a physical-safety benchmark, a small human study (even 15–20 domain-informed mechanics/DIYers/technicians) to validate that the model’s warnings are appropriate and non-redundant would significantly strengthen the empirical claims.

**Minor Comments**

- Image usage. Procedures often include images; but the generation task seems text-only. Say explicitly whether images are in the public release (some iFixit assets aren’t).

- A very intuitive baseline is “prepend domain-specific boilerplate chosen by IR from the taxonomy” (retrieve top-k warnings by BM25 over the step).

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses an important and overlooked problem: the inability of Large Language Models (LLMs) to generate context-aware safety warnings when acting as instructional assistants for complex tasks with physical risks (e.g., automotive or electronics repair). The authors point out that even advanced models like GPT-4o often fail to provide critical physical safety instructions.

To tackle this issue, the paper makes two main contributions:
1.  **The SAFETYCHAT Dataset**: A novel, multi-domain (automotive and electronics repair) large-scale conversational dataset. It is built upon real-world repair guides (such as iFixit, wikiHow, and TSBs) and collected through multi-turn, role-playing dialogues between human annotators and GPT-4o.
2.  **Safety Alignment Experiments**: The authors performed Supervised Finetuning (SFT) and Direct Preference Optimization (DPO) on open-source models (e.g., Llama-3.1-8B) using SAFETYCHAT.

Experimental results show that models trained on SAFETYCHAT significantly outperform (and even surpass) GPT-4o on tasks involving the classification and generation of physical safety warnings. This demonstrates that alignment with high-quality, domain-specific data can effectively enhance the physical safety awareness of LLMs.

### Strengths
1.  **Importance and Novelty of the Problem**: The paper addresses a critical and under-researched area: the **physical safety** of LLMs. As models are increasingly integrated into smart glasses, AR/VR, or embodied agents, the ability to foresee and warn against physical hazards during instructional tasks is paramount.
2.  **High-Quality Dataset Construction**: The SAFETYCHAT dataset is a core contribution of this work. It is built on authoritative, real-world repair guides (including professional TSBs) and employs a rigorous collection methodology. Notably, having human annotators **rewrite** GPT-4o's responses that missed safety warnings provides an exceptionally high-quality training signal for SFT and DPO.
3.  **Effective Alignment Strategy**: The experiments demonstrate that alignment using SFT and DPO on a domain-specific dataset is extremely effective. It is noteworthy that the finetuned 8B model surpasses GPT-4o on physical safety tasks, suggesting that for specific safety concerns, targeted data and alignment are more effective than larger, general-purpose models.

### Weaknesses
1.  **The Inherent Paradox of LLM-as-a-Judge Evaluation**: A fundamental weakness lies in the evaluation methodology. The paper first establishes that GPT-4o is deficient in identifying physical safety hazards, yet it paradoxically relies on this same "incapable" model as the primary "judge" for evaluating the safety generation tasks. This contradiction undermines the validity of the results, as a small-scale human verification is insufficient to resolve the concern that the judge model has the very blind spots it is supposed to be evaluating.

2.  **Lack of a Multimodal Evaluation**: Despite introductory scenarios like "smart glasses" and the use of images during data collection, all experiments remain purely textual. The evaluation framework fails to assess the model's ability to perceive physical danger from visual context, which is a critical component of the very problem the paper aims to solve.

3.  **The Dataset Name "SafetyChat" is a Significant Overclaim**: The name implies a general-purpose safety model, whereas the work is narrowly focused only on procedural physical safety for automotive and electronics repair. This is misleading and potentially dangerous, as it falsely suggests the model is suitable for other safety domains (like social, privacy, or even medical safety) where it has no training.

### Questions
How do the authors view the generalization capabilities of models trained on SAFETYCHAT? For instance, could a model trained on automotive repair data handle a physical safety warning for "bicycle repair," which it has never seen? Has the model learned specific textual patterns for "jack safety" and "battery safety," or has it grasped a more general concept of "physical harm avoidance"?

### Soundness
2

### Presentation
2

### Contribution
2
