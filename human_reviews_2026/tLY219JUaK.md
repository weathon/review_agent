# Existing Adversarial LLM Unlearning Evaluations Are Inconclusive

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Unlearning seeks to remove sensitive knowledge from large language models, with success often judged through adversarial evaluations. In this work, we critically examine these evaluation practices and reveal key limitations that undermine their reliability. First, we show that adversarial evaluations introduce new information into the model, potentially masking true unlearning performance by re-teaching the model during evaluation. Second, we show that evaluation outcomes vary significantly across tasks, undermining the generalizability of current evaluation methods. Collectively, these issues suggest that existing evaluations risk mischaracterizing unlearning success (or failure). To address this, based on our empirical findings, we propose two principles—*minimal information injection* and *downstream task awareness*—for future evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a critical analysis of adversarial evaluation methods for large language model unlearning. The authors argue that current evaluation practices are unreliable and can lead to inconclusive or misleading results. The central thesis is built on two primary findings: 1) The evaluation process itself, particularly through finetuning attacks (on retain data) and input-space attacks (via adversarial prompts), can inject new information into the model. 2) The success of an evaluation is highly sensitive to the specific task format. 

Through a series of empirical studies on standard benchmarks (TOFU, WMDP) and unlearning methods (RMU, NPO), the paper demonstrates these flaws. For example, it shows that finetuning on the retain set can spuriously improve accuracy on the forget set, and that conclusions about which unlearning algorithm is superior are completely contradictory depending on the evaluation task.

Based on these findings, the authors propose two guiding principles for future evaluations: (1) minimal information injection and (2) downstream task awareness. They also offer concrete recommendations, such as reporting an "injection budget" (via metrics like a Relearning Generalization Index (RGI) or a Prompt-Data Entropy Ratio (PDER)) and mandating cross-format metrics.

### Strengths
**Significance**: As unlearning becomes a critical component for LLM safety, privacy, and regulatory compliance (e.g., "right to be forgotten"), the reliability of evaluation protocols is paramount. This paper convincingly argues that a cornerstone of this subfield—adversarial evaluation—is built on shaky foundations. This is important for any researcher working on unlearning or, more broadly, on LLM safety evaluations.

**Clarity**: The paper is well-written with good clarity. The core arguments are presented upfront, and the supporting experiments are logical and easy to follow. The distinction between different MCQ evaluation methods (max letter probability vs. max text probability) is a subtle but critical point that the authors explain very well. The proposed principles and recommendations are clear and directly follow from the evidence.

### Weaknesses
**Practicality of Recommendations**: The proposed recommendations in Section 5 are conceptually sound, but their practical application needs more exploration. For instance, for the "Relearning Generalization Index (RGI)", the paper suggests a threshold $\tau=0.2$ is "reasonable" but doesn't provide a principled way to set this threshold.

**Limited Scope of Models and Methods**: While the experiments are convincing, their scope is naturally limited. The paper primarily uses smaller models (Zephyr-7B, Phi-1.5, Llama-3.2-1B) and two specific unlearning methods (RMU, NPO). To fully substantiate the claim that existing (plural) evaluations are inconclusive, it would be strengthening to demonstrate these findings on larger, more capable models (e.g., Llama-3-70B, Claude 3, GPT-4) and other classes of unlearning algorithms (e.g., gradient-based methods, exact unlearning approximations).

### Questions
You note that TOFU-MCQ was curated by prompting ChatGPT. Do you suspect this synthetic generation process is the primary source of the spurious correlations (e.g., ChatGPT using a consistent "style" for all authors)? Have you tried to quantify this, perhaps by training a simple classifier to distinguish between questions about different fictitious authors based on style alone? Would this issue be less prevalent in a "natural" (non-synthetic) unlearning dataset, if one existed?

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
This paper presents a critical examination of current evaluation practices for LLM unlearning, focusing on the reliability of adversarial evaluation methods. The authors argue that existing approaches such as finetuning attacks, input-space attacks, and memorization detectors often yield inconclusive or misleading results. Through systematic experiments, they identify two fundamental flaws in current evaluation pipelines: (1) information injection, where the evaluation process itself unintentionally teaches the model the very information it is supposed to have forgotten, and (2) task-format dependence, where evaluation outcomes vary drastically depending on whether the downstream task is multiple choice, open-ended generation, or another format.

### Strengths
1.  The paper identifies an important and overlooked problem in evaluating LLM unlearning, providing clear evidence that current adversarial methods can lead to unreliable conclusions.

2.  It proposes two practical principles, Minimal Information Injection and Downstream Task Awareness, offering useful guidance for more reliable and interpretable evaluations.

### Weaknesses
1. While the proposed Prompt-Data Entropy Ratio (PDER) serves as a useful heuristic for estimating explicit information injection, it may not adequately capture semantic-level reactivation. In many-shot jailbreak scenarios, models can reconstruct forgotten knowledge even when the prompt contains no explicit information related to the target content.

2. The observed performance improvement on the forget set after finetuning on the retain set may not necessarily indicate spurious generalization. Instead, it could reflect that the model is recovering its original task format, which was excessively forgotten during the unlearning process. In this view, the issue lies more with the unlearning algorithm—causing over-forgetting of task format—rather than with the evaluation method itself.

3. While the paper effectively highlights fundamental flaws in existing unlearning evaluations, it does not conduct a systematic re-evaluation of existing unlearning algorithms under the proposed principles. It remains unclear whether applying the new evaluation criteria would alter the relative performance rankings of popular methods.

### Questions
1. The paper effectively identifies biases in existing unlearning evaluations; however, it does not demonstrate how applying the proposed principles would alter these conclusions. For instance, it remains unclear whether controlling information injection would change the relative performance between methods such as RMU and NPO.

### Soundness
3

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
1

### Summary
The paper identifies two important issues in current evaluation practices for LLM unlearning.
First, the use of Adversarial LLM Unlearning Evaluation can unintentionally introduce target knowledge into the model during the testing phase, leading to inaccurate assessment of unlearning performance.
Second, the current evaluation metrics exhibit large numerical fluctuations under different conditions, revealing significant sensitivity and instability in the measurement of unlearning effectiveness.

The paper proposes two key considerations for improving LLM unlearning evaluation:
(1) minimizing data injection during the testing phase to avoid reintroducing unlearned knowledge, and
(2) incorporating task-aware estimation prior to evaluation to better account for downstream task sensitivity.

Given my limited familiarity with this specific research area, I do not feel confident providing a well-informed evaluation of this submission. I would suggest that the Area Chair assign this paper to a reviewer with more expertise in this topic.

### Strengths
Given my limited familiarity with this specific research area, I do not feel confident providing a well-informed evaluation of this submission. I would suggest that the Area Chair assign this paper to a reviewer with more expertise in this topic.

### Weaknesses
Given my limited familiarity with this specific research area, I do not feel confident providing a well-informed evaluation of this submission. I would suggest that the Area Chair assign this paper to a reviewer with more expertise in this topic.

### Questions
Given my limited familiarity with this specific research area, I do not feel confident providing a well-informed evaluation of this submission. I would suggest that the Area Chair assign this paper to a reviewer with more expertise in this topic.

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
3

### Summary
This paper evaluates the effectiveness of current unlearning evaluation methods and proposes two principles—minimal information injection and downstream task awareness—for future evaluations.

### Strengths
1. The paper provides a clear and well-structured summary of the definitions of unlearning.

2. I believe the proposal of evaluation principles for LLM unlearning is timely and quite valuable.

### Weaknesses
1. Would it be possible to conduct unlearning evaluation experiments on larger-scale models, such as a 14B-parameter model?

2. I believe that multiple-choice questions (MCQs) are not suitable for assessing whether a model truly retains or forgets knowledge [1]. Could you provide an additional set of experiments using *open-ended generation* to evaluate the unlearning effectiveness?

3. I strongly advise against using the TOFU dataset for unlearning research, for the following reasons:

   - All knowledge in TOFU is fictional, and must first be *introduced* into the model through finetuning. However, based on current work on mechanistic interpretability, finetuning is considered **ineffective at injecting new factual knowledge into model weights** [2].  
   - More importantly, the primary goal of unlearning is to remove knowledge acquired during **pretraining**, which is encoded in the model parameters—rather than removing knowledge artificially introduced afterward.



If the above weaknesses are addressed properly, I will reconsider my rating.

---
**References**

[1] Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above]  ACL 2025
[2] Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3
