# Are Medical Vision–Language Foundation Models Ready for Dermatology

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Medical Vision-Language models (VLMs) show significant promise for clinical image understanding, offering the promise of greater medical accessibility and interpretability. However, a critical performance gap in diagnostic accuracy exists between their strong vision encoders and the full multimodal model. This performance gap suggests that such VLM fails to make full use of the strength of its vision branch. Such misalignment also implies that these models often over-rely on their language priors, producing plausible-sounding diagnoses without sufficiently grounding their reasoning in visual evidence. Focusing on dermatology, we systematically investigate the root causes of this phenomenon. While fine-tuning can improve accuracy, it often compromises the model's essential reasoning capabilities. To address these challenges, we introduce a training-free inference pipeline designed to close the performance gap while preserving the model's reasoning abilities. Our pipeline enhances diagnostic accuracy and faithfulness without requiring additional training. These strategies are readily extensible, suggesting a path toward more reliable and interpretable VLMs in medicine and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the performance gap between the vision encoder capabilities and end-to-end diagnostic accuracy of MedGemma-4B in dermatology.

### Strengths
1. The performance gap between vision encoders and full VLMs in safety-critical medical applications is a significant issue that deserves investigation.
2. Evaluation across five diverse datasets (SD-260, Fitzpatrick17k, Derm7pt, eSkinHealth, PAD-UFES-20) covering different diseases, demographics, and data characteristics.

### Weaknesses
1. The generalizability claims are unsupported without evaluation on other VLMs (GPT-4V, LLaVA-Med, SkinGPT-4, SkinVL, Med-Flamingo, etc.).
2. Missing comparison with standard prompting baselines (few-shot prompting, chain-of-thought without describe-first).
3. The paper should clearly distinguish between truly training-free strategies (in-context, describe-then-decide) and the hybrid approach. There is misleading "Training-Free" and "Zero-Shot" Framing.
4. No confidence intervals, standard deviations, or significance tests reported despite "averaged over five runs."
5. No analysis of which improvements are statistically significant. And no ablation on key hyperparameters (why 8-shot?)
6. Table 1 shows LoRA fine-tuning dramatically underperforms linear probes (39.50% vs 57.22% on Derm7pt) . It is a surprising result deserves deep investigation but receives minimal analysis.
7. Evidence is circumstantial. The model describing diseases it cannot diagnose doesn't definitively prove vision training data issues - could be alignment/integration problems.
8. No evaluation of whether generated descriptions are actually useful for clinical decision-making. Only 100 images evaluated by a single dermatologist (no inter-rater reliability).

### Questions
1. Can you evaluate your strategies on at least 2-3 other medical VLMs (e.g., GPT-4V, LLaVA-Med, one other dermatology-specific model)? 
2. What percentage of test cases have the correct diagnosis outside the Top-5 candidates? How does this affect overall potential accuracy?
3. Can you provide statistical significance tests (e.g., paired t-tests or McNemar's test) for the improvements in Table 4?
4. Why does LoRA fine-tuning perform so much worse than a simple linear probe? Have you tried different LoRA ranks, learning rates, or full fine-tuning?
5. Can you provide concrete examples of "reasoning capabilities" that are lost during fine-tuning? Include qualitative comparisons of model outputs.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates whether current medical vision-language foundation models (VLMs) are ready for dermatology applications. The authors identify three potential sources of failure: 1. distribution mismatch between training and target data, 2. overreliance on language priors, and 3. misalignment between vision and language objectives.
To address these, they propose two prompt-based inference strategies—adding clinical context descriptions and a “Describe-then-Decide” reasoning approach—and a two-stage pipeline that combines a visual probe with VLM-based reasoning. The motivation is well-grounded: existing medical VLMs often perform poorly in diagnosis because they rely too heavily on textual correlations instead of actual image evidence.

### Strengths
1. The paper tackles a highly relevant problem: understanding and mitigating the role of language priors in medical VLMs. This is crucial for deploying such models in safety-critical clinical contexts.

2. The proposed techniques, though simple, lead to consistent accuracy gains across multiple dermatology datasets and improve the performance by a decent margin.

### Weaknesses
1. The work focuses mainly on adjusting prompts and modifying the inference pipeline rather than introducing any novel algorithmic or modeling framework. Although the paper correctly identifies the entanglement between image and language priors as a core issue, it does not explore how to decouple these modalities at the representation or optimization level.

2. The proposed strategies (prompt reformulation and two-stage decision) highly tailored to dermatology tasks and to specific models such as MedGemma. It is unclear whether these improvements would generalize to other modalities, datasets, or medical domains.

3. While the paper attributes VLM underperformance to language priors, the presented evidence does not conclusively prove that the three identified factors are the primary causes. Other factors, such as inadequate visual feature alignment or data imbalance, might explain the same phenomena.

### Questions
1. Do we have any further experiments on other medical image datasets except the dematology datasets. 
2. Do you think radiology dataset also have the same problem?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The presented work explores the ability of the MedGemma-4B vision-language model to effectively diagnose dermatological diseases. Initially, the authors show that MedGemma performs substantially worse than its MedSigLIP vision encoder tuned with linear probing. They postulate that this alignment gap cannot be overcome using model fine-tuning via LoRA. In response, they propose three training-free strategies to improve MedGemma’s performance on five different dermatological datasets. The first strategy includes short summaries of the diseases’ visual features in the prompt to MedGemma. The second strategy prompts MedGemma to first describe the visible pathological features in the provided image before linking these to a diagnosis. Finally, the authors split report generation into a two-stage pipeline: initially, the pre-trained image encoder is used to identify the five most likely diagnoses before the VLM is asked to make the final classification. Each of these strategies is shown to individually improve MedGemma’s performance, while their combination achieves the highest overall performance increase.

### Strengths
- Evaluating the performance of foundational vision-language models in real-world clinical settings is of interest to both the machine learning and medical scientific community.

- The proposed training-free strategies are shown to substantially improve the performance of MedGemma.

- The authors conduct extensive benchmarking experiments involving five different dermatological datasets as well as further human rater and ablation studies.

- The manuscript is pleasant to read with its clear narrative-driven structure, many informative figures and tables, and high-quality writing.

### Weaknesses
- Prompting strategies for VLMs have been widely researched. For example, requesting structured reports has already been proposed (Delbrouck et al. “Automated Structured Radiology Report Generation.” Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025), as has providing detailed clinical guidelines to VLMs (Holland et al. "Specialized curricula for training vision language models in retinal image analysis." NPJ Digital Medicine 8.1, 2025).

- In many cases the authors evaluate both the image encoder and VLM in a few-shot testing scenario, arguing that this reflects “real-world medical applications where labeled data is scarce.” However, I am not sure whether fine-tuning with single-digit-number of images is representative of a realistic deployment scenario.

### Questions
- The authors only include very little technical details how they conducted the LoRA fine-tuning of MedGemma.

- I was confused that the option to output “not sure” appears in some of the depicted prompts (e.g. Figure 4) but not in others (e.g. Figure 3 or Table 3). I understand that space is limited in figures, but at the very least the authors should denote incompletely listed prompts with ellipses.

- The authors should provide more detail and context on the human evaluation study. Without any baseline method for comparison or even knowing the instructions to the rater and evaluation criteria, it is difficult to interpret the results provided in Figure 6.

- It appears that in Section 3.3 the authors make two contributions: using a pre-trained vision encoder to select the five most likely diagnoses as well as prompting the VLM to output a structured report. These contributions should be disentangled, and their benefit should be measured and reported individually.

- I had to read Table 2 several times to map the column names to the individual methods. I believe that the authors should select more expressive short descriptors for the different strategies.

- Additionally, I feel that the inclusion of the top-5 to top-1 strategy in Figure 1 is premature as it has not been introduced yet when the figure is referenced in the introduction. Especially since several of the conducted experiments are reported in multiple figures and tables.

- On a minor note, the order of the datasets in Figure 1 and the Tables differs.

### Soundness
3

### Presentation
3

### Contribution
2
