# Identifying Truthful Inheritance in Family Models and Enhancing Truthfulness

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Recent advances in large language models (LLMs) have led to emergence of specialized multimodal LLMs (MLLMs), creating distinct model families that share a common foundation language models. 
This work investigates whether a core traits like truthfulness are inherited along this evolutionary trajectory. To quantify this trait, we employ linear probing on the models' internal representations. Our analysis of Vicuna and Qwen model families reveals a key finding: a strong correlation in truthfulness scores between LLMs and their finetuned MLLMs counterparts, even when they are finetuned or probed with different modalities and datasets. 
Building on this findings, we propose a soft gating method using the Truthfulness score to amplify the influence of these context-truthful heads to improve the context grounding ability while preserving the contributions of other heads.
We validate our approach on base LLMs on HaluEval benchmark, demonstrating an improved ability for context truthful reasoning.
We then show that the Truthfulness scores obtained from base LLMs can be effectively transferred and applied as a soft gate to its finetuned variants, demonstrating its improved performance on POPE and CHAIR benchmark. 
The performance gain from this transfer is comparable to that obtained by probing the MLLMs directly, highlighting the potential for a unified approach to enhance truthfulness across an entire model family.
Our work demonstrates a novel method for leveraging a model's inherent, inherited traits to systematically improve its truthfulness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates (1) whether the truthfulness of LLM can be inherited to the MLLM, and (2) whether it is possible to design a structure that leverages the truthfulness mechanisms to improves the MLLM's downstream performance on POPE benchmark. The experiments provided affirmative answers to both questions.

### Strengths
- The finding that truthfulness inherits from the LM to VLM is insightful.  
- The proposed TruthProbe improves the Acc, F1, Rec scores on the three POPE subsets (MSCOCO, A-OKVQA, GQA). It's an important advance for the interpretability field to show improvements in the model performance.

### Weaknesses
- The Introduction asks "Do these models inherit traits like truthfulness" which indicates other traits might also be studied, but only the truthfulness is studied in this paper. I recommend updating the questions in the abstract and the introduction correspondingly.  
- The TruthProbe method would benefit from a clearer explanation of details. The number of trainable parameters in the introduced soft gate could be written clearer. From my current understanding of Section 3, $S$, $\lambda$, and $g_l^h$ are the parameters. But how many of them are trained and how many of them are hyperparameters? I'd appreciate if more details are provided. Similarly, is this method applied during inference time?  
- Additionally, there are a lot of typos, and I think the paper could perhaps benefit from another round of proofreading:  
  - Line 193, "1" is it Table 1 or Figure 1?  
  - Line 233, "by" -> "By".  
  - Line 236, the LaTeX formatting is problematic here.  
  - Line 274, "equation 2" -> "Equation 2".  
  - It'd be great if the notations are unified. E.g., "Equation" vs "eq", "Figure" vs "Fig.", "Sec" vs "Sec."

### Questions
Please refer to my reviews in the above sections. Especially, clarifications about the details would be appreciated.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Previous work ITI shows that some attention heads in LLMs are highly associated with truthfulness, as they can distinguish the truthfulness of an input with high accuracy in linear probing. Following ITI, this paper finds that these truthfulness-associated heads are inherited in the MLLMs fine-tuned from a base LLM. They then propose a representation steering method to enhance the truthfulness of the MLLMs.

### Strengths
1. The authors pose the research questions clearly in the paper.

### Weaknesses
1. It seems that the first clarified contribution (Line 092) is not a contribution of this paper, as this is achieved in ITI. The paper simply applies ITI on Halu dataset and clarifies that the identified heads are context-truthful heads, not truthful heads.
2. One contribution of this paper is the finding on inheritance of truthfulness. However, how to derive the correlation scores that validate this finding is not explicitly clarified in Section 2.3. That is, only scores are given in Lines 236-240, while the definition or derivation of these scores are not clarified.
3. Overall, I feel this work is an application of ITI on the Halu dataset and MLLMs.

### Questions
1. In the experiment section, the authors show the effectiveness of the proposed TruthProbe on the fine-tuned MLLMs. Since the proposed method is designed to improve truthfulness for general LLMs/MLLMs, I wonder if there is any result showing the effectiveness on the base LLMs, especially compared with ITI? 
2. This paper shows that the truthfulness inheritance happens on fine-tuned MLLMs. I am curious if this also happens on the fine-tuned LLMs.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors investigate whether truthfulness mechanisms in large language models (LLMs) are inherited by their multimodal counterparts (MLLMs) and propose an intervention technique to enhance truthfulness performance. The authors employ linear probing at the attention head level to obtain truthfulness scores that indicate each head's responsiveness to truthful outputs. The authors then compute correlation coefficients between the truthfulness scores (probe accuracies) of base LLMs and their fine-tuned MLLMs to demonstrate inheritance. Building on these findings, they propose TruthProbe, a soft gating mechanism that amplifies contributions from high-truthfulness heads during inference by scaling their outputs in the residual stream according to normalized truthfulness scores. In the experiments, the authors evaluate the intervention method on the HaluEval benchmark for LLMs and the POPE benchmark for MLLMs.

### Strengths
- The paper presents an innovative perspective on truthfulness mechanisms and intervention techniques. The inheritance framework provides a novel angle to understand truthful behavior, and the findings from interpretability results inform the design of the intervention mechanism.

- The method is evaluated across multiple benchmarks and experimental settings, providing a comprehensive view of the proposed approach.

### Weaknesses
- The presentation of the paper could be improved to enhance clarity and include necessary methodological details.
  - The paper lacks clarity on the dataset splitting strategy for Table 1 evaluation. While the authors mention a 4:1 train/validation split for probe training and use a held-out validation set (20%) for hyperparameter tuning, it remains unclear whether Table 1 evaluates on truly independent test data or data potentially overlapping with the probe training/validation sets. 
  - In Figure 3, the evaluation results are obtained by aggregating across three benchmarks, but the paper does not specify how the aggregation is performed or provide breakdown scores for each individual benchmark.
  - Figure 2 presents truthfulness score breakdowns for each attention head across all layers, but the heatmap format offers a low information-to-noise ratio.

- The rationale and evidence should be provided to support the probing applied at the final answer token position, as it may not be the place where untruthfulness manifests.

- The methodology for evaluating truthfulness inheritance in MLLMs has limitations that weaken the paper's core claims.
  - Evaluating MLLMs' truthfulness mechanisms primarily through text-based hallucination benchmarks (HaluEval) with blank images does not provide meaningful insights into multimodal truthfulness. This evaluation setting does not reflect typical MLLM use cases, where models process informative visual content alongside text. The approach conflates text-only reasoning capabilities inherited from the base LLM with genuine multimodal grounding abilities.
  - The observed decrease in similarity when transitioning from text-only to multimodal settings (Figure 1(b) vs. Figure 3) may actually suggest that MLLMs develop new or altered truthfulness mechanisms distinct from their LLM counterparts, contradicting the inheritance hypothesis rather than supporting it.
  - The cross-modal similarity comparison does not constitute strong evidence for inheritance. The authors compare models trained from scratch (Mistral) with fine-tuned derivatives to demonstrate family-specific patterns, but this contrast is insufficient. Different base models trained on different datasets with different procedures naturally exhibit different internal representations. The low correlation with Mistral (≈0.02) could simply reflect differences in pre-training rather than validating that fine-tuning preserves specific functional properties. 

- The experimental results show marginal performance differences between intervened and non-intervened models (Table 2 and Figure 5), raising questions about the method's practical effectiveness. To establish that the intervention produces statistically significant improvements rather than noise-level fluctuations, the authors should report results from multiple runs with different random seeds and conduct appropriate statistical significance tests.

### Questions
Please see the above section.

### Soundness
3

### Presentation
2

### Contribution
2
