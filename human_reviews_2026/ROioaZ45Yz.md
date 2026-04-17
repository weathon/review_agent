# When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2, 6

## Abstract
Preference alignment is an essential step in adapting large language models (LLMs) to human values, but existing approaches typically depend on costly human annotations or large-scale API-based models. We explore whether a weak LLM can instead act as an effective annotator. We surprisingly find that selecting only a subset of a weak LLM's highly confident samples leads to substantially better performance than using full human annotations. Building on this insight, we propose **C**onfidence-**W**eighted **P**reference **O**ptimization (CW-PO), a general framework that re-weights training samples by a weak LLM’s confidence and can be applied across different preference optimization objectives. Notably, the model aligned by CW-PO with just 20\% of human annotations outperforms the model trained with 100\% of annotations under standard DPO. These results suggest that weak LLMs, when paired with confidence weighting, can dramatically reduce the cost of preference alignment while even outperforming methods trained on fully human-labeled data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates a cost-effective alternative to human annotation or large-scale model annotation for LLM preference alignment. The authors propose that a weak LLM can serve as an effective annotator when its confidence is taken into account. The core contribution, confidence-weighted preference optimization (CWPO) framework, which re-weights the training samples based on the confidence of a weak LLM's preference predictions. The key finding is that by using only a highly confident subset of a weak LLM's annotations, a stronger model can be aligned more effectively than with full human annotations. The empirical results, primarily using DPO (CW-DPO), show that this method outperforms models trained on 100% human-labeled data, even when the weak annotator is trained on only a small fraction (eg: 20%) of that human data.

### Strengths
- The method is novel and practical to reduce the cost of preference alignment. The idea of using a weak, computationally inexpensive LLM as an annotator is interesting. 
- The proposed annotation framework is simple and intuitive, which could also serve as a plug-and-play enhancement for existing preference optimization methods like DPO without altering their core algorithms.

### Weaknesses
- The evaluation relies on a pretrained GRA. While standard, this introduces a potential circular dependency as the reward model itself is a product of preference alignment. I think a analysis on human evaluation or performance on downstream tasks would strengthen the paper.

- The paper demonstrates robustness to the split ratio of labeled data, but it is unclear the method's sensitivity to the initial quality of the weak annotator. For example, how does performance degrade if the weak model is of extremely low quality or trained on a highly noisy/biased subset of human data?

- The generalizability of the framework to other domains, such as creative writing, reasoning, or code generation, remains unexplored. It is unclear if the confidence of a weak model is a reliable proxy for quality in other doamins.

- There is a potential risk of bias amplification. If the initial human-labeled subset used to train the weak annotator contains systemic biases, the model may learn to be confidently biased. The framework could inadvertently amplify these biases by up-weighting samples that strongly align with them, which could be detrimental to the final model's fairness.

### Questions
- Could you provide a qualitative analysis or examples of the types of samples where the weak LLM is highly confident but human annotators might disagree or be inconsistent? This could offer deeper insights into why this method is so effective.

- How does the performance of CWPO vary with the architectural and size gap between the weak and strong LLMs? For instance, what happens if the weak model is only marginally smaller than the strong model, or if they belong to entirely different architectural families?

- The confidence weighting function C is normalized to [0, 1]. Have you experimented with other weighting schemes, for example, a function that more aggressively penalizes low-confidence samples or a temperature-based scaling of the confidence scores?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Confidence-Weighted Preference Optimization (CW-PO), a framework for aligning large language models by utilizing preference annotations from weak LLMs and weighting each training sample by the annotator’s confidence. The method demonstrates that leveraging highly confident samples from weak models leads to stronger alignment performance compared to full human annotation, even with far fewer labeled examples. Experiments show that CW-PO significantly reduces annotation cost and consistently outperforms both human-supervised and previous weak-to-strong alignment baselines across various datasets and model families.

### Strengths
* **Logical Completeness**: Starting from an analysis of experimental results, the paper finds that high-confidence samples provided by weak LLMs are beneficial for improving annotation quality and thereby enhancing large model training. Building on this insight, the authors propose the CW-PO method, whose feasibility is validated through extensive experiments
* **Innovation**: The paper proposes the Confidence-Weighted Preference Optimization (CW-PO) framework, which weights training samples according to the confidence of weak LLMs. This is a strong complement to existing large-model preference alignment processes.
* **Cost and Scalability Advantages**: The weak annotators typically contain fewer than 0.5B parameters, resulting in much lower annotation and inference costs compared to API-level large models and human annotators, and can be reused for large-scale batch annotation

### Weaknesses
* **W1**: The explanation of why confidence weighting outperforms filtering or human annotation is primarily based on experimental observations, with theoretical analysis remaining relatively superficial. The underlying mechanism could be further explored in future work.
* **W2**: Although high-confidence samples can improve performance, there is a lack of case analysis on the impact of systematic bias in weak models. For complex, diverse, and noisy tasks, limitations in generalization ability still warrant attention

### Questions
* **Q1**: In practical large model training, how does the confidence distribution vary across different tasks and domains? How does the CW-PO method perform under extremely imbalanced or heterogeneous task scenarios?  

* **Q2**: In real-world applications, how can the confidence threshold be dynamically selected, and is task-specific fine-tuning necessary?

### Soundness
3

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
This paper proposes using a weak language model as an annotator to estimate confidence scores that reflect human preferences. These confidence scores are then integrated into existing preference optimization objectives to improve alignment performance. Experiments demonstrate that the proposed Confidence-Weighted Preference Optimization (CW-PO) achieves stronger alignment than human-labeled baselines, while utilizing only 20–30% of the human-annotated data.

### Strengths
- The paper identifies that leveraging a small subset of highly confident samples can significantly enhance alignment performance.
- The proposed CW-PO method is well-motivated and closely aligned with the empirical observations discussed in Section 3.1.

### Weaknesses
- The paper lacks sufficient discussion of related works on weak-to-strong supervision and generalization [1][2].
- Some evaluation details are unclear, such as sampling robustness and result variance.
- The presentation could be more concise — many key methodological details are deferred to the appendix, which slightly weakens the readability of the main text.
- The comparison in Table 5 needs clarification, since DPO and SFT+DPO optimize generative policies for text prediction, while BT is a discriminative objective that predicts preference scores.

References:

[1] *Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model*, ICLR 2025.

 [2] *Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models*, NeurIPS 2024.

### Questions
1. Have you tested CW-PO when the weak and strong models come from different architectures (e.g., OPT → Qwen)?
2. The OPT model series may be outdated. Have you evaluated CW-PO using more recent architectures such as LLaMA?

### Soundness
3

### Presentation
2

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
This study proposes a method to align large language models (LLMs) using a limited amount of human alignment data. Specifically, a weaker LLM is trained as a reward model rather than directly training the target LLM on the small labeled dataset. This annotation model labels the unlabeled data, and the resulting pseudo-labeled samples, together with their confidence scores, are used to further train the target LLM with Confidence-Weighted Preference Optimization.

### Strengths
1. The paper is well-written and easy to follow, making it straightforward to understand the motivation, methodology, and results.
2. The authors conduct evaluations on multiple models, including OPT and Qwen, as well as on major datasets such as HH-RLHF and TL;DR, which strengthens the credibility and reliability of the reported results.
3. The proposed method is simple and easily applicable.

### Weaknesses
1. In this paper, a small model is trained as a reward model using policy logits through BT modeling, similar to reference-based DPO approaches such as SimPO. However, this is not a novel idea. The subsequent incorporation of confidence-weighted DPO training is also conceptually similar to prior works such as WPO. Moreover, the notion that small models can effectively perform reward modeling has already been demonstrated in works like WS-DPO. From this perspective, the novelty of the paper appears quite limited.
2. The paper lacks sufficient baselines. The comparisons are restricted to DPO-based approaches, but since the proposed method involves training a small model as a reward model, it should also be compared against traditional reward modeling methods. For instance, studies such as PairRM have shown strong results in training high-performing reward models using small models, and such baselines would provide a fairer and more comprehensive evaluation.
3. The paper offers limited analysis. It does not clearly explain how the labels generated by the weak llm contribute to improved data filtering performance. For example, the weak LLM’s generated labels might enhance performance because they yield more accurate preference labels, or simply because they transform the data into an easier-to-learn form. Without such analysis, it is difficult to determine why the proposed method works or what specific mechanisms drive the observed improvements.

### Questions
1. Would the proposed method still maintain high performance if applied to online reinforcement learning settings, such as iterative DPO frameworks?
2. Would this approach remain effective when applied to more complex datasets, such as UltraFeedback, which involve richer and more diverse preference signals?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a confidence weighting of preference samples in various preference optimizaiton strategies, inspiried by the fact that even weak LLMs can surpass human annotations when it comes to preference data. The framework first trains a weak LLM to directly output scalar values as the rewards for the prompt and response, which are then used to obtain preference labels on a large amount of unlabeled preference data. During preference optimization, the per-sample loss is weighted based on the confidence of the weak LLM. Empirical validation on three datasets are given.

### Strengths
- The idea is useful in practical scenarios and straightforward to apply
- The analysis presented in this paper aligns with them of prior art, whilst also adding further results

### Weaknesses
- Most analysis done in this work uses models from the same family. It isn't entirely clear whether the proposed method will generalize across different model families.
- It's unclear on how exactly the weak LLMs are trained. The equation makes it seem like a regression problem, but the footnote says classification. Could you clarify this part?

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
