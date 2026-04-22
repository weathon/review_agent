# SelectLLM – Calibrating LLMs for Selective Prediction: Balancing Coverage and Risk

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Despite the impressive capabilities of large language models (LLMs), their outputs often exhibit inconsistent correctness and unreliable factual accuracy. In high-stakes domains, overconfident yet incorrect predictions can lead to serious consequences, highlighting the need for robust uncertainty estimation. To address this, we introduce SelectLLM, an end-to-end method designed to enhance the ability of LLMs to recognize and express uncertainty effectively. By integrating selective prediction into finetuning, SelectLLM optimizes model performance over the covered domain, achieving a more balanced trade-off between predictive coverage and utility.  Experimental results on TriviaQA, CommonsenseQA and MedConceptsQA show that SelectLLM significantly outperforms standard baselines, improving abstention behaviour while maintaining high accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SelectLLM, a method for selective prediction that allows LLMs to abstain from answering when uncertain, thereby balancing predictive risk and coverage. SelectLLM employs a dual-head architecture with separate decoding and selection heads. It is jointly fine-tuned using DPO for utility and a custom loss function for calibrated abstention. Extensive experiments on multiple QA benchmarks and LLMs show that SelectLLM outperforms existing baselines.

### Strengths
The paper addresses the critical problem of enabling LLMs to abstain when uncertain, which is fundamental for their safe deployment in high-stakes applications. The motivation is well-defined, and the proposed method offers a viable solution. The dual-head architecture, which decouples the generation task from the confidence estimation, is an elegant design choice.

### Weaknesses
1. The related work section omits several highly relevant papers on uncertainty quantification and selective prediction for LLMs, such as [1-3]. The paper doesn't provide citations for the LLMs used in the experiments, including Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen2.5-14B-Instruct, and DeepSeek-v3.

2. The paper introduces several key hyperparameters without adequate analysis. An ablation study on the modified risk terms in Eq.3 is mentioned (lines 270-271) but is not present in the appendix or main text. This lack of analysis makes it hard to understand why the method works and limits its claimed reliability.

3.  The evaluation of SelectLLM is confined to QA datasets. The paper provides no evidence or  discussion on whether the method generalizes to other tasks, such as summarization and open-ended generation. 

**References**:

[1] Uncertainty-aware Language Modeling for Selective Question Answering.  (Yang, et al., Arxiv 2023)

[2] Improving the reliability of large language models by leveraging uncertainty-aware in-context learning.  (Yang, et al., Arxiv 2023)

[3] Uncertainty in language models: assessment through rank-calibration.  (Huang, et al., EMNLP 2024)

### Questions
1.  Regarding the tone-confidence metric:

 - How would SelectLLM's performance be affected if a different or less powerful LLM were used to generate the tone-confidence preference labels instead of DeepSeek-v3?
 - Can the authors substantiate the reliability of the tone-confidence used to validate calibration in Section 5.4? 

2.  Regarding hyperparameters and implementation details:

 -   How sensitive are the results to the target coverage $c$ and the regularization hyperparameter $\lambda$? Can you provide risk-coverage curves for varying values of these hyperparameters?
 -   What is the specific architecture of the selection head $g(·)$ (e.g., a linear layer, an MLP)? What are the additional training and inference costs associated with this head?
 -   What is the justification for fixing the loss weighting parameter $\alpha$ to 0.5? Is there evidence that this choice is robust across different models and datasets?

3.  Regarding additional experiments and ablations:

 -   Could you provide the ablation study mentioned in the main paper (lines 270-271) that demonstrates the empirical impact of the additional risk terms introduced in Eq.3?
 -   Which component of SelectLLM is most critical to its performance gains? Is it the separate selection head, the explicit coverage constraint $c$ in the loss function, the modified risk formulation, or their combination?
 -   How might SelectLLM be adapted for generative tasks beyond QA? Have you performed any qualitative experiments on tasks like summarization or open-ended generation?

### Soundness
2

### Presentation
2

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
This paper proposes a confidence-head training strategy combined with the DPO loss. Their method is training a different head, which is for predicting the confidence and DPO loss for the generation. They evaluate their method on various datasets.

### Strengths
- Confidence Estimation is a timely and important topic.

### Weaknesses
- The paper is poorly written: The novelty is not properly presented. Some technical terms are misused, such as reward function at line 259. Sections are not separated properly: Section 4 only has one subsection, so why do you have a subsection?
- In line 256, the proposed loss addition to DPO is exactly the same loss in DPO because. logx-logy = log(x/y).
- Why do you use an additional model to set the ground truth confidence?
 - If the abstention decision is solely based on the confidence score coming from the confidence head, why do you combine it with DPO preference tuning?
- What is the purpose of defining the expected coverage? Do you evaluate it on your experiments?
- Please report TP/TN ratio rather than the actual magnitude.
- What is the novelty/new perspective of this paper with respect to the current literature?
- Citation of MARS on line 308 is wrong.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for training a confidence-estimation head in addition to the standard autoregressive decoding loss for an LLM. The paper proposes to use this confidence-estimation head to gate whether the model abstains from providing an answer. The loss function that is used to train the model is a combination of a DPO loss and a ‘select’ loss that attempts to maintain a target global coverage (i.e. abstention vs non-abstention) rate. The authors show that their approach achieves higher performance (defined by the sum of the rate of true positives and true negatives) than a range of prior methods, on three Q&A datasets, and across three open-source models.

Overall, in my view, there are several missing details in this paper, that make it difficult to properly evaluate. I hope that the authors are able to provide some of these details, as well as comprehensively answer my set of questions in the weaknesses below.

### Strengths
1. Strong performance on relevant metrics (TRUTH score).
2. Extensive set of baselines that are compared to.

### Weaknesses
There are several weaknesses in the paper, some of which I think are quite serious:

1. Line 270 states that “In the appendix, we include an ablation study to demonstrate the effectiveness of the two additional terms”, but there is no such appendix included.
2. There are missing experimental details, including hyperparameters such as learning rate, training epochs, optimizer. Most crucial is the missing batch size, because:
3. It is not clear how the empirical coverage (line 277) is calculated. Presumably this is not recomputed on the entire dataset at every training iteration. If it is estimated by the empirical coverage within a batch, then it is crucial to have a high enough batch size for this to be a low variance estimator; and I would like to have seen an ablation w.r.t. batch size.
4. What is the target coverage rate that is used in the experiments? I did not see this detailed anywhere.
5. It is also not clear to me exactly how the dataset is constructed. What constitutes a preferred and a dispreferred response? The statement is that a threshold of 0.7 is used and any response assigned a confidence score above that is considered ‘accepted’ and those below ‘rejected’. In which case, how are the pairs of {preferred, dispreferred} constructed precisely? If the pairing does not matter (e.g. it is done at random), then the use of DPO is not the most suitable choice to use; general margin maximisation algorithms such as [KTO](https://arxiv.org/abs/2402.01306) may be more appropriate.
6. The paper makes extensive reference to ‘human preferences’, yet, the experiments use a strong LLM (DeepSeek v3) to mark the confidence of the responses. The motivation of this design choice is not discussed in the paper at all. The use of this strong LLM labeller suggests that the method primarily benefits from distillation of confidence from a much stronger LLM.
7. Related to point 6) above, a standard approach of fine-tuning with LoRA to calibrate the LLM directly (as done in e.g. [[1]](https://arxiv.org/abs/2207.05221), [[2]](https://proceedings.neurips.cc/paper_files/paper/2024/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf) should be compared to the method used. It is claimed that the latter is done but there are no details given on the methodology used there, so it is impossible to judge if it is a fair comparison to the proposed method.
8. The discussion of and adjustment to the DPO loss function due to the possibility of the preferred responses’ likelihoods being reduced by the standard loss function would benefit from reference to prior works such as [[3]](https://arxiv.org/abs/2404.12358), [[4]](https://arxiv.org/abs/2404.04626), [[5]](https://arxiv.org/pdf/2402.13228), which all identify the issue as well as provide suggested ameliorations.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SelectLLM, an end-to-end method to calibrate LLMs for selective prediction by adding a selection head and training to balance coverage and risk, claiming strong results on TriviaQA, CommonsenseQA, MedConceptsQA. 

It begins with framing the risk–coverage trade-off in terms of four different outcome types (accept , reject) x  (correct, incorrect). It then motivates the need and pitfalls of abstention, and positions the proposed methods as optimizing that trade-off. The high level idea is to attach a selection head that reads the last hidden state of the question to produce a question-level confidence. The decoding head is trained with DPO, the selection head is optimized for risk–coverage, some illustrative cases are shown in table 1. Moving to more details into the method, the idea is to formalize coverage as proportion answered and risk as error rate over answered set, and then suggests DPO training with pairwise preferences, with the goal of a target coverage x and to abstain if the confidence is below a threshold. The selection head outputs confidence (between 0 and 1) from the last question token state. The loss is a combination of a DPO loss and a select loss. A modified empirical selective risk is defined with some additional terms (line 271), using a reward that prevents DPO from decreasing both probabilities. The two losses are combined with a detault weight of 0.5 (i.e. takes a simple average). 

For experiments, the following base models are chosen: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen2.5-14B-Instruct; QLoRA rank-16. The proposed method is compared against base, LACIE (DPO), LARS, MARS, TokenSAR, P(True), Semantic Entropy. The metrics used are TP/TN/Precision/Recall/Coverage, and TRUTH = TP+TN (upper bound 1000). For score-based methods, tune the threshold on validation to maximize TRUTH, then apply to test; non-score baselines accept anything unless model explicitly refuses. It is reported that SelectLLM substantially increases TN and Precision at the cost of Coverage/Recall; TRUTH is best across models/datasets. Some experiments for OOD are also reported.

### Strengths
The paper has a clear practical goal and contribution -- that of using train-time selective prediction for LLMs in a principled way, not just test-time thresholding.

The architecture proposed is simple and general. It has a single selection head with minimal changes needed to the decoding pipeline. 

The results show consistent empirical gains in TRUTH and precision across three base models and two ID datasets. The OOD results are also encouraging.

### Weaknesses
This is not really a weakness but cross my mind. Confidence uses only the question’s last token hidden state, it ignores evidence in the generated answer or early decoding signals. These can be highly informative for difficulty/uncertainty. While I understand that the authors made a choice, it may systematically miss cases where uncertainty emerges during generation (such as in multi-hop). Further, the authors argue that token probs are miscalibrated so a separate head is needed. However, then that head is trained without using token-level uncertainty or decoding statistics. I wonder if calibration performance could be better if so. 

Slight consistency issue: Earlier in the paper coverage is defined as the fraction answered. However, later, the empirical coverage is defined as the average of g(h), where g outputs a confidence score between 0 and 1 (not a binary accept indicator). This makes the empirical coverage an average score, not the fraction answered. This causes a slight consistency mismatch in what is argued and e.g. in the constraint terms (equation 4). 

There seems to be a circularity or leakage risk in the validation of confidence. Training pairs are constructed using DeepSeek-v3 "tone-confidence" thresholds and fallbacks. But later, SelectLLM’s confidence is validated by comparing to the same tone-confidence distribution (Fig. 3). This is not an independent validation signal. The risk is that it could be teaching the model to mimic tone signals and then "validating" that mimicry. For such empirical work this is a chicken and egg problem, but it would at least be useful to spell out. 

Using LDPO (pairwise margin) inside the selective risk is an indirect surrogate for correctness and might misalign with the risk–coverage target, especially OOD. (pages 5–6)

### Questions
See above. 

In addition. 

Why restrict g() to the question state only? Did you test variants that pool answer states or incorporate token-entropy/semantic entropy/variance across samples? If tested, please share,

### Soundness
2

### Presentation
2

### Contribution
3
