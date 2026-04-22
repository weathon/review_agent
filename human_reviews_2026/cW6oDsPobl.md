# Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Honesty alignment—the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence—is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, the latter demands costly, large-scale labeling.
We introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. This design substantially reduces annotation requirements while improving generalization across tasks. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals.
Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations ($\sim$0.18\% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed an annotation-efficient framework EliCal for the honesty alignment of LLMs.
And this paper also establish a large-scale benchmark HonestyBench for evaluating the performance of various confidence calibration methods. Notably, the proposed EliCal adopts a pretrain-then-finetune paradigm, which significantly reduces the need for labeled data by enhancing the generalization ability of confidence calibration.

### Strengths
1) This paper is well-written and easy to read.
2) After Elicitation-Then-Calibration, the model could express its confidence before generating any response token, which can efficiently reduce the computational overhead of sampling and consistency checking.
3) The method proposed by the authors is very simple and effective, while also easy to scale.

### Weaknesses
1) When constructing the Consistency Data, EliCal uses the greedy-search answer as the most confident response (Equation 5). However, the generation probability of the greedy-search answer is not necessarily the globally maximal. Using beam search to $\tilde r$ might be more reasonable.

2) It seems that authors does not explain how to evaluate whether two responses are semantically consistent in line 168 - line 175.

### Questions
1) In Equation 7, for $r$ from a sampled set $\hat{\mathcal
R} $, it is easy to obtain $p(r | q)$. Does using the estimation method in Equation 7 introduce additional error?

2) In Figure 2, the self-consistency confidence is highly correlated with true capabilities (Spearman coefficient = 0.789). Perhaps isotonic regression or platt scaling would be sufficient for calibration. It is recommended that the authors include experimental results using these simple baselines.

3) In line 252, does ETC means ELICITATION-THEN-CALIBRATION (ELICAL)?

4) The authors should show the performance of each model before and after confidence calibration in Table 3.

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
2

### Summary
The paper addresses the critical issue of honesty alignment in large language models, which is the ability of these models to accurately recognize their knowledge boundaries and express calibrated confidence. The authors introduce a two-stage framework called Elicitation-Then-Calibration (EliCal). This innovative approach first elicits the model's internal confidence using inexpensive self-consistency supervision and subsequently calibrates this confidence with a small set of correctness annotations. This method significantly reduces the annotation requirements while improving the model's generalization across various tasks.

### Strengths
1. The paper presents a novel approach to reward modeling by framing it as a text generation task. This innovative perspective allows for the generation of structured justifications for actions, which significantly enhances the interpretability of agent decisions.

2. The authors provide a thorough comparison of WebArbiter against existing models, showcasing its superior performance across various benchmarks. 

3. This paper addresses critical limitations in existing reward models and provides a more reliable framework for web agents.

### Weaknesses
1. The experiments primarily evaluate relatively small and weak open-source models (mainly LLaMA). It remains unclear whether the conclusions generalize to larger models (e.g., those with more than 30B parameters).

2. In Appendix G, there appears to be a missing or incomplete figure (indicated by “??”).

3. In the experiments, the authors use an LLM-as-a-judge approach to evaluate QA performance but do not provide any justification or evidence for its reliability.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To avoid exhausting human annotation in honest alignment, this work proposes an Elicitation-Then-Calibration strategy, which measures the self-consistency of LLMs’ multiple generations as an alternative signal for output correctness estimation and evaluation. The authors adopt a linear layer with the last hidden states for output consistency prediction. Experiments on ten datasets against seven baselines verify the effectiveness of the proposed method.

### Strengths
1. The authors regard the self-consistency of LLM outputs as an indicator of its confidence, which is further leveraged to assess the correctness of its generations. According to this setting, large-scale human annotation and LLM fine-tuning can be avoided.
2. Extensive comparison of 7 representative methods, including both training-free and traning-based, verifies the effectiveness on the HonestBench.

### Weaknesses
1. The method fundamental premise lacks sufficient theoretical justification and empirical validation regarding its effectiveness and robustness.
2. Important design factors such as the sampling number k, the necessity and contribution of Stage 2 (Confidence Calibration), and the accuracy of the linear-layer-based self-consistency prediction (e.g., MSE) are not adequately discussed and experimentally analyzed.
3. The evaluation lacks comparisons with recent baselines and omits results using true self-consistency (Eq. 7) versus predicted values (Eq. 8).

### Questions
1. My primary concern is regarding the actual effectiveness and validity of the basic assumption: the confidence of generation can be quantified through the self-consistency among the multiple outputs, given that it is the most important premise regarding the technical soundness of your method. Although the authors provide some reference for self-defense, more theoretical analysis and experiments are necessary to support this conclusion. 
2. How accurate or reliable is the self-consistency in estimating the correctness of LLM generations? Figure 2 does not present this relationship clearly or directly. Moreover, it is uncertain whether this prediction accuracy can lead to tangible performance gains in practical scenarios. As far as I am concerned, in most cases, unexpected or alignment-required outputs constitute only a small proportion, and thus, the potential improvement might be offset by incorrect predictions of the self-consistency.
3. Also, the number of sampled generations (i.e., k) seems important in your method. However, no discussion and experiments regarding this hyperparameter are provided.
4. Stage 2 Confidence-Calibration requires detailed elaboration. I am not sure of the necessity and improvements of this stage to the final performance.
5. How about the prediction accuracy of self-consistency through the linear layer? I recommend that the authors report the MSE metrics in experiments.
6. How would the performance on HonestBench change if the true self-consistency, computed directly from Equation (7), were used instead of the predicted values obtained from Equation (8)?
7. I suggest the authors supplement more recent baselines for a fair and comprehensive performance comparison.
8. I recommend the authors release the relevant data and code to enhance the reproducibility of the method.
9. Citation error: “Figure ??” (Appendix G, p. 20). Please perform a thorough proofreading pass and fix all broken cross-references.

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
The work proposes **EliCal (Elicitation-Then-Calibration)**, a two-stage framework for improving **honesty alignment** in large language models (LLMs)—the ability to express calibrated confidence aligned with correctness. Stage 1 (“Elicitation”) trains models to express internal confidence using *self-consistency* signals without annotations; Stage 2 (“Calibration”) fine-tunes this confidence using a small number of correctness-labeled examples. The work also introduces **HonestyBench**, a large-scale benchmark consolidating ten factual QA datasets (≈560k training and 70k evaluation pairs) annotated with both correctness and consistency signals. Empirical results show that EliCal attains near-upper-bound alignment performance using 0.18% of the labels required for full calibration, and can be generalized to unseen MMLU tasks.

### Strengths
1. **Well-motivated methodological design.** The proposed two-stage framework (EliCal) is well-motivated by the empirical correlation between self-consistency and correctness (Fig. 2), providing a principled basis for separating elicitation and calibration.
2. **Introduction of HonestyBench for universal evaluation.**  This benchmark combines 10 QA datasets, comprising 560k training samples and 70k evaluation samples, each annotated for correctness and self-consistency. It covers a variety of QA types—including single-hop, multi-hop, and template-generated questions—enabling systematic assessment both in-domain and out-of-domain.
3. **Significant experimental performance compared to the baselines.** The paper evaluates across three LLMs (Qwen2.5-7B, 14B; Llama-8B), showing consistent performance patterns (Table 2–4; Fig. 5–8), demonstrates strong cross-domain generalization to MMLU, and includes ablation studies probing the effects of elicitation dataset size and the limitations of the linear calibration head (Fig. 7–10).

### Weaknesses
1. **EliCal relies too heavily on the model's self-consistency performance.** However, self-consistency does not perform very well on all datasets [1], and for such datasets, this method still requires a large amount of labeled data for calibration.
2. **Even with LoRA used for training, the method is still very costly.** There is already a lot of work [2,3] applying token probability to open-ended QA tasks, and efficient post-hoc methods can be used for calibration, even without additional labeled data. In comparison, the cost of EliCal is much higher. Moreover, since it involves 2-stage training, this method is difficult to deploy on large-scale models in practice.
3. **The comparison of baselines may be unfair.** The purpose of EliCal is to assess the model's confidence in answering the current question correctly before the model generates an answer. However, in the baseline, both Prob and Verb require the model to generate an answer before providing a confidence score. I think there is a gap in the settings of these methods. 
4. **The ECE experiment results are incomplete.** Since ECE is a very important metric in confidence calibration, in addition to the ECEs of EliCal and Cal-Only, could you also report the ECEs of the other baselines?

[1] Manakul, Potsawee, et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." *The 2023 Conference on Empirical Methods in Natural Language Processing*.

[2] Shen, Maohao, et al. "Thermometer: Towards Universal Calibration for Large Language Models." *Forty-first International Conference on Machine Learning*.

[3] Luo, Beier, et al. "Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator." *The Thirty-Ninth Annual Conference on Neural Information Processing Systems.*

### Questions
1. **Can you compare ECE of the calibration method for token probability that I mentioned in the Weakness part with EliCal?** I'm curious how much the training method can improve compared to the post-hoc method.
2. **The setting of this article is a bit strange to me.** It is somewhat like uncertainty estimation and also like confidence calibration. In uncertainty estimation, we often use AUROC, while in confidence calibration, ECE is used as a metric. It is rare to evaluate both metrics at the same time. Could you explain to me the differences and connections between your task and these two tasks?

### Soundness
3

### Presentation
3

### Contribution
2
