# GradShield: Alignment Preserving Finetuning

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Large Language Models (LLMs) pose a significant risk of safety misalignment after finetuning, as models can be compromised by both explicitly and implicitly harmful data. Even some seemingly benign data can inadvertently steer a model towards unsafe behaviors. To address this, we introduce GradShield, a principled filtering method that safeguards LLMs during finetuning by identifying and removing harmful data points before they corrupt the model's alignment. It removes potentially harmful data by computing a Finetuning Implicit Harmfulness Score (FIHS) for each data point and employs an adaptive thresholding algorithm.We apply GradShield to multiple utility fine-tuning tasks combined with different levels of harmful data, and evaluate the safety and utility performance of the resulting LLMs under various metrics. Our results show that GradShield outperforms all baseline methods, as it consistently maintains a low Attack Success Rate (ASR) of under $6\%$, while preserving the utility performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a data filtration method to address the fine-tuning risk. Compared to typical guardrail filteration, the filtration performance is significantly increased.

### Strengths
1. The definition of  FIHS score looks interesting to me. The definition is constructed on a general safety function and measure the impact of one specific data point on the dataset. The definition seems to have some connection with influence function or shapley values. 

2. The final derivation of practical looks intincively correct to me in the first eye, though I have some comments on it (see the weakness part)

3. Experiment is sort of comprehensive though it can be improved. 

4. The detection threshold is automatically selected by a proposed method.

### Weaknesses
1. The FIHS score definition is not as fundamental as the SEAL score [1] in my view. Particularly, FIHS defines a safety score function S. Some casually designed safety function (e.g., a guardrail, and also the S that in you use in your experiment) may not accurately represent the oveall safety ability of the model (also for the S that you use in experiment) and can be circumvent by the adaptive attacker. The safety alignment loss use in SEAL should be a more reliable safety score function. I understand that it is also possible to directly apply the safety alignment loss here as the safety score. In that case, could you discuss more on how the FIHS becomes and how it connects with the data weight optimized by Seal?  Also, I would also mention that recently there are three  more very relevant papers Antibody[3], BDS [4], and Ref-Teacher[5]  that also explore weighting the fine-tuning data points. Please also consider to include them into your discussion.

[1] SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection (ICLR2025)

[3] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence
https://openreview.net/forum?id=qur2ef8MqQ 

[4] Adaptive Defense against Harmful Fine-Tuning via Bayesian Data Scheduler  (NeurIPS2025)

[5] Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks  https://openreview.net/forum?id=OK2GR1guwv

2. **Adaptive attack is not considered.** For detection based method, I typically have serious concern over its ability to circumvent adaptive attack, especially given my prior view that the FIHS score is not very fundamental. Particularly, I have two concerns here. 

* **Your safety score function seems to be overly casually designed.**  In Eq. (5), the safety score function is defined as the logits difference between unsafe token "Sure" and safe token "I".  This may not accurately reflects the safety capability of the model.  For example,  the of answer of a harmful query may not start with "Sure, .." but actually start with "I definitely can help you!", and in this case. why fine-tuning on this harmful sample with increase the logit over "Sure" but decrease that of "I"? 

*  **Your safety score function can be bypassed by a stronger adaptive attackers.** Let's assume that the attacker have access to the proxy dataset, and know the safety score S. Then the attacker can optimize a harmful query/answer using a similar way with Virus[6]. Specifically, the attacker can replace the F1 in Eq. (3) of Virus [6] with the objective to minimize its FIHS score. As you assume, the FIHS score is differentiable,  the attacker should be able to directly minimize the FIHS score by data optimization. I suggest the authors to conduct such adaptive attack experiments and it is okay that the defense fail in this stronger attack settings. 


[6] Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation   


3. Insufficient baselines. The authors should compare with SEAL[1] and Ref-Teacher[5]. Both are data filtration method and were first appeared before the ICLR2026 cycle.

4. Some related work on harmful fine-tuning defense should be discussed.

Detecting Adversarial Fine-tuning with Auditing Agents

Scaling Trends for Data Poisoning in LLMs

Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models

Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation

No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data

Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents

Eliciting Harmful Capabilities by Fine-Tuning on Safeguarded Outputs

Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

Self-Destructive Language Model

CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning

Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning

LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning

SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection

Safety alignment should be made more than just a few tokens deep

Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs

Shape it Up! Restoring LLM Safety during Finetuning

Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization

Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation

AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin

Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment

A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space

Detecting Instruction Fine-tuning Attack on Language Models with Influence Function

Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation when Fine-tuning LLMs

Locking Down the Finetuned LLMs Safety

Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation

Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets

Navigating the safety landscape: Measuring risks in finetuning large language models

ESTIMATING WORST-CASE FRONTIER RISKS OF OPEN-WEIGHT LLMS

Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models

When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

** There may be more relevant works (I just list above some more recent work), and I suggest the authors to read and discuss all of the relevant works on harmful fine-tuning when revising the paper.**

### Questions
1. I am also curious how the cosine similarity between the two gradient in (4) looks like on the initial aligned model and how it  evolve with the fine-tuning rounds (though you only use the initial aligned model θ_0 only to compute the FIHS score). 




I will consider to change my score if the authors can sufficiently address my concern. I overall like this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GradShield, a filtering framework designed to protect large language models (LLMs) from safety misalignment during fine-tuning. It addresses the problem that both explicitly harmful and seemingly benign data can inadvertently compromise model safety. GradShield operates by computing a Finetuning Implicit Harmfulness Score (FIHS) for each data point and using an adaptive thresholding algorithm to identify and remove potentially harmful samples before fine-tuning.

### Strengths
1. Safeguarding LLMs during API-based fine-tuning is an important and timely research area. It is increasingly common for developers to fine-tune LLMs on domain-specific datasets via APIs to improve their utility on specialized tasks, making this problem practically relevant.

2. The proposed FIHS framework is theoretically grounded. Although I did not verify the correctness of the proof in detail, the theoretical formulation appears sound and well-motivated.

3. The paper is clearly written and well-organized, making it easy to follow the overall methodology and contributions.

### Weaknesses
1. Time inefficiency and limited practicality.
 
The major limitation, in my view, is the method’s computational cost. The approach requires computing gradients for **each node**, performing repeated safety and utility evaluations, fitting the resulting scores to two Gaussian models, and possibly fine-tuning the model multiple times on different subsets. This iterative process is impractical for API-based settings, where users typically expect fast responses and cannot afford repeated fine-tuning cycles.

2. Unfair comparison with baselines.

The algorithm’s iterative filtering and fine-tuning steps, based on fixed safety and utility thresholds, give it a natural advantage over baselines that lack such adaptive refinement.As a result, the comparison may not be entirely fair, since the proposed method benefits from repeated optimization until desired thresholds are achieved.

3. Missing relevant baselines.

It would strengthen the evaluation to include comparisons with LLM-based filtering or guard models, such as Llama Guard or frameworks that use LLMs as safety judges. These are widely used in practice for safeguarding fine-tuned models.

4. Insufficient description of adaptive threshold computation.

The adaptive thresholding mechanism is a central component of the proposed approach, yet its presentation is limited. The paper currently provides a large algorithmic block without sufficient explanation of the design intuition or computational process behind it. A more detailed description (e.g., how thresholds are initialized, updated, and stabilized) would help readers better understand the method.

5.Limited justification for token selection (“I” as aligned token).

The rationale for selecting “I” as the aligned or compromised token appears entirely empirical, and its motivation is unclear. Additional analysis or intuition explaining why this token meaningfully represents alignment would make the choice more convincing.

6. Minor Comments

Please use consistent table formatting across the paper to improve readability.

The introduction could be strengthened by citing related works that highlight the trade-off between benign fine-tuning and safety degradation, such as:

[1] Benign Samples Matter! Fine-tuning on Outlier Benign Samples Severely Breaks Safety.

[2] What is in Your Safe Data? Identifying Benign Data that Breaks Safety.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explores the safety misalignment that arises during finetuning and proposes a filtering method that protects LLMs by identifying and removing harmful data points before they corrupt alignment. It computes the FIHS score and employs an adaptive thresholding algorithm. The paper shows that the proposed algorithm achieves a low attack success rate of under 6% while preserving utility performance.

### Strengths
- The paper tackles an important problem that has a significant effect on society.

- The paper proposes a novel approach for selecting a threshold that automatically adapts to the user dataset by employing Gaussian models and applying the Likelihood ratio test to determine a harmful change in the data.

### Weaknesses
- The presentation of the paper is confusing due to a lack of reasoning behind the techniques used and a vague description of the methods used. For example, why the two-component GMM is used should be highlighted more, and why binary search is used should clearly be explained. 

- The proposed algorithm is dependent on the safety score of the held-out dataset.

- Requires only one pass for each datapoint, which can be high for large-scale datasets and considering the LLMs require 3-4 epochs to be finetuned.

- The use of only one probing safety data point is inadequate and limited since it depends on which data point is selected. 

- Choosing "I" as “safe” and "Sure" as “unsafe” isn’t grounded in theory. To me, it seems ad. hoc choice. This selection is not generalizable since a prompt can start with "I" but can still be unsafe, for example, “I can help you do that.”

### Questions
- In line 181: why is a data point considered harmful if it raises the safety score?

- $x_s$ belongs to $D_s$, which is a safety benchmark dataset with harmful prompts. What is the reason behind selecting $x_s$ as the safety data point even though it is harmful?

### Soundness
3

### Presentation
1

### Contribution
3
