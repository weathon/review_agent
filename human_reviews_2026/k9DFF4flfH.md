# Detecting Distillation Data from Reasoning Models

- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Reasoning distillation has emerged as an efficient and powerful paradigm for enhancing the reasoning capabilities of large language models.  However, reasoning distillation may inadvertently cause benchmark contamination, where evaluation data included in distillation datasets can inflate performance metrics of distilled models. In this work, we formally define the task of distillation data detection, which is uniquely challenging due to the partial availability of distillation data. Then, we propose a novel and effective method $\textit{Token Probability Deviation}~(\textit{TBD})$, which leverages the probability patterns of the generated $\textit{output}$ tokens. Our method is motivated by the analysis that distilled models tend to generate near-deterministic tokens for seen questions, while often producing more low-probability tokens for unseen questions. Our key idea behind TBD is to quantify how far the generated tokens' probabilities deviate from a high reference probability. In effect, our method achieves competitive detection performance by producing lower scores for seen questions than for unseen questions. Extensive experiments demonstrate the effectiveness of our method, achieving an AUC of  0.918 and a TPR@1\% FPR of 0.470 on the S1 dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper addresses the problem of detecting distilled reasoning data, where reasoning chains from a LLM are distilled into a small language model (SLM). The authors aim to identify such distilled data given access to a fine-tuned SLM because some of this data may also appear in benchmark datasets, potentially contaminating the evaluation. They propose a method (TBD) and claim that for questions seen during SLM training, the model generates nearly deterministic answers with very high probabilities. By analyzing these probabilities, it is possible to determine whether a question was included in the SLM’s training data.

### Strengths
* Novelty compared to previous methods.
* Simple and intuitive approach.
* Clear mathematical formulas.

### Weaknesses
* Focuses on a single LLM (Qwen), though the method could potentially be applied to other models as well.

### Questions
* Do you have any evidence to support the claim: "Considering that the earlier generated tokens are likely to be more representative of the behaviour of the model for members and non-members"?

* The difference between your method and others in TPR@1%FPR appears unusually large. Do you have any intuition or explanation for this?

* You state: "In addition, among three models of different sizes, our method achieves the best AUC and TPR@1%FPR on the 32B model, likely because larger models are capable of memorizing training data." However, in Table 1, your method achieves better results with Qwen2.5-7B-Instruct than Qwen2.5-14B-Instruct on LIMO and S1.1. Can you clarify this discrepancy?

* You mention: "The figure 5a shows that the AUC initially increases with truncation length, but declines as the length continues to increase. Our method consistently reaches its optimal performance across various datasets when the truncation length is around 300. Similarly, Figure 5b show that the TPR@1%FPR score exhibits a similar trend. These findings indicate the robustness of our method across various datasets, allowing users to deploy our algorithm with a fixed truncation length value." Why does this behaviour occur? Why is the trend not monotonic? Do you have an intuition for this pattern?

* In Figure 5c, why does AUC behave this way with respect to $\tau$? Increasing $\tau$ allows tokens with higher probability to contribute, which should highlight member samples. On the other hand, the score for classifying a member should be low, but increasing $\tau$ seems to worsen the score (as it increases even for small values). Could you please explain this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of distillation data detection and the key challenge is partial availability: an auditor can query a distilled model with a question, but does not have access to the data used during training. The authors observe that distilled models tend to generate tokens with near-deterministic, high probability for seen questions during distillation, while producing more low-probability tokens for unseen questions. Based on this, they propose Token Probability Deviation (TBD), a scoring function that quantifies how far the probabilities of the first M generated tokens deviate from a high reference probability. A lower TBD score indicates a higher likelihood that the question was in the distillation set. Extensive experiments on models distilled from S1, S1.1, and LIMO datasets show that TBD significantly outperforms existing input-token-based membership inference methods, achieving an AUC of 0.918 and a TPR@1%FPR of 0.470 on the S1 dataset.

### Strengths
* The core idea of using the deviation of generated token probabilities instead of input-likelihood is novel
* The experimental evaluation is extensive and TBD consistently outperforms other existing methods across benchmarks
* The good TPR@1%FPR scores also demonstrate its practical utility for real-world auditing
* The paper is clearly written and easy to follow

### Weaknesses
* The experiments are conducted based on full access to the model's token-level probability distributions. It would be helpful if the authors could discuss on situations that only the generated text is available.
* It would be interesting to know whether sampling-based decoding strategies, e.g., top-k, temperature score, might affect the TBD scores.

### Questions
1. How would the performance of TBD if only generated text is available instead of full token probabilities?
2. Have you explored the performance change of TBD using sampling with different temperature scores?
3. Would it affect the performance of TBD if we change the distillation process from fine-tuning into DPO or RLHF?

### Soundness
3

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
The authors propose token probability deviation (TBD), a method for detecting instances which were used to distill reasoning data from long reasoning models (LRMs) into smaller LMs. Te authors propose a metric for detecting data used in training based on low probability tokens generated by the LM post-distillation. Higher likelihood of generating the reasoning sequence indicates that the instance was used in distillation. The authors evaluate the method using three distillation datasets (S1, S1.1 and LIMO) across Qwen models of different sizes, and show that it outperforms input-based methods as well as several baselines on detecting distillation data.

Overall, the paper is very well written and motivated. The authors motivate their approach well using emprical data, as well as describe the methodology clearly. My main issue with the work is that the contribution feels very limited and the method is a straightforward exstension of perplexity - TBD filters out high probability tokens, and scales the (1-) probabilities before averaging them. See questions for other concerns.

### Strengths
- Well written and motivated paper
- A novel method that performs well on detecting distillation data

### Weaknesses
- The contribution is very limited
- Methodological novelty is very low

### Questions
- Did you try paraphrasing the input questions before distillation, and then detecting reasoning? A simple way of bypassing this detection method would seem to generate reasoning for distillation based on paraphrased input, where the model should still perform well on data, but the reasoning chains might differ sufficiently to avoid detection. 
- Why are the perplexity and min-k scores computed on the first 1000, while TBD on the first 300 tokens? This makes the comparison unclear, especially since it is shown in Figure 5 that the performance of TBD decreases at higher truncation lengths.

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
This work introduces the task of distillation data detection, namely in a question-only setting where one is assessing whether a question was trained on during the distillation process. They propose token probability deviation (TBD) which is a score derived from the probabilities of *generated tokens* during greedy decoding. This method is motivated by an observation that questions that were seen during distillation (members) yield more deterministic output behavior than those not seen (non-members). Testing on three distillation datasets (S1, S1.1, LIMO) and Qwen2.5 models, they fine their  with Qwen2.5 models (7B/14B/32B), the method outperforms input-likelihood and simple output-based baselines on membership identification. They also conduct ablations and sensitivity analysis to understand reasonable gating thresholds and token caps to maximize AUC and TPR@1%FPR.

### Strengths
- **Problem formulation makes sense:** Data distillation detection seems like a valid and important task with the growing sigfnificance of distillation methods, particularly for with longer-CoT methods. There are cases where the actual distillation data itself could be unavailable (e.g., proprietary long-CoT outputs for a public task).
- **Intuitive and largely model-agnostic:** It operates on generated-token probabilities and is straightforward to implement, assuming model output probabilities are available.

### Weaknesses
- **Limited novelty:** TBD is largely a fairly intuitive heuristic (with a couple of parameters to e.g., gate token length, deviation penalty, etc.), however it seems fairly derivative given existing related work on conventional data contamination methods. 
- **Could use a wider variety of model and training configurations**: the work is tested only on QWEN-2.5 with conventional supervised fine-tuning and a small range of dataset sizes. To claim the robustness of this metric, it would be ideal to test this on a wider range of setups and given the metric is the primary contribution of the work, this area needs significantly more robust exploration before I would feel comfortable deploying it for this task.
- **Contribution scope**: While the scope is clear, I feel like this work would benefit from comparative analysis that consider multiple formulations of distillation data detection task -- i.e., one with full distillation information (q,c,a) and one with only the defined partial availability (q). I think it's reasonable to assume there may be parallels between these two formulations and I am not sure if it makes to address just the partial availability setting without performing some empirical comparison with the more conventional 'full availability' setting.

### Questions
- Can you provide results for the other popular open-weight model families?
- Similarly, have you explored any more nuanced distillation strategies beyond full-parameter supervised fine-tuning? 

It seems these two above dimensions could have a significant impact on distillation detection performance.
- Do you have any baseline performance numbers if conditioning on the full training sample (or perhaps just question+rationale / rationale)? I think some additional analysis in this direction would paint a clearer picture for the distillation data detection task as a whole.

Writing / Grammar Suggestions: 
- The acronym for the technique (TBD) does not match the name (Token Probability Deviation)? Is this intentional? If so, perhaps highlight the **b** in probability when introducing the term?
- I would appreciate if you could include a table outlining the key statistics of each distillation dataset and the key reasoning characteristics they are attempting to distill.

### Soundness
2

### Presentation
3

### Contribution
2
