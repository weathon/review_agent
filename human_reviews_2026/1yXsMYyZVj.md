# Forgetting: A New Mechanism Towards Better Large Language Model Fine-tuning

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Supervised fine-tuning (SFT) plays a critical role for pretrained large language models (LLMs), notably enhancing their capacity to acquire domain-specific knowledge while preserving or potentially augmenting their general-purpose capabilities. However, the efficacy of SFT hinges on data quality as well as data volume, otherwise it may result in limited performance gains or even degradation relative to the associated baselines. To mitigate such reliance, we suggest categorizing tokens within each corpus into two parts---**positive** and **negative** tokens---based on whether they are useful to improve model performance. Positive tokens can be trained in common ways, whereas negative tokens, which may lack essential semantics or be misleading, should be explicitly forgotten. Overall, the token categorization facilitate the model to learn less informative message, and the forgetting process shapes a knowledge boundary to guide the model on what information to learn more precisely. We conduct experiments on well-established benchmarks, finding that this forgetting mechanism not only improves overall model performance and also facilitate more diverse model responses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a simple yet effective token-level forgetting mechanism for supervised fine-tuning (SFT) of large language models. The method identifies “positive” and “negative” tokens based on loss differences between a base model and a lightly fine-tuned reference model, and applies a modified loss function: maximizing likelihood for positive tokens while decreasing likelihood for negative ones, with an adaptive coefficient λ(step). Experiments on five benchmarks using several LLaMA variants (1B, 3B, 8B) demonstrate consistent improvements over standard SFT and sequence-level filtering approaches.

### Strengths
Simplicity and clarity: The method is conceptually simple, easy to implement, and requires no additional labeling or complex architecture changes.

Innovative data generation: Automatically deriving token quality scores via model-to-model loss differences is a clever and generalizable idea, potentially applicable to diverse SFT or preference-optimization tasks.

Empirical improvements: Consistent performance gains across multiple LLaMA sizes and benchmarks, showing robustness to model scale.

Practical benefit: The approach can serve as a lightweight alternative to preference optimization, offering better noise control during fine-tuning without needing pairwise preference data.

### Weaknesses
[Limited novelty relative to prior work]
While the paper presents a clean formulation, its conceptual link to existing negative preference optimization (NPO) and token-level preference learning methods is not deeply analyzed.
For instance, TNPO (Token-level Negative Preference Optimization, Xu et al., 2024) also applies token-level weighting to downscale harmful or low-quality regions in text, and Unlearning approaches (e.g., Thakkar et al., 2024) similarly reduce log-likelihood for specific token spans.
The paper should clearly articulate how its influence-based token labeling differs fundamentally—is the improvement from the loss design itself, or from the way token labels are generated?

[Model diversity limitation]
All experiments use only LLaMA-based models. This narrow scope limits confidence in the method’s generality; it remains unclear whether the forgetting mechanism would generalize to different architectures (e.g., Qwen, Mistral, GPT-OSS) or even decoder–encoder models (T5, UL2).

[Lack of qualitative or diagnostic analysis]
The paper does not provide examples or visualization of which tokens are labeled negative, nor does it discuss possible biases introduced by the loss-based scoring.

### Questions
Clarify the conceptual delta from NPO/TNPO/unlearning-based fine-tuning and include controlled comparisons using matched compute and data.

Extend experiments to non-LLaMA architectures to test generality.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel token-level "forgetting" mechanism for SFT, where tokens are classified as "positive" or "negative" based on their influence scores derived from a reference model. While positive tokens are learned via standard gradient descent, negative tokens are actively "unlearned" through gradient ascent on their loss.

### Strengths
This paper is easy to follow and read.

### Weaknesses
1. The method applies gradient ascent on negative tokens, which may distort contextual representations and disrupt the internal consistency of sequential predictions. This could degrade fluency, coherence, or generalization, even if downstream task metrics improve. It needs more experiments to analyze the impact of the loss designed in the paper not only on the final metrics but also the training dynamics.

2. The influence-based token scoring relies on a reference model trained on a potentially noisy subset of data. Errors or biases in this reference model may misclassify high-quality tokens as negative, leading to harmful unlearning of useful information.

3. The evaluation is limited to only three LLaMA-3 variants and a narrow set of baselines (e.g., full SFT and a simple token masking baseline). It omits comparisons with established data filtering or other related baselines, weakening the claimed advantage. While, there are also various benchmarks to evaluate the general-purpose datasets (such as instruction-following, AlpacaEval, Arena-Hard) instead of these QA benchmarks.

4. The proportion of tokens classified as positive vs. negative (which depends on the thresholding parameter ρ and the influence score distribution) is not disclosed. Without this, it is difficult to assess the scale of the "forgetting" effect, reproduce the method, or understand whether performance gains stem from aggressive data reduction rather than the forgetting mechanism itself.

### Questions
I still have concerns about the token-level gradient ascent and would appreciate additional analytical experiments to better understand its specific effects.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a token-level “forgetting” mechanism for supervised fine-tuning (SFT) of LLMs. Each token is classified as positive or negative via a small proxy model; negative tokens are down-weighted with a reversed loss term. Experiments on two commonsense QA benchmarks show +1.3 % accuracy and +5.8 % diversity vs. standard SFT. The idea is intuitive and timely, but the submission suffers from incomplete evaluation, and weak baselines.

### Strengths
1. The research problem is interesting, and mitigating noisy SFT data is relevant to the community.
2.  Experiments on well-established benchmarks, show that this forgetting mechanism is promising.

### Weaknesses
1. The empirical experiment is not enough, lacking baseline methods mentioned in related works. Only one table shows the main result in the whole paper, which makes the experiment insufficient. Can you use some exploratory figures to show the effectiveness of your method?

2. The presentation is poor and the method part is hard to follow. For instance, Eq. (5) and Eq. (8) are hard to understand. Why do you set lambda(step) in that way?

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
