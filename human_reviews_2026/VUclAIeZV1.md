# Don't Lose Sight: Visually-Grounded Credit Assignment for Multimodal Reasoning

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6

## Abstract
Reinforcement Learning (RL) has shown promise for large language models, but its direct application to multimodal LLMs (MLLMs) faces unique challenges. Unlike text-only LLMs, MLLMs must jointly optimize for visual grounding and language reasoning. 
Our analysis reveals that RL primarily enhances textual reasoning, while the crucial visual
grounding aspect stalls, creating a bottleneck for overall model performance.
This observation highlights a critical mismatch: the learning challenge in MLLMs is concentrated in visually-grounded tokens, yet existing RL algorithms apply uniform optimization pressure across all tokens, thereby diluting the learning effort.
Motivated by this limitation, we propose Visually-grounded Credit Assignment (VICRA), a simple yet effective approach that reallocates optimization pressure toward visually-grounded tokens, explicitly correcting the token-level imbalance overlooked by prior methods. 
Extensive experiments across benchmarks, base models, and training data show that VICRA consistently enhances multimodal reasoning, achieving significant gains over strong RL baselines. Our work establishes a general framework for more balanced and effective reinforcement learning in MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper makes a point: directly applying RL to multimodal models leads to an uneven learning process. Optimization pressure gets spread out uniformly, which ends up improving textual reasoning but leaves visual grounding behind. The authors show that visually-dependent tokens get trapped in a high entropy state, bottlenecking the model's performance.To tackle this, they propose VICRA, a method for refocusing the optimization. It works by assigning a “Visually-grounded Score“ to each token, calculated from the probability difference with and without the visual input, and then uses that score to concentrate the learning on the visually-grounded parts of the output. This approach is designed to be a simple drop-in for RL algorithms like GRPO and DAPO, and its effectiveness is well-supported by results on multiple benchmarks.

### Strengths
This work astutely points out an optimization imbalance in MLLMs trained with RL. Specifically, these models have a tendency to bolster textual reasoning at the expense of visual grounding. To resolve this, the authors introduce VICRA, a simple yet highly generalizable method that strategically reallocates optimization pressure. For this approach, the authors report performance gains across diverse models, algorithms, and benchmarks.

### Weaknesses
The paper don't discuss the resource costs of VICRA's “Visually-grounded Score“” calculation. At the same time, it lacks a convincing explanation for why significant gains on reasoning benchmarks do not translate to meaningful improvements on perception tasks. Given the small margins, the authors should report evaluation variance and statistical significance to better support their conclusions.

### Questions
1. The model shows clear gains on reasoning-oriented benchmarks but only marginal gains on perception tasks, which is somewhat counterintuitive. Beyond the fact that these benchmarks have high baseline scores, why substantial improvements in visual reasoning do not translate into improved perceptual ability?
2. Building on the first point, the paper would be stronger if it reported evaluation variance and statistical significance.

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
4

### Summary
This paper addresses optimization challenges of reinforcement learning (RL) in multi-modal large language models (MMLLMs).

It identifies the phenomenon that visually-grounded tokens maintain high entropy during the training process, while text-related tokens follow the exploration-exploitation trajectory where entropy rises and then falls as the training converges.

To address the problem, the paper simply enforces a larger weight on the reward for visually-grounded tokens during training.
Specifically, the weights are proportional to the visually-grounded credits that are the score differences between with and without images as inputs.

### Strengths
(1) The observation is interesting: visually-grounded tokens maintain high entropy during the training process, while text-related tokens follow the exploration-exploitation trajectory where entropy rises and then falls as the training converges.

(2) The paper writes clearly and is easy to follow.

### Weaknesses
(1) The paper identifies the phenomenon that visually-grounded tokens keep high entropy during training while text-related tokens follow the exploration-exploitation trajectory, which means that visually-grounded tokens are more difficult to optimize. However, there is no deep analysis of the root causes.               

Does it come from the misalignment between language and vision?

There should be more analysis.

(2) It seems that the motivation is not consistent with the empirical analysis. As shown in Figure 4, KL_prop achieves a much smaller entropy while inferior performance than the proposed VICRA, which implies that lower entropy on visiually-grounded tokens does not mean higher performance. 

Moreover, the paper argues that KL_prop can suffer from entropy collapse. What does the 'entropy collapse' mean?

Why doesn't VICRA suffer from it? 

(3) The proposed VICRA training objective attaches more importance to visually-grounded tokens and improves model performance on reasoning tasks.
    It is interesting to show how the objective affects model performance on benchmarks, like VQA, SQA, GQA, and POPE.

### Questions
See weaknesses.

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
This paper identifies a key challenge in applying reinforcement learning (RL) to multimodal large language models (MLLMs): a performance bottleneck caused by an imbalance in optimization pressure between visually-grounded tokens and other tokens. The authors' analysis suggests that existing RL algorithms enhance textual reasoning but fail to sufficiently improve visual grounding, as evidenced by the consistently high entropy of visually-grounded tokens during training. To address this, the paper proposes Visually-grounded Credit Assignment (VICRA), a method that reallocates optimization pressure by amplifying the advantage function for tokens identified as visually-grounded. The experiments, conducted across several multimodal reasoning benchmarks, demonstrate that VICRA improves the performance of strong RL baselines like GRPO and DAPO on various base models and datasets.

### Strengths
1. The paper articulates a previously underexplored problem in the RL-based fine-tuning of MLLMs. The analysis identifying the optimization imbalance between visually-grounded and text-related tokens, supported by entropy-tracking experiments, provides a motivation for the proposed solution.
2. The proposed VICRA method is straightforward to implement and integrate into existing RL frameworks that use an advantage function (e.g., GRPO). It directly targets the identified problem by re-weighting the advantage based on a "visually-grounded score." The method shows consistent performance improvements across multiple benchmarks, base models (Qwen2.5-VL-7B, Qwen2.5-VL-3B, Llama-3.2-11B-Vision-Instruct), and training datasets (ViRL39k, MMK12).
3. The authors benchmark their method against a range of closed-source, open-source generalist, and open-source reasoning-specialized MLLMs. The consistent gains over strong baselines like GRPO and DAPO (e.g., an average improvement of +2.25 for GRPO w/ VICRA and +2.22 for DAPO w/ VICRA in Table 1) validate the effectiveness of the proposed approach.

### Weaknesses
1. The core motivation and the formulation of the visually-grounded score (Equation 4), which contrasts token probabilities with and without visual input, bear a strong resemblance to contrastive decoding methods[3] used to mitigate hallucinations in MLLMs. The paper would be significantly strengthened by acknowledging this connection, discussing the relationship between VICRA's training-time credit assignment and these decoding-time strategies, and citing relevant work. Furthermore, demonstrating VICRA's effectiveness on established hallucination benchmarks, such as POPE[1] and R-Bench[2], would provide a more direct and comprehensive evaluation of its ability to improve visual faithfulness.
2. The experimental validation is heavily focused on mathematical and logical visual reasoning benchmarks. While VICRA shows improvements in these areas, its effectiveness on more general multimodal tasks appears limited. The results on perception benchmarks in Table 3 show only marginal gains over the GRPO baseline. To establish VICRA as a truly general and robust framework, it is crucial to include experiments across a wider array of common vision-language tasks and demonstrate that the method's benefits are not confined to the niche of complex reasoning.

[1] Evaluating Object Hallucination in Large Vision-Language Models
[2] Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models
[3] Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
