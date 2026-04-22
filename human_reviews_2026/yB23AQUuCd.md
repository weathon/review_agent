# Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Despite the increasing use of large language models for creative tasks, their outputs often lack diversity. Common solutions, such as sampling at higher temperatures, can compromise the quality of the results. Dealing with this trade-off is still an open challenge in designing AI systems for creativity.
Drawing on information theory, we propose a context-based score to quantitatively evaluate value and originality. This score incentivizes accuracy and adherence to the request while fostering divergence from the learned distribution. We show that our score can be used as a reward in a reinforcement learning framework to fine-tune large language models for maximum performance. We validate our strategy through experiments considering a variety of creative tasks, such as poetry generation and math problem solving, demonstrating that it enhances the value and originality of the generated solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Disclosure: Claude is used to refine the review.

This paper proposes CoVO, a context-based score derived from mutual information to assess value and originality in neural text generation. The score combines two components: value (log p(x|y), measuring how well the input can be inferred from output) and originality (-log p(y|x), measuring surprisal). CoVO is used as a reward signal in GRPO to fine-tune LLMs for creative tasks. Experiments on poetry generation, math problem solving, and NoveltyBench demonstrate that the method can improve diversity without sacrificing quality.

### Strengths
- The tension between quality and diversity in LLM generation is important, and balancing value with originality is a meaningful objective.
- Experiments span multiple domains. Testing on poetry, mathematics, and NoveltyBench shows effort to validate across different creative tasks.
-  The proposed method is simple. Meanwhile, the paper provides detailed implementation guidance, e.g., how to compute the score with autoregressive models and integrate it with GRPO. 
- The release of GutenVerse dataset for poetry evaluation is a useful resource.

### Weaknesses
- My main concern is that this paper compares with no existing work on improving creativity of language models, such as DivPO, DRA-GRPO, and DARLING.
- The leap from mutual information to "creativity" lacks rigorous justification. I don't get the claim that log p(x|y) measures "value" - if y is relevant to x but y itself is ungrammatical/low-quality, won't log p(x|y) still be high?
- Computing p(x|y) for autoregressive models requires a workaround (adding prompt q to make y' = y + q) and is not guaranteed to approximate the true posterior.

### Questions
- Why is log p(x|y) evaluating "value"? If y is ungrammatical/low-quality but still relevant to x, would log p(x|y) still be high?

### Soundness
2

### Presentation
3

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
The paper proposes CoVO (Context-based score for Value and Originality), an information-theoretic score designed to evaluate and encourage both value and originality in neural text generation. The authors derive CoVO from mutual information between model inputs and outputs and demonstrate how it can be used as a reward in reinforcement learning (via GRPO) to fine-tune large language models for creative tasks. Experiments are presented across three domains: poetry generation, mathematical problem solving, and the NoveltyBench benchmark. The authors claim that optimizing for CoVO enhances creativity-oriented metrics without sacrificing correctness.

### Strengths
- Clear motivation: addressing the diversity–quality trade-off in creative LLM generation.
- Application of CoVO within reinforcement learning is technically sound and compatible with modern LLM fine-tuning frameworks (e.g., GRPO).
- The experiments span both creative (poetry) and analytic (math reasoning) domains, suggesting some generality.

### Weaknesses
- Lack of comparative evaluation. The paper does not benchmark CoVO against existing novelty/diversity metrics (e.g., Diversity Is All You Need, intrinsic rewards, novelty search).
- All experiments rely on similar foundation model families (llama). Experiments on a different family of foundation models would be useful (e.g., Qwen).
- Qualitative analysis or examples demonstrating that outputs are actually more creative or original would be useful to see.
- Reported improvements are small and not statistically significant.

### Questions
- The CoVO objective may be gameable (Goodhart’s law). Did the authors see any reward hacking happening? How might we prevent such reward hacking of quantitative measures from happening?
- Are there possible failure modes or adversarial cases where optimizing CoVO harms creativity instead?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CoVO, a context-based score rooted in mutual information theory, to quantify value as log p(x|y) and originality as -log p(y|x). This may promote divergence from the model's distribution in LLM-generated text. It employs CoVO as a reward in GRPO fine-tuning to enhance LLM creativity. Tested on poetry generation, math problem solving, and NoveltyBench tasks, the results demonstrate improvements: poetry shows higher tone adherence and lower reproduction rates; math yields better accuracy and diversity; NoveltyBench boosts novelty and quality.

Strength:
1. The paper is well written and easy to understand.
2. Comprehensive empirical validation across varied tasks.

Weaknesses:
1. The concepts of value and originality are not theoretically defined.
2. Using a reward model as a proxy is known to induce the reward hacking problem.

Given the vagueness in the task definition, I am leaning towards rejection.

### Strengths
1. The paper is well written and easy to understand.
2. Comprehensive empirical validation across varied tasks.

### Weaknesses
1. The concepts of value and originality are not theoretically defined. The use of P(x|y) and P(y|x) is unjustified. In addition, this also leads to the second issue.
2. Using a reward model as a proxy is known to induce the reward hacking problem. The reward score can be optimized to a higher level without necessarily improving the proxied quality. Also, the metrics in evaluation (EAD, T-LCS, SBERT, etc.) might suffer from a similar problem as reward hacking (following Goodhart's law).

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
