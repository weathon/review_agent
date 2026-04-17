# From Broad Exploration to Stable Synthesis: Entropy-Guided Optimization for Autoregressive Image Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Combining Chain-of-Thought (CoT) with Reinforcement Learning (RL) improves text-to-image (T2I) generation, yet the underlying interaction between CoT's exploration and RL's optimization remains unclear. We present a systematic entropy-based analysis that yields three key insights: (1) CoT expands the generative exploration space, while RL contracts it toward high-reward regions; (2) final reward is strongly negatively correlated with both the mean and variance of image-token entropy, highlighting the need to reduce uncertainty and instability; and (3) the entropy of the textual CoT directly governs downstream image quality, with lower-entropy CoTs leading to better generations. Motivated by these findings, we propose Entropy-Guided Group Relative Policy Optimization (EG-GRPO), a fine-tuning strategy that reallocates optimization budget by uncertainty: low-entropy tokens are excluded from reward-driven updates to preserve stability, while high-entropy tokens receive an entropy bonus that encourages structured exploration without collapse. Experiments on standard T2I benchmarks demonstrate that EG-GRPO achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an approach to balance exploration via CoT vs exploitation using RL for improving the text to image models and ultimately proposes a modified version of GRPO to combine these two insights.

### Strengths
The logic behind the paper makes a lot of sense, the combination of exploration vs exploitation.

The objective of the paper is clear and well communicated.

### Weaknesses
Missing references GRPO in autoregressive models: 

'Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization' 
'Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl.'


The relative improvement with respect to T2I-R1 is small. 

Experiments could be more comprehensive:
- Can you apply the same method to improve aesthetic score like in 'Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization' ?
- Can you try it with a more diverse set of text-to-image models? (I expected to work too, but its good for completeness even if its just on one benchmark to see how different base models improve with your technique)

### Questions
- Is Figure 6 representative across multiple prompts or is it a curated set of images? please add a disclaimer if so.
- In Figure 2 why does T2I-R1 translate the whole distribution rather than reshaping it around the original Janus distribution, the shift seems extreme. The entropy image mean goes down, could we include an appendix showing real images so we can qualitatively compare the diversity?

### Soundness
4

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
This paper studies the problem of autoregressive text to image generation, using chain of thought before generation. The authors devise three conclusions about the relationship between entropy and reward, before going on to devise an algorithm EG-GRPO. This algorithm nulls out what is called low entropy tokens to instead focus on high entropy tokens. The authors offer a theoretical explanation for "reinvesting" in high entropy tokens. The paper experimentally verifies this on T2I-CompBench and WISE. Where on most tasks it shows superior performance.

### Strengths
- this paper is well justified and for the most part I agree with the three points flagged to instantiate this algorithm.
- this paper does a good job justifying the motivation for the algorithm

### Weaknesses
- I am curious about the effect of EG-GRPO on diversity of the reward/image. Indeed, I worry that the by further optimizing high entropy tokens, the diversity of the rewards and images generated is lowered. 
As seen in fig 5, it seems that average entropy is lower, and thus the model is generating less diverge images. Some analysis and discussion would be preferred.
- The algorithms performance seems better than the other methods, but I worry this is at the cost of sample diversity.
- It is unclear to me why even entropy minimization in this case is the correct method? To me, this seems like reducing exploration?

### Questions
see weaknesses for questions.

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
2

### Summary
The paper investigates the relationship between CoT prompting and RL in autoregressive text-to-image generation models. The analysis shows that CoT broadens exploration, while RL narrows it toward high-reward regions. Both the mean and variance of image-token entropy are negatively correlated with final rewards, emphasizing stability and reduced uncertainty. The authors propose Entropy-Guided Group Relative Policy Optimization (EG-GRPO), which allocates optimization based on token-level entropy. Experiments show the proposed method balanced exploration and stability in RL fine-tuning.

### Strengths
The authors provide a thorough analysis of the proposed method to show how CoT and RL are balanced to reduce uncertainty and preserve knowledge, which seems convincing.

Paper is well organized. writing is clear and easy to follow.

### Weaknesses
1. The proposed approach adds nontrivial computational overhead due to token-level entropy computation, percentile thresholding, and per-batch bonus recalibration. However, the paper lacks a quantitative analysis of the resulting scaling, memory, and wall-clock costs, particularly for large-scale models or datasets.
2. While entropy is treated as a measure of uncertainty, the method primarily focuses on entropy reduction, potentially at the expense of output diversity and creative expressivity. A discussion of this trade-off is missing.

### Questions
1. Have the authors benchmarked the wall-clock or memory overhead of EG-GRPO (with batch bonus calibration) compared to vanilla GRPO or diffusion-based models, especially at scale? 
2. Are there cases where the additional computation outweighs the quality gains?
Could the authors provide a quantitative assessment of diversity (e.g., via FID/IS for diversity, or human preference rating) to determine whether entropy suppression negatively impacts creative diversity?

### Soundness
3

### Presentation
3

### Contribution
2
