# From Answer to Think: Multidimensional Supervision of Reasoning Process for LLM Optimization

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Large language models (LLMs) can develop strong reasoning ability when trained appropriately. Existing approaches are broadly categorized into outcome-level answer supervision and process-level reasoning supervision. 
However, the former provides only sparse binary feedback and overlooks intermediate step quality, while the latter scores individual steps but requires task-specific segmentation.
To this end, we propose a novel framework that assesses the quality of reasoning process along three dimensions: **Confidence** for uncertainty calibration, **Relevance** for semantic alignment and **Coherence** for logical consistency. 
Together, these dimensions capture aspects beyond final answer correctness and enable interpretable assessment without requiring ground truth answers.
Our framework serves as a Dimension-level Reward Model (**DRM**) that assigns scores to reasoning processes and provides supervision signals for both off-policy (e.g., DPO) and on-policy (e.g., GRPO) optimization.
Experimental results show that DRM provides effective supervision signals, guides the optimization of LLMs and enhances their reasoning ability.
In particular, DRM-supervised training achieves consistent gains on both in-distribution and out-of-distribution open-domain tasks, including mathematics, question answering, code execution and puzzles.
Our findings demonstrate that multidimensional supervision of reasoning process can improve the generalized reasoning ability of LLMs beyond the training distribution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DRM , a new framework that supervises large language models’ reasoning processes across three interpretable dimensions—confidence, relevance, and coherence—to more effectively improve reasoning ability and generalization than traditional outcome-only training methods.

### Strengths
- The idea is simple and reasonable. Experiments show the effectiveness of DRM.

- DRM can simple replace what orm did in different algorithm.

### Weaknesses
- In table 3, the improvement is limited, not sure whether the improvement only exist in given hyperparameter.

### Questions
- In the paper, the weights for the three dimensions are tuned empirically or via grid search. Were these weights optimized separately for each task, or were they fixed across all tasks? Also, did you compare whether different tasks (e.g., math vs. code generation) require different optimal weight configurations?

- The multi-dimensional supervision seems to rely on several external evaluators. Could you clarify how much additional computational overhead or latency this introduces during training and inference?

- Did you conduct any analysis on how DRM affects the reasoning process qualitatively — for instance, does it make the model’s explanations longer, more structured, or more confident?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Dimension-level Reward Model (DRM) that scores a model’s reasoning process along three complementary dimensions—Confidence, Relevance, and Coherence—and uses this multidimensional signal to supervise both off-policy (DPO + SFT) and on-policy (GRPO-style) optimization. Unlike answer-only reward schemes (RLVR) that deliver sparse, outcome-level feedback and often reward “correct answer, flawed reasoning,” and unlike process-level reward models (PRMs) that require task-specific step segmentation, DRM delivers dense, interpretable, ground-truth-free rewards over the entire chain of thought.

### Strengths
1. DRM directly targets two known gaps—sparse/answer-only rewards and PRM step-segmentation requirements—by shifting to dimension-level scoring that is dense, ground-truth-free, and interpretable.

2. The DRM reward can be integrated with standard training. It supervises off-policy DPO+SFT (pair selection) and augments on-policy GRPO (added advantage).

3. Across diverse tasks and backbones, DRM-supervised models outperform native and strong baselines.

### Weaknesses
1. Relevance depends on a reranker and Coherence on an ORM; the paper fixes dimension weights via grid search. While practical, robustness to judge/model choice and weight calibration is under-analyzed. 

2. Using log-prob as self-confidence is intuitive, but there’s limited study of calibration across domains/backbones or under distribution shift, and little comparison to alternative confidence estimators.

3. In GRPO combinations, a few reasoning-heavy or knowledge-intensive datasets see slight regressions vs. DRM alone (e.g., MuSR/GPQA), suggesting interaction effects between answer-only and reasoning rewards that merit deeper analysis.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors propose a novel framework that assesses the quality of the reasoning process along three dimensions: (1) Confidence for uncertainty calibration, (2) Relevance for semantic alignment, and (3) Coherence for logical consistency. Through extensive experiments, the authors show that Dimension-level Reward Model (DRM) can successfully provide supervision signals for both off-policy and on-policy optimization.

### Strengths
1. This work innovatively proposes Dimension-level Reward Model for both off-policy and on-policy optimization, and demonstrates the effectiveness of incorporating metrics of reasoning process (e.g., Confidence, Relevance, and Coherence) over vanilla outcome reward.
2. This work has done extensive experiments on the advantage of DRM, revealing new findings on process reward.
3. This work introduces a baseline to merge both process and outcome rewards for on-policy optimization, i.e., simply adding the advantage of both rewards. This opens up a new line of research, and is a significant contribution.

### Weaknesses
This work does not investigate deeply how to design a good process metric. Although the dimension-level ones (i.e., Confidence, Relevance, and Coherence) are proposed, more design choices should be compared in calculating the process reward metrics. Also, the final results heavily depend on the accuracy of the process metrics. For example, in cases where the Confidence score mistakenly assigns a flawed reasoning process with a high score, the RL training would be negatively affected. From this perspective, the authors should discuss more on how to calculate the scores.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
