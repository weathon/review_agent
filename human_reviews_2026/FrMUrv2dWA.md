# Omni-Thinker: Scaling Multi-Task RL in LLMs with Hybrid Reward and Task Scheduling

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The pursuit of general-purpose artificial intelligence depends on large language models (LLMs) that can handle both structured reasoning and open-ended generation. We present OMNI-THINKER, a unified reinforcement learning (RL) framework that scales LLMs across diverse tasks by combining hybrid rewards with backward-transfer–guided scheduling. Hybrid rewards integrate rule-based verifiable signals with preference-based evaluations from an LLM-as-a-Judge, enabling learning in both deterministic and subjective domains. Our scheduler orders tasks according to accuracy backward transfer (BWT), reducing forgetting and improving multi-task performance. Experiments across four domains show gains of $6.2\%$ over joint training and $12.4\%$ over model merging. Moreover, we demonstrate that simple assumptions on accuracy transfer yield accurate predictions of curriculum outcomes, with entropy dynamics explaining deviations due to generative tasks. These findings underscore the importance of BWT-aware scheduling and hybrid supervision for scaling RL-based post-training toward general-purpose LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Omni-Thinker, a unified reinforcement learning framework for training large language models across different task domains. The approach combines hybrid reward signals, integrating rule-based verifiable rewards with LLM-as-a-Judge preference evaluations, and employs a backward-transfer guided curriculum scheduler to order tasks and reduce catastrophic forgetting. The authors claim improvements over joint training and model merging, and propose that simple assumptions on accuracy transfer can predict curriculum outcomes, with entropy dynamics explaining deviations in generative tasks.

### Strengths
- The use of backward transfer matrices to guide curriculum ordering is principled and builds on established continual learning concepts.
- The paper is well written and easy to follow.

### Weaknesses
Please see my detailed questions and concerns below.

### Questions
- Assumption 2 requires that seeing the full dataset once saturates task accuracy to the same level as training from initialization. How realistic is this for complex reasoning tasks? What happens when tasks require multiple epochs to converge?
- Theorem 1 is stated without proof. A statement without proof should not be called a "Theorem" in mathematics.
- For creative writing, you compare against a reference response from the dataset. Doesn't this bias the reward toward a specific style rather than encouraging diversity or creativity?
- The paper claims that models "learn to emulate lower or higher temperatures" (lines 384-387). Are there any direct evidence for this mechanistic claim beyond just correlational observations?
- Table 2 shows temperature ablations for only QA and Writing. Why not include Math and Coding?
- The creative writing evaluation uses MT-Bench against GPT-4 from 2023. This is now quite outdated. How would results change against more recent models?
- The proposed method requires computing the full BWT matrix upfront, which means training on all task pairs. For K tasks, it seems to require O(K^2) training runs. How does this scale computationally as K increases?
- The paper focuses on post-training of already instruction-tuned models. Would your approach work for continued pretraining or for training from scratch?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Omni-Thinker, a unified RL framework that scales LLMs across diverse tasks using hybrid rewards and backward-transfer-guided scheduling. The method integrates verifiable rewards for structured tasks with preference-based LLM-as-a-Judge evaluations for open-ended tasks. Experiments across four domains show average gains of 6.2% over joint training and 12.4% over model merging. The proposed method offers an effective solution for unified multi-task learning in both structured and open-ended domains.

### Strengths
1. This framework addresses the inconsistency in optimization direction across different tasks in the reinforcement learning process, integrating verifiable rule-based rewards and preference-based LLM evaluation into a unified reinforcement learning paradigm.

2. The proposed BMT, by quantifying how learning a task influences the performance of previously learned tasks, provides a referable paradigm for the learning order in curriculum learning, mitigating the catastrophic forgetting problem to some extent.

3. In experiments across four different domains, the proposed method demonstrates stable performance improvements, outperforming existing approaches to model merging and joint training.

### Weaknesses
1. As mentioned in the article, the overhead of curriculum scheduling increases gradually with the increase in workload, and the scalability of the proposed method may be limited. Are there efficient strategies for real-world deployment?
2. The paper presents results using Qwen2.5-7B as the base model for all experiments. Would the same backward-transfer-guided scheduling strategy remain optimal for significantly smaller or larger models?
3. The overall framework, particularly the curriculum design and entropy analysis, appears somewhat heuristic and lacks tight integration with the core methodological contributions. To strengthen the contribution, could the insights from the entropy analysis be more formally integrated into the scheduling algorithm itself?

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper studies training curriculum for multi-task RL with both verifiable and preference-based tasks. It considers a multi-stage training setup, where the model is trained only on a single task at each stage and only enters the next stage after it finishes all training samples of this task. The proposed approach first measures the cross-task influence — the impact of training on each individual task on the performance of other tasks — and then determines an optimal task ordering by prioritizing the task that yields the highest average test accuracy across the remaining tasks. Experiments shows that this curriculum-based strategy outperforms no curriculum training, model merging, and SFT.

### Strengths
* This paper proposes a simple framework for ordering tasks. The final ordering heuristics make intuitive sense.

### Weaknesses
* This work relies on overly simplisitc assumptions (for both assumptions) and there are no sufficient evidence to justify them. Also see questions section.
* Authors claim that the predicted accuracy using test set backward transfers are surprisingly precise, however, table 3 shows relatively low correlations between test and predicted accuracies.

### Questions
If inter-task transfer effects are assumed to be constant (i.e., independent of starting ckpt), and we have task-wise saturation, then the cumulative effect of sequential training should be order-invariant. Could the authors explain this?

### Soundness
2

### Presentation
3

### Contribution
2
