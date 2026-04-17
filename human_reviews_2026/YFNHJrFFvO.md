# SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods—such as parallelization, objective- and data-driven modifications, and replay buffers—either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose **SPEC-RL**, a novel framework that integrates **SPEC**ulative decoding with the **RL** rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including GSM8K, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2–3$\times$ without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Spec-RL, a framework that leverages speculative sampling to accelerate the rollout phase in LLM-based reinforcement learning. Spec-RL stores trajectories from all rollouts. When the same prompt reappears, the response from the previous epoch is retrieved and tested to determine the reusable prefix. A lenience parameter is introduced to balance efficiency and accuracy. Experiments demonstrate 1.94× to 2.88× speedups in the rollout phase.

### Strengths
- The proposed method is both novel and practical. Since rollout is the major efficiency bottleneck in LLM-based reinforcement learning, the idea of accelerating it by reusing previous rollout prefixes is promising. The mechanism is well motivated through speculative sampling, and the method design is clear and easy to follow.

- Experiments are conducted on three different models, demonstrating the generality of the approach. The method achieves around 2× rollout speedup, which is strong. The paper also provides a detailed discussion on how the lenience parameter is tuned.

### Weaknesses
1. One unclear aspect is how the loss and importance sampling are handled. Suppose the reused prefix is o1 and the newly generated continuation is o2. Is the policy loss applied to o1 + o2 or only to o2? In addition, since the reused prefix has a log probability under the old model and two new log probabilities under the current policy, how are these values combined to form the importance sampling ratio?

2. The paper only considers reusing responses from the previous epoch. Would it be beneficial to also reuse responses from older epochs to further improve efficiency? Even if it brings limited gain, a discussion on this topic would be valuable.

### Questions
Please refer to the weakness part, and a following question:

- In my understanding, number of epochs in RL-for-LLM post training is typically small, and in some other cases, the data is generated during interaction with environment, which is hardly repeating. Thus I have a question that whether the proposed method is only workable in some settings while hardly workable in other. Could the authors have some discussion about their understanding of this concern?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SPEC-RL, a method that incorporates speculative decoding into the rollout stage of RL training to improve efficiency. The approach reuses previously generated segments as prefixes, thereby reducing redundant computation and achieving substantial speedup according to the reported experiments.

### Strengths
See weaknesses.

### Weaknesses
The submission does not comply with the ICLR formatting requirements. The page layout deviates from the official ICLR template. Specifically, the side margins are significantly smaller than in the standard format. As a result, the actual body text occupies more space per page, effectively exceeding the permitted page limit. This makes the submission non-compliant with the conference’s formatting and length guidelines.

Given the extent of the formatting irregularity and the fact that it affects the true content length, this paper should be considered for desk rejection on the grounds of violating submission format and page-limit policies.

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
2

### Rating Number
2

### Confidence
4

### Summary
The authors proposed a novel speculative decoding method by reusing previous rollouts in RLVR as draft proposals. In this way, the RLVR training is free from individual draft model and utilize the past rollout trajectories. The authors introduce lenience to finegrainedly control the level of acceptance. As a result, nearly doubled speedups are seen on different benchmarks.

### Strengths
- The SPEC-RL is free from deployment of a seperate draft model.
- It exhibited significant token savings for superior performance on benchmark.

### Weaknesses
1. The most concerning issue of this paper is that SPEC-RL is **not algorithmically equivalent** to vanilla training. Standard speculative decoding is already not lossless acceleration under non-greedy sampling. Not to mention the SPEC-RL uses old trajectories rather than draft outputs.
2. The accept length is generally over hundreds. This is very alerting because it is far from most speculative decoding work whose acceptance length is generally not more than 5.
3. The introduced lenience variable is empirically set. This variable is severly impacting training as the authors analyzed. The justification for this variable and its value lacks theoretical proof.
4. The RLVR training highly dependent on sample variability (Especially for grouped rollout in GRPO and Reinforce++). In contrast, reusing in SPEC-RL significantly harms sampling diversity due to shared prompts.
5. Complete training trajectories and number of traning steps are not shown. It is hard to confirm convergence.

### Questions
1. The variable control is not clear. The description at line 666 said "we control for the total amount of rollout data:". But how? Is this refer to same amount of output token (incl. draft acceptance) or same number of prompts?
2. Assuming the number of trajectories put into the model is identical for each entry in Tab. 1, to my understanding, SPEC-RL is receving trajectories with "older" tokens. In this way, how could it generally show superior performance over baseline? (In contrast, it is intuitively acceptable if SPEC-RL can achieve similar performance in shorter time.)
3. How was the importance sampling done? Are draft using the log probs from previous round or newly computed ones?
4. The dataset access order is also confusing. To make sure old trajectories can be used, the following step must use the prompt in previous step. Considering the grouped rollout (N>1), the prompt accessed in SPEC-RL is more heavily exploited than vanilla training. Is this carefully controlled?
5. Considering that canonical speculative decoding strategies like EAGLE or Medusa can already attain a good gain on RLVR training, is there convincing reasons SPEC-RL is superior?

### Soundness
1

### Presentation
2

### Contribution
2
