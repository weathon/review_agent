# Hierarchical Feedback Interface for Human-in-the-Loop Reinforcement Learning in Debugging

- Decision: Reject
- Scores: 0, 2, 2

## Abstract
We propose Hierarchical Feedback Interface (HFI) for human-in-the-loop
reinforcement learning in debugging which structures human feedback
grouped into high level objectives and low level refinements to cover the
subjectivity and inefficaciousness of ad-hoc corrections. The HFI employs a
two-tiered policy architecture, in which a high-level policy abstracts
debugging goals into ac a interpretable meta-objectives, and a low-level
policy translates these into actionable feedback thus grounding
human input to the ALigned-and-goal reasoning. The framework integrates a
hierarchical actor-critic mechanism - with the high-level policy
generating goal vectors over reduced state representations, while the
low level policy conditions of both code specific features and these
goals to generate context-aware feedback.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper presents a “hierarchical feedback interface” that is supposed to integrate human feedback at two levels for optimizing hierarchical RL policies. It targets a code debugging use case.

While the topic is potentially relevant, the submission is not in a reviewable state. The writing quality and structural issues prevent proper evaluation of the technical contribution. A major rewrite and clarification of the methodology (especially the technical contribution described in chapters 4 and 5) would be required before this work could be meaningfully reviewed.

### Strengths
+ The topic is interesting in general; making hierarchical RL work is a long-standing challenge. The use case seems practical and relevant.
+ There seems to be a novel contribution somewhere in there, but I was unable to assess it.
+ The proposed evaluation seems good in theory

### Weaknesses
- The paper reads like a draft (especially for sections 4 and 5). The convoluted writing, errors, sentence fragments, and usage of unexplained concepts (like Temporal Convolution Network) make it difficult to grasp the approach and contribution fully. Some key elements, like formula 12, are not explained (how is the weighting done?).
- Figures are not referenced in the text (Figures 1,2,3) or have meaningful captions
- The related work section is sparse, missing some crucial papers, e.g., on preference-based RL 
- For the experiments, many crucial details are missing

### Questions
-

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper does hierarchical RLHF for debugging. The feedbacks are abstracted into high level and low level goals. This is a good idea however the paper is not in an acceptable state.

### Strengths
Good idea.

### Weaknesses
The paper is clearly rushed with lots of typos. 
Pages 6-8 are almost empty.
Experiments are run on one seed (right?) so the results are null.

### Questions
Can you run your experiments on mutliple seed please and do a qualitative analysis of a single bug fix wth HFI please?
Can you detail the training details of the PPO ?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a hierarchical feedback interface for human-in-the-loop rl for code debugging. A high-level policy outputs goal vectors, while a low-level policy does concrete fixes. A PRM is used to translate human judgement into differentiable signals. Authors conduct experiments to show that their method has a higher bug-fix rates and fewer interventions than baselines.

### Strengths
* Tackles an important problem of incorporating human feedback for debugging. The solution of using hierarchical RL is relatively interesting combined with PRM.

### Weaknesses
* Section 4 is poorly written and very hard to understand. Key mechanisms like gating mechanism in eq (12) are not specified. There are also random phrases in Section 4, like "goal relevance", which suggests an unfinished state for the paper.

* Figure 1 is confusing as well: the architecture doesn't show a clear "human" component even though that's the core of the proposed method. The human expert only shows up in Figure 2.

* Evaluation details are omitted: PRM specification, episode length, etc. Only a few hyperparameters are given.

* Novelty is limited due to prior work on HRL + preference RL, especially since human is not actively in the loop for this method.

* Writing quality is overall very poor

### Questions
1. What's the PRM used for the experiments? Is it finetuned for the debugging task on new data that you collected?

### Soundness
2

### Presentation
1

### Contribution
2
