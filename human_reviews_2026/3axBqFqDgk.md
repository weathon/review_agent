# It Takes Two: Your GRPO Is Secretly DPO

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Group Relative Policy Optimization (GRPO) is a prominent reinforcement learning algorithm for post-training Large Language Models (LLMs). 
It is commonly believed that GRPO necessitates a large group size to ensure stable training via precise statistical estimation, which incurs substantial computational overhead.
In this work, we challenge this assumption by reframing GRPO as a form of contrastive learning, 
which reveals a fundamental connection to Direct Preference Optimization (DPO). 
Motivated by DPO's empirical success, we investigate the minimal two-rollout case (2-GRPO)—a configuration previously deemed infeasible. 
We provide a rigorous theoretical analysis to validate 2-GRPO and demonstrate empirically that it achieves performance on par with 16-GRPO, 
despite using only $1/8$ of the rollouts and reducing training time by over $70\\%$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper reinterprets Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO) as a form of contrastive learning. Based on this insight, the authors propose 2-GRPO, an extremely lightweight variant that uses only two rollouts per prompt instead of large groups. , They show the 2-GRPO is able to preserve gradient estimation and the same optimization direction as standard GRPO. Through theoretical analysis, they prove that 2-GRPO implicitly normalizes advantages and maintains stability, and empirically show across multiple reasoning benchmarks that it matches the performance of 16-GRPO while largely cutting training time and computational cost.

### Strengths
The work theoretically and empirically showed that 2-GRPO can be compatible with large GRPO while reduce the rollout.

### Weaknesses
Although 2-GRPO achieves comparable performance to standard GRPO with substantially lower computational cost, it also presents several limitations. From Table 1 and figure 1,2, the overall performance is dropped. Especially, AIME drops up to 9.56% and AMC dropped up to 4.85%.

### Questions
1. Is there any way to reduce the drop in performance? Could we increase the number of epochs for 2-GRPO to match the performance of standard GRPO? If so, approximately how many additional epochs would be required?

2. How can the theory be generalized to continuous reward settings?

### Soundness
3

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
3

### Summary
The paper analyzes GRPO with group size of 2 and finds that it is able to achieve similar performance to group size of 16 on math datasets while using 70% wall clock time. Furthermore, a contrastive learning framework is used to analyze GRPO's advantage estimates under differing group sizes.

### Strengths
- A theoretical analysis under a constrastive learning framework for 2-GRPO is provided
- Established math datasets and benchmarks are used for post-training and evaluation

### Weaknesses
- My primary concern with the work is that while a 70% wall clock time improvement is significant, it appears that experimentally all that was done was changing the group size to 2
- I would need to see a rather elaborate experiments section to verify that G=2 is equally performant and significantly faster on a wider variety of post training tasks to fully believe the claim

### Questions
- Please address weaknesses

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a variant of the GRPO algorithm which takes group size to be 2. The paper proved that similar to DPO, GRPO with group size 2 can be considered as a contrastive loss. The paper then derives a few property of 2-GRPO, and perform experiments on standard math reasoning tasks and demonstrates the gain in efficiency with small performance degradation.

### Strengths
1. The observation that GRPO’s within-group normalization induces a signed “positive vs negative” split is a useful lens and does connect to generic contrastive forms.

2. The compute focus is valid given rollout cost dominates; the table shows large wall-clock savings.

### Weaknesses
1. The paper proves that both objectives admit gradients of a contrastive form (Def. 3.1 / Props. 3.2–3.3), but stops there. There is no mapping of corresponding terms in DPO to GRPO’s objective. This is just a nominal equivalence.

2. The assumption of no clipping is rather strong. I understand it's for the convenience of the derivation, but in reality it plays an important role on shaping the gradient distribution. 

3. Honestly most of the observations / derivations in this paper is either quite well known or elementary (e.g., properties of Bernoulli random variables).

4. The results actually show that taking group size 4 induces both huge save in training time and no performance gap w.r.t. group size 8 or 16, that makes the choice of group size 2 less ideal. 

5. This paper is less principled than many previous papers with adaptive group size. E.g., [1]

[1] Zhang, Ruiqi, et al. "SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning." arXiv preprint arXiv:2506.09016 (2025).

### Questions
With group size 2, what is the percentage of prompts that have no gradients? How does this change along training?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors perform theoretical analysis and experiments in support of a variant of GRPO with group size 2.

### Strengths
(+) A mixture of theory and practice.

### Weaknesses
(-) I think you're missing a log in Eq. 2. Similarly, the expectation in Eq. 5 clashes with the finite-sample group.

(-) On line 272, I think you mean variance not bias. Also, I don't think you meant proportional on line 276. More broadly, it's not clear to me why Prop 4.1 implies that 2-GRPO performs normalization. I think there's also an extra nabla on line 286.

(-) The rationale for increasing Q to compensate for lower G in Sec. 4.2 seems to rely on a strong unstated assumption: equal variance across all prompts. I believe the same is true for Sec. 4.3 -- let me know if I've misunderstood.

### Questions
1. Why is the gradient zero outside of the clipping range as you write above Eq. 8?

2. Can you explain the justification for Prop. 3.2? It also seems like you might be missing a log in Eq. 10?

### Soundness
3

### Presentation
3

### Contribution
3
