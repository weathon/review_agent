# Smarter Not Harder: Generative Process Evaluation with Intrinsic-Signal Driving and Ability‑Adaptive Reward Shaping

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
Large reasoning models (LRMs) have shown strong performance in complex mathematical reasoning when optimized via reinforcement learning (RL). However, conventional outcome-only reward provides sparse feedback, leading to inefficient optimization. In this work, we investigate whether generative process reward models (GenPRMs) can accelerate RL training of LRMs by improving the utilization of reasoning trajectories. We first analyze critical limitations in existing GenPRMs, including their heavy reliance on reasoning ability during correctness judgment, and suppression of exploration as well as vulnerability to reward hacking during reward assignment. To address these limitations, we first propose a novel \textbf{intrinsic-signal-driven evaluation} mechanism, which judges reasoning steps using semantic cues from the solution, thus mitigating extensive dependence on GenPRM. Furthermore, we (i) adopt \textbf{thought-level rewarding granularity} to alleviate over-dense step rewards, and (ii) design a \textbf{difficulty-aware reward formulation} that dynamically balances exploration and exploitation and keeping the optimization target of key tokens to mitigate reward hacking. We integrate these innovations into the process reward-based GRPO, resulting in the proposed \textbf{TP-GRPO} algorithm. Experiments on LRMs with 1.5B and 7B parameters show that TP-GRPO achieves higher improvements while using significantly fewer training samples, and more analyses further confirm the effectiveness of our proposed process evaluation mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tries to address key challenges in incorporating generative process reward models (GenPRMs) into RLVR. The identified challenges are: (1) the requirement for a GenPRM with strong reasoning ability; (2) the risk of reward hacking from over-densified rewards; and (3) the potential discouragement of policy exploration. The authors tackle challenge (1) by using semantic matching to locate correct and incorrect steps. They tackle challenge (2) by merging continuous steps with the same correctness into a larger block. They tackle challenge (3) by using an adaptive process reward assignment. By incorporating the process rewards provided by their framework into GRPO, they proposed TP-GRPO. The main results are evaluated on mathematical reasoning benchmarks.

### Strengths
- The paper is clearly written and easy to follow.

- By identifying three crucial challenges for GenPRMs, this paper provides meaningful guidance for future research directions in the field.

- The idea of using semantic alignment for process evaluation is intuitively sound, and it could potentially reduce the difficulty of this task for GenPRM.

### Weaknesses
**Lack of Evidential Support**

The paper makes several assertions that lack sufficient supporting evidence. For instance:

- At line 236, the paper states that "static rewards can unintentionally suppress exploration." At line 248, the paper states that "when $\text{acc}_G=0$, $r^c$ vanishes, reducing to outcome supervision GRPO and prioritizing exploration." Why static rewards would suppress exploration? And why, when reduced to outcome supervision, is exploration prioritized? Empirical or theoretical evidence is needed to support these justifications.

- At line 220, the paper states that "If process rewards are assigned at the step level ... which could in turn mislead the optimization to incorrect directions." This conclusion requires further justification. For example, if the process reward is assigned correctly at the step level, would it still mislead the optimization direction?

**Strong Assumptions and Limited Application Scenarios**

- At line 203, the "Identify Effective Steps" component depends heavily on the LRM capacity for accurate self-reflection, a capability not universal among all models. How can the proposed method be applied to those LRMs without self-reflection behavior?

- At line 205, the paper states "During reflection, the LRM typically analyzes the causes of the reflected mistake". This assumes that LRM can correctly diagnose the cause of a mistake during reflection. The case of incorrect self-diagnosis is not addressed.

- Even if the LRM has self-reflection and can correctly identify the location of a previous error, the requirement for the GenPRM to identify "steps dependent on... incorrect previous steps" (line 207) might still raise concerns about its need for non-trivial reasoning capabilities.

- At line 211, the hypothesis that "all steps in the answer are erroneous" does not generally hold. As established by previous work [1], incorrect trajectories often contain valid reasoning steps before the first error occurs. 

[1] Let's Verify Step by Step, arXiv 2023.

**Incomplete Experimental Setting**

To robustly support the claim that the evaluation scheme has "low reasoning requirements on the PRM," additional settings should be added: using models like Qwen3-32B, Qwen3-4B, and Gemma-3-12B-it as the GenPRMs without applying the proposed "reducing reasoning requirements" methods.

### Questions
- At line 248, the paper states that "$\text{acc}_G$ is the accuracy of the G sampled solutions for the same problem." The definition of $\text{acc}_G$ is confusing. Is it the average accuracy across G sampled solutions for the same problem or something else?

- Also at line 248, the paper states that "when $\text{acc}_G=0$, $r^c$ vanishes, reducing to outcome supervision" I am a little bit confused about why  $r^c=0$ would cause it to reduce to outcome supervision.

- Could $\text{acc}_G$ also be applied to the computation of $r^{ic}_i$ to achieve adaptive penalization according to the difficulty of the problem?

- I am confused about the definition of the advantage function in this paper. According to Equation 3 in the paper, it is more like the definition of returns (or cumulative rewards). According to the definition in [1], the advantage function measures whether an action is better or worse than the policy's default (or expected) behavior. I hope the authors will clear this up for me.

[1] High-Dimensional Continuous Control Using Generalized Advantage Estimation, ICLR 2016.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel reinforcement learning framework for large reasoning models (LRMs) that enhances reasoning efficiency via process reward modeling (PRM). The authors identify key limitations in existing generative process reward models (GenPRMs), notably their dependence on reasoning ability, over-dense reward signals, and susceptibility to reward hacking. To overcome these issues, the paper introduces an Intrinsic-Signal-Driven Evaluation method and a Thought-Level Ability-Adaptive Reward Mechanism, which together form the proposed TP-GRPO algorithm. Specifically, thought-level

Experimental results on the DeepSeek-R1-Distill-Qwen (1.5B and 7B) models demonstrate significant improvements in training efficiency while maintaining comparable performance. Also, the proposed techniques are less sensitive to the GenPRM's reasoning ability, successfully mitigating the reasoning dependency issue.

### Strengths
1. Novelty: The techniques of (1) reflection localization for determining the granularity of thoughts and (2) thought-level advantage adjustment are new to me. The paper also introduces several useful methods in Sections 3.1.1 and 3.1.2.

2. Quality: The paper accurately pinpoints three major weaknesses in current GenPRM designs: excessive dependency on reasoning strength, over-dense reward granularity, and reward hacking. The analysis is precise and well-supported by empirical and conceptual insights.

### Weaknesses
1. Clarity: It is better to highlight the contribution of the paper with a more concise title, such as "Improving Reasoning Efficiency via Thought-Level Reflective Reward Shaping." The terms "intrinsic-signal" and "ability-adaptive" are somewhat misleading, as the former can be easily confused with the conventional term "intrinsic reward" [1], and the latter could be replaced with more commonly used terms such as "difficulty-aware," given that "ability" sometimes refers to more general categories of skills like coding, math, tool use, etc.

2. Significance, Quality: In Tables 2 and 3, TR-GRPO is compared only with methods based on outcome reward models (ORMs), and additional comparisons with other PRM baselines are needed, such as [2][3][4]. It is also recommended to include comparisons with stronger ORM methods, such as those in [5].

3. Novelty: The thought-level idea is similar to the one in [4]. It is worth discussing the differences between the two works and making an empirical comparison if applicable.

4. Quality: The effectiveness of TR-GRPO would be more evident if the comparisons in Tables 2 and 3 were made fair by using the same budget or by measuring the number of samples needed to achieve the same overall accuracy.

I think the paper provides many useful techniques, but also has obvious issues, as mentioned above. I would be happy to consider raising my score if the aforementioned concerns are well addressed.

### References

[1]: Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised prediction." International conference on machine learning. PMLR, 2017.

[2]: Zhao, Jian, et al. "Genprm: Scaling test-time compute of process reward models via generative reasoning." arXiv preprint arXiv:2504.00891 (2025).

[3]: Zhang, Hanning, et al. "Entropy-regularized process reward model." TMLR, 2024.

[4]: Xiong, Wei, et al. "Stepwiser: Stepwise generative judges for wiser reasoning." arXiv preprint arXiv:2508.19229 (2025).

[5]: Chen, Minghan, et al. "Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization." arXiv preprint arXiv:2505.12346 (2025).

### Questions
* In Figure 3, the accuracy fluctuates significantly and is not monotonically increasing. I am wondering if this is still the case under different random seeds and whether the conclusion still holds.

### Soundness
3

### Presentation
2

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
This paper revisits GenPRM for RL of large reasoning models by using intrinsic semantic cues in the solution trajectory to judge step correctness (locating reflections, tracing the earliest error, and interval labeling), merging consecutive steps with the same correctness into “thoughts,” and applying thought-level, ability-adaptive rewards; on incorrect solutions, only thoughts semantically aligned with the wrong answer are penalized. Combined with GRPO as TP-GRPO, the method attains higher or comparable accuracy with far fewer training solutions on DeepSeek-R1-Distill-Qwen 1.5B/7B (e.g., 5.6K vs 34K; 8.56K vs 16K) and introduces the Effic. metric to quantify improved sample efficiency.

### Strengths
The research question and proposed method are interesting, e.g., alleviating GenPRM’s reasoning burden and the observation on dense step-wise rewards.

### Weaknesses
The method's reliance on the LRM's own capacity for self-reflection and annotation is a fundamental limitation. This approach is inherently self-limiting, as any improvement is capped by the model's existing capabilities and the task's difficulty, hindering the acquisition of novel skills. The poor performance in Table 1 exemplifies this concern.

The experimental setup is questionable. The RL training is remarkably inefficient, yielding only a 2-4% accuracy gain over 400-1000 iterations—a potential artifact of an undersized group size for GRPO that undermines the paper's credibility. Furthermore, the efficiency comparisons in Tables 1 and 2 are inequitable due to mismatched hyperparameters (e.g., group_size, batch_size) across baselines, and a performance comparison at convergence is conspicuously absent.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper begins with an in-depth analysis of the challenges inherent in Generative Process Reward Model (GenPRM) based process evaluation. To mitigate these challenges, the authors propose a novel generative process evaluation mechanism. This mechanism features an intrinsic-signal-driven evaluation (judging reasoning steps based on semantic information) and thought-level, ability-adaptive reward schemes. The mechanism is integrated with the GRPO algorithm to form a new RL algorithm, termed TP-GRPO. Experimental results on mainstream reasoning benchmarks demonstrate that TP-GRPO achieves superior training efficiency and accuracy compared to baselines.

### Strengths
1. The paper is well-written and presents the topic clearly.

2. The proposed method is well-motivated and appears technically sound.

3. The empirical evaluation is promising. TP-GRPO is shown to outperform the GRPO baseline and most existing outcome reward-based RLVR methods in terms of both training efficiency and accuracy.

### Weaknesses
1. Notable performance gap remains when comparing TP-GRPO to state-of-the-art models with much larger training budgets (e.g., DeepScaler-1.5B-Preview and Skywork-OR1-7B). The paper does present initial scaling trend (i.e., Fig 3), but the experiment appears to be conducted with relatively low training budgets. This limited scope makes it difficult to conclusively determine whether the advantages of TP-GRPO will persist, widen, or saturate as model scale and training budgets increase significantly.

2. The introduction clearly articulates several "design pitfalls" of existing GenPRM-based process evaluation. While the paper does provide some analyses related to these points, the insights are somewhat scattered throughout the experimental section rather than being presented cohesively. The paper would be significantly strengthened if this analysis were consolidated and made more explicit, clearly demonstrating how the proposed method directly mitigates each of the identified pitfalls.

### Questions
Overall, this paper is well-written, well-motivated, and the proposed method is technically sound. The empirical results are promising, demonstrating clear improvements over strong baselines, even if they do not surpass all current SOTA methods. In order to maintain my rating, I would like the authors to address the points in the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
