# Towards demystifying the optimization landscape of RLVR methods

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
GRPO has achieved impressive success in the landscape of reasoning models. However, the motivation behind its origins along with the reasons for its effectiveness remain elusive. In this work, we fill some of the gaps and demonstrate that in on-policy setting, GRPO's optimization can be viewed as a weighted combination of maximization of likelihood for correct rollouts and minimization for the incorrect ones. This finding gives a different perspective about the optimization landscape of GRPO. Motivated by this, we analyze the positive and negative part of GRPO's objective function independently, and find that their global minima correspond to undesired solutions. While optimization of the positive term leads to entropy minimization and length collapse, optimizing for the negative term leads to entropy maximization and length explosion. Using this lens, we show the presence of instability in on-policy training of some recent algorithmic advances trying to simplify GRPO's objective. 
However, despite the presence of bad global minima in GRPO's objective function, it doesn't converge to either of them. We identify design choices in GRPO's advantages that aid convergence of GRPO to good minima. We also demonstrate the effectiveness of using clipping in stabilizing the optimization process, thereby preventing training instabilities even when training only for minimizing the likelihood of incorrect rollouts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper demystifies the GRPO algorithm, widely used for LLM reasoning, by re-framing its on-policy optimization as a weighted combination of maximizing the likelihood of correct rollouts and minimizing that of incorrect ones. The authors demonstrate that optimizing either of these objectives independently leads to unstable bad minima, characterized by either entropy collapse or entropy explosion. The study reveals that GRPO's success stems from its specific advantage calculation and the use of clipping, which work together to stabilize the training process and prevent convergence to these undesirable solutions.

### Strengths
1. The paper is well-written and easy to follow.

2. The paper offers an intuitive way to understand the optimization of RLVR methods. It reframes the objective as a balance between two competing forces: maximizing the likelihood of correct responses and minimizing the likelihood of incorrect ones. This model clearly explains why simpler methods are unstable, showing they can lead to "length collapse" or "length explosion" when one force overpowers the other.

3.  The paper presents a practical discovery: off-policy training, when combined with clipping, can be significantly more stable than its on-policy counterpart. This finding challenges common assumptions about on-policy methods and provides a valuable, counter-intuitive insight for practitioners.

### Weaknesses
1. While the paper's analysis of the problem is insightful, the solution it proposes—using token-level normalization—is not new. As the authors acknowledge, this technique is already a key component in several other recent and successful methods. Therefore, the paper's contribution feels more like a strong explanation for why an existing method works, rather than a new solution derived from its analysis indicating that, while the analysis is valuable, it doesn’t lead to any substantively new insight or proposal derived from the authors’ interpretation and findings.

2. The paper's core claims are about training instability and collapse, which are phenomena often highly sensitive to random seeds and initialization. The authors state that all experiments were run only once. This could be regarded as a significant limitation considering the instability of GRPO algorithm. The claims would be much stronger if they were supported by results averaged over multiple runs(seeds) to show variance and confirm that the observed collapses are consistent.

3. While the paper identifies that clipping is the key to off-policy stability, it admits a "do not have a complete understanding"  of the underlying mechanism why it works. Understanding how clipping "induces stability" is left as an "interesting future direction", making the paper's "demystification" partially incomplete.

4. I believe that "RLVR method" term in the title is too broad considering the algorithm handled in the paper. I would recommend that the authors consider changing the title. (for instance, changing "RLVR method" into GRPO)

### Questions
1. The paper attributes the observed instability primarily to the 'on-policy setting' itself. However, the GSPO[1] posit that the instability in GRPO does not stem from the on-policy setting, but rather from the high-variance noise introduced by its fundamentally flawed 'token-level importance sampling' design. GSPO algorithm is also an off-policy method, yet it achieves stable training where GRPO fails. Do the authors believe their perspective—that clipping is the critical component for stability in combined with off-policy methods—generalizes to sequence-level algorithms like GSPO as well? In other words, under a sequence-level optimization framework, do you still consider the clipping mechanism to be equally critical for maintaining stability than on-policy setting??

[1] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, & Junyang Lin. (2025). Group Sequence Policy Optimization.

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
The paper investigates the GRPO algorithm for optimizing LLMs in a reinforcement learning with verifiable rewards (RLVR) setting. The findings highlight differences between on-policy and off-policy training, the importance of training both on positive and negative samples, and the importance of likelihood ratio clipping. The authors show that removing certain components can lead to training instabilities.

### Strengths
The paper provides an in-depth analysis of different components of the RLVR optimization of LLMs. Evaluation is done on different datasets, and the trends seem to be similar across the different datasets.

### Weaknesses
Certain parts of the paper are not very clear (see Questions).

Furthermore, the paper claims that "PPO collapses in on-policy setting". This claim seems misleading. On-policy PPO here means that $\pi_{\theta_\mathrm{old}} = \pi_\theta$, which means that the ratio $\pi_\theta(a|s) / \pi_{\theta_\mathrm{old}}(a|s)$ is always 1. This, in turn, means that the clipping is never active, and what is left is essentially a vanilla policy gradient algorithm (similar to on-policy GRPO in eq. 6). Since the clipped loss is the central component of PPO, this notion of "on-policy PPO", therefore, does not bear a lot of resemblance to PPO anymore.

### Questions
1. Figure 1 is not clear. What exactly is 1(a) showing? Is this just an illustration, or is this some visualization of a loss landscape? Which loss is shown here? I assume the positive + negative likelihood loss? What is C_DL? The caption only says that it "leads to improved performance". (b) and (c) are also not clear. What do the black dots, arrows, and orange / brown curves represent? What are the little lines on the top right of the curves? 

2. The caption of Figure 1 states that "importance sampling reduces the norm of the gradients, resulting in slower convergence". However, I did not find any data in the paper backing this up.

3. C_NL and C_PL are the minima of L_NL and L_PL, respectively. This should not directly mean that the they are also minima of L_CL, which is the (weighted) sum of L_PL and L_NL, but the paper often treats these points as minima of L_CL or even GRPO's loss (e.g., in section 4.3 "Clearly, the two critical solutions C_PL and C_NL [...], are critical solutions of L_CL as well" or in the Figure 1(a)). I would appreciate it if the authors could clarify why these points are also minima of L_CL / GRPO's loss. 

4. Section 4.3.: At some point, the critical solutions are referred to as S_PL and S_NL, instead of C_PL and C_NL. Do both refer to the same thing?

5. The paper repeatedly claims that it is surprising that off-policy training is more stable than on-policy training, e.g., in section 7. To me, this does not seem surprising since in on-policy training, problematic updates have a very immediate and potentially catastrophic effect on the training data for the next updates, which makes it hard to recover from the suboptimal update. In the off-policy case, the data-collecting policy and the optimized policy are somewhat decoupled, which can help with this problem. Increasing the stability of training was also the reason why, e.g., DQN uses a replay buffer. I would appreciate it if the authors could elaborate on why they expect on-policy training to be more stable than off-policy training.

Typos:

1. Section 4.1: "decease" --> "decrease"

2. Section 4.1: "in length of model's entropy" 

3. Section 4.3.: "the gradients becomes zero" --> "the gradients become zero"

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper starts by analyzing GRPO. The authors "demystify" it by breaking its objective function into two parts: a positive term that maximizes the likelihood of correct answers and a negative term that minimizesthe likelihood of incorrect answers.

The paper's key findings are:
* GRPO is stable because its advantage calculation acts as a built-in stabilizer, preventing it from converging to these bad solutions.
* PPO's value estimators have "large errors from the true estimates".
* Off-policy training with clipping is more stable than standard on-policy training, as clipping also prevents these collapses.

### Strengths
1. The paper is easy to read and easy to follow.
2. The hypotheses raised by the authors are sound and accompanied by experiment observations.
3. The authors carry out experiments on multiple datasets, which cross validate their ideas.

### Weaknesses
1. The instabilities and collapses the authors identify are well-known failure modes that arise from the combination of function approximation, bootstrapping and off-policy learning, namely deadly triad [1]. 
2. PPO's value estimators having "large errors from the true estimates" is also a well known issue as critic models tend to overestimate values [2].
3. Becuase of 1 and 2, I question the novelty of this paper. i.e., I don't think this paper has enough new insights, nor does this paper offers novel solutions (clipping is not novel) to the above findings that achieve SOTA results.
4. I also think Figure 1 is really confusing. Why on the loss surface, C_DL is the most optimal trajectory but C_NL leads to the minimum entropy? Also I don't understand the illustration of Figure 1(c) completely.

[1] Van Hasselt H, Doron Y, Strub F, Hessel M, Sonnerat N, Modayil J. Deep reinforcement learning and the deadly triad. arXiv preprint arXiv:1812.02648. 2018 Dec 6.

[2] Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

### Questions
See the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper attempts to deliver concrete insights into the specific reasons that GRPO has proven to be an effective RLVR optimization technique for reasoning. The work presents a series of ablative experiments, focused primarily on elements of policy gradient algorithms deriving from PPO (e.g. clipping, reweighting with advantage functions, etc) to demonstrate the differences in training stability as well as investigate the differences between on- and off-policy training.

### Strengths
I am grateful for works such as this paper that attempt to explore the specific contributions of known methods. These papers are critically important for improved scientific understanding and help to develop improved algorithmic approaches. This papers sets an ambitious goal to study the effects of various elements of GRPO across datasets, model family as well as training paradigm. 

I felt that the most interesting section of the analysis came in Section 4.3 where the paper dug into the reweighting mechanism of GRPO which appears to balance between negative and positive generations. This led into a potentially deeper insight into the approximate advantage function used in GRPO. I was left wishing that the majority of the paper focused on this analysis rather than trying to cover every aspect of what the authors identified as contributions of GRPO for RLVR.

### Weaknesses
Overall, I'm not entirely sure that this paper introduced any novel insights beyond those identified in the literature they cited that have deeply investigated specific aspects of GRPO. While these unified surveys can be really useful if done rigorously and thoroughly, I do not feel that this paper meets those criteria. In many ways, it feels that the paper is trying to do too much at once and comes across as unfocused. This led to an overly distracted presentation in the paper where proposed insights are not deeply motivated or justified.

Perhaps the major flaw of this work is that it comes across as unaware of the RL theory underlying policy gradient methods, where many of the proposed insights about stability, variance and the trade-offs between on- and off-policy training have been well understood for decades. The discussion in the final paragraph largely restates the principled motivations that led to the development of trust region policy gradient approaches (of which TRPO and, later, PPO derive from). Policy gradient methods have been known to be reweighted MLE objectives since their introduction. Reformulations of policy gradient methods as weighted regression have further established this relationship (see, Peters and Schaal, "Using Reward-weighted Regression for Reinforcement Learning of Task Space Control" (2007)). 

Within the perspectives of LLM Reasoning, where I feel the authors are largely situated within, I think that there are some errors in the proposed development of the component losses for positive and negative generations individually. This is especially true within the lens that these are ablations of the GRPO objective since there is no controlling of simplified advantage function being a relative estimate of the group. Without this, it's not clear whether or not $L_{PL}$ and $L_{NL}$ are valid comparisons. There was not sufficient justification or grounding of these re-derivations to ensure that the policy gradient objective was not affected through the use of a biased baseline (for more about the use of baselines in policy gradient approaches, I'd highly recommend this recent blog post: https://fatemi.github.io/posts/pg-baseline/).

The GRPO objective provided in Equation 3 is incomplete as there is no aggregation over the group, including length normalization. Please revisit Shao, et al (2024). This omission leads me to have less confidence in the remaining development of the various objectives and the resulting analyses. This concern extends to the imprecise manner in which the advantage approximations are made throughout Section 4. 

The insights from Takeaway 2 are well established in the community, particularly those surrounding the effect of negative gradients. Please see the following two papers for detailed analysis.
- Setlur and Yang, et al (2025), "e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs"
- Fatemi, et al (2025), "Concise Reasoning via Reinforcement Learning"

### Questions
I do not have any further questions for the authors beyond the concerns raised in the "Weaknesses" section above.

### Soundness
2

### Presentation
1

### Contribution
1
