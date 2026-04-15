# Overcoming Knowledge Barriers: Online Imitation Learning from Visual Observation with Pretrained World Models

- Decision: Reject
- Scores: 5, 6, 3, 6

## Abstract
Pretraining and finetuning models has become increasingly popular in decision-making. But there are still serious impediments in Imitation Learning from Observation (ILfO) with pretrained models. This study identifies two primary obstacles: the Embodiment Knowledge Barrier (EKB) and the Demonstration Knowledge Barrier (DKB). The EKB emerges due to the pretrained models' limitations in handling novel observations, which leads to inaccurate action inference. Conversely, the DKB stems from the reliance on limited demonstration datasets, restricting the model's adaptability across diverse scenarios. 
We propose separate solutions to overcome each barrier and apply them to Action Inference by Maximising Evidence (AIME), a state-of-the-art algorithm.
This new algorithm, AIME-NoB, integrates online interactions and a data-driven regulariser to mitigate the EKB. Additionally, it uses a surrogate reward function to broaden the policy's supported states, addressing the DKB. Our experiments on vision-based control tasks from the DeepMind Control Suite and MetaWorld benchmarks show that AIME-NoB significantly improves sample efficiency and converged performance, presenting a robust framework for overcoming the challenges in ILfO with pretrained models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
**Problem setting**: have access to a demonstration data set without actions, an offline data set of suboptimal experience, and the ability to sample experience online without reward feedback

**Proposed approach**: (1) balance the offline and online datasets when updating the model, (2) combine AIME and AIL objectives (though also considers other alternatives to AIL)

**Experiments**: comparisons to relevant prior methods on DMC and MetaWorld tasks with visual observations

### Strengths
- Writing is fairly straightforward to understand 
- Results show clear improvements over relevant methods
- The paper includes fairly extensive ablations/empirical analysis

### Weaknesses
Overall, the technical contribution seems like a fairly basic combination of prior works. While many methods build upon prior works, this particular combination seems closer to an implementation choice than a significant technical contribution. The method is also more complex than prior methods as it introduces one hyperparameters and combines multiple objectives. 

Other weaknesses
- It’s a bit unclear how this connects to a real-world problem. In robotics, you typically either collected data from a robot, where you can get actions or something close to them, or videos of humans/animals, which lacks actions but also has significant domain shift. Connecting the problem statement to a real world problem would be helpful to motivate the paper’s significance.
- EKB and DKB are defined quite informally and then are used extensively, sometimes in a hand-wavy way. I’m not sure if introducing them is helpful for understanding. I think it would be better to either remove them or define them more formally and measure the extent to which they contribute to poor performance.
- I appreciate that the experiments are on visual observations, though they would be stronger if they included more complex tasks such as dextrous tasks, longer horizon tasks, or tasks with greater diversity/generalization

Some relevant related works to consider citing/discussing:
- Mobile (https://arxiv.org/abs/2102.10769), though uses low-dim observations
- VMAIL (https://arxiv.org/abs/2107.08829), though assumes actions in demos

### Questions
See suggestions in the weaknesses

### Soundness
3

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
4

### Summary
This paper studies the problem of imitation learning (IL) from (visual) observation, i.e., demonstrations do not contain action information using pretrained world models. The paper identifies two key bottlenecks in current IL algorithms, namely (1) OOD observations, and (2) OOD task configurations, which the authors refer to as the Embodiment Knowledge Barrier (EKB) and the Demonstration Knowledge Barrier (DKB), respectively. The key technical contribution is an algorithmic extension of the method Action Inference by Maximizing Evidence (AIME); the core idea is to finetune the learned IL policy via limited online interaction and a RL objective, with surrogate rewards derived from the demonstration dataset. Experiments are conducted on DMControl and Meta-World from visual observations, and the proposed method compares favorably to both AIME without finetuning as well as other IL methods that can be finetuned online (BCO, OT, PatchAIL).

### Strengths
- The paper is generally well written and easy to follow. Illustrations (Figure 1 in particular) are useful for understanding the proposed method, and Section 2 provides sufficient background for an unfamiliar reader to appreciate the contributions.
- Experiments are reasonably extensive, covering a variety of tasks from DMControl and Meta-World, as well as several (what appears to be) baselines appropriate for the problem setting. Empirical results are strong compared to baselines.
- It is a purely empirical paper, but there are sufficient ablations to understand how the algorithm behaves in different settings (e.g. number of demos), and which components are the main drivers of performance (e.g. choice of surrogate reward).

### Weaknesses
- I appreciate that the authors conduct experiments on both tasks from DMControl and Meta-World, and consider tasks with varying difficulty in the case of Meta-World. However, I would have liked to see a few examples of tasks that are more challenging, i.e., where the proposed method (along with baselines) struggle a bit more. For DMControl this could be e.g. any of the Humanoid or Dog tasks, and for Meta-World this could be e.g. Stick Pull / Push or Pick Place (Shelf).
- I find that some of the claims / conclusions are a bit exaggerated relative to what the experimental results show. For example, the authors claim that " the model pretrained on MW-mt50 offers much better results" (L424) while the results in Figure 4(c) show a fairly small difference between the two curves (absolute ~15% increase in success rate on avg it seems). I would prefer claims to accurately reflect the evidence.

### Questions
I would like the authors to address my comments listed in "weaknesses" above during the rebuttal. I have one additional question:

- Figure 10 indicates that the proposed method succeeds in all of the considered tasks except Cartpole Swingup. I would have expected this task to be the easiest of them all; can the authors please comment on why the method fails on this particular task?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper extends AIME to AIME-NoBarries, which addresses EKB with online interaction and DKB with online trajectories labeled with surrogate rewards. The authors evaluate the effectiveness of AIME-NoB in DMC and MetaWorld.

### Strengths
- AIME-NoB alleviates EKB through online interaction. To address DKB, the most natural way is to collect more expert demonstrations, which is expensive in terms of robot manipulation. Therefore, AIME-NoB circumvents this burden by providing reward signals using surrogate models and optimizes the policy with reward-labeled online trajectories.
- Experiments in DMC and MetaWorld show that AIME-NoB can bring significant improvement to AIME.

### Weaknesses
- The logic is confusing. The authors state that in order to alleviate EDK, they additionally use online interactions. It is obvious that online trajectories will bring more benefits since the behavior policy to collect offline datasets differs from the current policy. So the improvement compared to AIME might be largely caused by the setting difference rather than algorithm improvement.
- The setting is very complicated. To my understanding, AIME's setting includes an offline dataset $(s_{\mathbf{off}},a_{\mathbf{off}})$ for world model training, an expert dataset $(s_{\mathrm{expert}})$ for imitation learning. And AIME-NoB additionally assumes access to the environment to collect an online dataset $(s_{\mathrm{on}},a_\mathrm{on})$. The complexity of the setting makes it hard to make fair experimental comparisons as baselines might only assume access to a part of the datasets.

### Questions
NA

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies two barriers in imitation learning from observations, namely the Embodiment Knowledge Barrier (EKB) and Demonstration Knowledge Barrier (DKB). EKB refers to the gap when generalizing to new observations and DKB refers to the gap when generalizing from limited number of expert demonstrations.

It proposes to use online interaction to reduce EKB and introduces a weighted loss between new interactions and pre=training data. For DKB, it proposes to use a surrogate reward, such as a discriminator in AIL.

Experimental results suggest AIME-NoB outperforms baselines with significant margin in DMC and MetaWorld.

### Strengths
- Clearly identify the shortcomings of existing IFO methods and propose reasonable solutions to reduce the gaps.
- Extensive experiments on the different design choices, e.g. reward functions, and comparisons with baselines demonstrate solid improvement.
- Paper writing is clear and easy to follow.

### Weaknesses
- The ideas to reduce EKB and DKB seems a simple combination of previous methods, such as using online interaction with weighted sampling, adversarial training, etc. It's unclear the distinctions from previous methods (such as AIL, e.g.) is significant enough.
- These environments aren't hard to come up with a hand-designed reward or learning a surrogate reward. Lack of comparisons with Dreamer-like methods using hand-designed or surrogate reward.

### Questions
- How do you differentiate from previous work that uses online interactions and surrogate rewards?
- How does it compare with RL with some hand-designed reward or a learned surrogate reward?
- Does the method generalize to real-world environments, such as learning from videos?

### Soundness
4

### Presentation
3

### Contribution
3
