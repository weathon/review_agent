# Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with Stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are available at https://github.com/starreeze/latr.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Lookahead Tree-Based Rollout (LATR) to enhance trajectory-level diversity among rollouts in a group to improve the performance of GRPO and DAPO in policy refinement.

### Strengths
1- The paper is well written and easy to follow.
2- The diversity problem in RL is well known, and this paper clearly identifies it within the RLVR paradigm.

### Weaknesses
1- The literature review is not thorough. Many approaches that combine look-ahead reasoning with LLMs are not mentioned.

2- The baselines are neither strong nor state-of-the-art. Although the paper cites works addressing diversity at the token level, the comparisons use only stochastic sampling; it is unsurprising that look-ahead search would outperform simple stochastic sampling.

3- The performance gains are not substantial, especially in terms of correctness.

4- Although inference cost is reduced, there is no comparison of training costs between the stochastic and look-ahead approaches. I would expect the look-ahead method to require significantly more training time.

### Questions
1- The paper states that the tree-search approach is inspired by MCTS. Could you clarify how your tree search is similar to the MCTS used in AlphaZero [1]?



1- Mastering the game of Go with deep neural networks and tree search.

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
The paper presents LATR: Lookahead Tree-based rollouts for improving the diversity of the rollouts in algorithms like GRPO.
LATR operates in 3 stages: Branching which creates new trajectories when the model is highly uncertain, (2) Lookahead simulation to extend a bnrach and Pruning when branches that are similar are pruned.

Branching occurs at each point with the highest logit-based token being extended to capture the most likely trajectory. To encoureage diversity, if other logits satisfy a threshold, child branches are created. The authors use dual-thresholds to prevent diverging too far from the models distribution.

Since it is computationally expensive to branch at each logit/token, the authors uses lookahead simulation. After a branch, each trajectory along the branch is generated sequentially for a window or r tokens. Finally, pruning is performed for these sequential generations so that branches that are not sufficiently different are pruned.

The authors explore more optimizations like early stopping and hybrid rollouts akin to epsilon in RL settings.

The authors then conduct an empirical evaluation and compare policy optimization methods with LATR and compare to stochastic sampling. The authors also provide ablations to contrast the effect of different components.

### Strengths
1. The paper is clear and well-written

2. The proposed technique is intuitive and explained well. Using tree-based methods to focus on diverse trajectories based on model certainty is intuitive.

3. The experiments showcase model improvements in performance but significant improvements in the inference costs.

### Weaknesses
1. I think one major weakness in the paper is the empirical evaluation which is my primary reason for a lower score.

There is only 1 model considered. I think to better understand the performance of LATR you would need to evaluate more models.

2. It is not clear how statistically significant the results are since standard deviations are not provided. How many times were the models trained for these experiments.

3. Other relevant baselines like TreeRL are not considered. I believe TreeRL also seeks to improve diversity by employing entropy.

4. What is the threshold for EditDistance used in the experiments?

5. Im not sure how useful Ablation 5.3 is. You are comparing two different methods. While LATR is not sensitive to temperature and this is not super surprising, the interesting analysis IMO is to showcase the sensitivity to all these new hypermeters/thresholds you have introduced.

### Questions
Overall, this paper is interesting and the results seem okay.  There are a few weaknesses that the authors can clarify before I can increase my score. Happy to discuss further.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an enhancement to the reinforcement learning with verifiable rewards algorithm.

Motivation

Previous existing methods for RLVR, such as GRPO and DAPO, suffer from limited diversity in the sampled trajectories. Follow-up works mitigate this by (i) including increasing sampling temperature or (ii) filtering out and rejecting sequences with very similar samples. However the authors argue that  (i) i focuses on token-level diversity, rather than trajectory-leval diversity, and (ii) needs multiple rollouts leading to high cost, and  lacks withing-group diversity.

Method

To mitigate this, the authors propose  Lookahead Tree-Based Rollout (LATR) algorithm, inspired by MCTS that works as follows: 
starting from a root branch that corresponds to the prompt:
- first there is a branching stage, where each branch is extended with its highest-probability token (this is the parent branch) , as well as other tokens that have a "high-enough probability"  (these are the children branches), This is done until we have a set of K sequences.
- then, there is a pruning stage: given a lookahead step r , they look at each children branch that already had r rollouts from their parent branch, and they look at how different this children branch is from the parent branch (using edit distance). If differrent enough, the children branch and its descendants are kept. If not, it is removed along with its descendants.

They introduce further optimizations eg.. slowly decreasing the ratio of sequences generated with LATR during training in favor of standard stochastic sampling (more exploration in the start, more exploitation in the end of training) 

Experiments 

They run experiments on the CountDown,  DAPO-math, ACM23, MATH-500, Olympiad-Bench datasets. They run GRPO w LATR / DAPO w LATR, and they compare them to the baselines od GRPO w stochastic sampling / DAPO w stochastic sampling. Performance is measured in terms of (i) average correctness of the provided answer and (ii) average length of the answer provided (as well as format for CountDown). In almost all cases, LATR shows better performance in both metrics compared to the baselines. Taining dynamic results reveal faster learning (in terms of avg val reward) with LATR. 

They also run ablation studies on the diversity comparison across answers with / wo LATR, the effect of the pruning, and of sampling at different temperatures.

### Strengths
- Good presentation of the paper, clarity and ease of reading
- Good contextualization of the paper in previous work, helpful for those who are not experts in this field
- Simplicity of the introduced changes and strong evidence that it yields improvements 
- Clear experiments section with ablation studies

### Weaknesses
- I think an important ablation study is missing to show the claim of the paper that trajectory level lookahead is important and yields improvements over token-level lookahead.  
- Also, the authors mention that "token-level variations typically occur without lookahead ability, making local deviations (e.g., substituting “compute” with “calculate”)  " however this can also happen in trajectory level variations (with multiple such subsitutions). The authors use the edit distance to quantity diversity, however this distance assigns high value to word substitutions that mean the same thing.

### Questions
- Given point 1 above, Could you show ablation studies on the lookahead step r, in particular with r = 1 (i.e. only one-step lookahead but with the pruning that removed redundancy ) ? I would be curious to know if trajectory-level lookahead is important, or if token-level lookahead with diversity pruning is enough. 
- Given point 2 above, do you think edit distance a good measure of diversity between sequences? You could have 2 sequences that have high edit distance but mean the same thing because there are synonym words. Have you thought of using another distance e.g.  distance between the embeddings of the sequence, to make sure that they are semantically different? 
- In the diversity experiments, "two answer expressions ...  if their evaluated numerical outcomes differ". I am confused: for each math question there is only one correct answer, and you can have multiple diverse ways to reach this correct answer. Why aren't you looking at this diversity instead (rather than diversity over final answers, for which high diversity means that some answers are incorrect! )
- For the temperature-ablation experiments, do you keep the same theresholds tau_abs and tau_rel across temperatures? If yes, could you elaborate why is LATR less sensitive to temperature tuning? The branching quality in LATR will still be affected by the increase in temperature (because of the increased stochasticity in the new branches ), no?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a tree-based algorithm to increase the sampling diversity in the setting of RLVR.  The method focuses in generating trajectory-level diversity during the sampling process by rolling out and pruning semantically similar trajectories. The show that including this sampling strategy with GRPO and DAPO, the performance increases given the new variety of sampled reasoning trajectories. They evaluate their method on math and logical reasoning tasks.

### Strengths
- The paper is generally well written and the method is explained clearly.
- The empirical evaluation shows the benefits of the enhanced strategy with respect to token-level stochastic sampling

### Weaknesses
- The paper claims to care about semantic similarity but they prune based on Edit distance which doesn’t seem to me to be a good measure of semantic similarity. Maybe could the authors explain why this is working in their evaluation tasks?
- It seems that the new sampling process introduces could potentially introduce off-policy issues? Is this being taken into account in $\pi_{old}$, or am I misunderstanding  something?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
