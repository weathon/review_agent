# Leftover Lunch: Advantage-based Offline Reinforcement Learning for Language Models

- Decision: Accept (poster)
- Scores: 5, 6, 8, 6

## Abstract
Reinforcement Learning with Human Feedback (RLHF) is the most prominent method for Language Model (LM) alignment. However, RLHF is an unstable and data-hungry process that continually requires new high-quality LM-generated data for finetuning. We introduce Advantage-Leftover Lunch RL (A-LoL), a new class of offline policy gradient algorithms that enable RL training on any pre-existing data. By assuming the entire LM output sequence as a single action, A-LoL allows incorporating sequence-level classifiers or human-designed scoring functions as
rewards. Subsequently, by using LM’s value estimate, A-LoL only trains on positive advantage (leftover) data points, making it resilient to noise. Overall, A-LoL is an easy-to-implement, sample-efficient, and stable LM training recipe.

We demonstrate the effectiveness of A-LoL and its variants with a set of four different language generation tasks. We compare against both online RL (PPO) and recent preference-based (DPO, PRO) and reward-based (GOLD) offline RL baselines. On the commonly-used RLHF benchmark, Helpful and Harmless Assistant (HHA), LMs trained with A-LoL methods achieve the highest diversity while also being rated more safe and helpful than the baselines according to humans. Additionally, in the remaining three tasks, A-LoL could optimize multiple distinct reward functions even when using noisy or suboptimal training data.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The widely used RLHF algorithmic backbone, PPO, can incur additional computational overhead mode collapse due to its online learning nature. Viewing this problem, this paper proposes an offline policy gradient method, Advantage-Leftover Lunch RL (A-LOL), which optimizes LMs towards desired rewards using only static data. Specifically, A-LOL considers the entire output sequence as a single action, and calculates training data advantage before filtering unfavorable instances.
The proposed A-LOL is easy to implement over standard cross entropy loss by adding sequence-level reward-weighting and importance-sampling weights.
In experiments, the proposed method shows competitive results and data efficiency on our different language generation tasks.

### Strengths
1. The proposed method is clear and easy to implement, with relatively few assumptions. Therefore, it may have practical merits.
2. The experiments are relatively throughout and the results are promising.
3. Human study helps demonstrating the efficacy of the proposed method. 
4. The paper is generally easy to follow.

### Weaknesses
1. Unclear method contribution: the proposed method is nearly, if not exactly, a special case of TRPO/PPO method in the bandit setting. Such a special bandit instantiation has been widely considered in classical RLHF works, such as [1,2].

2. Advantage-weighted policy optimization is also a well-studied method in offline RL. For example, Eq. 5 in this paper is very similar to Eq. 4 in AWAC [3], except for the importance weighting that basically comes from TRPO/PPO.

3. The formulation of considering the entire output sequence as a single action step may suffer from exponentially large action space, which may make policy training harder and less stable. See for example [4, 5]. As an aside, recent works have already tried to learn a per-token reward function that incorporates arbitrary human-designed scoring function(s), which may better cope with the large action space in NLG problem, see, e.g., [6].

4. Weighted behavior cloning has been quite extensively used in prior NLP papers, e.g., [6,7,8,9,10,11]. It will make the algorithmic contribution of this paper more clear if the authors can have a paragraph discussing and comparing with such related works, instead of only citing the CRR paper from offline RL.

5. In Table 2, the comparison with PPO may not be fair, because the reward for PPO is a good-or-bad classifier. The offline RL methods, on the other hand, are fitted towards the original responses that themselves show high linguistic diversity, which would implicitly guide the offline RL methods, especially A-LoL, towards generating longer and more diverse sequences. In short, there is no guiding signal for PPO to generate such sequences, while the offline RL methods implicitly have the guidance.

6. There are several well-established exogenous components in the proposed method, such as (1) importance clipping, (2) discarding negative advantage datapoints, (3) prioritized sampling. It is unclear how each of those exogenous components contribute to the overall performance. It is also unclear if the baselines can also benefit from such exogenous components, e.g., (2) and (3). This again muds the algorithmic contribution of the proposed method and make the experiment results less convincing.

[1] Stiennon, Nisan, et al. "Learning to summarize with human feedback." Advances in Neural Information Processing Systems 33 (2020): 3008-3021.

[2] Ouyang, Long, et al. "Training language models to follow instructions with human feedback." Advances in Neural Information Processing Systems 35 (2022): 27730-27744.

[3] Peng, Xue Bin, et al. "Advantage-weighted regression: Simple and scalable off-policy reinforcement learning." arXiv preprint arXiv:1910.00177 (2019).

[4] Guo, Han, et al. "Text Generation with Efficient (Soft) $ Q $-Learning." (2021).

[5] Snell, Charlie, et al. "Offline rl for natural language generation with implicit language q learning." arXiv preprint arXiv:2206.11871 (2022).

[6] Yang, Shentao, et al. "Preference-grounded Token-level Guidance for Language Model Fine-tuning." arXiv preprint arXiv:2306.00398 (2023).

[7] Govardana Sachithanandam Ramachandran, Kazuma Hashimoto, and Caiming Xiong. 2022. [CASPI] Causal-aware Safe Policy Improvement for Task-oriented Dialogue. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 92–102, Dublin, Ireland. Association for Computational Linguistics.

[8] Feng, Y., Yang, S., Zhang, S., Zhang, J., Xiong, C., Zhou, M., & Wang, H. (2023). Fantastic Rewards and How to Tame Them: A Case Study on Reward Learning for Task-oriented Dialogue Systems. arXiv preprint arXiv:2302.10342.

[9] Norouzi, Mohammad, et al. "Reward augmented maximum likelihood for neural structured prediction." Advances In Neural Information Processing Systems 29 (2016).

[10] Sayan Ghosh, Zheng Qi, Snigdha Chaturvedi, and Shashank Srivastava. How helpful is inverse reinforcement learning for table-to-text generation? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 71–79, 2021.

[11] Marcin Junczys-Dowmunt, Roman Grundkiewicz, Shubha Guha, and Kenneth Heafield. Approaching neural grammatical error correction as a low-resource machine translation task. arXiv preprint arXiv:1804.05940, 2018.

### Questions
1. Is the proposed method an offline RL method or (online) off-policy RL method? I am a bit confused by some citations, e.g., "Degris et al., 2012" in the second paragraph of Section 1, which is titled as "Off-Policy Actor-Critic". 

2. How does the learned $\pi_\theta$ significantly improve over  $\pi_{ref}$? With the clipping technique in PPO, the learned  $\pi_\theta$ will be constrained within a "$\epsilon$-neighbourhood" around $\pi_{ref}$, which limits the room for possible improvement.
Note that in PPO $\pi_{ref}$ is constantly changing and hence allows the continuous improvement of $\pi_\theta$, but in this paper $\pi_{ref}$ is never updated during training (Appendix A). 

3. From Line 2 in Algo. 1, it's unclear how do the authors train the value function $V(x)$. Do the authors sample multiple $y'$ from $\pi_{ref}$ for *each* $x$? If yes, with this multi-sequence sampling, how would the proposed method save compute compared to standard PPO-style LM training? If no, then Line 2 in Algo. 1 will only regress to the reward $R$ of $y'$, which is a crude and high variance estimate of the state value $V(x)$.

4. Will "discarding the data points with negative advantages" worsen data scarcity and harm the quality of the LM generations? For example, even though those data points may be less advantageous with regard to the given reward, they may still be helpful for generating fluent text sequences.

5. Maybe Line 4 in Algo. 1 is "... not converge", instead of "... converges"?

6. How would you justify the approximate importance weight in A-LOL sequence? How is it different from the standard per-step importance sampling in RL? Given that it is the best performing method, it would be important if one can justify it.

7. Could you explain the term $\frac{\ln \pi_\theta}{\ln \pi_{ref}}$ in Eq. 6? Would it be better and easier to optimize if we use the log importance weight $\ln \frac{ \pi_\theta}{ \pi_{ref}} = \ln \pi_\theta - \ln \pi_{ref}$?

8. Will the offline RL methods converge if you only allocate one epoch of training steps?

9. In Fig. 2, is NLL trained on the same number of steps and plotted on the same step counts as other methods that are trained on top of the NLL-based reference policy? 

10. How is the training time of value function compared to the training time of policy $\pi_\theta$? Is it fair compared to NLL?

11. Why are preferences inherently unidimensional and cannot be extended to tasks where more than one aspect is important? Why couldn't humans make judgement based on multiple axes?

12. Would the success of A-LoL sequence contradict the single-action assumption?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces advantage-leftover lunch RL (A-LoL) which is a class of offline policy gradient algorithms. Crucially, different from most past work in RL+NLP literature, the assumption is that the entire LM output is a single action. 

The algorithm is essentially MLE loss (standard cross entropy loss for text generation) but multiplied by sequence-level advantage and importance weight. Algorithm 1 contains the pseudo code. A few tricks are used, including clopping importance weight, advantage priority sampling (simply discard examples with negative advantage). A few variants are proposed
- Regular
- “Ref free” variant (using 1 as importance weights)
- "Sequence” variant (see my concern below)
- “KL” variant (replacing importance weights with KL)

Baselines include PPO (on-policy RL), direct preference optimization (DPO; recent non-RL approach), preference ranking optimization (PRO), and GOLD (offline RL / learning from demonstrations). The generation models are based on llama-7b architecture. The reward models are taken from Pythia (trained on Open Assistant; see footnote 8 and surrounding footnotes). The algorithm is tested on four text generation tasks. The main text includes the helpful harmless assistant task (Anthropic’s dataset), and a Reddit response generation task. There’s improvement over PPO and approaching DPO performance on the helpful harmless assistant task. Different tasks used different baselines.

### Strengths
I appreciate the direction of exploring objectives that are not on-policy RL, which is the hot topic these days in fine-tuning LLMs. 

I’m glad the discussion of comparison with GOLD exists (Section 2.4), because the motivation and derivation are extremely similar to GOLD (except for a few differences like the per-token action vs. single action distinction, the different treatment of importance weights, etc., as described in Section 2.4). I think it’s totally fine even if it’s similar to GOLD – there are design differences, a few tricks are used, and more experiments are done. 

Related to the above point: for experiments, I especially appreciate the experiments on the helpful-harmless assistant task.

### Weaknesses
Approximations don’t seem justified mathematically. Maybe it’s alright given RL+NLP research has too many approximations in general – I’ll need to see what other reviewers think. 
- For the “ref free” variant: Is it justified to use 1 as importance weights, in the reference free variant of the algorithm? I can’t wrap my head around whether that’s an acceptable approximation, or whether that's making the derived Equation (3) or Equation (4) simply incorrect. 
- For the “sequence” variant: The approximation of importance rule is a bit strange. See line 6 of the “variants with alternative importance weight” paragraph on page 4. it’s essentially saying a1 * a2 * … * aT * (b1 + b2 + … + bT) = a1 * b1 + a2 * b2 + … + aT * bT. But this seems wrong? Am I understanding this correctly? Perhaps an explanation of why this is a good approximation will be helpful. But at the same time, the empirical results aren’t really impacted much, so I’m conflicted on how much I should treat this approximation seriously. 

A major issue: did the authors train PPO methods for more training steps (more than 1.3 times the training steps of offline methods)? If for more training steps, PPO results improve but your results stay stable, then we can’t say PPO is worse. 

Phrasing of the main question in paragraph 1 – the main question seems to be the italicized sentence at the end of the first paragraph: “can we perform rewarded learning, similar to PPO, while only using pre-existing data” but the answer is already yes given the literature in the past few years. 
- Cringe loss (https://aclanthology.org/2023.acl-long.493.pdf), as well as the older director loss (https://arxiv.org/pdf/2206.07694.pdf) and unlikelihood loss are relevant. The other algorithms the authors cited are also examples where we can learn from rewards while only using pre-existing data. I think the authors’ research question can be more specific & take prior work into account. 
- In addition, I’m also confused about what “similar to PPO” means: do the authors mean that PPO is a form of “rewarded learning” or do the authors mean “can we perform rewarded learning such that the results on X benchmark is similar to PPO performance?”

No on-policy RL performance on Reddit generation task. Is PPO helpful here (given that it’s so popular)?

### Questions
Can you elaborate what leftover lunch means? 

What amount of training trajectories have importance weights that are larger than 1+epsilon or smaller than 1-epsilon (given the bounds on page 4 in “clipping importance weight”)?

For the HHA task, the authors say that they filtered out 20K out of ~160K of training data with responses that abruptly end in a colon. Can the authors explain why this is helpful/necessary, or give an example? 

What kind of tasks would this method not make a difference or fail on? Or does this method work on any text generation task?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a better off-policy policy-gradient based algorithm called Advantage Leftover Lunch RL (A-LoL) in the context of Reinforcement Learning from Human Feedback (RLHF). The proposed A-Lol significantly simplifies PPO by using a fixed advantage function and treating each response as a single action. Experiments are carried out mostly in the HHA benchmark and the reddit response generation task, showing the effectiveness and stability of A-lol.

### Strengths
The single-action assumption made here is very reasonable in the RLHF setting because the transition kernel in natural language generation is deterministic and trivial, such that the standard RLHF is in effect a contextual bandit problem instead of an RL problem. RL algorithms like PPO are unnecessarily complicated in the standard RLHF setting, so it's nice to see a more stable contextual bandit problem algorithm. The proposed method is different enough from other alternatives.

In the common HHA benchmark, A-LOL beats other recent preference-based offline RL baselines such as DPO and PRO and other common baselines such as weighted behavior cloning and PPO. Experiments on reddit generation task also shows the advantage of the proposed method and its flexibility in terms of optimizing for versatile rewards.

### Weaknesses
A-LoL does not seem to perform better than DPO on the HHA benchmark with the common reward function. In particular, A-LoL seems to be less "Helpful" compared to DPO. Is there any explanation on that? The paper seems to be suggesting that there is an issue of reward hacking with the common reward function, is there a concrete example supporting this claim?

In the offline setting where all the data comes from existing offline datasets, the best that we can do seems only to be as good as the best trajectories in the offline datasets. Is it possible to modify A-LoL such that it can continue to improve itself with online data generated by itself and labeled by the reward model?


minor - The single-action assumption might not always hold especially when the dialogue involves multi-step interaction with the users.

### Questions
Why is this method called advantage leftover lunch RL?

I'm curious if other contextual bandit algorithms (such as those from https://arxiv.org/abs/1802.04064) can work well too in the RLHF setting?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Large language models have shown performance improvement due to training them with reinforcement learning (RL) from human feedback. But unlike other paradigms of learning, RL is very data-inefficient. The authors propose to address this issue by introducing a class of algorithms called Advantage-Leftover Lunch RL (A-LOL). The idea of these algorithms is to utilize better the SFT and the underlying data that the RL algorithm uses. In particular, A-LOL algorithms are offline algorithms that do not require any online samples, making their algorithm more sample-efficient than RL.

### Strengths
Strengths:
- The motivation of the paper and the technical contributions are clear.
- The authors perform a thorough empirical investigation of their proposed technique across several tasks. The authors conducted various ablation experiments of the proposed idea to show why the algorithm performed well (e.g., R-LOL versus A-LOL).
- The author studies an important question of comparing online and offline policy gradient algorithms.
- The authors also discuss an interesting issue around a subset of RLHF techniques needing human preference data, whereas other techniques do not.

### Weaknesses
Weaknesses:
- Given that in language, the transition function is trivial - it is unclear why offline algorithms are more sample-efficient than online algorithms. Offline algorithms assume access to a lot of quality data, while online algorithms can work with small amounts of data and well-designed reward functions.
- The paper relies on the assumption that each token is not an action but instead a sequence is an action, but it is unclear why this assumption matters. Most RLHF techniques optimize policies on sequence-level losses, not token-level losses because the reward functions are defined on a sequence.
- For the A-LOL algorithms to work, there is a set of assumptions that are not explicitly mentioned in the paper that could have a big impact on performance. In particular, if you don't have good data coverage and a good initial policy, then A-LOL will fail, which means that A-LOL and PPO in RLHF have the same assumptions.
- The experiments did not include PPO due to a seed collapsing, but there has been evidence in the literature that this does not happen, which means the authors did not tune this baseline algorithm properly [1]. 
- The authors claim that their proposed approach is more data-efficient because they filtered out 33% of good responses, but the same procedure can be done for other techniques. However, a similar procedure was not conducted for the baseline algorithms to show that with less data, their proposed approach performs better.

[1] PAIRWISE PROXIMAL POLICY OPTIMIZATION: HARNESSING RELATIVE FEEDBACK FOR LLM ALIGN- MENT  https://arxiv.org/pdf/2310.00212.pdf

### Questions
- There seems to be a typo in equation (2) $\mathbb{E}_{\boldsymbol{x} \sim d^{\pi^{ref}}, \boldsymbol{y} \sim \pi\_\theta} [R(\boldsymbol{x}, \boldsymbol{y}, \star)]$ because $y \sim \pi\_\theta$ in the expectation.
- What is the difference between a single action step and trajectory-base RL? Most RLHF algorithms assume we are performing trajectory-based RL and not token-based RL. The reward function is only defined on the trajectory, not the token level.
- Why is PPO, an online policy algorithm, only able to represent action on a token level, but all the offline policy algorithms can represent action on a sequence level?
- Why can you not optimize the PPO  objective with multiple rewards? If so, then how does PPO perform?
- Are you saying that equation (3) and equation (4) are equal? 
- In equation (3) are you ignoring the derivative concerning $\pi\_\theta$ in the ratio ($ \nabla_\theta \frac{\pi\_\theta}{\pi\_{ref}}$)?
- Why are the inputs $D_x$ in $D_{tr}$ satisfying this $D_x \subset d^{\pi^{ref}}$? If you are assuming that $\boldsymbol{x}$ is indepdnent of $\pi_{ref}$ then that seems to mean that $D_x = d^{\pi^{ref}}$.
- What is M, h and $A(\pi\_{ref})$ in algorithm 1 line 1?
- Did you run experiments with R-LOL with the advantage of the learner policy instead of the reward?
- Why can't you sum $\pi\_{ref}$ log probabilities and compute the reward for GOLD the baseline? 
- Why is A-LOL more data-efficient? You could also filter the data based on pairs with low rewards for the baseline algorithms and train them. But it is hard to understand if your algorithm is more data-efficient without training the baseline algorithm with the same data. 
- The average length of ppo is very odd compared to other algorithms. Do you have qualitative outputs to share? Did you include the kl-penalty into the objective? 



Missing citations:
- Pairwise Proximal Policy Optimization: Harnessing Relative Feedback For LLM Alignment Wu et al. 2023
- Learning to Generate Better Than Your LLM by Chang et al. 2023
- Calibrating Sequence Likelihood Improves Conditional Language Generation by Zhao et al. 2023

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
