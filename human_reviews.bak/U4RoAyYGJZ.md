# On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling

- Decision: Reject
- Scores: 5, 5, 8, 8

## Abstract
On-policy reinforcement learning RL algorithms perform policy updates using i.i.d. trajectories collected by the current policy. 
However, after observing only a finite number of trajectories, on-policy sampling may produce data that fails to match the expected on-policy data distribution. 
This sampling error leads to noisy updates and data inefficient on-policy learning.
Recent work in the policy evaluation setting has shown that non-i.i.d., off-policy sampling can produce data with lower sampling error than on-policy sampling can produce~\citep{zhong2022robust}.
Motivated by this observation, we introduce an adaptive, off-policy sampling method to improve the data efficiency of on-policy policy gradient algorithms.
Our method, Proximal Robust On-Policy Sampling (PROPS) reduces sampling error by collecting data with a \textit{behavior policy} that increases the probability of sampling actions that are under-sampled with respect to the current policy. 
Rather than discarding data from old policies -- as is commonly done in on-policy algorithms -- PROPS uses data collection to adjust the distribution of previously collected data to be approximately on-policy. 
We empirically evaluate PROPS on both continuous-action MuJoCo benchmark tasks as well discrete-action tasks
and demonstrate that (1) PROPS decreases sampling error throughout training and (2) improves the data efficiency of on-policy policy gradient algorithms.
Our work improves the RL community’s understanding of a nuance in the on-policy vs off-policy dichotomy: on-policy learning requires on-policy data, not on-policy sampling.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new algorithm, PROPS, which seeks to entail more efficient off-policy learning by drawing inspirations from on-policy learning algorithms, i.e., making the update data distribution stay close to the learner policy (on-policy). The paper provides a new algorithm, with experimental results that show improvements over baselines such as PPO.

### Strengths
The paper is interesting in that it tries to improve off-policy learning by drawing inspirations from on-policy learning algorithms, and the intuition that one should try to make the data to be as on-policy as possible. The paper hinges on an observation that for the learning to be on-policy, it suffices for the data distribution to be on-policy, and one does not always have to use the same policy for the update.

### Weaknesses
The idea of using near on-policy update or constrained update to improve off-policy learning has been extensively studied in the RL literature, and as a result, the constrained update rule proposed in the paper is not very novel.  Though the idea of using off-policy distribution that matches on-policy update is interesting, it does not appear enough to constitute a technically solid algorithmic contribution. Indeed, the main idea presented in the paper is not fundamentally different from the trust region policy updates, and does not present additional insight beyond what's already in the existing literature.

### Questions
=== **PPO objective and CLIP objective** ===

The PPO objective appears similar as the CLIP objective in Eqn 4. It is important to note that PPO takes $\theta$ and $\theta_{\text{old}}$ to be close by policy (in practice one iteration away from each other). In the limit when these two policies are very close, the update approximates the policy gradient update. In light of this, in order for the update to be sensible for CLIP, we also need $\theta$ and $\phi$ to be close to one another. In practice this would be softly enforced by the KL loss. Does that sound right? What would be the major difference between CLIP and PPO objectives, despite the technical differences on the clipping operation and KL regularizations.

=== **CLIP objective** ===

In Eqn 4 and 5 we specify a loss for both $\theta$ and $\phi$, do we update them both at the same time? Or is one updated at a slower rate compared to another.

If $\theta$ and $\phi$ are practically close to each other, then the CLIP objective is effectively very similar to the idea of trust region policy updates.

=== **On-policy and off-policy** ===

I like the comment that on-policy learning should really be to match the learning distribution, not the exact policy. However, it might not pay off much to read too much into the terminology itself -- I think much of the community would agree that the naive implementation of on-policy learning would be to learn from the same policy, whereas on-policy samples are all that's needed to perform the update. In practice since the policy keeps drifting away, it is very difficult to maintain even on-policy samples or sub-sample those from the replay buffer.

I think the paper has a very nice starting point to identify this conceptual insight, but didn't dive deep enough to address the issue. The CLIP objective is not fundamentally different from trust region policy updates, which have been in the literature since the beginning of deep RL.

=== **Comparison in Fig  5** ===

What's PPO-buffer exactly? Does it correspond to running PPO on off-policy samples naively? I'd feel that PROMPS is a slightly more adaptive variant of PPO-buffer, so that it can make use of buffer data as well. But I am surprised about the empirical gains, since PPO-buffer, though using off-policy data, still applies clipping and hence enforces a trust region constraint during the update. This means the policy update still maintains its stability over time, even when using very off-policy samples.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work focuses on sampling error problem of on-policy RL algorithms. Following the previous work ROS, this paper introduces the understanding: on-policy learning requires on-policy data, not on-policy sampling. This paper proposes a new method called Proximal Robust On-Policy Sampling (PROPS), aiming at adaptively correcting sampling error in previously collected data by increasing the probability of sampling actions that are under-sampled with respect to the current policy. Two more mechanisms are utilized to address the issues of destructive large policy updates and gaussian policy updates. The proposed method is evaluated in MuJoCo continuous control tasks and three discrete-action tasks, against PPO and PPO-Buffer, ROS.

### Strengths
- The paper is almost clear and overall well-written.
- Intuitive motivating examples are used which help smooth reading.
- The experiments are conducted with both learning performance comparison and sampling error analysis.

### Weaknesses
My main concern lies at the mismatch between *the idea* mentioned multiple times in the first half of the paper (i.e., adaptively corrects sampling error in previously collected data by increasing the probability of sampling actions that are under-sampled with respect to the current policy) and *the practical method* proposed later:

- The idea is to adapt the behavior policy so that the resultant data distribution can be more on-policy, as illustrated in Figure 1.
- However, the practical method is to adapt the behavior policy to fit the collected data (with the gradient $\nabla_{\phi}L=-\nabla_{\phi} \sum_{(s,a) \in D} \log \pi_{\phi}(a|s)$ and its improvement).

&nbsp;

For some specific point, the second paragraph of Section 5 says, ‘To ensure … updates to $\pi_{\phi}$ should attempt to increase the probability of actions which are currently under-sampled with respect to $\pi_{\theta}$… the gradient $\nabla_{\phi}L=-\nabla_{\phi} \sum_{(s,a) \in D} \log \pi_{\phi}(a|s)$ provides a direction to change $\phi$ such that under-sampled actions have their probabilities increased’. I do not get the point. Since the expectation is over the collected data, the gradient seems to be an MLE based on the collected data (or imitation learning) and it is almost the same as Equation 7 in the appendix (with a difference in max and min).

Please correct me if I misunderstand it.

&nbsp;

Besides, I feel that the example and discussion used in Section 4 are improper. The main idea of the paper and the illustrative example in Figure 1 focus on the distribution mismatch of *action*; while the discussion in Section 4 is mainly about the distribution mismatch of *outcome*, which cannot be known in practical learning process (i.e., we have no idea about the existence of a large reward). This turns to be a little bit off the track to me.

&nbsp;

The empirical results are not convincing enough to me, especially under the concerns I mentioned above. 

Important baselines ROS and GePPO are not included in Figure 5. Computational cost like wall-clock time is not discussed in the main body of the paper. Besides, the ablation on the KL term in learning performance is expected.

### Questions
1) Since there could be under-sampled actions, there should be also over-sampled actions. How are they considered in the proposed method? 

2) Why is ROS not included in learning performance comparison like Figure 5? In addition, I think it is necessary to include GePPO as a baseline, since the proposed method aims at improving learning efficiency of on-policy RL.

3) How is the computation cost of PROPS, especially the wall-clock time?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new adaptive sampling approach to reducing sampling error in off-policy policy optimization. Specifically, the paper proposes to collect data such that the data distribution is more on-policy to the target policy. They do this by training a separate behavior policy according to a objective function which encourages the behavior to take actions which correct the sampling distribution of the replay buffer. They show some empirical improvements in standard benchmarks compared to PPO.

### Strengths
Overall, I think the proposed method is quite reasonable and an interesting approach to correcting for the use of off-policy data in policy optimization. I see two weaknesses in the paper as currently presented.

### Weaknesses
I am updating my score based on conversations w/ the authors. Specifically, I appreciate the re-focus and the additions for clarity.


--- before edits ----

Overall, I think the proposed method is quite reasonable and an interesting approach to correcting for the use of off-policy data in policy optimization. I see two weaknesses in the paper as currently presented.

## Characterization of on-policy updates (i.e. data or sampling)

I believe the paper presents an inaccurate characterization of the history of on-policy vs off-policy algorithms, and this detracts from the overall presentation of the paper. The core of my issue is the characterization that the field is uncertain if on-policy updates require on-policy data or on-policy sampling. 

To understand my complaint, I want to begin with policy evaluation and think about how these principle ideas transitioned into generalized policy iteration and concepts surrounding the usual suspects of on-policy learning (i.e. those using “on-policy sampling”). If we look at the fundamentals of policy evaluation in dynamic programming we see a very clear picture of the ideal update (i.e. using the probabilities of the state distribution and policies directly), but at the cost of knowing the transition probabilities and having to sweep over all states and actions for a single update. When improving the policy we go between a policy improvement step (i.e. maximize the learned value function) and the policy evaluation step. Generalized policy iteration made this update more general by enabling the ability to improve a policy without running policy evaluation for every state and action. 

From generalized policy iteration we can get to many of our on-policy algorithms such as TD, sarsa, on-policy actor-critic algorithms, and many others with the inclusion of sampling from distributions instead of knowing the distributions ahead of time. This type of on-policy learning uses a transition and throws it away, meaning any policy improvement done will always appear in the data _and_ there is no stale buffer of data. From this trajectory of literature it is clear, on-policy algorithms are designed and work well with data that is distributed according to the target policy. The easiest way to get this data is through sampling according to the behavior.

With the inclusion of replay buffers, even on-policy algorithms are likely to have off-policy drifting in the data used to train them. We see these problems accumulate in bad behavior (see Deadly Triad) and hacks (e.g. Target Network) designed to navigate this problem to try and get more out of the data an agent experiences. 

The paper correctly notice a difference in the sampling error when the data isn’t infinitely large, where the adaptive sampling method presented here is perfectly suited to fill this gap and try and correct for the discrepancies in the distribution of actions (which might influence the distribution of states, but I’m unclear how this influence would evolve). 

To wrap this up, I don’t have a complaint about the method, and am personally very excited about more active techniques of the agent modifying its behavior to get more favorable data distributions. I just think the narrative is misleading, and a dichotomy presented by the authors is ill-conceived and distracting.

## Example in section 4
2. The example used in section 4 is quite confusing in the context of off-policy and on-policy policy optimization. The main confusion from my perspective is the confusion between a trajectory and an action, which propagates throughout the paper in that we are only correcting the distribution according to the action distribution and no the state distribution.


## Some suggestions and questions about the empirical section:
1. How were the hyperparameters chosen for all the methods?
2. I think the empirical section would benefit from the inclusion of an off-policy baseline to see how much progress is being made by the adaptive sampling method. While you don’t necessarily have to beat this baseline, including SAC could be beneficial to interpreting the results of your method!

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper looks at the problem of using replay buffers with on-policy methods. The replay buffer contains data generated from various (past) policies but on-policy methods (in practice) rely on data being on-policy.

Building on [Zhong et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/f2dbede0879b9d04ceb30f1b8b476b27-Paper-Conference.pdf), the paper explores adapting the behavior policy (the policy generating the data) in such a way that the replay, as a whole, looks on-policy with respect to the target policy (the policy being learned). (In this sense, the problem being studied can be seen as using on-policy methods for off-policy learning.)

While Zhong et al., 2022, considered the problem of policy evaluation, this paper considers the control problem. The proposed method, PROPS, works as follows:
train a policy to generate "diverse" data, with data from the replay. This policy solves an RL problem where the reward is -1 for taking the observed action and zero otherwise.
generate data with the policy from step 1.
train the target policy with data from the replay.

The policy in (1) is trained not to be too different from the target policy (the policy being trained in 3), but the the regime in (1) is slightly off-policy. The regime in (3) is also slightly off-policy. So the paper proposes to use PPO for both steps (1) and (3) as a way to mitigate the slight off-policiness.

The paper shows that the data generated by PROPS is more on-policy than the data generated by PPO. Moreover, PROPS outperforms PPO and the "ad-hoc" PPO with a replay in a control benchmark. Importantly, PROPS works well with the replay whereas the ad-hoc PPO with a replay works worse than PPO in some of the tasks and better in others.

Overall I am happy with this paper, but I am divided in my assessment of its significance. On the one hand, it is a good study of the problem of making on-policy methods work with replays (and thus become more sample efficient). On the other hand, it is unclear to me whether we should use PROPS, or other adapted on-policy methods instead of an off-policy method when learning with a replay (e.g., MPO).

### Strengths
The paper is well executed, and the method is described in enough detail that I feel confident I could reproduce it.

### Weaknesses
I don't think the paper has any serious flaws, but there are some areas it can be improved:
* the presentation can be improved to emphasize some points about the paper and the main contributions
* the significance of the results should be discussed in more detail

I note in passing that the paper is not specifically concerned with representation learning.

### Questions
### Comments

Thank you for your submission.

I think the introduction can be clarified a bit more for the distinction between on-policy sampling and on-policy data. My guess is that the focus on the distinction is not as helpful as getting to the point that on-policy methods need on-policy data for the updates, and the data collection should take into account what the data distribution in the replay will look like.  In the paper you show that if the method samples data on-policy the replay ends up looking quite off-policy.

I would even consider using PPO-buffer as the example to motivate the issue. "Consider the following method. Repeat: Add data to a replay with latest policy, update the latest policy with PPO updates. This method clearly samples the data on-policy, because the behavior and target policy are the same before the policy updates are applied. However, the data in the replay may not be on-policy, because the replay buffer collects data generated by various past policies. Therefore the example method may fail at policy improvement because PPO updates rely on near on-policy data in order to work as intended." A diagram of a replay buffer with "chunks" of data from various policies might also help quickly understand the issue.

Along these lines, I think the point in the beginning of section 4 would be easier to get across if you used an example with a replay buffer where the data comes from multiple policies. I think trying to argue that a sample from a single policy may look off-policy is trickier, and in practice it's not so much of a problem (PPO works just fine with on-policy sampling to generate the data for each update).

I partially disagree with paragraph 5 of section 4. I don't think off-policy corrections should be dismissed as "may increase variance". Some off-policy corrections that clip importance weights mitigate gradient variance in exchange for some bias, for example, PPO. I would say that the real problem is that, while off-policy corrections can mitigate off-policiness, methods that use them typically still suffer as data gets more off-policy. So they may still benefit from different sampling strategies for collecting experience, even with off-policy corrections in place.

I think there are typos in Eq. 4. I think the minus sign should be outside of clip.

In my opinion the KL penalty in Eq. 5 and the KL rule in line 9 of Alg. 2 are under-investigated. In my experience the KL regularizer in Eq. 5 contributes to make collected data more on-policy, so I think it would be helpful to understand how much the KLs are contributing to the replay buffer being on-policy.

We can also see the impact of the KL in Figure 3. The KL regularization actually increases the on-policiness of the data. In contrast, even though ROS was meant (by motivation) to also increase the on-policiness of the data, it does not. It is also worth noting that making the data more on-policy might also mean that the clipping and regularization are not letting the updated target policy deviate too much from the pre-update target policy. So maybe with regularization even on-policy sampling might generate more on-policy data.

Another point on ROS: I find the idea of the ROS loss to generate more diverse data a bit delicate to work with, in the sense that you want a policy that generates a bit less of the actions seen, but not completely optimizing the -1 reward. If you end up rewarding the agent with -1 all the time, the learning dynamics might take you to a uniform policy, but for the problem itself any policy is an optimal policy.

I suggest placing Figure 5 earlier, close to where it is discussed.

Please consider sorting the references alphabetically instead of by citation order, as it will be easier to refer to.

### Questions

What is the contribution of the paper in a broader context? The contributions are clear to me for the problem of on-policy methods with a replay, but what about the broader problem of increasing sample-efficiency by using a replay? Do you think adjusted on-policy methods with a replay are a suitable alternative to off-policy methods?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
