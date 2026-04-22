# Positive Transfer of Prior Knowledge in Deep Reinforcement Learning via Reward Shaping

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Effective learners improve task performance and acquire new skills more efficiently by leveraging related prior knowledge. Reward shaping is central to many such approaches and facilitates knowledge transfer. However, misidentifying or misusing prior knowledge can impair learning. To tackle this challenge, we propose a novel shaping method, Target value As Potential (TAP), which uses critic target value as the potential to operate within the canonical Potential-Based Reward Shaping (PBRS) framework. It integrates readily with policy-gradient deep reinforcement learning algorithms and requires only minor modifications to existing training pipelines. This endows TAP with the unique combination of policy invariance and simplicity in implementation, distinguishing it from many model-based methods. Our qualitative analysis and empirical evaluations demonstrate that TAP accelerates convergence compared to baseline DRL algorithms. Moreover, empirical results show that TAP leads to higher cumulative returns. We evaluate TAP-augmented TD3 and D4PG across a range of tasks in the DeepMind Control Suite. TAP significantly improves performance over the original TD3 and D4PG and consistently outperforms other reward shaping methods, including Heuristic-Guided Reinforcement Learning (HuRL) and Dynamic Potential-Based Reward Shaping (DPBRS).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposed to deal with the knowledge transfer problem through potential based reward shaping with Q function in source domain as the potential. The authors provide some theoretical analysis and experiments on DMC environments.

### Strengths
It is a long-standing problem on how to improve the sample efficiency for agents in real-world application and knowledge transfer considered in this manuscript seems to be one potential solution.

### Weaknesses
* The idea is not novel. At least this idea is quite similar with [1] and I believe there are more references related to the similar idea included in the manuscripts they refer or refer to them.

[1] Zou, Haosheng, et al. "Learning task-distribution reward shaping with meta-learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.

* From the current formulation, this method requires to share the state and action space from the source domain to target domain, which restricts the applicability for more practical scenarios. I'm wondering how to generalize the proposed methods to scenarios without shared states and actions.

* Theoretical analysis are weak and not novel. Policy Invariance can easily be derived from the original reward shaping paper from Ng et al. Convergence rate is of standard textbook style. One tricky part is the authors claimed benefits by assuming we perform Q-iteration from 0 initialization. But in practice this is never the case, and any kinds of positive initialization will be better than 0 initialization. There are no real emphasize on the importance and benefits of knowledge transfer. I would like to see what's the real benefits for the proposed methods.

* For the DMC experiments, there are no clear descriptions about how to generate the cross-task potentials. As not all the DMC environments share the same state action space I'm confused about what's the source task. Also, missing the introduction of cross-task potentials weaken the experimental results.

### Questions
To echo the weakness:

* How to compare the proposed methods with the existing work like [1] mentioned in the weakness part?
* What's the real theoretical benefits compared with other randomly initiated non-zero potential function?
* Can you provide a more clear experimental setup for the DMC experiments, like how the potential function from source domain is obtained, and how to deal with different state and action space?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Target value As Potential (TAP), a novel reward shaping method.
TAP uses the target critic network (action-value function) from actor-critic RL algorithms as prior knowledge to speed up training.
This is shown both in the same-task and cross-task setting for standard control tasks.
A theoretical analysis/discussion of TAP is provided, giving further insights into the expected benefit of using TAP.

### Strengths
- Easy to incorporate into existing methodology, which may facilitate uptake by the community.
- Interesting set of experiments, even if there are not really extensive in terms of number of datasets or baseline methods evaluated.
- Aims for a theoretical analysis to provide further insights in to the method, yet not with any hard guarantees.

### Weaknesses
- I did not check the details of the proofs in detail, so can not comment on their correctness. However, I feel they are a somewhat self-fulfilling prophecy, i.e. assuming the target Q-function is a useful prior the number of samples decreases when using it for reward shaping for learning the correct Q-function. Also, this discussion is missing any argument about whether it is necessary to know the Q-function exactly on the full state-action space or just good enough to allow training a good policy training. So overall my complaint would be that the result is somewhat vacuous and does not provide strong arguments for TAP, yet I would look forward to arguments by the authors why this is not the case.
- While it is ok to defer the proof to the appendix, I would greatly appreciate at least a proof sketch in the main paper to get an intuition why the presented result should hold.
- I do not think that Eq.(8) is correct, the expectation over the policy is missing.
- Am I missing something or does one not win anything through TAP in the experiments presented in Figure 1? What incorporating prior knowledge brings is exactly what was needed beforehand to learn the prior. E.g. using a prior obtained from 250k steps I immediately learn to always succeed after very few gradient steps, if I use a prior after 200k steps I need another 50k to close the gap and always succeed. This is not necessarily bad overall, transfering prior knowledge across tasks might still be possible, but it does not seem to bring any benefit when applied on the same task, the overall computation is roughly the same.
- I do not get the reasoning behind the results in Figure 4. (1) are the embedding spaces shared? T-SNE is non-linear and does not allow new datapoints to be projected into the same space. Also, the 3D representation is rather poor to judge, same for the color information. I would rather use 2D maps and work with alpha values to visualize density. Is PRS-IT in the figure TAP? Also, which TAP variant was used for this results? Also, there should be a comparison to the other reward shaping baselines, maybe rather quantitatively then qualitatively in the current visual way.


Presentation and Grammar should be improved. Some examples:
- There is no Remark 1 in the paper, it starts with Remark 2.
- There is no 2) after 1) in line 371
- Line 101: espiecially
- Notation errors such as line 193 where a subscript is used instead of a superscript
- $p_\pi$ in Eq.(3), (4) and (11) never introduced.

### Questions
- How exactly is TAP-Self implemented, is it just the current target network?
- Is there any practical purpose of TAP-Same? Don't I just learn the same policy again?
- For the results in Figure 2, what prior run was used for TAP-Same? Did this prior run attain the performance that TAP-Same is attaining or worse? If worse, how can it boost results so strongly?
- Why are only TAP-Same and TAP-cross used in the second set of experiments (Figure 3), I would be interested in the performance of the other baseline or a discussion why they are not applicable.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel reward shaping method that injects the prior knowledge from other tasks. By simple modification on reward signal using the critic function, the proposed method can effectively transfer the knowledge. Furthermore, as in the main theorem, it is guaranteed that TAP method can converge to the maximum return with less time. In the experiment, the proposed method can effectively adapt to the task more quickly, and also can  transfer the knowledge to the cross-task.

### Strengths
1. The overall method is simple and easy to implement. And this paper also proposes the theoretical guarantee on the speed of learning the tasks. Furthermore, the performance of TAP method is quite remarkable compared to the baselines.

### Weaknesses
1. What is the main motivation of deriving the TAP method? I know it is quite effective on injecting the prior knowledge, but it is hard to figure out why we should transfer the knowledge in this way. How can we derive this idea without any empirical or theoretical findings?

2. It would be better to show the results on other environments such as Meta-World which consists of multiple skills on robotic manipulation tasks. is the TAP method still effective on robotic manipulation tasks?

### Questions
Already mentioned in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
