# Reward Adaptation Via Q-Manipulation

- Decision: Reject
- Scores: 5, 6, 3

## Abstract
In this paper, we propose a new solution to reward adaptation (RA), the problem where the learning agent adapts to a target reward function based on one or multiple existing behaviors learned a priori under the same domain dynamics but different reward functions. 
RA has many applications, such as adapting an autonomous driving agent that can already operate either fast  (if transporting goods) or comfortable  (if carrying passengers) to operating both fast and comfortable (if transporting goods with human passengers onboard). Learning the target behavior from scratch is possible but often inefficient given the available source behaviors. Our work represents a new approach to RA
via the manipulation of Q-functions.  Assuming that the target reward function is a known function of the source reward functions, our approach to RA  computes bounds of the Q function. We introduce an iterative process to tighten the bounds, similar to value iteration. This enables action pruning in the target domain before learning even starts. We refer to such a method as Q-Manipulation (Q-M). We formally prove that our pruning strategy does not affect the optimality of the returned policy while empirically show that it improves the sample complexity. Comparison with baselines is performed in a variety of synthetic and simulation domains to demonstrate its effectiveness and generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers the reward adaptation in reinforcement learning. The task is to learn optimal policy for target reward function given behavior data under source reward. The method is developed based on the assumption that the target reward is a known function of the source reward to derive the upper and lower bound of the Q function to perform action pruning. The proposed method is compared with successor feature based reward adaptation method in several domains.

### Strengths
+ Theoretical analysis on the Q bound

### Weaknesses
+ Problem Definition: The problem the authors aim to address is not clearly described. In Definition 1, reward adaptation is defined as the task of learning an optimal policy for a target reward function, given a set of behaviors trained under source reward functions. However, it is unclear whether the agent still has access to the underlying MDP when learning the target policy. Is the proposed method an offline algorithm where the target policy is learned only based on the source data?

+ Assumption. The assumption that the target reward function is a known function of the source rewards (e.g., a linear combination) is not well-motivated and seems overly restrictive. Additionally, the experimental domains are not well-described, making it difficult for readers to interpret this assumption. For instance, in Race Track, $R_3$ assigns a positive reward (+3) for remaining at the initial location.. This is a relatively large reward compared to others in $R_1$ and $R_2$. In the target domain when $\mathcal{R}=R_1+R_2+R_3$, factors like the distance between the goal and initial location, maximum episode length, and discount factor significantly influence policy behavior. In extreme cases, such as with a short maximum episode length and low discount factor, the optimal policy could be to remain at the initial location.

+ Plots. For the Q-M method, pre-training and pre-computation of Q-functions under the source reward are required. However, in the visualizations, it appears that the computational costs associated with this stage are not included in the plots. This omission raises concerns about the fairness of the comparisons.

### Questions
When the target reward is a linear combination of the source rewards, this setup appears quite similar to multi-objective reinforcement learning (MORL), where each objective is summed linearly. Could you clarify if there is any relationship between your method and multi-objective RL?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes Q-Manipulation, a method for reward adaptation in RL which manipulates a Q-function to allow agents to adapt to new reward functions using prior learned behaviours. Upper and lower bounds for Q* for the new task can be determined when the combination function (the transform between the existing reward functions and the new one), which can be iteratively tightened and are guaranteed to reach a fixed point, similar to value iteration.

### Strengths
Overall, the strongest feature of this work is the theoretical guarantees provided on the convergence to a fixed point using the iterative method. This shows that the method is mathematically sound and motivates the extensions to noisy combination functions and continuous action/state spaces.

Pruning the unnecessary actions has the effect that the efficiency of the algorithm decreases with the stochastic branching factor of the MDP, leading to an interesting analysis of the SBF's contribution to convergence speed. As expected, convergence is instant with an SBF of 1, which is already an improvement over SFQL.

Empirically, Q-M outpaces the baselines in the discrete setting, with the linear combination function results being the most convincing. The robustness of the method to noisy reward functions is also convincing as the knowledge of the target reward function may be considered too restrictive in any practical settings.

The clarity of the work is good and the communications of the authors' claims and results are well-formulated.

### Weaknesses
The robustness of the method depends on the accurate initialisation of the Q bounds, and improperly placed bounds (which may occur if the target reward function is not well understood, for e.g.) might lead to inefficiency in the iterative pruning process. An analysis of the quality of the Q bounds to the final convergence speed in practice would be interesting.

Unfortunately, the good results in discrete environments don't carry over well to the continuous domain. The authors admit that the results for QM-DQN are only 'marginally better' for 2 environments, but even this is debatable for Lunar Lander. This is a crucial question for the future of this work because problem settings with known reward functions and convenient state discretisations may be difficult to find. In addition to the degradation of performance as the SBF increases, the scope of environments where Q-M is a viable solution seems limited. Have the authors considered other discretisation schemes, or perhaps a way to use Q-M in continuous environments without discretisation?

Finally, the lack of baselines makes the method a little hard to evaluate in the wider context of reward adaptation. However, as I was unable to identify any appropriate baselines apart from those in the paper, I cannot hold this against the authors as RA is largely an unexplored field as of yet.

### Questions
In Section 3.4, the following statement,

"In Pong, SF-DQN outperformed both QM-DQN and DQN. This was due to the choice of
source behaviors that are either keeping left or right. The target behavior requires the agent to move
to the left and right to catch the ball, which shares strong similarity with the source behaviors."

is confusing. Why doesn't the similarity between source and target functions also help the Q-M agent? Shouldn't the effects be felt more with QM-DQN than SF-DQN? If there is some insight here, a discussion on why QM-DQN didn't perform well would be useful to future readers as a gauge for determining which kinds of problems would be solved with either method.

### Soundness
4

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
5

### Summary
In the context of tabular RL, the paper addresses the problem of "reward adaptation" - we have multiple policies trained on different source tasks (together with the corresponding optimal Q-values and the Q-values of the worst possible policy for each reward) and we want to produce an optimal policy for a target task whose reward is a known function of the source rewards. 
The paper proposes a new method for this task called "Q-manipulation", which consists of establishing and then improving a lower and upper bound on the optimal Q-value for the target task. Where the lower bound for some actions is higher than the upper bound for other actions, this then allows us to prune the action space in each state, so when subsequently solving for the optimal Q-value on the target task, we have an easier task to solve. 
The paper gives a Bellman iteration algorithm for refining the upper and lower bounds and shows that it converges to a (non-unique) fixed point. They also evaluate on several simple simulated tasks against Q learning without any reward adaptation and one prior method with reward adaptation.

### Strengths
- the writing is fairly clear with only minor typos
- the method seems to be a valid method for solving the task as defined (at least in the case where target reward is a linear function of the source rewards)
- the paper provides a valid theoretical result showing that the upper/lower bounds converge to a fixed point.

### Weaknesses
- The importance of task is not addressed so the paper lacks good motivation. No example of a practical real-world task where this could be useful is given.
- This becomes even more pressing given the assumptions of the proposed method. Firstly, even before the method is applied, it assumes we have access to the Q-values of the worst possible policy for each source task (in addition to the optimal ones). However, such worst-case Q-values are not something we usually naturally have in hand, and in general, obtaining each such worst-case Q-function means solving the whole RL problem as many times as there are source tasks. Instead, we could just solve for the target task and be done.
	- one could argue that this would be worth it in cases where we have a fixed set of source tasks, and we need to repeatedly adapt to different combinations of those, but this is not even mentioned.
- Ok, assume that for some reason, we have the worst-case Q-functions. Then, the proposed method still involves running a form of value iteration to refine the upper and lower bound on the target optimal Q-values. The paper doesn't show that this is cheaper than solving for the target task directly. Furthermore, for the method to make sense, it needs to be cheaper to first solve for this and then also run the subsequent Q-learning.
- The timing results hidden in the appendix show that on most tasks, this is indeed slower than running Q-learning from scratch, defeating the purpose of the method.
- Furthermore, I see hiding these important negative results in the appendix without mentioning them in the main text dishonest. The main article shows training curves with iterations that, for the proposed method, include only iteration spent after the expensive pre-training, which may give a wrong impression of computational savings.
- A fully valid end-to-end method is provided only for target rewards that are linear combinations of the source rewards - a fact that is again somewhat hidden in the text. I would like to see that emphasized earlier as it is an important limitation on the contributions of the paper.
- Corollary 1 is never proven. To prove it, you can just give an example of an MDP with non-unique fixed points, which could also serve as a useful illustration.
- Not enough details are provided for reproducing the experiments. Code is not provided.

### Questions
Questions: 
- What are examples of real-world tasks on which your method could be practically useful? (if not in its current form, then at least aspirationally through future extensions)
- What is the break down of the training time using Q-M between the bound refinement and then the subsequent Q-learning?
- Do you disagree with any of the above outlined weaknesses and if so how?
- Why is the method called Q-manipulation? This doesn't seem particularly descriptive of what the method is doing (bounding the Q-function and pruning the actions) - in that light at least half of methods in RL (-adjacent) literature could be called Q-manipulation.


Minor comments and suggestions:
- The notion of "reachable states" is not properly defined/explained upon first usage. I'd recommend explaining as "reachable states" sometimes also refers to states that are eventually reachable (possibly over multiple steps).
- Before, or at least immediately after equations 5, 7, you should comment on the fact that we don't know Q*, otherwise it's really confusing.
- Table 5 in the appendix: I assume |A_p| is the total number of actions left summed over all states. Maybe it would be worth showing this as a percentage of the original number of actions?

### Soundness
2

### Presentation
3

### Contribution
2
