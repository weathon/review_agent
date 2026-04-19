# Off-Policy Primal-Dual Safe Reinforcement Learning

- Decision: Accept (poster)
- Scores: 5, 6, 8, 8

## Abstract
Primal-dual safe RL methods commonly perform iterations between the primal update of the policy and the dual update of the Lagrange Multiplier. Such a training paradigm is highly susceptible to the error in cumulative cost estimation since this estimation serves as the key bond connecting the primal and dual update processes. We show that this problem causes significant underestimation of cost when using off-policy methods, leading to the failure to satisfy the safety constraint. To address this issue, we propose conservative policy optimization, which learns a policy in a constraint-satisfying area by considering the uncertainty in cost estimation. This improves constraint satisfaction but also potentially hinders reward maximization. We then introduce local policy convexification to help eliminate such suboptimality by gradually reducing the estimation uncertainty. We provide theoretical interpretations of the joint coupling effect of these two ingredients and further verify them by extensive experiments. Results on benchmark tasks show that our method not only achieves an asymptotic performance comparable to state-of-the-art on-policy methods while using much fewer samples, but also significantly reduces constraint violation during training. Our code is available at https://github.com/ZifanWu/CAL.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The study conducted herein delves into the issue of underestimation in safe off-policy RL. In addressing this problem, the study conducts a series of experiments aimed at assessing the efficacy of the proposed methodology. The outcomes of these experiments hold substantial promise for enhancing RL applications within real-world environments.

### Strengths
1. The presentation is clear and straightforward, making it easy to understand.
2. The research conducted in this study is interesting.

### Weaknesses
1. The study could benefit from a more comprehensive exploration of convergence and stability, particularly concerning policy optimization. The theoretical analysis in this regard appears to be somewhat limited.

2. It is suggested that the on-policy experiments be completed, and the results of training for on-policy RL methods be included for a more comprehensive evaluation. The on-policy experiments for this method are not completed.

### Questions
1. Is it possible to extend Equation (5) into a third-order equation, and if so, what impact might this have on performance?
2. What is the connection between the variance in Q-cost value estimation and the process of policy optimization?
3. How does the value of "c" affect performance in terms of both cost minimization and reward maximization?

### Soundness
2 fair

### Presentation
3 good

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
This paper presents primal-dual method to solve constrained RL problem combining the conservative policy optimization and local convex optimization. By conservative policy, the estimated Q-value is calculated via averaging over several copies of Q-value and adds UCB-style bonus term. The local convexification comes from well-posed augmented Lagrangian problem.

### Strengths
The primal-dual approach is known to be difficult to train but combining several technique, the authors show improved performance compared to recent results in safe RL.

### Weaknesses
- I believe there needs to be some clarifications about the proof that ALM does not modify or alter the position of local optima. Regarding equation (20), in general expectation of max or square is not same as the max or square of expectation, i.e., $\mathbb{E}[X]^2\leq \mathbb{E}[X^2]$ and the same for the max operator. Hence, we cannot guarantee the following equation to hold:
$$ |\max\\{ 0,\lambda+c \mathbb{E} [ \hat{Q}^{UCB}_c(s,a) ] \\}  |^2 = \mathbb{E} [| \max\\{ 0,\lambda+c\hat{Q}^{UCB}_c(s,a) \\} |^2].$$

### Questions
- Why do we have to reformulate the augmented Lagrangian? What is the advantage of the reformulation according to Luenberger compared to the original augmented Lagrangian (equation (13))?

- For Figure (5), as for Hopper environment, when $c=0$, the test-cost seems to be not that bad in the sense that average is below the threshold. Does this imply the threshold constraint is too loose?

- Please provide page number or section for Luenberger 1984, about the derivation of augmented Lagrangian, which will be helpful to the readers.


- In section 3.1, about the underestimation, we have $\mathbb{E} [ \hat{Q}_c(s,\pi(s))]=\mathbb{E}[Q_c(s,\pi(s))+\epsilon]=\mathbb{E} [Q_c(s,\pi(s))]$. Why does taking minimum over $\pi$ implies underestimation bias? Please give some clarifications.

- It would be helpful if the authors formally introduce and rephrase the theorem in A.M. Duguid. 


A.M. Duguid. Studies in linear and non-linear programming, by k. j. arrow, l. hurwicz and h. uzawa. stanford university press, 1958. 229 pages. 7.50. Canadian Mathematical Bulletin, 3(3):196–198, 1960. doi: 10.1017/S0008439500025522.

### Soundness
1 poor

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the problem of constrained off-policy reinforcement learning and remedies the issue of inaccurate cost estimation in prior Lagrangian duality-based approaches in two steps: i) the cost value estimate is replaced by an upper-confidence bound, which is derived through the uncertainty, or lack of consensus, in an ensemble of cost value estimators, and ii) the regular Lagrangian is replaced by the augmented version of the Lagrangian in order to make the optimization landscape in the feasible region surrounding the local optimal policy. Essentially, the first component encourages the policy to be conservative and ensures better constraint satisfaction, while the second component encourages the policy to become as aggressive as possible in the space of feasible policies in terms of the primary reward function. Numerous experiments demonstrate the superiority of the proposed method, termed CAL, over multiple baseline methods in a variety of settings and environments.

### Strengths
- Training safe RL policies is a very important research direction, and alongside the aforementioned components of controlled conservativism and aggressiveness, the paper presents a sample-efficient constrained RL training procedure that also has safer behavior during **training**, something that has potentially been overlooked in prior work in this area.

- I very much enjoyed reading the manuscript. The content is very well presented, and the experimental results are, in my opinion, excellent and convincing as to why and how the proposed method works well.

### Weaknesses
One limitation I can think of is the inclusion of only one constraint in the problem formulation, solution, and experiments. The authors have mentioned that having a single constraint does not prevent generality, but I wonder how the method performs in settings with multiple competing constraints, especially as compared to baseline methods. A discussion on how the two introduced components interplay in the presence of several constraints would be helpful.

### Questions
- Could you please provide more details on how the cost value ensemble in (3) is created? Do you have multiple (i.e., $E$) parallel parameterized models that each estimate the cost in parallel? Should the expectation in (3) be replaced by an empirical mean (i.e., $\frac{1}{E} \sum_{i=1}^E$)?
- The last statement in Section 3.1 mentions that the UCB "encourages the policy to choose actions with low uncertainty in cost estimation." Could you comment on why that is the case? Based on (3), it seems that for state-action pairs with low uncertainty in cost estimation, the UCB is close to each of the ensemble estimates, but it does necessarily translate to the action leading to the highest estimated reward in that state.
- I suggest changing the notation for the hyperparameter $c$ in the augmented Lagrangian, as it has also been used to denote the cost function in the CMDP definition.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the problem of safe reinforcement learning in constrained Markov decision processes, where the agent tries to maximize reward while keeping a separate cost signal under a given threshold. The authors propose an off-policy primal-dual deep RL method that combines two techniques. The first, called conservative policy optimization, replaces cost-value estimates with an empirical upper confidence bound based on the variance of the estimates across an ensemble of Q networks. The second, called local policy convexification, applies the augmented Lagrangian method to the primal-dual formulation of constrained RL. While the first is intended to make the policy search more conservative reducing the number of constraint violations due to cost estimation, the second is intended to stabilize the learning process. The algorithm that combines these two techniques, called CAL (Conservative Augmented Lagrangian), is tested on benchmark tasks and on a real advertising system (in a semi-batch fashion) and compared with several baselines, showing a significant improvement over the state of the art both in terms of reward maximization and constraint violation.

### Strengths
The paper considers an important problem, that of safe RL, from a practical perspective.
Its main strength is in the extensive experimental campaign. The experiments are well designed and documented, and convincingly show the advantages of the proposed method over the state of the art. It is particularly impressive to see experiments on a real system, where the proposed method is also compared against several baselines.
The main point of originality is in using the augmented Lagrangian method in this setting.

### Weaknesses
The theoretical motivations are a bit brittle. The authors should be more clear on where the theoretical inspiration ends and where heuristics begin. For instance, in section 3.1, it is suggested that Equation (3) would be an actual upper confidence bound in a linear MDP, which is inaccurate, as the upper confidence bound for linear MDPs has a specific form that does not correspond to equation (3). 
Also, more words should be spent in explaining how the bootstrapped value ensemble is constructed, and why you chose this form of uncertainty estimation.
Regarding ALM, I think that the theory that is provided in the appendix is interesting and deserves more space in the main paper.

### Questions
How many Q networks did you use in the ensemble and how did you choose this hyperparameter?

Just a minor remark: the objective should be stated in terms of the true value functions and not their estimates (equation 1), even if they must be estimated from data in practice.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
