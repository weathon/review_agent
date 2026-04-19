# The Effective Horizon Explains Deep RL Performance in Stochastic Environments

- Decision: Accept (spotlight)
- Scores: 6, 6, 6, 5

## Abstract
Reinforcement learning (RL) theory has largely focused on proving minimax sample complexity bounds. These require strategic exploration algorithms that use relatively limited function classes for representing the policy or value function. Our goal is to explain why deep RL algorithms often perform well in practice, despite using random exploration and much more expressive function classes like neural networks. Our work arrives at an explanation by showing that many stochastic MDPs can be solved by performing only a few steps of value iteration on the random policy’s Q function and then acting greedily. When this is true, we find that it is possible to separate the exploration and learning components of RL, making it much easier to analyze. We introduce a new RL algorithm, SQIRL, that iteratively learns a near-optimal policy by exploring randomly to collect rollouts and then performing a limited number of steps of fitted-Q iteration over those roll- outs. We find that any regression algorithm that satisfies basic in-distribution generalization properties can be used in SQIRL to efficiently solve common MDPs. This can explain why deep RL works with complex function approximators like neural networks, since it is empirically established that neural networks generalize well in-distribution. Furthermore, SQIRL explains why random exploration works well in practice, since we show many environments can be solved by effectively estimating the random policy’s Q-function and then applying zero or a few steps of value iteration. We leverage SQIRL to derive instance-dependent sample complexity bounds for RL that are exponential only in an “effective horizon” of lookahead—which is typically much smaller than the full horizon—and on the complexity of the class used for function approximation. Empirically, we also find that SQIRL performance strongly correlates with PPO and DQN performance in a variety of stochastic environments, supporting that our theoretical analysis is predictive of practical performance. Our code and data are available at https://github.com/cassidylaidlaw/effective-horizon.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a new explanation of the success of deep RL algorithms that use random exploration strategies for stochastic environments. The key observation is that many environments can be solved by a few steps of value iteration starting with the uniformly random policy, meaning that sophisticated exploration is unnecessary. Based on the observation, this paper designs a novel provable RL algorithm called SQIRL that runs a few steps of fitted-Q iteration, and the sample complexity of SQIRL is only exponential in (roughly) the number of value iteration steps needed to solve the environment. This paper also shows empirically that the sample complexity of SQIRL correlates with the sample complexity of standard deep RL algorithms such as DQN and PPO.

### Strengths
-	This paper extends the effective horizon definition in Laidlaw et al. (2023) to stochastic environments, and consequently designs a novel provable algorithm SQIRL that depends exponentially only on the effective horizon. This paper also empirically verifies that the effective horizon is small for many realistic environments including Atari games. Hence, the SQIRL algorithm is the first algorithm that (a) has provable sample complexity upper bound on realistic environments, and (b) achieves non-trivial performance empirically.
-	Although the algorithm is very similar to the classic FQI algorithm, the sample complexity upper bound in Theorem 3.6 is novel. In addition, the assumptions on the oracle are mild and potentially can be satisfied by realistic neural networks.  
-	This paper is well-written and easy-to-follow.

### Weaknesses
-	Given that SQIRL has a provable and also computable sample complexity bound, this paper could benefit from more analysis on the comparison between the theoretical and empirical performance of SQIRL. For example, how does the percentage of environments that SQIRL can solve change with k? Is there a strong correlation between the actual sample complexity SQIRL and the sample complexity upper bound in Theorem 3.6?
-	For an unknown environment, computing the effective horizon or the sample complexity bound requires running a few steps of value iteration, which is not much easier than running the algorithm since we must iterate through the entire state space. This makes the instance-dependent complexity upper bound less helpful in determining the hardness of an environment / predicting the performance of deep RL algorithms on new environments. The bound could be more impactful if there is an efficient algorithm to estimate the effective horizon.
-	It is unclear whether the small effective horizon is an artifact of the sticky actions modification to the environment. A 25% chance of sticky action could potentially make long-term planning impossible, hence decreasing the effective horizon. Hence the conclusion of Figure 2 needs further justification.

### Questions
-	How does the performance of SQIRL change with the hyperparameter k? Is the best performance always achieved by choosing the smallest k such that the environment is k-QVI-solvable?
-	Is there a typo in the legend of Figure 2? Currently the figure shows that PPO succeeds less often when k=1.
-	It would be intriguing to see whether the sample complexity of the empirical version of SQIRL is still upper bounded by Theorem 3.6. For example, for all the environments with k=1, does SQIRL always solve the environment after $N^{SQIRL}$ steps?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors aim to provide theoretical explanations, supported by an empirical analysis, of the practical success of Deep-RL algorithms. In recent work, Laidlaw et al., 2023, "Bridging RL Theory and Practice with the Effective Horizon", the authors have shown that deep-RL methods succeed in (deterministic) MDPs, especially in cases in which it is sufficient to take few "greedyfication" steps from the Q-function of the random policy (i.e., the effective horizon). In this work, the authors show that these claims can be extended to stochastic MDPs by introducing the "stochastic effective horizon". More specifically, the paper is articulated in the following points:
1. The authors provide a formal definition of the "stochastic effective horizon" notion that presents itself as the natural extension of the effective horizon introduced by Laidlaw et al., 2023. 
2. Secondly, the authors propose a simple algorithm (SQIRL) whose sample complexity scales exponentially with the stochastic effective horizon (which is typically much smaller than the optimization horizon of the underlying problem).  
3. On a stochastic extension of the benchmark Bridge (Laidlaw et al., 2023), the authors show that modern deep-RL algorithms performance significantly correlates with the empirical success of SQIRL. This empirical phenomenon suggests that the concept of stochastic effective horizon can explain some of the reasons behind the success/failure of modern deep-RL methods.

### Strengths
1. Comprehending the factors contributing to the successes and shortcomings of deep reinforcement learning (Deep-RL) algorithms is of utmost importance. Analyzing the environments in which current methods excel provides insight into both their capabilities and constraints. This understanding serves as a foundation for inspiring and creating innovative approaches that address the limitations of current technologies. In this sense, the work done by the authors goes in this direction, as the problem deserves attention from the community (both practitioners and theoreticians).
2. The paper takes large inspiration from the recent work of Laidlaw et al., 2023, "Bridging RL Theory and Practice with the Effective Horizon". Nevertheless, in Laidlaw et al., 2023, the authors consider a deterministic setting. In this work, instead, the authors propose an extension to the more general (and challenging) stochastic environments. Although, in this sense, the novelty of the underlying idea is incremental, the challenges that are introduced from stochastic environments do not allow for a direct extension. 
3. The main text is overall well-written and easy to understand.

### Weaknesses
**1. Novelty**

First, I remark that the main contribution done by the authors goes into the direction of providing explanations behind the success of deep-RL algorithms. Although these explanations are highly appreciated, it has to be remarked that the main idea on which the work is built has been already proposed in Laidlaw et al., 2023. Indeed, in Laidlaw et al., 2023, the authors have shown that deep-RL succeed in (deterministic) MDPs especially in cases in which it is sufficient to take few "greedyfication" steps from the Q-function of the random policy. In this sense, the novelty, in terms of new explanations that are given to the success of deep-RL algorithms is somehow limited. The contributions of the authors, in this sense, is limited to to the extension to stochastic environments.

**2. Theoretical claims and analysis**

I have concerns regarding the theoretical claims done by the authors, especially regarding the main Theorem (i.e., Theorem 3.6). Specifically, I checked the proofs behind Theorem 3.6, and there are some steps that are unclear/unprecise/uncorrect (p.s., I haven't had a look in details to the other sections of the appendix, so I am unaware if there mistakes in those parts). 

First, I begin with the result on the sample complexity (Eq. 1). Using standard tools from the bandit community, it is possible to show that, for $\epsilon$-best arm identification with 2 arms, the Lower bound is given by $\widetilde{\Omega}[ \max \left( \Delta^{-2}, \epsilon^{-2} \right) ]$ (RL with 1 state, depth 1, and 2 actions). In Eq. 1, instead, the authors claim a complexity of $\widetilde{\mathcal{O}}(\epsilon^{-1})$, which is clearly impossible. It seems to me that the main problem is that the authors masked inside the $\widetilde{\mathcal{O}}$ instant dependent quantities such as $\Delta^{-2}$. 

Secondly, I invite the authors to provide details on the last step behind the proof of Lemma B.1. To me, it seems that they upper-bound $P_\pi(\mathcal{E}) \le \epsilon$, but, then it is unclear how they upper bound the cumulative sum of rewards to $1$.

(minor) some assumptions in Lemma B.1 are not used. For instance, k-solvability seems to be not used within the proof.

Overall, I currently believe that all these issues could be solved, leading to results comparable (or maybe, slightly worse) to the one presented in Theorem 3.6. Nevertheless, the paper, at its current status, seems to be lacking in formal correctness.

**3. Weaknesses of the proposed algorithm (minor)**.

It has to highlighted that the algorithm proposed by the authors needs to be aware of the k parameter of K-QVI-solvability property. This parameter is often unknown in practice. I consider this to be a minor weakness, as the purpose of this work is not to propose an algorithm (with theoretical guarantees) that can be applied in practice, but rather it focuses on explaining why deep RL algorithms succed in practice. 

**4. Minor comments:**
- Colors in Figure 2 seems to be swapped. I currently read the Figure as follows: PPO fails most likely with small values of k (however, I guess the opposite claim should be the correct one).

### Questions
See weakness section above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper extends the previous work GORP which studied the explanation of RL in deterministic environments by proposing and studying a new method under stochastic environments under function approximation. GORP shows that the success of both PPO and DQN can simply be explained by a simple procedure of greedily improving a few steps over a value function of random policy. SQIRL extends the algorithm and analysis for stochastic environments. The results demonstrate that the performance of the proposed method SQIRL correlates with the performance of PPO and DQN in stochastic environments.

### Strengths
1. The paper addresses the key limitation of prior work: GORP which showed that performance of deepRL can be explained by just acting greedily with respect to value function of a random policy in deterministic environments. GORP does not extend to stochastic environments directly and the authors propose one way to extend the algorithm to stochastic environments by using function approximation.
2. The authors theoretically explain the sample complexity of their proposed algorithm in terms of psuedo-dimension of the function class along with a combination of concentration and FQI analysis.  The authors also demonstrate that this sample complexity is closely related to sample complexity of Deep RL algorithms used in practice.
3. The method SQIRL is tested extensively over a set of 150 environments from previous work GORP. The environments are made stochastic by using sticky actions and show that whenever PPO or DQN does well, SQIRL does well 78% of the time too.

### Weaknesses
1. My major concern is around the novelty of the proposed approach:
a. To address stochasticity an open loop trajectory optimization is replaced by FQI. This is not new in my opinion:
Prior works:
[1] Considers a tree search with the FQI procedure along with analysis to account for stochastic environments.
[2] Considers a FQI analysis of H-step lookahead under empirical distributions for a more general setting of learned models (A.3) 
[3] Considers a similar FQI setting with learned models where exploratory policy is the dataset policy
A proper discussion on the novelty of using FQI to replace open-loop trajectory optimization along with a comparison with these prior works is warranted.
2. The strong claim of in-distribution generalization: On page 6 the authors claim that if we can properly regress for approximate-Q then we can also estimate the maximum action properly. The claim seems to be too strong without evidence: With finite data, the Q will almost always have errors. These errors will lead to perhaps suggesting actions OOD and lead to overestimation bias commonly observed in deep RL. This analysis seems to be missing in analysis of FQI for SQIRL.
3. A minor nitpick: Unlike GORP, the whole set of stochastic environments are not explained by SQIRL - I think the title and introduction implying something more stronger that it should?
4. Sticky action is a particular kind of stochasticity - More ways of inducing stochasticity and varying the noise std can make the empirical experiments stronger.
[1]: https://arxiv.org/pdf/2107.01715.pdf
[2]: https://arxiv.org/pdf/2107.01715.pdf
[3]: https://arxiv.org/pdf/2008.05556.pdf

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a setting where exploration is not needed. They formalize the setting in Def 3.1. They also design an algorithm and show that the algorithm is sample-efficient in the setting they consider. They also conduct experiment to validate their idea.

### Strengths
The setting they consider is closely related to the tasks in the application.

### Weaknesses
1. The theory is simple and straightforward. In fact, in the setting they considered, exploration is not needed. 

2. It is unclear whether their algorithm can be adapted to the scenario where exploration is needed.

### Questions
1. See the 'Weakness' section.

2. Figure 2 shows that PPO fails in most tasks with k=1, which contradicts the claim 'Furthermore, these are the environments where deep RL algorithms like PPO are most likely to find an optimal policy' on Page 5. Can you provide an explanation?

3. Apart from the tasks in Bridge, can you verify Def 3.1 for other tasks, including go and robotic?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
