# Learning Constraints from Offline Dataset via Inverse Dual Values Estimation

- Decision: Reject
- Scores: 3, 3, 5

## Abstract
To develop safe control strategies, Inverse Constrained Reinforcement learning (ICRL) infers constraints from expert demonstrations and trains policy models under these constraints. Classical ICRL algorithms typically adopt an online learning diagram that permits boundless exploration in an interactive environment. However, in realistic applications, iteratively collecting experiences from the environment is dangerous and expensive, especially for safe-critical control tasks. To address this challenge, in this work, we present a novel Inverse Dual Values Estimation (IDVE) framework. To enable offline ICRL, IDVE dynamically integrates the conservative estimation inherent in offline RL and the data-driven inference in inverse RL, thereby effectively learning constraints from limited data. Specifically, IDVE derives the dual values functions for both rewards and costs, estimating their values in a bi-level optimization problem based on the offline dataset. 
To derive a practical IDVE algorithm for offline constraint inference, we introduce the method of 1) tacking unknown transitions, 2) scaling to continuous environments, and 3) controlling the degree of constraint regularization.  Under these advancements, empirical studies demonstrate that IDVE outperforms other baselines in terms of accurately recovering the constraints and adapting to high-dimensional environments with diverse reward configurations.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an inverse constraint learning RL problem based on the agreement with expert demonstrations. The authors introduce a strange and potentially problematic way to define the cost, i.e., as the divergence to the expert occupancy measure ($D_f(d\|d^E)-\epsilon$). The paper then constructs a DICE framework and then models this problem as a bi-level optimization problem: at one level, solving an offline RL problem and on the other level, minimizing the divergence of the optimal state-action occupancy measure $d^*$ and the expert occupancy measure $d^E$. To solve this problem, however, the authors propose a practical implementation that completely deviates from the DICE formalism, that uses an in-sample learning offline RL framework like SQL. There are very large gaps between the theoretical derivation and the practical algorithm. Lastly, I think the paper shares quite a lot of similarities with RGM [1] from high-level ideas to problem formulation, but never mentioned RGM in the paper. This actually makes me wonder whether the proposed method is essentially performing reward correction or constraint learning as claimed by the authors. Please see the following strengths and weaknesses for detailed comments.

### Strengths
- Constraint learning in the offline setting is a meaningful problem and worth investigating.
- The paper is easy to read and well organized.
- Relative comprehensive experiments.

### Weaknesses
- The biggest concern I have is the way this paper models cost values. It approximates cost value as $\lambda_d (D_f(d\| d^E)-\epsilon)$. This is very problematic since being sub-optimal does not necessarily mean it is unsafe or a cost violation. Matching with expert occupancy measures is essentially doing some sort of inverse RL on rewards rather than conventional constraint learning.
- The model construction is highly similar to RGM [1], but it is not even mentioned in the paper. RGM considers a reward correction problem using a similar DICE-based framework. Both the proposed IDVE and RGM use an expert dataset and a large unknown/sub-optimal quality dataset as inputs. Both methods model the problem as similar bi-level optimization formulations: at one level minimize $D_f(d^*\| d^E)$ and on the other level solve an offline RL problem. The only difference is that RGM learns a reward correction term and this paper learns a cost value function $V^c$. Given the similar high-level formulation, I suspect the proposed method is essentially doing some sort of reward correction rather than constraint learning.
- There are many designs in this paper that look not very principled. For example, although the problem is formulated in a DICE framework, the authors use techniques from in-sample learning offline RL algorithms to construct the practical algorithm. First of all, the "state-value" $V$ in DICE-based methods is not the common meaning of state-value functions in typical RL problems, they are actually Lagrangian multipliers. If the authors are familiar with the DICE-class of algorithms, they will notice that the $V$ values in DICE algorithms take very different values and in many cases behave differently from those in common RL algorithms. Hence simply learning value functions using in-sample learning algorithms like IQL or SQL and then plugging them back into a DICE formulation is inappropriate. Also, there are lots of tricks and somewhat arbitrary designs in the practical algorithm, like introducing $\lambda$ in Eq.(12), and the treatment in Eq.(15)-(17). By contrast, RGM offers a much cleaner and more rigorous derivation from theory to practical algorithm.

### Questions
- Can you justify the difference between the proposed constraint inference as compared to the reward correction in RGM[1]?
- Why not consider learning $V^r$ and $V^c$ using typical DICE-based techniques, like RGM[1] or SMODICE[2]?

**References:**

[1] Li, J., et al. Mind the Gap: Offline Policy Optimization for Imperfect Rewards. ICLR 2023.

[2] Ma, Y., et al. Versatile offline imitation from observations and examples via regularized state-occupancy matching. ICML 2022.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the inverse constrained reinforcement learning (ICRL) problem. Previous works in ICRL primarily consider the online setting which allows online interactions with the environment. However, in safety-critical tasks, it is often dangerous to iteratively collect samples in the environment since the data-collecting policies may violate the constraints. To address this issue, this paper focuses on the offline ICRL and proposes an Inverse Dual Values Estimation (IDVE) framework. IDVE derives the dual values functions for both rewards and costs, estimating their values in a bi-level optimization problem. To implement IDVE, this paper introduces several techniques: 1) handling unknown transitions, 2) scaling to continuous environments, and 3) controlling the degree of sparsity regularization. The empirical results show that IDVE can accurately recover the constraints and achieve high returns.

### Strengths
1. This paper identifies the safety issues in online ICRL. To address this issue, this paper introduces the offline ICRL problem which is more practical.
2. The paper is well-written and easy to follow, providing clear explanations and detailed descriptions of the proposed method and experimental results.

### Weaknesses
The derivation of the proposed framework IDEV and its practical implementation introduces many unreasonable transformations, which makes the algorithm lack soundness.

1. In the first paragraph of Section 4.2, the authors approximate the expert regularizer $\lambda_d (D_f (d||d^E) - \epsilon)$ with $\mathbb{E}_d [\delta^c_V (s, a)]$. However, $\mathbb{E}_d [\delta^c_V (s, a)]$ is not a reasonable approximation of $\lambda_d (D_f (d||d^E) - \epsilon)$. There is even no connection between $\mathbb{E}_d [\delta^c_V (s, a)]$ and $\lambda_d (D_f (d||d^E) - \epsilon)$: the former is a temporal difference term w.r.t the cost function while the latter is a divergence between two distributions.
2. The introduction of the lower-level optimization in Eq.(11) is weird. The authors have replaced the expert regularizer with another term. However, they again add this regularizer in the bi-level optimization.
3. In the last line of Section 5.1, they introduce the approximation $Q^c(s, a)=\mathbb{E}_\{\left(s, a, s^{\prime}\right) \in \mathcal{D}^o}\left[\gamma V\_{\theta^c}^c\left(s^{\prime}\right)\right]$. This approximation is incorrect. The correct one is $Q^c(s, a)= c(s, a) + \gamma \mathbb{E}\_{s^\prime \sim p (\cdot|s, a)} [V^c\_{\theta^c} (s^\prime)]$.




Besides, the experiment setup is a little weird. In particular, the offline dataset in grid-world is collected by random policies, and the offline dataset in MuJoCo is collected by SAC policies. Such offline datasets may contain a large number of unsafe behaviors, contradicting the motivation of this paper. Thus, a more proper choice is to apply safe and conservative policies to collect the offline dataset.

### Questions
1. What is the meaning of the threshold $\hat{\epsilon}$ in Definition 1?
2. Typos:
    1. In the last paragraph of Section 3, “The demonstration dataset $\mathcal{D}_{O}$” should be “The demonstration dataset $\mathcal{D}^{O}$”.
    2. The notation of $\sum_{p_{\mathcal{T}}\left(s^{\prime} \mid s, a\right)} \gamma V^r\left(s^{\prime}\right)$ is confusing. The correct one should be $\sum_{s^\prime \in \mathcal{S}} p_{\mathcal{T}}\left(s^{\prime} \mid s, a\right) \gamma V^r\left(s^{\prime}\right)$ or $\mathbb{E}\_{s^\prime \sim p\_{\mathcal{T}}\left(\cdot \mid s, a\right) }  [\gamma V^r\left(s^{\prime}\right)]$.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the offline safe RL scenario where rewards are observable but costs are hidden. The objective is to derive the cost function from the dataset, akin to the IRL setup. The authors introduce IDVE, built on the DICE family's dual formulation of offline RL. It approximates the cost value function by measuring deviations from expert demonstrations since the cost signal isn't directly observed.

### Strengths
- The proposed setting appears to be realistic to me.

- The overall algorithmic designs make sense, although I do have concerns with a few choices.

- Section 6.1 effectively visualizes the recovered constraints and the impact of various components.

### Weaknesses
- Related works: 
Although IDVE is closely linked to the DICE family, formulated with the distribution correction estimation $w(s, a)$, there is no discussion and referencing to the DICE string of works [e.g., 1, 2]. Inclusion of OptiDICE [3] into discussion is also recommended as it also uses a closed-form solution for the inner maximization, in the DICE framework.

- Experiments: 

    - While considering no costs, the comparison between IDVE w/oA and offline RL is somewhat questionable. In scenarios like limited arm halfcheetah and blocked halfcheetah, IDVE w/oA shows notably superior cumulative rewards. The inferiority of offline RL suggests potential issues with the baseline's strength or its implementation. Given that the offline RL's objective solely maximizes returns, one would anticipate it to at least match, if not surpass, IVE w/oA in terms of returns.

    - The annotation for the dashed line in Figure 6 appears to be missing. I would also recommend the authors to plot average cumulative rewards/costs for both $D^E$ and $D^{\neg E}$. It will be helpful for the audiences to better understand the numbers in Table 2. 

    - (Continued) The gap in returns between expert and sub-optimal demonstrations, as well as between offline IL and expert demonstrations, is unclear. This ambiguity arises because the environments have been modified, eliminating the availability of standardized D4RL scores for comparison. Therefore, plotting rewards/costs of both $D^E$ and $D^{\neg E}$ would be helpful to improve clarity.


[1] Kostrikov, I., Nachum, O., and Tompson, J. (2019). Imitation learning via off-policy distribution matching. arXiv preprint arXiv:1912.05032.

[2] Nachum, O., Dai, B., Kostrikov, I., Chow, Y., Li, L., and Schuurmans, D. (2019). Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074.

[3] Lee, Jongmin, et al. "Optidice: Offline policy optimization via stationary distribution correction estimation." International Conference on Machine Learning. PMLR, 2021.

### Questions
- Section 5.2: The choice of $\pi \propto \delta^r-\delta^c$ is somewhat odd. Such a $\pi$ won't optimize the inner equation of Eq (7), given the current value estimations. I wonder why was this chosen over $\pi \propto \exp(\delta^r-\delta^c)$?

- Figure 3: 
In offline RL/IL, the value functions are solely asscociated with rewards. How, then, were the constraints derived from these methods?

- Table 5 (Appendix): The varying number of expert transitions for different tasks is appears a bit random to me. Could the authors provide clarity on this decision?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
