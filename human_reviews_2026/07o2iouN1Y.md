# Nash Policy Gradient: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Finding Nash equilibria in imperfect-information games remains a central challenge in multi-agent reinforcement learning. While regularization-based methods have recently achieved last-iteration convergence to a regularized equilibrium, they require the regularization strength to shrink toward zero to approximate a Nash equilibrium, often leading to unstable learning in practice. Instead, we fix the regularization strength at a large value for robustness and achieve convergence by iteratively refining the reference policy. Our main theoretical result shows that this procedure guarantees strictly monotonic improvement and convergence to an exact Nash equilibrium in two-player zero-sum games, without requiring a uniqueness assumption. Building on this framework, we develop a practical algorithm, *Nash Policy Gradient* (NashPG), which preserves the generalizability of policy gradient methods while relying solely on the current and reference policies. Empirically, NashPG achieves comparable or lower exploitability than prior model-free methods on classic benchmark games and scales to large domains such as *Battleship* and *No-Limit Texas Hold'em*, where NashPG consistently attains higher Elo ratings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper solves Nash equilibrium in two-player zero-sum extensive-form games by adding additional regularization. By switching the reference strategy periodically, the algorithm converges to the NE of the original game rather than the regularized game. The paper proves convergence theoretically and empirically evaluates its performance.

### Strengths
- The presentation is clear and easy to follow
- The paper includes deep learning experiments and achieves better performance compared to the baselines

### Weaknesses
My major concern is that the comparison with previous work is not enough. The most relevant paper I can think of, [1], is not included in this paper.

[1] added additional regularization and changed the reference strategy periodically. Moreover, [1] had theoretical guarantees on convergence.

[1] Abe, Kenshi, et al. "Adaptively perturbed mirror descent for learning in games." arXiv preprint arXiv:2305.16610 (2023).

### Questions
See weaknesses. May the authors explain the main contribution of this paper compared to [1]? I would be happy to re-evaluate the paper if the authors could clarify this.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Nash Policy Gradient (NashPG), a novel algorithm for finding Nash equilibria in two-player zero-sum games. The authors fix regularization at a large value for stability and achieve convergence through iterative refinement of a reference policy. Theoretically, the authors showed that NashPG demonstrates monotonic improvement and convergence to the Nash equilibrium. Experiments on benchmark and imperfect-information games show that NashPG not only exhibit faster convergence, but also achieves lower exploitability and higher Elo ratings than population-based and CFR-inspired baselines.

### Strengths
The idea of iterative refinement under fixed regularization strength was not previously explored in MARL. The analytical results seem rigorous and intuitive to the reviewer.
The empirical evaluation of NashPG is not limited to toy examples to validate the theoretical result. The authors also included more complicated games with imperfect information to demonstrate the performance and effectiveness of NashPG compared to baseline methods.
The paper is well written with clear motivations, and the theoretical and experimental results are also presented in a concise and readable manner.

### Weaknesses
Although the authors did provide a theoretical analysis, no convergence guarantees in finite time were provided; instead, there were only a few lemmas. This significantly hurt the theoretical contribution of this work, especially since existing policy gradient algorithms such as [1] have been shown to guarantee convergence to NE in finite time. The author also did not cite this paper and a few other closely related works.
[1] Zhang, Kaiqing, et al. "Model-based multi-agent RL in zero-sum Markov games with near-optimal sample complexity." Journal of Machine Learning Research 24.175 (2023): 1-53.

The authors should include synthetic examples in order to show convergence properties. Current experiments are very different from the algorithms in the previous sections. At the very least, the authors should have had experiments demonstrating the monotonic descent properties of NashPG in practice.

### Questions
- How about multi-agent RL beyond two-player zero-sum?
- How does this compare or relate to the anchor changing algorithm [2] in single-agent RL?
Zhou, Ruida, et al. "Anchor-changing regularized natural policy gradient for multi-objective reinforcement learning." Advances in neural information processing systems 35 (2022): 13584-13596.
- In the experiments, what does the step number represent? Since the proposed NashPG is a dual-loop method, whereas some other baseline methods are single-loop. Does step number represent T or KT? It could be an unfair advantage for the other algorithms. A better approach is to compare the number of gradient calculations across different algorithms.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines regularization-based equilibrium finding in two-player zero-sum games. The paper proves that iteratively replacing the reference policy in Magnetic Mirror Descent (MMD) with the fixed-point policy of the last learning round guarantees monotonic improvement and last-iterate convergence towards Nash equilibrium (NE). Based on this finding, the paper proposes a reinforcement learning algorithm called Nash Policy Gradient (NashPG) that iteratively reduces the exploitability of the learned policy for practically solving large-scale games. Experiments in benchmark or real-world games demonstrate the superiority of NashPG over MMD and average-iterate-based algorithms like Neural Fictitious Self-Play (NFSP) and Policy-Space Response Oracles (PSRO).

### Strengths
1. This paper is well-written and easy to follow. The background of solving regularized variational inequalities is well-introduced.

2. The theoretical results of iterative improvement are clearly demonstrated without assuming interior or unique NE.

3. The construction of NashPG is elegant, making the algorithm generally applicable under different single-agent RL algorithms.

4. Extensive experiments convince me that NashPG is efficient. The choices of performance measure are also reasonably explained.

### Weaknesses
1. The idea of iteratively replacing the reference policy is not that novel in view of the existing literature [1,2,3,4] in the field of equilibrium finding.

2. More recent RL algorithms for solving imperfect-information games (like NeuRD [5]) could also be considered as comparative methods.

[1] Julien Perolat, et al. From Poincare recurrence to convergence in imperfect information games: Finding equilibrium via regularization. ICML, 2021.

[2] Kenshi Abe, et al. Mutation-driven follow the regularized leader for last-iterate convergence in zero-sum games. UAI, 2022.

[3] Kenshi Abe, et al. Adaptively perturbed mirror descent for learning in games. ICML, 2024.

[4] Runyu Lu, et al. Divergence-regularized discounted aggregation: Equilibrium finding in multiplayer partially observable stochastic games. ICLR, 2025.

[5] Julien Perolat, et al. Mastering the game of Stratego with model-free multiagent reinforcement learning. Science, 2022.

### Questions
1. Could you compare the theoretical results with references [1,2,3,4] to further explain what leads to the advantage of your theory?

2. Could you also compare NashPG with NeuRD [5] or explain the reason why you prefer not to compare it in the experiment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Regularized Nash equilibriums are easier to compute compared their non-regularized counterpart but are exploitable in the original game. There exists a body of works that try to shrink this gap. Commonly, this is done via temperature decay. Unfortunately, lower temperatures are often unstable and converge slowly. R-NaD and FoReL proposed to iteratively regularize to the previous Nash equilibrium in an iterative process. This work proposes a policy space version of the same result.

### Strengths
On the more material side of things, the paper is well written, the accompanying Jax codebase is also easy to read. The paper proposes a natural way of updating the magnet of MMD which has been used successfully in R-NaD.

### Weaknesses
The theoretical results are not exciting. They are equivalent to the theorems of R-NaD (FoReL) but for mirror decent instead of follow the regularized leader (FTRL).

The experiments are questionable. For starters, the choice of using a linear policy is questionable. This is particularly concerning as the results do not match, for instance those of Rudolph et al 2025 (cited @965).  Calculating exploitability with neural networks has been shown to be extremely unreliable but the authors do not measure it for the 4 games @ 413 even thought exploitability calculation is readily available. Plotting standard deviation with 4 samples is somewhat questionable. Negative exploitability is a clear indicator of  bug or learning issues. In the worst case it should be zero by initializing the best response policies with the best policy available. The use of ELO for hold'em is questionable as the average win points is the quantity of interest. Similarly, given that imperfect information games are cyclic in nature, the use of ELO is also questionable. In all cases EVs should have been reported instead. Furthermore, for hold'em, a comparison with slumbot is necessary to gauge the strength of the policy. The "real games" are not necessary real games, for liars dice, the 6th dice is missing and the observations are not open spiel compatible.

### Questions
I think the paper would strongly benefit from having a table of sort describing the different contribution of this work vs mmd vs R-NaD (deep Nash, FoReL, etc).

Why are the MMD experiment ran with PPO? Is it unreasonable to do vanilla policy gradient directly? This is the first MMD implementation I've seen that uses PPO.

The claim @206 that the need to decrease learning rate must decrease with temperature is problematic is not evaluated. As the temperature decreases the movement of the regularized Nash also decreases (at least empirically) as it moves on a finite curve from the objective with alpha_0 regularization to the maximum entropy objective with no entropy. Are there any evidence that these vanishing step sizes are problematic?

### Soundness
2

### Presentation
2

### Contribution
2
