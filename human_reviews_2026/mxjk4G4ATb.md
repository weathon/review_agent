# Latency-Aware Contextual Bandit: Application to Cryo-EM Data Collection

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
We introduce a latency-aware contextual bandit framework that generalizes the standard contextual bandit problem, where the learner adaptively selects arms and switches decision sets under action delays. In this setting, the learner observes the context and may select multiple arms from a decision set, with the total time determined by the selected subset. The problem can be framed as a special case of semi-Markov decision processes (SMDPs), where contexts and latencies are drawn from an unknown distribution. Leveraging the Bellman optimality equation, we design the contextual online arm filtering (COAF) algorithm, which balances exploration, exploitation, and action latency to minimize regret relative to the optimal average-reward policy. We analyze the algorithm and show that its regret upper bounds match established results in the contextual bandit literature. In numerical experiments on a movie recommendation dataset and cryo-electron microscopy (cryo-EM) data, we demonstrate that our approach efficiently maximizes cumulative reward over time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors studied a latency-aware contextual bandit framework that extends standard contextual bandits by incorporating the action delays and formulates it as a special case of a semi-Markov decision process. The authors proposed the Contextual Online Arm Filtering (COAF) algorithm, which combines stochastic approximation and UCB exploration to balance reward and latency. The authors provided theoretical analysis of their algorithm, proving sublinear regret bounds. Finally, they conducted numerical experiments on MovieLens and cryo-EM data to demonstrate that COAF outperforms baselines and improves data collection efficiency.

### Strengths
- The proposed latency-aware model generalizes contextual and combinatorial bandits by explicitly accounting for temporal costs. This is a novel problem in the bandits literature.
- The theoretical analysis appears sound and comprehensive, though I have not checked every proof in detail).
- I also appreciate the discussion of the application to cryo-EM data collection, which highlights the real-world relevance of the framework. Modeling microscope exposure and movement as latency is a strong and realistic motivation that grounds the theoretical development.

### Weaknesses
- While the latency-aware formulation is novel, COAF primarily builds on existing tools such as UCB and stochastic approximation. The conceptual combination is interesting but may feel incremental without deeper theoretical or algorithmic insights. Could you clarify whether the current results provide any new algorithmic intuition or theoretical implications for the broader bandit literature?

- The numerical experiments, though illustrative, are relatively small-scale, so the insights they provide are somewhat limited. The cryo-EM evaluation appears to use simulated data, with experiments mainly comparing COAF to human microscopists. Could you offer some more comprehensive analysis here? E.g., an ablation study examining how COAF’s performance changes under different latency distributions. Similarly, the MovieLens experiments feel limited in scope, particularly in the choice of baselines. It would be informative to also compare against a number of standard contextual bandit algorithms.

- Finally, it would be valuable for the authors to discuss additional application domains where the proposed latency-aware bandit framework could be beneficial, beyond the cryo-EM setting.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study the contextual MAB problem where each action incurs a context dependent time cost and the goal is to maximize the reward per unit time. They develop new algorithm called COAF that jointly learns the reward model with UCB style confidence band and also learns the optimal average reward. They also showcase the performance of the algorithm with theoretical guarantees with regret bound in several regimes.  The theoretical work is supported with experiments conducted on two real world data from different domains showcasing the adaptability of the proposed setting.

### Strengths
The problem setting is clearly motivated with a proper use case of cryo em data collection and is designed to tackle similar use case.

The problem formulation has a generalization over contextual bandits, combinatorial semi bandits, which makes it solid. Also, COAF is supported by optimality equation and design with its dependence. 

The experimentation is supported by real world data to show the working validation of the motivating example. Along with it, they also show their performance on other domain with MovieLens data to showcase the wide adaptability of the setting. 

Having per arm feedback within combinatorial choice helps reduces variance and seems to be realistic for their application.

### Weaknesses
The setting allows for switching to a new decision sets but don't signify the regime when it is optimal as supposed to exploiting. 

The experimentation lacks proper baseline to compare the effectiveness of the proposed algorithm COAP. 

The problem setting has IID assumption with ($X_j$ ,$A_j$ , $l_j$), however this might applications where nonstationary has to dealt with and taken into account.

### Questions
Since Latency aware contextual bandits seems to be the special case of contextual bandits and contextual semi bandits, If the action space and context of the arm is reduced to be similar to stochastic bandits, Does Latency aware Contextual bandits reduce to Budgeted bandits ? If so, how does the regret bound compare in this scenario ? 

If a learner is allowed to request a new action set, how does this switch take latency into account, Is it already a part of the latency of the original selection action set ? 

Since the work is motivated by cryo em data collection, In latency aware contextual bandit setting with COAF can it exploit all the structure in the latency observed rather than treating it as arbitrary ?

Also, For the cryo-EM, Does COAP outperforms the strong domain specific heuristic or is there any advantage of using a learned policy for cryo em data collection application ? 

Also the numerical experimentation only involves a baseline comparison with the humans and Can any of the contextual bandits setting be adapted with mild relaxation to consider them for baseline evaluation ?

Often case, since cryo EM data collection involves human microscopists, drift in instrumentation or user's action changes mid run. In that case, under a IID assumption ($X_j$ ,$A_j$ , $l_j$), how does the algorithm behave ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper investigates a latency-aware contextual bandit problem, where each action (arm) incurs a random latency drawn from an unknown distribution. To capture the impact of latency on decision-making, the authors model the problem as a Markov Decision Process (MDP) and derive the corresponding Bellman optimality equation. Building upon this formulation, they propose an arm filtering algorithm that balances exploration and exploitation by accounting for both reward and latency. The proposed approach is demonstrated through experiments on the MovieLens 1M dataset and a Cryo-EM experimental setting.

### Strengths
The paper has the following strengths:
- The paper provides a theoretical formulation by modeling the latency-aware contextual bandit problem as an SMDP and deriving the corresponding Bellman optimality condition.
- The paper introduces a contextual online arm filtering (COAF) algorithm based on the derived Bellman condition and establishes regret bounds for both linear and general reward function settings.
- The problem is well-motivated by a real-world Cryo-EM application, and the proposed method is empirically validated on both MovieLens 1M and Cryo-EM datasets.

### Weaknesses
The weaknesses are described below.
- Although the paper formulates the latency-aware contextual bandit problem as an MDP, it does not clearly justify why the proposed method is preferable to existing MDP-based solutions.

- The arm filtering design and regret analysis follow relatively standard techniques, and the paper does not clearly articulate new analytical challenges introduced by latency or contextual dependencies.

- The study focuses solely on the stochastic setting, which can already be addressed by conventional MDP algorithms. Extending the formulation to adversarial or non-stationary environments would make the contribution more compelling.

- The impact of action latency on the learning rate or convergence behavior is not explicitly analyzed or reflected in the algorithmic design, despite being central to the problem motivation.

- The experimental evaluation is limited to the proposed method without comparisons against existing delayed-feedback bandits [1,2] or MDP-based algorithms, which weakens the empirical evidence supporting the algorithm’s effectiveness.


[1] Masoudian, S., Zimmert, J. and Seldin, Y., 2022. A best-of-both-worlds algorithm for bandits with delayed feedback. Advances in Neural Information Processing Systems, 35, pp.11752-11762.

[2] Lancewicki, T., Rosenberg, A. and Mansour, Y., 2022, June. Learning adversarial markov decision processes with delayed feedback. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 7, pp. 7281-7289).

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines a version of the contextual combinatorial bandit problem in which each action (a subset of arms) incurs a latency—a variable time cost. The goal is to maximize expected reward per unit of elapsed time rather than per round.

The authors model this as an average-reward semi-Markov decision process (SMDP) and derive a Bellman optimality equation of the form
E_{(X,A,l)}\!\left[\min_{A\in\mathcal{A}}\{l(A)\Gamma - \sum_{i\in A}\mu_i\}\right] = 0,
where \Gamma represents the long-run average reward rate.
They then propose an algorithm, COAF (Contextual Online Arm Filtering), that combines a Robbins–Monro–type stochastic approximation for estimating \( \Gamma^\* \) with UCB-style exploration for learning the arm rewards \mu_i(x).

Regret bounds of order O(T^{3/4}) are proved under both linear and general function classes, and experiments on synthetic data (MovieLens) and a cryo-electron microscopy (cryo-EM) simulation are presented.

### Strengths
1) The latency-aware formulation is conceptually relevant to real scientific workflows.
2) The mathematical derivations are careful and correct.
3) The paper is generally well written and easy to follow.
4) The cryo-EM example adds color and a nice application context.

### Weaknesses
1.	The regret bound is very likely suboptimal.
	2.	There is no lower bound or discussion of optimality.
	3.	The “latency” feature mostly amounts to a time-rescaling, I think; it is not clear why this warrants a fundamentally new theory. 
	4.	The experiments lack statistical rigor—no error bars or serious baselines.
	5.	Overall novelty is modest: the algorithm is a straightforward hybrid of known tools (UCB + stochastic approximation).

### Questions
1.	Do you believe the T^{3/4} rate is unavoidable, or is it simply an artifact of your analysis?
	2.	When latencies are known and bounded, why can’t the setting be reduced to a contextual bandit with a random time clock?
	3.	Could one obtain a sharper \sqrt{T}-type result using a ratio or Dinkelbach-style formulation?
	4.	What exactly does “throughput” measure in the cryo-EM experiment, and how does it relate to \(\Gamma^\*\)?
	5.	Please clarify whether the cryo-EM data come from real microscope logs or a synthetic simulator.

### Soundness
3

### Presentation
3

### Contribution
2
