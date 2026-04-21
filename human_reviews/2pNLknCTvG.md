# uniINF: Best-of-Both-Worlds Algorithm for Parameter-Free Heavy-Tailed MABs

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 8, 6, 8, 8

## Abstract
In this paper, we present a novel algorithm, `uniINF`, for the Heavy-Tailed Multi-Armed Bandits (HTMAB) problem, demonstrating robustness and adaptability in both stochastic and adversarial environments. Unlike the stochastic MAB setting where loss distributions are stationary with time, our study extends to the adversarial setup, where losses are generated from heavy-tailed distributions that depend on both arms and time. Our novel algorithm `uniINF` enjoys the so-called Best-of-Both-Worlds (BoBW) property, performing optimally in both stochastic and adversarial environments *without* knowing the exact environment type. Moreover, our algorithm also possesses a Parameter-Free feature, *i.e.*, it operates *without* the need of knowing the heavy-tail parameters $(\sigma, \alpha)$ a-priori.
To be precise, `uniINF` ensures nearly-optimal regret in both stochastic and adversarial environments, matching the corresponding lower bounds when $(\sigma, \alpha)$ is known (up to logarithmic factors). To our knowledge, `uniINF` is the first parameter-free algorithm to achieve the BoBW property for the heavy-tailed MAB problem. Technically, we develop innovative techniques to achieve BoBW guarantees for Parameter-Free HTMABs, including a refined analysis for the dynamics of log-barrier, an auto-balancing learning rate scheduling scheme, an adaptive skipping-clipping loss tuning technique, and a stopping-time analysis for logarithmic regret.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies parameter-free best-of-both-worlds (BOBW) for HT-MABs, where 1) HT means that the loss distributions can be unbounded but have $\\alpha$-th moment bounded by $\\sigma^{\\alpha}$, for some $\\sigma>0, \\alpha\in(1,2]$; 2) BOBW means that one single algorithm can enjoy logarithmic gap-dependent regret in the stochastic environment (loss distributions are fixed over time) and worst-case optimal regret in adversarial environment (loss distributions change over time), without knowing in advance whether the environment is sto. or not; 3) parameter-free means that the algorithm doesn’t now the value of $\\sigma>0, \\alpha\in(1,2]$, but can ensure the regret guarantee as if they were known.

An algorithm called uniINF is proposed, which ensures $\\tilde{O}(\\frac{K}{(\\Delta_{\\text{min}})^{\\frac{1}{\\alpha-1}}}) $ (expected pseudo-)regret in sto. env. (which is optimal up to log terms and the gap dependency), and near-optimal regret in adv. env. (which is optimal up to log terms) when the loss distributions of the optimal arm satisfy the truncated non-negative assumption (Assumption 1). This is the first parameter-free BOBW result in HTMAB. Previous results approach that in one single env. only (either sto. or adv.).

Technically, this is achieved by several components, including 1) iterative and adaptive learning rate scheduling; 2) adaptive clipping/skipping; 3) refined analysis for log-barrier regularizer.

### Strengths
The parameter-free BOBW bound in HTMAB is a quite strong guarantee, and to achieve this, several technical innovations are proposed.

### Weaknesses
I’m not happy with how Assumption 1 is justified. From Line 193 to 195, it says that "we make the following **essential** assumption. As shown in (Genalti et al., 2024, Theorems 2 & 3), without Assumption 1, there does not exist HTMAB algorithms that can … without knowing either $\\alpha$ or $\\sigma$." However, this statement could be misleading based on my understanding on (Genalti et al., 2024).

The negative result shown in (Genalti et al., 2024) is that, it’s impossible for one single algorithm to match the lower bound in (Bubeck et al., 2013) for all unknown $\\sigma>0$ or $\\alpha\\in(1,2]$. However, I don’t think it has been characterized that how weak the needed assumption to be "parameter-free" in HTMAB. In fact, in the conclusion part of (Genalti et al., 2024), it even says that "investigating the role of the truncated non-positivity assumption, especially, whether weaker assumptions can be formulated."

Therefore, I would urge the authors to refine the statements related to Assumption 1, as currently it may leave the impression that Assumption 1 is a necessary condition for "parameter-free", which as of now it’s still unclear yet.

### Questions
1. To my understanding, this paper claims to develop a new analysis for $DIV_t$ (also named stability term) when using log-barrier, and introduces an extra $(1-x_{t,i})^2$ factor in the bound, which is the key to get self-bounding property and BOBW. However, I don't quite understand what it means by “$S_t$ is adequately large compared to $||c_t||_{\infty}$”. I tried to find a formal lemma or theorem statement for this new bound (with exactly the same form) on $DIV_t$ term but I failed. Could the authors help explain this? Under what conditions does this new bound hold?

2. What’s the difficulty to get the refined gap dependency? From the appendix, I feel that in both DIV term and SHIFT term, we cannot achieve that. Could the authors elaborate more on that (from the analysis perspective)? Is it because the regularizer is log-barrier rather than Tsallis entropy?

Post rebuttal ===============================

I'm increasing the score from 6 to 8.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work establishes the first parameter-free algorithm for heavy-tailed multi-armed bandits (MABs) with best-of-both-worlds (BOBW) properties. This algorithm does not require prior knowledge of heavy-tail parameters $(\sigma, \alpha)$ and simultaneously obtains the (nearly) optimal regret in both the stochastic and adversarial environments.

### Strengths
1. **Technical innovations**: The proposed algorithm and its analysis incorporate new ingredients including a new skipping and clipping scheme of the loss estimates and a stopping time argument to bound the stability terms and the skipping errors, which seem to be technically valuable and might be of independent interest.
2. **Writing**: Generally, this work is well written.

### Weaknesses
1. **More comparisons with existing literature**: Most parts of the presentation in this work are clear. However, I would like to suggest the authors provide more discussions and comparisons with the techniques in existing literature. For instance, the exclusion of the optimal arm $i^\ast$ in Eq. (5) when using the log-barrier regularizer is also achieved by [1,2]. I am not very sure whether there are additional technical nuances between the exclusion of the optimal arm in Eq. (5) of this work and those in [1,2]. For the data-dependent learning rates, several works have also leveraged them to achieve BOBW results in various online learning problems (say, [2,3,4,5,6,7]). Besides, when bounding the stability term of OMD/FTRL, a key property required is to ensure the multiplicative stability of the update of the prediction. In this work, such a property is guaranteed by Lemma 4. However, it seems not appropriate to call such a lemma “novel” as on Line 423, since it has also appeared in previous works when using the log-barrier regularizer (say, Lemma 9 in [8]; Lemma 12 in [9]).

[1] Ito. Parameter-Free Multi-Armed Bandit Algorithms with Hybrid Data-Dependent Regret Bounds. COLT, 21.

[2] Ito. Hybrid Regret Bounds for Combinatorial Semi-Bandits and Adversarial Linear Bandits. NeurIPS, 21.

[3] Ito et al. Nearly Optimal Best-of-Both-Worlds Algorithms for Online Learning with Feedback Graphs. NeurIPS, 22.

[4] Tsuchiya et al. Best-of-Both-Worlds Algorithms for Partial Monitoring. ALT, 23.

[5] Ito et al. Best-of-Three-Worlds Linear Bandit Algorithm with Variance-Adaptive Regret Bounds. COLT, 23.

[6] Kong et al. Best-of-three-worlds analysis for linear bandits with follow-the-regularized-leader algorithm. COLT, 23.

[7] Ito et al. Adaptive Learning Rate for Follow-the-Regularized-Leader: Competitive Analysis and Best-of-Both-Worlds. COLT, 24.

[8] Lee et al. A closer look at small-loss bounds for bandits with graph feedback. COLT, 20.

[9] Jin et al. Simultaneously Learning Stochastic and Adversarial Episodic MDPs with Known Transition. NeurIPS, 20.

### Questions
1. If $(\sigma, \alpha)$ is known a priori, can we eliminate Assumption 1? What are the main technical difficulties when eliminating Assumption 1 in the case of known $(\sigma, \alpha)$?
2. In the previous work [10], the Tsallis entropy regularizer is used while the log-barrier regularizer is used in this work. Is it because the magnitude of the loss estimates in this work is larger than the magnitude of the loss estimates in [10]?

[10] Huang et al. Adaptive Best-of-Both-Worlds Algorithm for Heavy-Tailed Multi-Armed
Bandits. ICML, 22.

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
4

### Summary
The paper addresses the heavy-tailed MAB problem, a variant of the stochastic bandit problem where rewards are sampled from distributions having potentially infinite variance. The main contribution of this work is to provide an algorithm with tight regret guarantees in both the stochastic and adversarial HTMAB problem. While the performance in the stochastic setting is worse (not in terms of T) than existing algorithms (e.g. AdarUCB), the algorithm simultaneously deals with the two settings and is tight in both the instance-dependent and independent sense.

### Strengths
The paper is written, and it provides an exhaustive review of the existing literature.

The contribution is clear, and it is well highlighted which open question the paper addresses.

The paper also presents some nice technical contributions in the algorithm and in the proofs.

### Weaknesses
Overall, the contribution is limited when it comes to applications since adversarial HTMABs are uncommon in the real world and the literature. Instead, in the purely stochastic setting, the algorithm does slightly worse than AdaRUCB by a factor of  $\log \frac{\sigma^\alpha}{\Delta_{min}}$ (Genalti et al.)

### Questions
- How do applications justify the adversarial HTMAB? Can you please provide some examples?

- I think it would be interesting to highlight the trade-off between ADAR-UCB and UniInf more. How is the best-of-both-worlds property related to the extra factor in the stochastic setting's regret bound? Does your algorithm require extra round-robin turns (as in Adar-UCB)?

- Do you know what the optimal performance would be without the truncated non-negativity assumption? Are there any known lower bounds for the problem without this assumption?

- It would be interesting to understand if alternative (and possibly weaker) assumptions can lead to the same performances (I would like to point out that Theorems 2 and 3 from Genalti et al. don't necessarily imply that this specific assumption is required, but rather that without any assumptions such performances are unattainable).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies Heavy-Tailed MultiArmed Bandits (HTMAB) problem. The main contribution of the paper is to design an optimal algorithm that achieves both Best of-Both-Worlds (BoBW) and Parameter-free properties for HTMAB, where BoBW means that the algorithm performs optimally in both stochastic and adversarial environments and Parameter-free means that the algorithm do not need to know the the heavy-tail parameters in advance.

### Strengths
1. The paper is well written. The theoretical results and proof appear to be correct.
2. The paper achieves worst-case BoBW optimal regret for HTMAB, which improves previous results proposed in [Huang 2022].

### Weaknesses
1. The paper should include the comparisons with previous scale-free MAB works, e.g. [1-5]. Specifically, the algorithm structure proposed in the paper seems very close to the one proposed in [3], which also uses the clipping/skipping technique and inf regularization. The differences should be further clarified.
2. Assumption 1 is a bit weird. I can understand why it is unavoidable, but I suggest that the authors can give the best (not worst-case optimal) upper bounds we can get without this assumption.

### Questions
When $\alpha, \sigma$ known, it is trivial to use regular FTRL based algorithm to achieves (nearly) optimal worst-case regret for adversarial bandit problems with potentially heavy-tailed losses (fix the clipping bound $[-r,r]$ with $r=\sigma T^{1/\alpha}K^{-1/\alpha}$ and use Theorem 4 in [6]). When $\alpha, \sigma$ are unknown, intuitively, it suffices to use the adaptive clipping bound according to the empirical estimation of $\alpha, \sigma$ (Line 6 of ALG 1). Is the high-level idea of the algorithm in this paper the one I described?


References: 
[1] Putta, Sudeep Raja, and Shipra Agrawal. "Scale-free adversarial multi armed bandits." International Conference on Algorithmic Learning Theory. PMLR, 2022.

[2] Chen, Mingyu, and Xuezhou Zhang. "Scale-free Adversarial Reinforcement Learning." arXiv preprint arXiv:2403.00930 (2024).

[3] Chen, Mingyu, and Xuezhou Zhang. "Improved Algorithms for Adversarial Bandits with Unbounded Losses." arXiv preprint arXiv:2310.01756 (2023).

[4] Huang, Jiatai, Yan Dai, and Longbo Huang. "Banker online mirror descent: A universal approach for delayed online bandit learning." International Conference on Machine Learning. PMLR, 2023.

[5] Hadiji, Hédi, and Gilles Stoltz. "Adaptation to the range in k-armed bandits." Journal of Machine Learning Research 24.13 (2023): 1-33.

[6] Wei, Chen-Yu, and Haipeng Luo. "More adaptive algorithms for adversarial bandits." Conference On Learning Theory. PMLR, 2018.

### Soundness
3

### Presentation
3

### Contribution
3
