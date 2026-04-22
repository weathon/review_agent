# MSMR: Bandit with Minimal Switching Cost and Minimal Marginal Regret

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Effectively balancing switching costs and regret remains a fundamental challenge in bandit learning, especially when the arms exhibit similar expected rewards. Traditional upper confidence bound (UCB) -based algorithms struggle with this trade-off by frequently switching during exploration, incurring high cumulative switching costs.
Recent approaches attempt to reduce switching by introducing structured exploration or phase-based selection, yet they often do so at the expense of increased regret due to excessive exploitation of suboptimal arms. 
In this paper, we propose a new unified framework for bandit problems with switching costs, containing several classical algorithms, applicable to both Multi-Armed Bandits (MAB) and Combinatorial Multi-Armed Bandits (CMAB). Our approach is built on three key components: initial concentrated exploration, near-optimal exploitation, and predictive selection, which together achieve a principled balance between switching cost and regret.
Based on this framework, we introduce the Minimal Switching Cost and Minimal Marginal Regret (MSMR) family of algorithms. 
Theoretically, we show that MSMR algorithms achieve a regret upper bound of $\mathcal{O}(\log n)$ over horizon $n$, incur only $\mathcal{O}((\log n)^{1-\varepsilon})$ switching cost, and its marginal loss has an upper bound of $\mathcal{O}(\lambda \sqrt{\log n})$ by setting $\varepsilon = 1/2$, where $\lambda$ and  $\varepsilon \in (0,1)$ are hyper-parameters.
Experiments show that MSMR algorithms reduce switching costs to 1.0\% (MAB) and 1.3\% (CMAB) of those incurred by standard baselines, while maintaining comparable regret, demonstrating their practical effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes MSMR, a framework with three modules—initial concentrated exploration, near-optimal exploitation, and predictive selection. Instantiated for MAB and CMAB (MSMR-UCB / MSMR-CUCB), it claims log-like regret and sub-log switching, and gives marginal-loss bounds that balance the two via a tunable $\epsilon$. Experiments (cascading CMAB) show orders-of-magnitude fewer switches with comparable regret to CUCB.

### Strengths
1. Clear objective and unified recipe (phase-0 exploration; then long exploits; optional predictive selection). Easy to map onto CUCB.

2. Theoretical contributions.

### Weaknesses
1. Potential inconsistency on regret rate. Table 1 lists MSMR regret as $O( (\log n)^{\epsilon})$, but Theorem 5.2's bound contains the usual $O(\log n)$ term $\sum 8 \log n / \Delta_i$ (dominant for fixed gaps), suggesting $O( \log n)$ overall not $O( (\log n)^{\epsilon})$. 

2. Predictive selection practicality. The conditions (Eq. 5/6) reference future phase lengths $\gamma(\cdot)$ across many arms; it’s not yet clear how robust/cheap this is online beyond toy settings.

### Questions
I am not familiar with CMAB and I have a few questions to learn:

1. Computability of predictive selection. How do you implement Eq. (5)/(6) efficiently without peeking into future choices?

2. How to choose $\epsilon$, $M$, $N$. Beyond theory constraints, what practical tuning rules keep the switching bound while avoiding regret spikes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework for regret minimization under switching costs.

While the topic is interesting and relevant, the paper should be rejected because (1) its theoretical analysis appears to contradict established lower bounds for regret, and (2) the algorithmic description is incomplete and difficult to interpret.

**Main argument**
The paper’s theoretical results are implausible and insufficiently justified. In particular, the claimed regret bounds seem to violate known lower bounds for stochastic bandits. Moreover, key algorithmic details are either inconsistent or missing.

### Strengths
- The paper attempts to unify regret minimization with switching considerations in a single theoretical framework.

### Weaknesses
- **Unclear role of hyperparameters.**
The relationships among $\varepsilon$, $\gamma$, and $M$ (all seemingly tunable) are never specified, leaving tuning and theoretical interpretation ambiguous.

- **Implausible theoretical result.**
  The paper claims a regret bound $O((\log T)^{\varepsilon})$ with $\varepsilon < 1$. This is incompatible with the Lai–Robbins lower bound, which implies any consistent algorithm in the stochastic bandit setting must incur at least $\Omega(\log T)$ expected regret. The manuscript should either correct the statement or make explicit the additional assumptions under which such a sub-logarithmic rate would hold.

- **Weak empirical evaluation.**
  Only 20 independent runs are reported, with no justification for this choice and no analysis of variability. There is no ablation to show the effect of hyperparameters.


- **Inconsistencies.**
  - Line 023: the main text states $O(\log n)$, whereas Table 1 lists $O((\log n)^{\varepsilon})$; these are inconsistent.
  - Line 367: the range of $\varepsilon$ is inconsistent across the manuscript (e.g., line 367 vs. line 300).


Things to improve the paper that did not impact the score:
  * Line 167 (and elsewhere): citations are not referenced by author name, which reduces readability.
  - Line 630: the section presents only the algorithm, without accompanying explanation or motivation.
  * Line 1367: duplicated paragraph.

### Questions
- How does the algorithm achieve regret lower than established bounds for this setting? Are there additional assumptions that justify this claim?
- What is the precise role of the hyperparameters $\varepsilon$, $\gamma$, and $M$?
- How do the empirical results depend on these hyperparameters?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of switching costs in bandit algorithms. Traditionally, bandit algorithms do not consider this cost; in fact, they often benefit from rapidly switching between arms. However, this behavior is impractical or disallowed in many real-world applications. This paper aims to make a contribution by demonstrating how to maintain low regret while also keeping switching costs low. It shows this balance is possible by establishing theoretical results and providing empirical comparisons with baselines. The algorithm's design is intuitive and well-justified.

### Strengths
1. The paper is well-motivated, clearly structured, and easy to follow (main body).
2. The algorithm's design is highly intuitive, and the authors provide good intuition to support its expected strong performance.

### Weaknesses
The notation is heavily overloaded. The appendix includes a two-page list of notations, many of which are highly similar, making it almost impossible to follow the proofs.

### Questions
1. The description states that the algorithm takes the total number of time steps ($n$) as input, implying it is a fixed-horizon algorithm. How are the plots showing cumulative regret over time generated in this case?

2. In the first plot, I am surprised that the cumulative regret is so low during the initial phase of the experiment. Since the algorithm clearly separates its run into exploration and exploitation, shouldn't the regret grow rapidly during the exploration phase?

3. It might be also worth commenting on the standard Explore Then Commit (ETC) algorithm?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes the Minimal Switching cost and Minimal marginal Regret (MSMR) framework for stochastic and combinatorial multi-armed bandit problems, aiming to simultaneously minimize cumulative regret and switching cost. Traditional UCB-based methods achieve low regret but incur frequent arm changes, while batch or phased approaches reduce switching at the cost of higher regret. MSMR introduces a unified formulation based on marginal loss, defined as a weighted sum of additional regret and switching cost with trade-off parameter λ. The authors design MSMR-UCB and MSMR-CUCB algorithms that exploit an ε-controlled phase-length function to balance exploration and switching adaptively. Theoretical analysis establishes upper bounds of $O((\log n)^ε)$ for regret, $O((\log n)^{1-ε}$ for switching cost, and $O(λ(\log n)^{1/2})$ marginal loss when $ε = 0.5$ and λ is constant. Empirical results on standard and combinatorial bandit benchmarks show that MSMR achieves competitive regret and significantly lower switching cost compared with prior baselines such as Phased-UCB and B-FTRL.

### Strengths
The paper makes a clear and technically solid contribution to the study of bandits with switching costs. While the general problem of balancing exploration and switching has been studied in prior work, this paper introduces a conceptually neat and analytically tractable framework through the definition of marginal loss, which unifies regret and switching cost into a single evaluation metric. The proposed MSMR algorithms employ a simple UCB-based mechanism with an ε-controlled phase length, and the authors provide explicit theoretical bounds on both regret and switching cost, showing how their relationship is governed by ε and how the algorithm performs well without requiring prior knowledge of the trade-off parameter λ, while also deriving how ε could be optimally adjusted when λ is known to be dependent on $n$.

### Weaknesses
1. The introduction claims that “several recent works propose switching-aware bandit algorithms” but does not cite any specific papers. I also believe the idea of incorporating switching costs in bandit problems dates back decades, e.g., 

Manjari Asawa and Demosthenis Teneketzis. Multi-armed bandits with switching penalties.
IEEE transactions on automatic control, 41(3):328–348, 1996.

2. The paper briefly introduces the Riemann zeta function in the theoretical analysis but does not explain its role or why it appears in the regret bounds.

3. The pseudocode lists $n$, $M$, $α$, and the exploitation function γ(⋅) together as inputs, even though some of them are not problem parameters but algorithm hyperparameters. Presenting them all as inputs obscures which quantities are externally specified versus internally chosen. Moreover, the algorithm is expressed in terms of γ(⋅), while the theoretical analysis and discussion later in the paper are framed around ε. This inconsistency makes it difficult to follow how the practical implementation relates to the theoretical parameterization.

4. All experiments fix ε=0.5 and effectively operate under a constant-λ evaluation. Since the theory discusses how to adjust ε when λ depends on n, include studies that vary λ (several constants) and at least one $λ(n)$ schedule with the corresponding $ε(λ)$ choice.

5. Plots qualitatively show slow growth of regret and switching cost, but there are no log–log fits, slope estimates, or ablation over ε to test $(\log n)^ε$ vs. $(\log n)^{1−ε}$. Adding these analyses would substantiate the theoretical bounds and tighten the theory–experiment connection.

### Questions
Whether there is any conceptual or theoretical connection between MSMR and the dynamic allocation index (Gittins index) framework for multi-armed bandits, though the latter is derived from a Bayesian perspective? Both approaches involve balancing expected rewards and cumulative costs through a linear combination (or trade-off) of the two terms. The MSMR marginal loss formulation, which weights regret and switching cost by λ, appears structurally similar to how the Gittins index integrates discounted rewards and sampling costs.

### Soundness
3

### Presentation
3

### Contribution
2
