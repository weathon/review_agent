# Optimal Batched (Generalized) Linear Contextual Bandit Algorithm

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 2, 6

## Abstract
We study batched linear and generalized linear contextual bandits and introduce practical batched algorithms, aiming for methods that are both practical and provably optimal under limited adaptivity.
For linear contextual bandits, we propose the first algorithm that attains minimax-optimal regret (up to polylogarithmic factors in $T$) in both small-$K$ and large-$K$ regimes using only $O(\log\log T)$ batches, while our second algorithm removes the G-optimal design step—the dominant computational bottleneck—yet preserves the same order of statistical guarantees and achieves the lowest known runtime complexity.
We then adapt to the generalized linear contextual bandits and design an algorithm that is fully free of curvature parameter $\kappa$: neither the algorithm requires knowledge of nor its regret bound depends on $\kappa$, and it retains $O(\log\log T)$ batch complexity with near-optimal regret.
Collectively, these results deliver the first batched linear contextual methods that are simultaneously minimax-optimal across all regimes and computationally efficient, and the first generalized linear method that is both statistically and computationally efficient while remaining fully $\kappa$-independent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the problem of learning in batched linear and generalized linear contextual bandits, where decision updates can only occur at a limited number of time points. The goal is to design algorithms that preserve the statistical efficiency of fully adaptive methods while remaining computationally practical under restricted adaptivity.

To this end, the authors develop three algorithms—BLCE-G, BLCE, and BGLE—that together achieve near-optimal regret with minimal batch complexity. The first method, BLCE-G, integrates a near G-optimal design with an arm-elimination procedure and achieves an regret matching the known minimax lower bounds across both small- and large-$K$ regimes while requiring only $O(loglog T)$ batches.
Building upon this, BLCE removes the computationally expensive G-optimal design step and replaces it with an uncertainty-driven exploration mechanism. Remarkably, this simplification does not compromise statistical performance: BLCE retains the same minimax-optimal regret while reducing the overall computational cost to $O(Kd^2 T log log T)$, which is asymptotically the lowest among existing optimal algorithms.

The third contribution, BGLE, generalizes the approach to generalized linear contextual bandits, where rewards depend on a nonlinear link function. A key innovation lies in removing all dependence on the curvature parameter κκ, a challenging quantity that governs the local geometry of the link function. The corresponding regret bound is the tightest known for batched generalized linear bandits and is entirely κκ-free in both leading and transient terms.

Empirical results further support these theoretical claims: the proposed algorithms consistently outperform existing batched bandit methods in cumulative regret and runtime efficiency across diverse regimes. Overall, the paper offers the first unified framework that achieves minimax-optimal regret, $O(log log T)$ adaptivity, and practical computational scalability for both linear and generalized linear contextual bandits.

### Strengths
**Originality**
The paper’s originality is mainly from combining theoretical optimality with computational practicality for batched contextual bandits. Previous methods achieving minimax regret usually relied on computationally heavy G-optimal design or assumed access to curvature parameters (e.g. $\kappa$) as algorithm input in the generalized linear setting. Regarding the proposed algorithms: BLCE eliminates G-optimal design, and BGLE removes dependence on the curvature constant $\kappa$. 

**Quality**
This is a theoretical paper with good quality. The regret analysis seems complete and well matched with existing lower bounds in the related field. The mathematical proof indeed builds on prior techniques, but it also extends existing techniques to cover new cases such as the $\kappa$-free GLM analysis. The experimental evaluation on synthetic data corroborates with the statistical theory and provides credible empirical validation of both statistical and computational claims. Overall, the quality of the paper is good.

**Clarity**
The paper is clearly written and well structured. The motivation is clear and directly linked to practical limitations of existing literature in the field. Key results are presented in a clear way. For example, tables containing comparison of regret, batch complexity, and runtime are presented. Algorithm pseudocode is not too hard to understand. 

**Significance**
The work has significance for both theoretical contributions in bandit, an important topic in machine learning and statistical theory. Importantly, it effectively closes an open gap in the literature by showing that minimax-optimal regret can be achieved under minimal adaptivity without relying on computationally infeasible steps. 
The result also guides the direction for real world applications. To be specific, the removal of G-optimal design and the independence from $\kappa$ make the proposed algorithms very implementable in important real-world settings including statistical decision systems (e.g. recommendation systems or A/B testing platforms) that require good model generalization subject to limited adaptivity. The paper seems of interest to bandit theorists and also practitioners working on sequential decision-making under resource constraints.

### Weaknesses
I think this is a fine paper without major technical weakness. However, the following could be noted.

(1) The theoretical contribution is a bit incremental. The minimax-optimal regret and double-log batch complexity are existing results. I feel the major novelty of the paper’s is removing G-optimal design and achieving $\kappa$-independence. As mentioned in the ‘Strengths’ section, I think these are good refinement that have quite significant implications for real-world use cases.
Therefore, I feel it could be more appropriate to position the paper as a clever technical refinement.

(2) Absence of proof sketches or intuition.
Theoretical arguments are hard to follow. I think some proof sketches highlighting the key ideas behind Theorems 1–3 would improve readability.

### Questions
(1) The synthetic experiments show that the regret for BLCE-G and BLCE are very similar. This means the G-optimal step adds little in their specific experimental setup. I am curious under what setups one significantly outperforms the other.

(2) Regarding $\kappa$-independence: The proposed algorithm BGLE is shown to be $\kappa$-free, but in practice the bound still involves the expected inverse curvature $\hat{\kappa}$. In saturated regimes where $\mu’(<x, \theta^*>)$ is small, $\hat{\kappa}$ can still be large and may degrade the constant factor in the regret. This means $\kappa$ can still have an indirect impact on the statistical guarantee, is the understanding correct? 

(3) The theory defines $\hat{\kappa}$ using the curvature at the optimal arm, but in practice, non-optimal arms might have very different curvature. Does the regret guarantee assume that curvature does not vary too sharply across the arm set?

(4) The algorithms rely on an exploration–exploitation split controlled by $c$. How is this parameter chosen, and how does it affect the empirical regret?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper considers batched linear and generalized linear contextual bandits.
For linear contextual bandits, the proposed algorithm (BLCE-G) achieves the minmax regret up to log factors in both large and small number of actions, and only uses $O(\log \log T)$ batches.
The second algorithm (BLCE) does not require the computation of G-opt design while keeping the statistical guarantees. They further study the generalized linear contextual bandits. The proposed algorithm (BGLE) works without knowledge of the curvature parameter $\kappa$ and the regret bound is independent of it. The algorithm again only uses $O(\log \log T)$ batches and attains the nearly-optimal regret.

### Strengths
1. BLCE-G is the first algorithm to simultaneously match the minimax lower bounds in both large $K$-regime and the small $K$ regime up to logarithmic factors.

2. Keeping $O(\log log T)$ batches and min-max regret for both contextual linear and generalized linear bandits improve over existing results. 

3. Empirical performance is promising in various regimes $(K,d)$.

### Weaknesses
The result is the extension of Yu & Oh (2025) and see questions below.

### Questions
1: Yu & Oh (2025) is one of the most related works. The paper achieves, in the linear bandit,  the minimax regret by integrating arm elimination with a regularized G-optimal design. I understand that the different techniques are required to handle contextual setting, and also remove the G-optimal design in BLCE. It is surprising that results are almost similar with non-contextual settings. I could not fully understand why removing G-optimal design is still fine with theoretical guarantees . You are writing “BLCE instead lengthens the uncertainty-driven exploration phase to occupy those rounds.” in line 257, but I would appreciate a more intuitive and also rigorous explanation behind it. 

In the regret analysis, we need to control $\sum_{s} E[ \max_{y \in A^{\ell-1}} y ^{\top} H^{-1}_{s-1}y]$ (as in (8)) and we need to care about making Gram matrix not to grow excessively during Phase 2 of the algorithm. In order to have such a “good”  random event (line 952), employing G-optimal design seems essential. 

If I understand correctly, (13) is enough to remove G-optimal design and somehow surprising as the RHS in (13) is $O(d \log T)$ while in (8) the bound (dominating term) is $ $O(\frac{d \log T}{T})$. 
Can you comment on the critical idea in analysis for removing G-opt?
It may also be better to add the proof sketch in the main text.

2: What is the computational bottleneck of Zhang et al. (2025) which has time complexity dependent on $d^{7/2}$?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies batched linear and generalized linear contextual bandits under limited adaptivity. The authors propose three algorithms: BLCE-G (Batched Linear Contextual Bandit with Elimination and near G-optimal design), BLCE (same algorithm without the G-optimal design), and BGLE (Batched Generalized Linear Bandit with Elimination). The claimed contributions are: 
- The first linear contextual algorithms achieving minimax-optimal regret in both small-K and large-K regimes with only $O(\log \log T)$ batches, 
- Removal of the G-optimal design step while preserving the same statistical guarantees and achieving lowest runtime,
- The first $\kappa$-independent algorithm for batched generalized linear bandits, whose regret depends only on the average curvature $\hat \kappa$ rather than the worst-case curvature $\kappa$. 
Experiments show empirical improvements over prior batched baselines in regret and runtime.

### Strengths
Please see weakness.

### Weaknesses
The main issue of the paper is that the notion of "batched" in the paper differs from standard definitions in the literature. Section 3.2 claims the agent uses a "fixed policy within each batch", but Algorithms 1–2 adaptively update the Gram matrix $H_t$ at every round and select the arm with maximal matrix norm $\|x\|_{H{t-1}^{-1}}$. Thus, the action sequence within a batch depends on the full within-batch context history. In prior works (e.g., Ruan et al., 2021; Hanna et al., 2023; Zhang et al., 2025), "batched" meant memoryless within a batch: only decisions based on statistics from *previous* batches. Here, the algorithm is sequential within the batch. This difference invalidates a direct comparison with the "static grid" baselines summarized in Table 1, which potentially trivializes the problem.

### Questions
I have no further questions.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studied the batched linear contextual bandit problem.
1. It proposed algorithms for linear bandits that matched the minimax optimal regret bound in both small $K$ and large $K$ regime. 
2. The proposed algorithms also achieved the minimum possible number of policy switches, among algorithms with nearly minimax optimal regret bound.
3. The algorithms didn't need explicit construction of G-optimal design, and had improved time complexity compared to baselines.
4. The paper also extended the algorithm to generalized linear contextual bandits.

### Strengths
1. The paper improved upon previous paper in $\mathrm{poly}(\log d, \log \log T)$ factors in regret term. 
2. The paper utilized Sherman-Morrison formula to compute matrix inverse incrementally and reduced time complexity for the algorithm.

### Weaknesses
1. The regret bound improvements compared to baselines are quite small.
2. In light of point 1, it would be ideal if the paper could show that $\sqrt{\log \log T}$ or $\sqrt{\log d}$ dependency is inevitable in (batched) linear bandits.

### Questions
1. Line 110, lower bound should use $\Omega$ and I believe it shouldn't be $\tilde \Omega$. 
2. Empirical results show that the algorithm with G-optimal design (BLCE-G) is actually inferior compared to BLCE. Can you share some insight on why it's the case?

### Soundness
3

### Presentation
3

### Contribution
3
