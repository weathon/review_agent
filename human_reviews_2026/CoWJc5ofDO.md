# Bandit Learning in Matching Markets Robust to Adversarial Corruptions

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
This paper investigates the problem of bandit learning in two-sided decentralized matching markets with adversarial corruptions. In matching markets, players on one side aim to learn their unknown preferences over arms on the other side through iterative online learning, with the goal of identifying the optimal stable match. However, in real-world applications, stochastic rewards observed by players may be corrupted by malicious adversaries, potentially misleading the learning process and causing convergence to a sub-optimal match.  We study this problem under two settings: one where the corruption level $C$ (defined as the sum of the largest adversarial alterations to the feedback across rounds) is known, and another where it is unknown. For the known corruption setting, we develop a robust variant of the classical Explore-Then-Gale-Shapley (ETGS) algorithm by incorporating widened confidence intervals. For the unknown corruption case, we propose a Multi-layer ETGS race method that adaptively mitigates adversarial effects without prior corruption knowledge.  We provide theoretical guarantees for both algorithms by establishing upper bounds on their optimal stable regret, and further derive the lower bound to demonstrate their optimality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies bandit learning for two-sided decentralized matching markets with corrupted feedback, i.e., the feedback revealed to the learner could be corrupted from the ground truth within some certain corruption level $C$. Algorithms are presented for both known $C$ and unknown $C$ setting.

In the known $C$ setting, one can simply enlarge the confidence interval so that the ground truth still lies in it (w.h.p). However, it is not applicable to unknown $C$ case. Therefore, in the later setting, in the algorithm design, multiple algorithm instances are created, each of which are tailored to a specific corruption level and are sample with different probability. Eventually, one can show theoretical guarantee when coordinating all these instances together.

### Strengths
1. A new setup that has not been addressed.
2. Theoretical upper and lower bounds.

### Weaknesses
n/a. please see questions.

### Questions
1. In terms of how to address unknown corruption level, how would the design in this paper different from Lykouris et al. (2018)?
2. Since each instance is sampled with a fixed probability, how can one adapt to the best/correct one on the fly? (or do we need to?)
3. In Table 1, why are the regret bound and comm. cost exactly the same in the unknown $C$ setting?
4. From the lower bounds in Thm. 4.1, how can we claim Remark 4.2? The suboptimality gap has a different definition, and the unknown $C$ upper bound looks quite complicated, can the authors please give a breakdown?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates an online learning problem in two-sidd decentralized matching markets where stochastic rewards are subject to malicious adversarial corruptions. The goal is for N players and K arms to identify the player optimal stable match whilst minimizing the pseudo-regret. The problem is first studied when the corruption level C is known, where the proposed algorithm achieves a regret upper bound that is shown to be near optimal with corresponding lower bounds. The authors extend to the setting of unknown C by coordinating exploration among $O(\log T)$ parallel running instances of their original algorithm. They validate their results with some empirical results as well.

### Strengths
1. The introduction of adversarial robustness into decentralized matching market bandits is a nice direction. The paper addresses realistic adversarial corruptions beyond the common stochastic feedback.

2. The multi-layer extension is elegant to solve the unknown C. The need to introduce a synchronization mechanism to overcome conflicts inherent in randomized strategies is a crucial observation.

3. The optimality analysis for the known C setting is a strong point.

4. The emprical validations are nice.

### Weaknesses
1. For the case of unknown C, there exists a suboptimality gap. The authors acknowledge this multiplicative logarithmic gap and attribute it to the overhead caused from running $\log T$ paralllel instances.

2. Some important details regarding the algorithmic procedures are deferred to the Appendix, specifically Appendix B; such as Phase 1 index estimation, Phase 3 decentralized GS, leader selection and synchronization specifics. I believe the papers novelty rests heavily on adapting decentralized mechanics to robustness. Therefore, even if the pseudo code remains supplemental, more insights should be provided in the main body of the paper.

### Questions
1. Is the multiplicative redundancy $\log^2 T$ on the regret bound of unknown C fundamentally unavoidable given the layer based approach? Is there a possibility of reducing the overhead to tighten the optimality gap, mye by using alternative communication strategies or nonuniform synchronization intervals?

2. In Remark 3.5, the optimal choice ofor hyper-parameter $d$ is theoretically derived as $O(\sqrt{\log T})$ to minimize the regret bound. Figure 1(c) shows that larger $d$ values worsen the empirical regret compared to $d=1$. Can you provide a deeper empirical discussion on the optimal $d$ value?
3. The upper bounds utilize the minimum preferene gap $\Delta$, while the lower bound utilizes $\tilde{\Delta}$ specific to Optimally Stable Bandits. Can you clarify the relationship between these two metrics? Does the near-optimality established in Remark 4.2 hold because $\Delta$ and $\tilde{\Delta}$ are equivalent under OSB? Is the derived lower bound also applicable to the general setting characterized by $\Delta$?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a multilayer preference learning bandit algorithm that can adapt to an unknown level of corruption. Numerical experiment corroborated the theoretical findings and shows an advantage over existing approaches.

### Strengths
I find the paper to be well written, with a clear storyline, strong motivation, and sufficient explanation of the technical challenges. In particular, the authors effectively use concrete examples to illustrate why a two-sided market with corrupted feedback is an interesting and relevant setting. The discussion of the technical difficulties and the insights provided is also sufficiently detailed and informative.

### Weaknesses
- From the paper itself, it is not entirely clear how the algorithm's overall computation complexity scales as $K$ and $T$ grow. In particular, it is unclear how well the monitoring subroutine scales compared to existing approaches for online learning in a two-sided market.
- The authors commented on adversarial feedback. Would an FTRL/FTPL-type algorithm achieve a best-of-three-worlds guarantee in the sense that it achieves the same regret in all stochastic, corrupted, and adversarial environments? Can the authors comment on the related works in this direction and outline the technical difficulties in unknown corruption level, and why this is not a suitable algorithmic framework to consider for this setting?

### Questions
See the previous section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This tackles the problems of one-to-one decentralised matching with bandit feedback  under adversarial corruption, for the case where one side's (arms) preferences are known.

The decentralised matching works by players proposing to arms, and each arm accepting their most prefered player. The feedback here is corrupted, with stochastic rewards $r^S$ being modified to $r$ up to a total corruption budget $C$ over the horizon $T$.

The basic algorithmic idea is to take the corruption level into account when creating confidence intervals. This is first analysed for known corruption in 3.1, which results in $KC/\Delta$ additional regret and in 3.2 for the case of unknown corruption.

The paper refers to an index estimation phase in some algorithms, to take care of conflicts. I personally find the decentralised setting to just add unnecessary complexity, without making the problem more interesting. If you have to force everybody to do a round-robin according to some indices, why not just force them to play according to some centrally defiend schedule? This would also avoid the complications with sampling in the unknown corruption case.

### Strengths
+ It is a natural extension of the stochastic case.
+ The proof is not a completely straightforward adaptation of Lykouris to matching.

### Weaknesses
- Maybe the decentralised setting is too complicated.

### Questions
? The paper refers to an index estimation phase in some algorithms, to take care of conflicts. I personally find the decentralised setting to just add unnecessary complexity, without making the problem more interesting. If you have to force everybody to do a round-robin according to some indices, why not just force them to play according to some centrally defiend schedule? This would also avoid the complications with sampling in the unknown corruption case. But maybe then the proofs would be too straightforward from Lykouris.

### Soundness
3

### Presentation
3

### Contribution
2
