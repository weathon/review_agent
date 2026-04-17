# Delayed Adversarial Attacks on Stochastic Bandits

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
We study adversarial attacks on stochastic bandits when, differently from previous works, the attack of the malicious attacker is delayed and starts after the learning process begins. We focus on strong attacks and capture the setting in which the malicious attacker lacks information about the beginning of the learning process, a limitation that can dramatically affect the effectiveness of the attack. We introduce a more general framework to study adversarial attacks on stochastic bandit algorithms, providing new definitions of success and profitability of an attack, that account for variable corruption start time. We then analyze success and profitability for different families of algorithms such as UCB and $\epsilon$-greedy against an omniscient attacker. In particular, we derive upper and lower bounds on the number of target arm pulls, showing that our bounds are tight up to a sublinear factor. Finally, we identify an intuitive condition that characterizes when an attack can succeed as a function of its starting time and evaluate the tightness of our theoretical bounds on synthetic instances.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies adversarial attacks on stochastic bandits under a delayed attack setting, where the attacker starts corrupting rewards after the learning process has begun. It generalizes previous works that assume attacks start at the first round and introduces refined notions of attack successfulness and profitability. The authors analyze UCB and ε-greedy algorithms under an oracle attacker, derive tight upper and lower bounds on the number of target-arm pulls, and identify a threshold $\alpha^*(\Delta_{o,\tau},\epsilon)=\frac{\epsilon}{\epsilon+\Delta_{o,\tau}}$ that determines whether an attack can succeed.
Experiments on synthetic data validate the theoretical findings and illustrate how delayed attacks become ineffective when started too late.

### Strengths
- The delayed attack formulation relaxes an unrealistic assumption in existing works, offering a more practical perspective.
- The mathematical analysis is rigorous, and the success threshold is intuitive and interpretable.
- The paper provides solid theoretical analysis with tight upper and lower bounds.

### Weaknesses
- The analysis depends on a fully omniscient oracle attacker, limiting practical applicability. If this assumption could be relaxed, the work would be more impactful.
- This work requires a fixed horizon $T$. However, prior works on adversarial attacks (e.g., Jun et al. (2018) and Liu and Shroff (2019)) on bandits often consider the anytime setting, which is more practical.

### Questions
Could the authors discuss the practicality of the oracle attacker assumption? Are there potential extensions to more realistic attacker models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies a variant of attack policy on stochastic bandits where the attack may be delayed until the middle of the game. This paper studies how that delay will influence that attack policy and the attackibility,  on the UCB, $\epsilon$-greedy, and elimination. For UCB, the paper also identify a successfulness threshold $\alpha^*$. Some experiments are conductd to validate the performance.

### Strengths
1. The delay constraint on attacker is realistic and of practical interest. 
2. The writing of this paper is clear and easy to follow.

### Weaknesses
1. Lack of novelty. 
    1. This paper does not propose new attack policies, and it is mainly adapt the known attack policies. 
    2. This analysis of this paper is also mainly from that of prior work. 
    3. The results of this paper are also a nature extension of the standard attack ones. 
2. Missed important related works, which degrades the motivation of this paper. 
    1. Line 47 says the attack is not recoverable, which is wrong. See algorithms robust to adversarial attacks (https://openreview.net/pdf?id=vOFx8HDcvF) and references therein. 
3. The paper mentions that their results are tight up to a sublinear factor. However, as the results, say the regret or budget, are by default logarithms, “tight up to a sublinear factor” is not tight at all. The reviewer encourages the authors to detail the specific difference between upper and lower bounds.

### Questions
1. Can you present the detailed expression of $\alpha^*$?

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
4

### Summary
This paper studies adversarial attacks on stochastic bandits when the attacker starts after the learning process has begun. The authors propose a delayed attack model that generalizes prior strong-attack settings by introducing a variable start time. They define new notions of successfulness and profitability, analyze UCB and ε-greedy learners under oracle attacks, and derive lower bounds on the number of target-arm pulls. The paper also identifies a threshold beyond which attacks become ineffective and validates the theory through small-scale synthetic experiments.

### Strengths
1) The delayed attack model is a natural and realistic extension of existing adversarial attack frameworks for stochastic bandits.
2) The paper provides theoritical analysis on the number of target-arm pulls.
3) The experiments, though limited, illustrate the theoretical findings and intuitively show how attack delay diminishes effectiveness and increases cost.

### Weaknesses
1) The paper only analyzes oracle attacks (omniscient attackers with full knowledge of arm means). The more practical setting where the attacker has the same information as the learner is not addressed, which limits the real-world applicability of the results.
2) While the paper introduces the concept of profitability, it remains unclear how this differs substantively from previous metrics like the number of target-arm pulls or success probability used in prior attack works; the novelty seems mostly terminological.
3) The theoretical analysis appears to be a direct extension of prior work (Jun et al. 2018; Liu & Shroff 2019), with limited new technical ideas. The proofs mostly adapt existing inequalities and regret bounds to the delayed setting. 
4) Proof rigor requires clarification. For example, in Eq. (8), it is not clear how the bound behaves when $\epsilon < \eta_0$; the assumption in Eq. (21) is not well justified in Appendix A.
5) No analysis of attack cost is provided, which is crucial for evaluating the effectiveness of any attack model.

### Questions
You state that “we derive an upper and lower bound on the number of pulls of the target arm,” but this is somewhat confusing. According to Lemma 4.3, the so-called “lower bound” applies only to the specific oracle attack you analyze. However, in the standard sense, a lower bound should hold for any attack algorithm, which would then justify a claim of optimality. Could you clarify how the claimed optimality—“our bounds are tight up to a sublinear factor”—is established? As it stands, the oracle attack does not appear to be a truly optimal strategy, especially since attack cost is not modeled, making it unclear in what sense “optimal” is defined.

### Soundness
2

### Presentation
2

### Contribution
2
