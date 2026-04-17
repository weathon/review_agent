# The Cost of Delegation

- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
We study reinforcement learning and alignment through the lens of hierarchical coordination, where a principal steers many delegates with partial views and coupled effects. Starting from nonlinear dynamics, we identify the Cost of Delegation as the performance gap between centralized and decentralized control, decomposed into delegation, coordination, information, and surrogate mismatch components. We bound CoD, show that information value is decision-theoretic, and discuss implications for modern systems. Our work provides a theoretical foundation and new perspective for designing robust, scalable multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies hierarchical delegation in multi-agent control systems. It formalizes how a principal can guide multiple delegates under partial observability, analyzes the resulting Nash equilibrium, and quantifies the approximation error (termed “cost of delegation”) arising from quadratic surrogates of the full nonlinear dynamics. The framework distinguishes between epistemic (learning) errors and persistent structural errors, providing bounds and design principles to reduce the latter.

### Strengths
Rigorous formal treatment connecting multi-agent delegation, surrogate LQ games, and closed-loop stability.

Clear separation of epistemic versus persistent errors.

Provides practical guidance for designing hierarchical control systems.

### Weaknesses
The title “cost of delegation” is potentially misleading, conflating LQ surrogate approximation error with intrinsic delegation inefficiency. 

Heavy notation and long derivations reduce accessibility.

Lacks numerical or empirical examples to illustrate the bounds in practice.

### Questions
Can you explicitly explain the relationship between LQ approximation error and delegation inefficiency, e.g., due to partial observability or divergence between delegate and principal objectives? Are you assuming that under quadratic approximation the delegation inefficiency vanishes (even if the delegates are misaligned or adversarial), and thus the divergence from LQ approximation can indeed bound the cost of delegation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a theoretical framework to explain when delegation enhances or harms organizational performance, highlighting its dependence on information asymmetry and preference alignment. It models the trade-off between improved information acquisition by agents (when decisions are delegated) and the loss of centralized control by principals. The authors find that delegation can be costly when agents’ preferences diverge from the principal’s, leading to inefficiencies, but beneficial when it motivates better information gathering or reduces communication frictions.

### Strengths
1. The paper offers a clean, well-structured model that clearly formalizes the trade-off between control and information in delegation decisions.
2. It advances understanding of when delegation improves or harms efficiency, providing a foundation for later empirical and behavioral studies.

### Weaknesses
1. The model relies on strong assumptions of linear assumptions, full information, and simple preference structures.
2. The paper does not test its theoretical predictions with data or case studies, limiting practical relevance.
3. The paper has a discussion of its connection with modern systems like RLHF, multi-agent systems, and MoE training, but does not disclose detailed guidance on that.

### Questions
1. Can the authors raise some examples on how the cost of delegation affect current modern systems?
2. How might the main results change if the system dynamics were highly nonlinear or involved deep-learning systems?

### Soundness
2

### Presentation
3

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
The paper views multi-agent reinforcement learning and alignment through the lens of hierarchical coordination, where a principal directs multiple partially informed and interdependent delegates. Building on first principles with certainty-equivalence and local linearization, it develops a linear–quadratic surrogate in which the delegates’ interactions form a quadratic game whose Nash equilibrium determines overall behavior, yielding an explicit equilibrium map. The authors study three coordination structures (low-rank plus sparse, tree/DAG, and block-sparse), showing how each yields tractable solvers and predictable computational scaling. Finally, it formalizes the Cost of Delegation (CoD) as a persistent, non-vanishing penalty that decomposes into a cubic value-function remainder and a quadratic dynamics remainder, which distinguishes it from epistemic error.

### Strengths
1. The paper provides a clear framing of alignment as hierarchical coordination between a principal and many coupled delegates.
2. The paper give clean first-principles derivation of an LQ surrogate and explicit equilibrium map with uniqueness under standard conditions.
3. The paper novelly introduces the cost of delegation, which is a persistent penalty decomposed into a cubic value-function remainder and a quadratic dynamics remainder, distinct from epistemic error.

### Weaknesses
1. While I find the results very interesting, the writing is not strong. At present, the paper reads like a list of theorems, definitions, and assumptions. Adding more discussion and explanatory text would make it much more readable. The paper also needs a more detailed introduction and a more comprehensive discussion of related work.

2. As noted in the limitations paragraph, the analyzed games are quadratic surrogate games, which are far removed from modern AI systems.

3. See questions.

### Questions
1. In Theorem 3.1, shouldn’t it be $W_{mj} = (\Pi_m B)^\top Q_m (\Pi_m B)$ instead?

2. In line 360, the paper states that the effective dimension is (O(rMd + s)), whereas line 729 in the appendix gives (O(rMd + sM)). Could the authors clarify this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper presents a framework for understanding multi-agent reinforcement learning and alignment through hierarchical coordination, where a principal steers multiple delegates with partial observability. The core contribution is formalizing and quantifying the Cost of Delegation - an irreducible performance penalty inherent to indirect control.

### Strengths
1. Applicability - Despite being highly theoretical, the paper connects well to practical systems (RLHF, Constitutional AI, MoE architectures). The design principles in Section 5.2 offer actionable insights for system designer. I feel that there is likely some nuance in translating the theoretical nuance of this paper to modern systems, but nevertheless, its theoretical framing of hierarchical delegation costs is interesting and novel.
2. Theory - The analysis of computational complexity under different coordination structures (low-rank+sparse, tree/DAG, block-sparse) is thorough and provides practical guidance for scalable implementations.

### Weaknesses
1. No Empirical Investigation - As an ICLR submission, I would assume the paper has some experiments that demonstrate practical applicability. I feel the paper could be strengthened with some application to MoEs.

### Questions
In a scenario where we have several experts over several layers, how does the cost of delegation scale?
- are there ways of bounding cross-entropy loss according to the CoD framework?

### Soundness
3

### Presentation
3

### Contribution
3
