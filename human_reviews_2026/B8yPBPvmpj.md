# An Axiomatic Framework for N-Agent Ad Hoc Teamwork: From Shapley Axioms to Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Open multi-agent systems are increasingly relevant for modelling the emerging real-world domains such as smart grids and swarm robotics. This paper addresses the recently posed problem of n-agent ad hoc teamwork (NAHT), where only a subset of agents is controllable. We propose an axiomatic game-theoretic framework for the NAHT, formulated via a cooperative game model which differentiates between the learning objectives of NAHT and MARL. Within this framework, the axiomatic characterization of the Shapley value—Efficiency, Symmetry, and Linearity—is reinterpreted as structural constraints on individual value functions. This yields a principled design space: enforcing all axioms recovers the Shapley value, while dropping Efficiency yields the Banzhaf index, leading to our Banzhaf Machine variant. As concrete instantiations, we develop Shapley Machine and Banzhaf Machine, which enforce different subsets of axioms during learning. Implemented on IPPO and POAM, these algorithms provide stronger performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces an axiomatic framework for ad hoc teamwork (NAHT), a setting where the payoff structure of the task is cooperative but only a subset of the other agents are controllable. By enforcing individual constraints on the value functions of each agent, the authors are able to induce the Shapley axioms of efficiency, linearity and symmetry. At the same time, this paper explores the results of enforcing only the symmetry and linearity constraints which they term as Banzhaf machine.

### Strengths
Overall the mathematical axioms and propositions seem sound and correct to me, in such way the theoretical foundations of the paper are very solid. The main strengths of this paper in my opinion are:

1. **Theoretical contributions:** The paper cleanly maps Shapley axioms to training constraints (reward shaping + symmetry + TTD($\lambda$)), with a formal statement that enforcing all three recovers Shapley values in dynamic settings (Theorem 4.10).

2. **Empirical gains on strong Benchmarks:** The proposed  Banzhaf (BM) and Shapley (SM) machines show improved returns on two difficult tasks (MPE and SMAC) and across a variety of multi-agent reinforcement learning algorithms.

3. **Reproducibility details:** Hyperparameters ($\lambda$, m, $\alpha$, loss weights), episode lengths, and compute budgets are reported.

### Weaknesses
The main weakness is that the paper is very difficult to understand, building very little on intuitions and quickly skimming over the required background. As someone who is not an expert in this field, I had to re-read the paper several times before getting a notion of what the authors where trying to do and why. More specifically the following are my issues with this submission:

1. **Problem motivation**: The authors make no statement about the motivation of the problem or what they are attempting to achieve concretely. Although previous approaches rely on heuristics, and this paper claims to construct a theoretical grounding for ad hoc teamwork (NAHT), it is unclear what the purpose of constructing a theoretical grounding is. A theoretical grounding for its own sake is deeply un-satisfying. 

2. **Novelty**: Similar frameworks that decompose value functions according to the Shapley value have extensively been used in the literature with similarly strong experimental results. 

3. **Practical Utility**: The paper does show better performance of applying the proposed Banzhaf (BM) and Shapley (SM) machines in two tasks: MPE and SMAC. However it is unclear what the computational overhead of the proposed methodologies is, thus, the practical tradeoffs of this method are not clearly stated. The paper also notes stability issues for SM on larger maps, an issue that is not investigated and might constitute a major limitation of SM.

Another minor issue is that the authors claim in Figure 2 that t the number of n-step return components in TTD($\lambda$) can be determined by the number of non-empty basis games. But this is not clear to me from interpreting the results: in 5v6 and 8v9, there are no clear differences between the performance of different values of $m$ and in MPE-pp it is actually the value of $m=14$ the one with highest performance.

### Questions
My main questions for the authors are as follows:

1. What was the orginial motivation for constructinn this axiomatic framework? What is one practical advantage of this framework over the existing heuristical methods?
2. When should a practitioner prefer SM vs BM, what are the properties of the task where one machine may be preferrable over the other?
3. What have been your observations about the stability of these methods in practice? In particular, related to SM, what seem to be the environments where the instability becomes apparent and what might be causing it?

### Soundness
3

### Presentation
2

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
This paper introduces an **axiomatic framework for N-Agent Ad Hoc Teamwork (NAHT)** grounded in Shapley’s cooperative game theory. By reinterpreting three axioms as structural constraints on NAHT learning, the authors derive two theoretically principled algorithms, Shapley Machine (SM) and Banzhaf Machine(BM), which were evaluated on MPE-PP and SMAC in NAHT setting. The results demonstrate that these methods perform well. Furthermore, the experiments indicate that relaxing the Efficiency axiom can even lead to better agent type generalization than enforcing the full set of Shapley axioms. But the proposed methods do not explicitly address the challenges specific for NAHT.

### Strengths
1.  The paper achieves a theoretically rigorous fusion of NAHT and cooperative game theory, providing a clear axiomatic basis for agent value decomposition.
2. The experiments are well designed and cover multiple NAHT scenarios, demonstrating the practical feasibility and stability of both SM and BM.
3. The framework is internally self-consistent, with well-motivated propositions that link the axioms to learning method in a clean way.

### Weaknesses
1.  The formal axioms rely on strong assumptions such as additive reward decomposition and state-wise independence, which are unlikely to strictly hold in real NAHT environments.
2. Despite extensive proofs, many theoretical results merely reinterpret Shapley axioms to constrain the learning structure, without introducing fundamentally new optimization principles specific to NAHT. The proposed methods do not explicitly address the defining challenges of NAHT, such as teammate uncertainty, heterogeneity, adaptive coordination and so on. Thus limiting the framework’s task-specific effectiveness.

### Questions
**Q1.** How does the proposed axiomatic framework ensure robustness when agents operate in highly uncertain or partially observable environments?
Since the axioms (Efficiency, Symmetry, Linearity) are defined under idealized cooperative settings, it remains unclear how the framework maintains stable performance when the underlying dynamics or teammate behaviors are stochastic or adversarial.

**Q2.** Could the authors provide a more intuitive interpretation of how the proposed axiomatic structure concretely addresses the specific challenges of NAHT? A higher-level conceptual or visual explanation could help clarify why enforcing Shapley-inspired constraints improves the perfoemance specific in NAHT setting, rather than a generic framework suitable for both AHT and MARL.

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
3

### Summary
The paper proposes an algorithmic framework for a recently introduced multi-agent systems model of n-player ad hoc teamwork. The main results are Theorem 4.8 (a sufficient condition for resulting value functions) and Theorem 4.10 (saying that a particular algorithm recovers Shapley axioms). Furthermore, the paper presents simulations.

### Strengths
The paper works on building algorithms that construct cooperative game theory concepts in their algorithms, which is an interesting question.

### Weaknesses
1. The theory does (for my taste) not sufficiently motivate why the paper considers the questions it considers in the NAHT setting. This can be seen in the proof of one of the main results, Theorem 4.10, which relies on reducing the game's properties to the state conditioned game's properties. Why does the method not work more broadly? 
2. The writing of the paper is not clear enough yet for publication in a top venue. Proofs are informal. The important role of reduction of the state-conditioned game makes me believe that the paper is not using the appropriate scope for the generality of its method.
3. Many of the experiments don't reach statistical significance. Some are hard to read. For example, Figures 2 and 3, which are the main training curves, are too small for me to evaluate which methods perform best.

### Questions
- How is your analysis specific to NAHT? Can your method be applied more generally?
- Which of your simulation results are statistically significant?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes an axiomatic view of n‑agent ad hoc teamwork (NAHT). Instead of plugging the closed‑form Shapley value into MARL, it models each state as a cooperative game and enforces Shapley’s axioms—Linearity, Symmetry, Efficiency—as structural constraints on per‑agent value functions. This yields two learning templates: Shapley Machine (enforces all three axioms) and Banzhaf Machine (drops Efficiency, aligning with the Banzhaf index). A key insight is that Linearity maps directly to truncated TD(λ) with positive weights over n‑step returns; the number of return components is tied to the number of non‑empty coalitions. Built on IPPO and POAM, these variants generally improve training performance; notably, relaxing Efficiency can generalize better to unseen teammate types. Experiments on MPE‑PP and SMAC support these claims.

### Strengths
1. Recasting Shapley’s axioms as trainable constraints gives a principled design space: enforcing all axioms recovers Shapley; dropping Efficiency yields Banzhaf. This connects cooperative game theory to critic shaping in a way that’s easy to implement.
1. The paper shows Linearity to truncated TD(λ) with strictly positive weights, offering a game‑theoretic reading of λ‑return weighting. This is both novel and practically actionable.
1. Comprehensive experimental design. 
1. Clear training losses ablations on m and λ, and full hyperparameters make the work easy to replicate.

### Weaknesses
1. Several technical steps rely on superadditivity and variance/auto‑covariance assumptions (e.g., Lemma C.1, Assumption C.2) whose realism in complex NAHT environments is not empirically validated.
1. Implementing Efficiency requires accurate per‑agent value estimates for uncontrolled teammates, approximated via shared critics and a tunable coefficient α to stabilize learning; this may introduce bias and add a non‑axiomatic knob.
1. Evaluations focus on MPE‑PP/SMAC; no continuous‑control or real‑robot domains and limited comparison to the most recent open‑team frameworks beyond IPPO/POAM baselines—so external validity is still uncertain.

### Questions
How do you handle the uncertainty/bias in value estimation, especially for uncontrolled teammates?

### Soundness
3

### Presentation
3

### Contribution
3
