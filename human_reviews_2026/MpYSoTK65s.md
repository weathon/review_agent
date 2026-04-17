# Enforcing Axioms for AI Alignment under Loss-Based Rules

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
Recent alignment methods for large language models, most notably reinforcement learning from human feedback (RLHF), often train an auxiliary reward model to minimize a loss function on binary preference data over model responses. We study a theoretical setting inspired by principle-guided methods such as Constitutional AI, in which a small set of principles (e.g., helpfulness, toxicity) act as “voters” that guide binary comparisons---such as preferring the less toxic response. We model these principles as linear directions in an embedding space of responses, a simplifying assumption motivated by the Linear Representation Hypothesis---concepts are linear directions in representation-space---a useful first-order approximation in practice.
In this \emph{linear social choice model}, Ge et al. (2024) showed that an optimal linear reward model can violate Pareto optimality (PO): From the principles-as-voters lens, this means a response A can be less helpful and more toxic than B, yet still receive a higher reward. We analyze axiomatic violations in the linear social choice setting and probe the robustness of negative results under realistic assumptions. We show that added expressivity does not resolve the issue: polynomial reward models can still fail PO. We then offer a pragmatic alternative showing that when the data uniformly covers the embedding space, broad classes of loss-based rules in the limit exactly recover the axiomatic guarantees. This yields a recipe for constitutional-style alignment with provable guarantees: enforce balanced coverage \emph{via dataset design} to restore axiomatic guarantees without abandoning standard training pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies axiomatic violations in AI alignment, specifically how loss-based reward models can violate Pareto optimality even when all principles agree on preferences. The authors show polynomial rewards don't resolve this issue, but prove that uniform data coverage can restore axiomatic guarantees in constitutional-style alignment methods.

### Strengths
1. The paper provides rigorous mathematical analysis of axiomatic violations in AI alignment, extending the linear social choice framework with formal proofs and theorems.
2. The paper proves that even polynomial reward functions (beyond linear) still fail Pareto optimality, strengthening the robustness of the violation result.
3. The paper provides intuitive examples and simplified cases that clearly illustrate why Pareto optimality violations occur

### Weaknesses
Only examines Pareto optimality while other important social choice axioms (e.g., Pairwise Majority Consistency) receive limited attention.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes axiomatic guarantees (Pareto Optimality) for preference-based alignment under a loss-minimization view. It (i) shows PO violations persist even when moving beyond linear rewards to bounded-degree polynomial rewards; (ii) gives intuition via minimal counterexamples; and (iii) proves that in an idealized setting with uniform directional coverage of pairwise data, broad classes of loss-based rules do satisfy PO—suggesting a data-centric recipe (balanced coverage) for constitutional-style alignment without changing training pipelines.

### Strengths
1.  Crisp negative/positive results. Clear impossibility for polynomial rewards and a clean, sufficient condition (uniform coverage) restoring PO.
2.  Training-pipeline relevance. Positions guarantees within standard loss-based training, aligning with RLHF/RLAIF practice rather than bespoke voting rules.
3.  Intuition + formality. Minimal counterexamples illuminate why PO fails; proofs formalize the phenomenon.
4.  Actionable takeaway. Suggests data-centric interventions (balancing directions/lengths) as a path to axioms in practice

### Weaknesses
1.  Idealized coverage assumption. The uniform directional coverage condition is strong; guidance on how much imbalance is tolerable is missing (no finite-sample or robustness bounds).
2.  Empirical gap. No experiments (even synthetic) to quantify coverage diagnostics (e.g., spectrum of pairwise direction differences) or to validate the recipe on real preference data.
3.  Scope of axioms. Focuses on PO; discussion of other axioms (e.g., Pairwise Majority Consistency) is limited to related work, with no analogous guarantees.
4.  Practical measurement. The paper hints at PCA/geometry checks but lacks a concrete coverage diagnostic or reweighting algorithm practitioners can run.

### Questions
1.  Coverage diagnostics. How should practitioners measure directional coverage in real datasets (e.g., a concrete statistic or test), and what thresholds approximate the theorem’s regime?
2.  Finite-sample guarantees. Can you provide sample-complexity or robustness bounds showing how PO violation probability decays as coverage improves?
3.  Beyond PO. Do analogous results (negative or positive) hold for other alignment-relevant axioms such as Pairwise Majority Consistency under similar coverage assumptions?
4.  Practical recipe. Can you outline a reweighting/sampling procedure (pseudocode) that moves an arbitrary dataset toward the desired coverage while preserving data efficiency?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper take the impossibility result from Ge et al. (2024) regarding Pareto optimality of reward model training, and (i) generalize it to impossibility for polynomial hypothesis classes (instead of linear ones); (ii) show that the impossibility dissolves when we have preference samples that covers all directions of the embedding spaces uniformly and that the between-response distance in each sample is constant. By doing so, it argues that (i) the impossibility is real and not solved by simply adding representational capacity, yet (ii) can be resolved in practice via data distribution interventions.

### Strengths
- Significance: Ge et al. (2024) is a seminal work on an important topic (pluralistic alignment). The work addresses the most important shortcoming of Ge et al., which is its practical implications for today's model training pipelines. It shows what the data distribution needs to be like for the impossibility result to vanish, and the resulting suggestions (uniformly distributed embedding directions & constant embedding distances for preference pairs) seem highly actionable.
- Soundness: I followed the proofs in the body and spot no errors (except possible typo that I mentioned in the questions section). The "claims"/lemmas involved in the proofs seem generally easy to give proof sketches to, so I did not verify their proofs in the appendix.
- Presentation: The authors went with the simplest structure (first background, then present results in logical succession), and it worked well. The message is also clear.

### Weaknesses
- It is a pity, although definitely not a fatal flaw, that the paper gives no empirical validation on language models despite its aim at guiding practice. It would be helpful to see, e.g., comparisons of more uniform vs less uniform sampling methods for preference data collection, in terms of impact on downstream alignment performance against multiple constitutional principles.

**Minor points (don't affect score):**
- Presentation: Illlustrations would make following the proofs easier. For the proof of Thm 4.1, an illustration of the parallel lines + candidate points + voter vectors will be useful (same goes for Sec 3). For the proof of Thm 5.1, an illustration of a half-plane (i.e. $\\{x:\\langle x,\\theta\\rangle>0\\}$) gradually rotating towards the intersection $D$ of the voter half-planes will be useful.
- One unmentioned related work is [1] which found circularity in the hypothesis class's privilege graph as a condition for the impossibility. This may give a high-level reason why the impossibility continues to hold for the polynomial hypothesis class.

[1] Representative Social Choice: From Learning Theory to AI Alignment

### Questions
- Potential typo: In the proof of Thm 4.1, on line 298, should it be $j=\\lceil\\frac{i}d\\rceil$ instead of $j=(i-1)\\bmod d$?
- What do you think are direct implications for data collection practices in preference alignment / constitutional AI? Are there ways you can quickly test them?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates when loss-based training rules, such as RLHF, satisfy fundamental social-choice axioms, focusing on Pareto Optimality (PO). The authors work in the linear social-choice model of Ge et al. (2024). They extend the analysis to from linear to polynomial reward functions and prove that PO violations persist for any polynomial, establishing a general impossibility. Finally, they take a data-centric perspective: when pairwise-comparison data uniformly cover the embedding space, they prove that PO is satisfied. This yields a practical takeaway, axiomatic guarantees can be restored not by changing loss functions but by ensuring balanced sampling or reweighting of preference data.

### Strengths
The main strength of this paper lies in its two key theoretical results. 

Theorem 4.1 extends the findings of Ge et al. (2024) from linear to polynomial reward functions, showing that Pareto Optimality (PO) remains violated even when polynomial reward models are used. 

Theorem 5.1 demonstrates that under idealized conditions, specifically, unit-length representations and uniformly distributed directions, a linear reward model can in fact satisfy PO.

### Weaknesses
The main limitation of this paper lies in its significance comparing to previous results and its practical implications. First, several closely related works are not discussed [1,2,3,4]. As noted in [1,2], the nonparametric global solution of the reward-training loss does satisfy Pareto Optimality (PO). In practice, reward models are typically large language models (LLMs) implemented with transformer architectures and thus possess strong function-approximation capabilities. Consequently, practical reward models are much closer to expressive models than to the simplified linear or polynomial formulations considered in this paper. Hence, the extension from linear to polynomial reward functions has limited practical significance.

Second, Theorem 5.1 attempts to provide a positive result by showing that PO can be satisfied under certain conditions. However, the assumption of a uniform distribution is highly idealized and far from practical scenarios. Moreover, existing work do have some more practical positive results. They already indicate that when the reward model has sufficient approximation capacity, PO can in fact be satisfied.

Line 106: The statement “NLHF coincides with maximal lotteries” is missing an appropriate citation. The current reference (Fishburn, 1984) pertains to the definition of maximal lotteries, not to the connection between NLHF and maximal lotteries. Reference [3] establishes this equivalence and should be cited here. In addition, [4] applies these axioms to propose a new alignment approach. 

Lastly, the English writing and representation should be improved. The current manuscript does not fully adhere to the ICLR style guidelines, and the use of fonts and formal structure is inconsistent with the official template. 

[1] Ritesh Noothigattu, Dominik Peters, and Ariel D Procaccia. Axioms for learning from pairwise comparisons. Advances in Neural Information Processing Systems, 33:17745–17754, 2020.

[2] Jiancong Xiao, Zhekun Shi, Kaizhao Liu, Qi Long, Weijie J Su. Theoretical Tensions in RLHF: Reconciling Empirical Success with Inconsistencies in Social Choice Theory, 2025.

[3] Roberto-Rafael Maura-Rivero, Marc Lanctot, Francesco Visin, and Kate Larson. Jackpot! Alignment as a maximal lottery. arXiv preprint arXiv:2501.19266, 2025.

[4] K Kim, J Zhang, A Ozdaglar, PA Parrilo Population-Proportional Preference Learning from Human Feedback: An Axiomatic Approach, 2025.

### Questions
see weakness

### Soundness
2

### Presentation
1

### Contribution
2
