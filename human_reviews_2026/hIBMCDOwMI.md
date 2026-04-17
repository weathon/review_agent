# Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Well-trained multi-agent systems can fail when deployed in real-world environments due to model mismatches between the training and deployment environments, caused by environment uncertainties including noise or adversarial attacks. Distributionally Robust Markov Games (DRMGs) enhance system resilience by optimizing for worst-case performance over a defined set of environmental uncertainties. However, current methods are limited by their dependence on simulators or large offline datasets, which are often unavailable. This paper pioneers the study of online learning in DRMGs, where agents learn directly from environmental interactions without prior data. We introduce the Multiplayer Optimistic Robust Nash Value Iteration (MORNAVI) algorithm and provide the first provable guarantees for this setting. Our theoretical analysis demonstrates that the algorithm achieves low regret and efficiently finds the optimal robust policy for uncertainty sets measured by Total Variation divergence and Kullback-Leibler divergence. These results establish a new, practical path toward developing truly robust multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper propose a systematic and principled approach for online learning in distributionally robust Markov Games with environment uncertainty. This is done by first revealing the hardness in online DRMGs,  then proposing an online robust MARL algorithm, Multiplayer Optimistic Robust Nash Value Iteration (MORNAVI), which offers the first provable robust guarantee in the setting. The theoretical analysis demonstrates MORNAVI achieves low regret, finds the optimal robust policy and achieves high sample complexity.

### Strengths
Overall the paper is well-written, a little hard to understand due to the nature of a theoretical paper but clear enough to understand given the context. The theories given are rigorious and the threat model, algorithm and proofs seems correct, but I didn't check all the proofs.

1. The hardness of online DRMGs provides a clear motivation of the problem, and the contructed examples gives an intuitive thought experiment for future researchers to design their algorithms.

2. The derivations on sample complexity provides a rigorious theoretical foundation for establishing the regret of existing algorithms, and shows the result achieves complexity comparable with existing works, despite operating in online settings instead of easier generative and offline settings. The complexity bound is quite tight.

### Weaknesses
1. The main weakness I belive is that current online MARL parameterized by neural networks still do not have high enough sample efficiency to support this work that learns purely online, without any offline datasets available. However, accepting this paper would greatly benefit future research on this topic.

2. The paper is purely theoretical and might be designed for tabular case, instead of existing MARL using function approximations. While this is beneficial for theoretical analysis, it remains unknown how to adapt this algorithm to modern environments, such as MPE, SMAC and Multi-Agent Mujoco. This is clearly reflected in stage 1, Nominal Transition Estimation and stage 2, EQUILIBRIUM subroutine. Estimating the distribution of future states can be inaccurate in modern environments with large state space using simple empirical average, and computing the equilibrium such as CE or CCE can be hard in tasks such as SMAC, or continuous control.

3. What is the algorithmic differencce between MORNAVI and existing distributional robust approach? I am not very familiar with distributionally robust MARL literature, so I assume the main difference lies in Eqn. 5 and 6, since other parts of the algorithm is not too different from existing robust MARL approach.

To solve these problems, can you provide suggestions on empirical algorithms that leverage the advantage of MORNAVI, but use  function approximations? this would greatly strengthen this paper. However, I give a rating of 6 based on the current version.

### Questions
See weakness.

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
2

### Summary
This paper tackles online learning in Distributionally Robust Markov Games (DRMGs) and proposes f-MORNAVI—a model-based algorithm that learns empirical transition models, constructs optimistic and pessimistic value estimates under f-divergence uncertainty, and computes equilibria (Nash/CCE/CE) at each step. The authors prove hardness results, showing that support-shift uncertainty (e.g., TV sets) yields linear regret. They also derive upper bounds for total variation and KL uncertainty, establishing near-matching sample-efficient guarantees. The work provides the first theoretical framework for online DRMGs with rigorous proofs and detailed analysis.

### Strengths
- The paper supplies both lower bounds (separations) and matching upper bounds (for TV and KL) together. The proof structure is standard but carefully adapted to the robust multi-agent setting. 
- The paper identifies and formalizes the online DRMG problem (vs. prior offline/generative-model work) and isolates two distinct hardness phenomena (support shift and curse-of-multi-agency).

### Weaknesses
- All upper bounds and the lower bounds include the product of agent action counts. This is a severe scalability concern (exponential in number of agents if each has many actions). The paper acknowledges this as an open question but does not give practical guidance or alleviate it. This limits real-world applicability.
- The algorithm requires solving an equilibrium (Nash/CE/CCE) in the stagewise matrix game for each state and timestep. In practice large action space and many states make these subroutines expensive. The paper needs to discuss practical computational approaches and complexity per episode.
- The KL regret/sample complexity contains an $exp(O(H^2))$ term. For long horizons this is prohibitive; more discussion of whether this dependence is inherent (and how it scales in practice) is needed.

### Questions
- Lack of empirical validation despite practical motivation: The paper is motivated by bridging the sim-to-real gap in MARL, yet includes no experiments or case studies. This omission significantly weakens the claim that the approach improves practical robustness or connects theory and real-world performance. Would you be able to empirically show your proposed method compared with previous work in several simulation benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces f-MORNAVI, a model-based online algorithm for distributionally robust Markov games (DRMGs). The method estimates the dynamics from interaction with the environment and performs planning under uncertainty sets defined by f-divergences. It incorporates an equilibrium solver and provides regret bounds for online DRMGs, along with lower bounds that highlight inherent hardness.

### Strengths
+ The theoretical development is clear, with high-probability regret bounds for TV and KL uncertainty sets and sample-complexity corollaries to equilibrium under the NE, CE, and CCE.

+ The algorithmic design well-designed and motivated. It separates model estimation, robust optimistic planning with divergence-aware bonuses, and an equilibrium, and the mathematical treatment of support shift is interesting and well-written.

### Weaknesses
-- The paper lacks empirical validation. Although the theoretical results looks sound, there is a lack of experimental evidence that the proposed online method outperforms prior approaches or that the constants or overheads are practical. 

-- The practical comparison to generative or offline baselines and to out-of-distribution scenarios is unclear. It would help to quantify how the robust online procedure fares against strong non-robust or offline/generative methods on OOD tasks.

-- While the theory derives sample-complexity corollaries for reaching approximate equilibrium, the lack of experiments makes it hard to assess the real-world gap between these bounds.

### Questions
see the weaknesses section

### Soundness
2

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
4

### Summary
The paper tackles online distributionally robust Markov games (DRMGs) without a simulator or offline dataset and proposes f-MORNAVI, an optimistic-robust, model-based meta-algorithm for general f-divergence uncertainty sets (with concrete TV and KL instantiations). It proves (i) hardness results—linear regret with support shift and √K regret without shift but still scaling with the joint action size—and (ii) first regret bounds for online DRMGs in the TV/KL cases, plus corresponding sample-complexity bounds. Algorithmically, f-MORNAVI estimates a nominal kernel online, plans via robust Bellman operators augmented with UCB-style bonuses tailored to the uncertainty geometry, and computes an equilibrium (NE/CE/CCE) at each step.

### Strengths
Originality

The paper addresses a relatively unexplored problem: online distributionally robust Markov games (DRMGs) without access to simulators or offline data. While the formulation itself extends concepts familiar from single-agent robust RL and generative/offline DRMG studies, applying them to the online multi-agent regime is a natural but nontrivial step. The proposed MORNAVI framework—integrating optimism for exploration with robustness against uncertainty—is conceptually consistent with prior work on optimistic robust RL, though its multi-agent adaptation and generalization to 
𝑓
f-divergence sets give it modest originality. The contribution is incremental rather than groundbreaking but helps close an existing theoretical gap.

Quality

The theoretical development is careful and technically competent. The paper provides hardness results, upper bounds, and sample-complexity analysis that align with known results in related literature. The regret guarantees for both Total Variation and KL uncertainty sets are plausible extensions of existing robust RL theory. However, the analysis follows established proof techniques (empirical Bernstein inequalities, Bellman contraction arguments, etc.) rather than introducing fundamentally new analytical tools. Some aspects—such as the correctness of the correlated equilibrium definitions and the reliance on specific assumptions like rectangularity and failure states—should be clarified or corrected to fully validate the results. Overall, the quality is solid but not exceptional.

Clarity

The paper’s structure is standard and logical, with a clear flow from problem motivation to algorithm design and theoretical results. Nevertheless, the writing is mathematically dense, and several sections would benefit from higher-level intuition to guide readers through technical derivations. Certain notational inconsistencies (e.g., repeated theorem numbering, unclear equilibrium notation) reduce readability. While the main ideas can be followed by experts in the field, the exposition is unlikely to be easily accessible to a broader ICLR audience without additional clarifications or illustrative examples.

Significance

The significance of the paper lies primarily in its problem setting rather than in the methodological innovation. Establishing regret bounds for online DRMGs is useful for the theory of robust multi-agent learning, but practical impact remains limited given the heavy dependence on joint action space size and the lack of empirical validation. The work reinforces the difficulty of scaling robustness in multi-agent systems but does not yet provide clear strategies to mitigate these challenges. The theoretical results are incremental but may serve as a foundation for future improvements in scalability or algorithmic design.

### Weaknesses
1. Limited Conceptual Novelty Beyond Extension
While the paper presents a rigorous treatment of online Distributionally Robust Markov Games (DRMGs), its conceptual novelty is limited. The proposed MORNAVI algorithm largely repackages existing principles—namely optimism in exploration and robust Bellman operators—previously developed in single-agent robust RL (e.g., Wang & Zou, NeurIPS 2021; Dong et al., ICML 2022; Panaganti & Kalathil, ICML 2022). Extending these to the multi-agent setting is a logical next step but not a fundamentally new paradigm. The paper would benefit from a clearer articulation of what new technical difficulties arise in the multi-agent online case (beyond joint-action explosion) and how MORNAVI specifically overcomes them.

2. Overstated Claims of Firstness and Theoretical Gap
The paper repeatedly claims to be the first to address online DRMGs with provable guarantees, yet concurrent and closely related works—such as RONAVI (Farhat et al., 2025, arXiv)—already study similar formulations with optimism–robustness integration and provide comparable regret bounds. Moreover, previous generative or oracle-based DRMG works (Shi et al., 2024; Ma et al., 2023) have already laid much of the theoretical groundwork. The authors should moderate their 'first provable guarantee' claim and explicitly position their contribution in relation to these concurrent developments.

3. Incomplete or Inaccurate Definitions
There are important definitional inaccuracies that need correction before the theory can be considered reliable:
- The robust coarse correlated equilibrium (CCE) definition in Section 2 is identical to the robust Nash equilibrium definition, which is incorrect.
- The paper assumes existence of robust NE without proof. For general-sum DRMGs, NE existence is nontrivial; only CE/CCE are guaranteed.
Correcting these definitions and clarifying the associated assumptions would strengthen the theoretical soundness and interpretability of the results.

4. Dependence on Restrictive Assumptions
Several assumptions used to make the analysis tractable are strong and may limit practical relevance:
- The rectangular uncertainty set assumption eliminates coupling across states and agents, simplifying proofs but overlooking realistic uncertainty.
- The failure-states assumption for TV uncertainty ensures that unseen transitions are learnable, which may not hold in online exploration.
- The algorithm presupposes centralized model updates and full observability.
A section explicitly acknowledging these assumptions’ implications—and discussing potential relaxation strategies—would improve credibility.

5. Lack of Empirical or Illustrative Validation
Although the paper is theoretical, it would benefit from a minimal empirical illustration or simulation. A simple 2-player gridworld or coordination game with environmental noise could demonstrate how MORNAVI behaves in practice, whether the derived regret bounds are observable, and how robustness manifests under distribution shifts.

6. Unresolved Scalability Challenge
While the authors discuss the curse of multi-agency (joint action dependence), the paper stops short of offering even partial mitigation strategies. Theoretical exploration of structured policies (mean-field approximations, factored models, or correlated policies) could point toward reducing the exponential dependence on ∏Ai.

7. Exposition and Structural Issues
The exposition can be improved in several ways:
- Duplicate theorem numbering causes confusion.
- Several notations (e.g., σ_𝒫[V], ρ_min, P_min) are undefined when first introduced.
- The confidence interval construction is only in the appendix and should be summarized in the main text.
Clarifying these would improve readability and allow reviewers to verify correctness more easily.

Summary of Actionable Suggestions
1. Correct the CCE/CE definitions and restate all regret results accordingly.
2. Clearly differentiate the paper from concurrent work like RONAVI and moderate novelty claims.
3. Discuss the impact and realism of rectangular and failure-state assumptions.
4. Include a small synthetic experiment to illustrate algorithm behavior.
5. Explore scalability strategies to reduce dependence on joint action size.
6. Fix theorem numbering, ensure consistent notation, and summarize key proof steps in the main text.

### Questions
1. Clarification of the Equilibrium Definitions
The definition of robust coarse correlated equilibrium (CCE) in Section 2 appears identical to the Nash equilibrium condition. Could the authors clarify whether this was intentional, and if not, provide the correct formulation of the CCE obedience constraints? If the CCE definition is corrected, would this change any of the stated regret bounds or equilibrium existence claims? A clarification of whether the theoretical guarantees hold for all equilibrium notions (NE, CE, CCE) under the same assumptions would be very helpful.

2. Existence of Robust NE and Practical Computability
The paper defines a robust NE as a product policy, but general-sum robust games may not guarantee existence. Are there known conditions under which the robust NE considered in this paper is guaranteed to exist (e.g., convex–concave payoff structures or zero-sum cases)? How is the equilibrium computed in practice within the MORNAVI framework—via an exact solver or approximate methods? Including an explanation of computational feasibility would make the algorithmic contribution clearer.

3. Clarification of the Failure-State Assumption
The regret bound under TV divergence relies on the failure-state assumption, which seems to restrict uncertainty to transitions that are still reachable through exploration. Could the authors formalize this assumption more explicitly and discuss its implications? What happens if this assumption is violated—does the regret bound degrade gracefully, or does the algorithm fail entirely? A sensitivity analysis or a theoretical relaxation would strengthen the argument.

4. Scope of the Rectangular Uncertainty Set
The analysis assumes rectangular (decoupled) uncertainty sets across states and agents, which simplifies the dynamic programming recursion but limits expressiveness. Could the authors comment on whether their approach could handle non-rectangular (coupled) uncertainty sets, perhaps through approximate decomposition? Would any part of the regret proof break down under correlated uncertainties across agents?

5. Regret Bound Tightness and Scaling with Joint Actions
The regret bounds in Theorems 2 and 3 scale with the product of action space sizes (∏Ai), which makes them impractical for even moderate numbers of agents. Could the authors clarify whether this dependence is inherent or a proof artifact? Are there potential structural assumptions (e.g., mean-field or factored game structures) that could reduce this dependence while maintaining robustness?

6. Comparisons with Concurrent Work
The authors position MORNAVI as the first to provide online DRMG guarantees, but RONAVI (Farhat et al., 2025) and related works appear to address similar settings. Could the authors provide a more explicit comparison in terms of assumptions (e.g., oracle access, divergence type), theoretical guarantees, and computational complexity? If RONAVI uses similar optimism–robustness design principles, what distinguishes MORNAVI’s theoretical contribution?

7. Confidence Interval Construction and Proof Transparency
The proof of optimism for the robust Q-value relies on confidence intervals that bound the uncertainty-adjusted Bellman operator. Could the authors sketch the key steps or inequalities (e.g., dual form of σ𝒫[V]) in the main text to make the logic more transparent? How do the bonus terms differ in structure between the TV and KL cases, and what intuition explains these differences?

8. Empirical or Illustrative Demonstration
Even a small-scale experiment could provide insight into how the algorithm performs under model mismatch. Could the authors include or discuss results on a simple two-player coordination or adversarial environment? Observing whether the empirical regret trend aligns with the theoretical rate would help substantiate the practical value of the theoretical development.

9. Theoretical Open Questions
The paper concludes by raising the question of whether online DRMG algorithms can overcome the curse of multi-agency. Could the authors elaborate on potential directions—such as hierarchical decomposition, correlated equilibrium relaxation, or partial coordination—that might reduce this scaling? Are there theoretical obstacles (e.g., impossibility results) suggesting that sublinear regret without ∏Ai dependence might be unattainable?

10. Expository and Structural Improvements
There are a few presentation issues that would benefit from revision:
- Duplicate numbering of Theorem 1 for hardness and upper-bound results.
- Undefined symbols (e.g., σ𝒫[V], ρmin, Pmin) at their first appearance.
- Several long equations could use short textual interpretation lines to help readers follow the logic.
Addressing these would significantly improve readability and make the theoretical arguments easier to follow.

Summary
The main clarifications that could substantially change my evaluation are:
- Correcting and explaining the CCE/CE definitions and their impact on results.
- Providing clearer justification for key assumptions (failure states, rectangularity).
- Offering an explicit comparison with concurrent works.
- Demonstrating even minimal empirical validation or illustrating scalability considerations.
These improvements would make the contribution more transparent, the assumptions more credible, and the theoretical results easier to interpret and verify.

### Soundness
3

### Presentation
3

### Contribution
3
