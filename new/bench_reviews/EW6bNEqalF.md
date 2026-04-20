Now let me do calibration searches to anchor the score.Now I have sufficient information to write the consolidated review.

---

## Summary
This paper introduces two techniques for offline RL in Regular Decision Processes (RDPs): (1) a novel language metric $L_\mathcal{X}$ grounded in formal language theory, which defines distinguishability via probabilities of language families rather than individual traces; and (2) a Count-Min-Sketch (CMS) approach for compact storage of empirical probability distributions. The key theoretical result (Theorem 1) proves that $L_\infty^p$-distinguishability decays exponentially with corridor length $N$ while $L_{\mathcal{X}_{2,1}}$-distinguishability remains $\Omega(1)$ in T-maze RDPs; Theorem 3 translates this into a PAC sample complexity bound that replaces the $H$-dependent factor of the prior RegORL bound with $\log|\mathcal{X}|$. Experiments confirm that the language metric approach achieves better policy quality and scales linearly (vs. exponentially for CMS) in the T-maze, and outperforms FlexFringe in structurally complex domains.

---

## Strengths

- **Exponential distinguishability separation (Theorem 1 + Example 4):** Formally proves that $L_\infty^p$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$ in the T-maze family. The construction is explicit and directly motivates the algorithmic contribution.

- **Principled unifying framework (Definition 2):** The language metric generalizes all prior metrics ($L_\infty$, $L_1$, $L_\infty^p$, $L_1^p$) as special cases by choice of language family $\mathcal{X}$. The two-dimensional hierarchy $\mathcal{X}_{i,j}$, connected to the dot-depth hierarchy of star-free regular languages (Simon, 1972; Pin, 2017), gives the approach theoretically justified foundations.

- **PAC sample complexity bound (Theorem 3):** Establishes that when $j$ is a small constant, $\log|\mathcal{X}_{i,j}| = \tilde{\mathcal{O}}(1)$, removing the horizon dependence present in RegORL. Combined with Theorem 1's larger $\mu_0$, the net improvement can be exponential in $H$.

- **Empirical validation of scaling behavior (Figure 2):** Shows CMS time and RDP state count grow exponentially with corridor length while the language metric scales linearly in both, on a log scale. Table 1 shows the language metric achieves optimal reward in T-maze(c) (4.0 vs. 0.0 for FlexFringe) and best reward in Cheese (0.87) and Mini-hall (0.86), consistent with the theoretical claims.

- **Correction of prior work bug:** Identifies a mistake in Cipollone et al. (2023), with both the prior and proposed bounds carrying an additional $\sqrt{H}/\mu_0$ term. Transparency about this correction is a scientific contribution in its own right.

---

## Weaknesses

### Fatal
None.

### Major

- **The headline "sample efficiency" claim is not directly empirically validated.** The paper's central contribution is a provably better sample complexity (Theorem 3), yet all experiments fix the dataset size at $K = 100$ episodes and report policy quality and running time. There is no experiment varying $K$ and measuring the minimum number of episodes needed to achieve $\varepsilon$-optimal behavior, which is precisely what sample complexity quantifies. The paper's title is "Sample Efficiency via Language Metrics," but Figure 2 measures computational scaling (time vs. corridor length), not statistical scaling (reward vs. episodes). A natural and necessary validation would plot reward vs. $K$ for both the language metric and CMS/RegORL at several values of corridor length $N$, showing the claimed polynomial vs. exponential divergence. Without this, the theoretical advance remains unlinked to empirical reality.

- **CMS contribution is materially overstated in framing.** Theorem 2 shows CMS achieves the **same** asymptotic sample complexity as RegORL (corrected), while only reducing memory. In Section 5, the paper explicitly states: "the statistical test still has to iterate over all suffixes, which is exponential in $H$" and CMS exceeds the 1800-second budget for Mini-hall. Table 1 confirms CMS underperforms the language metric in policy quality across nearly all domains. Yet the abstract and contributions section present CMS as a co-equal original technique. The contribution would be more honestly framed as a secondary theoretical result demonstrating that memory savings are achievable without sacrificing sample complexity, not as an algorithmic advance of equivalent standing to the language metric.

### Minor

- **The bug correction in Cipollone et al. (2023) is handled in a single sentence** (Section 4.2: "Our new analysis uncovered a mistake in the proof of Cipollone et al. (2023), and as a result, both their and our sample complexity has an additional multiplicative term $\sqrt{H}/\mu_0$"). Since this correction changes the comparative picture between RegORL and the proposed methods—and constitutes a non-trivial claim about a published result—it deserves a clear statement of which lemma was incorrect and what the corrected proof changes.

- **No ablation over the hierarchy parameters $i$ and $j$.** All experiments use $\mathcal{X}_{3,1}$ without justification, sensitivity analysis, or guidance on how to select these parameters in practice. Since the sample complexity in Theorem 3 depends on $\log|\mathcal{X}_{i,j}|$ (which grows with $j$) and $\mu_0$ (which can increase with $i$ and $j$), these effects trade off and the optimal choice is environment-dependent. This is acknowledged nowhere in the paper.

- **The interaction between $d_m^*$ and the claimed exponential gain deserves analysis.** The paper acknowledges $1/d_m^*$ can be exponential in $H$ when states are hard to reach. In the T-maze, states $u_{i,g}$ in the corridor are reachable with probability $\sim 2^{-N}$ at time $t=N$ under a uniform observation policy—which could mean $1/d_m^*$ is exponential in $N$, potentially canceling the exponential improvement in $1/\mu_0$ claimed in Example 4. The paper notes this possibility but dismisses it without analysis. Addressing this specifically for the T-maze would strengthen confidence in the net theoretical gain.

### Trivial
None (formatting artifacts are parser issues, not author errors).

---

## Nice-to-Haves

- **Reward vs. episodes plot** for T-maze at several corridor lengths $N$: This would be the most direct empirical validation of the sample efficiency improvement and would significantly strengthen the paper.

- **Adaptive hierarchy selection:** The paper fixes $i=3, j=1$ in all experiments. An algorithm that starts at $j=1$ and increases as needed (when distinguishability is insufficient) would make the approach more practical and remove the need for manual tuning.

- **Detailed discussion of when CMS is useful:** Given that CMS has the same sample complexity as the original RegORL but exponential time complexity, it would help to articulate the conditions under which CMS offers a net advantage (e.g., memory-constrained environments where time is not the bottleneck).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Notational collision ("parameterised by $j$... and by $j$")** (Harsh Critic, Section 4.1): The extracted text reads "parameterised by $j$ for the granularity of the atomic symbols, and by $j$ for the sequential composition," but reading the formal definition $\mathcal{X}_{i,j}$ it is clear the hierarchy is indexed by $i$ for granularity and $j$ for sequential composition. This is a PDF-to-text parser artifact replacing "$i$" with "$j$" — the original submission does not have this error. Removed per the formatting artifact rule.

- **Generic strength: "T-maze running example makes formal language theory accessible"** (Strength Finder): This is a generic presentation praise without a specific figure/table citation that advances a core claim. Dropped.

- **FlexFringe comparison as a weakness** (Harsh Critic): The critic argues that when FlexFringe produces a smaller but lower-quality automaton, this is due to FlexFringe's heuristics underfitting, not a flaw in the proposed method. This is correct: the asymmetry (worse performance despite larger automata) cannot be attributed to the language metric approach and does not weaken its claims. Removed.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the **interplay between two independent sources of exponential improvement** in Theorem 3: (1) replacing the $H$-dependent suffix count with $\log|\mathcal{X}_{i,j}| = \tilde{\mathcal{O}}(1)$ when $j$ is constant, and (2) achieving an exponentially larger $\mu_0$ for structured RDPs (Theorem 1). While the paper mentions both effects, it does not combine them into a single explicit statement of the compound gain. At the same time, the $d_m^*$ factor—potentially exponential in $H$ for hard-to-reach states—acts as a potential counterweight that the paper leaves unanalyzed, leaving the net theoretical improvement in concrete problem families (including T-maze) unresolved at a formal level.

---

## Suggestions

1. Add a "reward vs. number of training episodes" experiment in the T-maze (varying $K$ at several corridor lengths $N$) as the direct empirical test of the sample efficiency claim.
2. Reframe the CMS contribution as "a theoretical complement showing memory savings are possible at no sample complexity cost," not a co-equal algorithm.
3. Add a precise statement of the bug in Cipollone et al. (2023) — which lemma, which step, and the corrected inequality — in Section 4.2 or Appendix C.
4. Include an ablation table varying $j \in \{1, 2, 3\}$ in one domain to quantify sensitivity to the hierarchy parameter choice.
5. Explicitly compute or bound $d_m^*$ for the T-maze as a function of $N$ to verify that the net improvement (after accounting for occupancy) is indeed polynomial.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| JSS9rKHySk | Offline RL, general function approximation, novel theoretical analysis | 8,8,6,6,6 (~6.8) | Accept (Spotlight) |
| 1hsVvgW0rU | POMDP sample complexity, new subclasses with PAC bounds | 6,6,6,6 (6.0) | Accept (Poster) |
| GrRo9uV3OH | RL sample complexity theory, very limited experiments | 3,5,5,3 (4.0) | Reject |
| Eqps25f8HU | Model-based offline RL, limited novelty, no empirical validation | 4.5 avg | Reject |

**Assessment:** This paper sits above GrRo9uV3OH and Eqps25f8HU (rejected, avg ~4) because its theoretical contribution is more novel — the language metric framework and exponential distinguishability separation are genuine advances, not incremental modifications. It is comparable to 1hsVvgW0rU (accepted poster, avg 6): both contribute novel theoretical ideas to RL in partially-observable settings with PAC bounds, both have limited but consistent experiments. The key gap relative to JSS9rKHySk (spotlight, ~6.8) is that the central sample efficiency claim is not directly validated experimentally, and the CMS contribution is overstated. The paper's experimental story (Figure 2) is consistent with the theory but tests running time/automaton size rather than the sample complexity axis. For a theory paper at ICLR, the theoretical novelty and soundness (Theorems 1 and 3, the unifying framework) are strong enough to merit acceptance, but the mismatch between the headline claim and the experiments keeps this at poster rather than spotlight level.

**Score: 6.0 — Marginally above acceptance threshold (poster-level)**

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>