Now let me look at the calibration papers to score the paper:Now I have enough information to write the consolidated review. Let me verify one specific claim from the harsh critic about the estimator notation.---

## Summary

This paper studies offline Reinforcement Learning in Regular Decision Processes (RDPs) and proposes two algorithmic improvements over the prior state-of-the-art RegORL. First, a novel **language metric** $L_X$ grounded in the dot-depth hierarchy of formal languages is introduced, replacing the $L_\infty^p$-based statistical test in state-merging algorithms and provably yielding exponentially better distinguishability on structured instances (e.g., T-maze). Second, a **Count-Min-Sketch (CMS)** variant alleviates memory requirements without changing statistical complexity. PAC sample complexity bounds are derived for both methods, and experiments across five POMDP/RDP domains validate the computational properties of the language metric.

---

## Strengths

- **Compelling separation theorem (Theorem 1):** The paper rigorously proves that in the T-maze family, $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$, demonstrating an exponential gain. This is a concrete and non-trivial result that motivates the entire framework.

- **Elegant unifying formalism:** Definition 2 unifies $L_1$, $L_\infty$, $L_1^p$, and $L_\infty^p$ within a single language-metric framework. The two-dimensional hierarchy $\mathcal{X}_{i,j}$ (varying atomic pattern granularity and sequential composition depth) is principled and provides a well-motivated interpolation between statistical power and computational cost.

- **Honest disclosure of corrected prior work:** The paper identifies a mistake in Cipollone et al. (2023) introducing an additional $\sqrt{H}/\mu_0$ factor in sample complexity, affecting both old and new bounds (Section 4.2). This is a valuable correction and reflects scientific integrity.

- **Experiments demonstrating computational advantage:** Figure 2 clearly shows exponential vs. linear scaling of runtime and automaton size in the T-maze as corridor length increases, directly validating the theoretical prediction. Table 1 shows the language metric consistently produces smaller automata and better policies than both CMS and FlexFringe on the harder domains.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Abstract and introduction overstate the theoretical result.** The abstract says "our algorithm is proven to be more sample efficient than existing results," but Theorem 3 replaces $\mu_0^{-2}$ (where $\mu_0$ is $L_\infty^p$-distinguishability) with $\mu_0^{-2} \log |\mathcal{X}|$ (where $\mu_0$ is $L_X$-distinguishability). This is a **parameter substitution**, not a uniform dominance result. The improvement is genuine in structured instances (T-maze), but there is no theorem stating that $L_X$-sample complexity is generally smaller than RegORL's corrected complexity. The claim should be scoped to "on instances admitting low-complexity languages."

- **No direct comparison with RegORL in experiments.** The paper's entire motivation is improving upon RegORL/ADACT-H, yet the sole empirical baseline is FlexFringe—a different algorithm class without the same theoretical guarantees. Comparing runtime, automaton size, and reward against ADACT-H using $L_\infty^p$ under matched datasets is essential to empirically validate the claimed improvement.

- **No sample-complexity study.** The core theoretical contribution concerns PAC sample complexity, yet all experiments use a fixed $K = 100$ episodes. There is no curve of policy quality or model recovery accuracy as a function of dataset size, making it impossible to empirically verify the exponential improvement in $H$ that the paper claims. This is the most important missing experiment.

- **The $\sqrt{H}/\mu_0$ correction is under-integrated.** The paper mentions at line 214 that a mistake in Cipollone et al. (2023) was uncovered, adding a $\sqrt{H}/\mu_0$ multiplicative factor to both old and new bounds. However, the paper never explicitly states what the mistake was, whether the corrected RegORL bound is still dominated by the new bound in the T-maze instance, or how this propagates through the comparative claims in the introduction. This correction directly affects the credibility of the headline comparison and needs explicit treatment.

- **Assumption 1 is not operationalizable.** The algorithm requires $\mathcal{X}_{i,j}$ as input and assumes $L_{\mathcal{X}_{i,j}}$-distinguishability $\mu_0 > 0$, but the paper provides no principle for selecting $(i,j)$ from data. In all experiments, $\mathcal{X}_{3,1}$ is used without justification. Since the improvement depends entirely on choosing a language class that increases distinguishability, the absence of a model-selection procedure leaves a significant gap between theory and deployable algorithm.

### Minor

- **The $1/d_m^*$ term can be exponentially large.** The paper notes briefly (after Theorem 3) that $1/d_m^*$ "depends exponentially on $H$ if there exists an RDP state that is very hard to reach" and that it "can be much smaller for structured RDPs." However, no characterization is given of which RDPs avoid this exponential dependence, and no analysis shows whether the T-maze improvement in $1/\mu_0$ is not cancelled by a large $1/d_m^*$. This deserves more than a brief remark.

- **CMS is strictly worse empirically but receives equal billing.** CMS failed to complete on Mini-hall, was substantially slower in all domains, and produced larger automata than the language metric in most cases (Table 1). The paper acknowledges this but does not demote CMS's billing accordingly. Clarifying that CMS's contribution is primarily memory-theoretic—with no practical performance advantage over the language metric—would sharpen the narrative.

- **Estimator notation in Section 4.1 is ambiguous.** The estimator $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$ appears to test trace membership in a set of languages rather than in a specific language $X \in \mathcal{X}_{i,j}$. Theorem 3's statistical test is correctly stated as $L_X(\mathcal{Z}_1, \mathcal{Z}_2) = \max_{X \in \mathcal{X}} |\hat{p}_1(X) - \hat{p}_2(X)|$, so the issue is presentation: the estimator should be defined per-language $X$.

- **No ablation over hierarchy parameters $(i, j)$.** Only $\mathcal{X}_{3,1}$ is evaluated. Given the two-dimensional hierarchy is the paper's central construction, understanding how performance and automaton quality change with $(i, j)$ is necessary to characterize the hierarchy's behavior.

### Trivial

- The paper's claim that the methods "can be directly applied to any algorithm that performs such statistical tests" (Section 3) is supported only for ADACT-H and ADACT-H-A. The scope of applicability to other state-merging algorithms should be stated more carefully.

---

## Nice-to-Haves

- **Experiments in regimes where the language metric offers no advantage** (i.e., where $L_X$-distinguishability ≈ $L_\infty^p$-distinguishability). This would establish that the method degrades gracefully rather than catastrophically when the structural assumption fails.

- **An adaptive hierarchy-selection heuristic**, e.g., starting from $\mathcal{X}_{1,1}$ and increasing $(i,j)$ until the test detects a split. Even a heuristic without full theoretical backing would substantially improve usability.

- **Explicit runtime analysis of the $L_X$ test.** The paper claims tractability versus $L_\infty^p$ due to $|\mathcal{X}_{i,j}| = \mathcal{O}((AOR)^j)$, but this can still be large for broader action/observation spaces. A brief complexity statement would clarify the regime where the language metric is practically tractable.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic — Comparison with deep RL baselines:** Removed. This is a PAC-theoretic paper for structured non-Markovian environments; deep RL (CQL, IQL, etc.) operates under entirely different assumptions and provides no formal guarantees in the RDP setting. Requesting such a comparison is scope creep.

- **Spark — "Scaling experiments beyond H≈15":** Removed. The paper explicitly states at line 257: "We also extend this experiment by increasing N up to 100, observing the same trend for our language-based approach." The concern is already addressed in the paper.

- **Harsh Critic — Notation inconsistencies in Definition 1/Example 2:** Removed. The reviewer notes these are "likely extraction artifacts." The definition and example are internally consistent when read carefully (strings of length 10 using $C_1^0$ with an alphabet of 3 letters produce 10-letter strings; the subscript convention is $C_k^\ell$ where $\ell$ is the total length and $k$ the number of inserted patterns). No genuine error.

- **Neutral Reviewer — CMS computational complexity not analyzed:** Removed as a standalone weakness. The paper's CMS contribution is primarily about memory (the statistical test is unchanged from $L_\infty^p$), and the memory reduction is analyzed via the CMS literature. The runtime performance is accurately reported in experiments.

- **Neutral/Harsh — Notation heavy, high barrier to entry:** Removed as a pure style concern. The formalism is standard in the automata-learning literature and required to state the theorems precisely.

---

## Novel Insights

The most genuinely novel observation in this work—and one that goes beyond simply replacing one test with another—is that the *structure of the sample space itself* (traces as strings over AO/R) can be exploited to define a metric hierarchy tied to the complexity of regular languages. The dot-depth hierarchy, a classical construct in formal language theory entirely outside the RL literature, turns out to be precisely the right tool to interpolate between the statistically powerful but computationally intractable $L_1$ and the tractable but statistically weak $L_\infty$. The T-maze construction (Theorem 1) is not just a toy counterexample; it reveals that the fundamental obstacle in horizon-dependent sample complexity is not the length of traces per se, but the *granularity* of the test events used to distinguish distributions over traces. This insight—that patterns (languages) rather than individual strings are the right unit of comparison for state-merging in RDPs—is likely to be transferable to other automata-learning and sequence-model-learning contexts.

---

## Suggestions

1. **Explicitly show the corrected RegORL complexity** side-by-side with Theorem 3 in the main text. Compute both bounds for the T-maze instance and verify numerically that the new bound is smaller after the correction.
2. **Add an experiment varying dataset size** (e.g., $K \in \{20, 50, 100, 200, 500\}$ episodes) in the T-maze and at least one other domain, plotting reward or state-recovery accuracy vs. $K$. This is the most impactful missing experiment.
3. **Compare against ADACT-H with $L_\infty^p$** (the direct predecessor) on the same datasets. Even a single domain (T-maze) at matched $K$ would substantiate the sample efficiency claim empirically.
4. **Scope the abstract:** Replace "our algorithm is proven to be more sample efficient than existing results" with language acknowledging the instance-specific nature of the improvement.
5. **Discuss the relationship between $\mu_0$ and $d_m^*$** in the T-maze: show explicitly that $1/d_m^*$ does not grow exponentially in this instance so that the exponential improvement is not spurious.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores | Reason for Comparison |
|---|---|---|---|
| `GnOLWS4Llt.md` | Accept (poster) | 5,5,5 | Offline RL with observation histories; similar setting, weaker theory |
| `1hsVvgW0rU.md` | Accept (poster) | 6,6,6,6 | PAC learning in POMDPs, comparable theoretical style |
| `txD9llAYn9.md` | Accept (poster) | 6,8,8,6 | Horizon-free bounds in RL; comparable theory quality |
| `CIcMuee69B.md` | Reject | 6,5,5,3,3 | Automata learning paper, weaker contribution, rejected |
| `9pW2J49flQ.md` | Accept (oral) | 8,8,8,8 | LTL+RL with strong experiments; stronger overall contribution |

**Assessment:** The paper's core theoretical idea—the language metric and Theorem 1—is more novel and concrete than `GnOLWS4Llt.md` (5,5,5) and comparable in depth to `1hsVvgW0rU.md` (6,6,6,6). The formal construction is clean, the separation theorem is compelling, and the identification of the corrected prior-work bound is a meaningful contribution. However, the experimental gaps (no RegORL comparison, no sample complexity curves) and the under-discussed correction to prior work place it below the cleaner papers at score 6. The major weaknesses are real and revision-relevant, but they are not fundamental flaws that invalidate the contributions. The FUNDAMENTAL ISSUES override is not triggered.

**Final score: 5.5** — Borderline, leaning toward weak accept pending the missing experiments and revised framing. This is above `GnOLWS4Llt.md` (poor experiments, 5,5,5) and below the cleanest posters at 6,6,6,6 given the uncorrected experimental and framing gaps.

**Originality:** High — the application of the dot-depth hierarchy to RDP learning is novel.
**Importance:** Moderate — addresses a concrete bottleneck in a well-defined sub-problem; primarily relevant to the automata-theoretic RL community.
**Claim support:** Moderate — Theorem 1 is well-supported; the overall "more sample efficient" headline claim is overstated.
**Experimental soundness:** Weak — demonstrates computational advantages but not statistical efficiency.
**Clarity:** Good — well-written with clear notation; some under-specification in estimator definitions.
**Community value:** Moderate-to-high — the language-metric formalism is likely to inform future work on non-Markovian RL.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>