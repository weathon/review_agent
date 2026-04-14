## Summary

This paper studies offline reinforcement learning in episodic Regular Decision Processes (RDPs), where environment dynamics are governed by a hidden finite-state automaton. Building on the ADACT-H framework (Cipollone et al., 2023), the authors propose two contributions: (1) a novel *language metric* $L_X$ grounded in the dot-depth hierarchy of formal language theory, which replaces the $L_\infty^p$ suffix-distance test with a structured family of language-membership events $\mathcal{X}_{i,j}$, and (2) a Count-Min-Sketch (CMS) approach to reduce the memory burden of storing empirical suffix distributions. Theorem 1 establishes an exponential separation between $L_\infty^\ell$-distinguishability and $L_{\mathcal{X}_{2,1}}$-distinguishability on the T-maze family, while Theorems 2 and 3 provide PAC sample complexity bounds for each variant. Experiments on five classical POMDP/RDP benchmarks support the theoretical results.

---

## Strengths

- **Formally grounded and genuinely novel language metric.** The paper introduces Definition 2 (language metric) as a unifying framework that captures $L_\infty$, $L_1$, and prefix distances as special cases of $L_X(p, p') = \max_{X \in \mathcal{X}} |p(X) - p'(X)|$. The two-dimensional hierarchy $\mathcal{X}_{i,j}$ built from basic patterns $\mathcal{G}_i$ and the $C_k^\ell$ operator is a principled and original construction, not a heuristic. This connection between the dot-depth hierarchy of formal languages and state-distinguishability in RDP learning is, to our knowledge, new.

- **Rigorous exponential separation (Theorem 1).** The T-maze family provides a concrete and complete proof that $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$. This is not just an illustration — it is a theorem establishing that sample complexity of prior methods can be exponential in horizon $H$ on a natural class of instances. The mechanism is transparent (the distinguishing event is "probability of positive reward upon North," a single-language membership check).

- **Identification and correction of a proof error in prior work.** The paper discovers an error in the proof of Cipollone et al. (2023), leading to a corrected bound with an additional $\sqrt{H}/\mu_0$ factor affecting both the prior and the present work. Surfacing and fixing such errors is a substantive scientific contribution.

- **Figure 2 provides clean, theorem-consistent empirical validation on T-maze scalability.** The linear vs. exponential scaling of running time and RDP size as corridor length increases is exactly what the theory predicts, and the experiment is quantitatively informative (log-scale plots, 20 runs, up to $N=100$).

- **Five-domain evaluation with a competitive external baseline.** The comparison against FlexFringe—a state-of-the-art algorithm for learning probabilistic-deterministic finite automata—is meaningful because FlexFringe uses domain-agnostic heuristics and sometimes uses cycles (which is favorable to FlexFringe), making the comparison conservative with respect to the proposed method.

---

## Weaknesses

### Fatal
None.

### Major

- **Headline sample efficiency claim is not empirically validated.** The paper's title and abstract center on "sample efficiency via language metrics," and Theorem 3 gives a bound that is polynomially better in $\log|\mathcal{X}|$ when $j$ is constant. However, *no experiment varies dataset size* to show that the language metric achieves better policy quality with fewer episodes. All experimental comparisons are at fixed dataset sizes. This is the single largest gap: the central theoretical contribution is not directly tested empirically, and without learning curves over $|\mathcal{D}|$ it is impossible to assess whether the sample complexity improvement is behaviorally meaningful in the tested domains.

- **No direct comparison to ADACT-H or RegORL.** The paper explicitly frames its contribution as improving over RegORL (Cipollone et al., 2023), yet RegORL/ADACT-H (exact, without CMS) does not appear as a baseline in any table or figure. The CMS variant partially fills this role (since it uses the same $L_\infty^p$ test), but the paper should either include the exact ADACT-H where feasible or explicitly argue why CMS is the appropriate stand-in. Without this, the claimed improvement over the precise algorithm being compared to is unverifiable experimentally.

- **No ablation over language hierarchy parameters $(i, j)$.** The hierarchy $\mathcal{X}_{i,j}$ is a key contribution, but only $\mathcal{X}_{3,1}$ is evaluated across all domains. The sensitivity of results to $i$ and $j$ is entirely uncharacterized. Without this, it is impossible to assess whether $\mathcal{X}_{3,1}$ is genuinely appropriate or whether it was selected post hoc, and whether the hierarchy design is well-motivated for domains other than T-maze.

- **Model selection for $(i, j)$ is unresolved and practically critical.** Assumption 1 requires that the behavior policy ensures $L_{\mathcal{X}_{i,j}}$-distinguishability $\geq \mu_0 > 0$ for a *known* $\mathcal{X}_{i,j}$ that is an input to the algorithm. In practice, the learner does not know which $(i,j)$ yields non-trivial distinguishability without oracle knowledge of the RDP. No heuristic, model selection procedure, or sensitivity analysis is provided. This is a genuine practical limitation that the paper should at minimum characterize empirically (e.g., what happens when the wrong $(i,j)$ is chosen on T-maze?).

### Minor

- **The $1/d_m^*$ term can itself be exponentially large in $H$, partially undermining the "overcoming exponential dependence" narrative.** The paper acknowledges this ("The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach") and discusses it after Theorem 3, but this acknowledgment is buried. The narrative in the abstract and introduction emphasizes avoiding exponential dependence on $H$, while the bound retains this exponential dependence through $d_m^*$. A more calibrated framing in the abstract and introduction would better reflect the scope of the improvement.

- **Notation imprecision in the estimator definition.** The paper defines $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$ (Section 4.1), but $\mathcal{X}_{i,j}$ is a *set of languages*, not a single language. The indicator $\mathbb{I}(e \in \mathcal{X}_{i,j})$ does not type-check: $e$ is a trace, $\mathcal{X}_{i,j}$ is a collection of languages. Presumably one computes $\hat{p}_1(X) = \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$ for each $X \in \mathcal{X}_{i,j}$, then maximizes. Since Theorem 3's threshold involves $|\mathcal{X}|$ via a union bound, this is not cosmetic.

- **Hierarchy definition contains a clear typographic error.** Section 4.1 reads: "parameterised by $j$ for the granularity of the atomic symbols, and by $j$ for the sequential composition." One dimension should be $i$. In a section whose contribution hinges on a two-dimensional hierarchy, this confuses the exposition.

- **The corrected proof error produces a bound that is strictly weaker than originally claimed in Cipollone et al. (2023)**, with an additional $\sqrt{H}/\mu_0$ factor in both bounds. The paper does not analyze how this additional factor changes the comparison with the lower bound from the prior work (which depends on $L_1$-distinguishability). The current gap between the upper and lower bounds is wider than previously thought.

### Tiny

- **Notation inconsistency between $L_\infty^\circ$ (used in Section 2.2's distinguishability definition) and $L_\infty^p$ (used in Section 3 and Theorem 2).** These appear to be the same metric under different notation; the paper should reconcile them.

- **The $\varepsilon$-optimality target in Section 2.3** ("finding $\hat{\pi}$ satisfying $V_0^*(h) - V_{\hat{\pi}}^*(h) \leq \varepsilon$ for each $h \in \mathcal{H}_0$") is stated as a per-history guarantee, which is stronger than the expectation-over-$h_0$ definition given earlier in the section. The paper should clarify whether the theorems prove the weaker or stronger form.

---

## Nice-to-Haves

- **Computational complexity of the language-metric test.** The paper emphasizes "tractability" but only formalizes sample complexity. A rough analysis of the per-iteration cost of evaluating $L_X$ in terms of $|\mathcal{X}_{i,j}|$, episode length, and dataset size would make the tractability claim more complete, especially for large action/observation spaces where $|\mathcal{G}_i|$ grows.

- **Characterization of RDP classes well-served by low-$(i,j)$ families.** Theorem 1 establishes the exponential gain for T-maze with $\mathcal{X}_{2,1}$. A broader characterization of *which* RDP structures or behavior policies lead to a bounded $j$ being sufficient would significantly strengthen the paper's scope claim and help practitioners choose $(i,j)$.

- **Evaluation on a domain where $L_X \approx L_\infty^p$ (no structure to exploit).** Showing that the language metric does not regress relative to CMS in unstructured domains would demonstrate robustness and complete the empirical picture.

- **Provide per-domain $\mu_0$ values** under both $L_\infty^p$ and $L_{\mathcal{X}_{3,1}}$, and $d_m^*$ estimates, to ground the theoretical bounds with concrete numbers rather than asymptotic statements alone.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED — parser artifact] Example 2 notation $C_1^0(\mathcal{G})$ vs. 10-letter strings.** The critic flags inconsistency between $C_1^0$ and strings of length 10. Given the plain-text rendering artifacts visible throughout the document (e.g., malformed figure captions), the superscript is almost certainly a rendering error (likely $C_1^{10}$). This is not a paper error.

- **[REMOVED — scope creep] Missing comparison to sequence-model-based offline RL or latent-state RL.** These approaches are mentioned in the introduction but lack sample complexity guarantees for the RDP class; demanding engagement with the dominant empirical paradigm goes beyond this paper's scope.

- **[REMOVED — scope creep] Insufficient engagement with POMDP/PSR literature at a technical level.** The related work discusses PSRs and POMDPs and explains why their bounds don't directly apply. Demanding further technical development of these connections exceeds the paper's stated scope.

- **[REMOVED — non-standard expectation] Demand for formal runtime complexity theorem.** The paper demonstrates tractability empirically (Figure 2) and via informal analysis. Requiring a formal complexity theorem for a systems/algorithmic paper is non-standard at ICLR when runtime is demonstrated empirically.

- **[REMOVED — unfair baseline criticism] FlexFringe comparison disadvantages the proposed method.** FlexFringe uses cycles and performance-optimizing heuristics not available to ADACT-H variants, making the comparison asymmetrically harder for the proposed method. Criticisms that the comparison is "not directly comparable" because FlexFringe sometimes outperforms are not valid weaknesses—this makes the proposed method's wins more impressive and its losses expected.

- **[REMOVED — factual misread] CMS width parameter $w = \lceil \varepsilon/\delta_c \rceil$ claimed unusual.** The critic compares to standard CMS parameterizations using different variable conventions. The paper's CMS description is internally consistent and the proof of Theorem 2 depends on it; without access to the appendix proof, there is no basis to claim incorrectness.

- **[REMOVED — generic] "The paper should more explicitly situate itself relative to mainstream offline RL concerns (coverage mismatch, pessimism)."** The paper adopts the concentrability framework standard in offline RL. Demanding broader situating in offline RL is generic advice not specific to this paper's weaknesses.

---

## Novel Insights

The most significant novel insight—which neither the authors nor reviewers fully develop—is the observation that the language metric $L_X$ implicitly performs *soft aggregation over exponentially many suffix events* by collapsing them onto a polynomial-cardinality family of language-membership events. This is why the testing complexity drops from $O((AOR)^H)$ to $O(|\mathcal{G}_i|^j)$: the language hierarchy replaces exact suffix probabilities with marginal event probabilities, trading discriminative power for statistical efficiency. Whether this aggregation principle can be made adaptive (choosing $(i,j)$ based on data without oracle knowledge) seems like the key open problem, and connects to model selection in automata learning more broadly. A data-driven variant—e.g., incrementally increasing $j$ until the distinguishability test stabilizes—would resolve the main practical limitation and would be a natural extension worth highlighting explicitly.

---

## Suggestions

1. **Add sample efficiency experiments**: Plot policy quality (reward) vs. dataset size $|\mathcal{D}|$ for at least T-maze and one other domain, for both CMS and language metric variants. This is the most direct test of the paper's central claim and is entirely absent.

2. **Add ADACT-H (exact) as a baseline** where computationally feasible (short horizons), or explicitly argue in the text why CMS is the appropriate computational proxy for the original method.

3. **Run ablation over $(i,j)$**: At minimum, test $j \in \{1, 2\}$ on T-maze and Cheese; report reward, automaton size, and runtime. Include a case where a suboptimal $(i,j)$ is chosen (e.g., $\mathcal{X}_{1,1}$ on T-maze) to characterize sensitivity.

4. **Provide a heuristic or discussion for model selection of $(i,j)$**: Even an informal procedure (e.g., validate on a held-out portion of $\mathcal{D}$, or increase $j$ until learned automaton size stabilizes) would substantially improve practical applicability.

5. **Fix the estimator notation**: Define per-language estimators $\hat{p}_1(X) = \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$ for each $X \in \mathcal{X}_{i,j}$, then state $L_X(\hat{p}_1, \hat{p}_2) = \max_{X \in \mathcal{X}_{i,j}} |\hat{p}_1(X) - \hat{p}_2(X)|$.

6. **Recalibrate the abstract and introduction** to clarify that the exponential improvement is (a) instance-dependent on favorable language structure and (b) present when $1/d_m^*$ does not itself dominate.

---

**Evaluation along key axes:**

- **Novelty**: High. The language metric and its connection to the dot-depth hierarchy is a genuinely original idea; the unification of $L_\infty$ and $L_1$ metrics as special cases of $L_X$ is elegant and previously unexplored in the RDP learning context.
- **Technical soundness**: Good. The theoretical framework is well-constructed; the corrected bounds are credible; notation issues and the estimator imprecision are refinements rather than errors.
- **Empirical support**: Moderate. The T-maze scaling experiment is convincing and theorem-consistent. However, the absence of sample efficiency experiments—the paper's core claim—and the lack of ablation over the hierarchy leave the empirical case incomplete.
- **Significance**: Moderate-to-high for the RDP/non-Markovian RL subfield. The language metric idea is likely to influence follow-up work; the paper moves RDP learning meaningfully closer to practical applicability for structured domains.
- **Clarity**: Moderate. The mathematical development is generally rigorous and well-structured; the motivating examples are effective. Several notation inconsistencies and the two typographic errors in the central definitions detract from the exposition of the paper's most important sections.