## Summary

This paper studies offline Reinforcement Learning in episodic Regular Decision Processes (RDPs) — non-Markovian environments whose hidden dynamics are captured by a finite-state automaton. The core contribution is a novel **language metric** $L_X$ grounded in formal language theory that replaces the $L_\infty^p$-based statistical test in the prior ADACT-H/RegORL framework, yielding PAC sample complexity bounds that scale with $\log|\mathcal{X}|$ rather than exponentially with the episode length $H$ for structured instances. A secondary contribution applies Count-Min-Sketch (CMS) to reduce the memory footprint of the original suffix-counting approach. Experiments across five domains demonstrate that the language-metric variant achieves superior reward and automaton compactness compared to FlexFringe and the CMS variant.

---

## Strengths

- **Formally proven exponential distinguishability gap (Theorem 1 + Example 4).** The paper rigorously constructs a family of RDPs ($\mathbf{R}_N$) and shows that the $L_\infty^\ell$-distinguishability decays as $\mathcal{O}(2^{-N})$ while the $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$ — an exponential gap. This is a concrete, non-trivial result that directly justifies the language metric rather than merely asserting superiority, and targets a specific structural pathology (distinguishing signal carried by event co-occurrence patterns rather than any individual suffix string).

- **Unifying language metric formalism.** Definition 2 cleanly unifies $L_\infty$, $L_1$, total variation, and both prefix distances as special cases of $L_X$ under different choices of $\mathcal{X}$. This consolidation is genuinely novel and provides a principled analytical lens on the $L_\infty$-vs-$L_1$ tension that motivates the paper.

- **Two-dimensional language hierarchy with formal language roots.** The $\mathcal{X}_{i,j}$ hierarchy — constructed via the $C_k^\ell$ operator inspired by the dot-depth hierarchy of star-free regular languages — is a creative and principled way to interpolate between cheap local tests and richer long-range pattern matching. The connection to formal language theory gives this hierarchy structural justification beyond ad hoc design.

- **Identification and correction of a mistake in Cipollone et al. (2023).** The analysis uncovers a missing $\sqrt{H}/\mu_0$ factor in the original RegORL sample complexity proof. This correction applies both to the prior work and to the new bounds in Theorems 2–3, adding scientific credibility.

- **Clear empirical validation of the scaling claim (Figure 2).** The T-maze corridor-length scaling experiment directly demonstrates linear vs. exponential growth in both runtime and automaton size, with the language-based approach reaching $N=100$ while the CMS approach exceeds 1800 seconds at $H=15$. This is clean, focused experimental evidence directly connected to Theorem 1.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No sample complexity experiments — the headline claim is unvalidated empirically.** The paper's central stated contribution is improved *sample efficiency*, yet every experiment in Table 1 uses a fixed dataset of $K=100$ episodes with no variation. There is no learning curve showing policy reward as a function of $|\mathcal{D}|$, no recovery accuracy vs. dataset size, and no empirical estimate of sample complexity improvement. For a paper whose primary theoretical contribution is Theorem 3 (a PAC sample complexity bound), this is the single most consequential gap between the stated contribution and the empirical evidence.

- **No comparison to RegORL / original ADACT-H.** The paper explicitly frames itself as improving upon RegORL (Cipollone et al., 2023), and the pseudocode of the original ADACT-H (the direct baseline) is included in Appendix A. Yet Table 1 compares only against FlexFringe — a general automata learner with no RL guarantees and different optimization objectives — and against the CMS internal variant. The original ADACT-H with the $L_\infty^p$ test, which is the method the paper claims to surpass, is absent from all empirical comparisons. Without this comparison, the claim to practical improvement over the direct prior method is unsubstantiated.

- **The $1/d_m^*$ term can dominate and may be exponential in $H$, undermining the overall bound.** Both Theorems 2 and 3 scale as $1/d_m^*$, where $d_m^* = \min_{u,a,o} d_t^*(u,a,o)$ is the minimum occupancy of the optimal policy. This can be exponentially small in $H$ if any RDP state is reachable only via a specific chain of transitions — even in fairly structured settings. The paper acknowledges this in one sentence ("The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach") but offers no further analysis. If $1/d_m^*$ is the dominant factor, the claimed removal of exponential $H$-dependence via $\log|\mathcal{X}|$ may be illusory end-to-end. At minimum, the authors should characterize the $d_m^*$ regime for the T-maze family, where the benefits are most prominently claimed.

- **Notation error in the estimator definition.** Section 4.1 writes $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$. This is type-inconsistent: $\mathcal{X}_{i,j}$ is a *set of languages*, not a language, so "$e \in \mathcal{X}_{i,j}$" would require the trace $e$ to be a language. The intended definition is clearly: for each fixed $X \in \mathcal{X}_{i,j}$, $\hat{p}_1(X) := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$, with the test maximizing over $X$. Since this estimator is the statistical object underpinning Theorem 3, the precise definition must appear correctly in the main text.

### Minor

- **CMS width formula appears inverted relative to standard parameterization.** The paper defines the column width as $w = \lceil \varepsilon/\delta_c \rceil$ (Section 2). The canonical Cormode & Muthukrishnan (2005) parameterization sets width $w = \lceil e/\varepsilon \rceil$, making the sketch *wider* for *smaller* error tolerance. The paper's formula has $w$ growing with $\varepsilon$, which has the opposite sense. If this is a deliberate reparameterization suited to the specific application, it requires explicit justification, since the CMS guarantees are invoked in the proof of Theorem 2.

- **No memory measurements for the CMS variant.** Theorem 2's claimed practical advantage over vanilla ADACT-H is *memory reduction*. Yet Table 1 reports only runtime and automaton size — not peak memory usage. The primary claimed benefit of the CMS contribution is empirically unquantified.

- **No ablation over language hierarchy parameters $(i,j)$.** All experiments exclusively use $\mathcal{X}_{3,1}$ without justification. The $\mathcal{X}_{i,j}$ hierarchy is presented as a key structural contribution, but there is no empirical demonstration of when $\mathcal{X}_{1,1}$ suffices or when larger $j$ is needed. The hierarchy remains a conceptual device rather than an empirically validated design tool.

- **No characterization of when Assumption 1 holds in practice.** Assumption 1 requires $\mu_0 > 0$ for the chosen $\mathcal{X}_{i,j}$. The paper demonstrates this for T-maze with $\mathcal{X}_{2,1}$ (Theorem 1), but provides no general characterization of which RDP classes admit positive $L_X$-distinguishability for small $j$ while having exponentially small $L_\infty^p$-distinguishability. Practitioners cannot determine when to apply the method without this guidance.

- **Chain from RDP recovery to policy optimality is implicit in the main text.** Theorems 2–3 establish recovery of the minimal RDP with high probability; the reduction to an $\varepsilon$-optimal policy guarantee is deferred entirely to the prior RegORL construction in Appendix A. The main text should state explicitly how RDP recovery translates into policy suboptimality bounds, including the contribution of the failure-probability event.

### Tiny

- The objective in Section 2.3 writes $V_\circ^*(h) - V_{\hat{\pi}}^*(h) \leq \varepsilon$ — the subscript appears to be a formatting artifact for $V_0^*(h) - V_0^{\hat{\pi}}(h)$ and should be corrected.
- The conclusion says the language approach "remov[es] the dependency on $L_\infty^p$-distinguishability parameters." More precisely, it *replaces* that dependency with $L_X$-distinguishability under the chosen $\mathcal{X}$, which must still be assumed positive.

---

## Nice-to-Haves

- **Guidance or heuristic for selecting $(i,j)$ at deployment time.** A model-selection criterion using held-out log-likelihood on a validation split of $\mathcal{D}$ could potentially identify the sufficient complexity level without prior knowledge of the RDP structure.

- **Visualization of learned automata on T-maze.** Showing the automaton recovered by $L_X$ vs. the CMS/$L_\infty^p$ approach on T-maze would make tangible why the language metric recovers correct structure while the baseline fails — directly connecting the motivating example to the learned artifact.

- **Results at larger $H$.** The paper claims exponential gains in $H$; all Table 1 domains have $H \leq 15$. Even two results at $H = 30$–$50$ would strengthen the practical relevance claim and demonstrate the approach beyond toy scales.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Comparison to sequence-model baselines (RNNs, transformers) is missing"** (Harsh Critic). This is scope creep. The paper explicitly scopes itself to methods with formal PAC guarantees and notes that sequence-model approaches "lack correctness guarantees." Evaluating against them is orthogonal to the stated contribution.

- **"PSR connection is underleveraged"** (Harsh Critic). The paper appropriately cites the PSR connection and notes that existing PSR bounds involve different parameters. Demanding deeper structural comparison to PSRs is outside the paper's scope.

- **"T-maze state definition is malformed / difficult to parse"** (Harsh Critic). The apparent structural ambiguity in the state set definition in Example 3 is almost certainly a PDF text-extraction artifact; the example is coherent in substance and the construction is consistent with the corridor dynamics described.

- **"Notation is inconsistent among $L_\infty^\circ$, $L_\infty^p$, $L_\infty^\ell$"** (Harsh Critic). These are genuinely distinct metrics in the paper: $L_\infty^\circ$ uses prefix matching (any trailing suffix, defined in Section 2.2), $L_\infty^p$ is the prefix distance used in ADACT-H's test, and $L_\infty^\ell$ is the $L_\infty$ over strings of exact length $\ell$. The notation is deliberate and internally consistent, not erroneous.

- **"FlexFringe comparison is unfair due to heuristics"** (implicit, Harsh Critic). The paper explicitly acknowledges that FlexFringe uses heuristics that do not preserve sample complexity guarantees, and notes "The RDPs output by FlexFringe are not always directly comparable." This comparison is intentionally asymmetric in FlexFringe's favor — FlexFringe being computationally less constrained makes it a strong practical baseline, not a weak one. Beating it strengthens the paper's claims.

- **"No statistical significance for entries reporting 1.0 and 4.0"** (Harsh Critic). In episodic environments such as Corridor ($H=5$) and T-maze(c) with a fixed goal structure, an optimal policy can achieve deterministic reward. Reporting a point value without variance in these cases is appropriate.

- **"Contribution statement mixes theory and implementation without clearly isolating novelty"** (Harsh Critic). The paper separates the language metric and CMS contributions clearly in both the abstract and the contributions paragraph. This is a stylistic critique without technical substance.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is that the standard $L_\infty^p$-based statistical test for state merging in automaton learning is not merely "practically inconvenient" but is *provably catastrophic* for a natural and important class of RDPs: those where the distinguishing signal is carried by event co-occurrence patterns (e.g., observing a specific reward upon a specific action *anywhere* in a suffix) rather than by any single suffix string. The language metric $L_X$ — by measuring probability mass over *sets* of strings defined by pattern-matching rather than individual strings — captures exactly the structural feature that makes T-maze-like domains tractable. The connection to the dot-depth hierarchy of star-free regular languages provides a theoretically grounded organizing principle, and the $\mathcal{X}_{i,j}$ hierarchy is a concrete implementable instantiation. A key open question raised by this synthesis is whether $1/d_m^*$ can be tamed for the same structured RDPs where $1/\mu_0$ improves via the language metric — if yes, the end-to-end sample complexity improvement would be genuinely polynomial in $H$.

---

## Suggestions

1. **Add learning curves.** Run each domain with dataset sizes $K \in \{10, 25, 50, 100, 250, 500\}$ and plot average reward vs. $K$ for Language metric, CMS, and original ADACT-H. This is the most important missing experiment and directly validates the sample efficiency claim.

2. **Include original ADACT-H ($L_\infty^p$ test) as a baseline in Table 1.** Since the pseudocode is already in Appendix A and all three variants share the same codebase, this comparison is straightforward and is necessary to substantiate the claim of improvement over RegORL.

3. **Fix the estimator definition.** Replace $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$ with the correct per-language formulation: for each $X \in \mathcal{X}_{i,j}$, define $\hat{p}_1(X) := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$, and state the test explicitly as the maximum discrepancy over $X \in \mathcal{X}_{i,j}$.

4. **Justify or correct the CMS width formula.** Clarify the parameterization $w = \lceil \varepsilon/\delta_c \rceil$ relative to the standard Cormode & Muthukrishnan (2005) formulation, and verify that downstream proofs use the correct guarantee.

5. **Add memory measurements for CMS.** Include peak memory usage alongside runtime in Table 1, since reduced memory is the primary theoretical advantage of Theorem 2.

6. **Ablation over $(i,j)$.** Add a table or figure showing how reward and automaton size vary as $(i,j)$ is changed on one or two domains, to ground the hierarchy empirically and guide practitioners.

7. **Characterize the $d_m^*$ regime for T-maze.** Add a short analysis or corollary showing what $d_m^*$ equals for the T-maze family as a function of $N$, to determine whether the end-to-end sample complexity bound is indeed polynomial in $H$ for this motivating example.