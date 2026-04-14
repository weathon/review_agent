## Summary

This paper proposes two new techniques for offline RL in Regular Decision Processes (RDPs): (1) a novel language-based metric $L_X$ grounded in the dot-depth hierarchy of star-free regular languages, which replaces the $L_\infty^p$ test in the ADACT-H state-merging algorithm, and (2) a Count-Min-Sketch (CMS) approach to compactly store empirical suffix distributions. The central theoretical result (Theorem 1) proves an exponential gap between $L_\infty^\ell$-distinguishability and $L_{\mathcal{X}_{2,1}}$-distinguishability for the T-maze family of RDPs, and Theorem 3 derives a PAC sample complexity bound that depends on $\log |\mathcal{X}|$ rather than horizon-dependent suffix-space size. The authors also discover and correct an error in a prior proof, adding a $\sqrt{H}/\mu_0$ factor to the RegORL sample complexity.

---

## Strengths

- **Provable exponential gap via concrete family (Theorem 1).** The paper rigorously constructs a family of RDPs $(\mathbf{R}_N)$ and behavior policies $(\pi_N^b)$ for which $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$. The T-maze running example grounds this in an intuitive and widely-used benchmark domain; the argument is not merely asymptotic but is tied to a specific structural property (aggregating over the language of "North-action-then-positive-reward" traces).

- **Two-dimensional hierarchy that unifies common metrics.** The $\mathcal{X}_{i,j}$ hierarchy is a principled and elegant construction: $j=1$ on singleton patterns recovers $L_\infty^p$; taking $\mathcal{X} = 2^{\Gamma^\ell}$ recovers total variation ($L_1$). The interpolation via the operator $C_k^\ell$ and the basic pattern sets $\mathcal{G}_1, \mathcal{G}_2, \mathcal{G}_3$ provides a concrete and actionable family of tests, each with $|\mathcal{X}_{i,j}| \in \mathcal{O}((AOR)^j)$, growing polynomially in problem parameters for fixed $j$.

- **Identification and correction of a prior proof error.** The analysis in Theorem 2 uncovers a missing $\sqrt{H}/\mu_0$ multiplicative factor in the RegORL proof of Cipollone et al. (2023). Identifying and correcting such errors is a meaningful contribution to the reliability of the theoretical foundation in this area.

- **T-maze scaling experiment supports the core claim.** Figure 2 provides direct empirical evidence that the language metric approach scales polynomially in corridor length $N$ (both in runtime and automaton size), while CMS scales exponentially. The gap is observed consistently across 20 runs, with the language approach handling $N=100$ while CMS times out beyond $H=15$—precisely the behavior predicted by the theoretical analysis.

---

## Weaknesses

### Fatal
None.

### Major

- **No direct empirical comparison with the baseline being improved (ADACT-H / RegORL with $L_\infty^p$).** The paper's central empirical claim is that the language metric improves on existing RDP offline RL algorithms. However, the only comparator in experiments is FlexFringe, a grammar/PDFA learner with no PAC guarantees and different objectives (it sometimes learns cyclic automata). The original ADACT-H with $L_\infty^p$ is never included as a baseline in Table 1 or Figure 2. Without this comparison, there is no direct empirical validation that the proposed method yields better automata or policies than the prior approach on matched datasets—this is especially important given that the theoretical improvement is conditional (on $\mu_0$ and $d_m^*$).

- **No sample complexity experiments: reward vs. dataset size $K$ is never measured.** The core theoretical claim is improved PAC sample complexity (Theorem 3). Yet the experiments report only fixed-dataset reward, runtime, and automaton size. The natural validation—sweeping $K$ (number of offline episodes) and measuring policy quality as a function of $K$ for each method—is entirely absent. Without this, the empirical support for the paper's primary contribution is indirect at best.

- **No guidance for selecting $(i, j)$ in $\mathcal{X}_{i,j}$.** Assumption 1 presupposes that the chosen $\mathcal{X}_{i,j}$ satisfies $L_{\mathcal{X}_{i,j}}$-distinguishability $\geq \mu_0 > 0$. In practice, the user has no means to verify this from data, and the paper provides no adaptive selection, cross-validation criterion, or statistical test for sufficiency of $(i,j)$. If $(i,j)$ is chosen too small, $\mu_0 = 0$ and the algorithm silently fails. All experiments use $\mathcal{X}_{3,1}$ without justification. This gap significantly limits practical applicability.

### Minor

- **The $d_m^*$ term can independently be exponentially small in $H$, potentially negating the $\mu_0$ gain.** Theorems 2 and 3 depend on $d_m^* = \min_{u,a,o} d_t^*(u,a,o)$, the minimum occupancy of the optimal policy. The paper briefly notes that "$1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach," but provides no analysis of when this is well-behaved or how it interacts with the language family choice. In the worst case, the exponential improvement in $1/\mu_0$ is entirely offset by an exponential $1/d_m^*$, yet the paper presents the bounds without clearly quantifying this interaction.

- **$L_X$ is only a pseudometric, and the consequences for correctness are unaddressed.** Footnote 1 acknowledges that $L_X$ is only a pseudometric. This means distinct RDP states can have $L_X$ distance zero for a given $\mathcal{X}_{i,j}$, in which case the algorithm would incorrectly merge them. The paper does not analyze whether or under what conditions this is avoided, nor how it affects the model recovery guarantee beyond assuming $\mu_0 > 0$.

- **The CMS memory advantage is not measured empirically.** CMS is introduced with the stated benefit of reducing memory requirements (Theorem 2). However, Table 1 reports only runtime, automaton size, and reward—not memory. The theoretical memory benefit is never validated or quantified experimentally, leaving the CMS contribution empirically unsupported on its primary claim.

- **CMS width parameter $w = \lceil \varepsilon / \delta_c \rceil$ appears non-standard.** The standard CMS (Cormode & Muthukrishnan, 2005) sets width $w = \lceil e/\varepsilon \rceil$, which is inversely proportional to desired accuracy: smaller $\varepsilon$ requires larger $w$. The paper's formula $w = \lceil \varepsilon / \delta_c \rceil$ gives a smaller $w$ for smaller $\varepsilon$, which is the opposite of standard behavior. This may reflect a non-standard parameterization or a typesetting issue, but as presented it is inconsistent with the cited data structure and should be clarified. The resulting approximation property used in the proof of Theorem 2 is also not explicitly stated.

- **The correction to the prior proof ($\sqrt{H}/\mu_0$ factor) weakens the narrative without being properly contextualized.** The paper says both RegORL and its new bounds have an additional $\sqrt{H}/\mu_0$ factor due to a corrected prior proof. However, it does not clearly show whether the new bounds in Theorem 3 are still strictly better than the corrected RegORL bounds across the relevant parameter regimes. A direct corollary comparing the two in matched notation would resolve this.

### Tiny

- **The estimator definition for $L_X$ is ambiguous as written.** The paper states $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j}) / |\mathcal{Z}_1|$, but $\mathcal{X}_{i,j}$ is a *set of languages*, not a single language. The intended empirical estimator for $L_X$ should be $\hat{L}_X(\mathcal{Z}_1, \mathcal{Z}_2) = \max_{X \in \mathcal{X}} |\hat{p}_1(X) - \hat{p}_2(X)|$ where $\hat{p}_i(X) = \sum_{e \in \mathcal{Z}_i} \mathbb{I}(e \in X) / |\mathcal{Z}_i|$. The distinction matters for understanding the computational cost.

- **Notation instability for the reference metric ($L_\infty^\circ$, $L_\infty^p$, $L_\infty^\ell$).** These appear to refer to related but distinct metrics, yet the notation shifts across sections without a clear mapping. This adds friction when following the comparison between old and new bounds.

- **Monotonicity of $L_{\mathcal{X}_{i,j}}$ is stated but not proved or cited in the main text.** The claim $L_{\mathcal{X}_{i,j}} \leq \min(L_{\mathcal{X}_{i+1,j}}, L_{\mathcal{X}_{i,j+1}})$ presumably follows from $\mathcal{X}_{i,j} \subseteq \mathcal{X}_{i+1,j}$ and $\mathcal{X}_{i,j} \subseteq \mathcal{X}_{i,j+1}$, but this should be made explicit given its central role.

---

## Nice-to-Haves

- An ablation over $(i,j)$ (e.g., comparing $\mathcal{X}_{1,1}$, $\mathcal{X}_{2,1}$, $\mathcal{X}_{3,1}$ on T-maze) would directly illustrate the practical effect of language family choice and provide intuition for practitioners—particularly showing what happens when $\mathcal{X}_{i,j}$ is too small to distinguish relevant states.

- Visualizing the learned automata against ground-truth RDP structure (e.g., the two parallel $\top/\perp$ components in T-maze) for each method would give qualitative evidence that the language metric correctly recovers RDP structure.

- A discussion of when $d_m^*$ is expected to be well-behaved (e.g., structured RDPs with good coverage) and how it interacts with the $\mu_0$ improvement would make the bounds more interpretable.

- Extending experiments to domains with $H \geq 50$–$100$ where the exponential improvement in $\mu_0$ should manifest not just in runtime but also in policy quality with realistic dataset sizes.

- An end-to-end theorem connecting dataset size $K$ to $\varepsilon$-optimal policy return in the main paper (currently deferred to the appendix via RegORL), making the offline RL contribution self-contained.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's concern about practical relevance of the algorithm contributions vs. model reconstruction.** The paper clearly states its contribution is a tractable statistical test for ADACT-H, which is then incorporated into RegORL for policy learning. This framing is consistently maintained and is a legitimate, well-scoped contribution.

- **Criticism that algorithm pseudocode is in the appendix.** This is standard practice and not a weakness for an ICLR submission.

- **Criticism that broader impact is not discussed.** Not a standard ICLR requirement.

- **Criticism that sequence-model or recurrent-policy offline RL baselines are absent.** Such methods lack PAC guarantees and operate under different objectives than RDP model recovery. Their absence is within the paper's scope; including them could be a nice-to-have but is not a core flaw.

- **Criticism of "unfair comparison" with FlexFringe (FlexFringe uses heuristics that can hurt its performance).** The paper explicitly acknowledges that FlexFringe uses heuristics that do not preserve guarantees (Section 5). The asymmetry is disclosed and works *against* the authors' method (FlexFringe is freed from PAC constraints), making any win over it a stronger result.

- **Generic strength: "well-written" and "topic is important."** Removed per synthesis rules.

- **Critic concern about lack of broader impact section.** Removed.

---

## Novel Insights

The most genuinely novel insight in this paper—beyond presenting the $L_X$ framework—is the structural observation that the exponential blow-up of $L_\infty^p$ in RDPs like T-maze is not an intrinsic property of the problem's difficulty but an artifact of the metric: the *same* RDP states that are exponentially hard to separate by singleton-string probabilities can be $\Omega(1)$-separated by a single aggregated language event (the probability of "action North followed eventually by positive reward"). This shifts the conceptual frame from "the problem is hard" to "the test is the wrong granularity." The two-dimensional $\mathcal{X}_{i,j}$ hierarchy then provides a concrete realization of this insight, showing that the right level of aggregation can be extracted from the algebraic structure of episode traces (temporal ordering of basic action/observation/reward patterns) rather than from sufficiency-style or bisimulation-style notions. The connection to dot-depth hierarchy from formal language theory is a nontrivial cross-domain link that could inspire further work on complexity-theoretic characterizations of learnability in non-Markovian settings.

---

## Suggestions

1. **Add ADACT-H with $L_\infty^p$ as a direct experimental baseline.** On each domain, include the original ADACT-H variant as a third column in Table 1 using the same dataset. This is the method being improved and its absence is the most significant empirical gap.

2. **Add reward-vs-dataset-size curves.** For at least T-maze and one other domain, plot policy quality as a function of $K$ for the language metric and ADACT-H with $L_\infty^p$. This directly validates the sample complexity claims.

3. **Clarify or correct the CMS width formula $w = \lceil \varepsilon/\delta_c \rceil$.** If this is an intentional non-standard parameterization, state explicitly what approximation guarantee it provides and how it enters the proof of Theorem 2.

4. **Provide at least a heuristic for selecting $(i,j)$.** Even a monotone-search procedure ("start at $(1,1)$, increase until the statistical test is ever triggered") would substantially improve the paper's practical usability and address the most common practitioner objection.

5. **Add memory measurements to experiments.** Report peak memory usage for ADACT-H (language metric), ADACT-H (CMS), and FlexFringe. Without this, Theorem 2's stated advantage cannot be empirically verified.

6. **Fix the estimator notation in Section 4.1.** Define $\hat{p}(X)$ for each $X \in \mathcal{X}_{i,j}$ individually and write the empirical $\hat{L}_X$ as the max over $X$.

7. **Include a brief end-to-end corollary** combining Theorem 3 with the RegORL bound to state the dataset size needed for an $\varepsilon$-optimal policy, making the offline RL guarantee visible in the main text.

---

## Evaluation

**Originality:** High. The language metric framework and its connection to the dot-depth hierarchy are conceptually novel contributions not previously applied in RL or automata-based policy learning. The two-dimensional $\mathcal{X}_{i,j}$ hierarchy is a principled and non-trivial construction.

**Importance of research question:** Moderate. The exponential sample complexity barrier in offline RDP learning is a genuine and previously identified open problem. The broader RDP/non-Markovian RL setting is scientifically important, though it remains a specialized subfield.

**Claims well supported:** Partially. The theoretical claims (Theorems 1–3) are well-supported, and the T-maze scaling experiment is convincing for the tractability claim. However, the primary sample efficiency claim is not validated empirically (no reward-vs-K curves, no comparison with the exact baseline being improved), which is a significant gap between the paper's framing and its experimental evidence.

**Soundness of experiments:** Weak. Five domains are evaluated, but the baseline set is insufficient (FlexFringe is not an offline RL algorithm), no sample complexity experiments are conducted, CMS memory is not measured, and only a single $(i,j)$ is tested. The experiments support the tractability story but not the sample efficiency story.

**Clarity of writing:** Moderate. The high-level narrative is clear and the T-maze example threads the paper effectively. However, there are notation inconsistencies across sections, an ambiguous estimator definition, and a potentially erroneous CMS formula that create friction in the technical sections.

**Value to the research community:** Moderate-to-high contingent on revision. The language metric framework is a valuable conceptual contribution with potential applicability beyond offline RDPs (e.g., online learning, PSR identification). In its current form the empirical gaps reduce the paper's immediate utility.

**Contextualization relative to prior work:** Good. The paper positions itself clearly against RegORL and ADACT-H, and the related work section is appropriately broad. The identified error correction in prior work strengthens the theoretical foundation of the area.