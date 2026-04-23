Now let me do calibration searches to determine the appropriate score.Now I have enough to write the final consolidated review. Let me synthesize everything.---

## Summary

GLIDE is a causal discovery framework that recovers causal graphs from observational data by testing whether the conditional distribution $P(X \mid Z)$ remains invariant when the prior over source/basis variables is varied. The method generates synthetic data augmentations via a principled downsampling scheme, then uses a maximal-clique-based search over Markov blanket neighborhoods to efficiently identify parent sets with claimed $O(d^2)$ complexity. Empirical evaluation spans synthetic (L-G, nL-nG) and categorical data, as well as large-scale real-world Bayesian network benchmarks including Munin (1041 nodes).

---

## Strengths

1. **Principled and elegant invariance test:** Theorem 1 establishes a fundamental causal property (the conditional $P(X \mid \text{Pa}[X])$ is invariant to changes in the prior $P(\mathbf{B})$ over source variables), and the downsampling machinery in Theorems 4–6 provides a clean closed-form implementation. The convex hull characterization of valid source priors (Theorem 6, Eq. 5–6) is a genuinely useful theoretical result.

2. **Demonstrated scalability on very large benchmarks:** GLIDE successfully runs on the 1041-node Munin dataset (Table 2) achieving SHD 883.2 and 1.8% spurious rate, while PC times out entirely and GIES produces 42.4% spurious rate. Scaling a constraint-based-style method to this regime is a meaningful practical contribution.

3. **Broad applicability across data models without model-specific assumptions:** Unlike NOTEARS (linear), SCORE (score-based), or FCI (continuous), GLIDE applies uniformly to L-G, nL-nG, and categorical data; this is validated experimentally across all three regimes (Figures 2, 3, 4, Table 2).

4. **Low spurious rate across settings:** GLIDE achieves 4.2% spurious rate vs. 11.75% for NOTEARS in the 500-node nL-nG normal setting (Figure 2), and 1.8% vs. 42.4% for GIES on Munin (Table 2), indicating reliable causal identification relative to baselines.

---

## Weaknesses

### Fatal

None.

### Major

1. **Basis-variable substitution for source variables lacks formal justification.** Theorem 1 is proven for *source variables* $\mathbf{B} = \{B : \text{Pa}[B] = \emptyset\}$, but the algorithm uses a *basis set* (Definition 3), which is only guaranteed to be a maximal independent set of the same size as the sources — not the sources themselves. The paper justifies this substitution informally: "changing the prior over basis variables will have similar effect to changing prior over source variables" (Section 4.2.1). No theorem formalizes this. If a basis variable $B'$ has parents, the causal resampling interpretation changes: $D_i$ is still a valid subsample but does not correspond to an intervention on source priors in the sense Theorem 1 requires. This is a structural gap between the proven theorem and the implemented algorithm. The empirical success suggests the approximation is reasonable in practice, but the theoretical foundation claimed for GLIDE is not fully established by the stated theorems.

2. **PC is excluded from synthetic continuous data experiments (Figures 2–3) without adequate justification.** Table 1 explicitly marks PC as applicable to both L-G and nL-nG settings, yet the continuous synthetic benchmarks (Figures 2–3) include FCI, GIES, NOTEARS, MLP-NOTEARS, DAS, and SCORE but not PC. The footnote (Footnote 3) offers only the vague explanation that excluded baselines "produce very poor performance not meaningful for comparison," but this is implausible for PC on standard L-G benchmarks (where PC is state-of-the-art) and directly contradicts Table 1's applicability markings. PC appears in categorical experiments (Figure 4) and real-world data (Table 2) — including the latter where it times out on Pathfinder and Munin — but no such timeout explanation is offered for the continuous case. This omission leaves open the possibility that PC is competitive with GLIDE on the L-G/nL-nG synthetic benchmarks where most of the headline performance comparisons are made.

### Minor

1. **One-directional theorem and argmin uniqueness.** Theorem 1 establishes a *necessary* condition ($\mathbb{V} > 0 \Rightarrow Z \neq \text{Pa}[X]$). The selection rule (Eq. 3) uses an argmin to identify $\text{Pa}[X]$, which requires the true parent set to uniquely achieve near-zero variance among all maximal clique candidates. The paper correctly notes (after Eq. 3) that this becomes exact only as $m \to \infty$. What is not analyzed is the false-selection rate for finite $m$: any set $Z$ for which $P(X \mid Z)$ empirically coincides with $P(X \mid \text{Pa}[X])$ (e.g., supersets containing redundant ancestors) could achieve comparably low variance. No ablation measures how often the argmin selects the wrong candidate, which limits understanding of when the method is reliable.

2. **Identifiability in L-G settings is unaddressed.** For linear-Gaussian data, the true DAG is only identifiable up to Markov equivalence class (CPDAG) without additional assumptions (e.g., non-Gaussianity). GLIDE produces a DAG and is evaluated by SHD against the ground-truth DAG, the same convention used by all baselines. However, the paper provides no discussion of what GLIDE recovers theoretically in this setting or how SHD should be interpreted relative to Markov-equivalent DAGs. This is a secondary issue since the comparison remains internally consistent across methods.

3. **GIES runtime advantage in real-world experiments.** Table 2 shows GIES is 10–100× faster than GLIDE across all real-world datasets (e.g., 61.5s vs. 6200s on Munin). The paper concludes GLIDE has "the best balance between performance and scalability" but does not analyze the regime where GIES's massive speed advantage makes it preferable despite lower accuracy. For practitioners, this tradeoff is important and is unaddressed.

### Trivial

1. **Complexity constant is empirically bounded, not structural.** The $O(d^2)$ claim relies on the empirically verified bound $p \leq 13$ (Section 4.3). The paper correctly validates this empirically, but describing the result as a "provable $O(d^2)$" characterization overstates its generality — it is $O(d \cdot 3^{p/3})$ with $p$ controlled empirically.

---

## Nice-to-Haves

- **Empirical measurement of argmin false-selection rate.** Constructing controlled synthetic graphs with known Markov equivalence structures and measuring how often Eq. (3) selects a wrong maximal clique would directly characterize the practical impact of the one-directionality limitation in Issue (Minor 1).
- **Ablation on bin width for continuous data.** The paper ablates $\gamma_0$ and $m$ but not the bin width used to discretize continuous basis variables, despite this being a substantive algorithmic choice affecting resampling granularity and conditional density estimates.
- **Discussion of robustness to latent confounders.** GLIDE assumes causal sufficiency. A brief analysis of how the invariance test degrades under unobserved confounders — and whether any partial robustness can be claimed — would be valuable for practitioners.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"PC times out on Pathfinder/Munin — GLIDE's advantage here is unexplained"** *(harsh critic's Section 5.3 note)*: Table 2 already reports N/A for PC on Pathfinder and Munin. This is presented transparently as a scalability advantage for GLIDE, not a hidden omission. The concern about whether "PC would be competitive if given more time" is speculative; GLIDE's superior SHD and spurious rate at feasible runtime is a valid finding.

- **"Abstract overstates claims ('up to 25×', '73.3% reduction')"**: The "up to" qualifier is used throughout and the extreme nL-nG 500-node setting does yield 25.52× speedup over MLP-NOTEARS (Section 5.1). The framing is within normal bounds for empirical claims and does not constitute misrepresentation.

- **"SHD 16.2% behind MLP-NOTEARS in normal L-G"**: The harsh critic raises this as a notable gap, but the paper explicitly acknowledges it (Section 5.1) and frames it as "second best" while noting much larger speedup. This is a known limitation, not an undisclosed one, and is within range of "comparable" for an efficiency-focused method.

- **Any concern about reference/appendix availability**: Proofs are deferred to appendix, which is a normal practice. The parser strips these sections.

---

## Novel Insights

The most genuinely novel observation in this work is the use of downsampling-without-replacement as a *causal intervention surrogate*: by selecting subsets of observational data that reflect different marginal distributions of source/basis variables while keeping the conditional structure intact, the method effectively creates pseudo-interventional datasets from purely observational data. This resampling-as-intervention interpretation is a creative bridge between observational and interventional causal inference and could find applications beyond graph discovery (e.g., invariance testing for individual causal edges or covariate shift analysis in causal models).

---

## Suggestions

1. **Add PC to Figures 2–3 or explicitly justify its exclusion.** If PC times out in the continuous synthetic setting, report its last completed datapoint and mark the cutoff. If it does not time out but performs poorly, report the result — it strengthens the paper's claims.
2. **Prove or empirically characterize the basis-source substitution.** At minimum, add an experiment comparing GLIDE's performance when the true source set is known (oracle basis) vs. the computed basis, to bound the approximation error introduced by the substitution.
3. **Add a false-selection rate ablation** varying $m$ to show convergence toward unique argmin, directly addressing the one-directional implication concern.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison to GLIDE |
|------|----------------|---------------------|
| `/human_reviews/lk2Qk5xjeu.md` | 7.0 | Invariance-based CRL, theoretically complete, well-presented; GLIDE has stronger empirics but weaker theoretical completeness |
| `/human_reviews/xByvdb3DCm.md` | 8.0 | Guaranteed causal discovery algorithm with full proofs; stronger theory, more complete than GLIDE |
| `/human_reviews/or8wkKoBP4.md` | 4.0 | Causal DAG detection with unfaithfulness; had math mistakes and unclear utility; GLIDE is clearly stronger |
| `/human_reviews/xbUlKe1iE8.md` | 4.8 | Doubly-robust structure identification with theoretical issues; similar pattern to GLIDE but narrower scope and weaker empirics |
| `/human_reviews/8QTpYC4smR.md` | 1.0 | No scientific contribution; GLIDE is far above this |

GLIDE sits between the 4.0–5.0 (weak causal discovery papers with theoretical/empirical gaps) and 7.0–8.0 (theoretically complete causal papers with strong results). The novel invariance test idea, broad applicability, and compelling large-scale empirics push it well above the reject cluster. However, the unproven basis-source substitution — which creates a gap between the stated theorem and the implemented algorithm — and the unexplained absence of the PC baseline from the primary continuous experiments prevent it from reaching the 7.0 bar. The anchor cluster suggests a score of 5.5–6.0 for a paper with this pattern of novelty + empirical strength + theoretical incompleteness.

**Originality:** Moderate-to-high. The invariance-based downsampling idea is genuinely novel.
**Importance of research question:** High. Scalable, assumption-free causal discovery is a key open problem.
**Support for claims:** Partial. Empirics are strong and broad; theoretical support has a notable unproven step.
**Soundness of experiments:** Moderate. Strong across categorical and real-world settings; missing PC in continuous experiments is a gap.
**Clarity of writing:** Good. The method is clearly described and the framework is well-organized.
**Value to the research community:** Moderate-to-high. Demonstrated scaling to 1041 nodes is an important practical result.

**Final score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>