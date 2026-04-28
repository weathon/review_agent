## Summary

This paper proposes GLIDE, a causal discovery framework that tests distributional invariance of effect-cause conditionals under prior shifts created via data resampling, combined with a quadratic-complexity graph search using Markov blankets and maximal cliques. The method claims to recover causal DAGs from observational data across Linear-Gaussian, Non-linear-Non-Gaussian, and Categorical settings, with empirical validation on synthetic benchmarks and large-scale real-world datasets including Munin (1041 variables).

## Strengths

- **Scalability with quadratic complexity**: The parent-finding algorithm leverages Markov blankets and maximal cliques on augmented bidirectional graphs to achieve O(d²) search complexity (Theorem 7, Section 4.3). This enables processing of the Munin dataset (1041 variables) in ~6200 seconds, whereas PC and FCI fail to complete within 48 hours (Table 2).

- **Strong empirical performance on large-scale benchmarks**: GLIDE achieves best SHD on 7/7 real-world datasets and best spurious rate on 5/7, with particularly large gaps on high-dimensional datasets (e.g., 1.8% vs 42.4% spurious rate on Munin compared to GIES, Table 2).

- **Versatility across data generation models**: The framework applies to Linear-Gaussian, Non-linear-Non-Gaussian, and Categorical data without functional form assumptions (Table 1), unlike NOTEARS-family methods which degrade in non-linear settings (Figure 2, nL-nG columns).

## Weaknesses

### Fatal

None

### Major

- **Unexplained success in Linear-Gaussian settings contradicts identifiability theory**: The paper claims DAG recovery from observational Linear-Gaussian data (Table 1 shows ✓ for L-G; Figure 2 shows positive SHD/spurious rate results). However, it is a fundamental result in causal discovery that observational L-G distributions only identify the Markov Equivalence Class (CPDAG), not the specific DAG orientation. Footnote 1 acknowledges "Finding the true causal graph is not possible without running intervention," yet the paper presents positive empirical results in L-G settings without explaining how the identifiability barrier is overcome. The resampling mechanism creates weighted subsets of the *same* joint distribution, not independent environments with genuinely different priors. This suggests the L-G empirical success may stem from undisclosed non-Gaussianity in the synthetic data generation (the Zheng et al. 2018/2020 code may introduce slight non-Gaussianity) or finite-sample artifacts, rather than the proposed method's validity. This undermines the core claim of general observational DAG recovery.

- **Resampling does not create independent environments for invariance testing**: Theorem 1 requires different priors P_i(B) over source variables, but the method constructs these via downsampling from a single dataset D (Theorem 5, Section 4.2.2). Downsampling produces weighted subsets of the same i.i.d. data, not genuinely independent environments with novel distributions. For Markov Equivalent L-G structures, the conditional P(X|Z) is invariant to marginal shifts in *both* directions due to Gaussian likelihood symmetry. The method effectively attempts to extract interventional information from observational resampling without additional assumptions (known heterogeneity or non-Gaussianity), which is theoretically unsupported. This gap between the theorem's assumptions and the practical implementation weakens the theoretical foundation.

### Minor

- **Complexity analysis overlooks conditional density estimation cost**: The O(d²) claim (Section 4.1.D) accounts for graph search but treats the invariance test's conditional density estimation as O(1). For continuous variables using binning (Section 4.2.2), estimating P_i(X|Z) scales exponentially with |Z| (the conditioning set size). While sparsity may keep |Z| small empirically, the worst-case complexity is O(m·d·cost(P(X|Z))), not O(md²). A runtime breakdown showing density estimation vs. graph search costs would clarify whether the O(d²) claim holds in practice.

### Trivial

None

## Nice-to-Haves

- Evaluate performance on recovering CPDAG (skeleton + oriented edges where identifiable) rather than full DAG for L-G data to align claims with theoretical limits.

- Explicitly test performance on purely Gaussian data vs. non-Gaussian data to isolate whether directional accuracy comes from the invariance test or implicit non-Gaussianity in the data generation.

- Report the actual divergence (e.g., KL divergence) between P(B) and P_i(B) achieved by resampling to verify the prior shifts have sufficient magnitude for the invariance test to have power.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point about "Basis Identification Stability"**: The critic questions whether the basis-finding step correctly identifies true roots. However, the paper explicitly uses basis variables as a *surrogate* for sources (Section 4.2.1: "As finding the true sources is not possible without intervention, we will use a basis set as a surrogate"). This is acknowledged, not a hidden flaw.

- **Strength Finder claim about "4.2% vs 11.75% spurious rate"**: This specific comparison is accurate (Section 5.1), but the framing as "principled invariance test replacing heuristic scores" overstates the theoretical grounding given the identifiability concerns above.

- **Generic strengths about "important problem" or "unified framework"**: Removed as these are superficial without specific evidence.

- **Critic's concern about "Spurious Rate metric undefined for CPDAG"**: The paper evaluates against baselines that also output DAGs (NOTEARS, GIES), so the metric is consistently applied. This is a comparison methodology issue, not a paper flaw.

## Novel Insights

The paper's core insight—that effect-cause conditionals remain invariant under prior shifts while cause-effect conditionals do not—is well-known in causal inference (related to Independent Mechanism Principle). The novel contribution is attempting to operationalize this via resampling from a single dataset rather than requiring multiple true environments. However, this operationalization faces the theoretical barrier that resampling does not create the independent environments required for the invariance principle to distinguish Markov Equivalent structures in L-G settings. The empirical success despite this theoretical gap suggests either (a) real-world and synthetic data contain sufficient non-Gaussianity to enable identifiability, or (b) finite-sample effects break the symmetry. Disentangling these would strengthen the contribution.

## Suggestions

1. **Clarify identifiability assumptions**: Revise claims to state that DAG recovery requires either (i) sufficient non-Gaussianity in the data, (ii) genuine multi-environment data, or (iii) additional assumptions beyond observational L-G. Present L-G results with the caveat that empirical success may reflect slight non-Gaussianity in the data generation process.

2. **Add non-Gaussianity ablation**: Explicitly measure the degree of non-Gaussianity in synthetic datasets (e.g., using skewness/kurtosis or normality tests) and correlate with directional accuracy to test whether the method's success correlates with deviation from Gaussianity.

3. **Provide runtime breakdown**: Report time spent on density estimation vs. graph search to verify the O(d²) scalability claim holds when conditioning set sizes vary.

4. **Compare to CPDAG recovery baselines**: Include evaluation against methods that explicitly output CPDAGs (e.g., PC algorithm's pattern output) to show whether GLIDE's oriented edges exceed what is theoretically identifiable from observational data alone.

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Comparison to GLIDE |
|-------|-----------|---------------------|
| ta8BKRa1bl | 6.00 | Has clear identifiability theory requiring true multi-environment data; GLIDE has stronger empirical but weaker theoretical grounding |
| mA78uXqcnl | 7.00 | Strong theory + empirical on real data; GLIDE has comparable empirical but unexplained L-G success |
| wnFbqvUJ6D | 5.00 | Novel approach without non-Gaussianity assumption; similar theoretical ambition but GLIDE has better empirical |
| BNHplerBYE | 5.33 | Score-based with equivalence class guarantees; more honest about identifiability limits |
| V7pT2ZRoTB | 4.50 | Narrow theoretical scope, synthetic-only; GLIDE has better empirical but similar theory-practice gap |
| aS7EVadvZD | 3.00 | Poor theoretical grounding, no identifiability discussion; GLIDE is better but shares some issues |
| EzHPHhSQMD | 2.00 | Overclaimed guarantees, trivial results; GLIDE is substantially better |

**Reasoning**: GLIDE has genuinely strong empirical results on large-scale real-world datasets (comparable to mA78uXqcnl's 7.0), but the unexplained L-G success represents a more serious theoretical gap than ta8BKRa1bl (6.0), which explicitly requires multiple true environments. The paper is better than V7pT2ZRoTB (4.5) due to real-world validation, but worse than wnFbqvUJ6D (5.0) which is more careful about identifiability claims. The empirical strength prevents scoring as low as aS7EVadvZD (3.0).

**Final Score**: 4.5 — The strong empirical results on large-scale datasets are valuable, but the fundamental theoretical gap regarding L-G identifiability prevents acceptance without significant revision. The paper should either (a) provide theoretical justification for how resampling breaks L-G symmetry, or (b) restrict claims to non-Gaussian settings and present L-G results as empirical observations requiring further investigation.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>