## Summary
This paper proposes GLIDE, a causal discovery framework that exploits the invariance of effect-cause conditional distributions under changes in the prior distribution of causes. Parent candidates are identified via Markov blankets and maximal clique enumeration (quadratic complexity under a sparsity assumption). An invariance test selects the true parents by minimizing the variance of the estimated conditional across multiple augmented datasets. Experiments on synthetic continuous and real categorical benchmarks show high accuracy, but on the largest real-world graph (Munin, 1041 nodes) GLIDE is 100× slower than GIES, contradicting the claimed scalability. The theoretical foundation does not guarantee that the variance-minimization rule recovers true parents.

## Strengths
- **Novel invariance-based test**: Theorem 1 establishes a necessary condition – non-zero variance of \(P(X|Z)\) across augmentations implies \(Z\) is not the parent set. This provides a principled, non-heuristic alternative to score-based methods.
- **Efficient candidate generation**: By restricting to cliques in an augmented Markov blanket graph and exploiting empirically low degeneracy (\(p \leq 13\)), the method achieves quadratic complexity in practice (Section 4.3).
- **Strong empirical accuracy**: GLIDE achieves the lowest structural Hamming distance (SHD) on all 7 real-world datasets (Table 2). On Munin it reduces SHD from 1235.0 (GIES) to 883.2 and spurious rate from 42.4% to 1.8%.
- **Rigorous data augmentation**: Theorems 4 and 5 derive minimal downsampling rates to achieve target cause priors while preserving the conditional \(P(\text{effect}|\text{cause})\), avoiding heuristic manipulations.
- **Clear presentation**: Figure 1 concisely illustrates the two-phase workflow (invariance test preparation and parent finding), making the methodology accessible.

## Weaknesses

### Fatal
None. No fundamental error that entirely invalidates the core idea or results.

### Major
1. **Incomplete theoretical guarantee for parent selection**: Theorem 1 is a one-way implication (\(\mathbb{V}[P(X|Z)] > 0 \Rightarrow Z \neq \text{Pa}[X]\)). It does **not** prove that true parents uniquely minimize variance, nor that non-parent sets (e.g., supersets of \(\text{Pa}[X]\) containing conditionally independent variables) cannot achieve near-zero variance. The parent-finding rule in Eq. (3) relies on minimization without a consistency theorem, leaving the core mechanism theoretically ungrounded. (Section 4.1, Theorem 1; Section 4.3, Eq. 3)
2. **Critical dependency on perfect Markov blanket recovery with no error analysis**: The candidate generation assumes the Markov blanket \(\mathbb{M}(X)\) from Edera et al. (2014) is correct under causal sufficiency. Markov blanket identification from observational data is error-prone; no analysis of error propagation is provided. If \(\mathbb{M}(X)\) is inaccurate, true parents may be excluded from candidates and recovery becomes impossible. (Section 4.3, paragraph starting “As previous work in Markov blanket identification…”)
3. **Scalability claim contradicted by authors’ own results**: The abstract and introduction state “up to 25× reduction in processing time” and “better scalability.” On the Munin dataset (1041 nodes), GLIDE takes **6200.3 seconds** while GIES takes **61.5 seconds** – a **100× slowdown**. This directly falsifies the general scalability claim, especially on large graphs. (Abstract; Section 5.3, Table 2)
4. **Unjustified restriction to maximal cliques may exclude true parents**: Theorem 7 shows \(\text{Pa}[X]\) is a clique in \(G'(X)\), but the algorithm only tests **maximal** cliques. If \(\text{Pa}[X]\) is a non-maximal clique, it is omitted from candidates, making recovery impossible. No argument is given that \(\text{Pa}[X]\) must be maximal. (Section 4.3, “Restricting the set of plausible parent sets…”)

### Minor
1. **Missing ablation study referenced in text**: Section 4.2.3 claims “our ablation studies in Section 5 shows the impact of \(\gamma_0\),” yet Section 5 contains no such ablation. This misleads readers about the evidence for hyperparameter robustness. (Section 4.2.3, last sentence)
2. **Limited baseline comparison for categorical data**: Only GIES and PC are evaluated on categorical datasets, justified by “failure to produce meaningful results.” No quantitative evidence is provided; categorical adaptations of continuous baselines (e.g., NOTEARS with categorical neural networks) are not attempted. (Section 5.2, Table 1)
3. **Notation and clarity issues**: Theorem 1’s notation \(\mathbb{V}_{P_+(\mathbf{X})} \sim P [P_+(X|Z)]\) is unconventional and could be clarified. Definition 1’s perfect-map assumption is stronger than necessary for causal discovery under causal sufficiency. (Section 3, Definition 1; Section 4.1, Theorem 1)
4. **Hyperparameter \(\gamma_0\) not reported**: The threshold controlling downsampling rate is a key tuning parameter; its value(s) for experiments are absent from the main paper, hindering replication. (Section 4.2.3)

### Trivial
1. Minor formatting inconsistencies (e.g., equation formatting, figure caption details) that do not affect understanding.

## Nice-to-Haves
- **Theoretical consistency proof**: Prove that under causal sufficiency, faithfulness, infinite data, and sufficiently diverse augmentations, Eq. (3) converges to \(\text{Pa}[X]\).
- **Robustness to Markov blanket errors**: Analyze or propose methods to tolerate inaccuracies (e.g., expand candidate search beyond estimated blanket, iterative refinement).
- **Clarified complexity claim**: Explicitly state the assumption that degeneracy \(p\) is bounded; report \(p\) for all datasets and discuss when it might grow.
- **Complete ablation**: Include effects of \(\gamma_0\), number of augmentations \(m\), and isolation of the invariance test ( vs. Markov blanket alone) in the main paper.
- **Fair categorical baseline comparison**: Implement categorical versions of continuous baselines (e.g., NOTEARS with categorical MLPs) to level the comparison.
- **Statistical significance reporting**: Add hypothesis tests for observed differences in SHD and spurious rate.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Perfect-map definition (Definition 1) is too strong**: Actually, the paper later assumes causal sufficiency, which implicitly requires a perfect map, so this is a deliberate modeling choice rather than an error.
- **Basis construction relies on unreliable pairwise independence tests**: While finite-sample noise exists, the paper uses established tests from causal-learn; this is an empirical concern, not a theoretical flaw.
- **Downsampling reduces effective sample size**: The paper account for this via Theorems 4–5 deriving minimal downsampling rates that preserve information, so the concern is already addressed.
- **Strength finder overstated Theorem 1 as bi‑directional**: The theorem is one‑way; the strength description “directly distinguishes true causal relationships from false ones” is partially inaccurate.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- **Theoretical section**: Clearly state that Eq. (3) is a heuristic without proven consistency; discuss potential failure modes (e.g., non-parent supersets, non-maximal cliques).
- **Experiments**: Either report speedup numbers only for continuous synthetic data or clearly separate categorical results; explain the Munin runtime anomaly (e.g., categorical encoding overhead, growth of \(p\)).
- **Reproducibility**: Move all hyperparameter values (\(\gamma_0\), \(m\)) and ablation figures into the main text or explicitly reference appendix locations.
- **Limitations**: Acknowledge the assumption of causal sufficiency and perfect Markov blanket recovery; discuss how violations might impact accuracy.

## Score and Decision
MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>