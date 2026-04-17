Now I have enough information to compose my final review. Let me synthesize:

**Calibration papers and their scores:**
- RelChaNet (neural FS): Rejected, scores 5,5,6,5 — incremental novelty, unfair comparisons, limited evaluation
- Gumbel-Softmax concrete layer for band selection: Rejected, scores 5,5,3,3 — similar technique, limited novelty, lack of strong theoretical justification
- Decoupled ST-GS: Rejected, scores 3,6,5,3 — similar Gumbel-based method, lacked depth
- LCEN feature selection: Rejected, scores 5,5,5,1 — unfair comparisons, overclaims
- MI-based feature selection: Rejected, scores 1,3,3 — weak method, missing baselines
- TabReD (tabular DL benchmark): Accepted (Spotlight), scores 3,8,8,8 — strong benchmark contribution
- TabM: Accepted (Poster), scores 8,6,6,6 — incremental but effective, strong experiments

The AutoNFS paper has good empirical results but significant overclaiming issues, similar to the LCEN paper (unfair comparisons, overclaims) which was rejected. The core method is a reasonable combination of Gumbel-Sigmoid + cardinality penalty, similar in spirit to STG/Hard-Concrete.

Now let me write the final review.

## Summary

AutoNFS proposes a neural feature selection method that combines a Gumbel-Sigmoid-based masking network with a task network, trained end-to-end with a cardinality penalty that encourages sparsity. The key claims are: (1) it automatically determines the number of features to select (no need to specify k), and (2) it achieves nearly constant computational overhead regardless of input dimensionality. Evaluation is performed on 11 OpenML-based benchmarks with synthetic corruption, plus 24 metagenomic datasets.

## Strengths

- **Well-motivated problem**: Automatically determining the number of features to select is a genuine and practical pain point in feature selection. The L_select penalty offers a simple mechanism for this.
- **Sound empirical evaluation on structured benchmarks**: The use of the Cherepanova et al. (2023) benchmark with three corruption scenarios (random, Gaussian, second-order) provides a systematic way to test whether FS methods can identify genuinely relevant features. The misselection analysis (Figure 3) directly evaluates selection quality rather than only downstream accuracy.
- **Diverse real-world evaluation**: Testing on 24 metagenomic datasets with two downstream classifiers (MLP and RF) demonstrates applicability beyond synthetic benchmarks and shows the selected features transfer across model families.
- **Clear algorithm specification**: Algorithm 1 is well-defined and the overall architecture is straightforward, which aids reproducibility.

## Weaknesses

### Major

- **Experimental comparison is confounded by different feature budgets**: The paper explicitly states (Sec. 4.1): "all baseline methods select the same number of features as were in the initial representation (before corruption), whereas our method automatically chooses a much smaller subset." This means baselines are forced to keep exactly D features (the original count) while AutoNFS selects fewer — including potentially dropping some original features. This confounds feature selection quality with feature budget: AutoNFS may perform better because (a) it selects better features, or (b) it selects fewer features and thereby gains a regularization benefit. Without running baselines at matched feature counts (or across a Pareto frontier), the claim that AutoNFS "consistently outperforms both the classical and neural FS methods" cannot be substantiated. This is the central experimental claim of the paper, and it is undermined by this design.

- **"Automatically determines the minimal set of features" is a mischaracterization**: The cardinality of the selected set is controlled by the hyperparameter λ in L_total = L_task + λL_select. The paper concedes in the conclusion that "the balance between sparsity and accuracy [is] controlled through a single λ parameter," which directly contradicts the claim of automatic, minimal determination. λ is a sparsity-accuracy trade-off knob, analogous to the k parameter in top-k methods or the regularization coefficient in L1 methods — the method does not eliminate the need for hyperparameter tuning, it merely shifts it from k to λ. Additionally, the "minimal" claim is contradicted by the metagenomic results (e.g., KeohaneDM_2020 MLP accuracy drops from 0.469 to 0.344; ThomasAM_2018a from 0.733 to 0.567), showing the selected set is not always sufficient.

- **"Nearly constant computational overhead regardless of input dimensionality" is overstated**: The claim appears in the abstract, introduction, and contributions. Conceptually, each forward pass must compute a D-dimensional mask and process D-dimensional inputs, which is O(D). The empirical claim of α ≈ 0.08 (Sec. 4.3) is a curve fit in a limited range that depends on implementation details (e.g., whether network hidden sizes are held fixed). The paper provides no theoretical justification for near-constant scaling, and the improvement over baselines is more accurately described as "better scaling exponent" rather than a qualitatively different complexity class. This overclaim is structurally embedded in the paper's narrative.

- **Missing comparison with closely related differentiable neural FS methods**: The paper discusses STG (Yamada et al., 2020), Concrete Autoencoders (Balın et al., 2019), INVASE (Yoon et al., 2018), and LassoNet (Lemhadri et al., 2021) in related work, but the experimental comparison includes only "10 established FS methods" whose identities are deferred to Appendix C (not in the main text). These architecturally similar differentiable methods are the most natural baselines and their absence from the experimental results weakens the claim of superiority. Without comparing against STG or Hard-Concrete gates with the same cardinality penalty framework, it is unclear what the Gumbel-Sigmoid specifically contributes over existing differentiable FS approaches.

### Minor

- **λ sensitivity analysis is deferred to Appendix**: Given that λ is the critical hyperparameter controlling the sparsity-accuracy trade-off, the fact that the main text only states "λ = 1 gives satisfactory results across datasets" without showing the sensitivity analysis is a gap. A single fixed λ value working across datasets is an important claim that should be supported in the main text.

- **No variance or stability analysis**: Results are reported as single numbers (Tables 2-5), without standard deviations across multiple runs. Given the stochasticity from Gumbel noise and random initialization, feature selection results could vary significantly across seeds. For a method claiming interpretability through its selected features, stability of those features across runs is important.

- **Global mask limitation is not discussed**: AutoNFS learns one mask for all instances (through a single learned embedding e). While this makes the method efficient, it means AutoNFS cannot perform instance-specific feature selection, which may be suboptimal for heterogeneous data. The related work section discusses INVASE (instance-specific) but no comparison or discussion of when a global mask is insufficient is provided.

- **Metagenomic results show substantial per-dataset variance**: While averages are reported as improvement, individual datasets show large drops (KeohaneDM_2020 MLP: 0.469→0.344, ThomasAM_2018a MLP: 0.733→0.567), which undermines the claim that AutoNFS "maintains predictive performance."

### Trivial

- The distinction between Gumbel-Sigmoid and the Gumbel-Softmax/Concrete distributions used in prior work is acknowledged but could be more explicit about the specific advantages over Hard-Concrete gates and STG's Gaussian gates.

## Nice-to-Haves

- Running baselines at matched feature counts (e.g., same k as AutoNFS selects) to isolate selection quality from budget effects.
- An ablation comparing Gumbel-Sigmoid vs. Hard-Concrete gates vs. STG's Gaussian gates with the same cardinality penalty framework.
- Stability analysis (e.g., Jaccard index of selected features across seeds) to support interpretability claims.
- Evaluation on datasets with >>1K features (e.g., gene expression, text classification) to substantiate scalability claims beyond the 136-718 feature range tested.

## Removed Points

- **"Insufficient architectural details" (reproducibility concern)**: The paper references Appendix C for setup details and provides Algorithm 1 with specific hyperparameters (τ₀=2.0, α=0.997, λ=1). While the masking/task network architectures aren't fully in the main text, this is a standard completeness concern, not a fundamental methodological gap.

- **"The masking network takes a single learned embedding vector" is "unusual"**: While the global embedding design is somewhat unusual, it is clearly described and motivated by the goal of global (not instance-specific) feature selection. This is a design choice with trade-offs, not a flaw.

- **Demands for theoretical guarantees of convergence or support recovery**: The paper is primarily an empirical contribution. Requiring theoretical proofs would go beyond the scope of an empirical methods paper. The lack of theory does not invalidate the empirical findings, though the stronger theoretical claims (automatic minimality) should be removed.

- **"No details on how dimensionality is varied" (computational complexity)**: While the methodology of the complexity analysis could be more precise, the empirical demonstration of near-constant scaling is a valid practical contribution even without complete methodological specification of the scaling experiments.

## Novel Insights

The most notable observation from reviewing this paper is the fundamental tension between the "automatic" framing and the architectural reality: AutoNFS replaces the problem of specifying k (number of features) with the problem of specifying λ (sparsity-accuracy trade-off). While this reframing has practical value — λ is a continuous parameter that can be more intuitively tuned than an integer budget, and the end-to-end training allows k to emerge from optimization — it does not constitute truly automatic feature count discovery. The paper would be stronger if it honestly positioned λ as a more convenient and principled hyperparameter than k, rather than claiming the feature count is "automatically determined."

## Suggestions

- **Most critical**: Re-run the OpenML benchmark experiments with baselines allowed to select the same number of features that AutoNFS selects (or at multiple budgets), to enable a fair comparison at matched sparsity levels. This single experiment would dramatically strengthen or weaken the paper's core claims.
- Replace all instances of "automatically determines the minimal set" with more accurate language like "learns a sparse feature set controlled by the λ parameter" or "automatically determines feature count via a sparsity-accuracy trade-off."
- Replace "nearly constant computational overhead" with "efficient scaling" or similar language that reflects the empirical finding without implying O(1) complexity.
- Add STG and/or Concrete Autoencoders as direct neural FS baselines to demonstrate the specific advantage of the Gumbel-Sigmoid design.
- Report mean ± std over multiple random seeds, at minimum for the main benchmark results.

## Score and Decision

**Calibration**: Papers with similar issues (overclaiming + unfair comparisons + incremental novelty) like LCEN (scores 5,5,5,1 → rejected) and the Gumbel-Softmax band selection paper (scores 5,5,3,3 → rejected) were in the reject range. The Decoupled ST-GS paper (similar Gumbel-based method, scores 3,6,5,3) was also rejected. AutoNFS has stronger empirical evaluation than some of these (diverse benchmarks, real-world data, good selection quality analysis), but its core method is incrementally novel over STG/Hard-Concrete + cardinality penalty, and the overclaiming on both "automatic minimality" and "constant overhead" significantly weakens the contribution. The unfair baseline comparison is the most damaging issue as it undermines the central experimental claims.

Against higher-quality tabular DL papers like TabM (scores 8,6,6,6 → accepted poster) and TabReD (3,8,8,8 → accepted spotlight), AutoNFS falls short in both contribution depth and experimental rigor.

I rate this paper at **4.5** — it has a reasonable method with useful empirical characteristics, but the overclaimed contributions and confounded experimental comparison prevent accepting it as-is.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>