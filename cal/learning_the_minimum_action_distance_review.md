=== CALIBRATION EXAMPLE 30 ===

# Final Consolidated Review
## Summary

This paper proposes two algorithms (MadDist and TDMadDist) for learning the Minimum Action Distance (MAD)—the minimum number of actions to transition between states—from state trajectories alone, without action labels or rewards. The key innovations are: (1) supporting asymmetric distance metrics (quasimetrics) to capture directional structure in environments with irreversible dynamics, (2) a novel simple ReLU-based quasimetric ($d_{simple}$), and (3) a benchmark suite of environments with known ground-truth MAD for controlled evaluation.

## Strengths

- **Principled formulation with rigorous foundations.** The MAD is defined as the solution to a constrained optimization problem (Eq. 1), and the paper provides a complete uniqueness proof (Appendix A). This establishes a clean theoretical grounding that prior work on MAD approximation lacked.

- **Explicit handling of asymmetry is both well-motivated and empirically validated.** The paper correctly identifies that symmetric distance metrics fundamentally cannot capture MAD in environments with irreversible dynamics (KeyDoorGridWorld, CliffWalking). Figure 3 shows Hilbert (symmetric) catastrophically fails on these environments while quasimetric methods succeed—a result that is not merely expected but substantively demonstrates the necessity of the design choice.

- **The $d_{simple}$ quasimetric is an effective, parsimonious contribution.** Appendix E.2 shows $d_{simple}$ consistently outperforms both Wide Norm and IQE within the MadDist framework, despite being structurally simpler. The proof that it satisfies the triangle inequality (Appendix B) is clean. This is a case where less is more—simpler inductive bias better matches the problem structure.

- **Benchmark suite with known ground-truth MAD enables rigorous evaluation.** The environments span discrete/continuous state spaces, deterministic/stochastic dynamics, noisy observations, and symmetric/asymmetric transition structures (Appendix G). This is a genuine contribution that future work can build on.

## Weaknesses

### Major:

- **Statistical inconsistency in reported experimental results.** Section 7 states "All reported results are means over **five** independent runs," yet Figure 3, Figure 11, and Figure 12 captions state "Shaded regions minimum and maximum values across **three** random seeds." Table 1 reports standard deviations without specifying the number of runs. It is unclear whether the reported means and deviations are over 3 or 5 seeds, and why figures and text disagree. This directly undermines confidence in the statistical robustness of the claims.

- **Critical planning experiment parameter ($H$) is missing, making results unreproducible and difficult to interpret.** Appendix H describes a random-shooting MPC planner with $K{=}100$ candidate action sequences "of length $H$," but **never specifies the value of $H$**. Table 1 reports perfect success rates (1.00 ± 0.00) for MadDist on PM Large Navigate/Stitch environments. In a large maze, achieving 100% success with only $K{=}100$ random action sequences is surprising—the result is entirely conditioned on the unspecified horizon $H$. Without this value, the planning evaluation is not reproducible and the perfect scores cannot be properly assessed.

- **TDMadDist consistently underperforms MadDist without adequate analysis.** TDMadDist achieves lower success rates than MadDist on 5 of 6 OGBench environments (Table 1) and lower correlations in Figure 3. The paper states this fact but provides no analysis of *why* TD bootstrapping hurts performance—whether due to target network instability, bootstrapping bias in the distance domain, or fundamental incompatibility with the loss formulation. Since TDMadDist is presented as one of two core algorithmic contributions, its consistent underperformance needs explanation.

### Minor:

- **Limited baseline comparisons.** Only QRL and Hilbert are evaluated. The related work section discusses successor features (Dayan, 1993; Myers et al., 2024), time-contrastive representations (Eysenbach et al., 2022), Laplacian-based methods (Wu et al., 2019; Wang et al., 2021), and bisimulation approaches (Dadashi et al., 2021; Agarwal et al., 2021)—none of which appear in experiments. While these measure different quantities than MAD (e.g., diffusion similarity, on-policy visitations), comparing against them would clarify whether MAD-specific inductive bias offers unique advantages over other temporal distance notions.

- **No ablation on individual loss components.** The MadDist objective combines three terms ($L_o + w_r L_r + w_c L_c$), each serving a distinct purpose (objective matching, contrastive separation, constraint enforcement). While the paper ablates quasimetric choice and latent dimension (Appendix E), it does not isolate the contribution of each loss term. The contrastive loss $L_r$ in particular introduces the hyperparameter $d_{max} \in \{100, 500\}$—its sensitivity and necessity are not analyzed.

- **Downstream evaluation is limited to a single task type.** The introduction motivates MAD for policy learning, reward shaping, and option discovery, but only goal-reaching planning is evaluated. Whether accurate MAD translates to improved performance on reward shaping or hierarchical RL remains unverified.

- **No evaluation on high-dimensional or image-based observations.** All environments use low-dimensional vector observations (2D–4D). The method's applicability to the realistic setting of learning MAD from pixel observations—a setting where representation learning matters most—is not demonstrated.

### Trivial:

- The constraint horizon $H_c{=}6$ means explicit upper-bound penalties only apply to pairs within 6 steps on a trajectory. Long-range consistency relies entirely on the quasimetric's inherent triangle inequality. This works empirically but the choice of 6 is not analyzed.

## Nice-to-Haves

- End-to-end goal-conditioned RL experiment using learned MAD as reward/distance heuristic (the paper claims this as a key application but provides no such evidence).
- Theoretical convergence or approximation bounds for MadDist/TDMadDist.
- Comparison to successor feature or Laplacian-based temporal distance methods to clarify MAD's unique benefits.
- Visualization of learned embedding asymmetry (e.g., directed arrows showing $d(s,s') \neq d(s',s)$) to confirm the quasimetric captures asymmetry rather than just fitting regression targets.
- Experiments with limited data coverage or distribution shift between training and evaluation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Evaluation limited to toy/synthetic environments"** — The environments are deliberately designed for controlled evaluation with known ground truth, which is a feature enabling rigorous quantitative assessment. This is not a flaw; the real gap is the lack of high-dimensional observations, which is already captured above.

- **"Lack of theoretical convergence analysis"** — This is an empirical paper with a theoretical formulation (MAD as constrained optimization). Demanding convergence proofs is scope creep beyond the paper's stated contributions.

- **"Non-Markovian observations in NoisyGridWorld"** — The i.i.d. Gaussian noise does not actually make the observation process non-Markovian (noise is independent at each step). The encoder learns to ignore it, which is demonstrated empirically. Not a real weakness.

- **"QRL may be under-tuned for long-horizon tasks"** — The paper reports that short-horizon QRL hyperparameters performed best overall, including on long-horizon environments. This is an empirical finding, not an unfair comparison. Per rules, criticisms of unfair comparison where asymmetry favors the baseline are removed.

- **"Relative vs. absolute loss scaling justification"** — The paper provides a reasonable justification for scale-invariant loss ("scaling makes the loss invariant to the magnitude of the estimation error, which typically increases as a function of $j{-}i$"). While an ablation would strengthen this, the design choice is well-motivated.

- **Formatting/style nitpicks** and **reproducibility complaints about hyperparameters** — Removed per hard rules.

## Novel Insights

The consistent underperformance of TDMadDist relative to MadDist reveals an interesting tension: bootstrapping—typically beneficial in value function approximation—appears harmful in the distance estimation setting. This may be because distance targets are not stationary in the same way value targets are (the embedding space geometry changes during training, making bootstrapped targets unreliable), or because the TD objective's `min(j-i, 1+d')` truncation introduces bias when `d'` is underestimated early in training. This finding suggests that the benefits of temporal difference learning do not straightforwardly transfer from reward estimation to geometric distance estimation, a distinction the community should be aware of.

## Suggestions

- **Immediately resolve the seed count inconsistency** (Section 7 says 5, figure captions say 3). Clarify which is correct and ensure all results use consistent reporting.
- **Specify the planning horizon $H$** in Appendix H and discuss why perfect success rates are achievable with $K{=}100$ random samples.
- **Add a loss component ablation** removing $L_r$ and $L_c$ individually to quantify their contributions, especially given the sensitivity to $d_{max}$.
- **Analyze why TDMadDist underperforms**—even a simple analysis of target stability or bias during training would illuminate whether TD is fundamentally ill-suited here or just needs better tuning.
- **Include at least one experiment with higher-dimensional observations** (e.g., D4RL AntMaze with proprioceptive state) to demonstrate the method scales beyond coordinate-based inputs.

## Evaluation Summary

- **Novelty:** Moderate—$d_{simple}$ is a clean, effective contribution; the algorithmic framework combines known ideas (trajectory upper bounds from Steccanella & Jonsson 2022, quasimetrics) in a well-engineered way. The TD variant adds little.
- **Technical soundness:** Generally sound, but undermined by the statistical inconsistency and missing planning parameter.
- **Empirical support:** Strong on representation quality metrics (correlation, CV); weaker on downstream utility due to the planning evaluation's reproducibility gaps.
- **Significance:** The benchmark suite and the demonstration that simple quasimetrics outperform complex ones are valuable. The practical impact is currently bounded by limited downstream evaluation.
- **Clarity:** Good overall, but the seed inconsistency and missing $H$ are significant omissions for a paper centered on empirical claims.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 2.0, 4.0]
Average score: 4.5
Binary outcome: Reject
