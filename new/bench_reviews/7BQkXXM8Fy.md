## Summary

This paper conducts a large-scale empirical study of design choices in diffusion planning for offline reinforcement learning, training over 6,000 models to systematically evaluate four key axes: guided sampling algorithms, denoising network architecture, action generation strategy, and planning stride. The work identifies that Transformers outperform U-Net backbones, separate inverse dynamics action generation outperforms joint state-action modeling for high-dimensional tasks, Monte Carlo unconditional sampling with selection (MCSS) can outperform gradient-based guidance, and dense-step planning (stride=1) is optimal for the proposed configuration. Based on these insights, it proposes the Diffusion Veteran (DV) baseline, which achieves state-of-the-art average scores across Kitchen, AntMaze, and Maze2D benchmarks.

## Strengths

- **Unprecedented empirical scale and systematic scope:** Training and evaluating 6,000+ models across three D4RL task families (Kitchen, AntMaze, Maze2D) while sweeping four orthogonal design axes provides a dense empirical map of the diffusion planning design space that individual ablation studies cannot match. (Sec. 3.2–3.3, Table 1)
- **Actionable, counter-intuitive findings grounded in data:** The paper demonstrates several results that challenge common practice in diffusion planning: (i) MCSS outperforms CG/CFG on most tasks (Fig. 7a), (ii) Transformer backbones outperform U-Net on 8 of 9 sub-tasks (Fig. 5a, Table on lines 219–232), and (iii) "separate" action generation via inverse dynamics dramatically outperforms "joint" modeling in high-dimensional action spaces (Kitchen Δ=+30.3 for Mixed, AntMaze Δ=+52.7 for M-D in Fig. 3).
- **Strong, simple baseline:** The DV configuration achieves the best normalized average scores across all three task families (Kitchen: 83.8, AntMaze: 83.2, Maze2D: 163.6 in Table 1), demonstrating that the identified design choices combine into a practically superior system.
- **Transparent and reproducible methodology:** The control-variable experimental procedure is clearly described (Sec. 3.2), Algorithm 1 provides full pseudocode for DV, and Appendix deferrals for implementation details (App. A–B) are standard for conference submissions.

## Weaknesses

### Fatal
None.

### Major

- **Contradiction between planning-stride claim and main-figure data (Sec. 4.2, Fig. 4, Takeaway 3):** The paper states that "jump-step planning (Sect. 3.1) is beneficial in almost all cases" (line 205) and frames this as motivation for longer-timescale planning (the neuroscientific analogy in line 211). However, Figure 4 — the paper's own primary figure for this axis — shows performance *monotonically decreasing* as stride increases from 1 to 4 in all environments, and the star indicating DV's choice is at Stride=1 (dense-step). Takeaway 3 similarly advises "Implementing jump-step planning can be highly beneficial" (line 305), despite the data showing the opposite within DV. The paper hedges by pointing to "extensive results" in Appendix D and suggesting readers "try jump-steps and sweep the stride" (line 209), but this does not resolve the core contradiction between the textual Takeaway and the evidence the paper itself presents. This undermines readers' confidence in the reliability of Takeaway 3 and risks misleading future researchers into exploring an unhelpful direction.

### Minor

- **Control-variable ablation measures conditional sensitivity, not main effects (Sec. 3.2):** The experimental design — "modify only one component of the best model at a time" (line 92) — evaluates how fragile the globally optimized DV configuration is to each individual design choice, rather than isolating the independent contribution of each component. In diffusion planning, component interactions are non-trivial (e.g., guidance effectiveness likely depends on backbone architecture and stride), so the broad generalizations presented as Takeaways 1–7 reflect *local observations conditional on the specific DV configuration* rather than general design principles. A factorial or partial-factorial design over at least the most critical axes would help disentangle main effects from interactions.
- **MCSS framing conflates unconditional sampling with value-based reranking (Sec. 4.5, Algorithm 1):** Algorithm 1 (lines 9–10) generates N candidate plans unconditionally, then selects the best using a critic network V_φ. This is Monte Carlo sampling with value-based selection, not unconditional diffusion. The paper's claim that "non-guidance can be better than guidance" (line 272) is somewhat of a mischaracterization: MCSS still relies on a separately trained value function as a ranking signal, which is a form of conditioning applied post-hoc. The comparison to CG/CFG is fair in spirit (no gradient guidance during the reverse process), but the framing should more precisely describe MCSS as value-based trajectory reranking to avoid overstating the "unconditional" aspect.
- **SOTA claim lacks matched-protocol baseline evaluation (Table 1):** DV's headline results are compared against literature-reported baseline numbers (Table 1, line 130: "The results of other methods are obtained from literature"). Offline RL on D4RL is notoriously sensitive to evaluation protocols — number of evaluation episodes, fixed vs. random seeds, reward normalization constants, and tuning budgets. Training and sweeping 6,000 models for DV while comparing against single-report literature baselines introduces potential selection bias. Reproducing 2–3 strongest baselines (e.g., HD, DQL*) under DV's exact evaluation pipeline would strengthen the SOTA claim. That said, this is a common limitation in empirical RL papers, not a fatal flaw.

### Trivial
None beyond what is covered above.

## Nice-to-Haves

- Report the training-time and parameter-cost difference between "joint" and "separate" (inverse dynamics) action generation, as DV adds a second diffusion model for inverse dynamics. This would help practitioners weigh the performance gain against compute overhead. (Sec. 4.1 mentions similar performance between MLP and diffusion inverse dynamics but does not quantify compute trade-offs.)
- Provide qualitative trajectory rollouts comparing Stride=1 vs. a higher stride on AntMaze or Kitchen to visually explain *why* the stride=1 setting is optimal, which would help readers reconcile the stride analysis with the paper's other takeaways.

## Removed Points

These points were flagged by the harsh critic but removed after verification against the paper:

- **Criticism of CG/CFG distinction in offline RL:** The critic claims "the distinction between classifier guidance and classifier-free guidance is underspecified for the offline RL context where 'classifiers' approximate value functions." The paper treats CG and CFG as established techniques from image generation (Sec. 2, lines 53–54) applied to offline RL with learned reward/value networks. This is standard in the diffusion RL literature, and the paper's treatment is consistent with prior work. Removed as scope-creep.
- **"Attention length observation is a trivial scaling artifact":** The critic argues that 6 steps × stride 4 ≈ 25 steps × stride 1 is simply a scaling artifact since the model attends to ~24 environment steps of context regardless. The paper's observation that the Transformer discovers stride-invariant temporal correlations (measured in planning-step indices that scale with stride) is a genuine finding — it shows the model adapts its attention window to maintain consistent temporal coverage. Removed as a disagreement with the paper's interpretation rather than a factual error.
- **Sustainability section as "post-hoc justification":** The critic dismisses the sustainability discussion (Sec. 5, line 327) as not adding scientific value. This is a style/positioning critique; the authors' framing that the energy investment provides a reusable foundation for future research is a reasonable discussion point. Removed as pure editorial opinion.
- **Demands for 2×2 factorial ablation matrix:** While the paper's control-variable method has limitations (addressed in Minor weaknesses above), demanding a full factorial design across all axes exceeds the paper's stated scope and is not standard practice for empirical configuration studies. Weakened to a Methodological note above.
- **Missing baseline re-runs under identical protocol:** This overlaps with the Minor weakness about protocol parity but is framed as a demand. Moved to the Minor section with softened framing, as this is a nice-to-have, not a core flaw.

## Novel Insights

One paragraph synthesizing genuinely novel observations.

The paper's most genuinely novel observation — beyond confirming existing practices — is the counter-intuitive result that MCSS (unconditional diffusion sampling followed by value-based selection) systematically outperforms gradient-guided sampling (CG/CFG) on planning-oriented tasks where the offline dataset contains substantial near-optimal trajectories (Fig. 7a–b). This challenges the prevailing assumption in diffusion planning that gradient guidance during generation is essential for reward maximization, suggesting instead that when the dataset is sufficiently rich, selecting from diverse unconditional samples is more effective than steering a single sample with potentially mis-specified gradient signals. This insight — paired with the empirical demonstration that "separate" inverse-dynamics action generation dramatically outperforms joint modeling in high-dimensional action spaces (up to +52.7 normalized points on AntMaze M-D) — provides genuinely actionable guidance for future diffusion planner design.

## Suggestions

1. **Revise Takeaway 3 and the planning-stride analysis:** Acknowledge explicitly that within the DV configuration on the main benchmarks, dense-step planning (stride=1) is optimal (as shown in Fig. 4). Qualify the jump-step recommendation as context-dependent — perhaps advantageous in other configurations or with different datasets — rather than broadly "beneficial in almost all cases."
2. **Clarify MCSS framing:** Re-label or reframe MCSS throughout as "value-based trajectory reranking" (or similar) to accurately reflect that it uses a critic network for selection, distinguishing it from pure unconditional sampling. This will also strengthen the comparison with CG/CFG.
3. **Add a discussion of ablation limitations:** In Sec. 4.8, briefly note that the takeaways are based on control-variable analysis from a specific optimized configuration, and that full interaction effects between components remain an open question. This will improve scientific rigor without undermining the paper's contributions.
4. **Include a brief compute-overhead note for inverse dynamics:** A sentence or short paragraph in Sec. 4.1 or the Discussion quantifying training time / parameter increase for the separate-action approach would help practitioners adopt the recommendation.

## Calibration and Scoring

I compared this paper against several anchors:
- **btpgDo4u4j (LatentDiffuser, scores 8, 5, 8, 6; Avg ≈ 6.75):** A diffusion planning paper on D4RL with a novel method (latent-space planning) and strong empirical results. That paper scored higher because it proposed a new *method*, not just a study. Our paper is purely empirical (no new algorithm), so it scores below.
- **tGQirjzddO (LatentDiffuser variant, scores 5, 8, 6; Avg ≈ 6.3):** Similar scope — diffusion modeling for offline RL — scoring in the 6 range for solid empirical contribution without a breakthrough novel method.
- **d8w0pmvXbZ (small-scale proxies, scores 8, 8, 8, 8; Avg = 8.0):** An exemplary systematic empirical study (oral acceptance). This paper scored at the top because it provided small-scale proxies useful for the entire ML community. Our paper, while systematic, is narrower (diffusion planning on D4RL) and has the stride contradiction.

The paper under review is an honest, thorough empirical study with real practical value for the diffusion planning community. Its main weaknesses — the stride contradiction and the conditional-sensitivity limitation — are substantive but do not invalidate the core contribution. It sits above borderline papers (e.g., 1zuJZ1jGvT, scores 5.5, avg, rejected for methodological issues) but below breakthrough novel-method papers in this space. Positioned against the ~6-range anchors for solid empirical studies, I score it slightly above due to the unusual scale (6,000 models) and actionable findings, but below the high-scoring anchors due to the contradiction and lack of novel methodology.

## Score and Decision

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>