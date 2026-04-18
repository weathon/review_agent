Now I have enough context from the calibration papers. Let me write the final review.

## Summary
This paper presents a systematic empirical study of design choices for diffusion-based planning in offline reinforcement learning, training and evaluating over 6,000 models across D4RL benchmarks (Maze2D, AntMaze, Franka Kitchen). The study identifies several counter-intuitive findings—unconditional sampling with selection (MCSS) can outperform guided sampling, Transformers outperform U-Net backbones, separate inverse dynamics outperforms joint state-action modeling—and proposes Diffusion Veteran (DV), a simple baseline achieving strong results on these benchmarks.

## Strengths

- **Scale and thoroughness of empirical study.** Training and evaluating over 6,000 diffusion models across multiple design axes (guided sampling, architecture, action generation, planning strategy) represents a substantial empirical effort. This scale enables more reliable comparative conclusions than typical single-design-point papers. The one-variable-at-a-time ablation methodology (Section 3.2) is a principled approach for isolating component effects.

- **Counter-intuitive and practically valuable findings.** The discovery that MCSS outperforms classifier/classifier-free guidance (Section 4.5, Fig. 7) directly challenges prevailing practice in diffusion planning. Similarly, the finding that Transformers outperform U-Nets (Section 4.3, Fig. 5a) and that separate inverse dynamics beats joint state-action modeling (Section 4.1, Fig. 3) in higher-dimensional action spaces are actionable insights for practitioners. These results are well-supported by the data presented.

- **Clean and useful baseline.** DV is conceptually simple (Algorithm 1) and achieves strong performance across all three task categories (Kitchen: 83.8, AntMaze: 83.2, Maze2D: 163.6 in Table 1). The transparent acknowledgment that diffusion policy outperforms diffusion planning on MuJoCo locomotion (Fig. 8) adds credibility.

- **Insightful analysis of domain suitability.** Section 4.6 provides a useful functional characterization of when diffusion planning vs. diffusion policy should be preferred (planning-heavy, sparse-reward tasks vs. dense-reward locomotion), supported by concrete experimental evidence.

## Weaknesses

### Fatal
None.

### Major

- **Inconsistent jump-step claim vs. data presented.** Section 4.2 states "jump-step planning is beneficial in almost all cases," yet Figure 4 shows DV's optimal stride is 1 (i.e., dense-step planning), with performance generally decreasing as stride increases across all environments. The paper references Appendix D for supporting evidence but the main text figure directly contradicts the headline claim. This creates genuine confusion about the actual finding: is the takeaway that stride>1 helps (which Fig. 4 does not show for DV), or that one should sweep stride (a weaker claim)? The practical takeaway (Takeaway 3—"experimenting with different planning strides is encouraged") is reasonable, but the broader claim is unsupported by the evidence in the main paper.

- **SOTA claim with heterogeneous baseline comparison and missing variance.** Table 1 aggregates baseline numbers "obtained from literature" while DV results are newly computed "averaged over 500 episode seeds." Variance is explicitly omitted ("We omit the variance over seeds for simplicity"). This matters because some performance margins over strong baselines are moderate (e.g., DV scores 83.8 Kitchen avg vs. HD's 72.5 on Kitchen-mixed, but HD scores 88.7 on AntMaze-medium, and DV vs. IDQL* differences on some tasks are smaller). Without variance estimates or statistical tests, it is difficult to assess whether observed improvements are meaningful rather than due to random seed variation. The appendix reportedly contains variance information, but for a paper whose central claim is SOTA performance, this should be in the main table.

- **No controlled ablations showing which DV components actually drive improvements over prior methods.** DV packages several design choices simultaneously (Transformer, MCSS, inverse dynamics, jump-step). While Section 4 ablates each component from DV's best configuration, no experiment takes the strongest prior method (e.g., HD or DD) and incrementally adds DV's design choices to see which actually close the gap. This makes it impossible to distinguish whether DV's gains come from genuinely better principles (e.g., MCSS being superior) or from combining modern architectural choices (Transformer) with more tuning budget than prior work invested.

### Minor

- **MCSS explanation is plausible but unvalidated.** Section 4.5 hypothesizes that MCSS outperforms guidance "if the dataset contains a substantial portion of expert demonstration," supported by value distribution histograms (Fig. 7b). This is a reasonable hypothesis but lacks controlled experiments (e.g., systematically varying expert data fraction in a single environment) to confirm causation. The hypothesis is appropriately qualified ("we can propose a hypothesis"), which mitigates this—however, the practical implications (Takeaway 7) are stated more categorically than the evidence warrants.

- **Computational cost of MCSS is not discussed.** MCSS requires sampling N trajectories and selecting the best, scaling inference cost by N. The paper does not quantify or discuss this tradeoff, which matters for practical deployment. This is acknowledged in Section 5 ("Computational efficiency" paragraph) but only to defer to future work, with no numbers provided.

- **"Separate vs. joint" conclusion may not generalize beyond the tested configurations.** The recommendation to use separate inverse dynamics for high-dimensional action spaces (Section 4.1) is based on comparing one specific joint model against one specific separate model. The joint model may simply be under-parameterized for the higher-dimensional output. Without capacity-matched experiments or analysis of failure modes, the general principle is supported only for the tested configurations.

- **Adroit validation relegated to appendix without main-text summary.** The claim of generalizability beyond D4RL (Section 4.7) deserves at least a summary table or key trends in the main text, not just a promise that "results are deferred to Appendix C."

### Trivial

- The neuroscientific analogy in Section 4.2 (prefrontal vs. motor cortex timescales) is engaging but purely anecdotal—not grounded in any empirical measurement from the model. This is fine as discussion flavor but should not be read as evidence.

## Nice-to-Haves

- Controlled experiments varying dataset quality/composition to validate the MCSS hypothesis, making the practical guidance more precise (e.g., what constitutes "substantial" expert data, and where is the crossover point?).
- Inference cost comparison (FLOPs or wall-clock time) between MCSS with different N values, CG, and CFG, enabling practitioners to make informed speed/quality tradeoffs.
- Interaction analysis between components (e.g., does MCSS benefit more from Transformer backbone than from U-Net?) to strengthen the generalizability of the takeaways.

## Removed Points

- **"CFG implementation details are not in main text"** — Relegating implementation details to the appendix is standard practice and not a weakness. The paper adequately describes CFG conceptually and references the appendix for specifics.
- **"The paper doesn't test on vision-based tasks"** — The paper explicitly scopes to state-based offline RL (Section 3.3) and acknowledges this limitation in Section 5 ("vision-based decision making... remains an open problem"). Criticizing absence of work outside stated scope is scope creep.
- **"Diffusion planning vs diffusion policy comparison is asymmetric"** — The paper does not claim a rigorous paradigm-level comparison; Section 4.6 presents an empirical observation about domain suitability. The System 1/System 2 discussion (Section 5) is explicitly speculative and positioned as future work direction.
- **"Depth scaling claims are speculative"** — The paper itself qualifies these claims ("This may be due to... which requires further study to systematically address"), so this is not overclaiming.
- **"6000 models claim needs more transparency about search space"** — While valid in principle, the paper describes its search procedure in Section 3.2 and references Appendix B for hyperparameter details, which is standard for empirical studies.
- **"Unfair comparison favors DV over baselines because DV uses modern design choices"** — Per the hard rules, this asymmetry actually proves a stronger point (that combining correct design choices matters), so this is not a valid weakness.

## Novel Insights

The most novel and impactful finding is the empirical evidence that unconditional sampling with selection (MCSS) can outperform classifier guidance and classifier-free guidance in diffusion planning for offline RL. This inverts the prevailing assumption in the diffusion planning community that guided generation is essential, and the hypothesized connection to dataset expert quality—while not experimentally validated—is a testable and thought-provoking starting point for future work. The finding that Transformers consistently outperform 1-D U-Nets in this domain is also notable and aligns with trends in image generation, but carries the additional observation that long-range attention patterns in Transformers may be particularly valuable for planning tasks with long-horizon credit assignment.

## Suggestions

- **Clarify the jump-step finding.** Either revise the claim in Section 4.2 to reflect that the main evidence shows stride>1 does NOT help DV specifically (the star is at stride=1), with the caveat that Appendix D shows benefits in other configurations, or present a figure in the main text that actually demonstrates the benefit of stride>1.
- **Add variance to Table 1.** Even a ± format in parentheses would allow readers to assess the significance of DV's improvements over the closest baselines (HD, IDQL*).
- **Report inference cost.** A simple table of inference time per decision for MCSS (with different N), CG, and CFG would greatly strengthen practical applicability claims.
- **Summarize Adroit results in the main text.** A brief table or 2–3 sentences highlighting consistent/inconsistent findings would make the generalizability claim more credible without consuming much space.

## Score and Decision

**Calibration comparison:**

- **ChemRLformer** (Accept, scores 8/6/8/6): Systematic empirical study of RL design choices for molecular design, achieves SOTA, provides actionable insights. Similar scope and contribution type to the current paper. The current paper is comparable in empirical effort but has more overclaiming issues.

- **EDM2+** (Reject, scores 5/6/6/3): Design space exploration for diffusion model architecture. Similar empirical study nature, but less impactful findings and limited novelty. The current paper has stronger and more counterintuitive findings.

- **Efficient Planning with Latent Diffusion** (Accept poster, scores 8/5/8/6): Novel algorithm for diffusion-based offline RL planning with strong results. More algorithmic novelty than the current paper, which is primarily empirical.

- **DMEMM** (Reject, scores 3/6/3/3): Diffusion planning paper with weak SOTA claims and limited novelty. The current paper is substantially stronger—larger empirical scope, clearer insights, more honest assessment of limitations.

The current paper sits between EDM2+ (rejected for being incremental) and Efficient Planning with Latent Diffusion (accepted for novel algorithm + strong results). Its main contribution is empirical, not algorithmic, which limits its novelty compared to LatentDiffuser. But it provides genuine, counterintuitive insights that will influence how practitioners build diffusion planners. The overclaiming issues (jump-step, SOTA without variance, lack of controlled baselines) are significant but not fatal—they mainly affect how the results should be interpreted rather than whether the results are real. The paper would be substantially stronger with more careful claim qualification and variance reporting.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>