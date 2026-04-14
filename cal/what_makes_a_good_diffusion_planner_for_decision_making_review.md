=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary

This paper conducts a large-scale empirical investigation into the design choices of diffusion planners for offline reinforcement learning, training and evaluating over 6,000 diffusion models across Kitchen, AntMaze, and Maze2D benchmarks. The study evaluates four key components—action generation strategy, planning stride, denoising network backbone, and guided sampling algorithm—and yields several counter-intuitive insights (e.g., Transformer beats U-Net, unconditional Monte Carlo sampling with selection (MCSS) outperforms classifier/classifier-free guidance). Building on these findings, the authors propose Diffusion Veteran (DV), a simple baseline that achieves state-of-the-art performance among diffusion planners on D4RL.

---

## Strengths

- **Scale of systematic ablation rarely seen in offline RL.** Training and evaluating 6,000+ diffusion models across multiple task families, with DV results averaged over 500 episode seeds, gives the findings substantially more statistical grounding than typical single-run ablations in this space.

- **Counter-intuitive, field-correcting findings.** The finding that MCSS (unconditional generation + selection) outperforms CG and CFG—which are the dominant choices in prior diffusion planning work—is a genuine, actionable surprise. The accompanying dataset-quality hypothesis (Figure 7b), showing that environments with more near-optimal demonstrations favor MCSS, provides a principled (if observational) explanation.

- **Transformer-over-U-Net with mechanistic insight.** The result that Transformer outperforms U-Net in 8/9 sub-tasks is backed by attention weight visualizations (Figure 5b) revealing that (a) the model preferentially attends to long-range trajectory elements, consistent with U-Net's local-convolutional bias being a bottleneck, and (b) the characteristic attention length is stride-invariant (6 steps × stride 4 ≈ 25 steps × stride 1), suggesting the Transformer discovers an environment-level temporal abstraction rather than an artifact of the planning horizon.

- **Principled action-space analysis for Separate vs. Joint action generation.** The finding that "Separate" (inverse dynamics) substantially outperforms "Joint" in higher-dimensional action spaces (e.g., AntMaze-M-D: 87.4 vs. 36.3) is well-supported by quantitative results across multiple sub-tasks, and the explanation—joint state-action modeling is increasingly ill-conditioned as action dimension grows—is intuitive and transferable.

- **Honest characterization of the diffusion planning/policy frontier.** Section 4.6 clearly identifies where diffusion planning fails (MuJoCo locomotion), providing a practical scope boundary rather than overselling the method.

---

## Weaknesses

### Fatal
None.

### Major

- **Internal contradiction between Takeaway 3 and Figure 4.** Section 4.2 states: *"One crucial result we found is that jump-step planning is beneficial in almost all cases"* and cites Figure 4 as evidence. Yet Figure 4's own caption states *"performance generally decreases as the planner stride increases"* and marks DV's choice as Stride=1 (dense-step). Takeaway 3 then recommends implementing jump-step planning. The paper does cite Appendix D for "extensive results" supporting the general claim, but the only evidence shown in the main text (Figure 4) directly contradicts it for the DV configuration. This creates a serious credibility problem: readers cannot know whether the jump-step recommendation applies to U-Net-based planners, other architectures, or DV itself. The paper must either (a) clarify that jump-step helps other architectures but not DV, and explain the interaction, or (b) re-examine whether the claim is correct for DV at all. As-is, Takeaway 3 and Figure 4 give opposite recommendations.

- **MCSS computational cost is not analyzed.** MCSS requires generating N full diffusion rollouts per step, while CG and CFG require a single rollout with gradient guidance. The paper recommends MCSS as the preferred guidance strategy but does not provide any inference-time comparison—not FLOPs, not wall-clock time, not even a note on what N is used. The value of the MCSS recommendation is significantly undercut without knowing how much more expensive it is: if it requires N=10× more compute than CG to match or exceed it, a practitioner needs to know this. Algorithm 1 lists N as a hyperparameter but never specifies its value or sensitivity.

### Minor

- **DV's depth=2 choice is not justified given Figure 6.** Figure 6 shows that depth=4 outperforms depth=2 on Kitchen, and depth increases appear uniformly helpful on AntMaze, yet DV uses depth=2. The paper states only "a deeper model is not always better"—which does not explain why depth=2 was selected as the DV default rather than, say, depth=4. If the choice is based on average across all tasks, that aggregation should be shown.

- **Inverse dynamics ablation asserted without data.** Section 4.1 states: *"We tested both diffusion models and vanilla MLP as the inverse dynamics, and found similar performance between them."* This is an architecturally relevant finding—DV uses diffusion inverse dynamics—but no quantitative evidence is provided for this claim. Given that it justifies a design decision in DV, at least a brief table or figure should accompany it.

- **Dataset quality hypothesis for MCSS is observational.** The explanation that MCSS benefits from near-optimal data (Figure 7b) is interesting and plausible, but it is supported only by correlational evidence. No controlled variation of dataset composition is performed to test the hypothesis. The paper should be explicit that this is a hypothesis, not a validated finding, especially since Takeaway 7 presents it as actionable guidance.

- **Parameter parity between U-Net and Transformer stated but not shown.** The paper notes "the amount of parameters in U-Net is comparable to that in Transformers" (Section 4.3 caption) without reporting actual counts or FLOPs. If the Transformer has meaningfully different memory or training-time cost for a similar parameter count, this should be reported.

- **Adroit generalization results deferred entirely to appendix.** Section 4.7 asserts generalizability to the Adroit dataset but provides no quantitative summary in the main text. For a paper whose core contribution is empirical insights, a single summary table or bar chart of Adroit results should appear in the main body.

### Tiny

- The control variable methodology (fix best config, vary one factor) cannot capture interaction effects between components (e.g., whether Transformer is better than U-Net specifically in combination with MCSS, or also with CG). The paper does not acknowledge this limitation. A brief note would improve transparency.

- DV is compared to baselines using their published numbers, while DV itself was selected after sweeping 6,000 models. This comparison is not perfectly controlled; the authors should acknowledge that some of DV's advantage over reported baselines may reflect hyperparameter tuning effort rather than strictly architectural superiority.

---

## Nice-to-Haves

- A compute-normalized comparison between MCSS and CG/CFG (e.g., fixing the total number of diffusion function evaluations per environment step) would substantially strengthen the MCSS recommendation.
- An analysis of component interactions—specifically, whether the Transformer advantage holds when using CG/CFG guidance rather than MCSS—would help disentangle architectural from sampling-strategy effects.
- Synthetic dataset quality experiments (mixing expert and random data in controlled proportions) to test the MCSS hypothesis directly would elevate the dataset quality finding from hypothesis to validated claim.
- A brief summary table of Adroit results in Section 4.7 would make the generalizability claim more credible without requiring readers to consult an appendix.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[REMOVED] Lack of MuJoCo locomotion ablations.** The harsh critic raises the absence of per-component ablations on locomotion tasks. However, the paper explicitly scopes its ablations to planning-oriented task families, explicitly notes in Section 4.6 that diffusion planning underperforms on locomotion (and provides a plausible explanation), and lists locomotion extension as a future direction. Criticizing the absence of locomotion ablations within a paper that identifies locomotion as outside its intended scope is scope creep.

- **[REMOVED] Sim2Real transfer demanded.** The spark finder suggests at least one real-robot experiment. This is entirely outside the paper's stated scope (offline RL benchmarks on D4RL) and would not be expected for this type of empirical study at ICLR.

- **[REMOVED] Demand for statistical significance tests (t-tests, p-values) on ablations.** For RL benchmarks where single-run evaluation with 500+ seeds is already rare and commendable, demanding formal significance testing is not standard. The paper's 500-seed evaluation for DV is already rigorous.

- **[REMOVED] Reproducing competing baselines under identical compute budgets.** Using literature-reported numbers for comparisons is standard practice in offline RL empirical papers. The comparison is asymmetric in ways that favor baselines (they haven't been through the same tuning pipeline), making DV's gains conservative rather than inflated.

- **[REMOVED] "Unfair" MCSS advantage from using two models (planner + inverse dynamics).** The spark finder suggests the "Separate" approach has an unfair capacity advantage. However, the paper controls the comparison by using comparable parameter counts, and inverse dynamics networks in this setting are typically small (conditioned on two consecutive states, predicting one action). The concern is not substantiated.

- **[REMOVED] Criticism of the speculative neuroscience analogies.** The System 1/System 2 and prefrontal cortex analogies in Sections 4.2 and 5 are speculative but clearly framed as interpretive and motivational, not empirical claims. This is a style preference, not a substantive error.

---

## Novel Insights

The most genuinely novel insight across all three reviews—and one worth amplifying—concerns the **stride-invariant attention length** finding in Section 4.3. The observation that the Transformer's characteristic attention horizon in environment-clock-time remains constant regardless of planning stride (6 steps × stride-4 ≈ 25 steps × stride-1) suggests that the Transformer is not merely capturing local trajectory correlations but is discovering an intrinsic temporal scale of the environment's dynamics. This is more than an architectural win: it implies that the planning model is learning an implicit temporal abstraction, which is a genuinely interesting finding for the intersection of sequential decision-making and Transformer research. Combined with the MCSS finding, the paper implicitly argues that diffusion planners can operate effectively as near-optimal trajectory samplers when given sufficient data coverage—making the "guidance" scaffolding less critical than previously assumed. These two findings together suggest that future diffusion planners should prioritize data quality and Transformer-based backbones over sophisticated guidance engineering.

---

## Suggestions

1. **Resolve the Stride=1 contradiction immediately.** In Section 4.2, explicitly state that Figure 4 shows jump-step planning does not benefit DV (Stride=1 is optimal for this architecture/configuration), but Appendix D results show it helps other configurations. Rewrite Takeaway 3 accordingly: e.g., "Jump-step planning can be beneficial, particularly for U-Net-based planners; for Transformer-based planners with MCSS, dense-step may be preferred."

2. **Add inference cost analysis for MCSS.** Report the value of N used in DV's MCSS, wall-clock inference time relative to CG/CFG, and a brief sensitivity analysis of performance vs. N. Even a one-paragraph analysis would address the most practically important question a reader would have before adopting DV.

3. **Justify the depth=2 choice for DV explicitly.** If depth=2 is chosen based on average performance across all tasks, show this aggregation (e.g., a bar chart averaging across all sub-tasks by depth). If chosen for efficiency, state so. The current presentation leaves this choice unexplained.

4. **Add at least one quantitative data point for the MLP vs. diffusion inverse dynamics claim.** The assertion in Section 4.1 that MLP and diffusion perform similarly as inverse dynamics models should be backed by at least a brief table, since DV's choice of diffusion inverse dynamics is otherwise unmotivated.

5. **Move a summary of Adroit results to the main text.** Even a 3-row table showing the key findings (e.g., Transformer > U-Net, Separate > Joint) replicated on Adroit would substantiate the generalizability claim without expanding the paper significantly.

---

**Evaluation Summary:**

- **Novelty:** Moderate-to-high for an empirical study; the MCSS-beats-guidance and Transformer-over-U-Net findings are genuine corrections to field consensus, and the stride-invariant attention length is a genuinely new observation.
- **Technical soundness:** Moderate; the control variable methodology is appropriate but the planning stride contradiction represents a real inconsistency in the paper's own recommendations that must be resolved.
- **Empirical support:** Generally strong given the scale of experimentation, but weakened by missing MCSS cost analysis, unquantified inverse dynamics ablation, and deferred Adroit results.
- **Significance:** High for practitioners; the practical takeaways—if the stride contradiction is resolved—provide clear, evidence-backed guidance for building diffusion planners.
- **Clarity:** Good overall, but Section 4.2 is substantially misleading as written, and the depth selection for DV lacks justification.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 10.0, 6.0]
Average score: 7.5
Binary outcome: Accept
