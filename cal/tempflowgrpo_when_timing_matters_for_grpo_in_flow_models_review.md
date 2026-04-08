=== CALIBRATION EXAMPLE 43 ===

# Final Consolidated Review
## Summary

TempFlow-GRPO introduces temporal awareness into GRPO for flow matching models via three mechanisms: (1) trajectory branching, which attributes terminal rewards to specific timesteps by switching from ODE to SDE sampling at designated branching points, eliminating the need for trained process reward models; (2) noise-aware policy reweighting, which scales gradient contributions by the intrinsic noise level at each timestep to balance optimization across the generation trajectory; and (3) a seed group strategy that groups trajectories by shared initial noise to isolate exploration effects. The method demonstrates substantial improvements over Flow-GRPO on Geneval, PickScore, and HPDv2 benchmarks across multiple base models.

## Strengths

- **Trajectory branching as a parameter-free process reward mechanism.** The core insight—that switching from ODE to SDE at a single timestep while keeping the rest deterministic localizes the source of reward variation to that timestep—is creative and practically valuable. It avoids the well-known difficulty of training process reward models for semantically ambiguous intermediate states (Section 4.1.1). The "Credit Localization" property is intuitively sound: since all stochasticity is concentrated at the branching point, reward differences between branches are causally attributable to that point's exploration.

- **Theoretical grounding for noise-aware reweighting.** The policy gradient derivation (Section 4.2, Appendix A.1) identifies a concrete mechanism: in standard GRPO, the natural gradient scale term is proportional to $\frac{\Delta_k(1-k)}{k}$, which causes low-noise refinement steps to dominate optimization (Figure 5, Right). Reweighting by $\sigma_t\sqrt{\Delta_t}$ simplifies this scale term to $\Delta_k$, balancing gradient contributions. The empirical correlation between noise level and reward standard deviation (Figure 5, Left) provides independent support for using noise as a proxy for exploration capacity.

- **Breadth of empirical validation.** The paper tests across three base models (FLUX.1-dev, SD3.5-M, Qwen-Image), five reward models (PickScore, HPSv2, HPSv3, Geneval, OCR), and multiple resolutions (512 and 1024). Consistent improvements across all combinations (Figures 3, 7, 10, 11, 16) strengthen the claim that temporal awareness is a generally beneficial inductive bias rather than a niche trick.

## Weaknesses

- **Computational efficiency claims are inconsistent with acknowledged sampling overhead.** Appendix A.6 states that trajectory branching incurs ~4.5× more sampling per iteration than Flow-GRPO (for K=10), yet also claims "the training time per iteration remains identical to Flow-GRPO." These statements are contradictory: 4.5× more forward passes per iteration should increase iteration time unless there is unexplained parallelization or the backward pass dominates runtime. While Figures 3 and 12 show favorable GPU-hour curves, the accounting is opaque. Without clarification of how per-iteration cost can remain identical despite multiplicative sampling overhead, the efficiency claims are difficult to trust. This matters because computational cost is a primary selling point of the method.

- **The "Credit Localization" theorem overstates what is proven.** The proposition correctly notes that the *source* of randomness is isolated to the branching point $k$. However, the *magnitude* of reward change at $x_0$ depends on the Jacobian of the deterministic ODE flow from step $k$ to step 0. If the flow dynamics amplify perturbations (sensitive dependence), a small noise injection at $k$ could produce a large reward shift, over-crediting step $k$; if the flow dampens perturbations, step $k$ is under-credited. The paper assumes the reward variance magnitude correctly reflects timestep importance, but without normalizing by the flow's local sensitivity, the credit assignment may be biased by downstream dynamics rather than purely reflecting exploration value at step $k$. The noise-aware reweighting partially compensates, but the two mechanisms (branching for credit, reweighting for scale) are not clearly decoupled in theory.

- **Circular evaluation risk: training reward models double as evaluation metrics.** In the primary PickScore experiment, PickScore serves as both the training reward and the evaluation metric. Similarly, Geneval's reward model is used for both training and scoring. The near-perfect Geneval score of 0.97—even exceeding GPT-4o's 0.67—raises the question of whether this reflects genuine compositional improvement or optimization of the specific reward model. The multi-reward experiment (Table 2) and the cross-reward analysis in Appendix A.14 (PickScore drops when training on Geneval) provide partial evidence, but no evaluation on held-out compositional benchmarks (e.g., T2I-Bench, DrawBench) or human studies is included. For a paper claiming SOTA human preference alignment, this is a significant evidential gap.

- **Narrow baseline comparison in the main body.** The primary comparisons are against Flow-GRPO and its "Prompt" variant only. DanceGRPO—the most directly comparable concurrent method—is relegated to Appendix A.4. DPO-based alignment methods (e.g., Diffusion-DPO) are not compared at all. While the paper's scope is GRPO-based methods, the broad SOTA claims in the abstract and conclusion ("state-of-the-art performance in human preference alignment") are not supported by comparisons against the full spectrum of alignment approaches.

- **No analysis of failure modes or when temporal awareness might hurt.** The paper reports consistent improvements but provides no investigation of scenarios where trajectory branching or noise-aware reweighting degrades performance. For instance, does aggressive reweighting toward early timesteps ever cause the model to neglect fine-grained detail that matters for certain prompts? Understanding failure modes is important for assessing the reliability and practical limits of the approach.

## Nice-to-Haves

- **Ablation restricting branching to early timesteps only.** The paper motivates that early timesteps carry the most reward variance (Figure 2), yet branching is applied at all timesteps. An ablation where branching is restricted to the first K steps would simultaneously test the temporal-awareness hypothesis and potentially reduce computational cost if later branching is unnecessary.

- **Quantitative diversity analysis.** Figure 19 provides qualitative diversity visualizations, but a quantitative metric (e.g., LPIPS diversity across samples for the same prompt) would strengthen the claim that RL alignment does not sacrifice diversity.

- **Statistical significance testing.** Results appear to be single runs. While this is standard practice for large-scale generative model training, confidence intervals or multi-seed runs would strengthen claims about marginal improvements (e.g., the 1.0–1.7% PickScore gains).

- **Human evaluation study.** All metrics are proxy-based. A small-scale human preference study comparing TempFlow-GRPO vs. Flow-GRPO outputs would provide the strongest evidence for the paper's claims about human preference alignment.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Theoretical justification depends on specific noise schedule $\sigma_k = a\sqrt{(1-k)/k}$.** The paper explicitly addresses this in Appendix A.16, deriving a generalized reweighting coefficient for arbitrary $\sigma_k$ schedules and validating it experimentally (Figure 20). The concern is already handled.

- **Weakness: "Flow shift" terminology undefined.** This is a minor clarity point about terminology that appears in Figure 5 and Section 4.2; it relates to parameter $a$ in the noise schedule and is inferable from context. Formatting nitpick.

- **Weakness: Baseline score discrepancy between "0.63 to 0.97" and "Flow-GRPO reaches 0.88."** The reviewer misread the text. Table 1 shows SD3.5-Medium's base score is 0.63. TempFlow-GRPO improves it to 0.97. Flow-GRPO achieves 0.88. These numbers are consistent: base (0.63) → Flow-GRPO (0.88) → TempFlow-GRPO (0.97). Factually incorrect criticism.

- **Weakness: Distinction between Flow-GRPO and Flow-GRPO (Prompt) introduced too late.** This is a presentation/clarity nitpick. The distinction is explained in Figure 3's caption and in the experimental section, which is the natural place for it.

- **Weakness: Missing cross-architecture validation.** The paper tests on FLUX.1-dev, SD3.5-M, and Qwen-Image (Appendix A.12). This is reasonable architectural diversity for the scope.

- **Weakness: Reproducibility concerns about undisclosed hyperparameters or missing code.** Per hard rules, these are removed as nitpicks about reproducibility.

- **Weakness: Method limited to flow matching and may not generalize to standard diffusion models.** The paper is explicitly scoped to flow matching GRPO. Criticizing absence of diffusion model experiments is scope creep.

## Novel Insights

The correlation between noise level and reward standard deviation (Figure 5, Left) is not just an empirical observation but reveals a structural property of flow-based generation: the information-theoretic capacity for reward-relevant exploration is inherently coupled to the signal-to-noise ratio at each timestep. This suggests that for any flow-based RL method—not just GRPO—the noise schedule itself encodes a natural curriculum for exploration, and methods that ignore this structure are fighting against the generative dynamics. The policy gradient derivation making this explicit (the inverse-noise scaling of gradient contributions in standard GRPO) is a finding that could inform future algorithm design beyond this specific method.

## Suggestions

- **Reconcile the computational cost accounting.** Add a clear paragraph or table to the main text explaining: (a) the per-iteration sampling cost multiplier from branching, (b) whether forward passes are parallelized across branches (and if so, the memory cost), (c) what "training time per iteration remains identical" means precisely, and (d) the resulting wall-clock comparison. This is the most critical clarification needed.

- **Add at least one held-out evaluation benchmark.** Evaluate on a compositional or preference benchmark not used as a training reward (e.g., DrawBench, T2I-CompBench) to address the circular evaluation concern. Even a small-scale evaluation would significantly strengthen the SOTA claims.

- **Qualify the Credit Localization theorem.** Acknowledge that while the *source* of stochasticity is localized, the *magnitude* of credit assigned depends on downstream flow sensitivity. This would make the theoretical claims more precise without undermining the method's practical value.

- **Include a per-timestep gradient contribution visualization during training.** Show how gradient norms or reward improvements distribute across timesteps before and after reweighting. This would directly validate the core temporal-awareness hypothesis during actual optimization, complementing the initial variance analysis of Figure 2.

---

**Axis Assessments:**

- **Novelty:** High. Trajectory branching for credit assignment without PRMs is a genuinely novel mechanism for flow-based RL. The noise-aware reweighting with its policy gradient justification goes beyond heuristic scheduling.

- **Technical soundness:** Moderate. The core mechanisms are well-motivated and the theoretical derivation is directionally correct, but the Credit Localization claim is overstated, and the computational cost accounting contains internal contradictions that undermine confidence in the efficiency results.

- **Empirical support:** Moderate. Improvements are consistent and substantial across many settings, but the circular evaluation setup (training reward = evaluation metric) and narrow baseline comparisons limit the strength of the SOTA claims. No human evaluation is provided for a paper centered on human preference alignment.

- **Significance:** Moderate-to-high. Addressing temporal uniformity in flow-based GRPO is an important problem, and the proposed solutions are practical and well-motivated. If the efficiency claims hold after clarification, this could meaningfully improve the feasibility of RL-based alignment for large flow models.

- **Clarity:** Moderate. The paper is well-structured with good use of figures, but the density of mathematical notation in Section 4.2/Appendix A.1, the contradictory efficiency statements, and some unclear terminology ("flow shift") reduce accessibility.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 6.0, 6.0]
Average score: 7.5
Binary outcome: Accept
