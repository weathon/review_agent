=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary

TempFlow-GRPO addresses temporal uniformity in existing flow-based GRPO methods by introducing three mechanisms: (1) trajectory branching, which injects SDE stochasticity at individual timesteps along an otherwise deterministic ODE trajectory to attribute terminal rewards to specific intermediate actions; (2) noise-aware policy weighting, which reweights the GRPO loss by each timestep's noise level to align optimization intensity with exploration potential; and (3) a seed group strategy, which groups trajectories by shared initial noise to isolate exploration effects from initialization variance.

## Strengths

- **Principled identification of temporal imbalance in flow-GRPO.** Figure 2 (Left) provides direct empirical evidence that reward variance is heavily concentrated in early timesteps, and Figure 5 (Right) shows that the natural gradient scale terms in standard GRPO are inversely proportional to noise level—causing late refinement steps to dominate optimization despite minimal impact. This diagnosis is specific, well-visualized, and directly motivates the proposed reweighting.

- **Elegant workaround for process reward models.** Trajectory branching avoids the notoriously difficult problem of training intermediate reward models on semantically ambiguous noisy states by instead using outcome-based rewards attributed via the ODE-SDE-ODE construction. This is a practical and conceptually clean contribution that lowers the barrier to temporally-aware credit assignment.

- **Consistent and substantial empirical gains.** Across multiple models (SD3.5-M, FLUX.1-dev, Qwen-Image), benchmarks (GenEval, PickScore, HPDv2), and reward models (PickScore, HPSv2, HPSv3, GenEval), TempFlow-GRPO consistently outperforms Flow-GRPO. The GenEval improvement from 0.63 to 0.97 is particularly striking, and convergence speedups of 2–4.5× in step count are demonstrated throughout.

- **Component-wise ablation with additive gains.** Figure 8 shows that trajectory branching, noise-aware reweighting, and seed grouping each contribute incrementally on both PickScore and GenEval, with reweighting alone providing a ~10% gain on GenEval (0.82→0.92 at 1200 steps).

## Weaknesses

### Major:

- **The "provable guarantees" and "Credit Localization Theorem" are informal and overstated.** Section 4.1.1 states that "total reward variance and all parameter-dependent improvements are entirely attributable to the outcome of noise injection at k," framing this as a theorem. However, no formal proof is provided—the statement is presented as a proposition without rigorous derivation. While the *variance* in final rewards across branches indeed originates from the noise at step k (since the ODE remainder is deterministic), calling this "provable guarantees" at ICLR standard requires either a formal theorem statement with proof or substantially softened language. The current presentation blurs the line between a well-motivated design choice and a rigorous theoretical result.

- **Computational overhead accounting is insufficient for the default configuration.** Appendix A.6 states that for K=10, sampling overhead is ~4.5× that of Flow-GRPO. However, the main experiments use K=6 (4 seeds × 6 branches). The actual per-iteration overhead for this default configuration is never reported. Figure 3 and Figure 12 plot performance vs. GPU hours and show net efficiency gains, but without a transparent breakdown of per-step sampling cost, gradient cost, and parallelization strategy for the 4×6 configuration, the reader cannot independently verify whether the GPU-hour comparisons account for all branching costs. Providing wall-clock time per iteration for the default setup would resolve this ambiguity.

### Minor:

- **Ambiguity in how advantages are assigned across timesteps.** The paper states that trajectory branching localizes credit to the branching point k, yet Equation 7 sums the loss over all timesteps t=0,…,T−1 with the same advantage Â_i_t. The text in Section 4.1.1 says the reward for the k-th step is replaced with R(ODE_{k-1}(SDE(x_k, ε)), c), but it is unclear whether this means (a) the advantage is non-zero only at the branching timestep and zero elsewhere, or (b) the same advantage computed from the branched trajectory is applied to all timesteps. If (b), the "credit localization" claim is at odds with the dense loss. Algorithm 1 (lines 18–21) computes advantages per timestep using branched rewards, which suggests per-timestep advantages, but this should be stated explicitly in the main text.

- **Branching description oscillates between "at a designated timestep k" and "at each step."** Section 4.1.1 describes branching at "a designated branching timestep k," while Section 5.2 states "branching is performed at each step." These describe different things—the former refers to a single branch point for one trajectory, while the latter means the method branches at every timestep of the ODE trajectory, each producing its own set of exploratory samples. This distinction is critical for understanding computational cost and should be clarified unambiguously in the method section.

- **No systematic evaluation of diversity or mode collapse.** RL-based fine-tuning of generative models is known to risk reduced diversity. Appendix A.14 shows qualitative diversity examples and notes a PickScore drop of ~0.234 when optimizing for GenEval (comparable to baseline), but no quantitative diversity metrics (e.g., FID, LPIPS diversity, precision/recall) are provided. Given the increased optimization pressure on early timesteps, a quantitative diversity analysis would strengthen confidence that the method does not sacrifice generative breadth.

- **Lower KL divergence is shown but not analyzed.** Figures 7, 10, and 14 consistently show TempFlow-GRPO maintains lower KL divergence than Flow-GRPO. The paper interprets this as better distribution preservation, but an equally valid interpretation is that the method underfits relative to the reward signal. Since the method achieves higher reward scores *and* lower KL, underfitting is unlikely, but the paper should briefly discuss why rather than leaving the interpretation implicit.

### Trivial:

- The limitations section (Section 6) focuses solely on reward model enhancements and does not acknowledge the method's reliance on having sufficient timesteps for effective branching, which Appendix A.13 shows degrades in few-step settings.

## Nice-to-Haves

- Comparison against non-GRPO diffusion RL baselines (e.g., Diffusion-DPO, SPO) to contextualize gains beyond the GRPO framework.
- A sensitivity analysis on branching factor and seed count beyond the 2×12 / 4×6 / 6×4 configurations already tested.
- Systematic reward hacking evaluation across multiple reward models, as temporal credit assignment could exacerbate or mitigate this differently than uniform GRPO.
- Validation on non-image flow matching tasks (e.g., audio, video) to support the generality of the "generative dynamics" framing.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Claim that the Norm() function in Equation 7 introduces inter-sample dependency.** The paper states in Section 5: "we normalized the weights applied to the policy loss to have a mean of 1 at all timesteps." This is a pre-computed normalization of the noise schedule, not batch-statistics-dependent. The criticism misunderstands the implementation.

- **Concern that Flow-GRPO baseline hyperparameters may not be optimally tuned.** The paper explicitly states "experimental setup is kept consistent with that of Flow-GRPO," meaning the same hyperparameters from the original Flow-GRPO paper are used. This is standard practice for fair comparison; there is no evidence the authors tuned one method differently from the other.

- **Demand for comparisons with DPOK, Diffusion-DPO, etc.** The paper's stated contribution is improving GRPO for flow models. Comparing to methods operating under entirely different optimization frameworks is scope creep. The relevant baselines are Flow-GRPO and DanceGRPO, which the paper does compare against.

- **Concern about missing related works.** Per hard rules, this is removed.

- **Formatting and style nitpicks** about garbled PDF parsing artifacts. Per hard rules, removed.

- **Demand for cross-domain validation (video, audio).** The paper's scope is explicitly text-to-image flow models. Demanding evaluation on other modalities is scope creep.

- **Demand for confidence intervals or multiple runs.** Single-run evaluation is standard practice in large-scale generative model training; requesting confidence intervals is a nice-to-have at most.

## Novel Insights

The most interesting observation across the reviews, confirmed by the paper, is the fundamental mismatch between the natural gradient scale in standard flow-GRPO and the actual exploration capacity of each timestep. Figure 5 (Right) shows that without reweighting, the gradient contribution is dominated by late refinement steps (where noise is low and the scale term ∝ Δ_k(1−k)/k is large), while early structural steps (where exploration variance peaks) contribute minimally. The noise-aware reweighting simplifies this scale term to ∝ Δ_k, yielding approximately uniform gradient contributions when flow shift=1. This suggests that the gains from TempFlow-GRPO may be less about "credit assignment" per se and more about correcting a systematic bias in the vanilla GRPO gradient that causes the optimizer to waste capacity on low-impact refinement steps—a distinction the paper does not fully surface but which could sharpen the narrative and guide future work on RL for generative models more broadly.

## Suggestions

- Replace "provable guarantees" language with "under the deterministic ODE assumption, reward variance localizes to the branching point" and either provide a formal theorem + proof in an appendix or reframe the claim as a well-justified design principle.
- Report per-iteration wall-clock time and memory usage for the default 4×6 configuration, and include a small table breaking down sampling cost vs. gradient cost vs. total cost, so readers can verify the GPU-hour efficiency claims independently.
- Add one sentence in Section 4.1.1 clarifying whether advantages are non-zero only at branching timesteps or dense across all timesteps, and if dense, explain why this does not contradict the credit localization claim.
- Add quantitative diversity metrics (e.g., LPIPS diversity, precision/recall) in a short table or plot to address the mode collapse concern that is standard for RL-based generation methods.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 6.0, 6.0]
Average score: 7.5
Binary outcome: Accept
