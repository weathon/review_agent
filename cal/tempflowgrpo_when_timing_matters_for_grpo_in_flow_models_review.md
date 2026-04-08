=== CALIBRATION EXAMPLE 42 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract

The subtitle "When Timing Matters for GRPO in Flow Models" is somewhat casual and vague for an ICLR submission; it reads more like a blog headline than a precise technical descriptor. The abstract makes three strong claims: (1) temporal uniformity is the "key impediment" to GRPO in flow models, (2) the method provides "process rewards … without requiring specialized intermediate reward models," and (3) it delivers "state-of-the-art performance." All three claims are partially supported by the experiments, but (1) is an empirical assertion stated as a principled diagnosis without showing that temporal uniformity—rather than, e.g., reward model quality or group size—is the bottleneck, (2) is accurate but perhaps overstated since the approach still depends entirely on terminal reward quality, and (3) is contested given that comparisons to non-GRPO methods (DPO-based, diffusion RL) are superficial and the human-evaluation gap is unaddressed.

---

### Introduction & Motivation

**Strongest part of the paper.** Figure 2 (left) is a convincing empirical motivation: applying SDE only at one timestep and measuring reward standard deviation reveals a sharp peak at early steps. This cleanly establishes that different timesteps have different exploration potential. The identification of two specific failure modes—sparse terminal rewards and uniform optimization pressure—is crisp.

However, one subtlety is elided: the experiment in Figure 2 applies SDE at step k and measures variance in the TERMINAL reward. This is not the same as showing that credit should be assigned differently to each step in the context of GRPO training—it is possible that the effect on training dynamics is mediated by other factors (e.g., gradient magnitude, trajectory correlation). The paper would be stronger if it connected Figure 2's observation directly to training behavior (e.g., showing that in Flow-GRPO, early steps actually receive smaller effective gradient updates).

The claim that existing work "treats the multi-step generation process as a black box" somewhat overstates the contrast with DanceGRPO and SPO, which do differentiate timesteps to some degree.

---

### Method

**Section 4.1.1 — Trajectory Branching**

The core idea is elegant: run the ODE trajectory deterministically to step k, inject one SDE step at k, and continue deterministically to completion. The terminal reward then localizes the credit to step k. This is a valid and practical mechanism for attributing credit without a process reward model.

However, the paper labels an immediate consequence of the design as a **"Theorem (Credit Localization)"**—which will damage credibility with technically rigorous reviewers. The statement is: since stochasticity is only at step k, reward variance is attributable to step k. This is true by construction, not by proof. It is not a mathematical theorem—it is a definitional property. Calling it a theorem risks appearing either naïve or misleading.

More substantively: the "credit" assigned to step k is the outcome of the full remaining ODE trajectory starting from x_{k-1}, which itself depends on the entire prior ODE trajectory (x_T → … → x_k). The method measures the *marginal contribution* of the stochastic choice at step k given a fixed prefix and fixed suffix policy, not the true Q-value at that state. For short trajectories this is a reasonable proxy, but the relationship to true policy gradient credit assignment is only formalized under the policy gradient justification in Section 4.2—and that analysis has its own limitations (see below).

**A critical missing experiment:** The paper does branching at *every* timestep, but the motivation (Figure 2 left) argues specifically that early steps matter most. There is no ablation showing the effect of restricting branching to only early timesteps vs. only late timesteps vs. all timesteps. This would directly validate or complicate the core motivation.

**Section 4.1.2 — Noise-Aware Policy Weighting**

This is the most technically grounded contribution. The observation in Figure 5 (left) that reward std is strongly correlated with noise level is compelling and well-motivated. The reweighting in Eq. 7 is simple and easy to implement.

The theoretical justification in Section 4.2 derives the policy gradient and shows that without reweighting, the scale term is proportional to √(Δk(1−k)/k)—which for flow matching with k ∈ (0,1) diverges as k → 0 (clean image end). With noise-level reweighting, the scale term simplifies to Δk, yielding uniform gradient weighting when step sizes are equal. The claim "when flow shift equals 1, our method achieves perfect equilibrium" is correct but represents a special case. For other flow shift values, the reweighting does not achieve perfect equilibrium, and the paper does not examine how performance degrades as flow shift deviates from 1. This limits the generality of the theoretical claim.

The derivation in Appendix A.1 contains a Taylor expansion approximation of the reward (Eq. 26) that requires the perturbation magnitude σk√Δk to be small. In early high-noise stages—exactly where the paper claims the method has most impact—this approximation may be crude. The derivation should clarify when this is valid or how large the approximation error is.

**Section 4.3 — Seed Group Strategy**

The seed group strategy (grouping trajectories sharing the same initial seed) is a sensible variance reduction technique for the GRPO advantage computation. The motivation is sound: without seed grouping, reward differences across group members could reflect initial noise variation rather than policy quality.

One question that deserves more careful treatment: in the GRPO advantage (Eq. 3), the normalization uses mean and std over the group. With seed grouping, the group is defined by (prompt, initial seed) pairs. If the group size K=6 branches per seed is small, the advantage estimate may have high variance. The paper does not report the variance of advantage estimates across configurations.

---

### Experiments & Results

**Section 5.1 — Main Results**

The improvements on GenEval (0.63 → 0.97 overall) and PickScore are substantial and would be exciting if the experimental setup is sound.

**A significant issue with Table 1:** The paper announces a multi-model comparison table but the actual rows for autoregressive models, flow matching models, and "GRPO-based methods" are absent from the extracted text (except for a few diffusion model baselines from 2022). While this may partly be a PDF extraction artifact, it makes it impossible to fully evaluate the comparative claims. The key question is: what are the GenEval scores for Flow-GRPO and DanceGRPO specifically, and on which model (SD3.5-M vs. FLUX.1-dev)?

**The "improved baseline" issue:** The paper introduces "Flow-GRPO (Prompt)" as an "improved baseline with group-wise standard deviation stabilization." This is the paper's own improvement to the baseline, not a published variant. Defining an improved version of the comparison method and then demonstrating superiority against it is methodologically questionable—it may create the appearance of a more rigorous comparison while making it harder to assess the true margin over published Flow-GRPO.

**Section 5.1 — FLUX.1-dev / HPSv3**

Training uses 10 steps but evaluation uses 50 steps (Appendix A.2). This discrepancy—training at 10 steps and evaluating at 50—should be flagged prominently. The behavior of the trajectory branching mechanism may differ substantially when deployed at 5x more denoising steps than it was trained with.

**Section 5.2 — Ablations**

The ablation in Figure 8 shows the incremental value of: trajectory branching, noise-aware reweighting, and seed group. This structure is clean and informative. However, the ablation for noise-aware reweighting is done ON TOP OF trajectory branching, not independently from the Flow-GRPO baseline with only reweighting. A clean 2×2 ablation (branching × reweighting) from the same Flow-GRPO (Prompt) baseline would be more informative and is standard practice.

**Missing statistical significance:** All performance curves are shown as single training runs. For a stochastic training procedure with random initialization, no confidence intervals or standard deviations are reported. Given that the differences between methods can be small (e.g., ~1% PickScore improvement), even small random seed effects could explain the differences. This is a significant experimental weakness.

**Human preference evaluation:** All comparisons are via automated metrics (PickScore, HPSv2, HPSv3, GenEval). No human evaluation is reported. While PickScore and HPSv2 are validated proxies, for a paper claiming "superior photorealism and enhanced fine-grained detail" (Figure 1 caption), a small-scale human study would strengthen the claims considerably.

---

### Computational Cost

Section A.6 (Appendix) acknowledges that the sampling overhead for K=6 branches is approximately 4.5x that of Flow-GRPO—this is a significant cost. The main paper Figure 3 claims "superior computational efficiency" while comparing wall-clock GPU hours, but the axes in those curves are compressed (e.g., Figure 3 left PickScore y-axis spans 0.65–0.95, visually amplifying small differences). The paper should present the raw throughput difference more explicitly in the main text, not bury it in the appendix.

The claim of 3.75x convergence speedup (requiring only 80 steps vs. 300 for Flow-GRPO) needs to be reconciled against the 4.5x sampling overhead: the net wall-clock speedup is therefore modest at best (3.75/4.5 < 1), meaning the method may not actually be faster in absolute GPU time per performance point.

---

### Theoretical Framing

Section 4.2 presents a policy gradient analysis that is conceptually useful but incomplete. The key result—that noise-aware reweighting cancels the scale imbalance—is derived under the assumption that E_ε[ε Â_k] is timestep-invariant (Eq. 28). This requires the reward gradient gk to satisfy a specific relationship. While the derivation shows ||E_ε[ε Â_k]|| is invariant, the analysis assumes a first-order Taylor expansion of R around the mean, which may not hold well in early high-noise stages. The paper should discuss the regime where this approximation breaks down.

The phrase "invariant among the timesteps" (following Eq. 28) is stated informally; a stronger statement would characterize the bias introduced by the approximation as a function of noise level.

---

### Limitations & Broader Impact

The limitations section (Section 6) mentions only one limitation: the focus on "algorithmic innovations rather than reward model enhancements." This is inadequate. Missing from the discussion:

1. **Computational overhead:** ~4.5x sampling cost is a real barrier to deployment.
2. **Reward hacking:** Optimizing PickScore or HPSv3 directly likely introduces biases or artifacts not captured by these metrics. No analysis of reward hacking or out-of-distribution generalization of the trained model is provided.
3. **Dependency on flow shift:** The theoretical optimality claim holds only for flow shift = 1 (a special case).
4. **Training vs. inference step mismatch:** Training at 10 steps, evaluating at 50 steps—this gap is not analyzed.
5. **Generalization beyond T2I:** The paper is framed as a general GRPO improvement for flow models but only tested on text-to-image.

---

### Overall Assessment

TempFlow-GRPO presents a coherent set of modifications to Flow-GRPO that address a genuine and well-motivated problem: the temporal non-uniformity of exploration potential in flow matching models. The trajectory branching mechanism is practically sensible and easy to implement; the noise-aware reweighting has a plausible theoretical basis; and the seed group strategy is a reasonable variance reduction technique. The empirical improvements—particularly the ~3.75x convergence speedup and the large gain on GenEval—are impressive if reproducible. However, the paper has several weaknesses that would likely place it on the borderline for ICLR acceptance: (1) the "Credit Localization Theorem" is not a theorem and should not be presented as such; (2) the ~4.5x computational overhead is understated in the main paper and partially undercuts the efficiency claims; (3) there are no statistical significance measures and only single-run curves; (4) the comparison baseline ("Flow-GRPO (Prompt)") is the authors' own improvement, complicating head-to-head assessment; (5) the train/eval step count mismatch (10 vs. 50) is unacknowledged in the main text; and (6) the ablation design is not fully orthogonal. The core contribution—particularly noise-aware reweighting—is likely valuable to the community, but the presentation needs to be tightened, the theoretical claims should be calibrated appropriately, and the experimental methodology needs strengthening before this work is ready for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the inefficiency of uniform timestep optimization existing Group Relative Policy Optimization (GRPO) methods applied to flow matching models. The authors propose TempFlow-GRPO, which introduces a trajectory branching mechanism for precise step-level credit assignment, a noise-aware loss reweighting scheme that aligns optimization intensity with each timestep's intrinsic exploration potential, and a seed-grouping strategy to control initialization variance. Extensive experiments across multiple text-to-image benchmarks demonstrate that the method achieves superior human preference alignment and compositional fidelity with substantially improved sample and wall-clock convergence efficiency compared to baseline GRPO variants.

### Strengths
1. **Well-Motivated Problem Identification & Effective Credit Assignment:** The observation that reward variance is heavily concentrated in early, high-noise timesteps (Figure 2, Left) effectively motivates rejecting uniform updates. The trajectory branching mechanism (switching ODE→SDE→ODE at specific steps) is an elegant solution that circumvents the need for training separate, often noisy, intermediate reward models while enabling rigorous credit localization (Section 4.1.1, Theorem: Credit Localization).
2. **Strong Empirical Rigor & Transparency:** The experimental evaluation is comprehensive, covering diverse base models (SD3.5-M, FLUX.1-dev, Qwen-Image), reward functions, and tasks. Systematic ablations (Figure 8) successfully isolate the contributions of branching, noise reweighting, and seed grouping. Crucially, the authors report wall-clock training time (GPU hours) in Figures 3 and 12 rather than just optimization steps, which is a strong practice for RL efficiency claims and aligns with ICLR's emphasis on fair compute comparisons.
3. **Theoretical Grounding for Optimization Dynamics:** Section 4.2 and Appendix A.1 provide a policy gradient derivation that mathematically explains why standard GRPO's natural gradient scale becomes imbalanced across timesteps. Showing that the unweighted scale term heavily under-prioritizes early structural exploration adds theoretical depth beyond a purely empirical heuristic.
4. **High Reproducibility:** The inclusion of explicit algorithmic pseudocode (Algorithm 1), detailed hyperparameter configurations in the appendix, and clear experimental protocols matching prior work ensures that the method can be readily implemented and verified by the community.

### Weaknesses
1. **Compute/Memory Overhead & Practical Scalability:** While overall training time is reduced due to faster convergence, the branching mechanism inherently multiplies forward-pass memory and compute per iteration by a factor of $K$ (number of branches). The paper acknowledges the per-sample cost (Section A.6) but does not discuss VRAM bottlenecks, gradient accumulation strategies, or how this scales to significantly larger models or higher resolutions without distributed branching.
2. **Incomplete Diversity & Reward Hacking Analysis:** Section A.14 briefly mentions reward hacking and shows a minor PickScore drop when optimizing for GenEval. However, quantitative diversity metrics (e.g., pairwise LPIPS, intra-prompt variance, or FID-distribution metrics) are absent. Given that RLHF for generation frequently induces mode collapse, stronger empirical evidence that TempFlow-GRPO maintains output diversity under aggressive alignment is needed for ICLR standards.
3. **Ambiguous Baseline Definition:** The primary comparison is often made against "Flow-GRPO (Prompt)," described as an "improved baseline with group-wise standard deviation stabilization" (Figure 3 caption). The exact formulation and any hyperparameter modifications applied to create this variant are not fully detailed in the main text. This ambiguity makes it difficult to definitively attribute all gains to the proposed temporal mechanisms versus baseline tuning differences.
4. **Loose Coupling Between Theory and Implementation:** The theoretical analysis derives a natural gradient coefficient proportional to $-\Delta_k(1-k)/k$, yet the implemented reweighting factor (Equation 7) uses a normalized noise level justified primarily by empirical correlation (Figure 5, Left). The paper claims this "simplifies the scale term," but the precise mathematical mapping from the derived correction to the actual normalized heuristic used in practice is not explicitly demonstrated.

### Novelty & Significance
- **Novelty:** Moderately High. Integrating temporal awareness into GRPO for flow models addresses a genuine gap. While concepts like process rewards and noise-aware scheduling exist in isolation, synthesizing ODE-SDE trajectory branching for credit localization with gradient-scale-aligned reweighting is a principled and novel contribution to the growing Diffusion/Flow RL literature.
- **Clarity:** High. The narrative flows logically from motivation to method, theory, and experiments. Figures effectively communicate the temporal variance and mechanism. Some theoretical derivations are dense, but the core intuitions are accessible.
- **Reproducibility:** Very High. Detailed algorithms, hyperparameters, ablation setups, and clear benchmark protocols meet ICLR's rigorous standards. Code release is not mentioned, but the textual description is sufficient for independent implementation.
- **Significance:** High. As preference alignment moves toward continuous-time generative models, moving beyond "black-box," uniform-step RL is a critical algorithmic step. This work provides a practical, theoretically-motivated framework that improves both training dynamics and final alignment quality, making it highly relevant to ICLR's focus on learning dynamics, sample efficiency, and foundational generative model optimization.

### Suggestions for Improvement
1. **Clarify Baseline Protocols:** Explicitly define the "Flow-GRPO (Prompt)" variant in Section 5, detailing any architectural or hyperparameter changes made to the original Flow-GRPO. Ensure comparisons isolate TempFlow-GRPO's contributions rather than baseline configuration effects.
2. **Quantitative Diversity Evaluation:** Add standard generation diversity metrics (e.g., average pairwise LPIPS or KID across 50+ generations per prompt) to complement preference scores. This will rigorously address concerns about potential mode collapse or reward hacking under the proposed optimization.
3. **Detail Compute/Memory Trade-offs:** Provide a breakdown of peak VRAM usage per iteration as a function of branch factor $K$ and trajectory length. Discuss practical mitigation strategies (e.g., checkpointing, distributed sampling, or gradient accumulation) to broaden the method's applicability to larger-scale training.
4. **Strengthen Theory-Implementation Link:** In Section 4.1.2, explicitly show how the chosen normalized noise weighting approximates or bounds the theoretically derived scale term correction. If it is an empirical heuristic, clarify why normalization was preferred over direct use of the analytical scale factor.
5. **Test on Compressed Dynamics (Few-Step Models):** Since distillation and consistency models effectively compress the generation trajectory, briefly evaluating how TempFlow-GRPO behaves with highly compressed step counts (e.g., $\leq 4$ steps) would demonstrate the method's robustness and clarify the limits of temporal credit assignment when the noise schedule is truncated.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Total Compute Cost (FLOPs):** Section A.6 admits a 4.5x sampling overhead per step due to branching, yet claims efficiency based on convergence speed. Provide a explicit table of total FLOPs or wall-clock time to reach target performance to verify if the convergence speedup genuinely outweighs the per-step branching cost.
2. **Human Preference Evaluation:** The core claim is "Human Preference Alignment," but all results rely on automated proxies (PickScore, HPS). Conduct a blind human evaluation study to verify if higher automated scores actually correlate with human preference for this specific method.
3. **Non-GRPO Baselines:** Comparisons are limited to Flow-GRPO variants. Include standard Diffusion-DPO or DDPO baselines to establish whether the GRPO+Temporal framework is superior to alternative alignment paradigms, not just an improvement on a specific baseline.
4. **Reward Model Sensitivity:** Validate the method using non-CLIP-based rewards (e.g., regression-based aesthetic scorers) to ensure the credit assignment signal isn't overfitted to the specific geometry of CLIP-based reward landscapes.

### Deeper Analysis Needed (top 3-5 only)
1. **Empirical Credit Localization:** The "Theorem (Credit Localization)" assumes reward variance stems solely from the branching step. Empirically measure the correlation between noise injected at step $k$ and final reward variance to verify non-linear ODE propagation doesn't obscure this signal.
2. **Taylor Expansion Validity:** The theoretical justification (Appendix A.1) relies on a first-order Taylor expansion of the reward model (Eq 26). Analyze the error bound of this approximation given the high non-linearity of deep vision-language reward models used in experiments.
3. **Gradient Norm Verification:** Plot the actual measured gradient norms per timestep during training to confirm the noise-aware weighting successfully equilibrates contributions as claimed in Figure 5, rather than relying on theoretical derivation alone.
4. **Noise Schedule Robustness:** The weighting scheme depends on $\sigma_t$. Analyze performance stability across different flow matching noise schedulers (e.g., linear vs. cosine) to determine if the method is robust or scheduler-dependent.

### Visualizations & Case Studies
1. **Latent Trajectory Divergence:** Visualize the latent space trajectories at branching points to show how much the paths diverge after the SDE injection, confirming the "exploration" claim physically.
2. **Failure Case Gallery:** Display examples where TempFlow-GRPO underperforms (e.g., prompts requiring precise late-stage refinement like text) to expose the limits of prioritizing early-stage exploration.
3. **Reward Landscape Heatmap:** Visualize the reward landscape around the branching points to show if the method indeed navigates higher-variance regions compared to uniform GRPO.

### Obvious Next Steps
1. **Formal Theoretical Proof:** Replace the heuristic "Theorem" and Taylor approximation with a rigorous bound on credit assignment error for non-linear ODE solvers to meet ICLR theoretical standards.
2. **Reward Hacking Mitigation:** Deepen the analysis in Appendix A.14 to explain *why* reward hacking occurs and propose a mitigation strategy specific to temporal branching, rather than acknowledging it as inherent.
3. **Sparse Branching Strategy:** Analyze the trade-off of branching only at high-variance steps vs. all steps to optimize the compute/performance ratio, as branching at every step appears computationally wasteful.

# Final Consolidated Review
## Summary

This paper proposes TempFlow-GRPO, a temporally-aware extension of Group Relative Policy Optimization for flow-based text-to-image models. The authors identify that existing GRPO methods apply uniform optimization across generation timesteps despite varying exploration potential, and address this through: (1) trajectory branching for step-level credit assignment without requiring intermediate reward models, (2) noise-aware policy reweighting that prioritizes high-variance early timesteps, and (3) a seed-grouping strategy for variance reduction. The method achieves state-of-the-art performance on GenEval (0.63 → 0.97) and PickScore benchmarks with faster convergence than baseline GRPO variants.

## Strengths

- **Well-motivated problem identification:** Figure 2 (left) compellingly demonstrates that reward standard deviation varies dramatically across timesteps—peaking at early steps (0-2) and approaching zero at late steps (6-8). This provides clear empirical motivation for rejecting uniform timestep treatment and establishes a genuine gap in existing GRPO methods.

- **Elegant trajectory branching mechanism:** The ODE→SDE→ODE design enables precise credit assignment to intermediate actions without training specialized process reward models. By injecting stochasticity only at designated branching points and completing trajectories deterministically elsewhere, the method correctly localizes reward variance to specific timesteps. This is a practical and principled alternative to the "semantic ambiguity" problem of evaluating noisy intermediate states.

- **Theoretical grounding for noise-aware reweighting:** Section 4.2 and Appendix A.1 derive that the natural gradient coefficient in standard GRPO is proportional to −Δk(1−k)/k, which heavily underweights early structural exploration. The proposed noise-level reweighting simplifies this to Δk, achieving balanced gradient contributions. Figure 5 (right) visualizes this correction effectively, showing how standard GRPO's scale terms are dominated by late refinement steps.

- **Comprehensive empirical validation:** Experiments span multiple base models (SD3.5-M, FLUX.1-dev, Qwen-Image), reward functions (PickScore, HPSv2, HPSv3, GenEval), and tasks (human preference alignment, compositional generation, visual text rendering). The consistent improvements across settings suggest the approach generalizes well.

- **Clear ablation structure:** Figure 8 systematically isolates contributions of trajectory branching, noise reweighting, and seed grouping. The results show each component provides incremental gains, with noise reweighting providing the largest improvement (10% GenEval gain over Flow-GRPO at 1200 steps).

## Weaknesses

- **Mislabeling of "Theorem (Credit Localization)":** Section 4.1.1 presents a "Theorem" stating that credit localizes to the branching point because all stochasticity is concentrated there. This is not a mathematical theorem—it is a definitional consequence of the design. Calling it a theorem risks misleading readers and undermines the paper's technical credibility. The observation is valid and useful, but should be presented as a design property or proposition, not a theorem.

- **Author-defined improved baseline:** The primary comparison is against "Flow-GRPO (Prompt)," which the paper explicitly describes as "an improved baseline with group-wise standard deviation stabilization." This is the authors' own modification to Flow-GRPO, not the published baseline. While comparing to a strengthened baseline can be methodologically sound, the exact modifications should be clearly detailed in the main text, and comparisons to the original Flow-GRPO should also be presented to isolate the contribution of temporal mechanisms from baseline tuning.

- **Absence of statistical significance measures:** All performance curves report single training runs with no confidence intervals or standard deviations. Given that reported improvements can be relatively small (~1-2% on PickScore) and RL training is inherently stochastic, this makes it difficult to assess whether observed gains are reproducible or attributable to random variation.

- **Train/eval step count discrepancy:** Appendix A.2 states that training uses 10 sampling steps while evaluation uses 50 steps. The trajectory branching mechanism may behave differently when deployed at 5× the denoising steps it was trained with. This discrepancy should be discussed in the main text with analysis of its implications.

- **Incomplete ablation factorial design:** The ablations in Figure 8 add components sequentially (Flow-GRPO → +Branching → +Reweighting → +Seed Group). A fully orthogonal 2×2×2 factorial design would better isolate each component's contribution and potential interactions.

## Nice-to-Haves

- **Human preference evaluation:** All comparisons rely on automated metrics (PickScore, HPS, GenEval). While these are established proxies, a small-scale human preference study would strengthen claims about improved "photorealism and fine-grained detail."

- **Diversity analysis beyond visual examples:** Appendix A.14 shows sample images but lacks quantitative diversity metrics (e.g., pairwise LPIPS, intra-prompt variance). For an RL-based alignment method, demonstrating that output diversity is maintained is important for practical deployment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weakness about missing DanceGRPO/non-GRPO comparisons:** The paper explicitly compares to DanceGRPO in Appendix A.4 and Figure 11, showing TempFlow-GRPO achieves 1.3% improvement and 2× convergence speedup. Requests for Diffusion-DPO or DDPO comparisons are scope creep—the paper specifically improves GRPO methods for flow models.

- **Weakness about Taylor expansion approximation error:** While the first-order Taylor expansion in Appendix A.1 may be less accurate in high-noise early stages, the method's strong empirical performance suggests this approximation is sufficiently valid in practice. The derivation provides intuition; it need not be mathematically tight.

- **Criticisms about Table 1 being incomplete:** This appears to be a PDF extraction artifact. The actual paper includes complete GenEval comparisons across model categories as evidenced by the results discussed throughout.

- **Request for confidence intervals on large-scale benchmarks:** Single-run evaluation is standard practice in generative model papers at ICLR. While additional statistical measures would strengthen the paper, their absence is not a critical flaw.

- **Demand for rigorous theoretical proof:** This is an empirical methods paper with theoretical motivation. The current derivation suffices to explain why noise-aware reweighting helps. Replacing it with formal proofs would be out of scope.

## Novel Insights

The paper makes an underappreciated observation about the *imbalanced gradient contributions* in standard GRPO: while credit assignment treats timesteps uniformly, the natural gradient coefficient actually varies by orders of magnitude across the generation trajectory. Figure 5 (right) reveals that late refinement steps receive ~5× larger gradient contributions than early structural steps under standard GRPO—precisely backwards from the exploration potential shown in Figure 2 (left). This mismatch between where gradients *should* be largest (early, high-variance steps) and where they *actually* end up largest (late, low-variance steps) provides a mechanistic explanation for GRPO's sample inefficiency that goes beyond simply observing "early steps matter more." The correction through noise-level reweighting doesn't just weight timesteps differently—it fundamentally rebalances the effective learning signal to match the intrinsic exploration capacity of each generation stage.

## Suggestions

- **Rename "Theorem (Credit Localization)" to "Property (Credit Localization)" or "Proposition"** to accurately reflect its status as a design consequence rather than a formal result.

- **Report standard deviations across multiple random seeds** (at minimum n=3) for main results. Even small variance estimates would substantially strengthen claims about convergence speedup and final performance differences.

- **Clarify the "Flow-GRPO (Prompt)" baseline** in the main text: specify exactly what modifications were made (group-wise standard deviation stabilization) and provide a comparison to the original Flow-GRPO to isolate TempFlow-GRPO's contribution.

- **Add a discussion of train/eval step mismatch** in the main paper: analyze whether and why trajectory branching transfers from 10-step training to 50-step evaluation, and whether this generalization is an artifact or an intended feature.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 6.0, 6.0]
Average score: 7.5
Binary outcome: Accept
