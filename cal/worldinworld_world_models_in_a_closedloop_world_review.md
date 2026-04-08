=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
Now I have enough to write a thorough review. Let me compose my assessment.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "World-In-World: World Models in a Closed-Loop World" is creative but slightly redundant ("World" appears three times). More substantively, the abstract makes three bold empirical claims ("surprises"): (1) visual quality doesn't guarantee task success, (2) post-training data scaling beats upgrading pre-trained generators, and (3) inference-time compute helps. These are well-supported by the experiments overall, but the framing as "surprises" is oversold—especially claim (1), which has been anticipated by several prior works in model-based RL and has been explicitly noted for video-based world models (e.g., Kang et al., 2024 "How Far is Video Generation from World Model"). The abstract also claims to present "the first data scaling law for world models in embodied settings," but Figure 6 shows three curves over four data sizes (400, 4K, 40K, 80K)—this is a scaling *trend*, not a scaling *law* (which typically involves fitting a power-law or log-linear functional form with fitted coefficients and generalization claims). This is a notable overclaim.

---

### Introduction & Motivation

The motivation is compelling and clearly articulated: most benchmarks are open-loop and measure visual quality, not task success. The gap is real. The contributions listed are concrete, though the third one—discovering that visual quality doesn't predict task success—reads more as a finding than a contribution. The related-work positioning against VP2 (Tian et al., 2023) is fair; however, the paper could be more thorough in discussing DreamerV3 and other model-based RL systems that have long used learned world models for closed-loop control, even if those are not video-generation models per se. The boundary between "video generation models used as world models" and "latent world models for RL" is worth clarifying because it affects what the "first comprehensive closed-loop benchmark" claim means.

---

### Method / Approach (Sections 2.1–2.4)

**Closed-loop planning framework (Sec. 2.1):** The proposal–simulation–revision formulation is clean and general. Presenting it as "policy-guided beam search" is apt. However, Equations (1)–(4) effectively describe standard model-predictive control (MPC) / CEM-style planning with a learned world model and learned scoring function—this is not a novel algorithmic contribution in itself. The paper should acknowledge this more directly rather than framing it as a core contribution. The framework's main value here is integrative, not algorithmic.

**Unified action API (Sec. 2.2):** The three-mode API (text prompt, camera trajectory, low-level action) is practical and clever. A key concern is the text-prompt modality: translating discrete navigation primitives (Forward, Turn-Left, Turn-Right) into text strings and passing them to a text-conditioned video generator is a very noisy control signal. The paper acknowledges this implicitly (zero-shot text models have limited controllability), but doesn't quantify how much controllability degrades across the three API modes, which would be a useful ablation.

**Task selection (Sec. 2.3):** The four tasks (AR, ImageNav, A-EQA, Manipulation) are diverse and well-motivated. However, there are important asymmetries:
- AR (551 episodes), A-EQA (184 episodes), ImageNav (144 episodes), Manipulation (200 episodes). The evaluation sets are small—particularly ImageNav (144 episodes) and Manipulation (200 episodes across 4 tasks = 50 each). Statistical reliability of the reported differences is not addressed (no confidence intervals, no standard deviations, no significance tests). For ImageNav, a 1-episode difference corresponds to a ~0.7% SR change, yet differences of 1–3% are used to draw conclusions.
- The tasks are simulation-only (Habitat-Sim, CoppeliaSim). Real-world evaluation is absent, which is noted in discussion but not in the limitations section.

**Post-training (Sec. 2.4 / Appendix C):** The fine-tuning setup is clear and reproducible (official repos, documented hyperparameters in Table 8, computational costs in Table 10). Using LoRA for 14B models but full fine-tuning for smaller models is a confound: the 14B models may be under-adapted relative to their capacity, which could affect the claim that "larger models benefit more" (Sec. 3.2). This should be discussed.

A deeper concern: the post-training data for Habitat tasks is collected using the Habitat oracle shortest-path planner and a custom trajectory-sampling procedure (Appendix D). The resulting trajectories represent systematic, efficient coverage of indoor scenes—quite different from the exploratory, potentially suboptimal trajectories an actual agent would generate. The distribution mismatch between post-training trajectories and evaluation-time agent behavior is not analyzed.

---

### Experiments & Results (Section 3)

**Table 1 (AR and ImageNav):** The clearest gain is for the heuristic base policy (39% → 61% SR in AR with post-trained SVD). These are large improvements. But notably, for the VLM base policy, most zero-shot video generators offer only marginal AR gains (50.27% baseline vs. 55–59% with zero-shot WMs), raising a question: is the world model genuinely doing something useful, or is the improvement mostly attributable to the increased number of samples (M=2 candidate plans) in the planning loop? The paper does not ablate the number of candidates M while holding the world model fixed at a "dummy" or random model. This is a critical missing control—if randomly generated rollouts, evaluated by the VLM revision policy, also improve performance, then the benefit is from test-time sampling, not from world-model fidelity.

**Table 3 (Manipulation):** The results are notably weak. With a VLM base policy, zero-shot world models produce no gain or even slight degradation (SVD: 44.0% vs 44.5% baseline). Post-trained models improve slightly (SVD†: 46.5%, Cosmos-P2†: 45.0% vs 44.5%). The 3D Diffusion Policy results are more interesting (24% → 44.7% with SVD†), but the baseline of 24% is suspiciously low—the paper notes Stack Cups is excluded from the 3D-DP evaluation (only 3 of 4 tasks), which makes direct SR comparison with the VLM baseline (which uses all 4 tasks) unclear and potentially misleading. This should be clarified in the table.

**Controllability finding (Sec. 3.2, Fig. 5):** This is the strongest empirical finding. The comparison between SR vs. generation quality (Fig. 5a) and SR vs. controllability measured by 1−LPIPS (Fig. 5b) is compelling. However, using LPIPS between ground-truth and predicted observations as a measure of "controllability" conflates two things: (i) how accurately the model predicts the future conditioned on the action, and (ii) how faithfully the model follows the intended action. These are related but distinct. A model that predicts the correct scene but drifts in viewpoint would score poorly on LPIPS but might still be useful. A dedicated controllability metric (e.g., angular error in viewpoint, translation error in position) would be more rigorous.

**Data scaling (Fig. 6):** The scaling trend is clear and interesting. But the claim "data scaling law" in the abstract requires fitting a functional form. The figure shows four points (400, 4K, 40K, 80K) per model without any fitted curve or extrapolation, without stating whether the curve follows a power law or log-linear form, and without error bars. This is an empirical observation, not a "law." Moreover, 80K is the largest dataset evaluated—it's unclear whether performance has saturated or would continue to improve with more data.

**Inference-time scaling (Fig. 7):** This is convincing and clearly presented. However, the x-axis is "average inference count per episode," which is correlated with M (number of candidates) but also with episode length. Controlling for one while varying the other would strengthen the claim.

**Table 5 (Revision policy ablation):** The LPIPS-based revision policy outperforms the VLM-based revision policy on ImageNav (SVD†: 47.92% vs 43.05% SR). This is an important result suggesting that task-specific, perceptual reward functions are more effective than general VLM scoring for goal-conditioned navigation—but this finding is somewhat underplayed. It also raises a question: for AR and A-EQA, where VLM scoring is used, would a better-tailored reward function substantially change the results?

**Runway Gen4 (proprietary model):** Including Runway Gen4 in the AR table without full ImageNav results (entries are blank) and without reporting Manipulation results creates an incomplete comparison. The paper should either include all results or explain why only partial results are available.

---

### Writing & Clarity

The paper is generally well-written. Section 2.3 is thorough, and the appendices are unusually detailed and transparent—a strength. However, Section 3.2 presents several ablations (Tables 4, 5, 6) inline with text that is difficult to parse because the tables are interleaved with explanation paragraphs. The presentation of Table 4 (duplicated in the appendix at lines 772–798) is confusing.

The A-EQA results (Table 2) are presented with far fewer model comparisons than Tables 1 and 3. Only NWM and two image generators are shown under the VLM base policy—none of the zero-shot video generators (Wan, Hunyuan, LTX-Video, Cosmos) appear. The paper should either include all models or explain the omission. For a benchmark paper, consistency of coverage across tasks is important.

---

### Limitations & Broader Impact

Section 4 discusses four future directions (generalization, long-horizon planning, precise dynamics, stronger policies) and one practical concern (compute). These are genuine and relevant. However, some important limitations are not acknowledged:

1. **Simulation gap**: All evaluation is in simulation. The gap to real-world environments—including sensor noise, action execution errors, and domain shift—is not discussed.

2. **Confounded improvement from sampling**: As noted above, the paper does not isolate the contribution of world-model quality from test-time sampling (using M>1 candidates).

3. **Task coverage**: Manipulation is essentially the only task involving physical contact and precise kinematics, and the results are near-baseline. The benchmark does not cover key embodied AI domains such as outdoor navigation, multi-agent settings, or manipulation with deformable objects.

4. **Evaluation set size**: The small evaluation sets (especially ImageNav at 144 episodes) limit the statistical reliability of conclusions.

5. **Use of proprietary models (Runway Gen4)**: Benchmark papers that include proprietary, closed-source models with no parameter count or implementation details create reproducibility issues and competitive imbalances.

---

### Overall Assessment

World-In-World makes a genuine and timely contribution by introducing a closed-loop evaluation framework for generative world models in embodied tasks. The core insight—that visual quality is a poor proxy for embodied utility—is empirically validated across multiple tasks and models, and the benchmark's integration of diverse heterogeneous world models via a unified action API is practically valuable. The paper is transparent about implementation details and computational costs.

However, several issues weaken the contribution at the ICLR standard. First, the central empirical claims are incomplete without a control condition that isolates world-model quality from test-time sampling (M>1 candidates). Second, evaluation sets are small and no statistical significance testing is reported, making fine-grained performance comparisons unreliable. Third, the "data scaling law" framing in the abstract is unsupported—there is no fitted functional form, and what is shown is a trend over four data sizes. Fourth, the manipulation results are nearly null, and the 3D-DP comparisons are not presented consistently across models. Fifth, the benchmark's coverage is uneven across tasks, especially in Table 2 (A-EQA), where several major models are missing. These gaps do not undermine the paper's core value as a benchmark contribution, but they do reduce confidence in the specific quantitative claims. A major revision addressing the confounded evaluation design and the overclaiming on scaling laws would substantially strengthen the submission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces World-In-World, a closed-loop benchmark that evaluates generative world models based on their practical utility for embodied decision-making rather than open-loop visual quality. It proposes a unified planning strategy (proposal-simulation-revision), a standardized action API, and a lightweight post-training protocol to adapt heterogeneous video generators across four embodied tasks (active recognition, image-goal navigation, embodied QA, and manipulation). Empirical results demonstrate that fine-grained controllability and action-conditioned post-training scale more predictably for task success than raw visual fidelity or base model upgrades, and that allocating additional inference-time compute yields consistent closed-loop performance gains.

### Strengths
1. **Targets a Clear and Timely Evaluation Gap:** The paper convincingly argues that open-loop visual metrics (e.g., aesthetic scores, controllability against camera trajectories) are insufficient proxies for embodied utility. By explicitly testing whether world models help agents succeed in closed-loop interaction, it addresses a pressing need in the generative and embodied AI communities (Sec. 1, Fig. 2).
2. **Unified and Flexible Integration Framework:** The standardized action API successfully bridges diverse world model interfaces (text prompts, camera trajectories, low-level discrete/continuous actions) into a single evaluation pipeline. This enables fair, apples-to-apples comparison across fundamentally different architectures, from proprietary text-to-video models to panoramic navigation simulators (Sec. 2.2, Tables 1-3).
3. **Actionable Empirical Insights & Scaling Analysis:** The paper delivers three well-supported findings: (i) controllability (measured via 1-LPIPS between intended vs. predicted motion) correlates strongly with task success, unlike aesthetic quality (Fig. 5); (ii) post-training with action-observation data follows a clear scaling law and outperforms simply upgrading base generators (Fig. 6); (iii) increasing inference-time compute (planning rollouts) reliably boosts success rates, establishing a clear compute-performance trade-off (Fig. 7). These insights directly inform future model design and deployment strategies.
4. **Rigorous Experimental Design and Ablations:** The evaluation spans four distinct tasks, multiple base policies (VLM, heuristic, 3D diffusion), and over ten world models. The ablations systematically isolate key components, including input context (panorama vs. front view, Tab. 4), revision policy design (VLM vs. LPIPS scoring, Tab. 5), and cross-domain generalization (synthetic-to-real post-training, Tab. 6), demonstrating methodological thoroughness expected at ICLR.

### Weaknesses
1. **Limited Algorithmic Novelty in Planning Component:** The core decision-making loop is explicitly described as a policy-guided beam search analogous to Model Predictive Control (Sec. 2.1). While effective, the paper's contribution lies primarily in the evaluation framework and empirical analysis rather than a novel planning algorithm. Reviewers may expect either a new planning formulation or a more explicit justification of why standard MPC suffices for benchmarking diverse WMs.
2. **Marginal Gains and Underexplored Limitations in Manipulation:** The paper acknowledges that world models struggle with precise contact-rich dynamics (Sec. 3.1), but the empirical gains are modest (~1-2% SR over baselines, Tab. 3) and limited to short-horizon tasks. The analysis does not systematically categorize failure modes (e.g., collision, physics violation, temporal drift) beyond referencing appendix figures, leaving a gap in understanding *why* and *where* video-based WMs break down for manipulation.
3. **Heavy Reliance on Large VLMs for the Revision Policy:** The default revision/scoring policy uses Qwen2.5-VL-72B (Appendix B.5.2), which introduces significant computational overhead and potential accessibility barriers. While the LPIPS alternative is shown to work well for navigation (Tab. 5), the paper does not thoroughly analyze how revision policy capability bottlenecks or distorts the benchmark's assessment of the *world model itself*, especially for complex QA or reasoning-heavy tasks.
4. **Scaling Law Analysis Lacks Compute-Normalization:** Figure 6 shows clear data-scaling trends, but does not account for training FLOPs, parameter efficiency across different model sizes (e.g., 1.5B SVD vs. 14B Wan2.1), or data quality vs. quantity. The claim that "scaling post-training is more effective than upgrading pretrained generators" would be stronger with compute-normalized comparisons or explicit disentanglement of capacity vs. data efficiency.

### Novelty & Significance
The paper demonstrates high significance for the machine learning and embodied AI communities. As world model generation rapidly advances, the lack of a standardized, closed-loop utility benchmark has become a critical bottleneck. World-In-World fills this gap with a well-engineered, open evaluation platform that shifts the community's focus from visual plausibility to embodied decision-making efficacy. The novelty is primarily empirical and systemic rather than theoretical: the unified action API, the comparative analysis across heterogeneous generative families, and the derivation of controllability and inference-time scaling laws constitute a substantive advancement. This aligns well with ICLR's growing emphasis on rigorous empirical evaluation, reproducible benchmarks, and insights that steer model development toward practical robustness.

### Suggestions for Improvement
1. **Quantify Computational Trade-offs Explicitly:** Provide latency, VRAM, and FLOPs estimates for the planning loop across different beam widths and world model sizes. A compute-performance curve would help practitioners determine the cost-benefit of inference-time scaling and make the benchmark more actionable for resource-constrained settings.
2. **Deepen the Manipulation Failure Analysis or Reframe Scope:** Given the modest gains, add a systematic breakdown of rollout failures (e.g., physics inconsistency, contact error, temporal drift) using quantitative metrics or confusion matrices alongside qualitative examples. Alternatively, clearly position the manipulation suite as a "stress test" that exposes current video generation limits rather than a primary success story.
3. **Strengthen Scaling Law Rigor:** When claiming post-training data scaling outperforms base model upgrades, include parameter-normalized or compute-normalized comparisons. Discuss whether saturation points are driven by model capacity limits, action space coverage, or dataset diversity, and clarify the marginal returns beyond 40K-80K samples.
4. **Ensure Full Reproducibility Infrastructure:** ICLR places high importance on reproducibility. Explicitly commit to releasing the unified action API, environment wrappers, proposal/revision policy implementations, and post-training scripts with fixed seeds and dependency specifications. Detail how researchers can replicate the closed-loop setup without proprietary API access, and provide clear instructions for swapping world models and tasks programmatically.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to non-WM model-based planning baselines** — Without this, you cannot claim improvements come from the world model specifically rather than just rollout-based planning; add a dummy/random rollout baseline with identical compute budget.

2. **Add statistical significance testing with confidence intervals** — ICLR requires rigorous evaluation; all tables report point estimates only, making it impossible to determine if claimed improvements (often 1-3%) are meaningful or noise.

3. **Test on real robot hardware, not just simulation** — The core claim is about embodied utility, but all evaluation is in Habitat/CoppeliaSim; without real-world validation, the "closed-loop embodied" claim remains unproven.

4. **Ablate planning horizon (L) and candidate count (M) systematically** — These vary arbitrarily across tasks (L=4 for AR, L=5 for ImageNav, L=14 for A-EQA) without justification; show sensitivity curves to prove results aren't hyperparameter-tuned.

5. **Include compute-matched model-free RL baselines** — Claim (3) about inference-time scaling is undermined without showing whether the same compute budget spent on training a stronger policy would outperform WM-based planning.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify when WM planning hurts vs. helps performance** — Show failure cases where WM predictions mislead the planner; without this, readers cannot assess reliability or know when to trust the method.

2. **Analyze the controllability-success correlation mechanistically** — Figure 5 shows correlation but doesn't explain why; add analysis of which action types (rotation vs. translation) are hardest to control and how this affects specific tasks.

3. **Measure uncertainty calibration of world model predictions** — For embodied decision-making, knowing when predictions are unreliable is critical; current work treats all predictions as equally trustworthy.

4. **Report actual wall-clock inference time and memory footprint** — Claim (3) about inference-time scaling is meaningless without quantifying the computational cost; ICLR reviewers need to assess practical feasibility.

5. **Analyze cross-task generalization of post-trained models** — Post-training is task-specific; show whether a model post-trained on navigation helps manipulation tasks, or if each task requires separate adaptation.

### Visualizations & Case Studies
1. **Side-by-side episode rollouts showing WM prediction vs. ground truth divergence over time** — This would reveal whether failures come from early prediction errors that compound, or sudden catastrophic hallucinations.

2. **Case studies of high-visual-quality models failing specific tasks** — Figure 2 shows the correlation but individual examples would make the claim concrete and actionable for readers.

3. **Visualization of planning trajectories with and without WM for the same episode** — Show exactly which decisions changed due to WM predictions and whether those changes improved outcomes.

4. **Failure mode gallery organized by error type** — Categorize failures (physics violations, object permanence, action ignoring) to guide future improvement directions beyond generic "controllability matters."

### Obvious Next Steps
1. **Include at least one real-robot manipulation experiment** — For an embodied AI paper at ICLR, simulation-only evaluation significantly weakens the contribution; even a small real-world validation would strengthen claims substantially.

2. **Compare against stronger closed-loop baselines from prior work** — The paper mentions VP2 and other benchmarks but doesn't integrate their methods; this is needed to establish genuine novelty over existing closed-loop evaluation.

3. **Test generalization to unseen action spaces** — Post-training uses the same action space as evaluation; show whether models transfer to new action primitives without additional fine-tuning.

4. **Provide a detailed compute budget breakdown** — Show training costs, inference costs per episode, and total wall-clock time; ICLR reviewers need this to assess whether the method is practically deployable.

5. **Release the post-training dataset and code for reproducibility** — The benchmark's value depends on community adoption; without open resources, the "first open platform" claim cannot be verified or built upon.

# Final Consolidated Review
## Summary

World-In-World introduces the first closed-loop benchmark for evaluating generative world models (WMs) through embodied task success rather than open-loop visual metrics. The paper proposes a unified planning framework (proposal-simulation-revision), a standardized action API that bridges heterogeneous WM interfaces (text prompts, camera trajectories, low-level actions), and a lightweight post-training protocol to adapt video generators. Across four embodied tasks (Active Recognition, Image-Goal Navigation, Active EQA, Robotic Manipulation), the paper finds that controllability correlates more strongly with task success than visual quality, that post-training data scaling is more effective than upgrading pretrained generators, and that inference-time compute yields consistent gains.

## Strengths

- **Addresses a timely and important evaluation gap.** The paper convincingly argues that existing benchmarks (VBench, WorldModelBench, WorldScore) measure visual plausibility but not embodied utility. By centering evaluation on closed-loop task success, the benchmark targets the core question: "do world models actually help agents succeed at embodied tasks?" (Sec. 1).

- **Unified integration of heterogeneous world models.** The action API successfully bridges fundamentally different control interfaces—text-conditioned generators (Wan2.1, Hunyuan), trajectory-conditioned models (NWM), and viewpoint-based synthesizers (PathDreamer, SE3DS)—into a single evaluation pipeline, enabling fair comparison across model families (Sec. 2.2, Tables 1–3).

- **Substantive empirical findings with practical implications.** Three core results are well-supported: (i) controllability (1−LPIPS between ground-truth and predicted motion) correlates with task success while aesthetic quality does not (Fig. 5); (ii) post-training data scaling yields predictable gains, with 40K–80K samples achieving near-peak performance (Fig. 6); (iii) inference-time compute (more candidate rollouts) improves success rates across tasks (Fig. 7). These directly inform model design and deployment.

- **Rigorous ablations and transparent methodology.** The paper systematically ablates input context (panorama vs. front view, Table 4), revision policies (VLM vs. LPIPS scoring, Table 5), and cross-domain transfer (Table 6). Computational costs for post-training are reported (Table 10), and implementation details are thorough (Appendices C–D).

## Weaknesses

- **Missing control for test-time sampling vs. world-model quality.** The planning loop uses M>1 candidate action sequences. Without a control condition showing whether randomly generated rollouts (or dummy predictions) evaluated by the same revision policy also improve performance, it remains unclear how much gain comes from world-model fidelity versus simply from increased sampling. This is a critical experimental gap for attributing improvements to the WM itself.

- **Statistical reliability of small evaluation sets.** ImageNav uses only 144 episodes, A-EQA 184, and Manipulation 200 across four tasks (50 per task). No confidence intervals, standard deviations, or significance tests are reported. For ImageNav, a 1-episode swing corresponds to ~0.7% SR change, yet conclusions are drawn from differences of 1–3 percentage points. This limits confidence in fine-grained model comparisons.

- **"Data scaling law" claim is overstated.** Figure 6 shows four data points per model without fitted curves, functional form specification, or extrapolation claims. The abstract's claim of "the first data scaling law for world models in embodied settings" is not supported—this is an empirical scaling trend, not a scaling law. The terminology should be corrected.

- **Inconsistent model coverage across tasks.** Table 2 (A-EQA) omits all zero-shot video generators (Wan, Hunyuan, LTX-Video, Cosmos) shown in Tables 1 and 3, and Table 1 omits ImageNav results for the proprietary Runway Gen4. For a benchmark paper claiming comprehensive evaluation, this uneven coverage makes cross-task comparison incomplete.

- **Controllability metric conflation.** Using LPIPS between ground-truth and predicted observations as a "controllability" measure conflates prediction accuracy with action-following fidelity. A model that generates visually plausible frames but drifts in camera viewpoint would score poorly on LPIPS yet might still be useful. A dedicated controllability metric (e.g., angular/translation error in viewpoint) would be more rigorous.

- **Confounded comparison for 14B models.** Table 10 notes that 14B models use LoRA fine-tuning while smaller models use full fine-tuning. This introduces a confound when comparing scaling across model sizes—the 14B models may be under-adapted relative to their capacity, affecting claims about model size and data efficiency.

## Nice-to-Haves

- **Compute-normalized scaling comparisons.** When claiming post-training data scaling outperforms upgrading base generators, parameter-count or FLOP-normalized comparisons would strengthen the claim and help practitioners assess cost-benefit tradeoffs.

- **Deeper analysis of manipulation failures.** Table 3 shows near-null gains for manipulation (SVD†: 46.5% vs. 44.5% baseline). A systematic categorization of failure modes (physics violations, contact errors, temporal drift) beyond the appendix figures would help readers understand *why* video WMs struggle in this domain.

- **Systematic ablation of planning hyperparameters.** The candidate count M and horizon L vary across tasks without justification (M=2, L=4 for AR; M=3, L=5 for ImageNav; M=3, L=14 for A-EQA). Showing sensitivity to these choices would ensure results aren't artifact of hyperparameter tuning per task.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No algorithmic novelty in planning component"** — The paper explicitly frames the planning framework as policy-guided beam search (Sec. 2.1) and does not claim algorithmic novelty. The contribution is integrative (unified interface for diverse WMs), not algorithmic. Criticizing lack of novel algorithms attacks a straw man.

- **"Missing comparison to DreamerV3 and model-based RL systems"** — The paper's scope is clearly defined as evaluating generative video models used as world models for embodied tasks. Latent world models for RL (DreamerV3) are a different research direction. This criticism demands work outside the paper's stated contribution.

- **"No real-robot evaluation"** — The benchmark focuses on simulation environments, which is standard for this research area. While real-world validation would strengthen the work, its absence does not invalidate the simulation-based findings. This is a nice-to-have for future work, not a core weakness.

- **"Compute-matched model-free RL baselines"** — The paper evaluates whether world models help embodied agents; comparing to alternative approaches (model-free RL) is outside its scope. The relevant baseline is agents *without* world models (VLM-only, heuristic-only), which the paper includes.

- **"Missing related work references"** — As an AI reviewer, I cannot verify whether specific related works exist or are missing from citations. The paper cites 90+ references covering visual generation, world models, and embodied AI.

## Novel Insights

The core insight that controllability predicts embodied success while visual quality does not is empirically validated through multiple ablations. The revision policy experiment (Table 5) reveals that a simple LPIPS-based goal-matching score outperforms VLM-based scoring for ImageNav (SVD†: 47.92% vs. 43.05%), suggesting that task-specific perceptual rewards can be more effective than general-purpose VLM judgment for goal-conditioned tasks. This has practical implications: for navigation tasks, lightweight perceptual metrics may suffice, reserving VLM scoring for tasks requiring semantic reasoning (AR, A-EQA). Additionally, the cross-domain generalization results (Table 6) show that models post-trained on synthetic HSSD scenes transfer to real-world HM3D/MP3D environments with only moderate performance degradation, suggesting the post-training learns action-conditioned visual representations that are somewhat robust to domain shift.

## Suggestions

- **Add a random-rollout control condition.** Run experiments where predictions are replaced by random/noise frames while keeping the same planning loop and compute budget. This isolates whether gains come from world-model quality or from test-time sampling.

- **Report confidence intervals or standard errors.** For evaluation sets with ≤200 episodes, bootstrap confidence intervals should be computed and reported to establish statistical significance of claimed improvements.

- **Correct "scaling law" terminology.** Replace with "scaling trend" or "scaling behavior" throughout, or fit and report a functional form (power-law coefficients) to justify "law."

- **Include full model coverage or explain omissions.** Either add missing zero-shot video generators to Table 2 (A-EQA) or provide a brief explanation for why certain models were excluded from specific tasks.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
