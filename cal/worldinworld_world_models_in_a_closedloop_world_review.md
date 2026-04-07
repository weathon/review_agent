=== CALIBRATION EXAMPLE 76 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "World-In-World: World Models in a Closed-Loop World" is cute but slightly redundant. The abstract is largely accurate: it correctly foregrounds the gap between open-loop visual quality evaluation and closed-loop embodied utility, and the three "surprises" are genuinely interesting findings that are supported by the experiments. However, the abstract claims this is "the first open platform that benchmarks WMs in a closed-loop world," which is not strictly true — the VP2 benchmark (Tian et al., 2023) already evaluated video prediction in model-based planning. The distinction (VP2 is narrower in scope and uses older models) is acknowledged in the related work, but the "first" framing in the abstract is overreaching.

The abstract also describes "the first data scaling law for world models in embodied settings." As discussed below under Section 3.2, this claim is overstated — what is shown is a 4-point scaling trend, not a fitted functional law.

---

### Introduction & Motivation

The motivation is clear and well-articulated: the community has been measuring world models primarily through visual quality metrics (VBench, WorldModelBench, WorldScore) without asking whether these models actually help agents succeed at embodied tasks. This is a genuine and important gap. Figure 2, showing the poor correlation between visual quality and task success, is the most rhetorically effective contribution in the paper, even if it is hard to read due to parsing artifacts.

The three contributions are well-defined. However, Contribution 3 — discovering that visual quality ≠ task success and that training-time/inference-time scaling helps — could be argued to be a *finding* rather than a *contribution*, since it does not introduce a novel method. This distinction matters at ICLR, where methodological or theoretical novelty is valued alongside empirical findings.

---

### Method: Unified Closed-Loop Planning Strategy (Section 2.1)

The proposal-simulation-revision framework (Equations 1–4) is clearly presented. It is a well-known paradigm — model-predictive control + learned proposal/revision policies — but the formalization is clean and general enough to accommodate heterogeneous world models. The extension beyond classical MPC (where decisions can be answers or recognition labels, not just action sequences) is a notable and useful generalization.

**Concern 1 — Beam width M is very small:** The experiments use M=2 (AR), M=3 (ImageNav, A-EQA), and M=5 (Manipulation). With such small beam widths, the "planning" aspect is limited. It is essentially a 1-step look-ahead with a small candidate set. The paper does ablate inference count in Figure 7, showing that more inferences help, but the computational cost of larger M is never analyzed. Readers need to know whether further scaling M yields diminishing returns quickly.

**Concern 2 — The horizon L is short:** L=4 (AR), L=5 (ImageNav), L=14 (A-EQA). For long-horizon tasks (A-EQA allows 250 low-level actions), using the terminal predicted frame of a 14-step horizon for scoring is a coarse signal. This design choice is acknowledged in the Discussion as a limitation but deserves more explicit treatment in the method section.

---

### Method: Unified Action API (Section 2.2)

The action API bridges three control paradigms (text prompt, camera trajectory, low-level actions). This is the most practically important contribution of the paper, as it enables fair comparison across heterogeneous world models.

**Concern 3 — Critical confound between control type and model capability:** The comparison between zero-shot models (which receive text prompts like "move forward") and post-trained models (which receive explicit low-level action sequences) conflates two factors: (1) the quality of the pretrained model and (2) the precision of the control interface. When post-trained Wan2.1† outperforms zero-shot Wan2.1, is this because of domain adaptation, because of action-conditioned training, or because of the richer control signal? The paper attributes it to "aligning the model to the target domain and action space" (Section 3.1), but the controllability ablation in Figure 5(b) and the section on panoramic vs. front-view inputs (Table 4) suggest the control signal precision is dominant. This confound is only partially addressed.

**Concern 4 — Text prompt translation is under-specified:** For models that use text control, the API "converts each primitive action into a phrase" using "a predefined template." The actual templates are in Appendix F but their quality is not analyzed. A crude text template could fundamentally limit what zero-shot text-conditioned models can achieve — independently of the model's intrinsic capabilities. This makes the zero-shot baselines potentially unfair to text-conditioned models.

---

### Method: Post-Training Recipe (Section 2.4)

The post-training procedure is straightforward: fine-tune pretrained video generators on domain-specific action-observation data from Habitat-Sim (panoramic trajectories) and CoppeliaSim (RLBench demonstrations). The key design choice — training scenes disjoint from evaluation scenes — is appropriate and the dataset statistics (858 scenes, 763K panorama frames) are impressive for a self-collected dataset.

**Concern 5 — LoRA vs. full fine-tuning inconsistency:** Table 10 notes that 14B variants use LoRA fine-tuning while smaller models use full weights. This is not discussed as a potential confound. LoRA may limit how much Wan2.1† (14B) can adapt compared to SVD† (1.5B, full fine-tuning). Yet Table 1 shows Wan2.1† outperforms SVD† substantially. The conclusion that "larger models benefit more from action-conditioned post-training" (Section 3.2) could be partly an artifact of the LoRA-vs-full distinction rather than model capacity.

---

### Experiments: Benchmark Tasks (Section 2.3 / Appendix B)

The four tasks are well-chosen and cover complementary aspects of embodied intelligence. Some specific concerns:

**Concern 6 — Small evaluation sets:** 144 ImageNav episodes and 184 A-EQA questions are small. With no variance reported across trials (no random seeds, no confidence intervals), it is impossible to assess statistical significance. Consider: in ImageNav, the best post-trained model achieves 48.61% SR vs. the base VLM at 35.42% — a 13% absolute gain that seems meaningful. But in Table 5, moving from LPIPS-based to VLM-based revision on the *same model* (SVD†) changes SR from 47.92% to 43.05% — a 5% swing from the revision policy alone. With 144 episodes, a 5% swing corresponds to ~7 episodes. These numbers could easily flip with different random seeds.

**Concern 7 — Runway Gen4 is evaluated only on AR:** Runway Gen4 achieves the highest AR SR (64.79%) and is the dominant outlier in Figure 2, anchoring the claim that proprietary models perform best. Yet it is absent from ImageNav, A-EQA, and Manipulation tables. Given that Gen4 is proprietary, this is understandable, but it creates an asymmetric evaluation. The abstract refers to a comprehensive benchmark but the flagship proprietary model is only tested on one of four tasks.

**Concern 8 — Manipulation results are essentially null:** Table 3 shows that zero-shot video generators consistently *fail* to improve over the base VLM policy in manipulation (SVD: 44.0% vs. base 44.5%, Hunyuan: 44.5% vs. base 44.5%). The post-trained models show only marginal gains (best: SVD† at 46.5% vs. 44.5%). The paper frames this as "less pronounced gains" and attributes it to the difficulty of modeling contact-rich physics. But a 2% absolute gain on 200 episodes (4 more successes) is arguably noise. If world models cannot meaningfully help manipulation — by far the most industrially important embodied task — the benchmark's generality claim is weakened. The section on manipulation deserves a frank acknowledgment that current video-based WMs are *not* yet useful for this task class.

**Concern 9 — A-EQA metric depends on a proprietary LLM judge (GPT-4o):** The Answering Score is graded by GPT-4o on a 1–5 scale. The paper does not report inter-annotator agreement, GPT-4o version, or prompt sensitivity. This is not reproducible in the strictest sense. Minor changes to GPT-4o's outputs (version updates, sampling temperature) could shift scores. A more robust metric or a fixed evaluation model would be preferable.

---

### Experiments: Ablations and Findings (Section 3.2)

**Concern 10 — "Scaling law" terminology is inappropriate:** Figure 6 shows AR success rate vs. number of post-training examples at 4 data points (400, 4K, 40K, 80K examples). The paper calls this "the first data scaling law for world models in embodied settings" (abstract). A scaling *law* conventionally means a power-law (or similar) functional relationship fit to data, with analysis of the exponent and extrapolation range (cf. Kaplan et al., Chinchilla). What is shown here is a monotonically increasing trend across 4 points — there is no curve fit, no analysis of the exponent, and no evidence that the relationship is power-law rather than log-linear or sublinear. This terminology is a significant overclaim. Calling it a "data scaling trend" would be accurate; "scaling law" is not.

**Concern 11 — Inference-time scaling (Figure 7) is confounded:** Figure 7 plots SR vs. "average number of world-model inferences per episode." The number of inferences is *determined by* the number of decision steps the agent takes, which itself depends on whether the agent succeeds quickly (early termination) or struggles. Agents that fail often take more decision steps and thus use more inferences. If early-stopping episodes (successes) use fewer inferences, the average inference count for a higher-SR run would actually be *lower*, not higher. The paper does not discuss this direction of causality. To cleanly study inference-time scaling, the authors should fix the beam width M and vary it directly across experimental conditions, reporting aggregate SR with fixed episode budgets.

**Concern 12 — Panoramic vs. front-view ablation is inconclusive:** Table 4 shows mixed results: for AR, panoramic input consistently helps (60.98% vs. 57.89% for SVD†). For ImageNav, the picture is inconsistent: Wan2.1† favors front-view (48.61% vs. 45.14%) while SVD† favors panorama (43.05% vs. 38.19%). The authors explain this as "panorama-to-perspective conversion introduces resolution loss." This is plausible but speculative and not tested directly. A cleaner ablation would fix resolution and test field of view.

**Concern 13 — Controllability metric (1 − LPIPS)**: The paper uses 1 − LPIPS between ground-truth and predicted observations as a measure of controllability (Section 3.2, Figure 5b). This is actually a measure of *perceptual similarity*, not controllability per se. Two models could have the same LPIPS score but differ in whether their predictions follow the intended action direction. A metric like ego-motion consistency (does the camera actually move forward when the action is "Forward"?) would be a more direct measure of controllability. The correlation in Figure 5(b) is suggestive but the controllability proxy is imprecise.

---

### Writing & Clarity

The paper is generally well-written. The main body is dense but logically organized. Tables 4 and 5 appear *twice* in the paper (once in the main text and once after the appendix table of contents), which is a formatting issue. The A-EQA table (Table 2) only shows results for a subset of world models compared to Table 1, without explanation in the main text — readers must dig into Appendix B.3 to understand why some models are missing.

One clarity issue: the heuristic policy is only described in Appendix B.5.1, not the main body. Yet Table 1 reports results for "Heuristic (w/o WM)" without explanation in the main text. The heuristic policy achieves only 2.08% SR on ImageNav — far below the VLM policy at 35.42% — suggesting these are essentially different experimental conditions, not a comparable baseline.

---

### Limitations & Broader Impact

Section 4 identifies five future directions (generalization, long-horizon planning, precise dynamics, stronger policies, computational cost). These are honest and appropriate. However, the paper does not discuss:

- **Reproducibility of the benchmark itself**: Are the episodes fixed and publicly released? Can other researchers evaluate new models on the same episodes? A benchmark's value depends on a stable, publicly accessible evaluation protocol.
- **The confound between base policy strength and WM gains**: All conclusions about world model utility depend on the specific base policy (Qwen2.5-VL-72B). A weaker base policy might show larger WM gains; a stronger base policy might show none. The paper acknowledges "the base policy sets the performance floor" but does not systematically study this.
- **Selection bias in task and model choice**: The four tasks were chosen by the authors, and the models evaluated are a selective sample. The benchmark's generalization to other tasks (e.g., object rearrangement, social navigation) and other model families (e.g., autoregressive world models like IRIS) is not addressed.

---

### Overall Assessment

World-In-World addresses a genuine and timely gap: existing world model benchmarks reward visual fidelity while embodied agents ultimately need accurate, controllable prediction for decision-making. The closed-loop evaluation framework, unified action API, and post-training recipe are solid engineering contributions, and the central finding — that controllability matters far more than visual quality — is both clear and important. The benchmark itself, with four diverse tasks and 11+ evaluated models, is the most comprehensive closed-loop WM evaluation to date.

That said, several issues limit the paper's impact. The two headline claims beyond the central finding — a "data scaling law" and "inference-time scaling" — are overstated relative to the evidence (4-point trend curves, a confounded experimental design). The manipulation results are essentially null, which is a significant limitation for a benchmark claiming to cover embodied AI broadly. The evaluation sets are small (144–551 episodes) and no statistical testing is reported, so many of the performance differences in the tables may not be reliable. The confound between control interface type and model quality in the zero-shot vs. post-trained comparison is only partially resolved. Finally, the paper's claim of being an "open platform" and "benchmark" is not fully substantiated without a clear description of public infrastructure (leaderboard, fixed episode sets, evaluation server). The paper is above the ICLR acceptance bar in terms of motivation and breadth, but the experimental claims need to be more carefully calibrated before publication.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **World-In-World**, a comprehensive benchmark and framework for evaluating generative World Models (WMs) within closed-loop embodied tasks, moving beyond isolated visual quality metrics. It proposes a unified online planning strategy, a standardized action API to bridge heterogeneous models, and a post-training protocol for action-conditioned adaptation. Empirical results on four embodied tasks reveal that closed-loop task success depends more on controllability and inference-time scaling than on raw visual fidelity, establishing a new standard for measuring the utility of world models in embodied AI.

### Strengths
1.  **Critical Evaluation Paradigm Shift:** The paper correctly identifies and addresses a significant gap in the literature: the evaluation of WMs on visual quality (e.g., FID, aesthetics) versus their practical utility in decision-making. By establishing closed-loop task success as the primary metric, it forces the community to confront whether "smart" videos actually enable "smart" agents. This aligns well with ICLR's focus on meaningful AI advancement.
2.  **Robust System Design and Reproducibility:** The technical contributions extend beyond benchmarking. The **Unified Action API** (Section 2.2) and **Closed-Loop Planning Strategy** (Section 2.1) provide a concrete, reusable interface for integrating diverse models (video generators, image generators) into a single framework. The detailed post-training recipe (Appendix C) and data construction algorithm (Appendix D, Algorithm 1) ensure high reproducibility, with resources and project pages provided.
3.  **Actionable Empirical Insights:** The paper delivers concrete, quantitative findings rather than just qualitative observations. Specifically, the data scaling laws (Figure 6) showing that post-training on domain-specific action-observation data improves utility more than larger pretraining, and the inference-time scaling (Figure 7) showing performance gains with more rollout computations, provide clear directions for future model development. The finding that visual quality correlates poorly with task success (Figure 5) is a vital insight for the field.

### Weaknesses
1.  **Attribution of Performance Gains:** The evaluation methodology relies heavily on the quality of the base policy (e.g., VLM-based proposal/revision policies) to drive performance. While the paper claims WMs enhance the base policy, it is difficult to disentangle the contribution of the world model's predictive accuracy from the revision policy's scoring capability (e.g., using LPIPS vs. VLM scoring in Table 5). Stronger ablation on independent world model fidelity (e.g., prediction errors without policy integration) would strengthen the claim.
2.  **Limited Manipulation Performance:** The Robotic Manipulation results (Table 3) show only marginal improvements over the baseline (e.g., Wan2.1 † improves SR from 44.0% to 44.5% in one setting). While the authors attribute this to the difficulty of modeling contact dynamics, the benchmark could benefit from more sophisticated metrics for manipulation tasks (beyond Success Rate, which can be noisy for high-horizon tasks) or a discussion on why current visual WMs fundamentally fail here in a way that isn't solvable by the proposed framework.
3.  **Computational Overhead:** The closed-loop planning strategy requires multiple rollouts per decision step (up to $M=5$ in manipulation). While the paper acknowledges compute scaling, it does not report wall-clock latency or inference cost per decision step against real-time constraints. For ICLR, which values practical feasibility, understanding the trade-off between the planning overhead and the closed-loop gains (e.g., SR vs. Time-per-episode) is crucial for adoption.

### Novelty & Significance
*   **Novelty:** Moderate to High. While closed-loop evaluation of world models exists in niche areas (e.g., VP2), this work generalizes the concept to recent large-scale generative video models (Wan2.1, LTX-Video) and provides a unified architectural interface (API + Planning) that was previously fragmented. The specific post-training and data sampling methodology adds methodological novelty.
*   **Significance:** High. The paper shifts the metric of success for world models from "pretty videos" to "useful agents." This is essential for the field of Embodied AI, which is stalling on purely generative metrics. The findings regarding scalability and controllability directly influence how researchers should allocate resources toward WM development.
*   **Clarity:** The paper is well-structured with clear definitions of the planning loop and API. Despite the OCR artifacts in the provided text, the logical flow of the argument from problem identification to solution to empirical validation is coherent.
*   **Reproducibility:** High. The provision of algorithms for dataset construction, detailed hyperparameters for post-training, and a public project page suggests that other groups can replicate the benchmark.

### Suggestions for Improvement
1.  **Quantify Compute Efficiency:** Include a section or table detailing the inference time, GPU memory usage, and energy cost per episode for different configurations (e.g., Zero-shot vs. Post-trained, varying $M$). This contextualizes the "Inference-Time Scaling" finding and helps readers assess the trade-offs.
2.  **Disentangle Model vs. Policy Contributions:** Add an ablation study where the revision policy is fixed (e.g., using a simple heuristic score) across different WMs. This would more rigorously prove that gains come from the World Model's internal predictive capability rather than the VLM's ability to score generated videos.
3.  **Address the "Manipulation Stagnation" Deeper:** Expand the discussion on why WMs fail at manipulation compared to navigation. Is it a data modality issue (lack of physical priors) or an architectural one (autoregressive video models not capturing physics)? Proposing specific architectural modifications or data augmentation strategies for this specific failure mode would add technical depth.
4.  **Cross-Task Generalization:** While Table 6 tests cross-domain training, discuss how a WM trained on Navigation transfers to Manipulation (or vice versa) without fine-tuning. This would better illuminate the "generalist world model" hypothesis versus task-specific models.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add statistical significance testing with variance bars** — All tables report single-point metrics without confidence intervals or multiple seeds; without this, claims about post-training improvements (e.g., 58.26% → 62.61%) could be noise rather than meaningful gains.

2. **Include model-free baseline policies without any world model augmentation** — The paper claims WMs enhance performance, but doesn't compare against stronger non-WM baselines (e.g., fine-tuned VLA policies); improvements may stem from the planning framework, not the world models themselves.

3. **Evaluate on at least one real-robot task** — All experiments are in simulation (Habitat-Sim, RLBench); claims about "embodied utility" are undermined without any real-world validation where dynamics and perception gaps matter more.

4. **Add comparison to VP2 and other control-centric benchmarks** — The paper claims to be "first" at closed-loop WM evaluation, but VP2 (Tian et al., 2023) exists; without direct comparison, the novelty claim is weakened.

5. **Test with more diverse task categories beyond navigation/perception** — Four tasks are heavily skewed toward navigation; missing manipulation-heavy or multi-agent tasks limits generalizability claims about the benchmark.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze when world models hurt rather than help performance** — No failure mode analysis; if WMs sometimes degrade decisions due to hallucination, this fundamentally affects whether they should be deployed in safety-critical embodied settings.

2. **Report computational cost per episode for inference-time scaling** — Figure 7 shows more inference improves SR, but without FLOPs/time cost, the practical value of this scaling law is unclear for real-time applications.

3. **Decouple planning framework improvements from world model improvements** — The unified planning strategy itself may drive gains; without ablating the planner separately from the WM, you cannot attribute improvements to world model quality.

4. **Analyze prediction accuracy vs. task success correlation quantitatively** — Figure 5 shows controllability correlates with SR, but no quantitative metric (e.g., regression R²) establishes how much prediction error is tolerable before task performance collapses.

5. **Examine post-training data quality, not just quantity** — Figure 6 shows scaling with data size, but without analyzing trajectory diversity or coverage, it's unclear if more data or better data drives improvements.

### Visualizations & Case Studies
1. **Show side-by-side prediction rollouts vs. ground truth for high-SR and low-SR models** — This would reveal whether high-performing models actually predict more accurate futures or just produce more consistent (but wrong) predictions.

2. **Include failure case visualizations where visual quality is high but task success is low** — This directly supports the claim that "visual quality doesn't guarantee task success" with concrete evidence, not just correlation plots.

3. **Visualize what the revision policy selects across candidates** — Show which trajectories are chosen and why; this reveals whether the revision policy is making meaningful distinctions or selecting arbitrarily.

### Obvious Next Steps
1. **Add uncertainty estimation to world model predictions** — Without calibrated uncertainty, agents cannot know when to trust vs. ignore model rollouts, which is critical for safe closed-loop deployment.

2. **Include long-horizon task evaluation beyond 10-20 steps** — The paper acknowledges long-horizon planning is challenging but doesn't evaluate it; claims about planning utility are incomplete without testing extended horizons.

3. **Release the post-training dataset and code for reproducibility** — ICLR expects reproducibility; without public release of the 40K trajectory dataset and training scripts, the post-training claims cannot be verified.

# Final Consolidated Review
## Summary

World-In-World introduces a closed-loop benchmark for evaluating generative world models (WMs) through embodied task success rather than visual quality metrics. The paper proposes a unified planning framework (proposal-simulation-revision), a standardized action API to integrate heterogeneous WMs, and a post-training protocol for adapting video generators to embodied domains. Empirical evaluation across four tasks (Active Recognition, Image-Goal Navigation, A-EQA, Robotic Manipulation) reveals that visual quality correlates poorly with task success, and that controllability—achieved through action-conditioned post-training and inference-time scaling—matters more for downstream utility.

## Strengths

- **Identifies and addresses a genuine gap in world model evaluation:** The paper correctly observes that existing WM benchmarks (VBench, WorldModelBench, WorldScore) optimize for visual fidelity without asking whether generated predictions actually help agents make better decisions. Figure 2 demonstrates this disconnect convincingly—models with high aesthetic scores (e.g., Wan2.2 5B) achieve lower task success than post-trained counterparts with lower visual quality but better action alignment.

- **Unified action API enables systematic comparison:** Section 2.2 provides a concrete interface that translates agent actions into control signals for diverse WM architectures (text prompts, camera trajectories, low-level actions). This standardization is a practical contribution that allows fair comparison across models that were previously incomparable due to incompatible conditioning interfaces.

- **Important empirical finding on controllability vs. visual quality:** Figure 5(b) shows a clearer positive correlation between controllability (measured as 1-LPIPS alignment between predicted and ground-truth observations) and task success than Figure 5(a) shows for visual quality. This finding—that models must follow actions reliably rather than just look realistic—has direct implications for how researchers should allocate WM development resources.

- **Post-training recipe and dataset construction are reproducible:** Algorithm 1 (waypoint sampling for panoramic trajectories) and Tables 8-10 (training configurations, model sizes, GPU-hours) provide sufficient detail for replication. The training data uses scenes disjoint from evaluation scenes, appropriately testing generalization rather than memorization.

## Weaknesses

- **Overstated "first" and "scaling law" claims:** The abstract claims "the first open platform that benchmarks WMs in a closed-loop world," but VP2 (Tian et al., 2023) already evaluated video prediction models for model-based control—acknowledged in the paper's related work, contradicting the "first" framing. Similarly, Figure 6 shows 4 data points (400, 4K, 40K, 80K) labeled as "the first data scaling law," but a scaling *law* conventionally requires fitting a functional relationship (e.g., power law) and analyzing exponents. What is shown is a monotonic trend—valuable, but not a scaling law.

- **Small evaluation sets without statistical significance testing:** ImageNav uses 144 episodes, A-EQA uses 184 questions. Tables report single-point metrics without confidence intervals or multiple random seeds. A 5% SR swing (e.g., Table 5: SVD† with LPIPS revision achieves 47.92% vs. VLM revision at 43.05%) corresponds to ~7 episodes—differences that could flip with different random seeds. Without variance reporting, the reliability of comparisons is unclear.

- **Confound between control interface and model capability:** Zero-shot models receive text prompts (e.g., "move forward") while post-trained models receive explicit action sequences. When Wan2.1† outperforms zero-shot Wan2.1, is this due to domain adaptation, action-conditioned training, or the richer control signal? Figure 5(b) suggests control precision is dominant, but the confound is only partially resolved.

- **Manipulation results show marginal gains:** Table 3 shows zero-shot WMs provide essentially no improvement over the base VLM policy (44.0-44.5% SR range). Post-trained models yield ~2% absolute gains (best: SVD† at 46.5% vs. base 44.5%). On 200 episodes, this is ~4 additional successes. The paper frames this as "less pronounced gains," but the results suggest current video-based WMs are not yet useful for contact-rich manipulation—a significant limitation for a benchmark claiming breadth across embodied tasks.

- **Inference-time scaling analysis may be confounded:** Figure 7 plots SR vs. average inferences per episode, but episode length depends on success (successful episodes terminate early). If high-SR runs use *fewer* inferences due to early termination, the positive correlation could be artifact. The paper does not discuss this direction of causality or report results with fixed episode budgets.

- **Controllability metric measures perceptual similarity, not action alignment:** The paper uses 1-LPIPS between ground-truth and predicted observations as a controllability proxy (Section 3.2). This measures visual similarity, not whether the predicted motion follows the commanded action direction. A model could achieve high LPIPS similarity while violating action semantics (e.g., moving forward when commanded to turn left).

- **A-EQA evaluation depends on proprietary LLM judge without reproducibility guarantees:** Answering scores are assigned by GPT-4o on a 1-5 scale. The paper does not report GPT-4o version, temperature, or inter-annotator agreement. Minor API changes could shift scores unpredictably.

- **Asymmetric evaluation of proprietary models:** Runway Gen4 achieves the highest AR accuracy (64.79%) and anchors Figure 2's visual quality claim, but is evaluated only on AR—not ImageNav, A-EQA, or Manipulation. This limits the benchmark's comprehensiveness.

## Nice-to-Haves

- **Report wall-clock latency and FLOPs per episode:** The paper shows inference-time scaling improves SR (Figure 7), but practical deployment requires understanding compute costs. Adding a table with latency vs. SR trade-offs would help readers assess real-world feasibility.

- **Test with alternative revision policies beyond VLM-based and LPIPS:** The revision policy (how candidate rollouts are scored) is central to performance. Ablating with heuristic scorers or fixed policies would better isolate the WM's contribution from the scoring mechanism.

- **Include real-robot validation:** All experiments use simulation (Habitat-Sim, RLBench). Real-world evaluation would strengthen claims about "embodied utility," where dynamics gaps and perception noise matter more.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Formatting/style nitpicks":** The harsh critic noted Tables 4 and 5 appear twice and A-EQA has fewer models than Table 1. These are minor presentation issues, not substantive problems.

- **"Heuristic policy only described in appendix":** While true, this is organizational, not a methodological flaw. The base policy comparison is still valid.

- **"Beam width M is small":** The critic notes M=2-5 is limited. The paper acknowledges this and shows scaling helps (Figure 7). The concern about diminishing returns is reasonable but not a flaw—the ablation exists.

- **"LoRA vs. full fine-tuning inconsistency":** While the training approach differs between model sizes, this reflects practical reality (full fine-tuning 14B models is expensive). The gains shown are still meaningful.

- **"VP2 comparison missing":** The paper does cite and discuss VP2 in Related Work. A direct numerical comparison would strengthen novelty claims but is not required.

## Novel Insights

The central insight—that world models must follow actions reliably rather than produce visually stunning outputs—emerges clearly from the empirical results. The correlation analysis in Figure 5 demonstrates this quantitatively: models that respond predictably to low-level controls achieve higher task success than models with higher aesthetic scores but poorer controllability. This has practical implications: the field should invest in action-conditioned training (domain adaptation data is more impactful than larger pretrained models, per Figure 6) rather than chasing visual fidelity benchmarks.

A secondary insight from Figure 7 is that inference-time compute (more candidate rollouts) improves decision quality, suggesting world models could benefit from test-time scaling analogous to language models. However, the practical value depends on latency constraints, which the paper does not analyze.

## Suggestions

- **Replace "scaling law" with "scaling trend" throughout the paper.** This accurately describes Figure 6's 4-point monotonic relationship without overclaiming.

- **Add confidence intervals to all table metrics.** Given small evaluation sets, reporting standard errors or bootstrap intervals would establish statistical significance.

- **Disentangle control interface from model quality with an additional ablation.** Evaluate zero-shot models with a structured text API that encodes action sequences as prompts (e.g., "move forward 0.2m, then turn left 22.5°") to isolate the effect of action representation from post-training.

- **Clarify the manipulation limitation explicitly.** The Discussion should state that current video-based WMs provide negligible utility for manipulation tasks, and identify specific failure modes (contact dynamics, physics violations) that future work must address.

- **Use a fixed, open-source evaluation model for A-EQA scoring.** Replacing GPT-4o with a fixed checkpoint (or providing the exact prompt and version) would improve reproducibility.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
