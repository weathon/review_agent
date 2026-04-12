=== CALIBRATION EXAMPLE 42 ===

# Final Consolidated Review
## Summary
This paper introduces **World-In-World**, a benchmark and interface for evaluating visual/generative world models in **closed-loop embodied tasks** rather than by open-loop video quality alone. The core contribution is a unified planning-and-action interface that lets heterogeneous world models participate in four embodied settings (active recognition, ImageNav, active EQA, and robotic manipulation), together with an empirical study showing that post-training on action-observation data and increased test-time planning compute can materially improve task success.

## Strengths
- **The paper makes a meaningful shift from open-loop visual evaluation to closed-loop embodied utility.** This is not just a rhetorical reframing: the benchmark’s primary reported outcomes are task metrics such as AR success, ImageNav SR/SPL, A-EQA answer score/SPL, and manipulation SR, and the paper directly shows dissociation between generation quality and task performance (Figure 5a / Figure 2).
- **The framework genuinely integrates heterogeneous model interfaces rather than benchmarking a single narrow model family.** The unified action API covers text prompts, camera trajectories/viewpoints, and low-level actions (Section 2.2), allowing direct comparison among image-generation models (PathDreamer, SE3DS), video generators (SVD, LTX-Video, Hunyuan, Wan, Cosmos), and task-oriented world models (e.g., NWM) within one closed-loop protocol.
- **The post-training study is one of the strongest empirical aspects of the paper.** Across multiple models and tasks, action-conditioned post-training consistently improves embodied performance, e.g. Wan2.1 improves from 58.26% to 62.61% on AR and from 38.19% to 45.14% on ImageNav; similar gains appear for SVD and for A-EQA. This is a practical and nontrivial finding for the community.
- **The paper surfaces an important and specific empirical lesson: controllability matters more than aesthetic quality for embodied use.** While the exact proxy can be debated, the benchmark usefully demonstrates that visually impressive models are not necessarily the most useful planners, and that action-conditioned adaptation changes this ranking.
- **The benchmark includes several task regimes with qualitatively different demands.** Navigation/perception tasks show clearer gains, while manipulation remains much harder. That asymmetry is itself informative and is honestly discussed by the authors rather than hidden.

## Weaknesses

### Fatal
None.

### Major:
- **Benchmark scores conflate world-model quality with the quality of the action-interface translation, especially for text-conditioned models.**  
  This concern is real and visible in the paper’s setup. Section 2.2 explicitly maps the same agent action sequence into different control modalities: “text prompt,” “camera trajectory/viewpoint,” or “low-level actions.” For text-conditioned models, the interface relies on a predefined template that converts actions into phrases; for other models, the same agent action is represented much more directly. This means that performance differences can reflect not only the predictive capability of the world model, but also the fidelity of the control representation injected into that model.  
  The paper partially acknowledges heterogeneity but does not isolate how much of the gap comes from the model versus the translation layer. This matters because a benchmark claiming to compare embodied utility across architectures should distinguish “model cannot predict useful futures” from “the API failed to express the intended action precisely enough.”

- **Closed-loop performance is substantially mediated by the revision policy, and the paper does not fully disentangle world-model quality from scorer quality.**  
  The paper itself states in Section 4 that “the agent’s overall performance depends on both world-model fidelity and the strength of the proposal and revision policies.” Table 5 makes this concrete: in ImageNav, simply replacing the VLM-based revision policy with LPIPS-based scoring increases SR substantially (e.g., SVD† from 43.05 to 47.92; Wan2.1† from 45.14 to 48.61).  
  This is an important result, but in the current presentation it also weakens the claim that the benchmark cleanly ranks world models. If the revision policy is a strong bottleneck, then measured “WM utility” is really the utility of the **WM + revision policy pair**. The paper should analyze this more directly, because otherwise model rankings may change with the scorer rather than with the simulator quality itself.

- **The main “controllability matters more than visual quality” conclusion is suggestive but not as rigorously supported as claimed, because the controllability metric is only a proxy for action fidelity.**  
  In Section 3.2 / Figure 5(b), controllability is defined as **1 − LPIPS between ground-truth and predicted observations**. That is closer to frame-level predictive alignment than to direct action-faithfulness. A model could score well by producing perceptually similar frames without truly respecting the commanded action semantics, and conversely a model could follow action commands but differ visually in ways LPIPS penalizes.  
  So the qualitative takeaway is plausible, but the metric does not fully support the stronger interpretation that “controllability” per se is the causal driver. This weakens one of the headline findings.

- **The inference-time scaling claim is potentially confounded by extra compute rather than uniquely demonstrating the value of world-model rollouts.**  
  Figure 7 shows better performance with more world-model inferences per episode, which is useful operationally. However, the current experiments do not compare against a compute-matched alternative that spends similar extra budget on a planner without a world model (e.g., more proposal samples, repeated rescoring, or stronger search over the base policy alone).  
  As a result, the paper supports the statement “more WM-assisted planning compute helps,” but does not fully establish the stronger interpretation that simulated rollouts are uniquely responsible for the gains, rather than the more general fact that more test-time search improves decisions.

- **The manipulation evaluation is the least convincing part of the paper, both because the gains are modest and because the action adaptation may itself introduce distortion.**  
  The paper is transparent that manipulation remains hard, and Table 3 indeed shows only small improvements over the VLM baseline in that setting. Moreover, Appendix B.4 states that when the candidate sequence length does not match the world model’s required conditioning length, the action API “linearly interpolates” or uniformly samples actions. For 7-DoF manipulation, this is a fairly strong design choice that may affect realism of the conditioned motion.  
  This does not invalidate the benchmark, but it makes it harder to attribute weak manipulation performance solely to limits of current world models. Some of the degradation may come from the interface mismatch itself.

### Minor
- **The paper’s strongest empirical conclusions are concentrated in a subset of tasks and settings, especially AR and ImageNav.**  
  The benchmark is broad, but the clearest scaling and correlation analyses are primarily shown for AR, with more limited depth on why results transfer similarly—or fail to transfer—to the other tasks.
- **The computational trade-off is acknowledged but not quantified enough for practical deployment questions.**  
  Section 4 discusses efficiency concerns, and Figure 7 shows a performance/compute trend, but the paper does not report end-to-end latency, throughput, or per-step runtime cost of the planning loop. Given that the method explicitly advocates more inference-time computation, this missing systems view matters.
- **Cross-model fairness is imperfect because some models receive richer inputs than others.**  
  Appendix B.6 notes that SE3DS receives ground-truth depth and PathDreamer receives depth plus semantic labels, while others operate on RGB only. This is understandable given model requirements, but it means “fair” should be interpreted as “integrated under a common closed-loop protocol,” not as perfectly matched sensory input conditions.
- **The benchmark mostly measures whether adding a world model improves a given base policy, not whether the world model alone is a strong decision-making substrate.**  
  This is consistent with the paper’s stated design, but it does mean conclusions should be phrased as augmentation utility rather than standalone planning capability.

### Trivial
- **The “data scaling law” language is somewhat stronger than the evidence supports.**  
  Figure 6 clearly shows monotonic improvement with more post-training data, which is useful. But this is closer to an empirical scaling trend over the explored regime than to a deeply characterized scaling law.

## Nice-to-Haves
- Add an ablation isolating the **action API translation effect**, e.g. compare native action-conditioned inputs versus text/trajectory abstractions where possible.
- Report **latency / GPU-time / memory** for closed-loop planning at several inference budgets, especially for the Figure 7 scaling study.
- Include a **compute-matched non-WM baseline** to determine how much of the inference-time scaling gain is specific to simulated rollouts.
- Strengthen analysis of **failure accumulation over planning horizon**, especially comparing navigation vs manipulation.
- For A-EQA, provide more detail on the robustness of the **LLM-as-judge** evaluation, such as prompt stability or agreement checks.
- Expand revision-policy ablations, since Table 5 already shows this component can materially change outcomes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper is not the first closed-loop/control-centric benchmark because VP2 or prior planning papers exist.”**  
  Removed because the paper’s concrete claim is narrower and more specific: “the first comprehensive closed-loop benchmark” / “the first open platform” for this class of visual world-model evaluation. The cited prior work does not, from the paper text itself, obviously invalidate that scoped claim.
- **Complaints about missing related work or ignored baselines such as DreamerV3/OCTO.**  
  Removed per instruction: missing related-work claims cannot be verified externally here.
- **Reproducibility concerns about proprietary models, release status, API access, or unverifiability.**  
  Removed per hard rule.
- **Pure statistical-significance complaints requiring confidence intervals for all benchmark numbers.**  
  Weakened/removed as a core criticism. While variance would help, single-run reporting is common in this style of large benchmark. The more substantive point retained is that the manipulation gains are modest and should be interpreted cautiously.
- **Simulation-to-real criticism as a core weakness.**  
  Weakened and removed from the main review because the paper is explicitly a benchmark in simulated closed-loop environments, and it already includes a cross-domain transfer study (Table 6). Demanding real-world validation would be scope creep.
- **Claim that stronger baselines are required because gains over heuristic/VLM policies are otherwise invalid.**  
  Removed in this form. The paper’s claim is mainly about evaluating WM utility when plugged into proposal/revision loops, not about beating the entire embodied-AI literature.

## Novel Insights
The most interesting synthesis across the evidence is that the paper is strongest not as a pure “leaderboard benchmark,” but as a **measurement framework for interface-limited world models**. Its own experiments suggest that embodied performance is shaped by three coupled bottlenecks: (1) how expressively actions can be encoded for a given generator, (2) how faithfully the model translates that control into future observations, and (3) how well the revision policy can extract decision value from those predictions. This means the benchmark is revealing an important systems-level truth: today’s “world model utility” is not a single property of the generator alone, but of the whole control–prediction–scoring stack. The paper would be even stronger if it explicitly embraced this as a central framing rather than presenting rankings as primarily model-intrinsic.

## Suggestions
- Add a targeted ablation where the **same model** is driven by different control interfaces (or different prompt realizations) to measure how much the unified API affects rankings.
- Expand Table 5-style analyses across more tasks to show how sensitive conclusions are to the **revision policy**.
- Replace or supplement the current controllability proxy with a more direct **action-faithfulness metric**.
- Include a **compute-matched baseline without world-model rollouts** for the inference-time scaling claim.
- For manipulation, evaluate whether the current interpolation/sampling scheme for action sequences is degrading results; if so, adopt a more kinematically faithful control conversion.
- Make the paper’s claims more explicit about what is being benchmarked: **closed-loop utility of WM-integrated systems**, rather than a pure intrinsic ranking of simulator quality.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
