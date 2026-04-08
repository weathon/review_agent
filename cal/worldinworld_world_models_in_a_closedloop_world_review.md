=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

The paper introduces World-In-World, a closed-loop benchmark that evaluates generative world models through embodied task success rather than open-loop visual quality metrics. It provides a unified online planning strategy (proposal-simulation-revision loop) and a standardized action API enabling heterogeneous world models—image-based, video-based, text-conditioned, action-conditioned—to be integrated and compared across four embodied tasks: Active Recognition, Image-Goal Navigation, Active Embodied Question Answering, and Robotic Manipulation. The study reveals that visual quality poorly predicts task success (controllability matters more), that post-training data scaling is more effective than upgrading pretrained generators, and that inference-time compute scaling via more rollouts improves closed-loop performance.

## Strengths

- **Paradigm shift in evaluation philosophy:** The paper provides compelling evidence that the dominant evaluation paradigm for world models—open-loop visual quality—is misaligned with embodied utility. Figure 2 and Figure 5 demonstrate this disconnect quantitatively, and the controllability-vs-visual-quality analysis (Figure 5b vs. 5a) showing stronger correlation with the former is a concrete, novel finding that should reshape how the community thinks about world model quality.
- **Unified framework enabling heterogeneous comparison:** The unified action API (Section 2.2) and standardized planning loop (Section 2.1) allow models with fundamentally different conditioning interfaces (text prompts, camera trajectories, low-level actions) to be compared under a single evaluation protocol. This is a non-trivial engineering and conceptual contribution—prior work (e.g., VP2) covered narrower task diversity and older architectures, and no prior benchmark enables this breadth of cross-model comparison in closed-loop settings.
- **Actionable empirical findings with practical implications:** The data scaling law (Figure 6) showing that Wan2.2† (A14B) with substantially more web-video pretraining reaches nearly the same performance as Wan2.1† after post-training—demonstrating that scaling action-conditioned post-training is more effective than upgrading the pretrained generator—is a concrete, counterintuitive result with clear practical implications for resource allocation in world model development. Similarly, the inference-time scaling result (Figure 7) provides a clear recipe for improving closed-loop performance.

## Weaknesses

### Major:

- **No statistical significance testing or confidence intervals:** Across all four tasks, results are reported as single numbers without error bars, standard deviations, or significance tests. This is most concerning for tasks with small episode counts: ImageNav has only 144 episodes, where a 2% SR difference represents ~3 episodes. For A-EQA with 184 episodes, the top two post-trained models (LTX-Video†: 48.6 vs. Wan2.2† A14B: 48.4) differ by 0.2 answer-score points on a [0,100] scale. Without any measure of variance, it is impossible to determine whether reported improvements are meaningful or within noise. This is a significant methodological gap for a benchmark paper whose core contribution is empirical evaluation.

- **Planning framework lacks mechanisms to handle unreliable world model predictions:** Section 4 acknowledges that models "may revert to training priors or ignore action controls, yielding plausible but physically or semantically inconsistent rollouts," and Figures 13–14 visualize such failures. Yet the planning pipeline (Equations 1–4) treats all world model predictions as equally trustworthy—the revision policy scores and selects based on predicted rollouts regardless of their fidelity. If the world model hallucinates a favorable outcome, the agent will commit to a bad plan. No uncertainty quantification, prediction quality filtering, or confidence-weighted scoring is incorporated. This is not just a limitation but a structural gap in the proposed framework, since the paper's own evidence shows hallucination is common and directly misleading for planning.

- **Counterintuitive revision policy results are under-analyzed:** Table 5 shows that a simple LPIPS-based revision policy (47.92% SR) substantially outperforms a VLM-based revision policy (43.05% SR) for SVD† on ImageNav—a ~5% absolute gap. This is surprising because a VLM presumably understands task semantics better than pixel-level perceptual distance. The paper presents this as a secondary finding but does not investigate *why* a weaker scoring signal produces stronger decisions. Possible explanations include VLM reward noise, misalignment between VLM judgments and navigation success, or LPIPS being a more reliable proxy for geometric progress. Understanding this failure mode of VLM-based revision would significantly strengthen the paper's contribution, as the revision policy is a core component of the framework.

### Minor:

- **Compute-performance tradeoff for inference-time scaling is unquantified:** Figure 7 shows a clear positive correlation between inference count and SR, but no wall-clock time, FLOPs, or cost analysis accompanies this result. Going from 3 to 11 inferences per episode (for SVD†, SR: 53.36%→60.98%) involves generating 3.7× more video rollouts per decision. Without quantifying the latency or compute cost, the practical implications of this "scaling law" are unclear—especially for real-time robotic applications where planning latency directly impacts performance. Section 4 acknowledges this concern but provides no data.

- **Manipulation results show minimal or negative gains from zero-shot WMs, with insufficient analysis:** Table 3 shows that zero-shot video generators sometimes *decrease* manipulation performance (e.g., SVD zero-shot: 44.0% vs. VLM baseline 44.5%). Post-trained models provide only modest gains (SVD†: 46.5%). The paper briefly notes "gains are less pronounced" and attributes this to "contact-rich interactions," but does not break down which manipulation subtasks fail most, what specific prediction errors occur, or whether the failure is due to visual hallucination, dynamics inaccuracy, or action-control misalignment. Given that manipulation is a major embodied capability, this gap deserves deeper treatment—especially since it represents the primary limitation of current world models for real robotic use.

- **Cross-domain generalization claim is weakly supported by the evidence:** Table 6 shows that post-training on synthetic HSSD scenes and testing on real HM3D/MP3D achieves 58.98% AR SR (SVD†) vs. 60.98% for in-domain HM3D post-training—only a 2% gap. The paper frames this as evidence that "post-training learns action-conditioned visual representations that transfer across scene distributions." However, the small gap may instead indicate that the evaluation scenes are not distributionally distinct enough from the training scenes to constitute a meaningful generalization test. Both HSSD and HM3D contain similar indoor room structures; a stronger test would involve a more radically different domain. The claim of cross-domain transfer should be tempered accordingly.

### Trivial:

- The fixed action-to-control conversion parameters (0.2m translation, 22.5° rotation in Section 2.2) are standard for Habitat-based navigation and do not require independent justification for this benchmarking context.

## Nice-to-Haves

- Quantification of prediction failure rates: What fraction of world model rollouts contain hallucinations or action-control violations, and how does this correlate with downstream task failure? This would directly address the structural gap in the planning framework.
- Error accumulation analysis: Measure how prediction quality (e.g., LPIPS to ground truth) degrades with rollout horizon length, which would explain the observed difficulty of long-horizon planning and justify the chosen horizon limits.
- Minimum viable post-training data: The scaling law in Figure 6 shows consistent gains, but identifying the data floor needed to outperform zero-shot baselines would have practical significance for resource-constrained settings.
- Semantic controllability metrics beyond LPIPS: Object-state consistency or physical plausibility scores would better capture whether the world model understands causal dynamics, not just pixel similarity.
- Real-robot validation or cross-simulator generalization (e.g., testing Habitat-trained models on AI2-THOR) to strengthen generalization claims.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"First closed-loop benchmark" claim conflicts with VP2:** The paper explicitly acknowledges VP2 in Related Work and differentiates based on task diversity, modern architectures, and embodied task scope. The abstract's claim of "the first open platform that benchmarks WMs in a closed-loop world that mirrors real agent-environment interactions" is qualified by the specific framing ("mirrors real agent-environment interactions"). The differentiation from VP2 is stated, not ignored.
- **Proprietary model (Runway Gen4) lacks post-trained variant, creating unfair comparison:** The asymmetry here does not clearly favor the authors—Runway Gen4 zero-shot already outperforms all open post-trained models on AR, which actually weakens the authors' post-training narrative. This is a completeness concern, not a bias favoring the authors.
- **Figure rendering/formatting issues (Figures 2, 5, 6, 7; Table 4 duplication):** Per rules, pure formatting/style nitpicks are removed. The instructions explicitly note that PDF parser artifacts are not paper problems.
- **Missing real-robot validation:** The paper's stated scope is benchmarking world models in simulated closed-loop environments. Criticizing the absence of real-robot experiments is scope creep.
- **Missing task-specific RL baselines (e.g., PPO on Habitat):** The paper benchmarks world models' utility for planning, not whether world-model planning outperforms all possible alternatives. The appropriate baselines are policies with and without world model augmentation, which are provided.
- **Heavy citation of 2025 work:** Per rules, all cited references are assumed to exist as of April 2026. This is not a valid criticism.
- **Missing broader impact discussion:** Nice-to-have at best; not a core flaw for a benchmarking paper.
- **Evaluation code not explicitly linked:** Per rules, nitpicks about reproducibility such as undisclosed implementation details or large artifacts impractical to include are removed. The paper provides substantial reproducibility information including project page, repository links, training configs, prompt templates, and dataset construction algorithms.

## Novel Insights

The most striking underexplored finding is the LPIPS-revision-outperforms-VLM-revision result (Table 5). This suggests that for goal-conditioned navigation, a pixel-level similarity signal to the goal image may be more reliable than a VLM's subjective judgment of progress. This could imply that current VLMs, despite their semantic understanding, are poorly calibrated as reward models for spatial navigation—possibly because they judge visual plausibility rather than geometric progress, echoing the paper's central thesis that visual quality ≠ task success even within the scoring mechanism itself. If the revision policy itself suffers from the same visual-quality bias that the paper critiques in world model evaluation, this creates a recursive problem: the very tool used to judge world model rollouts may be misaligned with task success. Investigating this would deepen the paper's conceptual contribution substantially.

## Suggestions

- Add confidence intervals or bootstrap-based error bars to all main results tables, especially for tasks with fewer than 200 episodes. This is essential for a benchmark paper whose contribution is empirical evaluation.
- Investigate the LPIPS-vs-VLM revision policy gap in depth: test whether VLM revision fails because it rewards visual plausibility over geometric progress, and whether a goal-image-aligned reward model closes the gap.
- Quantify inference-time cost (wall-clock seconds or GPU-hours per episode) for the inference-time scaling analysis, and report the latency per decision step for each world model to clarify real-time feasibility.
- Add a failure-mode analysis for manipulation: break down which subtasks (e.g., Push Buttons vs. Stack Cups) benefit most/least from world model augmentation, and correlate with prediction quality metrics specific to dynamics (e.g., object displacement error).
- Incorporate a simple prediction-confidence heuristic into the planning loop—for example, downweighting rollouts whose first predicted frame has high LPIPS from the current observation—to address the structural vulnerability to world model hallucination.

---

**Evaluation by axis:**

- **Novelty:** Moderate-to-high. The closed-loop evaluation paradigm for world models is a genuine contribution. The unified API and planning framework are well-designed though the individual components (beam search, VLM scoring) are not novel in isolation.
- **Technical soundness:** Moderate. The framework is cleanly formalized, but the absence of statistical rigor and the structural gap regarding prediction reliability are significant concerns for a benchmark paper.
- **Empirical support:** Moderate-to-strong. The breadth of models and tasks is impressive, and the core findings (visual quality ≠ task success, post-training scaling, inference-time scaling) are well-supported. However, the lack of error bars and the under-analysis of key counterintuitive results weaken the evidential base.
- **Significance:** High. This benchmark addresses a clear gap in the field and the findings have immediate practical implications for how the community should allocate resources in world model development.
- **Clarity:** Good. The paper is well-organized with clear formalization, though some ablation results deserve deeper discussion.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
