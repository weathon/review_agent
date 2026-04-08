=== CALIBRATION EXAMPLE 38 ===

# Final Consolidated Review
## Summary

World-In-World introduces the first benchmark that evaluates generative world models (WMs) through closed-loop task success in embodied settings rather than open-loop visual quality metrics. It provides a unified action API and an online proposal-simulation-revision planning framework that enables heterogeneous WMs (text-conditioned, camera-conditioned, action-conditioned) to be integrated into four embodied tasks spanning perception, navigation, and manipulation, while also analyzing post-training data scaling and inference-time compute scaling.

## Strengths

- **Paradigm shift in WM evaluation with systematic evidence.** The paper demonstrates that visual quality (aesthetic + image quality scores) correlates poorly with closed-loop task success across multiple models (Figure 2, Figure 5a), while controllability (action-conditioned alignment measured by 1-LPIPS) shows a clearer positive correlation (Figure 5b). This is a concrete, evidence-backed finding that directly challenges how the community currently evaluates world models, going beyond informal observations.

- **Unified framework enabling fair cross-architecture comparison.** The unified action API (Section 2.2) translates agent actions into the diverse input modalities expected by different WMs (text prompts, camera trajectories, low-level actions), and the proposal-simulation-revision planning strategy (Section 2.1) provides a common evaluation protocol. This enables direct comparison of 10+ world models with fundamentally different control interfaces under identical task conditions—a non-trivial engineering and design contribution.

- **Counterintuitive finding on revision policies.** Table 5 shows that a simple LPIPS-based revision policy substantially outperforms the more expensive VLM-based revision on ImageNav (SVD†: 47.92% vs. 43.05% SR; Wan2.1†: 48.61% vs. 45.14% SR). This suggests that perceptual similarity to the goal can be more effective for planning than language-model scoring, an insight with practical implications for WM-based planning design.

## Weaknesses

- **"Scaling law" claim is overstated for the evidence provided.** The abstract claims "the first data scaling law for world models in embodied settings," but Figure 6 presents only 4 data points (400, 4K, 40K, 80K) per model. True scaling laws (in the sense used in scaling-law literature) require consistent trends across several orders of magnitude and ideally some theoretical grounding. The empirical trends are positive and useful, but labeling them a "scaling law" overstates the evidence. This matters because it sets reader expectations for rigor that the data does not support.

- **Inference-time scaling lacks a compute-matched baseline, making causal attribution unclear.** Figure 7 shows that increasing the average number of WM inferences per episode improves AR success rate. However, the paper does not compare against a compute-matched control that samples more candidate action plans and evaluates them *without* WM rollouts (e.g., using only the VLM proposal/revision policy with more samples). Without this control, it is impossible to determine whether the gains come from the WM's predictive value or simply from exploring more candidates under any evaluator. This undermines one of the paper's three main findings.

- **The controllability-vs-visual-quality finding is demonstrated on only one task.** The key claim that controllability matters more than visual quality (Figure 5) is supported exclusively by Active Recognition results. This finding is not verified on ImageNav, A-EQA, or Manipulation, where different task demands (long-horizon navigation, open-ended QA, contact-rich dynamics) may shift the relative importance of different WM properties. Without cross-task validation, the generality of this central insight remains uncertain.

- **Inconsistent and incomplete model coverage across tasks weakens the benchmark's comprehensiveness claim.** The paper evaluates a large suite of WMs on AR and ImageNav (Table 1) but a dramatically reduced subset on A-EQA (Table 2: only PathDreamer, SE3DS, NWM for zero-shot; 7 post-trained models) and Manipulation (Table 3: only 2 zero-shot and 2 post-trained models). This inconsistency makes it impossible to draw unified conclusions about how different WMs compare across all four tasks and limits the benchmark's value as a comprehensive evaluation platform.

- **Marginal manipulation improvements with limited analysis of failure modes.** WM-augmented manipulation shows only 2-4% SR improvement over the VLM baseline (44.5% → 46.5% for the best model), and the 3D diffusion policy baseline (24.0% SR) shows larger gains with SVD† (44.7%) but this may reflect the weaker starting point. The paper acknowledges that "precise modeling of interactions and dynamics remains difficult" but provides no quantitative failure mode analysis—e.g., what fraction of episodes the WM actively *hurts* performance, what types of physics violations occur, or whether gains concentrate in specific subtasks. Without understanding when and why WMs fail to help, the practical utility of WM-based manipulation planning remains poorly characterized.

- **No computational cost analysis for inference-time scaling.** The paper advocates inference-time scaling as a key finding but provides no wall-clock time, FLOPs, or latency measurements. Generating multiple video rollouts per decision step using models like Wan2.1 (14B parameters) is computationally expensive. Without cost reporting, practitioners cannot assess whether the SR improvements justify the computational overhead, or compare against alternative approaches on a compute-normalized basis. Section 4 acknowledges this concern qualitatively but does not provide the quantitative analysis needed to evaluate it.

- **LPIPS-based revision outperforms VLM-based revision, yet VLM is the default—unexplained.** Table 5 shows that LPIPS revision achieves substantially higher SR and SPL than VLM revision for both SVD† and Wan2.1† on ImageNav, yet the paper uses VLM revision as the default throughout all other experiments. The paper does not discuss why a simpler, cheaper, and more effective revision policy is not adopted as the standard, nor whether this finding generalizes to other tasks. This is a substantive design choice that affects the reported results and deserves analysis.

## Nice-to-Haves

- **Statistical variance reporting.** The paper reports single-run results on fixed evaluation episodes. While this is common in embodied AI benchmarks, reporting confidence intervals or standard deviations across multiple sampling runs would strengthen confidence in the reported improvements, especially where gains are marginal (e.g., manipulation: 2-4%).

- **Comparison to traditional (non-generative) model-based planning baselines.** Showing whether simpler dynamics models or heuristic planners achieve similar gains at lower cost would help establish the unique value of generative WMs over cheaper alternatives, but this is outside the paper's stated scope of benchmarking existing WMs.

- **Direct experimental comparison with VP2.** The paper distinguishes itself from VP2 in related work but a direct empirical comparison on shared task dimensions would further clarify the benchmark's specific advances.

- **Failure mode quantification.** Reporting what fraction of episodes WM integration decreases success rate, and analyzing what conditions lead to harmful predictions, would strengthen the discussion of limitations.

- **Real-world pilot experiment.** All evaluation is in simulation; even a qualitative demonstration on a physical robot would help bridge the stated goal of mirroring "real agent-environment interactions," though this is clearly beyond the paper's benchmarking scope.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Title is opaque.** (Formatting/style nitpick—removed per hard rules.)
- **Weakness: Notation inconsistency between o_t and **o**_t.** (Formatting nitpick—removed per hard rules.)
- **Weakness: Appendix dependency for key details.** (The paper provides post-training configs in Appendix C-D and prompts in Appendix F, which is standard for ICLR. Removed as style nitpick.)
- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** (Hyperparameters are provided in Appendices C-D and Table 8-10. Removed per hard rules on reproducibility nitpicks.)
- **Weakness: Code availability and model checkpoint availability.** (Project page is cited; post-training procedure is fully described. Removed per hard rules—cannot question availability of cited resources.)
- **Weakness: Including proprietary Runway Gen4 model.** (Including proprietary models for completeness is standard practice; paper clearly marks it. Not a substantive weakness.)
- **Weakness: Episode counts lack power analysis.** (Generic one-size-fits-all weakness; episode counts are comparable to standard embodied AI benchmarks. Removed per soft rules.)
- **Weakness: No broader impact / negative societal impact statement.** (This is a formatting/convention concern, not a substantive weakness of the research. Moved to nice-to-have territory but ultimately removed as it's a meta-convention rather than a paper flaw.)
- **Weakness: Exploration-exploitation trade-off not specified in π_revision.** (The paper describes how π_revision is instantiated for each task in Appendix B.5.2; the scoring function implicitly handles this. Partially addressed by paper—removed per soft rules.)
- **Weakness: Action discretization asymmetry between VLM and 3D diffusion policy in manipulation.** (The paper explicitly describes both policy configurations in Appendix B.4. This is a design necessity for different policy types, not an unaddressed flaw. Weakened and removed.)

## Novel Insights

The most striking insight from this work—underappreciated even in the paper's own framing—is the tension revealed in Table 5: simpler perceptual-matching revision (LPIPS) substantially outperforms expensive VLM-based revision for goal-conditioned navigation. This suggests that for tasks where a clear reference signal (the goal image) exists, the world model's value lies primarily in generating *comparable* visual states rather than in producing outputs that a VLM can reason about linguistically. This has a counterintuitive implication: the current trend toward ever-larger VLM-based planning pipelines may be overkill for goal-conditioned embodied tasks, where lightweight perceptual similarity suffices. A second underexplored insight is that the cross-domain transfer results (Table 6: HSSD→HM3D) show that post-training on synthetic scenes yields meaningful gains on real-world scans, suggesting that action-conditioned representations transfer more robustly than visual appearance—consistent with the paper's controllability finding but with broader implications for sim-to-real transfer in embodied world models.

## Suggestions

- **Replace "scaling law" with "scaling trend"** in the abstract and throughout, or provide additional data points spanning more orders of magnitude with analysis of saturation behavior to justify the stronger claim.

- **Add a compute-matched baseline** for the inference-time scaling experiment: sample M×K candidate plans from the proposal policy and evaluate them using the revision policy *without* WM rollouts, to isolate the contribution of world model predictions from the contribution of simply exploring more candidates.

- **Replicate the controllability-vs-visual-quality correlation analysis** (Figure 5) on at least ImageNav and A-EQA to validate whether this finding generalizes beyond Active Recognition.

- **Discuss why LPIPS-based revision outperforms VLM-based revision** and test whether this finding holds across all four tasks; if it does, consider making LPIPS revision the default to present the strongest possible WM-augmented results.

- **Report wall-clock time or FLOPs per decision step** for the inference-time scaling analysis, enabling practitioners to assess the practical viability of WM-augmented planning.

---

**Axis assessments:**

- **Novelty:** Moderate-to-high. The closed-loop evaluation paradigm and unified API are genuinely new contributions; the specific empirical findings (controllability > visual quality, LPIPS > VLM revision) are novel and counterintuitive. However, the planning framework itself (propose-simulate-revise) follows established model-predictive-control principles, and the post-training recipe is straightforward fine-tuning.

- **Technical soundness:** Moderate. The framework is well-designed and experiments are extensive, but several key claims lack rigorous support: the "scaling law" has minimal data points, inference-time scaling lacks a compute-matched control, and the controllability finding is task-limited. The LPIPS-vs-VLM revision discrepancy is unexplained.

- **Empirical support:** Mixed. Strong evidence for the core claim that visual quality ≠ task success and that post-training improves WMs. Weaker evidence for the scaling law and inference-time scaling claims due to missing controls and limited data points. Manipulation results are marginal and under-developed in both model coverage and failure analysis.

- **Significance:** High. This work addresses a genuine and timely gap in how the community evaluates world models, and its findings—if confirmed across tasks—could redirect research priorities from visual fidelity toward controllability. The benchmark infrastructure could become a standard evaluation tool.

- **Clarity:** Good. The paper is well-organized with clear section structure, formal notation, and comprehensive appendices. The unified framework is clearly described. Minor issues with claim calibration (overstated "firsts" and "scaling law") do not significantly impede understanding but do affect trust in the conclusions.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
