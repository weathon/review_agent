=== CALIBRATION EXAMPLE 42 ===

# Final Consolidated Review
## Summary

The paper introduces **wd1**, a ratio-free reinforcement learning method for diffusion-based large language models (dLLMs) that reformulates the standard GRPO objective as a weighted log-likelihood, avoiding the need to approximate policy ratios (which amplify likelihood approximation errors exponentially in dLLMs). The weights combine positive (exp(ψÂ)) and negative (exp(−ψÂ)) advantage terms to reinforce high-advantage completions while penalizing low-advantage ones. The method is further extended to **wd1++**, which leverages intermediate denoising-step completions. Theoretically, the paper shows wd1 is equivalent to energy-guided discrete diffusion training combined with negative-sample unlearning. Experiments on LLaDA-8B show substantial gains over the d1 baseline on Sudoku/Countdown, and wd1++ achieves SOTA on MATH500 (44.2%) and GSM8K (84.5%) with only 20 training steps.

## Strengths

- **Ratio-free formulation eliminates exponential error amplification.** The core insight—that computing policy ratios r ≈ exp(ϕ_πθ − ϕ_π_old) in dLLMs amplifies approximation error exponentially (formally shown in Appendix A.1, Eq. 15)—is well-motivated and the solution of replacing this with a weighted log-likelihood requiring only a single likelihood approximation is clean and effective. The error bound for wd1 scales linearly (Eq. 16) rather than exponentially, which is a genuine algorithmic advantage over ratio-based methods.

- **Theoretical depth: energy-guided diffusion + unlearning interpretation.** Theorem 1 and Remarks 1–2 provide a principled interpretation of wd1 as jointly performing advantage-weighted denoising concrete score matching (energy-guided sampling) and negative-sample unlearning via ELBO minimization. This goes beyond standard RL framing and connects dLLM alignment to the broader energy-based modeling literature.

- **Significant empirical improvements without SFT.** On Sudoku (256 tokens), wd1 achieves 76.4% vs. d1's 17.6%—a dramatic improvement—without requiring the SFT stage that d1 depends on. The ablation in Table 4 confirms that removing SFT from wd1 barely hurts performance, while removing the negative weight w⁻ causes catastrophic degradation (Sudoku drops from 76.4% to 6.69%), validating the core design.

- **Strong efficiency gains.** Table 2 shows wd1 reduces per-step time (81.16s vs 103.5s), FLOPs (8.89×10¹⁵ vs 9.92×10¹⁵), and NFEs (µ vs µ+2) compared to d1, while eliminating the 2-hour SFT stage entirely. This is a meaningful practical improvement.

## Weaknesses

- **Ambiguity between theoretical sampling distribution and implementation.** The theoretical derivation (Eq. 7) requires sampling from the geometric mixture π_old^ref ∝ π_old^{λ/(λ+β)} · π_ref^{β/(λ+β)}, and Section 3.2 explicitly states completions are "sampled from geometric mixture π_old^ref." However, Algorithm 1 (Line 4) writes "o_i ∼ π_old(·|q)." The resolution appears to be that β=0 and λ=1 are used in all experiments (Section 5, Implementation), making π_old^ref = π_old and thus resolving the inconsistency in practice—but the paper never explicitly states this. Readers must infer it. This creates confusion about whether the "single approximation" claim holds generally or only when β=0. *This matters because the core selling point of requiring only one likelihood approximation is undermined if β>0 requires reference likelihood estimation (as Appendix B.3 acknowledges).*

- **Compute accounting for wd1++ is incomplete.** Table 3 (right) claims wd1++ uses "10× fewer rollouts" than baselines (1,280 vs 12,000–30,000). However, wd1++ leverages all intermediate denoising-step completions (Section 3.3), meaning each "rollout" generates L additional training sequences (L = number of diffusion steps, up to 128 per Table 6). The rollout count metric obscures the actual computational cost of wd1++. Table 2 provides FLOPs for the base wd1 but **not** for wd1++, making it impossible to assess whether the efficiency gain comes from algorithmic design or from simply extracting more training signal per trajectory.

- **Task-dependent effectiveness of base wd1 is striking and unexplained.** Base wd1 improves Sudoku by +58.8 pp and Countdown by +16 pp over d1, but achieves **zero improvement** on MATH500 (39.0% for both wd1 and d1 at 256 tokens). Only wd1++ (with intermediate steps and full fine-tuning) achieves meaningful MATH500 gains. This asymmetry is acknowledged in the results but never analyzed. Understanding why wd1's weighted log-likelihood is so effective on symbolic reasoning tasks but ineffective on mathematical reasoning without the wd1++ extension would strengthen the paper's contribution and inform practitioners about when to apply each variant.

- **Single-seed results for RL benchmarks.** All main results (Tables 1, 3) appear to be from single training runs. Figure 4 (right) shows one additional seed for MATH that reveals different early training dynamics, and the paper acknowledges this is a limitation. For RL fine-tuning—where variance is notoriously high—and especially when making SOTA claims (44.2% MATH500), reporting at least 3 seeds with standard deviations would significantly strengthen confidence in the results. This is a community-standard expectation for empirical RL papers at ICLR.

- **Reverse-KL trust region choice needs stronger justification.** Equation (4) uses D_KL(π_θ ‖ π_old) (reverse-KL) rather than the forward-KL D_KL(π_old ‖ π_θ) used in standard TRPO. While Theorem 2 proves monotonic improvement, reverse-KL is zero-forcing—it allows π_θ to place mass in regions where π_old has low density, where the advantage estimate A_π_old is unreliable. The paper should justify why reverse-KL is appropriate here (e.g., because the closed-form solution in Eq. 5 enables the weighted log-likelihood formulation, or because the weight normalization in Eq. 9 implicitly constrains support). Without this, the theoretical guarantee may be looser than forward-KL alternatives in practice.

## Nice-to-Haves

- **Comparison with AR-based RL on identical benchmarks.** Showing whether dLLM + wd1 is competitive with, e.g., GRPO on a comparable autoregressive model (LLaMA-8B) on the same math tasks would contextualize the dLLM RL paradigm vs. the standard AR approach. This is outside the paper's stated scope (RL for dLLMs) but would help practitioners assess the viability of the dLLM + RL pipeline.

- **Empirical variance measurements.** Figure 1 shows ratio variance for baselines, but direct measurements of gradient variance or objective variance for wd1 vs. d1 across training steps would empirically validate the core variance-reduction claim.

- **Likelihood approximation ablation.** Since the biased d1-style approximation is a shared component, ablations varying the approximation quality (e.g., ELBO with different sample sizes of t) would demonstrate wd1's robustness to this remaining source of error.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Weakness: "Residual bias from likelihood approximation undermines theoretical guarantees."** The paper explicitly acknowledges this in Limitations (Section D, paragraph 3) and shows empirically that wd1 works well despite the bias. The theoretical guarantees assume exact likelihoods, which is standard for all RL-for-LLM papers. The ablation on ψ (Figure 4) partially addresses robustness. Weakening: the authors already address this reasonably.

- **Weakness: "No comparison with AR-based RL methods (PPO/GRPO on LLaMA)."** This is outside the paper's stated scope (RL for dLLMs). The paper is evaluating whether wd1 is better than existing dLLM RL methods, not whether dLLMs are better than AR models. Demanding this is scope creep.

- **Weakness: "Statistical significance requirements (confidence intervals)."** Single-run evaluation is the norm in the dLLM RL literature (the baseline d1 also reports single runs). While multiple seeds would strengthen the paper, demanding confidence intervals for large-scale benchmarks where single-run evaluation is standard is applying a higher bar than the community norm.

- **Weakness: "wd1++ depends on specific inference mechanics (confidence-based remasking)."** This is an inherent property of wd1++'s design, not a flaw. The paper clearly scopes wd1++ as an extension that leverages intermediate denoising steps, and the base wd1 method does not depend on this.

- **Weakness: "Baseline reproduction underperforms reported d1."** The paper transparently reports both reproduced and original d1 numbers (Table 7). The reproduced version is used for fair comparison under identical conditions. This is a methodological strength, not a weakness.

- **Weakness: "Missing limitation about mode collapse from aggressive negative weighting."** The ablation study (Table 9, Figure 2) empirically addresses this by showing balanced weights (0.5) perform best and training is stable. This concern is not borne out by the evidence.

## Novel Insights

The dual-weight formulation (w⁺ − w⁻) in wd1 can be understood as a form of **soft rejection sampling** that simultaneously upweights high-advantage and downweights low-advantage trajectories, rather than the hard binary split used in methods like RAFT. This is subtly different from standard advantage-weighted regression (AWR): AWR's exp(A) weighting suffers from vanishing gradients on negative samples (w⁺ ≈ 0), effectively wasting them. The w⁻ term recovers gradient signal from these samples by treating them as explicit unlearning targets. The theoretical equivalence to energy-guided diffusion (Remark 1) then reveals that this is not merely a computational trick—it corresponds to steering the entire denoising trajectory toward high-advantage regions at every diffusion timestep, not just at the sequence level. This insight suggests that future dLLM RL methods could benefit from richer per-timestep advantage signals rather than relying solely on outcome-level rewards.

## Suggestions

- **Clarify the sampling distribution explicitly.** Add a sentence in Section 3.2 or the Implementation subsection stating that with β=0, λ=1 (as used in all experiments), π_old^ref = π_old, so Algorithm 1 is consistent with the theoretical derivation. Also clarify whether the "single approximation" claim holds only when β=0, and discuss the cost when β>0.

- **Report FLOPs or wall-clock time for wd1++.** Even an approximate measurement would allow readers to assess whether wd1++'s rollout efficiency translates to actual compute efficiency, or whether the intermediate-step extraction adds significant overhead.

- **Add 2–3 seeds for the main benchmarks** (at minimum MATH500 and GSM8K with wd1++), even if reported in an appendix, to establish reliability of the SOTA claims.

- **Analyze the Sudoku-vs-MATH discrepancy.** A brief discussion of why wd1 excels on symbolic tasks but not math reasoning (e.g., hypothesis: Sudoku rewards are denser/more informative, enabling the exponential weights to better separate positive/negative samples) would help practitioners set expectations and guide future improvements.

---

**Quality Assessment (verbal):**

- **Novelty:** High. The ratio-free formulation specifically designed for dLLMs' intractable likelihoods, combined with the energy-guided diffusion interpretation, is a genuine contribution that moves beyond naive GRPO adaptation.

- **Technical soundness:** Moderate-to-good. The theoretical derivations are solid, but the gap between theory (sampling from π_old^ref with general β, λ) and practice (β=0, λ=1) is insufficiently acknowledged, and the reverse-KL choice warrants more discussion.

- **Empirical support:** Moderate. Results on Sudoku/Countdown are very strong, but the base wd1's negligible improvement on MATH500 and single-seed reporting weaken the empirical case for the method's generality. wd1++ results are impressive but the compute accounting is incomplete.

- **Significance:** High for the dLLM RL community. The paper provides a principled, efficient alternative to ratio-based methods at a time when dLLMs are rapidly gaining traction.

- **Clarity:** Moderate. The core ideas are presented clearly, but the sampling distribution ambiguity and the wd1/wd1++ conflation in the abstract create unnecessary confusion.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
