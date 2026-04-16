## Summary

The paper proposes end-to-end learnable Gaussian Mixture Priors (GMPs) for diffusion-based sampling methods within a variational inference framework. By replacing the standard single-Gaussian prior with a learnable mixture model, the method aims to address three challenges: poor exploration (C1), large discretization errors from prior-target support mismatch (C2), and mode collapse from reverse KL minimization (C3). The authors also introduce an Iterative Model Refinement (IMR) strategy that progressively adds mixture components during training, initialized via a heuristic using candidate samples. Experiments across multiple diffusion samplers (DIS, MCD, CMCD, DBS) and benchmarks show consistent ELBO/ESS improvements, with IMR enabling strong mode coverage on the high-dimensional Fashion target.

## Strengths

- **Clear problem identification and practical framing:** The three challenges (C1–C3) are well-identified with an effective illustration (Figure 1), making the case for learned priors intuitive and compelling.

- **Generality and consistency:** The framework applies cleanly to four distinct diffusion samplers (Table 1) and shows improvements across all of them on the Funnel and six real-world benchmarks (Tables 2, Figure 3). The gains from GP over fixed Gaussian and from GMP over GP are consistent and meaningful—for instance, DBS-GMP achieves the best ELBO on 4 of 6 real-world tasks.

- **Empirically validated on diverse benchmarks:** The paper evaluates on both synthetic (Funnel d=10, Fashion d=784) and real-world tasks (d=25–61) with multiple metrics (ELBO, ΔlogZ, ESS, W₂², EMC), following the Blessing et al. (2024) evaluation protocol.

- **The Fashion + IMR experiment is informative:** The color-coded visualization in Figure 4 clearly demonstrates that IMR enables per-component mode specialization, and the EMC metric meaningfully quantifies mode coverage. This experiment also honestly reveals the tradeoff between ELBO/ΔlogZ and mode coverage.

- **Technically sound VI formulation:** The extension of the extended ELBO to learnable priors via reparameterization (Eqs. 12–13) is straightforward and correct. Proposition 1 provides a clean theoretical basis for extending denoising diffusion to arbitrary stationary distributions.

## Weaknesses

### Major:

- **The "GMP addresses mode collapse" narrative is overstated.** The paper motivates GMPs partly as a solution to C3 (mode collapse from reverse KL), but the Fashion experiment—the only strongly multimodal benchmark—shows that GMP without IMR barely improves mode coverage (EMC: 0.007→0.012 vs. DIS-GP). The only configuration achieving good coverage (DIS-GMP+IMR, EMC=0.78) critically relies on the IMR scheme with MALA-generated candidate samples. The paper's framing in the abstract and introduction implies that GMPs themselves counteract mode collapse, but what actually works is GMP+IMR+MALA. Furthermore, the extended ELBO being optimized is a path-space KL rather than a marginal reverse KL, so the classic "reverse KL is mode-seeking" argument does not straightforwardly apply. The paper should reframe C3 as addressed by the combination of expressive priors + iterative refinement, not by GMPs alone.

- **"No additional target evaluations" is misleading for the IMR setting.** The abstract claims improvements "without requiring additional target evaluations," but IMR as evaluated on Fashion uses MALA chains that require gradient evaluations of the unnormalized target density. Section 6.2 notes this cost is "comparable to a single gradient step," but this is per candidate sample per component addition—and the most compelling empirical result (Fashion) depends on it. The base GMP method does work without extra evaluations on other benchmarks, but the paper should clearly separate the "free" core contribution (learnable GMP) from the "costly" enhancement (IMR+MALA) and avoid implying the strong Fashion results come at no extra cost.

- **ELBO/ΔlogZ degradation with GMP+IMR on Fashion is inadequately explained.** DIS-GMP+IMR achieves worse ELBO (−62.482 vs −24.712 for DIS-GP) and worse ΔlogZ (27.645 vs 10.581) while improving W₂² and EMC. The paper attributes this to ELBO being "not well-suited for quantifying model performance for multi-modal targets," but since ELBO is the very training objective, its degradation signals a genuine tension in the method. The paper acknowledges this briefly but does not analyze whether this is an optimization issue, a consequence of the KL objective being misaligned with mode coverage, or an artifact of IMR's component-addition schedule. This deserves deeper investigation given that ELBO is the only tractable objective for training.

### Minor:

- **GP→GMP improvements are marginal on many real-world tasks.** In Table 2, MCD-GP vs MCD-GMP on Credit: −585.350 vs −585.276; CMCD-GP vs CMCD-GMP on Ionosphere: −111.687 vs −111.682. These differences are within error bars. The method's value above a learned single Gaussian is small for non-multimodal targets, suggesting the main benefit of GMPs is for multimodal or support-mismatched distributions.

- **Proposition 1's stationary distribution argument has a gap for finite-N.** The proposition covers T→∞, but experiments use N=128. Learning δt mitigates but does not eliminate this gap, and no analysis of the mismatch between the learned prior density and the backward marginal at time 0 is provided. The paper is transparent about this ("empirically, we observe substantial improvements") but the theoretical motivation is weaker than presented.

- **No comparison with alternative expressive priors.** GMPs are one choice among many expressive prior families (normalizing flows, hierarchical priors, mixtures of Student-t). The paper does not discuss why Gaussian mixtures are the right choice beyond computational convenience and reparameterizability, nor compare against alternatives.

- **Computational overhead not quantified.** Evaluating a K-component GMP density at each diffusion step costs O(Kd). With K=10 and N=128, this is nontrivial, especially compared to a single Gaussian. No wall-clock times or per-step cost analysis is provided.

## Trivial

- The claim that GMPs reduce the number of diffusion steps needed (C2) is not directly tested—the ablation (Figure 5) shows that increasing K and N both improve ESS, but does not show that GMPs allow fewer N for the same ESS compared to fixed priors.

## Nice-to-Haves

- Ablation of the IMR heuristic (Eq. 22) against simpler alternatives (e.g., random initialization, high-ρ/low-p₀ samples) to assess whether the full objective is necessary or whether MALA samples alone drive the improvement.
- Training curves comparing GMP vs. GP convergence speed to assess the claim about C2 (faster convergence due to better support matching).
- A synthetic multimodal benchmark with known ground truth Z, where the number and separation of modes can be controlled, to cleanly test the mode-collapse narrative.

## Removed Points

- **"The comparison with baselines appears computationally unbalanced."** The paper follows the Blessing et al. (2024) evaluation protocol for fair comparison; the claim of unfairness lacks evidence that baselines were given less hyperparameter tuning. Without concrete evidence of inequitable treatment, this is speculative. That said, reporting wall-clock times would still strengthen the paper.

- **"The theoretical justification for learning arbitrary priors in denoising diffusion (Section 4.1) is hand-wavy around finite-time behavior."** I kept a version of this in Minor weaknesses, but the harsh critic's framing as a "Critical Issue" goes too far. The paper is transparent about the empirical nature of the finite-N claim, and the stationary result (Prop. 1) correctly motivates the construction even if it doesn't guarantee finite-time behavior.

- **"IMR's component-initialization heuristic is under-motivated and not cleanly evaluated."** I moved this to Nice-to-Haves since the heuristic works empirically and the paper provides intuitive justification; demanding ablations of every design choice goes beyond what's standard for this venue.

- **"The paper lacks comparisons with alternative approaches to mode collapse."** This asks the paper to address problems outside its stated scope—it focuses on improving existing diffusion samplers via better priors, not on comparing alternative divergence objectives. This would strengthen the paper but is not a core flaw.

- **"Scalability concerns in very high dimensions."** The paper includes a d=784 experiment (Fashion). Demanding additional d=500+ tasks goes beyond what's standard; the d=784 experiment already pushes beyond typical benchmarks in this area.

## Novel Insights

The Fashion experiment reveals an important and underappreciated tension: the extended ELBO training objective—standard in diffusion-based VI—can be fundamentally misaligned with mode coverage. DIS-GP achieves the best ELBO/ΔlogZ among diffusion-based methods on Fashion while having near-zero EMC, and improving EMC via IMR degrades ELBO. This suggests that optimizing the standard path-space reverse KL is insufficient for multimodal targets, and that the community needs metrics and objectives that better capture distributional coverage. This finding echoes observations in Blessing et al. (2024) but is made more concrete here by showing that more expressive priors alone do not resolve this misalignment.

## Suggestions

- Reframe C3 as addressed by GMP+IMR combined, not by GMP alone. Qualify the abstract's "without requiring additional target evaluations" to specify this applies to the base GMP method, and clearly separate the computational cost of IMR+MALA.
- Add a paragraph analyzing the ELBO/EMC tension on Fashion rather than dismissing it as a metric issue. Discuss whether alternative divergences (forward KL, α-divergences) might better align training with mode coverage.
- Report wall-clock times for the K=10, N=128 configuration vs. fixed-prior baselines, including any IMR overhead, to contextualize the practical cost of GMPs.

## Score and Decision

**Calibration:** I compared against papers with similar profiles. "Improved sampling via learned diffusions" (h4pNROsO06, scores 6/6/6/8, Accept) and "ELBOing Stein" (2rBLbNJwBm, scores 6/6/6/8, Accept) are diffusion-sampling papers with solid empirical contributions but incremental novelty—they received ~6.5 averages. "Neural Diffusion Models" (hkL8djXrMM, scores 5/5/6/5, Reject) had more fundamental issues. The current paper has consistent empirical improvements and a clean technical formulation, but its core claims about mode collapse are overstated relative to the evidence, and the "no additional target evaluations" framing is misleading for the most compelling experiment. The marginal GP→GMP gains on real-world tasks also temper enthusiasm. This paper is below the accepted calibration papers in terms of how well the claims match the evidence, but above the rejected papers in terms of technical contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>