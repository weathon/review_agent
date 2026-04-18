Now I have a good calibration. Let me compile the final review.

Key calibration summary:
- "Improved sampling via learned diffusions" (h4pNROsO06): avg ~6.5, accepted. Concerns about log-variance theoretical motivation but still accepted due to solid unification framework and good experiments.
- CMCD (PP1rudnxiW): avg ~7.2, accepted. Stronger paper, novel framework, good experiments.
- DGFS (OIsahq1UYC): avg ~7.5, accepted. Good theory and experiments.
- "Adaptive teachers for amortized samplers" (BdmVgLMvaf): avg ~6.5, accepted. No convergence theory, limited high-dim evidence.
- Neural Diffusion Models (hkL8djXrMM): avg ~5.25, rejected. Not convincing effectiveness.
- "Avoiding mode collapse in diffusion RL" (fXkoROek1M): avg ~4, rejected. Incremental, poor presentation.

This paper falls in the category of "solid practical contribution with some theoretical gaps." It's comparable to the "Improved sampling via learned diffusions" paper (scores 6,6,6,8=avg 6.5) and "Adaptive teachers" (scores 5,8,8,5=avg 6.5), both accepted. It has a clear practical contribution (learning mixture priors for diffusion samplers), comprehensive experiments across multiple methods and benchmarks, but some theoretical gaps and mixed results on multimodal targets.

The paper is above the acceptance threshold but not by a huge margin. The key weakening factors are:
1. The Fashion results show GMP without IMR can be worse than GP, undermining the abstract's broad claim about "counteracting mode collapse"
2. The theoretical gap around finite-N coupling for denoising diffusion models
3. Limited evidence for IMR beyond one dataset
4. No compute comparison

I'd place this around 6.0-6.5, comparable to the "Improved sampling via learned diffusions" paper.

Let me now write the review.</think>## Summary

This paper proposes end-to-end learnable Gaussian mixture priors (GMPs) for diffusion-based sampling methods that use variational inference. The key idea is to replace the standard Gaussian prior (which is fixed) with a learned Gaussian mixture that adapts to the target distribution's support, addressing exploration difficulty (C1), discretization error from support mismatch (C2), and mode collapse from reverse KL minimization (C3). The authors extend both denoising-type and annealed Langevin diffusion samplers to support arbitrary learnable priors, and propose an iterative model refinement (IMR) strategy that progressively adds mixture components during training. Experiments across four diffusion-based sampling methods and multiple synthetic/real-world benchmarks demonstrate consistent improvements from learned priors, particularly on the challenging Funnel target.

## Strengths

- **Well-motivated and clearly formulated contribution**: The three challenges (C1–C3) are cleanly identified, and GMPs directly address each one. The idea of learning the prior in diffusion-based VI is natural but had not been systematically explored in this setting.

- **Unified framework across methods**: The approach cleanly integrates with four existing diffusion-based samplers (MCD, CMCD, DIS, DBS) requiring only modifications to the prior and backward process (Proposition 1 for denoising-type models). The presentation in terms of the general SDE framework (Eq. 2) makes the adaptation transparent.

- **Comprehensive experimental evaluation**: The paper evaluates across 8 benchmark tasks, 4 diffusion-based methods, and 3 external baselines (SMC, CRAFT, FAB). The structured X / X-GP / X-GMP comparison protocol provides clear isolation of the contribution of learned priors. The Funnel results (Figure 3) show substantial and consistent improvements across all four methods and all four metrics.

- **Honest evaluation metric discussion**: The paper acknowledges that ELBO and ΔlogZ are mode-seeking and not well-suited for multi-modal evaluation (Section 6.2, citing Blessing et al. 2024), and provides EMC and Sinkhorn distance as complementary metrics onFashion.

- **Qualitative evidence supports mechanism**: Figure 3's visualization of GMP components adapting to the Funnel target's neck and opening provides intuitive validation that the learned prior meaningfully matches the target structure, rather than just adding free parameters.

## Weaknesses

### Fatal
None.

### Major

- **GMP alone fails to address mode collapse on the primary multimodal benchmark, contradicting the abstract's core claim**: On the Fashion target (d=784, the only truly multi-modal benchmark), DIS-GMP without IMR performs *worse* than DIS-GP on ELBO (−38.9 vs −24.7), ΔlogZ (18.1 vs 10.6), and has near-zero EMC (0.012). DIS-GMP even has worse W₂² than plain DIS on this task. The abstract claims GMPs "counteract mode collapse" but the evidence shows this only holds with the additional IMR procedure, which requires external candidate samples. This substantially weakens the broad claim that GMPs are sufficient to address C3.

- **Theoretical gap in finite-N coupling for denoising diffusion models**: Proposition 1 guarantees that the backward process has stationary distribution p₀^φ only in the limit T→∞. The paper uses this to justify learning arbitrary priors via Eq. (18) and learning δt alongside φ, but acknowledges the relaxation time is "unknown for general p^φ." No analysis—either theoretical or empirical—is provided on how well the finite-N coupling holds. This matters because if the marginal at time 0 under the backward SDE does not approximately equal p₀^φ, the ELBO in Eq. (18) may not have the intended interpretation, and claims about "reducing discretization error" (C2) are under-justified. The paper's footnote that "Empirically, we observe substantial improvements for finite values of N" addresses this only anecdotally. This is a meaningful theoretical gap for a methods paper at a top venue.

- **IMR initialization heuristic (Eq. 22) lacks principled justification and has limited evidence**: The heuristic for placing new component means is described as "balancing exploration and exploitation" but has no derivation linking it to an objective that directly measures mode coverage or global ELBO improvement. The only empirical support is the single Fashion experiment with qualitative visualization and EMC. No ablation isolates the effect of Eq. 22 against simpler alternatives (e.g., initializing at high-density MALA samples, or random perturbation of existing component means). This is a claimed contribution that is under-supported.

### Minor

- **"Without requiring additional target evaluations" claim in the abstract is misleading for the full method**: While the basic GMP method indeed requires no additional target evaluations beyond those in standard diffusion-based VI, the IMR procedure uses MALA-generated candidate samples which require evaluating ρ(x) and ∇log ρ(x). The paper acknowledges this in Section 6.2 but the abstract creates a different impression.

- **Marginal GP→GMP improvements on some real-world tasks**: On several real-world benchmarks (e.g., Credit: MCD-GP −585.350 vs MCD-GMP −585.276; Cancer: DIS-GP −85.005 vs DIS-GMP −84.061 with overlapping standard deviations), the improvement from learned single Gaussian to learned GMP is modest relative to variability, suggesting the primary benefit often comes from learning the prior at all (GP) rather than the mixture structure (GMP).

- **No computational cost analysis**: GMPs with K=10 components require evaluating p₀^φ (which scales with K) at each of N diffusion steps, and IMR adds MALA sampling overhead. Without wall-clock or FLOPs comparison, it is unclear whether GMP improvements justify the increased cost relative to, say, running a fixed-prior method with more diffusion steps or larger networks.

### Trivial

- **Notation inconsistency**: The switch from p̃^θ, p̃^γ in Section 3 to p̄^{θ,φ} in Section 4 is not immediately obvious and could confuse readers.

## Nice-to-Haves

- A convergence analysis or empirical study of how ELBO quality depends on N when using learned priors vs. fixed priors, to directly evaluate the C2 (discretization error) claim.
- An ablation of Eq. 22 vs. simpler initialization strategies for IMR.
- Full covariance or low-rank covariance experiments on low-dimensional targets to assess when mixture expressiveness matters vs. component flexibility.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Lack of normalizing flow or other expressive prior baselines"**: The paper's contribution is specifically about Gaussian mixture priors. Requesting comparison with other expressive prior families is scope creep—the appropriate comparison is X vs. X-GP vs. X-GMP, which the paper provides.

- **"No convergence rate guarantees"**: This is a methods paper with an empirical focus. The related work (CMCD, CRAFT, FAB) also lack convergence rate guarantees for their practical implementations. Demanding theoretical convergence guarantees is outside the norm for this community.

- **"Insufficient evidence on truly high-dimensional problems"**: The paper does include a d=784 experiment (Fashion), and the real-world tasks range up to d=61. This is comparable to the evaluation scope of prior work (CMCD, FAB, etc.). While higher-dimensional evaluation would strengthen the paper, d=784 already exceeds what many diffusion-sampling papers evaluate on.

- **"Diagonal covariance limitation"**: This is a design choice discussed in the paper as a practical efficiency consideration. It is not a flaw—the paper never claims full covariance is necessary.

- **"Related work on adaptive priors in VI/SMC"**: Per the rules, I cannot confirm whether specific related works exist and should not flag their absence.

## Novel Insights

The observation that learned GMPs without IMR can *degrade* performance on multimodal targets (Fashion: DIS-GMP worse than DIS-GP on ELBO and ΔlogZ) while IMR with the same GMPs dramatically improves sample quality (as measured by W₂² and EMC) is a genuinely important finding. It suggests that for multi-modal targets, the initialization of mixture components matters far more than the expressiveness of the prior family itself—a nuance that complicates the paper's narrative but is valuable for the community. Additionally, the tension between ELBO optimization and actual mode coverage (GMP+IMR improves W₂²/EMC but worsens ELBO on Fashion) highlights a fundamental limitation of reverse-KL-based VI for multi-modal targets that even mixture priors cannot fully resolve.

## Suggestions

- **Temper the abstract claim**: Replace "GMPs counteract mode collapse" with "GMPs, combined with iterative model refinement, can counteract mode collapse." Acknowledge that mode coverage requires careful initialization (IMR), not just a more expressive prior.

- **Add a compute comparison**: Report wall-clock training times for at least one method (e.g., DIS vs. DIS-GP vs. DIS-GMP vs. DIS-GMP+IMR) to inform practitioners about cost-benefit tradeoffs.

- **Provide finite-N empirical analysis**: Show how ELBO and ΔlogZ change as N varies for learned priors vs. fixed priors to substantiate the C2 claim that GMPs reduce discretization error. This could be a small addition to the existing ablation (Figure 5 already varies N).

- **Ablate IMR initialization**: Compare Eq. 22 against simpler alternatives (e.g., random perturbation of existing means, MALA sample means without the argmax scoring) on at least one task.

## Score and Decision

Calibration anchors:
- "Improved sampling via learned diffusions" (h4pNROsO06): avg 6.5, accepted poster — solid but incremental framework, concerns about theoretical motivation of new divergence, good experiments.
- CMCD (PP1rudnxiW): avg 7.2, accepted poster — strong unifying framework, broader methodological impact, concerns about compute.
- DGFS (OIsahq1UYC): avg 7.5, accepted poster — good theory and experiments for partial trajectory optimization.
- "Adaptive teachers for amortized samplers" (BdmVgLMvaf): avg 6.5, accepted poster — practical method with limited theory, moderate experiments.
- Neural Diffusion Models (hkL8djXrMM): avg 5.25, rejected — not convincing effectiveness.

This paper has comparable strengths to "Improved sampling via learned diffusions" (consistent empirical improvements, clear practical contribution) but with weaker theoretical grounding (the finite-N gap) and a more significant empirical caveat (GMP alone failing on the multimodal benchmark). It is somewhat stronger than rejected papers like "Neural Diffusion Models" in terms of experimental breadth and practical contribution, but weaker than CMCD in terms of theoretical depth and novelty. I rate it marginally above the acceptance threshold, comparable to "Adaptive teachers" (~6.5) and "Improved sampling" (~6.5).

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>