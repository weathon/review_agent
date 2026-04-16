Now I have sufficient context from the paper and calibration papers. Let me synthesize the final review.

## Summary

This paper proposes end-to-end learnable Gaussian Mixture Priors (GMPs) for diffusion-based sampling methods that generate samples from unnormalized target densities. By replacing the standard fixed Gaussian prior with a learnable Gaussian mixture, the method aims to address three challenges: poor exploration (C1), large discretization errors from prior-target support mismatch (C2), and mode collapse from reverse KL divergence (C3). The authors also introduce an iterative model refinement (IMR) strategy that progressively adds mixture components during training, with a principled initialization heuristic for new components.

## Strengths

- **Well-motivated and cleanly formulated.** The three challenges C1–C3 are clearly identified and GMPs are a natural, practical solution. Proposition 1 provides a principled way to extend denoising diffusion models to arbitrary stationary distributions by modifying the drift, directly generalizing the OU process. The unified VI framework (Section 3) is clearly presented and consistent with prior work.

- **Generality of the approach.** The method improves four distinct diffusion-based samplers (MCD, CMCD, DIS, DBS) without method-specific modifications, demonstrating that learnable priors provide broad utility across the diffusion sampling family.

- **Strong empirical results on Funnel and real-world tasks.** On the Funnel target (Figure 3), every method equipped with GMPs substantially outperforms its fixed-prior and learned-Gaussian-prior counterparts across all metrics (ELBO, ΔlogZ, ESS, W₂²). On real-world Bayesian inference tasks (Table 2), DBS-GMP achieves the best or near-best ELBO on 5 of 6 datasets, outperforming or matching FAB. Improvements of GMP over GP and fixed priors are consistent and often large.

- **Effective iterative model refinement.** The IMR strategy with its initialization heuristic (Eq. 22) produces compelling mode coverage on the 784-dimensional Fashion target: DIS-GMP+IMR achieves EMC=0.780 and W₂²=213.8 vs. DIS-GMP without IMR at EMC=0.012 and W₂²=1703.0. The visualization in Figure 4 confirms that mixture components specialize to distinct modes.

- **Useful ablation study.** Figure 5 systematically varies K (mixture components) and N (diffusion steps), providing practical guidance on the trade-offs.

## Weaknesses

### Major:

- **The claim that GMPs alone address mode collapse (C3) is overstated.** On the genuinely multimodal, high-dimensional Fashion target (d=784), DIS-GMP without IMR achieves EMC=0.012 and W₂²=1703.0, barely better than DIS-GP's EMC=0.007. Only DIS-GMP+IMR (which relies on MALA-generated candidate samples) achieves strong mode coverage. The paper's abstract and introduction strongly suggest that GMPs themselves counteract mode collapse, but the experimental evidence shows this is true only for less severely multimodal targets (Funnel, the real-world tasks which appear effectively unimodal). The C3 claim needs to be qualified: GMPs help with moderate support mismatches, but severe multimodal mode collapse requires IMR with external candidate generation. (Sections 1, 5, 6.2; Figure 4)

- **The ELBO–sample quality tension on multimodal targets is a fundamental concern.** On Fashion, DIS-GMP+IMR achieves the best mode coverage (EMC=0.780, W₂²=213.8) but significantly worse ELBO (-62.482) and ΔlogZ (27.645) compared to DIS-GP (ELBO=-24.712, ΔlogZ=10.581). Since ELBO is the training objective, this reveals a tension: optimizing ELBO with reverse KL drives the model toward mode collapse, and the GMP structure alone does not resolve this. The paper acknowledges this (Section 6.2, last paragraph) but frames it as ELBO being "not well-suited" rather than recognizing it as a limitation of the reverse KL objective that their proposed method does not fully address. This matters because ELBO and partition function estimation are core use cases of diffusion-based sampling.

- **The learned prior construction for DIS-type methods lacks direct validation.** Proposition 1 guarantees that for a time-independent drift, the SDE has a given stationary distribution, but says nothing about convergence rate or how closely the finite-T marginal matches p₀^φ. The paper addresses this by learning δt as a parameter, but no diagnostic experiment checks whether the backward process actually reaches p₀^φ at time 0. This is a notable theoretical gap for the DIS variants, since the correctness of the extended ELBO (Eq. 18) relies on this coupling. While the downstream performance metrics suggest the method works in practice, a simple validation on low-dimensional problems (comparing the empirical backward marginal at time 0 to the parametric prior) would strengthen the claim significantly. (Section 4.1)

### Minor:

- **No computational cost analysis.** The paper does not report wall-clock training times or memory consumption for any method. GMPs require evaluating K Gaussian components at each diffusion step, and IMR adds MALA-generated candidates plus retraining. Without cost comparisons, the practical trade-offs are unclear.

- **Limited guidance on choosing K and the IMR schedule.** K=10 is used uniformly, and Figure 5 shows ESS can decrease with larger K on some datasets (e.g., Fashion), but this is not discussed. The IMR hyperparameters (when to add components, how many candidates) are also opaque.

- **GP improvements are often marginal.** On real-world tasks, learned single Gaussian (GP) often provides negligible improvement over fixed priors (e.g., CMCD-GP vs CMCD on Credit: -585.178 vs -586.956; MCD-GP on Funnel: slightly worse ELBO than MCD). Understanding why GP fails to help would clarify what specifically makes GMPs effective—the mixture structure, the increased parameter count, or the multi-modal expressiveness.

### Trivial:

- The abstract's claim "without requiring additional target evaluations" technically refers to GMPs without IMR, but could be misread as applying to all results. A brief clarification in the abstract or introduction would help.

## Nice-to-Haves

- A comparison with alternative expressive prior families (e.g., normalizing flow priors, mixture of Student-t components) would clarify whether GMPs are uniquely suited or just one viable option.
- Experiments on multimodal targets in moderate dimensions (d=10–50, well-separated modes) would isolate the GMP contribution to C3 more cleanly than Fashion (d=784) or Funnel (effectively unimodal).
- Alternative divergence objectives (forward KL, α-divergences) could potentially align ELBO optimization with multimodal coverage—a brief exploration would be valuable.
- Diagnostic experiments checking the backward marginal at time 0 against the learned prior in low dimensions.

## Removed Points

- **"No additional target evaluations" claim is false for IMR experiments.** The harsh critic claims this is "false" and structurally undermines the paper. On re-reading, the abstract states "significant performance improvements across a diverse range of real-world and synthetic benchmark problems when using GMPs without requiring additional target evaluations"—this specifically refers to the GMP experiments on Table 2 and Figure 3, which indeed don't use extra evaluations. IMR is an add-on strategy evaluated separately (Fashion, Figure 4). The claim is technically correct, though could be stated more clearly. **Kept as a minor presentation issue rather than a structural problem.**
- **Limited improvement over non-diffusion baselines (FAB).** The harsh critic flags that improvements over FAB are mixed. This is true but not a weakness of the paper per se: the contribution is showing that learnable GMPs consistently and substantially improve existing diffusion samplers, which is a meaningful contribution even if FAB remains competitive on some tasks. Removed as a standalone weakness since the paper's core claim is about improving diffusion samplers, not about beating all non-diffusion methods uniformly.
- **Comparison with alternative expressive priors.** The spark reviewer requests comparison with normalizing flow priors and other prior families. This is a nice-to-have but not a core flaw; GMMs are a natural, efficient choice given the three desiderata (reparameterizable, efficient to evaluate, multi-modal). Removed as a standalone weakness.
- **Higher-dimensional real-world experiments.** Requesting d>100 real-world Bayesian tasks would be scope creep; the d=784 Fashion experiment already pushes into high dimensions. Removed.
- **Scalability concerns beyond d=784.** While valid as a concern, the paper does demonstrate results in d=784, which is substantial. The neutral reviewer's point is weakened by this existing evidence.

## Novel Insights

The most insightful finding from the reviews and the paper is the fundamental tension between ELBO optimization and multimodal coverage in learnable-prior diffusion samplers. On the Fashion target, better mode coverage (via GMP+IMR) comes at the cost of worse ELBO and partition function estimates. This is because reverse KL—which ELBO maximizes—is mode-seeking and thus favors fitting a single mode well over covering all modes. GMPs partially address this by providing a multi-modal prior, but on truly multimodal targets the reverse KL gradient still pushes toward mode collapse unless augmented by an iterative component-adding strategy (IMR) with external MCMC proposals. This suggests that the community should consider combining learnable priors with alternative divergence objectives (forward KL, α-divergences) or mixed-objective approaches to fully resolve the C3 challenge.

## Suggestions

- **Qualify C3 claims.** Clearly distinguish between what GMPs alone achieve (improved support matching on moderately challenging targets) versus what GMP+IMR+MALA achieves (mode coverage on severely multimodal targets).
- **Add a prior validation experiment.** On a 2D or 5D problem, compare the learned prior p₀^φ to the empirical marginal of the backward process at time 0 (via kernel density estimation). This directly validates the key construction in Section 4.1.
- **Report wall-clock training times.** Even a simple table comparing training time per method would help practitioners assess cost-benefit trade-offs.
- **Discuss the ELBO–coverage trade-off.** Rather than attributing worsened ELBO on Fashion to ELBO being "not well-suited," provide a more thorough analysis of why this happens and what it implies for practitioners who care about both partition function estimation and mode coverage.

## Score and Decision

Calibration:
- CMCD (PP1rudnxiW): Strong diffusion sampling paper with novel framework; avg ~7.2, accepted poster.
- Improved sampling via learned diffusions (h4pNROsO06): Novel framework for sampling via diffusions with alternative divergence; scores 6,6,6,8 avg ~6.5, accepted poster.
- DGFS (OIsahq1UYC): Novel diffusion sampling with partial trajectory optimization; scores 8,8,8,6 avg ~7.5, accepted poster.
- Structured Diffusion with GMM Prior (ZqM9mZkrRB): GMM prior for diffusion (generative modeling, not sampling); scores 3,5,5,5 avg ~4.5, rejected — this is the most directly comparable paper, but it was much weaker (poor motivation, limited theory, algorithm equivalent to running per-cluster diffusion models).
- GM-DDPM (xi4qWLNbhs): GMM noise for DDPMs (generative modeling); scores 5,5,5,3 avg ~4.5, withdrawn/rejected — also related but for generative modeling rather than sampling.

This paper is substantially stronger than the rejected GMM-prior diffusion papers: it has a sound theoretical grounding (Proposition 1), a general framework applicable to multiple methods, strong empirical results on real-world tasks, and a novel IMR strategy. It is somewhat below CMCD and DGFS in terms of depth of theoretical contribution, but offers a practical, broadly applicable improvement. The main issues are (1) overclaiming about C3, (2) the ELBO-coverage tension, and (3) the lack of validation for the learned-prior construction. These are real but not fatal weaknesses—the core contribution of learnable GMPs improving diffusion samplers is solidly demonstrated across multiple methods and benchmarks.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>