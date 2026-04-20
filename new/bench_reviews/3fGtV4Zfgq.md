Now I have a thorough understanding of the paper. Let me construct the consolidated review.

## Summary

This paper introduces three contributions for training, evaluating, and sampling from equilibrium Restricted Boltzmann Machines (RBMs) on highly structured, multimodal datasets: (1) Trajectory Annealing Importance Sampling (Tr-AIS), an online log-likelihood estimation method that exploits the training trajectory as an annealing path; (2) Parallel Trajectory Tempering (PTT), a sampling algorithm that exchanges configurations between models saved at different training stages rather than across temperatures; and (3) a low-rank RBM pretraining strategy with a trainable bias extension to mitigate critical slowdown in early training. The paper demonstrates clear empirical gains on multiple structured datasets (MNIST01, HGD, Ising 2D, Mickey) and honestly acknowledges limited benefits on less clustered data (CelebA).

## Strengths

- **Compelling physical diagnosis of PT failure on multimodal data.** The paper provides a concrete and visually compelling demonstration (Fig. 4A–B, left panels) that standard Parallel Tempering can completely forget minority clusters (the East Asian population in HGD disappears) while maintaining high swap acceptance rates (>0.8). This is explained via the first-order phase transition in temperature scaling, and contrasted with continuous transitions along the training trajectory—a principled motivation for the trajectory-based approach.

- **PTT substantially improves mode exploration on clustered datasets.** Figure 3 (column C) presents quantitative evidence that PTT achieves orders-of-magnitude more inter-mode "class exchanges" per unit computation than AGS, PT (with various configurations), and Stacked Tempering across MNIST01, HGD, and Ising 2D. On HGD, AGS barely produces any jumps over 10⁴ steps, while PTT shows hundreds—a dramatic empirical gap.

- **Tr-AIS provides accurate and low-cost LL estimation in controlled settings.** Figure 2 demonstrates on 20-hidden-unit RBMs (where exact LL is computable by enumeration) that online Tr-AIS achieves lower estimation error than temperature-based AIS with any number of temperatures tested (T=2 to 100), at negligible additional cost since "updating the partition function only involves calculating the energy difference between the new and old parameters" (Section 4.1). This is verified across 10 independent runs with narrow standard deviations.

- **PTT is theoretically sound with detailed balance.** Equation 3 specifies the Metropolis acceptance rule, and the paper argues (correctly) that this move satisfies detailed balance with the target equilibrium distribution, ensuring theoretical correctness of the sampling scheme.

- **Practical low-rank pretraining with bias extension.** Section 5 (Eq. 4–5) extends the Decelle & Furtlehner (2021a) low-rank RBM framework by adding a trainable bias direction u₀, which "effectively adds an additional direction at minimal cost" and is shown to be "crucial to obtain reliable low-rank RBMs for image data." Figure 4 (right panels C–E) confirms pretraining+PCD properly balances modes and achieves higher test log-likelihood across all five datasets.

- **Honest assessment of limitations on less structured data.** The paper clearly states that pretraining "offers little benefit" for less clustered datasets like CelebA, with supporting experiments in SI J, demonstrating intellectual honesty rather than overclaiming.

- **Code and dataset availability.** Code is provided at a public GitHub repository, enabling reproducibility.

## Weaknesses

### Fatal

None.

### Major

- **Limited experimental scale for the central claims.** The core quantitative validation of Tr-AIS (Fig. 2) is performed exclusively on 20-hidden-unit RBMs where exact LL is computable by enumeration. Similarly, the PTT vs. PT/Stacked Tempering comparisons in Fig. 3 use these small models. While the qualitative findings likely generalize, the headline claims of "reliable and computationally efficient log-likelihood estimates" and that PTT "outperforms previously optimized MCMC methods" are not demonstrated at scales where RBMs are typically deployed on real scientific data (e.g., genomics datasets with hundreds of hidden units). The paper does show results on five datasets, but all at the toy-scale model size.

- **Absence of compute-fair (wall-clock/FLOP) comparisons.** The computational comparison in Fig. 3 normalizes by $N_{\text{model}} \times \text{AGS steps}$, treating all AGS steps as equal cost regardless of model state. However, PTT evaluates multiple distinct parameter sets in parallel, which can have different computational characteristics than temperature-scaling a single model. Without wall-clock or FLOP-normalized comparisons—particularly when the paper is competing against established methods like PT and Stacked Tempering—the efficiency claims remain difficult to assess in practical terms. This is especially relevant since one of the paper's key selling points is computational efficiency.

### Minor

- **No ablation of PCA directions retained during low-rank pretraining.** The pretraining constrains the RBM coupling matrix to $d$ PCA directions (Eq. 4). Performance gains on pretraining+PCD over PCD alone could partly reflect that the low-rank model bypasses early critical slowdown by forcing the model into a specific low-dimensional subspace. Without ablating $d$ or comparing against unconstrained initializations, it is difficult to isolate whether the improvement comes from better initialization quality versus structural bias toward a faster-reachable linear basin. The paper notes this is most beneficial for "highly clustered" data, but does not quantify how performance degrades or saturates as $d$ varies.

- **Sensitivity of Tr-AIS and PTT to training hyperparameters is not characterized.** The paper states that "updating the partition function only involves calculating the energy difference" for Tr-AIS and that PTT exchange acceptance is maintained around 0.25, but does not analyze how these methods behave under different learning rate schedules, batch sizes, or gradient noise levels. If the methods are sensitive to these standard training choices, their practical utility on non-toy models may be constrained.

### Trivial

- **Limited to RBMs.** The PTT method is claimed to be "fully applicable to other energy-based models" (Section 4.2), but is only demonstrated on RBMs. The exchange criterion (Eq. 3) assumes a Hamiltonian structure that may require adaptation for more general EBMs, and this generalization is not discussed.

## Nice-to-Haves

- Including higher-dimensional distributional diagnostics (e.g., t-SNE/UMAP visualizations, or feature-space metrics beyond 2D PCA projections) would strengthen confidence that PTT samples the true equilibrium distribution and not just the PCA-projected structure.

- Formalizing an adaptive step-size or path-spacing mechanism for Tr-AIS (akin to the adaptive schedules used in standard AIS) could improve robustness but is not essential given the strong empirical results at fixed step sizes.

## Removed Points

These points are flagged to be removed—treat them with caution.

- **Harsh Critic point 1 (structural: online Tr-AIS estimator "mathematically ill-posed" for stochastic training):** The critic claims that "compounding stochastic bias and high variance" causes Tr-AIS to "drift systematically rather than converge," and that the paper "masks this by validating Tr-AIS only on a 20-hidden-unit model trained with likely deterministic, single-batch updates." This is a misreading. The paper explicitly states in Section 4.1: "Since updating the partition function only involves calculating the energy difference between the new and old parameters, this can be efficiently computed online during each parameter update, allowing for very small integration steps." The method works precisely because the integration steps are small. The critic's claim that the model uses "deterministic, single-batch updates" is unsupported—the paper describes PCD training with persistent chains in the standard way. The concern about stochasticity is a valid discussion point but does not constitute a "mathematically ill-posed" foundation, especially given the empirical accuracy demonstrated across 10 independent runs with narrow variance. → Moved here.

- **Harsh Critic point 2 (sampling efficiency claims "lack compute-normalized and high-dimensional validation"):** The concern about 2D PCA projections is partially valid (moved to Major/Minor above), but the critic's claim that counting "class exchanges" between PCA-defined clusters "does not verify convergence to the true high-dimensional equilibrium distribution" mischaracterizes what the experiment demonstrates. The paper uses PCA-defined clusters as a proxy for mode separation—a standard analysis technique for RBMs on structured data. The paper does not claim these metrics prove convergence in full dimensional space; rather, it shows dramatically improved inter-mode transitions relative to all baselines. → Partially moved to Major above; the rest is overreach.

- **Harsh Critic point 3 (low-rank pretraining "restrictive linear inductive bias"):** The critic argues this bias is problematic for genomics/neural data where "true generative manifolds are non-linear." This is a scope-creep criticism. The paper explicitly focuses on clustered datasets where the dominant structure is captured by leading PCA components (Fig. 1). The low-rank pretraining is presented as a fast initialization, not a replacement for full-rain training—the RBM continues to train with PCD after pretraining. The paper does not claim this captures all structure. → Moved to Minor above (ablation); the expressivity constraint concern is weakened here.

- **Harsh Critic claim that Tr-AIS "cannot be reliably deployed for standard-scale, mini-batch RBM training without fundamentally altering the optimizer":** This is an unsupported assertion. The paper uses standard PCD training throughout. The estimator works because the parameter changes per step are small (learning rate is small), making the accumulation of $\log \langle \exp(-\Delta \mathcal{H}) \rangle$ well-behaved. The critic is demanding an adaptive path-control mechanism that is not standard for AIS variants. → Removed as a nice-to-have above.

- **Harsh Critic claim that PTT "depends on a bipartite structure or easily factorizable Hamiltonian, which does not hold for general EBMs":** The paper claims PTT "can be fully applied to other energy-based models" as a statement of principle, not as an empirical contribution. The core method is designed for RBMs. Extending or not extending to general EBMs is not a flaw. → Removed.

## Novel Insights

The paper's key insight—that the progressive feature-learning dynamics of RBM training can be repurposed as a thermodynamic trajectory for both evaluation (Tr-AIS) and sampling (PTT)—is genuinely novel and well-motivated by the physics of continuous phase transitions. The diagnosis that temperature-based annealing encounters first-order phase transitions where modes disappear discontinuously, while the training trajectory encounters only continuous transitions where modes gradually separate, provides an elegant explanation for why trajectory-based methods should outperform temperature-based ones on clustered data. The empirical demonstration that PT can "forget" minority clusters despite high swap acceptance rates is a practically important observation for practitioners using PT on structured data.

## Suggestions

- Add a small table or figure reporting wall-clock times for PTT vs. PT vs. Stacked Tempering for equivalent number of class exchanges, to make the efficiency claims concrete and practically interpretable.
- Include an ablation on the number of retained PCA directions $d$ in the low-rank pretraining stage to show how pretraining quality varies and whether the benefits saturate or degrade.
- Consider reporting the number of saved models used by PTT across datasets to give readers a sense of the memory overhead of the method.

## Calibration and Scoring

I compared this paper against several anchors:

**High-scoring anchors (Accept/Poster, ~6-6-8 range):** The Stacked Tempering RBM paper (kXNJ48Hvw1) scored 6, 6, 8 with an Accept (Poster) decision. The EBM with cooperative diffusion paper (AyzkDpuqcl) scored 8, 6, 6, 6, 8 (Accept/Spotlight). These papers had strong novelty and empirical results on standard-scale models with clear baselines.

**Borderline anchors (~5-6 range):** Papers with good framing but limited experiments or only toy-scale validation (TgeVptDYAt, Uavy4DLrXR, rYhDcQudVI) were mostly rejected, scoring in the 3-5 range.

**Low-scoring anchors (~3):** Weak experimental validation and methodological gaps led to rejects (gS0XOu0JKs, iQHL76NqJT).

This paper sits between the high and borderline anchors. It is conceptually stronger than most borderline-rejected papers—its physics-inspired framing is genuinely novel and its empirical results are clear and compelling *for the model sizes tested*. However, the validation is limited to 20-hidden-unit RBMs, which is too small to conclusively demonstrate that the methods scale to practical scientific datasets (e.g., genomics). The absence of wall-clock comparisons is also notable. Compared to the Stacked Tempering RBM paper (scored 6, 6, 8) which also worked on MNIST and Ising models, this paper is comparable in scope but slightly weaker on demonstration breadth.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>