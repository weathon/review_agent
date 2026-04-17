Now let me carefully verify specific claims against the paper text before finalizing my review.

## Summary

LS-Merge proposes merging large language models in a learned latent space rather than directly in weight space. A Transformer-based VAE encodes model weights into a shared latent representation—using a two-stage training curriculum and structured chunking to handle heavy-tailed weight distributions—where interpolation and merging operations are performed before decoding back to parameters. For heterogeneous (cross-architecture) merging, the paper introduces dimensionality-matching projections and optimal-transport-based alignment of latent distributions. The method demonstrates strong improvements on homogeneous LoRA expert merging and initial but limited evidence for cross-architecture merging.

## Strengths

- **Novel and well-motivated formulation.** Shifting merging from weight space to a learned latent manifold is a genuine conceptual advance that directly addresses a known limitation (architectural homogeneity) of prior merging methods. The weight statistics analysis (Table 1) showing high kurtosis and heavy-tailed distributions in LLM weights is a concrete, actionable insight that motivates the VAE design choices.

- **Strong homogeneous merging results.** Table 3 is the paper's strongest empirical contribution: LS-Merge (lerp and soup variants) clearly and consistently outperforms Uniform Soup, Greedy Soup, SLERP, DARE-TIES, and Data Merge across 7 benchmarks on Gemma-7B-it LoRA expert fusion. These are fair comparisons against relevant recent baselines.

- **Compelling PCA vs. VAE ablation.** Table 8 provides direct evidence that linear compression (PCA) collapses functional performance even at mild compression (r=1.6), while the non-linear VAE maintains near-original accuracy even at r=4.0. This supports the claim that the weight manifold is fundamentally non-linear and validates the choice of a non-linear encoder.

- **Principled alignment mechanism.** The use of optimal transport to align latent distributions from heterogeneous architectures is a theoretically grounded approach to the manifold registration problem, even if the current empirical validation is limited.

## Weaknesses

### Major

- **Evidence for heterogeneous (cross-architecture) merging is thin and oversold.** The paper's central differentiating claim—"architecture-agnostic" merging enabling "robust cross-scale and cross-family model merging for the first time"—is supported by only two experiments: (1) intra-family Gemma-4B→Gemma-1B, where results are reported only via a figure reference without aggregate metrics, and (2) cross-family LLaMA-3.2-1B→Gemma-3-1B at a single λ=0.1 on three benchmarks (Table 5). Moreover, "OT only" in Table 5 *degrades* performance substantially (Winogrande 56.83→51.13, ARC-C 42.78→34.25), and recovery requires adding only 10% of the source model. This pattern suggests the gains may arise from very light regularization/injection rather than robust knowledge transfer across architectures. The language about "architecture-agnostic" and "for the first time" far exceeds what the data support, which show a single-point improvement in one direction of one cross-family pair.

- **Self-merging lacks rigorous baselines to isolate the mechanism.** Table 2 compares LS-Merge against the original model and a single VAE reconstruction, but does not disentangle whether improvements come from the claimed latent-space exploration or from simpler explanations like averaging multiple noisy decodings or VAE-induced regularization. Without controlling for "decode k independent samples and average in weight space" or "pick the best of k samples," the 4% gain cannot be attributed to the latent merging mechanism specifically. This matters because the self-merging claim—that one can improve a model by encoding and re-sampling its own weights—is both striking and easy to confound.

### Minor

- **Tension between non-Gaussian weight motivation and Gaussian OT approximation.** Section 3.1 carefully documents that LLM weights are heavy-tailed and non-Gaussian (kurtosis up to ~15), motivating a non-linear VAE over simple linear methods. Yet Section 3.3 adopts a Gaussian approximation for the OT map (affine map using empirical mean and covariance). The paper does not analyze how well this Gaussian approximation fits the actual latent distributions, nor does it compare against a non-parametric OT solver. This is a conceptual inconsistency that is acknowledged but not resolved.

- **Computational cost and training data requirements are unreported.** The Transformer-VAE must be trained on collections of pretrained weight snapshots, but the paper does not specify how many checkpoints are needed, the training time/cost, or how this compares to the cost of weight-space merging itself. This makes it difficult to assess the practical viability of the framework, especially at scale.

- **The two-stage curriculum (deterministic AE → VAE) is claimed as crucial but not ablated.** Section 3.2 states this is important for stability on heavy-tailed weights, but no experiment compares it against standard single-stage β-VAE training. Without this ablation, it is unclear whether this design choice is essential or incidental.

- **OT alignment and dimensionality-mapping details remain under-specified.** Algorithm 1 references "algorithm 2" which does not appear in the paper text. The dimensionality-matching projection ratio r = ntN/(nsM) is stated formally but never instantiated concretely for the evaluated architectures. The level at which Gaussian statistics (per-layer? per-token?) are computed for the OT map is not made explicit.

### Trivial

- **Evaluation protocols vary across experiments** (custom code for Tables 2–3, lm-eval for Tables 4–5, 7–8). While the paper acknowledges this, it complicates cross-experiment comparison.

## Nice-to-Haves

- Ablation of VAE vs. deterministic autoencoder for *merging* (not just reconstruction), to determine whether the variational structure is essential or whether the encoder/decoder architecture and chunking scheme alone suffice.
- Expanded cross-architecture experiments with larger size gaps, different architectural families (e.g., Mistral), and reverse transfer directions.
- λ-sweep for cross-family merging (currently only λ=0.1 tested), and comparison against simple baselines like truncation + padding.
- Text generation quality evaluation (perplexity, human/LLM-judge) beyond multiple-choice benchmarks.
- Statistical significance tests or confidence intervals for results with small numerical differences.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released" or availability concerns about cited methods/benchmarks**: Several reviewers questioned whether certain referenced approaches could be independently verified. Per policy, if the paper cites it, it exists.

- **Formatting and notation inconsistencies**: References to Algorithm 2, minor notation inconsistencies between sections, and formatting artifacts are editorial issues, not substantive weaknesses.

- **Demand for missing related work citations**: Per policy, we cannot verify whether omitted references exist, so this is excluded.

- **Reproducibility concerns about hyperparameters**: Exact chunk size, positional encoding details, etc., are implementation specifics that can be released with code. The paper states they will open-source implementation.

- **Demand for comparison against fine-tuning/knowledge distillation in the cross-architecture setting**: These are fundamentally different approaches (require data and training) from training-free merging. The comparison scope is weight-space merging methods.

- **Claims that Table 7 contradicts the main experiments**: The table shows generalization to *unseen* models degrades at high compression, while the main experiments use models the VAE was trained on or with appropriate compression ratios. There is no direct contradiction.

- **Strict reproducibility standards (number of seeds, confidence intervals)** for all experiments: Single-run or few-run evaluation is the norm in large-scale LLM benchmarking, and the paper does report some standard deviations.

## Novel Insights

The PCA vs. VAE comparison (Table 8) reveals an important structural insight: even though LLM weight matrices exhibit approximate low-rank structure (as shown by the Eckart–Young analysis in Section 3.1), the *functional* weight manifold is fundamentally non-linear. PCA collapses to near-random accuracy even at mild compression (r=1.6), while preserving exactly as much dimensionality as the VAE does successfully at r=4.0. This implies that low-rank structure in individual weight matrices does not translate to a linear subspace structure in the joint parameter space—valid weight configurations occupy a curved manifold that requires non-linear encoding to navigate. This is a meaningful geometric insight for the weight-space learning community.

## Suggestions

1. **Temper the heterogeneous merging claims**: Replace "architecture-agnostic" and "robust cross-family" language with more precise descriptions like "initial evidence for cross-architecture transfer in limited settings" and "a principled alignment framework that shows promise in small-scale experiments."

2. **Add self-merging baselines**: Include "average k decoded weights in weight space" and "select best of k decoded samples" as controls. If LS-Merge still outperforms these, the latent-merging mechanism is genuinely contributing above and beyond noise averaging.

3. **Report concrete metrics for intra-family heterogeneous merging**: Fig. 4a is described qualitatively but no table of numbers is provided. Include the same benchmark suite as Table 5.

4. **Report VAE training cost**: Even approximate GPU-hours and the number of weight checkpoints used for training would significantly help assess practical deployment feasibility.

## Evaluation

**Originality**: The latent-space merging paradigm for LLMs is genuinely novel. The combination of VAE + OT alignment for cross-architecture merging is creative, though the individual components are well-established.

**Importance of research question**: Heterogeneous model merging is an important open problem. The self-merging question (can one improve a model by manipulating its own latent distribution?) is intriguing.

**Claim support**: The homogeneous LoRA merging claims are well-supported (Table 3). The heterogeneous and self-merging claims are oversold relative to the evidence. The non-linear manifold claim is well-supported (Table 8).

**Experimental soundness**: Solid for homogeneous settings; thin for heterogeneous settings; incomplete for self-merging mechanism isolation.

**Clarity**: Generally well-written, though the OT alignment and dimensionality-matching sections are ambiguous in places.

**Value to community**: The idea of operating in a learned weight manifold is valuable even if the current execution has limitations—it opens a clear research direction.

**Calibration**: Compared to model merging papers in similar venues: KnOTS (Accept Poster, avg ~6.3) had a clearer, narrower contribution with solid experiments; AdaMerging (Accept Poster, avg ~6.5) had strong empirical results on a focused problem; CCA Merge (Reject, avg ~4.5) had theoretical gaps and limited validation; Foldable SuperNets (Reject, avg ~5.5) had limited practical scope. LS-Merge has a more novel core idea than CCA Merge and Foldable SuperNets, with genuinely compelling homogeneous results, but its headline claims are disproportionately strong relative to the evidence—particularly the heterogeneous merging story. This places it below the accepted merging papers but with real promise.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>