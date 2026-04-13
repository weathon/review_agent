=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
## Summary

PIN proposes using Prolate Spheroidal Wave Functions (PSWFs) as activation functions in Implicit Neural Representations, leveraging their optimal space-frequency energy concentration property. The authors argue that PSWFs' simultaneous localization in both spatial and frequency domains addresses limitations of existing INR activations (SIREN, WIRE, GAUSS) that struggle with balancing fine detail representation and smooth regions, and with generalizing to unseen coordinates in sparse reconstruction tasks. Experiments span image representation, 3D occupancy fields, image inpainting, and neural radiance fields.

## Strengths

- **Grounded theoretical motivation**: The use of PSWFs for optimal joint space-frequency concentration is well-founded in classical signal processing theory (Slepian-Pollak), providing a principled alternative to heuristically-chosen activation functions like sinusoids or Gabor wavelets.
- **Consistent image representation improvements**: On the 24-image Kodak dataset (Figure 2), PIN achieves 36.00 dB PSNR compared to WIRE's 31.81 dB and SIREN's 33.10 dB, demonstrating substantial and consistent improvements across a reasonably-sized benchmark.
- **Broad task coverage**: The paper evaluates PIN across multiple relevant tasks—image representation, 3D occupancy fields, image inpainting, and novel view synthesis—demonstrating the proposed activation's versatility beyond single-task optimization.

## Weaknesses

- **Contradiction between claims and data in inpainting experiments**: The abstract states PIN "significantly outperforms existing methods in various vision tasks that require INR generalization, including image inpainting," and Section 7.4 claims "PIN is the only architecture that maintains the highest PSNR value in both instances." However, Figure 5's table shows WIRE achieving 25.56 dB PSNR versus PIN's 23.18 dB on one inpainting experiment. The paper never clarifies which experimental protocol (70% random sampling vs. text mask) the reported numbers correspond to, creating ambiguity and an apparent contradiction between textual claims and presented data. This discrepancy must be resolved for the empirical claims to be credible.

- **Missing computational efficiency analysis**: PSWFs have no closed-form expression and require numerical approximation via Legendre polynomial expansion. The paper provides no discussion of training time, inference cost, or memory overhead compared to SIREN (sinusoidal activations) or WIRE (Gabor wavelets). For INRs—where efficiency is a practical concern—this omission prevents readers from assessing the trade-off between performance gains and computational cost.

- **No ablations on critical PSWF-specific design choices**: The paper uses only order-0 PSWFs throughout all experiments without justification or exploration of higher orders. The bandwidth parameter c, which fundamentally determines what "optimal" concentration means for PSWFs, is never discussed in the main text. Additionally, there is no ablation comparing the proposed adaptive formulation $\tilde{\psi}(x) = T\psi(wx) + b$ against fixed-parameter PSWF to validate that this adaptive mechanism is responsible for performance gains.

- **Limited NeRF evaluation**: The novel view synthesis experiment evaluates only the "drums" scene from a single dataset, comparing against SIREN, WIRE, and GAUSS but omitting modern NeRF baselines such as Instant-NGP, TensoRF, or Mip-NeRF. A PSNR improvement of 0.49 dB over GAUSS on one scene is insufficient to establish robust advantages in this domain.

- **Theoretical gap between motivation and claims**: The paper argues that better space-frequency concentration in activations leads to better generalization from partial observations (inpainting), but provides no theoretical justification for this causal link. Theoretical analysis (Theorem 1) shows that PIN outputs can be expressed as polynomials of the first-layer activations, but this result applies to any polynomial-approximable activation and does not uniquely characterize PSWFs or explain their empirical advantages.

## Nice-to-Haves

- Comparison against modern NeRF architectures (Instant-NGP, TensoRF) to establish relevance beyond vanilla INR baselines
- Ablation study on PSWF order (why only order 0?) and bandwidth parameter c sensitivity
- Larger-scale NeRF evaluation across multiple scenes from standard benchmarks like Mip-NeRF 360
- Wall-clock training time comparison to quantify the practical cost of PSWF numerical evaluation

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Theorem 1 is mathematically trivial"**: While the theorem is not groundbreaking, it does characterize the expressivity of the INR architecture. However, the weakness about the theorem not uniquely advantaging PSWFs is retained above.

- **"SSIM contradiction invalidates the paper"**: Figure 3 shows PIN achieving higher PSNR (28.10 dB) than SIREN/GAUSS (26.18 dB) but lower SSIM (0.749 vs 0.862). This is an interesting observation that merits discussion but does not invalidate the results—PSNR and SSIM measure different aspects of reconstruction quality, and the paper does not claim uniform superiority on all metrics.

- **"Polynomials are not band-limited"**: The critic's claim that polynomial approximation undermines the band-limited proof mischaracterizes the paper's argument. PSWFs themselves are band-limited; the polynomial approximation is for computational purposes, not theoretical replacement.

- **"Missing modern NeRF baselines is a fatal flaw"**: This is a valid criticism for NeRF-specific contributions, but retained above as a weakness rather than rejection criterion since the paper's primary contribution is the activation function itself, not a NeRF architecture.

## Novel Insights

The paper identifies a genuine phenomenon: existing space-frequency compact INR activations (Gabor/Gaussian) tend to focus on fine details at the expense of smooth regions, introducing noise-like artifacts. PSWFs' provably optimal energy concentration within finite bandwidth constraints offers a principled solution to this trade-off. However, the empirical evidence for this theoretical advantage is undermined by the inpainting data contradictions and limited NeRF evaluation, leaving the core claim plausible but not convincingly demonstrated.

## Suggestions

- **Correct the inpainting section immediately**: Audit Figure 5's table against the experimental protocols (70% random sampling vs. text mask). If the reported numbers show WIRE outperforming PIN, revise the text to accurately describe the results rather than claiming uniform superiority. If there are two experiments with different winners, report both sets of numbers clearly.

- **Add computational efficiency metrics**: Report training time per 1000 iterations and inference time per image for PIN vs. SIREN, WIRE, and GAUSS on a standardized benchmark. This is essential for practical adoption.

- **Provide PSWF implementation details**: State the number of Legendre polynomial terms used for PSWF approximation in the main text, and discuss any numerical stability considerations.

- **Clarify the adaptive parameter scope**: Specify whether T, w, b are learned per-layer, per-neuron, or globally, and report their learned values or distributions across layers to demonstrate that meaningful adaptation occurs.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
