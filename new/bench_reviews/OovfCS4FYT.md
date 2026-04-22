## Summary

The paper proposes a computationally efficient formulation of divisive normalization (CH DivNorm) for CNNs, using non-overlapping neuronal neighborhoods with a squared numerator and two learnable per-neighborhood parameters (γ_p, σ_p). Inspired by Carandini & Heeger's canonical model, CH DivNorm achieves comparable or better accuracy than Miller et al.'s prior formulation at roughly half the runtime overhead on ImageNet/AlexNet, and can replace ReLU in shallow networks while improving categorization performance. The paper also claims that divisive normalization causes the emergence of learned competition between orientation-selective and color-opponent cell types.

## Strengths

- **Real and substantial computational efficiency gains over prior work**: Table 2 shows CH DivNorm + ReLU runs at 1280 sec/epoch vs. Miller et al.'s 2800 sec/epoch on AlexNet/ImageNet, while achieving the same 61.3% Top-1 accuracy and using less GPU memory (1.9G vs. 2.9G). This is a genuine practical improvement.
- **DN can replace ReLU in shallow networks with performance gains**: Table 1 provides striking evidence — on a 2-layer CNN/CIFAR-10, CH DivNorm + BN achieves 73.07% vs. 69.50% for ReLU + BN; on AlexNet/CIFAR-100, 62.7% vs. 57.9%. The squaring operation in DN turning all responses positive appears sufficient as an activation in these settings.
- **Clean, modular formulation**: Eq. 2 requires only two learnable parameters per neighborhood per layer, making it trivial to insert into existing architectures. The design closely follows the canonical Carandini & Heeger model.
- **Improved robustness to pixel-level image corruptions**: Figure 4 shows statistically significant robustness improvements across most ImageNet-C corruption types for AlexNet, particularly for pixel-wise variations (brightness, saturation, gaussian noise), consistent with DN's role in normalizing within local receptive fields.
- **Honest discussion of limitations**: Section 5 explicitly acknowledges the lack of quantitative filter similarity metrics, limited architectural scope, and the failure of DN-without-ReLU at VGG-16 depth.

## Weaknesses

### Fatal
None.

### Major

- **The "emergent competition" claim is potentially confounded by the multivariate Gaussian initialization**: Section 3.4 describes a multivariate Gaussian initialization that "promotes greater similarity within each neighborhood while ensuring more diversity between neighborhoods." This initialization *by construction* makes filters within a neighborhood similar — the very pattern presented as "emergent" in Figure 2. The paper never specifies which initialization produced Figure 2, and there is no experiment comparing filter structure under Kaiming vs. Gaussian initialization. Section 4.3 states "this behavior emerges from the model," but without disentangling initialization from learning, the claim that DN *causes* emergent competition is unsupported. The paper's own Discussion (Section 5) acknowledges lacking "a robust metric to quantify this similarity," but this does not excuse the absence of the basic control experiment (Kaiming-init + DN vs. Gaussian-init + DN filter comparison).

- **All experimental results are on outdated architectures; the core utility claim is limited**: The paper tests only AlexNet, VGG-16, and a 2-layer CNN. VGG-16 with CH DivNorm (no ReLU) fails to converge (Table 3), which directly limits the headline claim that "divisive normalization eliminates the need for a non-linear activation function like ReLU" (Abstract). No modern architecture (ResNet, EfficientNet, ViT) is tested despite being standard since 2016. For a method paper claiming general applicability, this gap is significant — the DN-as-activation claim is restricted to shallow networks, and the DN-as-improvement claim is only demonstrated on two architectures, one of which (AlexNet) is rarely used in practice.

### Minor

- **No quantitative analysis of filter similarity**: The paper's most distinctive neuroscience claim — emergence of competition between orientation-selective and color-opponent cell types — rests entirely on visual inspection of Figure 2. A within-neighborhood vs. cross-neighborhood similarity metric (e.g., average pairwise cosine similarity) with statistical testing would substantially strengthen this claim. The paper acknowledges this gap (Section 5) but presents the claim as a central result regardless.

- **The "DN replaces ReLU" claim requires more nuance**: The paper shows DN + BN outperforms ReLU + BN in shallow networks (Table 1), but the comparison conflates the effect of replacing ReLU with DN. The incremental benefit of *adding* DN to ReLU is modest (~1% on CIFAR-10, row 1 vs. row 2 in Table 1). The failure at VGG-16 depth without ReLU (Table 3) significantly limits the scope, and no analysis of *why* DN fails at depth is provided beyond a one-sentence speculation about "the depth of the model" (Section 4.2).

- **The asymptotic complexity comparison is misleadingly framed**: Section 3.3 claims O(N²) worst-case for Miller et al. vs. O(N) for CH DivNorm, but the worst case (λ = N/8) is not representative — λ is a learned parameter that remains small in practice. The practical efficiency advantage is real and well-demonstrated (Table 2), so the asymptotic framing adds rhetorical weight that the actual numbers don't fully support.

- **Suspiciously small standard deviations in Table 1**: Rows 2 and 4 show 70.53 ± 0.00 and 73.07 ± 0.01 over 10 random seeds. Standard deviations of essentially zero across different seeds are extremely unlikely in floating-point arithmetic and suggest either a reporting artifact or an issue with seed variability. This deserves clarification.

### Trivial
None.

## Nice-to-Haves

- Testing on at least one residual architecture (e.g., ResNet-18 on ImageNet) to establish scalability beyond 2012-era architectures — especially important given the VGG-16 convergence failure.
- Ablation of multivariate Gaussian vs. Kaiming initialization with quantitative filter similarity metrics to disentangle the "emergent competition" claim from initialization confounds.
- Analysis of why DN without ReLU fails at depth (gradient flow analysis, loss landscape investigation, or systematic hyperparameter exploration).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Veerabadran et al. (2022) not sufficiently acknowledged"**: The harsh reviewer claims the paper doesn't adequately acknowledge Veerabadran et al.'s DivNormEI. However, the paper explicitly discusses DivNormEI at lines 80-83: "Veerabadran et al. (2022) introduced DivNormEI, which performs divisive normalization within the spatial neighborhood of each channel and applies lateral inhibition and excitation by weighted sum across channels, demonstrating improved performance in large-scale object recognition tasks." This is a fair acknowledgment; the claim that it's insufficient is subjective.

- **"Miller et al.'s sliding window can be implemented as O(N) convolution"**: This is speculative and not substantiated. The paper's practical runtime comparison (Table 2) is the more relevant evidence and already supports the efficiency claim.

- **"Interaction between squaring operation and batch normalization not analyzed"**: This is a fair observation but not a weakness — it's a nice-to-have analysis that goes beyond the paper's stated scope. The paper explains why BN helps after DN (Section 3.5).

- **"Figure 3 sharpened edge detection claim is qualitative"**: While true, this is presented as a supporting observation, not a central claim. The paper does not overstate this finding.

- **"Robustness only tested on AlexNet"**: This is already captured under the broader "limited architectures" concern. Testing robustness on more architectures would strengthen but not invalidate the current findings.

- **Strength claim "Scalability to deeper architectures" from the Strength Finder**: CH DivNorm does work on VGG-16 *with* ReLU, but the "scalability" framing is misleading when the more interesting claim (DN without ReLU) *fails* at this depth. This strength is partially valid but overstates the case.

- **Strength claim "Neighborhood-specific parameters capture heterogeneous dynamics"**: Table 2 shows 61.3% vs. 61.2% accuracy — a 0.1% difference that is within noise. This does not meaningfully demonstrate the value of per-neighborhood parameterization.

## Novel Insights

The observation that CH DivNorm's squaring operation — which makes both excitatory and inhibitory signals contribute to the numerator — might be more informative than half-wave rectification for shallow networks is genuinely interesting. This suggests that the ReLU-induced information loss from discarding negative responses is more harmful than the redundancy that DN's local competition removes, at least in low-depth regimes. The depth-dependence of this trade-off (it reverses at VGG-16 depth) points to a fundamental but unexplored interaction between DN's normalization dynamics and gradient flow in deep networks that could explain why biological circuits combine DN with rectification rather than replacing it.

## Suggestions

- Add a single experiment comparing Kaiming vs. Gaussian initialization for Figure 2's filter visualization, with a quantitative metric (e.g., within-neighborhood vs. cross-neighborhood cosine similarity). This would directly address the most damaging confound.
- In the Abstract, qualify the "eliminates the need for ReLU" claim with "in shallow networks" more prominently, and explicitly state the depth limitation observed in VGG-16.
- Report whether Table 1's near-zero standard deviations reflect rounding or genuinely deterministic behavior across seeds; if the former, report more decimal places.

## Evaluation

**Originality**: Moderate. The CH DivNorm formulation is a straightforward adaptation of Carandini & Heeger's model to non-overlapping channel neighborhoods. The computational efficiency advantage over Miller et al. is meaningful but incremental. The color-opponency emergence claim is novel but not convincingly established.

**Importance of research question**: High. Understanding how divisive normalization interacts with deep learning is important for both neuroscience and ML. Bridging canonical neural computations with practical CNN components is a valuable direction.

**Claims well supported**: Partially. The efficiency and accuracy claims are well-supported (Tables 1-3). The "emergent competition" claim is potentially confounded. The "DN replaces ReLU" claim is supported only for shallow networks and undermined by VGG-16 failure.

**Soundness of experiments**: Adequate for the efficiency and accuracy claims, insufficient for the neuroscience claims. Lack of Kaiming vs. Gaussian ablation for filter analysis is a significant gap. Standard deviations in Table 1 need clarification.

**Clarity of writing**: Good. The paper is well-organized, the formulation is clean, and limitations are honestly discussed.

**Value to research community**: Moderate. The efficient DN layer is a practical contribution, but limited to outdated architectures. The neuroscience claims, if validated, would be more impactful.

## Calibration

Anchors used for scoring:

| Paper | Score | Comparison |
|-------|-------|-----------|
| TopoNets (THqWPzL00e) | 7.50 | Brain-like topography tested on ResNet-18/50, ViT, GPT-Neo with modern architectures and strong neuroscience validation. Clearly above the current paper. |
| Bio-inspired local learning (CLE09ESvul) | 7.50 | Novel framework with strong empirical validation. Clearly above. |
| Higher-order convolutions (dPbJb9XdI1) | 4.50 | Bio-inspired with limited small datasets (MNIST, CIFAR). Current paper has ImageNet results, slightly stronger. |
| DISTA (mjDROBU93g) | 4.50 | Spiking transformer with intrinsic plasticity, CIFAR only. Current paper has larger-scale evaluation. |
| Neural robustness/D6Htk1rwkK | 4.25 | Neuroscience-inspired robustness analysis, limited model scope. Similar limitations. |
| Retina-inspired log-polar CNN (uYuoqHxtAW) | 2.50 | Trivial robustness gains, unfair comparisons, limited evaluation. Current paper is clearly better. |
| Brain-inspired body/background (epFk8e470p) | 1.67 | Extremely weak experiments. Current paper is clearly better. |

The paper sits in the medium band, comparable to the bio-inspired papers scoring 4.25-4.75 that also have genuine ideas but limited architectural validation and partially unsupported claims. The ImageNet evaluation pushes it slightly above some of these, but the confounded neuroscience claim and lack of modern architecture testing prevent it from reaching the 5.5+ range where papers with solid, well-validated contributions sit. The paper is marginally better than the ~4.5 anchors due to ImageNet-scale results, but the confounded core neuroscience claim is a meaningful weakness.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>