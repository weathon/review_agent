## Summary

DeeperForward proposes a redesign of the goodness function in the Forward-Forward (FF) training framework, replacing squared-sum goodness and vector-length normalization with mean goodness and LayerNorm/GroupNorm. The central claim is that this change addresses feature-scaling and deactivated-neuron pathologies that prevent FF-style local learning from working in deeper architectures. The method is demonstrated on VGG-like and ResNet-like CNNs of up to 17 layers, substantially outperforming prior FF-based baselines on CIFAR-10 (86.22% vs. 75.28% for CwComp at comparable depth), and is accompanied by a pipeline model-parallel strategy enabled by local updates.

---

## Strengths

- **Concrete and isolated ablation evidence for mean goodness.** Table 4 shows a 6.84% CIFAR-10 accuracy improvement (79.38% → 86.22%) from the use of mean goodness over squared goodness within the same 17-layer ResNet architecture — a specific, non-trivial delta that substantiates the core claim.

- **Actual depth scaling milestone within FF literature.** Prior layer-wise FF methods are universally shallow (2–4 layers). This paper is one of the first to report competitive training on 14- and 17-layer CNNs without block-wise backprop, which is a meaningful step for the local-learning subfield even if absolute accuracy still trails BP.

- **Clear mechanistic argument for decoupling.** The mean-goodness design has a coherent internal logic: LayerNorm fixes mean/std of the output to zero/one, making the pre-normalization mean a natural goodness statistic that is "erased" by normalization before being passed to the next layer. This is more principled than CwComp's batch normalization, which does not zero out the cross-layer signal, leading to observed overfitting in deeper CwComp reproductions (75.28% at 14 layers).

- **Honest benchmarking.** The authors reproduce CwComp on their own 14-layer CNN under matched conditions, directly exposing the overfitting issue (75.28%) rather than comparing against the original 4-layer results. They also openly acknowledge the gap with BP and the worsening under data augmentation.

---

## Weaknesses

### Fatal
None that outright invalidate the paper's contribution.

### Major

- **Table 4 appears to contain a formatting error that makes the ablation ambiguous.** Two rows show identical checkmarks (MEAN ✓, SIP ✓, RESIDUAL ✓) but yield 79.38% and 86.22% respectively. Since the text states "mean goodness achieves a substantial performance increase of 6.84%," the row at 79.38% must use squared goodness — meaning the MEAN column is mislabeled (should be ✗) for that row. This formatting error makes the ablation table misread at face value and needs to be corrected.

- **Ablation conflates LayerNorm/GroupNorm with mean goodness, preventing attribution.** The "MEAN" checkmark toggles both the goodness statistic and the normalization scheme simultaneously (mean goodness + LayerNorm vs. squared goodness + vector-length norm). A reviewer cannot determine whether the 6.84% gain comes from the goodness statistic, the normalization choice, or their interaction. A row isolating LayerNorm with squared goodness, or mean goodness with vector-length normalization, is needed to validate the paper's core mechanistic claim.

- **The mathematical justification for solving deactivated neurons (Eq. 6) is incomplete and potentially incorrect.** The paper states ΔW_ij = Cx_i (∂L/∂g) and claims "C is a constant," arguing that this allows updates even when y_j = 0. However, if g = Σ_i y_i = Σ_i ReLU(Wx)_i, then ∂g/∂W_ij = x_i · 𝟙{y_j > 0}, which is still zero for a deactivated ReLU unit — identical to the squared goodness case. For the claim to hold, C must be something other than a constant (e.g., a straight-through estimator). The paper does not clarify this, and Appendix I (discussing dead neurons) is not visible to reviewers in the main paper. The empirical improvement is real (as evidenced by Table 4), but the theoretical justification in Section 3.1 is misleading or incomplete as written.

- **Depth plateau beyond 17 layers is acknowledged but undiagnosed.** The paper notes in Section 4.5 that 33- and 100-layer models yield no significant improvement (Appendix D), yet provides no analysis of why. A paper titled "DeeperForward" that cannot scale past 17 layers without degradation has an obligation to characterize this bottleneck — is it gradient vanishing through the residual shortcut? Layer-wise prediction pressure? Goodness signal collapse? Without this, "enhancing FF for deep networks" is only established up to a narrow depth range with no guidance on further scaling.

- **CIFAR-100 results reveal a fundamental architectural bottleneck.** The ResNet-ours achieves 53.09% vs. 58.01% for BP; recovering to 60.28% requires tripling channels (ResNet-CHx3). The paper acknowledges this in its limitations section, attributing it to "inadequate allocation of neurons to each class." This is not a minor caveat: the CW-Conv design ties channel count directly to class count, meaning the method as presented does not generalize gracefully to realistic multi-class problems without aggressive width inflation. This is a fundamental constraint on applicability.

### Minor

- **Eq. 5 notational inconsistency.** In Eq. 5, g = Σ_i y_i is defined as a sum, and then z = (y − g)/√(σ²+ε) subtracts a scalar sum from each element, which is not standard centering. The CNN version (Eq. 7) correctly uses the mean (1/(HWC) Σ h). The discrepancy between the scalar formulation in Section 3.1 and the convolutional formulation in Section 3.2 should be resolved with consistent notation.

- **SIP module provides negligible improvement and introduces methodological ambiguity.** Table 3 shows gains of 0.06%, −0.01%, and +0.15% after SIP, which are practically insignificant. Moreover, the paper reports "best trial results" (not mean ± std) for SIP comparisons, a weaker standard than used elsewhere. The biological analogy to "synaptic pruning" is strained — SIP is essentially validation-set model selection over O(L²) layer-combination candidates. Whether the selected (Start, End) is stable across seeds is not reported.

- **No data augmentation on main benchmarks limits practical relevance.** The paper acknowledges in Appendix C that the augmentation gap with BP is substantial, but this is not quantified in the main paper. If the method cannot benefit from standard augmentation — a cornerstone of modern training — this is a practical limitation that deserves more prominent discussion, not appendix treatment.

### Tiny

- **Figure 1 "Ours" panel is misleading.** The diagram shows a single backward error signal L_e flowing through the network, visually resembling global backprop. The actual training uses independent local losses at each layer (Eq. 13), which is quite different. The figure should be updated to show per-layer local losses.

- **Training objective framing.** The method uses local per-layer cross-entropy (Eq. 13), not classical FF positive/negative contrastive passes. The paper is largely transparent about this (Figure 1e, Section 3.3) but vacillates between calling itself "FF" and "FF-inspired local learning." Being explicit that the optimization is greedy local supervised training (with FF-inspired goodness as intermediate statistic) would clarify the contribution's position relative to the broader local-learning literature (DTP, decoupled greedy learning, etc.).

- **Parallelism evidence is limited.** Table 5 covers only one dataset and architecture with no throughput-vs-depth scaling or comparison to pipeline-parallel BP baselines. The 4-GPU speedup (2.48× vs. DDP's 2.61×) is acknowledged but not analyzed. This section establishes feasibility rather than strong systems evidence.

---

## Nice-to-Haves

- **Ablation with standard data augmentation.** Showing whether the FF local-learning gap with BP under augmentation is specific to goodness design or is fundamental to local training would clarify the method's long-term trajectory.

- **Neuron activation rate visualization.** A heatmap showing dead-neuron rates across layers for squared vs. mean goodness (as mentioned in Appendix I) would directly validate the paper's central mechanistic motivation and should be promoted to the main paper.

- **Cross-layer goodness correlation analysis.** A quantitative measure of "goodness leakage" (correlation of goodness values between adjacent layers) for CwComp vs. DeeperForward would substantiate the decoupling claim empirically rather than arguing it by construction.

- **Scaling study for CIFAR-100.** A systematic analysis varying channels-per-class would determine whether the CIFAR-100 gap is an optimization failure or a capacity-allocation problem, informing future design of more class-count-agnostic architectures.

- **Deeper layer diagnostics.** Gradient norms, feature rank, and goodness trajectories at 17 vs. 33 vs. 100 layers would explain the depth plateau and provide the community with concrete failure modes to address.

---

## Removed Points

*These points are flagged as removed — treat them with caution.*

- **"The method is not really FF" as a core weakness.** The harsh critic devotes substantial space to arguing the method is "local supervised training, not FF." However, the paper itself is explicit about this departure: Figure 1(e) clearly labels the method as using a single forward pass with local error signal, distinct from FF's positive/negative passes. The paper positions itself as extending/inspiring from FF, not as a faithful reimplementation. The broader FF literature already includes many departures from Hinton's original formulation (CwComp, Trifecta, etc.). This is a framing preference, not a substantive flaw.

- **"Bio-plausibility is overstated."** The paper's primary motivation is removing update locking and enabling local parallelism, not strong neuroscientific realism. The harsh critic applies a more rigorous bio-plausibility standard than the paper claims to meet. The paper does acknowledge gradient-based optimization remains; removing this criticism is appropriate.

- **"Missing related works."** Per review policy, references are not independently verifiable and claims of missing citations are excluded.

- **"Comparison with DTP/recLRA is across different architectures."** The comparison in Table 1 is context-setting across the broader local-learning landscape, not a primary claims table. The paper's main comparisons are within FF-type methods at matched depths (reproducing CwComp on the same 14-layer CNN). Cross-architecture comparisons in a survey table are standard practice.

- **"Memory comparison is unfair — should compare against activation checkpointing."** The memory advantage of local updates over standard BP is stated as a feasibility result, not a claim of superiority over all engineering baselines. Activation checkpointing is an optimization that independently applies to local methods too. This is an asymmetry favoring the baseline, and the paper does not claim to beat all BP memory-saving techniques.

- **"Benchmark suite is too small for ICLR."** CIFAR-10/MNIST/FMNIST are the dominant benchmarks in the FF/local-learning literature precisely because the methods are not yet competitive on ImageNet. Criticizing paper X for not solving what paper X acknowledges as future work is scope creep. The CIFAR-100 experiment is included. Demanding TinyImageNet/ImageNet without the architecture modifications to support it is not a reasonable bar within this subfield.

- **"The architecture's 'no FC layer needed' claim is unsubstantiated."** The paper says "experiments in Appendix E reveal that CW-Conv outperforms FC" and explicitly defers to the appendix. This is a design choice with supporting evidence — calling the claim "too strong" for having its evidence in an appendix is a formatting nitpick.

---

## Novel Insights

The most valuable insight that emerges beyond standard benchmarking is the *coupling between normalization design and goodness signal stability across depth*. By showing that batch normalization in CwComp leaks goodness (empirically evidenced by overfitting at depth) while GroupNorm after mean-based goodness extraction prevents this, the paper implicitly suggests that the choice of normalization for FF-type training is not merely a performance tuning knob but a structural choice that determines whether deeper layers receive meaningful learning signal or inherit already-resolved information. This framing — that goodness decoupling is an architectural requirement for depth, not an optimization convenience — is a useful organizing principle that the local-learning community could build on when designing future deep forward-only architectures. The failure of the method beyond 17 layers and its class-count scaling bottleneck also implicitly point to a deeper open problem: local classification objectives may impose an information bottleneck that prevents rich intermediate representations from emerging, suggesting that future local-learning methods for deeper networks may need explicit mechanisms to promote representation diversity rather than per-layer discriminability.

---

## Suggestions

1. **Fix the Table 4 formatting error.** Correct the duplicate checkmark row so the ablation table unambiguously shows one row with and one row without mean goodness.

2. **Add a standalone ablation row for LayerNorm + squared goodness.** This is the minimum experiment needed to separate the normalization effect from the goodness statistic effect and validate the paper's core mechanistic claim.

3. **Clarify Eq. 6 and the deactivated-neuron gradient argument.** Either derive ∂g/∂W_ij rigorously (addressing the ReLU indicator issue) or explicitly state that the method uses a straight-through estimator / ReLU relaxation. Move the Appendix I dead-neuron analysis (activation rate comparison) into the main paper, as it directly supports the central motivation.

4. **Standardize SIP reporting.** Report mean ± std (over five trials) in Table 3 instead of best-trial results, consistent with the rest of the paper.

5. **Provide a depth diagnostic.** Add a brief analysis in the main paper explaining why 33/100-layer models do not improve over 17 layers — gradient behavior, goodness signal collapse, or feature saturation — rather than deferring entirely to Appendix D.

6. **Quantify the augmentation gap.** Include at minimum one row in Table 1 (or a new table) showing DeeperForward's accuracy under standard CIFAR-10 augmentation alongside the BP baseline, so readers can assess the practical cost of the representation-learning gap.