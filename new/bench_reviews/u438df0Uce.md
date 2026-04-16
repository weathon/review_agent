Now I have a good understanding of the paper and the landscape of similar work. Let me synthesize the final review.

## Summary

SpikeZIP proposes an ANN-to-SNN conversion framework that establishes theoretical model-level equivalence between a Quantized ANN (QANN) and its converted SNN under specific conditions (ST-BIF neurons, analog input/bias encoding, average pooling, BN fusion). It also introduces Paths-Ensemble Training (PET) to improve the accuracy-latency Pareto frontier, and Residual Connection Re-routing (RCR) for latency reduction. Experiments show state-of-the-art results on ImageNet (74.21% on ResNet-34 at 11 time-steps) and CIFAR-100.

## Strengths

1. **Comprehensive and strong experimental results**: The paper provides extensive comparisons across multiple architectures (VGG-16, ResNet-20/34), datasets (CIFAR-100, ImageNet), and methods (conversion-based and learning-based baselines), consistently achieving Pareto-frontier improvements. The reductions in required time-steps (43.75% for VGG-16, 33.3% for ResNet-34 on ImageNet) are meaningful practical gains.

2. **Theoretical grounding with precise conditions**: The paper defines model-level equivalence (Theorem 1) with clearly stated prerequisites, which is more rigorous than many prior conversion works that rely on approximate mappings. The explicit specification of conditions (ST-BIF neuron, even-release encoding, $V_{t=0}=0.5V_{thr}$, etc.) makes the framework reproducible and the claims verifiable.

3. **PET is a well-designed and empirically validated technique**: The multi-path quantization-aware training approach, inspired by slimmable networks, is carefully designed with parameter sharing schemes (Table 5) and thorough ablations. The idea of simultaneously optimizing for multiple quantization levels to improve low-latency SNN performance is practically effective.

4. **Thorough ablation studies**: Table 5 and Figure 6 systematically disentangle the contributions of PET, RCR, parameter sharing strategies, loss coefficients, and quantization levels, providing valuable insight for practitioners.

## Weaknesses

### Major:

1. **The "practical advantage of SNN over QANN" question is unanswered**: The core premise of ANN-to-SNN conversion research is that SNNs offer efficiency advantages on neuromorphic hardware. However, Theorem 1 establishes that the SNN is *mathematically equivalent* to the QANN—meaning that for inference, the SNN computes the exact same function as a low-bit quantized ANN. This raises the fundamental question (also raised in reviews of closely related QAC work: "if the quantized ANNs are equivalent to SNNs with soft reset, why don't we just use the quantized ANNs?"): **what is the practical advantage of deploying the SNN instead of just running the QANN?** The paper's SOP-based energy analysis (Figure 7) does not address this, as it compares SNN vs. ANN floating-point operations—not SNN vs. QANN integer inference. Memory access costs, which dominate energy consumption on many platforms, are also omitted.

2. **Equivalence only holds at equilibrium ($T_{eq}$), not at the practical low-$T$ regime where main claims are made**: The paper's headline results report accuracies at $T \ll T_{eq}$ (e.g., 73.92% at $T=9$ for VGG-16 on ImageNet). Footnote 4 explicitly states that "peak accuracy of SNN is not achieved at $T_{eq}$ but a time-step $T < T_{eq}$." This means the proven equivalence at equilibrium does not directly explain the strong low-$T$ performance—yet the paper rhetorically connects them. The theoretical framework and empirical operating regime are misaligned.

3. **Component novelty is incremental**: The ST-BIF neuron model is adopted from prior work (Li et al. 2022; Hu et al. 2023); the Q-ReLU training follows LSQ (Esser et al. 2020); PET is an adaptation of slimmable networks (Yu et al. 2018) to quantization levels; and RCR is acknowledged as similar to SEW-ResNet's topology. While the combination is effective, each individual component has limited novelty. The Theorem 1 proof is essentially the composition of existing neuron-level equivalence (Lemma 1, imported from Hu et al. 2023) with standard BN fusion—a valid but modest theoretical step.

4. **Architectural modifications required for equivalence are under-discussed**: The "SNN-friendly morphing" (max-pooling → average-pooling, RCR) changes the functional form of the original ANN. The reported "(Q)ANN accuracy" in Table 3 is for the *morphed+PET QANN*, not the original floating-point ANN, making it difficult to assess how much accuracy is lost to these structural changes alone. The conclusion suggests the method could apply to large models "where retraining or fine-tuning are not feasible" via post-training quantization, but all strong results rely on PET fine-tuning and morphing—no evidence is provided for direct conversion without retraining.

### Minor:

1. **ST-BIF neuron deployability on existing neuromorphic hardware is unaddressed**: The bipolar spikes and spike tracer require hardware support not available on most current neuromorphic platforms (e.g., Loihi 2 supports graded spikes but not necessarily the ST-BIF mechanism natively). While Table 7 shows low power overhead for the spike tracer in 65nm CMOS, no discussion of real hardware deployment is provided.

2. **Object detection experiments use SpikeZIP-N (without PET or RCR)**: Table 6 validates object detection without the paper's two main technical contributions, yet claims "general applicability of the conversion theory" in the conclusion.

3. **No variance reported**: Tables 3–6 report single numbers without error bars or standard deviations across runs, making it difficult to assess the significance of some improvements (e.g., 74.21% vs. 74.14% on ImageNet ResNet-34).

### Trivial:

1. The notation $S_t$ is overloaded between "spike tracer" (eq. 1) and "accumulated spikes" in other contexts, which can cause momentary confusion.

## Nice-to-Haves

1. **Comparison with the QANN run directly on hardware** (e.g., low-bit integer inference on CPU/GPU/edge processors) to ground the practical motivation for SNN deployment.
2. **Experiments on modern architectures** (e.g., MobileNet, EfficientNet) to demonstrate scalability beyond VGG/ResNet, though these are standard benchmarks in the field.
3. **Broader comparison with recent learning-based SNN methods** (beyond SEW and MS-ResNet from 2021) and across multiple time-step values, not just $T=4$.
4. **Discussion of the applicability to event-based (asynchronous) data**, since the even-release encoding assumes synchronous, frame-based input.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **"No experiments on modern architectures (MobileNet, EfficientNet, ViT)"** (from Spark reviewer): Demanding modern architectures beyond VGG/ResNet is a generic expansion request. VGG-16 and ResNet-34 are the standard architectures used across ALL compared conversion-based methods (QCFS, Offset, Fast-SNN, etc.), so this is not a missing baseline but rather a scope expansion request.

2. **"Missing related works from 2022-2024"** (from Spark reviewer): Per the rules, I cannot confirm the existence of specific uncited works, and the paper already compares against the most directly relevant concurrent methods (Offset, QCFS, Fast-SNN, QFFS).

3. **"Peak accuracy at $T < T_{eq}$ contradicts the practical value of equivalence"** (from Spark reviewer, rephrased to claim it renders equivalence useless): This overstates the issue. The equivalence *does* provide value: it guarantees that the SNN eventually matches the QANN exactly, establishing an upper bound and a termination condition for inference. The fact that accuracy often peaks earlier is an empirical observation, not a contradiction of the theorem.

4. **"Compounding rounding errors across layers are not addressed in the proof"** (from Spark reviewer): This is incorrect. The proof of Theorem 1 shows exact equality at each layer because the ST-BIF neuron with $S_{max} = n$ and appropriate $V_{thr}$ perfectly implements Q-ReLU at equilibrium. There are no accumulating rounding errors—each layer's output is exactly representable as accumulated spikes, which then serve as exact integer input to the next layer. The L1 distance of 0.5 in Figure 5 confirms near-exact matching (attributed to floating-point hardware).

5. **"Report standard deviations across multiple runs"** (from Spark reviewer): This is a standard but not universally adopted practice in the SNN conversion community, and the paper reports results consistent with community norms (single-run evaluations). The absolute gaps in key comparisons (e.g., 74.21% vs. 74.14%) are indeed small, but this is already noted in the paper.

6. **"RCR changes model capacity"** (from Spark reviewer): The paper acknowledges RCR is similar to SEW-ResNet's topology and explicitly footnotes the difference in motivation. The ablation (Figure 6a) shows RCR helps SNN performance, and the architectural change is clearly described. This is a known design choice, not a hidden confound.

## Novel Insights

The paper reveals an interesting tension that is common in the QANN-to-SNN conversion literature: as the theoretical framework becomes tighter (proving exact equivalence), the practical motivation for SNN deployment becomes murkier—since an exactly-equivalent SNN offers no information-theoretic advantage over the QANN it came from. The practical value of SpikeZIP's SNNs at low time-steps (before equilibrium) appears to stem from PET's multi-path training rather than the equivalence theorem, suggesting that the engineering contribution (PET) is more impactful than the theoretical one (Theorem 1) for the paper's actual empirical claims.

## Suggestions

1. **Directly compare SNN energy consumption against QANN integer inference** (not just ANN FLOPs) to ground the practical motivation. If neuromorphic deployment is the goal, articulate what hardware advantage the SNN provides over a 4-bit integer QANN running on an edge processor.
2. **Clearly separate the equivalence claim from the low-$T$ performance claim** in the abstract and introduction. State explicitly that the Pareto-front improvements come from PET engineering rather than the equivalence guarantee.
3. **Run SpikeZIP-PR (the full method, with PET and RCR) on object detection** to validate that the core technical contributions transfer beyond classification.

## Score Justification and Calibration

**Calibration against similar papers:**

- **QAC** (GTzP2GC7NR): ANN-SNN conversion with quantization-aware approach, scores 5/6/6/6 → avg ~5.75, rejected. SpikeZIP is more complete (provides formal equivalence proof, broader experiments), but shares the same fundamental question about SNN vs. QANN advantage.
- **Timesteps meet Bits** (KjiNHPinrS): Similar conversion framework, scores 5/5/3/6 → avg ~4.75, withdrawn/rejected. SpikeZIP is substantially stronger empirically and has a genuine theoretical contribution.
- **OneSpike** (QRWrvzRU4w): Claims mathematical equivalence between SNN and quantized ANN, scores 3/3/3/6 → avg ~3.75, withdrawn/rejected. SpikeZIP's equivalence is more carefully stated and the experimental work is far stronger, but the "just a quantized ANN" concern is similar.
- **Spatio-Temporal Approximation** (XrunSYwoLr): SNN conversion for Transformers, scores 8/8/6/6 → avg ~7.0, accepted. This work tackled a genuinely new domain (Transformers) with clear practical novelty.
- **QP-SNN** (MiPyle6Jef): Quantized and pruned SNN, scores 8/5/6/8 → avg ~6.75, accepted poster.

SpikeZIP is stronger than the rejected conversion papers (QAC, Timesteps meet Bits, OneSpike) in empirical performance and theoretical rigor, but shares their core limitation about the practical motivation for SNN deployment. It is weaker than the accepted papers in conceptual novelty—PET is a well-executed engineering contribution (adaptation of slimmable networks to quantization levels) rather than a fundamentally new idea, and the Theorem 1 proof is a careful but incremental formalization of prior neuron-level work. The misalignment between the theoretical claim (equivalence at equilibrium) and the empirical operating regime (low time-steps before equilibrium) is a genuine concern that weakens the paper's narrative coherence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>