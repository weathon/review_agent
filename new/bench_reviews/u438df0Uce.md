Now I have enough calibration data. Let me compose the final review.

## Summary

SpikeZIP proposes a framework for converting pre-trained ANNs to SNNs via a two-step ANN→QANN→SNN pipeline. Its two claimed core contributions are: (1) a rigorous proof of model-level equivalence between QANN and SNN under specified conditions (Q-ReLU activation, ST-BIF neurons, analog input/bias encoding, and SNN-friendly operator replacements), and (2) a Paths-Ensemble Training (PET) algorithm with Residual Connection Re-routing (RCR) that together improve the accuracy–latency Pareto frontier, achieving SOTA results such as 74.21% on ImageNet with ResNet-34 at 11 time-steps.

## Strengths

- **Strong empirical Pareto-frontier results**: SpikeZIP consistently improves the accuracy-latency tradeoff across datasets (CIFAR100, ImageNet) and architectures (VGG-16, ResNet-20/34). The ResNet-34 ImageNet result of 74.21% at 11 time-steps vs. Offset's 74.14% at 16 time-steps represents a meaningful latency reduction (Table 3).

- **Well-integrated system design**: The combination of SNN-friendly morphing, input/bias encoding, PET, RCR, and BN fusion forms a coherent, implementable pipeline. Each component is cleanly defined and the ablation studies (Fig. 6a, Table 5) demonstrate their individual and combined contributions.

- **Comprehensive internal ablations**: Table 5 provides detailed ablations over PET parameter sharing schemes (Q-ReLU scales and batch-norm statistics), loss coefficients, and label types, lending credibility to design choices. Fig. 6a cleanly isolates PET and RCR contributions.

- **Extension beyond classification**: The object detection experiments on PASCAL VOC and MS COCO (Table 6) and energy consumption analysis (Fig. 7, Table 7) demonstrate broader applicability and practical relevance.

- **Feature map verification**: Figure 5 provides both qualitative visualization and quantitative L1-norm comparison between QANN and SNN feature maps, offering empirical evidence for the closeness of the conversion.

## Weaknesses

### Fatal
None.

### Major

- **The "model-level equivalence" theorem is overstated relative to its technical depth and operational applicability.** Two interconnected issues undermine this central claim:

  (1) **The theorem operates at equilibrium (T_eq), not at the practical operating point.** Theorem 1 proves that the *time-accumulated* SNN output equals the QANN output at T_eq. However, Footnote 4 explicitly states "the peak accuracy of SNN is not achieved at the T_eq but a time-step T < T_eq," and all headline results (e.g., 74.21% at 11 steps, 73.92% at 7 steps) use T < T_eq. The theoretical guarantee therefore does not directly justify the practical performance gains. The proof establishes asymptotic equivalence after sufficient integration time, while the Pareto-front advantage comes from early-exit behavior that falls *outside* the theorem's scope. The paper frames the equivalence as a foundational contribution explaining the results, but the theorem and the experiments address different regimes.

  (2) **The theorem's proof is largely incremental over prior work.** Lemma 1 (neuron-level equivalence) is taken directly from Hu et al. (2023). The Theorem 1 proof composes this block-wise equivalence after assuming BN fusion, weight alignment, and the specific evenly-release encoding — the extension from a single block to the full network is stated as "by extending the equivalence between blocks to the network" (Proof 3.1). Crucially, the proof does not address inter-layer temporal dependencies (layer l cannot reach equilibrium until layer l−1 has stabilized), does not bound T_eq or prove its existence across arbitrary network depths, and does not handle the effect of RCR on the computation graph. The combination of (1) and (2) means the paper's central theoretical narrative — that rigorous model-level equivalence explains the strong Pareto-front results — does not hold as stated.

- **Comparisons are not fully controlled for fairness.** In Table 3, SpikeZIP's QANN accuracies differ significantly from baselines (e.g., VGG-16 on ImageNet: 77.07% vs. 76.28%/70.03%/68.16). Since part of the SNN improvement may stem from inheriting a stronger QANN (due to PET, distillation, etc.), the attribution of Pareto-front gains to superior conversion rather than better upstream training is ambiguous. Additionally, SpikeZIP uses ST-BIF neurons (bipolar spikes) while baselines like QCFS and OPI use standard IF neurons (binary spikes). Since bipolar spikes inherently carry more information per spike, this creates an asymmetric comparison that favors SpikeZIP, yet the paper does not address this.

### Minor

- **PET's claimed connection to SNN temporal information is heuristic.** The abstract states PET "considers the SNN temporal information when fine-tuning QANN," but PET as described is a multi-resolution quantization ensemble with shared weights and an additive loss (Eqs. 3–6). The link between quantization level distribution and SNN convergence dynamics is observed empirically but not theoretically grounded. PET is a sensible and effective training recipe, but the "temporal information" framing overstates the mechanism.

- **T_eq is uncharacterized.** The paper defines T_eq as the time when "neurons of entire SNN are static" but never analyzes its value, proves it is bounded, or shows how it depends on network depth or input. Without bounding T_eq, the practical latency implications of the equivalence guarantee remain unclear.

- **Limited architecture diversity.** Experiments are restricted to VGG-16 and ResNet-20/34. No results on modern architectures (ResNet-50/101, MobileNet, ViT) are presented, limiting confidence in scalability.

- **Detection experiments use only SpikeZIP-N (without PET/RCR).** Table 6 shows YOLOv3 results only for SpikeZIP-N, making it unclear whether PET and RCR transfer to detection tasks or are specific to classification.

### Trivial

- The claim in Section 4.2 that L1-norm = 0.5 is "resulted from the intrinsic computing error of GPU hardware" is stated without verification (no precision analysis is provided).

## Nice-to-Haves

- Evaluation on more diverse or modern architectures (e.g., ResNet-50, MobileNet-V2) would strengthen generalizability claims.
- Formal characterization or empirical measurement of T_eq across datasets and architectures would make the equivalence guarantee more actionable.
- A per-layer breakdown of QANN-SNN feature map discrepancy (rather than aggregate L1-norm) would reveal where equivalence holds or degrades.
- Discussion of hardware feasibility for ST-BIF neurons (spike tracer, bipolar spikes) on platforms like Loihi 2 would improve practical relevance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Why not deploy QANN directly instead of SNN?"** (from Human Finder): This is a generic concern applicable to the entire ANN-SNN conversion literature, not specific to this paper's contributions. The paper provides energy analysis (Section 4.5) as motivation, and this question is scope creep beyond what the paper claims to address.

- **"No actual neuromorphic hardware deployment/experiments"** (from Spark): This is a common limitation across the entire SNN conversion subfield and not a specific weakness of this paper. Demanding actual hardware deployment is above the community's standard for a methodology paper.

- **"Variance/confidence intervals not reported"** (from Spark): Standard practice in the SNN conversion literature is single-run evaluation on ImageNet. Demanding multiple seeds for large-scale benchmarks is not the norm in this area.

- **"Paper should acknowledge restrictions of equivalence (no max-pool, specific encoding)"** (partially from Harsh Critic): The paper actually does enumerate all four conditions explicitly in the contribution list (Section 1, bullet 1) and in Table 2. While the scope could be discussed more prominently, the paper does not hide these conditions.

- **"Different quantization levels in Table 3 vs Table 4"** (from Spark): Using {8,4,2} for the main Pareto comparison (targeting higher accuracy) and {3,2,1} for the learning-based comparison (targeting T=4 with low latency) is reasonable experiment design, not cherry-picking.

- **"The ~3% ANN→SNN accuracy drop is unanalyzed"** (from Spark): The paper explicitly explains this gap in Section 4.3: "there exists a small accuracy gap between the native ANN and peak accuracy achieved by the SpikeZIP variants. This is mainly resulted from the low quantization levels (e.g., n_mp=4) chosen to train the QANN for a fair comparison with the competing works."

## Novel Insights

The most insightful observation across the reviews is the tension between the equilibrium regime of Theorem 1 and the pre-equilibrium operating point where all practical gains are realized. This reveals that SpikeZIP's practical success stems not from the formal equivalence guarantee but from PET's empirically effective training that produces QANNs whose SNN approximations converge quickly toward QANN behavior — a property the theorem does not establish. The formal contribution and the empirical contribution are thus more decoupled than the paper's narrative suggests.

## Suggestions

- Re-frame the theoretical contribution honestly: present Theorem 1 as a *sufficient condition* for exact convergence that motivates the pipeline design, and separately acknowledge that the Pareto-front gains at T < T_eq are empirical achievements enabled by PET, not direct consequences of the theorem.
- Provide empirical measurements or bounds on T_eq across architectures and datasets to connect the theorem to practice.
- Add a baseline that trains QANN with standard LSQ (no PET) at the same quantization level and converts it, to isolate PET's contribution from the overall training recipe.
- Acknowledge the ST-BIF vs. IF neuron asymmetry in comparisons and discuss its implications.

## Score and Decision

**Calibration anchors:**

- **Spatio-Temporal Approximation** (accept poster, scores 8,8,6,6): Genuinely novel SNN conversion for Transformers with real methodological innovation. Significantly stronger contribution than SpikeZIP.
- **CSS Coding** (reject, scores 3,6,6,6): SNN conversion with fairness concerns in comparisons and limited novelty. Weaker than SpikeZIP.
- **QAC** (reject, scores 5,6,6,6): ANN-SNN conversion with mixed-timestep, moderate novelty; theoretical claims oversold relative to content. Comparable scope to SpikeZIP.
- **Error-Free ANN-SNN** (withdrawn/reject, scores 6,6,5,6): ANN-SNN conversion, somewhat similar overclaim pattern on theoretical side.
- **Timesteps meets Bits** (withdrawn/reject, scores 5,5,3,6): ANN-SNN conversion with novelty concerns.

SpikeZIP has substantially stronger empirical results than QAC and CSS Coding, and a cleaner system design than Error-Free or Timesteps meets Bits. However, the theoretical overclaim is a significant issue — the paper positions model-level equivalence as a core contribution, but the theorem is incremental and does not cover the operating regime where results are achieved. This is similar to the pattern seen in other rejected papers in this space (QAC, Timesteps meets Bits). The empirical contributions (PET, RCR, strong results) bring it above those, but the overclaim is a real drag. Compared to Spatio-Temporal Approximation (which earned ~7 avg), SpikeZIP is weaker on novelty but comparable on empirics.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>