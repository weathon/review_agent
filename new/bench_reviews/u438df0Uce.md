Now let me read the calibration papers for scoring.Now I have enough to write the consolidated review. Let me produce it.

---

## Summary

SpikeZIP proposes a two-step ANN→QANN→SNN conversion framework designed to improve the accuracy–latency Pareto front of converted SNNs. The method introduces (1) Paths-Ensemble Training (PET), which simultaneously trains a QANN across multiple quantization levels using shared weights and an ensemble loss, and (2) Residual Connection Re-routing (RCR), which reorganizes residual-block topology to reduce the spiking-transmission bottleneck in converted ResNets. The paper additionally claims a rigorous model-level equivalence proof between QANN and its converted SNN via the ST-BIF neuron with evenly-release analog encoding. Experiments on CIFAR-100 and ImageNet show competitive or superior results compared to prior conversion-based methods.

---

## Strengths

- **Strong Pareto-front empirical results.** Table 3 reports 74.21% accuracy on ImageNet with ResNet-34 at 11 time-steps and 73.92% with VGG-16 at 9 time-steps, improving over the prior best conversion-based results in terms of accuracy per time-step. These are non-trivial gains on a large-scale benchmark.

- **PET is a well-motivated and carefully ablated contribution.** Section 3.4, Figure 6, and Table 5 together present a convincing case that sharing weights across quantization-level paths regularizes the major-path QANN for low-latency SNN inference. The parameter-sharing design (for Q-ReLU scales and batch-norm) is thoughtfully justified, and the ablation over path configurations and loss coefficients is thorough.

- **RCR addresses a concrete architectural bottleneck.** Figure 3 and accompanying ablations (Figure 6a) show that re-routing the residual addition before the spiking neuron layer reduces the integration delay in converted ResNets, with meaningful accuracy gains at small T.

- **Training cost advantage.** Figure 4 demonstrates roughly 43.5× GPU-hour reduction over SEW (BPTT) at equivalent or better accuracy, which is practically significant.

- **Reasonably comprehensive evaluation.** The paper evaluates both VGG-16 and ResNet architectures on CIFAR-100 and ImageNet, includes object-detection results on PASCAL VOC/MS COCO, and provides energy consumption estimates.

---

## Weaknesses

### Fatal
*None. The paper's practical contributions remain credible. However, the combined weight of the Major issues below means the theoretical framing substantially overclaims.*

---

### Major

- **SNN accuracy exceeds QANN accuracy — direct empirical contradiction of the claimed equivalence.** In Table 3 (ResNet-34 / ImageNet), SpikeZIP-PR reports SNN accuracy of **74.21%** at T=11, while the corresponding QANN accuracy is only **73.65%**. Theorem 1 states that SNN accumulated output equals QANN output at T_eq; under this equivalence, SNN accuracy at T_eq cannot exceed QANN accuracy. Footnote 4 explicitly acknowledges that "the peak accuracy of SNN is not achieved at T_eq but a time-step T < T_eq." This means the empirically superior Pareto-front performance is produced at *sub-equilibrium* states where the equivalence theorem does not apply, and the theorem therefore does not explain the headline result. The paper presents PET/RCR as the practical driver and equivalence as the theoretical foundation, but these two threads are never reconciled: if the best performance arises before T_eq, the equivalence result is disconnected from the reported gains. This tension is the most important issue the authors must address.

- **Proof of Theorem 1 is informal and skips the key inductive step.** Proof 3.1 sets V^in = W̃_l X_{l-1} + b̃_l and concludes "by extending the equivalence between blocks to the network, eq. (8) is proven." In a multilayer SNN, layer l does *not* receive X_{l-1} as a static quantity — it integrates the spike outputs from layer l-1 as they are emitted over time. The argument that accumulated output is order-independent for the ST-BIF neuron (so total accumulated input still equals W̃_l X_{l-1} + b̃_l) is plausible but is *not stated* in the proof. The inductive structure, the condition that a common finite T_eq exists for the whole network, and the validity of the substitution across layers are all omitted. This does not necessarily mean the theorem is *false*, but the proof as written does not establish it; the key step is hand-waved with "by extending." Given that model-level equivalence is advertised as the primary theoretical contribution, this is a significant gap in rigor.

- **Architecture modification (RCR) is not isolated in the headline SOTA comparison.** SpikeZIP-PR, the variant used for the strongest ResNet results, applies both PET and RCR. RCR fundamentally alters the residual topology relative to all comparison baselines in Table 3. The paper is transparent that SpikeZIP-PR uses RCR, and readers can compare SpikeZIP-P vs. SpikeZIP-PR; however, the headline claim of "better Pareto-front performance than prior conversion works" for ResNet mixes backbone-topology improvements with conversion improvements, and is not a purely method-level comparison. (Note: max→avg pooling substitution is standard across nearly all SNN conversion methods and is *not* a new asymmetry.)

---

### Minor

- **Object-detection experiment is too limited to support "generalization of conversion theory."** Section 4.4 only tests SpikeZIP-N (no PET, no RCR) on YOLOv3 after replacing LeakyReLU with ReLU. The comparison baselines use different architectures (Tiny YOLO, YOLOv2). The claim that this "shows the potential of the conversion theory" is modest in the paper, but is too loosely supported to draw conclusions about generalizability; PET/RCR benefits on detection are completely untested.

- **No mechanistic explanation for why PET improves the major path.** The paper invokes the slimmable network analogy (Section 3.4) but does not explain how gradients from lower-quantization sub-paths improve major-path representation. Without this, PET appears as a successful empirical trick rather than a principled contribution, weakening the technical narrative even if the effect is real.

- **Feature-map equivalence claim overstated.** Section 4.2 concludes "only SpikeZIP shows the equivalence" from L1-norm comparisons against QCFS and Fast-SNN on 100 ImageNet samples. However, those methods differ from SpikeZIP in neuron model, training, encoding, and architecture; the experiment does not isolate model-level equivalence as the sole factor. The claim should be narrowed to "SpikeZIP's feature maps converge faster to the QANN" rather than serving as proof of equivalence.

- **Energy analysis uses analytical SOP estimates, not hardware measurements.** The comparison in Figure 7 and Table 7 against QCFS is done via SOP-based analytical estimation (using the ROLLS processor energy figures from 2015), not measured on deployed neuromorphic hardware. This is a known limitation of the field and is acceptable if framed carefully, but the paper should explicitly acknowledge that SOP counts underestimate the cost of managing spike sparsity and that results may not transfer to current accelerators.

---

### Trivial

- The L1-norm residual of 0.5 in Figure 5 (at equilibrium) is attributed to GPU floating-point error. While plausible, this should be distinguished from potential algorithmic discrepancies (e.g., floor vs. round, initial membrane potential offset). A brief quantitative argument would be more rigorous than "GPU computing error."

---

## Nice-to-Haves

- **Ablation over PET quantization-level configurations** ({8,4,2} vs. {16,8,4} vs. {4,2,1}) to guide practitioners on tuning this component. Figure 6b ablates $n_{mp}$ but not the choice of sub-path levels or number of paths.

- **Per-layer convergence analysis** showing at which layer and time-step the SNN feature maps diverge from the QANN, to validate whether the equivalence holds uniformly or only aggregated across layers.

- **Hardware implications of the evenly-release encoding strategy.** Existing neuromorphic chips often have constraints on temporal input patterns; a brief discussion of whether the proposed encoding is implementable on Loihi/TrueNorth without large overhead would strengthen the deployment claim.

- **Evaluation on DVS/event-camera datasets** to verify that the method extends to the temporal, asynchronous domain that motivates SNN research.

---

## Removed Points

*These points were identified by reviewers but are not upheld after verification against the paper:*

> **Harsh Critic — "Proof is entirely unsupported / central theorem is unsupported as written."** The critic claims the proof is *fundamentally* broken. After reading Proof 3.1, the underlying logic (ST-BIF output depends only on total accumulated input, not spike timing, enabling an inductive layer-by-layer argument) is plausible and has been used implicitly in related work (Hu et al. 2023, Lemma 1). The proof is *informal and incomplete* (the inductive step is not stated), which is kept as a Major weakness, but characterizing it as "completely unsupported" is too strong. The harsh critic's claim that "this is not a clarification issue" overstates the case.

> **Harsh Critic — "Architecture changes make the entire SOTA comparison invalid."** Max-pooling→average-pooling substitution is standard in SNN conversion, used by virtually all comparison methods (Rueckauer 2017 and descendants). Flagging this as a key unfairness asymmetry is not supported. The RCR concern is kept (as a Major weakness), but the broader claim that the "whole" comparison is invalid is exaggerated.

> **Human Finder — "No neuromorphic hardware implementation provided; Loihi compatibility unproven."** This is a standard limitation of GPU-evaluated SNN papers and reflects scope, not error. The paper claims compatibility in principle; demanding actual hardware deployment goes beyond what is standard in this sub-field.

> **Human Finder — "Comparison with 'more recent' BPTT methods (Spiking Transformers, TEILU 2023-24) is missing."** Per review rules, missing related works are not flagged as we cannot independently verify existence. The paper does compare with BPTT methods in Table 4.

> **Spark Reviewer — "PET training cost vs. conversion baselines not reported."** The paper reports training cost vs. SEW (BPTT) as its primary comparison target for training efficiency. Conversion-based methods (QCFS, Offset) do not use BPTT so their training cost is lower by default — requiring this comparison would favor the baseline, which per rules is removed.

> **Neutral Reviewer — "RCR may degrade feature representation quality."** The ablation in Figure 6a provides direct empirical evidence of RCR's effect on accuracy vs. time-step curves. The concern is addressed by the paper's own data.

---

## Novel Insights

The most genuinely novel and underappreciated insight in the three reviews is from the Spark Reviewer: that the paper's practical contribution (superior Pareto-front at T < T_eq) and its theoretical contribution (equivalence at T_eq) operate in different regimes and may be fundamentally disconnected. If the model-level equivalence theorem holds but peak accuracy is achieved before equivalrium, the theorem is not the mechanism behind the headline empirical result. This is not a fatal flaw — PET and RCR are valid contributions in their own right — but it means the paper has two separable contributions (theory + practice) that are presented as unified when they may not be. Clarifying whether the theory motivates the specific design of PET/RCR, or whether PET/RCR are purely empirically justified, would substantially strengthen the paper's coherence.

---

## Suggestions

1. **Directly address the SNN > QANN accuracy anomaly** in a dedicated paragraph. Either argue that the QANN accuracy of 73.65% reflects quantization at n_mp=8 while the SNN at T=11 produces an intermediate effective quantization leading to better generalization, or explicitly state that the Pareto-front claim is orthogonal to the equivalence theorem and operates at T < T_eq.

2. **Strengthen Proof 3.1** by (a) explicitly stating the inductive hypothesis, (b) showing that layer l's total accumulated input in the SNN equals W̃_l X_{l-1} + b̃_l regardless of the temporal distribution of spikes from layer l-1 (invoking the order-independence of ST-BIF accumulation), and (c) bounding T_eq for the full network in terms of T_off and the depth of the network.

3. **Separate the Pareto-front performance claim into a "PET/RCR" section** and the "model equivalence" claim into a "conversion theory" section, and be explicit that the former drives the headline results while the latter provides theoretical support for the conversion at equilibrium.

4. **Add an experiment testing SpikeZIP-P or SpikeZIP-PR on the object detection task**, or else clearly frame the detection result as a preliminary proof-of-concept and not as a general validation.

5. **Explain the mechanism of PET** beyond the slimmable network analogy — e.g., via a CKA similarity analysis or gradient analysis showing how sub-path gradients regularize the major-path representations.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Score | Decision |
|---|---|---|---|
| GTzP2GC7NR | ANN-SNN conversion, analog encoding | 6,6,5,6 (avg 5.75) | Rejected |
| KjiNHPinrS | ANN-SNN conversion, similar weaknesses | 5,5,3,6 (avg 4.75) | Rejected |
| lGUyAuuTYZ | SNN efficiency, moderate weaknesses | 5,6,6 (avg 5.67) | Accepted (poster) |
| eoSeaK4QJo | SNN pruning, strong empirical | 8,3,8,6 (avg 6.25) | Accepted (spotlight) |

SpikeZIP compares favorably to GTzP2GC7NR and KjiNHPinrS in empirical quality — its ImageNet results are stronger, its ablations are more thorough, and PET is a more original contribution than the methods in those papers. However, it shares their weakness of an informally justified proof and, critically, adds an unexplained empirical anomaly (SNN > QANN accuracy) that the calibration papers do not have. The paper's practical contributions (PET + RCR) are genuinely useful and match the standard for a poster-level acceptance in this field, comparable to lGUyAuuTYZ. However, the theoretical contribution is the primary advertised selling point, and it has both a proof-rigor issue and the fundamental T_eq vs. peak-accuracy tension. The paper therefore sits between lGUyAuuTYZ (accepted poster, 5.67 avg) and GTzP2GC7NR (rejected, 5.75 avg) — borderline, leaning slightly below acceptance given the Major weaknesses.

**Final Score: 5.0** — Marginally below acceptance threshold. The empirical contributions merit attention, but the theoretical framing requires either substantially strengthening the proof and addressing the SNN>QANN anomaly, or reframing the paper to not center the equivalence theorem as the primary contribution.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>