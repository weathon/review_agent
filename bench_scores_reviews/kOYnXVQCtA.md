## Summary
DeeperForward modifies the Forward-Forward (FF) algorithm by replacing squared-sum goodness and vector-length normalization with mean goodness and layer normalization, enabling layer-wise local training on CNNs up to 17 layers — well beyond prior FF-based work limited to 4-layer networks. The paper also introduces parameter-free residual structures adapted for local learning, a Signal Integrating and Pruning (SIP) module for layer selection, and a pipeline model-parallel strategy. Experiments on CIFAR-10, MNIST, and Fashion-MNIST show meaningful improvements over existing FF-based methods, particularly an 8.11% gain on CIFAR-10 over the best reproduced FF baseline.

---

## Strengths

- **First FF-based method to demonstrate systematic depth scaling.** The paper includes a direct same-architecture, reproduced comparison: CwComp* at 14 layers achieves 75.28±0.54% on CIFAR-10, while CNN-ours at 14 layers achieves 81.76±0.30%, a clean apples-to-apples gain. Extending to a 17-layer ResNet reaches 86.22±0.17%, which far exceeds all prior FF-based work and confirms that the mean-goodness design genuinely enables deeper training, not merely a numerical increment.

- **Mean goodness + layer normalization is a principled, targeted fix for two concrete failure modes of squared goodness.** The paper correctly identifies that squared goodness (i) suffers from feature-scaling issues (LN vs. vector-length normalization) and (ii) produces zero gradients for deactivated (zero-activation) neurons under ReLU (Eq. 3 vs. Eq. 6). The ablation in Table 4 supports this: removing mean goodness drops accuracy by 6.84 percentage points on CIFAR-10 within the same 17-layer ResNet.

- **Practical parallel training with concrete, measured efficiency gains.** The model-parallel strategy exploits local-update independence to pipeline layers across GPUs. Table 5 shows DeeperForward is faster than BP-DDP at 1 and 2 GPUs (36.38s vs. 51.98s single-GPU; 20.77s vs. 32.70s at 2 GPUs), with a concrete explanation for the 4-GPU bottleneck (load imbalance). This is not just a theoretical claim but a measured result.

- **Parameter-free residual structures adapted for local learning.** The addition and concatenation residual shortcuts (Eq. 10–11) are a practical adaptation that avoids learnable shortcut weights conflicting with local updates. The ablation shows residuals are the single largest contributor after mean goodness (86.08% with, 81.02% without).

---

## Weaknesses

### Fatal
None.

### Major

- **CIFAR-100 comparison is uncontrolled for model capacity.** Table 2 shows ResNet-ours at 53.09% (below BP at 58.01%) and ResNet-CHx3-ours at 60.28% (above BP at 58.01%), but CHx3 triples channel count only for the proposed method, giving it substantially more parameters than the BP baseline. There is no experiment giving BP the same tripled parameter budget. Because the paper's own text notes that "inadequate allocation of neurons to each class results in sharp performance decline," the CIFAR-100 gain may be entirely attributable to parameter overprovisioning, not the training rule. This must be controlled to make the CIFAR-100 result interpretable.

- **Ablation does not disentangle layer normalization from mean goodness.** Table 4 removes both LN and mean goodness simultaneously (falling back to squared goodness + vector-length normalization). A row with "LN + squared goodness" is needed to determine whether the 6.84% gain is attributable to the normalization change, the goodness formulation, or their interaction. Without this, the paper's central mechanistic claim cannot be verified.

- **The gradient derivation for Eq. 6 — the paper's core claim about deactivated neurons — is absent.** The paper asserts that mean goodness yields "Δ W_ij = C x_i ∂L/∂g," i.e., a weight update independent of y_j, thereby avoiding dead-neuron stalling. This is one of the two main technical contributions, yet no derivation is provided in the main paper or appendix. The claim is plausible but must be shown, not asserted, given it underpins the entire motivation.

- **Scaling beyond 17 layers fails, but this is buried in the appendix.** Section 4.5 mentions that 33- and 100-layer experiments "do not observe significant performance improvements" (Appendix D). A paper titled "DeeperForward" that plateaus at 17 layers has an unresolved depth-scaling problem. This result belongs in the main text with an analysis of why scaling stalls, not hidden in an appendix. As written, the "Deeper" promise in the title is only partially fulfilled.

- **Mathematical notation in Eq. 5 is inconsistent with the paper's stated design.** Eq. 5 defines g = ∑_i y_i (a scalar sum), but then writes z = (y − g)/√(σ² + ε), where subtracting a scalar sum from a vector is not standard layer normalization (which subtracts the mean). Eq. 7 correctly defines ŷ = (1/HWC)∑h (the mean). The discrepancy between Eq. 5 and Eq. 7 creates ambiguity about what "mean goodness" is at the core theoretical level. The notation should be corrected and made consistent.

### Minor

- **Data augmentation gap is relegated to an appendix.** Section 4.2 acknowledges that "the improvement is limited after data augmentation, as detailed in Appendix C," but does not show or discuss these results. Data augmentation is standard practice; if augmentation substantially widens the gap with BP, this is a practically important limitation that deserves in-text analysis and not just a footnote reference.

- **Parallel comparison conflates different parallelism paradigms.** Table 5 compares the proposed pipeline/model-parallel strategy against BP with DDP (data parallel). These paradigms have different tradeoffs. The paper does not report memory usage in the pipelined setting (FIFO queues buffer activations between stages), whether effective batch sizes are identical, or what a BP pipeline-parallel baseline would achieve. The memory savings claim (618.64 MB vs. 1314.49 MB) lacks methodology detail — it is unclear whether optimizer state and queue buffers are included.

- **SIP's held-out data cost is unanalyzed.** The SIP module reserves 10,000 training samples (MNIST/F-MNIST) and 5,000 (CIFAR-10) purely for layer selection, not training. Given that SIP's accuracy gains are already marginal (Table 3: 86.45→86.51 on CIFAR-10), the trade-off between data held out and the selection benefit is not established.

### Tiny

- **Table 4 first row appears identical to last row in the extracted text** (both show "✓ ✓ ✓"), which is a formatting artifact from the paper's table — but the body text clarifies the intent. This should be corrected in the final version for unambiguous readability.

---

## Nice-to-Haves

- **Layer-by-layer accuracy curves for CwComp (14-layer) vs. DeeperForward (14-layer)** would directly illustrate the claimed "goodness leakage → overfitting in deeper layers" mechanism. The current Figure 5(d) shows per-layer accuracy for DeeperForward with/without residuals, but the most compelling visualization — CwComp collapsing in deeper layers vs. DeeperForward remaining stable — is absent.

- **Quantitative dead-neuron statistics across layers.** Appendix I is referenced but absent from the main text. Reporting the fraction of zero-activation neurons per layer for squared vs. mean goodness across training epochs would make the deactivated-neuron claim falsifiable rather than asserted. A training-epoch heatmap of dead neurons for both variants would directly validate the method's core motivation.

- **Convergence analysis (total wall-clock time and total epochs to target accuracy)** alongside epoch-time comparisons. If DeeperForward requires substantially more epochs to converge than BP, the per-epoch efficiency advantage diminishes in total training time. This is especially relevant for comparing against BP-DDP holistically.

- **Parameter counts for all compared model configurations.** The paper does not report parameter counts for any of its own architectures or those it reproduces. Given that channel count is a crucial sensitivity (CIFAR-100 results), parameter-matched comparisons would substantially strengthen the empirical claims.

- **An analysis of why data augmentation disproportionately hurts FF-style training.** Whether this is a fundamental limitation of greedy local objectives or a correctable design issue would be valuable to the community.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **"DeeperForward is not a true FF algorithm because it uses local cross-entropy."** The paper explicitly frames DeeperForward as an FF-inspired extension; the use of local CE is stated and intentional. Criticizing it for departing from canonical FF semantics treats an acknowledged design choice as a hidden flaw. REMOVED.

- **Biological plausibility of layer normalization.** Reviewer 2 notes that LN requires computing population statistics, which is debated in the strict bio-plausibility literature. The paper acknowledges bio-plausibility as one motivation among several (efficiency, parallelism), and does not claim strict neuronal realism. Demanding that layer norm itself be biologically justified is scope creep. REMOVED.

- **Demanding a formal mathematical invariance proof that downstream layers cannot recover goodness.** This is a non-standard theoretical requirement for an empirical local-learning paper. REMOVED; noted as nice-to-have (goodness leakage quantification).

- **Criticisms of the heterogeneous Table 1 comparisons with non-reproduced baselines (PEPITA, DTP, rec-LRA, etc.) as though they constitute the paper's primary claims.** The paper correctly identifies its main head-to-head as the CwComp* reproduced comparison. The broader table entries serve as historical context, as is standard in this literature. REMOVED as a major weakness (retained only as a contextual note).

- **Requesting significance tests or confidence intervals beyond 5-run mean±std.** Five-run evaluation with standard deviations is the norm for this class of experiments. Demanding formal significance testing is not a standard expectation for CIFAR-scale empirical results in this community. REMOVED.

- **"Claims that a cited reference does not exist / a method is not yet released."** No such claims are made in the sub-reviews here, but the instruction is noted; none are applicable. N/A.

- **"The paper should be evaluated as a general deep learning breakthrough, not just within FF."** The stated scope is explicitly FF-based local learning. Criticisms premised on matching SOTA BP performance are scope creep. REMOVED as a major weakness (retained as minor contextual note in Strengths).

---

## Novel Insights

The most insightful observation across the three reviews — not fully developed in the paper itself — is the **interaction between goodness decoupling and depth scaling as a double-edged design constraint**. The paper shows that CwComp's goodness leakage causes overfitting in deeper layers, while mean goodness with layer normalization fixes this. However, the paper also reveals (in Appendix D) that scaling beyond 17 layers yields no further gain, and CIFAR-100 results show that channel-per-class allocation is a hard bottleneck. Taken together, these findings suggest a deeper tension: the channel-grouping design that enables per-class goodness scores in CNNs fundamentally limits representational capacity per class and creates a hard ceiling on both depth and breadth scalability. This is not a criticism of the method within its stated scope, but an insight that points toward the next necessary architectural innovation for FF-based local learning to become truly general.

---

## Suggestions

1. **Add a controlled ablation row: LN + squared goodness, same architecture.** This single experiment would decompose the 6.84% gain and validate whether mean goodness or layer normalization (or both) is responsible.

2. **Show a parameter-matched BP baseline for CIFAR-100.** Train a ResNet-BP with the same tripled channel count and report its accuracy alongside ResNet-CHx3-ours. Without this, the CIFAR-100 result is uninterpretable.

3. **Move 33-layer and 100-layer results into the main paper** with an analysis of why scaling stalls. A model claiming to enable "deeper" training must address where and why the depth ceiling lies.

4. **Fix and clarify Eqs. 5–6.** Rewrite Eq. 5 so that g is explicitly the mean (1/N ∑ y_i), making it consistent with Eq. 7, and add a brief derivation (or clear appendix pointer) for Eq. 6 showing why the update is independent of y_j.

5. **Move augmentation results (currently Appendix C) into the main paper** and provide at least a paragraph analyzing why augmentation disproportionately hurts the local-learning approach relative to BP.

6. **Report per-layer accuracy curves for CwComp (14-layer) vs. DeeperForward (14-layer)** to directly visualize the goodness-leakage and overfitting claim that motivates the method.

---

**Evaluation axes:**
- **Novelty:** Moderate. Mean goodness replacing squared goodness is a targeted, principled change rather than a new paradigm, but it addresses a real failure mode and is not obvious.
- **Technical soundness:** Partially solid. The core gradient claim (Eq. 6) is unproven, and the notation is inconsistent, but the overall design logic is coherent and supported by ablation.
- **Empirical support:** Reasonably strong within the FF niche, particularly the CwComp*-14L direct comparison. The CIFAR-100 result is confounded by parameter mismatch. The depth-ceiling issue (no gain beyond 17 layers) is a genuine empirical gap.
- **Significance:** Meaningful to the local-learning and bio-plausible training community; moderate for the broader ICLR audience. The efficiency results are a practical plus.
- **Clarity:** Adequate overall; specific notational issues in the key equations undermine the precise technical argument.