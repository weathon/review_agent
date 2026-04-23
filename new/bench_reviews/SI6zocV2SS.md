## Summary

The paper introduces CAN (Continuously Adapting Networks), a continual learning architecture that combines Hebbian learning (with Oja's rule and lateral inhibition) to compute local importance scores for neurons, and then uses selective neuron freezing and gradient scaling to mitigate catastrophic forgetting. Task-specific binary masks define sub-networks, and inference requires manually selecting the correct mask. Experiments on CIFAR-10 (2-task split, MLP architecture) report average accuracies of 27.1% (task-incremental) and 27.01% (domain-incremental), compared to 22.5% and 15.5% for vanilla sequential training.

## Strengths

- **Conceptually reasonable core idea**: Using local Hebbian co-activation patterns as an importance signal for parameter protection is a distinct approach from gradient-based importance methods like EWC or SI, and aligns with the paper's biological motivation. The pipeline (Hebbian scoring → importance-based gradient scaling → selective freezing) is logically coherent.
- **Correct identification of Hebbian stabilization needs**: The paper recognizes that raw Hebbian learning diverges and incorporates Oja's rule (Section 3.1.2) for weight normalization and lateral inhibition (Section 3.1.3) for representational diversity—these are theoretically grounded choices that address known failure modes.

## Weaknesses

### Fatal

- **Near-chance-level accuracy undermines the central claim**: Table 1 reports CAN achieves 27.1% average accuracy on task-incremental CIFAR-10, where each task has 5 classes (random chance = 20%). A model performing only 7 percentage points above random on a 5-class problem does not demonstrate "significant reduction in catastrophic forgetting" as claimed in the abstract. While CAN outperforms vanilla ANNs (22.5%), the absolute performance is so low that it indicates the method is not functioning effectively—either due to a fundamental issue with the approach or a flawed experimental setup (e.g., using an MLP on CIFAR-10, which is known to perform poorly on image data).

### Major

- **No comparison to any established continual learning baseline**: The only comparison is against vanilla sequential training (Table 1), the weakest possible baseline. The paper cites EWC (Section 2.1), SI (Section 2.1), replay methods (Section 2.2), and Progressive Networks (Section 2.3) but does not compare against any of them. Showing improvement over a network with no anti-forgetting mechanism is a trivially low bar. Without comparison to at least one standard CL method, there is no evidence that CAN is competitive with even simple alternatives.

- **Figure 3 mislabels cross-entropy loss as accuracy**: Figure 3's y-axis is labeled "Acc" but shows values in the range 1.90–2.45 (the accompanying table confirms these values). These are clearly cross-entropy loss values, not accuracy. Notably, Figures 4–6 correctly label their y-axes as "loss" with similar value ranges. This mislabeling means the paper provides no training trajectory of actual task accuracy—the visual evidence for reduced forgetting is absent. This compounds the concern about the overall reliability of the experimental evaluation.

- **Extremely limited experimental scope**: The evaluation uses only 2 tasks, only CIFAR-10 (plus MNIST mentioned but not numerically reported), only an MLP with 256-128-64 hidden units, and only a single random seed (720). Two-task evaluation is insufficient for validating continual learning claims; standard benchmarks use 5–20+ tasks. Using an MLP on CIFAR-10 is a poor architectural choice for image data and likely contributes to the near-chance accuracy. No standard CL benchmarks (e.g., Permuted MNIST, Split CIFAR-100) are employed.

### Minor

- **Method description lacks key implementation details**: The threshold for converting Hebbian importance scores into binary masks is described only as "a pre-defined threshold" (Section 3.3) without specifying its value or how it was chosen. Lateral inhibition is described in Section 3.1.3 but never explicitly confirmed as implemented in the experiments. The architecture diagram (Figure 2) is a high-level flowchart without layer specifications.
- **Oracle task identity at inference**: The method requires manually selecting the correct mask at inference time (Section 3.2, Section 4.3), which is the easiest continual learning scenario. The paper acknowledges gating as future work, but this significantly limits practical applicability.

### Trivial

None beyond what is covered above.

## Nice-to-Haves

- Comparison to at least 2–3 CL baselines (EWC, SI, simple rehearsal) on standard benchmarks with proper CNN architectures would substantially strengthen the evaluation.
- Ablation of Hebbian scoring vs. random masking vs. magnitude-based masking to demonstrate the Hebbian component's contribution.
- Experiments with more than 2 tasks and proper continual learning metrics (backward transfer, forgetting measure) as recommended in the CL literature.
- Use of CNNs for image tasks and reporting actual accuracy curves over training epochs.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic claim about "overselling" in abstract/introduction**: While the abstract does overclaim, this is already captured by the near-chance accuracy weakness. The "repetitive" biological motivation criticism is a style/formatting nitpick.
- **Harsh critic claim that gradient scaling and hard masking are "conflated"**: The paper describes these as two steps in a pipeline (Section 3.2: "multiply the scaled hebbian updated values with the gradients" and separately "activation from irrelevant neurons... is multiplied by zero"). They are distinct mechanisms used sequentially, not conflated.
- **Strength Finder's "Evaluation across both task-incremental and domain-incremental settings"**: While technically true, both settings yield near-chance results, so the breadth of evaluation does not meaningfully strengthen the paper when the absolute performance is this poor. Moved to Removed Points.
- **Strength Finder's "Modular architecture design"**: This is a generic strength that any three-component system would have, and it conflicts with the verified fatal weakness that the system doesn't actually work well.

## Novel Insights

The core tension in this paper is instructive: the conceptual pipeline (Hebbian importance → selective freezing) is reasonable and biologically motivated, but the execution reveals that biological plausibility does not translate to engineering effectiveness without careful calibration. The near-chance accuracy combined with the mislabeled figures suggests the work was submitted prematurely, before the method was properly validated. The gap between the stated ambition ("significantly reduces catastrophic forgetting") and the empirical reality (27% on 5-class problems) is the paper's defining feature.

## Suggestions

1. **Fix Figure 3** to report actual accuracy (0–100%) over training epochs, or correctly label it as loss. This is essential for any credible experimental evaluation.
2. **Switch to CNN architectures** for CIFAR-10 experiments and compare against at least EWC, SI, and a simple replay buffer. Without this, the paper cannot establish that CAN offers any advantage over existing methods.
3. **Scale up experiments**: Use at least 5 tasks on Split CIFAR-10/100 or Permuted MNIST, and report standard CL metrics (average accuracy, backward transfer, forgetting measure) with multiple seeds and error bars.

## Calibration

**Anchors compared against:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| DIRAD/PREVAL (CL, weak experiments, MNIST/FMNIST) | ZHTYtXijEn.md | 2.33 | Very similar: weak CL experiments, limited datasets, poor baselines. CAN is comparably weak. |
| Projected Subnetworks (CL, subnetwork approach, weak experiments) | WM5G2NWSYC.md | 2.00 | Similar approach (subnetworks for CL), but even that had more rigorous evaluation than CAN. |
| Continual Weighted Sparsity (CL, meta-plasticity, proper benchmarks) | DaUsIJe2Az.md | 4.25 | Much stronger: proper CL benchmarks, baselines, and ablations, yet still rejected. CAN is significantly weaker. |
| Dying Neurons for Pruning | 2wFXD2upSQ.md | 5.50 | Different topic but shows the bar for ~5 scores: thorough experiments across multiple settings. CAN falls well below. |
| Budgeted Online CL (adaptive layer freezing) | dOAkHmsjRX.md | 7.50 | Comprehensive experiments, proper baselines, code available. Vastly stronger than CAN. |
| Hebbian Spiking NN (orientation maps) | rySLejeB1k.md | 7.33 | Also uses Hebbian plasticity with biological motivation but with thorough validation. |

The CAN paper is most similar to the low-scoring CL papers (2.00–2.33) that share its weaknesses: limited experiments, poor baselines, and near-trivial results. It is significantly weaker than the medium-scoring anchors (4.25+), which at minimum include proper CL baselines and standard benchmarks. The near-chance accuracy and mislabeled figures place it at the bottom of the quality range.

## Score and Decision

**Originality**: Low. Hebbian importance scoring for CL has been explored (Amato et al. 2019, cited by the paper itself). Selective freezing and sub-network masking are standard parameter isolation ideas. The combination is incremental.

**Importance of research question**: Moderate. Catastrophic forgetting is important, but this work does not advance solutions meaningfully.

**Claims well supported**: No. Near-chance accuracy (27% on 5-class problems) does not support the claim of "significant reduction in catastrophic forgetting." Figure 3 mislabels loss as accuracy. No CL baselines.

**Soundness of experiments**: Very poor. Only 2 tasks, only MLP, single seed, single dataset, no CL baselines, mislabeled figures.

**Clarity of writing**: Adequate. The high-level pipeline is readable, though method details are underspecified.

**Value to community**: Minimal in current form. The method does not demonstrate working continual learning.

MY FINAL SCORE: <pineapple>2.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>