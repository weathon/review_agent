## Summary

CAN (Continuously Adapting Networks) proposes to mitigate catastrophic forgetting by combining selective neuron freezing with Hebbian learning. The method maintains a separate Hebbian weight matrix updated by local co-activation signals, averages those weights per neuron to produce importance scores, and uses those scores to scale SGD gradients via PyTorch hooks — effectively constructing binary masks that prevent previously-assigned neurons from being modified when learning subsequent tasks. Experiments are conducted on 2-task splits of MNIST and CIFAR-10 under task-incremental (TIL) and domain-incremental (DIL) settings, compared only against a vanilla ANN baseline.

---

## Strengths

- **Using unsupervised, local Hebbian co-activation as the importance signal is a genuinely distinct mechanism for continual learning.** Most parameter-isolation methods compute importance from global, supervised signals — Fisher information (EWC), path integrals of gradient norms (SI), or loss-based saliency. Anchoring importance scoring entirely in unsupervised local activations, requiring no global error signal, is a different design point that is task-agnostic by construction and avoids the computational overhead of second-order gradient methods. This is a specific choice, not a generic observation.
- **Evaluating both TIL and DIL with different masking protocols for each is more complete than single-setting evaluation.** The paper correctly identifies that in the DIL setup the output heads overlap and masking at inference is inapplicable, and adjusts the protocol accordingly (gradient scaling only, no forward masks). This distinction is handled consistently.

---

## Weaknesses

### Fatal

- **Near-random performance makes the paper's core claim unverifiable.** Table 1 reports CAN at 27.1% average accuracy on CIFAR-10 TIL, on a 5-class problem where random chance is 20%. The vanilla baseline is 22.5% — also near-random. An improvement of 4.6 percentage points over a nearly non-functional baseline does not demonstrate that catastrophic forgetting has been meaningfully reduced; it demonstrates that neither model is learning the tasks. A system cannot be said to "remember" tasks it never successfully learned. The paper provides no single-task accuracy upper bound (i.e., what accuracy is achieved if the model is trained only on one task with no CL constraint), making it impossible to determine whether the bottleneck is the continual learning mechanism, the architecture, or a fundamental implementation bug. Until single-task accuracy is established as reasonable, no claim about forgetting can be evaluated.

- **Figure 3's y-axis is labeled "Acc" but displays values of 2.0–2.45, which are physically impossible for accuracy on any standard scale (0–1 or 0–100%).** These values fall in exactly the same numerical range as the loss curves in Figures 4–6, which are correctly labeled as loss. The accompanying table inside Figure 3 also lists "Task 1 (Acc)" and "Task 2 (Acc)" values such as 2.05 and 2.45. This is almost certainly a loss curve mislabeled as an accuracy curve. The consequence is that all visual evidence in the results section is either mislabeled or uninterpretable, and the only usable quantitative evidence is Table 1 — whose values are themselves near-random.

### Major

- **No comparison against any established continual learning method.** The sole comparison in the paper is against a vanilla ANN trained with standard SGD. For a continual learning submission, comparison against EWC, SI, Experience Replay, or parameter-isolation methods such as PackNet is a minimum expectation. Without such baselines, there is no way to know whether the modest improvement over vanilla SGD is competitive, redundant, or worse than trivial alternatives. The absence of these baselines, combined with near-random absolute performance, makes the empirical section essentially uninterpretable.

- **Oracle task identity at inference — the system is incomplete as presented.** Section 4.3 explicitly states: *"we are manually selecting the mask and measuring the metrics but it can be done using a gating system that automatically selects the relevant gate according to the given task during inference"*, and identifies the gating mechanism as future work. Requiring a human to select the task mask at test time means the reported TIL results are not reproducible in any realistic deployment scenario. The paper's broader language about "allowing the network to learn continually" overstates the current system. This should be clearly scoped to task-ID-known TIL evaluation.

- **Core methodology is underspecified and not reproducible.** After reading the full paper, the following questions remain unanswered:
  - Are the Hebbian weights a separate parameter matrix maintained in parallel to the ANN weights, or are they the ANN weights themselves? (Section 3.1.1 refers to "Hebbian parameter" *w* but this is never clarified structurally.)
  - Are Hebbian updates accumulated per sample, per batch, or per epoch, and over what window?
  - How exactly are Hebbian weights converted into gradient scaling values (normalization procedure, layerwise vs. global)?
  - Is Oja's rule (Section 3.1.2) actually used in the experiments? The paper says it "can be used in practice" but never states which rule is used experimentally.
  - Is lateral inhibition (Section 3.1.3) implemented in the reported experiments? No equation, no hyperparameter, and no mention in the experiment section is provided. Written in present tense as a method contribution ("we introduce competition"), yet absent from any implementation detail.
  - What threshold defines "selected" neurons (Section 3.3 references "a pre-defined threshold" but no value is given)?
  - What loss function is used? The paper specifies "Sigmoid activation at the end" for multi-class classification, which is non-standard (softmax + cross-entropy is the norm), and the loss is never named. This likely contributes directly to the near-random performance.
  - What optimizer configuration, batch size, weight decay, and learning rate schedule are used?
  - No pseudocode or formal algorithm is provided.

- **MNIST results are missing.** The paper states "All our experiments were conducted on the MNIST and the CIFAR-10 Dataset" but Table 1 — the only accuracy table — explicitly says "The above values are the results by using the CIFAR-10 dataset." No MNIST accuracy results are reported anywhere.

- **Single seed; no variance estimates.** All results come from one fixed seed (720). For a small MLP on CIFAR-10, running 3–5 seeds is inexpensive and expected. A single run cannot distinguish true performance from noise, especially given the small margins reported.

### Minor

- **Two-task evaluation only.** Both CIFAR-10 and MNIST are split into exactly two 5-class subsets. With only two tasks, long-term forgetting dynamics, capacity exhaustion, and mask accumulation behavior over many tasks cannot be assessed. Standard CL benchmarks (Split CIFAR-100, Permuted MNIST with 5–20 tasks) exist precisely for this reason.

- **The stated constraint "we can't use a continuous stream of data belonging to a variable number of classes" (Section 4.1.1) is a significant architectural limitation** mentioned only in passing. This restricts the method to fixed-output-head, fixed-task-count settings — a narrower scope than the introduction implies. It should be stated prominently as a limitation.

- **Abstract claim "new tasks can be trained without changing parameter weights" is inaccurate.** The method does change parameter weights — it trains the selected subset. The correct characterization is that *previously-assigned* parameters are not changed. This should be corrected.

- **"Time to Stability" (Section 5.3)** provides no quantitative table, no operational definition, no baseline comparison, and no analysis. The observation that Task 2 requires 20 epochs vs. 10 for Task 1 is presented without interpretation.

### Tiny

- Citing Zenke et al. (2017) (the Synaptic Intelligence paper) under meta-learning approaches in Section 2.5 is a citation misplacement; it belongs under regularization, where it is correctly cited in Section 2.1.
- The loss notation `L(θ) = L_n(θ_n) ∈ T_n` is ill-formed (a loss value is not an element of a task set); the meaning is recoverable from context but should be stated precisely.

---

## Nice-to-Haves

- **Ablation on the Hebbian importance mechanism.** Without comparing Hebbian-based neuron selection against random selection, magnitude-based selection, or gradient/Fisher-based selection, there is no evidence that the Hebbian component specifically — rather than the freezing strategy alone — contributes to any gains.
- **Capacity utilization analysis.** Report the fraction of neurons consumed per task, the overlap (if any) between masks, and a projection of how many tasks the 256-128-64 network supports before exhaustion.
- **Visualization of learned masks.** Heatmaps showing which neurons are selected per task per layer would reveal whether the method is finding meaningful task-specific subnetworks or partitioning neurons near-randomly.
- **Per-task single-task accuracy as upper bound.** Reporting the accuracy of a model trained solely on one task establishes a ceiling against which CL performance can be measured.
- Implement and evaluate the autoencoder-based gating mechanism to remove the oracle task identity requirement.

---

## Removed Points

*These points were raised in sub-reviews but are flagged for removal. They are preserved here for transparency.*

- **"Hou et al. (2025) is a future-dated citation"** (Review 2): The reference appears in the bibliography with a valid DOI (Information Sciences, 687:121368, July 2025). Per review policy, if the paper cites a reference, it is assumed to exist. Removed.
- **"The paper should use CNNs for CIFAR-10"**: The paper explicitly uses and scopes itself to MLPs. Demanding CNNs imposes architectural requirements outside the stated contribution. The low accuracy is a genuine concern but should be attributed to the near-random performance issue, not to the architecture choice. Removed as a standalone criticism.
- **Demand for theoretical convergence proofs**: This is an empirical systems paper. Theoretical guarantees on Hebbian convergence are not a standard expectation for this type of contribution. Removed.
- **Generic strengths** ("the paper is well-written," "the topic of continual learning is important"): These apply to any paper in the field and are not evidence of specific quality. Removed.

---

## Novel Insights

The three reviews collectively point to a deeper diagnostic concern worth highlighting directly for the authors: the paper's experimental results likely reflect an implementation-level problem rather than a merely weak method. The combination of (1) a non-standard sigmoid output layer for multi-class classification with an unnamed loss function, (2) loss curves for all conditions hovering near or above the entropy of a random 5-class predictor, (3) accuracy figures that are visually identical to loss figures yet labeled differently, and (4) final accuracy only marginally above random chance collectively suggest the network may not be converging to a working classifier at all — independent of the continual learning mechanism. This means the paper is not yet in a position to evaluate whether Hebbian importance scoring is a good strategy for subnetwork selection: the experiment cannot distinguish "Hebbian scoring is unhelpful" from "the network isn't trained correctly." Establishing a working single-task baseline with proper loss function and evaluation protocol is the essential first step before any CL claim can be made.

---

## Suggestions

1. **Establish single-task accuracy first.** Train the model on each 5-class split in isolation and confirm it achieves reasonable accuracy (>85% on MNIST, >55–60% on CIFAR-10 with an MLP). If it does not, fix the architecture and loss function before making any continual learning claims.
2. **Use softmax + cross-entropy** as the standard for multi-class classification. The sigmoid output is almost certainly causing the near-random performance. Explicitly state the loss function in the experimental setup.
3. **Correct Figure 3.** Replace mislabeled "Acc" curves with actual accuracy curves (0–100% scale) for both the vanilla and CAN models, measured on held-out test data after each epoch.
4. **Add at minimum EWC and a simple replay baseline** on the same 2-task setup. This is the single most important addition for the empirical section.
5. **Provide a complete algorithm description** specifying: (a) whether Hebbian weights are maintained as a separate matrix, (b) which Hebbian rule (basic or Oja's) is used in experiments and why, (c) whether lateral inhibition is implemented, (d) the exact normalization procedure for converting Hebbian weights to gradient scaling factors, and (e) the numerical threshold used for neuron selection.
6. **State the evaluation protocol explicitly** as task-ID-known TIL and remove language suggesting a more general continual learning solution until a gating mechanism is implemented and evaluated.
7. **Report MNIST results** alongside CIFAR-10 results, or remove the claim that experiments were conducted on MNIST.
8. **Run at minimum 3 seeds** and report mean ± standard deviation for all accuracy numbers.