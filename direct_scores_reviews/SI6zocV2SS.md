## Summary
CAN (Continuously Adapting Networks) proposes to address catastrophic forgetting in continual learning by combining Hebbian learning-based importance scoring with selective neuron freezing and gradient scaling. For each sequential task, a Hebbian importance calculator identifies the most activated neurons, which are then selectively trained via gradient hooks; previously trained neurons are frozen via binary masks stored per task. Experiments are conducted on MNIST and CIFAR-10 under two-task task-incremental and domain-incremental settings.

---

## Strengths
- **Hebbian-guided gradient scaling is a concrete implementation choice.** Combining an unsupervised local importance signal (Hebbian activations) with SGD gradient hooks is a specific, implementable design that distinguishes this method from pure regularization-based approaches such as EWC. The idea of letting Hebbian co-activation patterns naturally surface task-relevant neurons—without requiring gradient or Fisher computation—is intuitive and has plausible computational advantages.
- **Dual evaluation of TIL and DIL settings.** The paper makes a deliberate effort to distinguish task-incremental learning (with per-task masks) from domain-incremental learning (Hebbian-scaled gradients without masking) and evaluates both, recognizing they require different mechanisms. This is a sensible structural choice even if the execution is incomplete.

---

## Weaknesses

### Fatal
- **Near-random absolute performance undermines the central claim.** Table 1 reports 27.1% average accuracy for CAN on the CIFAR-10 two-task split (classes 0–4 and 5–9). For a 5-class problem, random chance is 20%. An improvement from 22.5% to 27.1% over vanilla ANNs—both barely above random—does not constitute evidence that the method "significantly reduces the risk of catastrophic forgetting." If the model is not learning the tasks in the first place, the forgetting comparison is meaningless. The loss curves (Figures 4–6) all hover around 2.1–2.3, which is consistent with near-random performance on a 5- or 10-class problem. No explanation is provided for why the method learns so poorly, nor is any investigation conducted (threshold sensitivity, capacity analysis, architecture ablation) that would help diagnose the failure. This directly invalidates the core empirical claim.

- **Figure 3 mislabels loss as accuracy.** Figure 3 is explicitly captioned "accuracy (Acc) on the y-axis (ranging from 2.0 to 2.4)," yet values of 2.0–2.45 are impossible for any standard accuracy metric (bounded in [0,1] or [0,100%]). These are clearly cross-entropy loss values. The same values appear in data tables accompanying the figure. This is not a parsing artifact: the text caption and table headers both use the label "Acc" for values that are manifestly loss. Since Figure 3 is presented as the primary visualization of the baseline's performance in the task-incremental experiment, mislabeled axes make it impossible to interpret the comparison between the two methods.

### Major
- **Oracle task identity at inference makes the system incomplete as a continual learning solution.** Section 4.3 explicitly states: "we are manually selecting the mask." The proposed autoencoder-based gating mechanism is relegated to future work and has not been implemented or tested. Without automatic task identification at inference, CAN does not actually solve the continual learning problem it claims to address; it is a task-aware memorization system. The paper should clearly scope its claims to the task-incremental setting with known task identity, rather than presenting broader continual learning capabilities.

- **No comparison against any established continual learning baseline.** The only baseline is "Vanilla ANNs" (plain SGD finetuning). Standard ICLR-level CL papers compare against EWC, Synaptic Intelligence, PackNet, Progressive Networks, or replay-based methods under matching protocols. Without such comparisons it is impossible to assess whether CAN provides any advantage over well-known methods, especially given the very low absolute performance.

- **The algorithm is insufficiently specified for reproduction.** Critical details are absent: (1) how Hebbian importance scores are aggregated over mini-batches or epochs; (2) how they are normalized before thresholding; (3) what the threshold value is and how it is chosen; (4) whether Oja's rule or vanilla Hebbian updates are actually used in experiments; (5) the exact gradient scaling formula (described only as "scaled by the locally received feedback"); (6) whether lateral inhibition is implemented and if so how; (7) optimizer hyperparameters, batch size, weight decay, learning rate value, and scheduler type. The paper mentions seed 720 for reproducibility but without these details, the seed alone cannot enable reproduction.

- **Sigmoid output activation for multi-class classification.** Section 4 states that all experiments use "Sigmoid activation at the end" on a network classifying 5 or 10 classes. Sigmoid applied element-wise to a multi-class output does not produce a proper probability distribution and is a non-standard choice that likely degrades performance. The extremely low accuracy (near random) may be partly attributable to this configuration error. No justification is provided.

- **Single-seed evaluation with no uncertainty quantification.** All results are from seed 720 only. On small benchmarks with modest networks, single-run results carry substantial variance. Given that the improvement over the baseline is 4.6 percentage points (27.1% vs 22.5%), even moderate variance from different seeds could eliminate this margin. No error bars, confidence intervals, or multi-run statistics are reported.

### Minor
- **Equation `L(θ) = L_n(θ_n) ∈ T_n` is mathematically incoherent.** A loss value is not "an element of a task." The intended meaning appears to be that the objective for task $n$ is to minimize $L_n$ over the subset $\theta_n \subseteq \theta$, which should be written explicitly.

- **Two-task evaluation is insufficient to assess scalability.** Testing only two sequential tasks prevents evaluation of cumulative forgetting, capacity saturation (a limitation the paper itself acknowledges), or performance as task count grows. This is not merely a "larger benchmark" request—it is necessary to validate the stated goal of continual learning over task sequences.

- **Standard CL metrics are absent.** No backward transfer, forgetting measure, or per-task accuracy matrix is reported. Average accuracy alone (especially when both values are near-random) does not allow assessment of how much forgetting actually occurs on Task 1 after training on Task 2.

- **"Time to stability" is not formally defined or measured.** Section 5.3 describes this metric conceptually but provides no table, threshold criterion, or formal quantification. Observing that the second task takes 20 epochs to converge vs 10 for the first is stated as a finding but not systematically analyzed.

### Tiny
- **Domain-incremental and task-incremental approaches use fundamentally different mechanisms** (masks used in TIL, no masks in DIL), but this divergence is not explicitly discussed or justified. It is unclear whether the two experiments are testing the same method.
- **The data flow diagram (Figure 2) is too high-level to convey the technical mechanism**; no pseudocode or algorithm box supplements it.

---

## Nice-to-Haves
- A neuron assignment heatmap per task per layer would visually confirm whether distinct sub-networks form or whether neuron selection is degenerate (e.g., always the same neurons selected regardless of task).
- Ablation isolating Hebbian gradient scaling alone versus masking alone versus both combined would clarify which component carries the benefit.
- Comparison of Hebbian importance scores against a random neuron selection baseline would confirm that the Hebbian signal adds meaningful signal beyond random subnetwork allocation.
- Capacity analysis reporting what fraction of neurons are consumed after each task would quantify the practical scalability horizon.
- An investigation of whether Oja's rule and lateral inhibition (described in the methodology but not clearly used in experiments) improve performance over vanilla Hebbian updates.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Title is too broad"** (Harsh Critic): Pure style/naming preference. Not a substantive weakness.
- **"Toneva et al. (2018) is poorly matched as a citation"** and **"Hou et al. (2025) is about information retrieval"** (Harsh Critic): Per review instructions, if the paper cites a reference, we assume it exists and is the author's prerogative to use it. Critiquing citation match quality is a style nitpick without access to verify relevance.
- **"Related work does not discuss specific families of methods"** (Harsh Critic): The related work covers the major CL paradigms at a survey level. While a tighter review situating CAN relative to the closest prior work would strengthen the paper, the absence is not a standalone weakness given that the paper's primary gaps are empirical.
- **"The method cannot handle variable-class continuous streams"** as a weakness (Harsh Critic): The paper explicitly scopes this as a limitation in Sec. 4.1.1 ("One of the constraints of our architecture is that we can't use a continuous stream of data belonging to a variable number of classes"). Criticizing a stated limitation is scope creep.
- **Strength: "Modular design with three components"** (Positive Reviewer): This is a generic architectural description that applies to any modular system. Not a specific distinguishing strength.
- **Strength: "Dual TIL/DIL evaluation demonstrates the method across settings"** — partially removed as a strong claim; retained as a minor positive only because the two settings use materially different mechanisms, limiting the generalizability claim.

---

## Novel Insights
None beyond the paper's own contributions. The concept of using Hebbian co-activation as a task-importance proxy for neuron selection in a subnetwork isolation framework is the paper's stated contribution. The reviews do not surface additional insights not already present in the paper itself. The concerning observation that near-random loss values (~2.2 CE) throughout training, combined with the sigmoid output on a multi-class head, suggests the network architecture may be fundamentally misconfigured—potentially explaining the uniformly poor performance independent of the CL mechanism—is an important diagnostic insight that the authors have not themselves identified or discussed.

---

## Suggestions
1. **Fix the output activation**: Replace sigmoid with softmax (or use logits with cross-entropy loss directly) for multi-class classification on MNIST and CIFAR-10. Investigate whether this alone substantially changes accuracy before attributing low performance to the CL mechanism.
2. **Publish the full algorithm as pseudocode**: Define the exact Hebbian update accumulation (per batch vs per epoch), normalization formula, threshold selection, gradient scaling equation, and mask update rule. This is prerequisite for credibility.
3. **Add standard CL baselines**: Implement EWC and SI as comparisons under the same 2-task protocol. These are straightforward to implement and the community will not accept a CL paper without them.
4. **Implement or remove the gating mechanism claim**: Either implement and evaluate the autoencoder-based gating for automatic task selection, or explicitly restrict all claims to the task-incremental setting with known task identity. Do not present a partially complete system as a continual learning solution.
5. **Report per-task accuracy matrix**: Show accuracy on Task 1 before training Task 2, and after training Task 2, to directly quantify forgetting—the paper's central claim.
6. **Run with at least 5 seeds**: Report mean ± standard deviation. The current single-seed result does not support the word "significantly" in the abstract.
7. **Evaluate on at least 5 sequential tasks**: The current 2-task setting cannot distinguish a method that avoids forgetting from one that simply doesn't have enough tasks to degrade. Split-CIFAR-100 or Permuted MNIST with 5+ tasks would provide a more meaningful evaluation.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Low–moderate. Combining Hebbian importance with subnetwork isolation is a reasonable idea, but it is adjacent to several existing approaches (EWC, PackNet, HAT, prior Hebbian CL work). No clear differentiation from closest prior methods is established. |
| **Importance of research question** | High. Catastrophic forgetting is a central open problem in deep learning. |
| **Claims well supported** | No. The main claim—significant forgetting reduction—is not supported by the near-random accuracy results, mislabeled figures, and single-seed evaluation. |
| **Soundness of experiments** | Poor. Two-task toy evaluation, no standard baselines, single seed, mislabeled metric axes, likely incorrect output activation, no ablations. |
| **Clarity of writing** | Below acceptable. The algorithm is never fully specified; key design choices (Oja's rule vs vanilla Hebbian, lateral inhibition) are described but not confirmed as used; figures mislabel axes. |
| **Value to the research community** | Very limited in current form. The experiments do not demonstrate the method works, let alone that it advances the state of the art. |
| **Contextualized relative to prior work** | Weak. Related work is survey-level; no direct comparison to the methods most similar in mechanism (parameter masking, importance-based freezing, Hebbian CL). |

---

MY FINAL SCORE: <pineapple>2.2</pineapple>