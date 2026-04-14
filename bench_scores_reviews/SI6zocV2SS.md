## Summary
CAN (Continuously Adapting Networks) proposes to address catastrophic forgetting by combining Hebbian-learning-based importance scoring with selective neuron freezing and gradient masking. The method computes per-neuron importance scores using local Hebbian updates, selects task-relevant neurons, and freezes their gradients during subsequent task training. Experiments on MNIST and CIFAR-10 with 2-task splits report modest improvements in average accuracy over a vanilla ANN baseline.

---

## Strengths

- **Biologically-motivated, unsupervised importance scoring**: Using local Hebbian activity as an importance signal rather than gradient- or Fisher-based metrics (EWC, SI) is a conceptually distinct approach. Hebbian scoring requires no global error signal, operates on activations, and is in principle usable on unlabeled data — a concrete differentiator from the dominant regularization-based paradigm.
- **Transparent disclosure of limitations**: The paper honestly acknowledges the absence of automatic task gating, the manual mask selection at inference, and the fixed-capacity constraint. Placing these in "Future Scope" rather than hiding them is appropriate; it allows the reader to assess the actual scope of the contribution.

---

## Weaknesses

### Fatal
None that invalidate the core idea in principle.

### Major

- **Critical measurement error in Figure 3**: The y-axis is labeled "Accuracy (Acc)" but displays values in the range 1.9–2.45. These values are consistent with cross-entropy loss, not accuracy (which would be 0–1 or 0–100%). Figure 4 (CAN on TIL) is correctly labeled "loss" with similar value ranges (~2.05–2.23), making it nearly certain that Figure 3's axis is simply mislabeled. This corrupts the primary visualization of the task-incremental training dynamics, and it raises doubts about the integrity of the broader experimental pipeline.

- **Performance barely above random chance**: CAN achieves 27.1% average accuracy on a 5-class CIFAR-10 split where random chance is 20%. Even a trivially small MLP of size 256-128-64 should reach well above 50% accuracy on 5 CIFAR-10 classes without any CL mechanism. A difference of only ~7 percentage points above random is not evidence of meaningful learning. The paper provides no analysis of why accuracy is so low — whether it stems from too aggressive a masking threshold, a flawed importance estimator, an implementation bug, or the architecture being genuinely too small. Until this is explained, the results cannot be interpreted as demonstrating that the method works.

- **No standard continual learning baselines**: The only comparison is against vanilla SGD. EWC, SI, PackNet, HAT, or even naive rehearsal are absent. Given that the paper explicitly cites EWC and SI in Related Work (Sections 2.1 and 2.3) and the key claim is "significantly reduces catastrophic forgetting," the absence of any comparison against these methods makes it impossible to assess whether CAN offers any improvement beyond trivial isolation effects.

- **Evaluation severely limited to 2 tasks**: CIFAR-10 classes 0–4 vs. 5–9 constitutes a single transition, not a continual learning sequence. With only 2 tasks, forgetting is observable only once, and the scalability claim ("can be generalized with N number of tasks") is entirely unvalidated. Section 4.1.1 further notes that the architecture explicitly cannot handle "a continuous stream of data belonging to a variable number of classes," which is the dominant formulation of the problem.

- **Method underspecified for reproducibility**: The paper lacks pseudocode and omits critical implementation details: the gradient scaling rule is never written as an explicit equation; the threshold selection procedure (global, per-layer, percentile-based?) is unspecified; whether Oja's rule (Section 3.1.2) or lateral inhibition (Section 3.1.3) are actually used in experiments is never confirmed; the Hebbian learning rate, batch size, optimizer, and weight decay are all absent. As a result, it is not possible to verify, reproduce, or build on the method.

- **Oracle task identity required at inference**: The paper explicitly states that masks are "manually selecting... during inference" (Section 4.3) and defers automatic gating to future work. This restricts the method to task-incremental learning with oracle task ID — a well-known simplification. While the authors disclose this, the broader paper language ("continuously adapting," "continual learning") overstates the scope of what is demonstrated.

### Minor

- **Equation `L(θ) = L_n(θ_n) ∈ T_n` is not mathematically well-formed**: Equating a loss value to a set-membership relation is meaningless as written. The intended meaning (minimize `L_n(θ_n)` for task `T_n` using the subset `θ_n ⊆ θ`) should be stated precisely.

- **Inconsistency between Figure 1 caption and methodology text**: The Figure 1 caption states "red neurons only receive zero gradients *until they become relevant for the next task*," implying previous neurons can be reused. However, Section 4.2.1 says "it ensures that no neurons selected for the current training were used for prior tasks." These statements directly contradict each other. If neurons can be reused for later tasks, the guarantee of preventing forgetting is compromised.

- **"Time to Stability" is anecdotal**: The claim that Task 2 required 20 epochs vs. 10 for Task 1 is reported without any quantitative stability threshold, no comparison to the baseline's convergence speed, and no systematic measurement. This metric adds no rigorous empirical content in its current form.

### Tiny

- The abstract states "new tasks can be trained without changing parameter weights." In context, previously-trained weights are frozen, but weights for the new task are trained from scratch. The correct phrasing would be "without changing *previously learned* parameter weights."

---

## Nice-to-Haves

- **Ablation study**: Separate the contributions of (a) Hebbian importance scoring, (b) gradient scaling, and (c) hard neuron freezing. Without this it is impossible to know which component drives any observed reduction in forgetting.
- **Mask visualization**: Show which neurons are selected per task, the degree of overlap between task masks, and whether high-importance neurons correspond to meaningful patterns vs. noise.
- **Capacity analysis**: Measure how performance degrades as tasks accumulate and the neuron pool fills up. This is core to the method's practical feasibility.
- **Per-task accuracy curves over the full training timeline**: Plot accuracy on Task 1 *while* Task 2 is being trained to directly verify that forgetting is being prevented (as opposed to both tasks learning poorly from the start).
- **Implement and evaluate the autoencoder gating mechanism**: This would make the system self-contained and remove the oracle task ID dependency.

---

## Removed Points
*These points were flagged for removal; treat them with caution.*

- **"Title overstates the contribution"** (Reviewer 1): This is a style/framing nitpick. The paper's actual scope is clear from reading Section 4.
- **"Introduction claim that ANNs must be retrained from scratch is wrong"** (Reviewer 1): This is a standard rhetorical simplification in the CL literature, not a substantive scientific error.
- **"Dynamic architectures section misclassifies the contribution"** (Reviewer 1): The related work section is contextualizing the broader space; this is a positional quibble, not a substantive flaw.
- **Demand for theoretical proofs** (Reviewer 1): Demanding formal guarantees from an empirical systems-oriented paper imposes a standard not expected in this subfield.
- **"Demand for larger/more diverse datasets if current ones are insufficient"** — kept as a Major weakness because the issue here is not scale per se but that 2-task evaluation is insufficient to demonstrate any CL claim, and the near-random accuracy suggests the method may not be functioning correctly.

---

## Novel Insights

The juxtaposition of unsupervised/local (Hebbian) importance estimation vs. supervised/global (Fisher/gradient) importance estimation for parameter isolation in continual learning is the genuinely interesting conceptual angle in this paper. In principle, a Hebbian importance estimator that requires no labeled data and no backward pass could be advantageous in few-shot or unlabeled-data CL settings. However, this potential advantage is completely undeveloped: the paper never designs an experiment that tests it, and the near-random performance of the current implementation makes it impossible to assess whether the Hebbian estimator is identifying meaningful task structure at all. The key open question the paper leaves unaddressed — *is unsupervised local importance scoring a viable substitute for gradient-based importance when correctly implemented?* — is more interesting than what the paper actually demonstrates.

---

## Suggestions

1. **Fix Figure 3 immediately and audit the entire experimental pipeline**: Confirm whether the accuracy/loss mislabeling is a plot artifact or reflects a deeper implementation error. Then verify that the vanilla baseline achieves expected accuracy (>50% on 5 CIFAR-10 classes).
2. **Diagnose the low accuracy regime**: Systematically ablate the masking threshold to determine whether it is too aggressive (too few neurons active) and report per-task accuracy before and after sequential training to separate "poor learning" from "high forgetting."
3. **Add a pseudocode/algorithm block**: Specify the exact gradient scaling rule (e.g., `∇θ ← H ⊙ M ⊙ ∇θ`), threshold selection, mask storage format, and the status of Oja's rule and lateral inhibition in actual experiments.
4. **Add at least EWC and SI as baselines** on the same 2-task split before expanding to more tasks; this is the minimum needed to show the method is not regressive.
5. **Expand to 5+ task settings** on a standard benchmark (Split CIFAR-100, Permuted MNIST) once performance on 2-task splits is validated.
6. **Resolve the Figure 1 neuron-reuse inconsistency**: Decide explicitly whether previously-used neurons can be reused for later tasks and make the method description, figure caption, and training protocol consistent.

---

**Evaluation summary:**
- **Novelty**: Low — the combination of parameter isolation with importance-based masking is well-trodden; the Hebbian angle is interesting but undeveloped.
- **Technical soundness**: Weak — the method is underspecified, key components (Oja's rule, lateral inhibition) have unconfirmed experimental status, and a central equation is malformed.
- **Empirical support**: Very weak — performance near random chance, a critical axis mislabeling, only 2 tasks, and no established baselines.
- **Significance**: Very limited in current form — without evidence the method works at a basic level, no broader significance can be claimed.
- **Clarity**: Adequate at a high level, but the gap between the conceptual exposition and the actual implementation is too wide for a research publication.