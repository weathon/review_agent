## Summary

CAN (Continuously Adapting Networks) proposes to mitigate catastrophic forgetting in neural networks by combining Hebbian-learning-based importance scoring with selective gradient masking. The method computes per-neuron importance scores from local Hebbian activations, stores a binary mask of the task-specific sub-network after each task, and zeros out gradients for previously frozen neurons during subsequent training. Experiments are reported on MNIST and CIFAR-10 in two-task task-incremental and domain-incremental settings.

---

## Strengths

- **Principled motivation for local importance scoring.** Using co-activation statistics (Hebbian learning) rather than gradient-based Fisher information or path integrals to derive importance scores is a legitimate and underexplored design choice. It is computationally cheaper and task-agnostic, which could be advantageous in an online setting. This distinguishes the approach from EWC/SI-style importance measures in a non-trivial way.

- **Lightweight implementation via PyTorch hooks.** The described strategy of scaling gradients in-place via hooks is simple to implement and compatible with any standard SGD-based training pipeline, which is a practical advantage over methods that require architectural changes (e.g., Progressive Nets).

- **Explicit acknowledgment of key limitations.** The paper openly discloses two unresolved issues — the lack of an automatic gating mechanism and the absence of network-growth support — rather than glossing over them. This intellectual honesty is appreciated, though the limitations are severe enough to constrain the contribution significantly.

---

## Weaknesses

### Fatal

- **The primary results figure (Figure 3) is fundamentally mislabeled.** The y-axis is labeled "Accuracy (Acc)" but the plotted values range from approximately 1.90 to 2.45. Accuracy is bounded in [0, 1] or [0%, 100%]. These values match the cross-entropy loss range visible in Figures 4–6 for the same setup. This means the paper's graphical evidence for catastrophic forgetting in the vanilla baseline is showing *loss*, not accuracy. The actual per-task accuracy trajectory — the central quantity for measuring forgetting — is never correctly reported in the figures. This is not a parser artifact; the data table in the paper explicitly lists values like "Task 1 (Acc): 2.05" at epoch 0, confirming the mislabeling.

- **Reported accuracy is near-random, calling into question the method's basic functionality.** Table 1 reports 27.1% average accuracy for CAN on CIFAR-10 in the task-incremental setting. Each task is a 5-class sub-problem (classes 0–4 and 5–9), so random chance is 20%. The "improvement" over the vanilla baseline (22.5%) is a marginal 4.6 percentage points on top of near-chance performance. The paper never diagnoses why performance is this low — possibilities include a defective Sigmoid output head for multi-class classification (the architecture uses sigmoid rather than softmax for the final layer, with no explanation), insufficient model capacity, or a broken importance-scoring mechanism. Without diagnosis or resolution, the paper's affirmative claims about "significantly reducing catastrophic forgetting" are not supported by evidence.

### Major

- **No comparison to any established continual learning baseline.** Despite citing EWC, SI, replay methods, Progressive Nets, and CLAW in the related work, the paper compares only against vanilla SGD fine-tuning. This makes it impossible to assess whether the proposed method is competitive, redundant with existing approaches, or actually worse. For ICLR, this is a disqualifying omission: a new CL method must be compared to at least a few representative baselines (EWC, SI, and one replay method would be the minimum expectation).

- **Method is severely underspecified and not reproducible.** Key implementation details are absent:
  - How exactly is the importance score computed from Hebbian weights? ("average of all the weights relevant to one particular neuron" is ambiguous: incoming, outgoing, both? absolute or signed? per-batch, per-epoch, or cumulative?)
  - What threshold is used to select neurons, and how is it set (global, per-layer, tuned)?
  - How does lateral inhibition work in practice inside a standard ANN? No equation, no layer description, no confirmation that it is actually used in experiments.
  - How are Hebbian updates integrated with SGD steps — simultaneously, alternately, once per epoch?
  - What are the Hebbian learning rate, batch size, and other hyperparameters?
  - Is Oja's rule actually used in the experiments, or only described as background?
  No pseudocode or formal algorithm is provided. The method cannot be independently reproduced.

- **Only two tasks evaluated, undermining the "continual" claim.** The entire experimental section uses a single two-task split of MNIST or CIFAR-10. A method claiming to enable continual learning must be evaluated on at least 5+ tasks (e.g., Split-MNIST with 5 binary tasks, Permuted-MNIST with 10+ permutations). The paper itself acknowledges the method cannot handle variable-class streams (Section 4.1.1), which further restricts the setting. It is unknown whether the sub-network allocation strategy degrades or saturates beyond two tasks.

- **Inference requires oracle task identity (manual mask selection).** Section 4.3 explicitly states: "Currently, to analyze the performance of the model, we are manually selecting the mask." This means the evaluation is performed in a task-incremental setting with oracle task identity — a strictly easier setting than most practical CL deployments. The paper frames the gating mechanism as "future scope" but it is architecturally necessary for the method to function without human intervention. The proposed autoencoder-based gating is unevaluated and speculative.

- **No ablation study.** There is no experiment isolating the contribution of Hebbian importance scoring versus simple alternatives (e.g., activation magnitude, gradient norm, random sub-network selection), nor any comparison of gradient scaling versus hard binary masking, nor any test of whether Oja normalization changes results. Without ablations, the specific claimed contribution — Hebbian scoring as the key ingredient — is unsubstantiated.

### Minor

- **No standard CL evaluation metrics.** Backward Transfer (BWT) and Forward Transfer (FWT) are standard metrics in continual learning evaluation and directly quantify forgetting and transfer. The paper reports only average accuracy (correctly defined in Eq. 1), but this metric alone cannot distinguish between low forgetting and low learning. BWT in particular would directly measure the paper's central claim.

- **Single-seed evaluation.** All results use seed 720 only. For a two-task setup on MNIST/CIFAR-10, variance across seeds is small, but reporting mean ± std over 3–5 seeds is standard for credibility.

- **Architectural configuration unexplained.** The network uses a Sigmoid activation at the output for what appears to be a 5-way classification problem. Sigmoid with multi-label interpretation is non-standard for this task. The loss function is never stated. The very low absolute accuracy (27%) is at least partially attributable to this design choice, which is neither justified nor ablated.

- **Informal language throughout.** Phrases such as "really good at learning patterns," "very easily explodes," and "the concept of a growing network comes into the picture" are informal and weaken the precision of technical claims.

### Tiny

- The paper's constraint about variable-class streams (Section 4.1.1) is acknowledged but not integrated into the abstract or contributions framing, creating a gap between the headline claims and the actual scope.
- The time-to-stability subsection (Section 5.3) reports "the second task needed 20 epochs to converge" with no operational definition of stability, no plot, and no baseline comparison — it contributes no meaningful evidence.

---

## Nice-to-Haves

- **Implement and evaluate the proposed autoencoder-based gating mechanism.** The paper outlines a concrete approach (reconstruction error from per-task autoencoders). Evaluating this would convert the method from a proof-of-concept with oracle task identity to a practically usable CL system.
- **Visualize sub-network masks per task.** Heatmaps showing which neurons are selected for each task would reveal whether the method discovers meaningful, non-overlapping partitions or degenerates to overlapping subsets — providing insight into the method's actual behavior.
- **Capacity and overlap analysis.** Report what fraction of neurons are consumed per task and when capacity is saturated. This would quantify the method's scalability limits directly.
- **Comparison of Hebbian importance vs. simpler alternatives** (activation magnitude, gradient norm). This would both motivate the design choice and potentially yield insight about the value of local learning signals for CL.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **[Removed: Title scope criticism]** The harsh critic argued the title "CAN - Continuously Adapting Networks" is too broad. This is a style/framing nitpick rather than a substantive technical concern.
- **[Removed: Zenke et al. citation placement]** Reviewer 1 noted that Zenke et al. (SI) is cited under meta-learning rather than regularization. This is a minor citation organization issue and not a substantive weakness.
- **[Removed: Missing related works]** Reviewers mentioned PackNet, Progressive Neural Networks, HAT, GEM, iCaRL, etc. as missing comparisons. Per review policy, missing related work is not included because we cannot confirm their relevance without external sources.
- **[Removed: Storage cost of masks]** Reviewer 1 flagged mask storage overhead as a missing analysis. For a small network (256-128-64 hidden units) with binary masks, the storage is negligible and this is not a meaningful concern at this scale.
- **[Removed: Broader impact absent]** The paper does not include a formal broader impact statement. This is not a content weakness at ICLR.
- **[Strength weakened: "well-written" / "topic is important" type generic claims]** All three reviewers made general statements about the paper addressing "an important problem" in continual learning. Catastrophic forgetting being important is not a specific strength of this paper.

---

## Novel Insights

The one potentially interesting observation not fully developed by the paper is the use of *local, unsupervised* importance signals (Hebbian co-activation statistics) as a substitute for *global, supervised* importance measures (Fisher information, path integrals) for parameter isolation in CL. If this idea were implemented rigorously and shown to work comparably to gradient-based measures, it would reduce the computational overhead of importance estimation and potentially generalize better to online/streaming settings where gradient statistics are costly. However, the paper does not provide the evidence to confirm this — the method as implemented does not outperform vanilla SGD by a meaningful margin on any standard benchmark, so the potential insight remains speculative.

---

## Suggestions

1. **Fix Figure 3 immediately** — plot true classification accuracy (not loss) on the y-axis, and verify all other figures are correctly labeled. Report per-task accuracy before and after sequential training, not just loss curves.
2. **Diagnose and fix the near-random accuracy.** With 27.1% accuracy on a 5-class subproblem, debug the output layer (replace Sigmoid+? with Softmax+cross-entropy), verify training converges for each task individually, and confirm that the Hebbian masking is not inadvertently zeroing out too many neurons.
3. **Add at least three CL baselines.** EWC (Kirkpatrick et al., 2017), SI (Zenke et al., 2017), and fine-tuning without any CL method are the minimum required to position the contribution. Use established benchmark protocols.
4. **Provide a formal algorithm box** specifying (a) the Hebbian score computation step-by-step, (b) how the threshold is applied, (c) how the mask is formed and stored, (d) how gradient hooks interact with standard backprop, and (e) where Oja's rule and lateral inhibition appear in the pipeline.
5. **Extend to 5+ task experiments** (e.g., Split-MNIST with 5 binary tasks, or 5-task Split-CIFAR-10). This is required to make any claim about continual learning.
6. **Report BWT and FWT** alongside average accuracy to separately quantify forgetting and forward transfer.

---

## Evaluation on Key Axes

| Axis | Assessment |
|---|---|
| **Originality** | The concept of using Hebbian importance scores for gradient masking has a plausible niche between parameter isolation and Hebbian-inspired dynamic networks. However, the combination is not clearly differentiated from closely related prior work, and the novelty claim is not empirically substantiated. Originality is *low to moderate*. |
| **Importance of research question** | Catastrophic forgetting is a genuine and important problem. The research question is well-motivated. |
| **Claims well-supported** | Claims are *not well-supported*. The central claim — "significantly reduces catastrophic forgetting" — rests on a mislabeled figure and near-random accuracy results. |
| **Soundness of experiments** | *Poor*. Two-task setup, no standard baselines, mislabeled metrics, single seed, near-random performance, and no ablations collectively render the empirical section unreliable. |
| **Clarity of writing** | *Below standard for ICLR*. The methodology section is too vague to be reproducible, key algorithmic steps are missing, and informal language weakens precision. |
| **Value to the research community** | *Minimal in current form*. No reproducible algorithm, no competitive results, and no new benchmark contribution means the community cannot build on this work as presented. |
| **Contextualization relative to prior work** | *Insufficient*. The related work survey is descriptive rather than comparative, and the experimental section does not situate CAN's performance relative to any existing CL method. |