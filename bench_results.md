# ICLR Benchmark Results

Date: 2026-04-13 23:12
Critic/Merger: openai/gpt-5.4 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## SI6zocV2SS

- GT: Reject (avg 1.5)
- Predicted: Reject (2.3/10)
- Match: YES

### Final Review

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

---

## slZZnzlITo

- GT: Reject (avg 4.8)
- Predicted: Accept (5.0/10)
- Match: NO

### Final Review

## Summary
This paper introduces Multimodal Open Set Recognition (MMOSR), a new task extending OSR to multimodal data settings. The authors empirically identify a "fusion degradation" phenomenon where naively combining multimodal fusion with OSR regularization degrades both closed-set accuracy and unknown rejection ability. To address this, they propose the Multimodal Representation Reactivation Network (MRN), combining bidirectional cross-attention (mutually enhanced fusion) with a Mixture-of-Experts module (adaptive fusion). Experiments on four datasets spanning image-text, audio-visual, and RGB-depth modalities demonstrate competitive performance over multimodal fusion and single-modal OSR baselines, with gains up to 5.23% OSCR.

---

## Strengths

- **Concrete empirical diagnosis of a real failure mode.** Table 1 shows a clear degradation pattern where Fusion-OSR underperforms both Fusion alone and single-modal OSR (AUROC drops of 0.01–5.60 points depending on setting), providing direct motivation for a specialized approach rather than simply asserting that naïve combination fails. This is a useful and specific insight for the community.

- **Multi-modal, multi-dataset scope.** Evaluating across three fundamentally different modality pairs (image-text on Food-101/Flower-102, audio-visual on CREMA-D, RGB-depth on SUN RGB-D) with matched backbones is more thorough than most OSR works, which tend to be single-modality and single-dataset in their primary evaluation.

- **MRN as a plug-in backbone.** The paper shows that MRN improves performance not only standalone but also when combined with existing OSR methods (ARPL-MRN, CSRR-MRN outperform ARPL-ADD, ARPL-CAT, ARPL-GQA consistently), suggesting the fusion design is genuinely beneficial rather than a standalone artifact.

- **Robustness across openness levels.** Table 3 systematically varies known class count from 5 to 80, showing MRN maintains consistent gains over multimodal baselines at all openness levels — more thorough than a single fixed-split result.

---

## Weaknesses

### Fatal
None. The core direction is sound and the empirical results, while imperfect, are not fabricated or contradictory at a fatal level.

### Major

- **The motivating "fusion degradation" experiment (Table 1) is built on a single, weak fusion baseline.** The diagnostic uses only addition-based fusion + max-softmax (OpenAUC) on Food-101. The paper never tests whether more sophisticated fusion methods — including GQA, TMC, or MLA — also suffer degradation when combined with OSR. If cross-attention-based fusion (like GQA) combined with ARPL does not degrade, the entire premise collapses and MRN is solving a self-imposed problem. This is the most critical gap in the paper: the problem must be demonstrated on the same class of methods the solution is targeting.

- **No statistical significance — many gains are within noise range.** Results are reported as single-run point estimates with no standard deviations across seeds. Critically: SUN RGB-D gains are 0.37 AUROC / 0.01 OSCR; CREMA-D shows MRN *underperforming* MLA by 1.05 AUROC and 0.18 OSCR; Food-101 gains over MLA are 0.72 / 1.38. Without error bars over at minimum 3 seeds, none of these small margins can be claimed as reliable improvements. The claim "(1) MRN consistently demonstrates exceptional MMOSR performance across various datatypes" is contradicted by MRN's own underperformance on CREMA-D.

- **The MoE (adaptive fusion) module is never ablated.** Table 4 ablates only the cross-attention directions C₁ and C₂ within the mutually enhanced fusion module, while starting from a baseline that already includes the adaptive fusion. The contribution of the MoE module — which is framed as addressing "limited representation capability" from fusion degradation — is therefore unvalidated. Without an ablation row removing the MoE and replacing it with a simple MLP, it is impossible to know whether the gains come from the MoE structure, raw parameter capacity, or the cross-attention modules alone. This is a significant omission given that MoE with 15 experts is the architecturally heavier component.

- **CREMA-D underperformance is unexplained.** On the audio-visual dataset, MRN scores 66.78 AUROC / 57.32 OSCR versus MLA's 67.83 / 57.50 (MLA is best). This directly contradicts the claim of "consistent" superiority. The paper provides no analysis of why MRN fails on this modality pair, which raises questions about whether the fusion degradation framing generalizes to audio-visual settings or whether the cross-attention design has specific failure modes.

### Minor

- **No quantitative measurement of fusion degradation.** The core diagnostic evidence relies on t-SNE visualization (Figure 2), which is inherently qualitative and can be misleading in high-dimensional settings. Quantitative measurements — intra-class variance, inter-class margin, per-modality feature norm/activation statistics before and after OSR regularization — are needed to rigorously establish that degradation occurs and that MRN alleviates it.

- **Threshold protocol underspecified.** Section 4.3 states "The threshold τ is set to ensure 95% of the known samples are correctly classified" but does not clarify on which split this is calibrated, whether any validation set is used, or whether the threshold is fixed across all random test folds. Given that OSCR (unlike AUROC) is threshold-sensitive, this needs explicit protocol description.

- **Ablation table metric mismatch.** Table 4 reports AUROC and ACC only, while the paper's primary claim and Table 2 emphasize OSCR as the main metric. The ablation should report OSCR to be internally consistent with the evaluation framework.

- **Grad-CAM comparison targets the wrong baseline.** Figure 7 compares MRN against ARPL (a single-modal OSR method), rather than against Fusion-OSR. To demonstrate that MRN specifically recovers representations suppressed by fusion degradation, the comparison must be MRN vs. Fusion-OSR, showing what the reactivation mechanism concretely recovers over the problem configuration it is designed to fix.

- **Cross-attention equation (Eq. 1) notation is nonstandard.** The formula `Softmax(W₁^Q z₁ z₂ W₁^K / √d)(W₁^V z₂)` appears to conflate query-key projection and score computation in a way that does not match standard cross-attention formulations. The missing transpose and projection ordering ambiguity hinder reproducibility. This may be a typographical artifact of PDF extraction, but should be verified.

### Tiny

- Table 3 uses "CSSR" while Table 2 uses "CSRR" — one is a typo.
- The text in Section 3.2 says Fusion-OSR causes unknown samples to "closely resemble known clusters," but the extracted figure caption for Figure 2(d) describes unknowns as "more dispersed again." This internal inconsistency should be clarified (likely a figure-labeling confusion between Figures 1 and 2).

---

## Nice-to-Haves

- **Capacity-matched MLP control.** Include an ablation replacing the MoE with a single MLP of equivalent total parameter count. This isolates whether the MoE *structure* (specialized experts with routing) matters versus pure capacity increase.

- **Computational cost analysis.** A brief table of parameter counts and inference latency for MRN vs. strongest baselines (MLA, TMC) would address practical viability concerns for the embedded/robotics use case in the introduction.

- **Ensemble of single-modal OSR models as baseline.** A simple prediction-ensemble of per-modality OSR classifiers is a natural competitive baseline that the paper does not include. Demonstrating that MRN beats this would more directly justify the fusion-based approach.

- **Theoretical or geometric intuition for fusion degradation.** Even a loss-landscape sketch or mutual information analysis showing why cross-entropy + OSR regularization conflicts with multimodal fusion alignment would strengthen the motivation beyond the single empirical observation.

- **Failure case analysis.** Showing examples where MRN still fails (misclassified unknowns, incorrect rejections of knowns, audio-visual cases) would delineate the method's scope and guide future work.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Criticism that multimodal OOD detection literature is absent as a baseline.** The paper cannot be expected to compare against every adjacent setting; the OSR vs. OOD detection distinction is well-established in the field. Absence of MCM-style baselines is at most a nice-to-have, not a flaw undermining the contribution.

- **Criticism that OSR baselines (ARPL 2022) are "dated."** The paper includes CSRR (TPAMI'23), ASH (ICLR'23), MLA (CVPR'24), and OpenAUC (NeurIPS'22), which is a reasonable suite spanning the last three years. The criticism that ARPL is outdated ignores that it remains a relevant strong discriminative baseline and is used in combination experiments specifically.

- **Criticism about missing theoretical proof for fusion degradation.** This is an empirical systems paper proposing a new task and method. Requiring formal theory is outside the standards of the OSR/multimodal community for this type of contribution.

- **Criticism of Figure 1 as a "conceptual cartoon."** All motivation figures are conceptual illustrations; this is standard practice. The figure is appropriate for its purpose.

- **Demanding multi-modality-count scalability analysis (3+ modalities).** The paper explicitly scopes to dual-modality for the main contribution and notes the pairwise extension. Criticizing the absence of a 3-modality experiment is scope creep.

- **Criticism that CLIP/CoOp/MaPLe comparison is "unfair."** The direction of any potential unfairness benefits the baselines: CLIP uses far larger pretraining data than MRN. That MRN outperforms these pretrained models despite less supervision is, if anything, a stronger result.

---

## Novel Insights

The most genuinely valuable insight in this paper — which neither sub-reviewer fully articulated — is that the failure of Fusion-OSR is *directional*: fusion improves closed-set ACC (by pooling modality information) while simultaneously *hurting* unknown detection (AUROC decreases vs. best single-modal baseline in most settings of Table 1). This asymmetric failure pattern is a specific and actionable diagnostic: it suggests that OSR regularization and multimodal alignment objectives are geometrically at odds in the representation space. The cross-attention + MoE combination addresses this by maintaining modality-specific discriminative signals through the cross-attention read-out, while the MoE provides diverse representation pathways that resist the homogenization imposed by OSR compactness constraints. Whether this mechanism is genuinely the cause of the fix or whether the improvement comes simply from capacity and better optimization remains unresolved by the ablations, but the directional failure observation itself is a meaningful and testable hypothesis for the community.

---

## Suggestions

1. **Replicate Table 1's diagnostic with stronger fusion methods (GQA, MLA combined with OSR).** If GQA+ARPL and MLA+CSRR also degrade relative to their fusion-only counterparts, the fusion degradation claim becomes substantially more credible and general. If they don't, revise the problem framing accordingly.

2. **Add a full ablation row removing the MoE (replace with single MLP, matched parameters).** This is the single most important missing experiment. Results in Table 4 should include: encoder-only, encoder + single MLP, encoder + MoE only (no cross-attention), and the full MRN.

3. **Report mean ± std over at least 3 random class splits/seeds for all main results.** Given margins of 0.01–1.38 OSCR on some datasets, this is necessary to make any quantitative claim credible.

4. **Add a quantitative degradation measure.** Report intra-class feature variance and mean pairwise inter-class distance for (a) Fusion-only, (b) Fusion-OSR, (c) MRN — this directly validates the core diagnostic claim with a single table.

5. **Explain and analyze CREMA-D underperformance.** Analyze why audio-visual data breaks the pattern, whether the cross-attention design is ill-suited to asynchronous audio-visual sequences, and what modification could address it.

6. **Fix Eq. 1 notation.** Verify that the cross-attention formula matches the implementation; provide explicit tensor shapes in an appendix to support reproducibility.

---

**Evaluation summary:**
- *Novelty:* Moderate — formalizing MMOSR as a task is useful; the architectural components (cross-attention, MoE) are standard, but their combination is purposefully motivated.
- *Technical soundness:* Weak-to-moderate — the core diagnostic experiment is insufficiently thorough, the ablation is missing a key component, and the cross-attention equation has notation issues.
- *Empirical support:* Moderate — multi-dataset evaluation is a strength, but lack of statistical testing and the CREMA-D underperformance weaken the empirical case materially.
- *Significance:* Moderate — MMOSR is a real and important problem; the paper opens a research direction more than it closes one.
- *Clarity:* Acceptable — the paper is generally readable, but internal inconsistencies between Figure 2 labels and prose, and between the ablation metrics and main evaluation metrics, detract from overall coherence.

---

## E6kQ51yfAj

- GT: Reject (avg 4.5)
- Predicted: Accept (5.4/10)
- Match: NO

### Final Review

## Summary

This paper proposes Game-theoretical Preference Optimization (GPO), which frames LLM alignment as a two-player zero-sum game between an adversarial agent (generating challenging prompts) and a defensive agent (generating safe responses). The adversarial agent is equipped with diversity rewards (SelfBLEU and sentence-embedding-based) to prevent collapse onto a narrow attack distribution. The authors prove that an idealized variant of the iterative algorithm converges to an ε-approximate Nash equilibrium at rate O(T^{-1/2}), and demonstrate empirical gains over RLHF and static baselines on safety and jailbreak benchmarks.

---

## Strengths

- **Adaptive prompt distribution as a principled RL objective.** Unlike prior two-player alignment work (SPIN, self-play reward-hacking mitigation) that fixes the prompt set and lets the two agents compete on responses, this paper makes *prompt generation itself* a learned RL objective. The max-min formulation in Eq. (3.1) is a clean formalization of this idea, and it addresses a real gap in standard RLHF where D_PPO is static and cannot target the current defender's blindspots.

- **Diversity mechanism with clear empirical motivation.** The combination of SelfBLEU and semantic-embedding diversity rewards is concretely motivated and Table 2 / Figure 2 provide unambiguous evidence that without diversity rewards the adversary collapses to a narrow, high-toxicity distribution—reducing its utility as a training partner for the defender. The ablation over diversity intensity (k ∈ {0,1,5,10}) provides useful practical guidance.

- **Convergence analysis with FTRL grounding.** The no-regret/FTRL argument connecting Algorithm 1 to Nash equilibrium convergence is technically sound for the idealized variant. Leveraging the fact that the zero-sum objective is linear in both π and μ (enabling the minimax theorem and CCE→NE reduction) is the right approach. This is more principled than most heuristic adversarial training papers.

- **Attack transfer evaluation.** Testing the trained adversarial agent on three *held-out* target models (Llama-2-7b-chat, Vicuna-7b, RLHF model) rather than only the opponent it trained against is a strong experimental design choice. GPO+Div consistently achieves higher ASR with competitive or better diversity than RLHF+Div, suggesting the game dynamic—facing a stronger, co-evolving defender—induces better generalization in the attacker than training against a fixed weaker opponent.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison against MART (Ge et al., 2023), the most directly comparable baseline.** The paper explicitly calls out MART in related work ("MART iteratively conducts red teaming and safety enhancements but relies on supervised fine-tuning, which makes it difficult to balance the capabilities of attackers and defenders") and positions GPO as its successor with RL-based co-training. Yet MART is never included in Tables 1–3. Without this comparison, it is impossible to determine whether GPO's gains stem from the *game-theoretic RL co-training* or simply from doing any form of iterative red-teaming with an RL objective. This is the central empirical gap.

- **Cosine similarity formula in Eq. (3.5) appears incorrect.** The denominator is written as ‖φ(x)‖² ‖φ(x′)‖², whereas cosine similarity requires ‖φ(x)‖ · ‖φ(x′)‖ (product of norms, not sum of squared norms). If the formula as written is what was implemented, the embedding diversity reward is not measuring cosine similarity but something else entirely, which would affect training dynamics and the interpretation of diversity results in Table 2. The authors should clarify what was actually implemented and correct the formula if it is a typo.

- **Diversity reward magnitude grows unboundedly with history.** In Eq. (3.5), the sum runs over *all* previously generated attack prompts X. As training progresses, |X| grows, so the diversity reward's scale increases monotonically while the safety reward r(x,y) is bounded. This creates a non-stationary optimization landscape where later-round adversaries are penalized for diversity-deficiency at a much higher scale than early-round adversaries. The paper does not describe any normalization (e.g., dividing by |X|, windowing) or discuss this instability. Combined with the asymmetric step schedule (200 defense / 400 attack), this could explain why "selecting a moderate intensity is found to be more effective"—at high k, unbounded diversity penalty would eventually dominate—but no analysis is provided.

### Minor

- **Theory-practice gap is acknowledged but under-analyzed.** Section 3.3 explicitly states that the theoretical version of Algorithm 1 differs from the practical one in four ways: averaged vs. last-iterate policies, uniform initialization, exact argmax/argmin, and no optimization error. The practical PPO procedure satisfies none of these. The paper treats this honestly, but stops short of discussing whether the NEGap actually decreases empirically during training—reporting this quantity (or even qualitative training curves of both agents' rewards) would substantially strengthen the connection between theory and practice.

- **Adversarial agent action space ambiguity.** Equation (3.1) treats μ_φ as an unconditional prompt generator (x ~ μ_φ(·)), but the experimental setup describes the adversary as *conditionally transforming* original harmful prompts into "similar but more harmful variations." These are materially different setups: the former is open-ended generation, the latter is conditional rewriting. The discrepancy affects the interpretation of the theoretical result (which assumes unconditional generation) and needs explicit reconciliation.

- **Potential reward circularity unaddressed.** Both training rewards and evaluation metrics (ASR, r_safe) appear to derive from the same safety classifier family (Llama-Guard / ToxiGen-based). Whether safety improvements reflect genuine robustness or calibration to that specific classifier is not evaluated. Testing against a second, independent safety classifier or using human judgments for a subset of outputs would substantially increase confidence in the results.

- **Computational cost unquantified.** The paper acknowledges in the conclusion that training two LLM agents simultaneously is a limitation, but provides no wall-clock time, GPU-hour, or total parameter-update comparison against standard RLHF. For readers considering adoption, this information is necessary.

### Tiny

- The claim "as far as we know, our work is the first to investigate two-player games from this perspective" in Section 5 is plausible but should be more carefully qualified, given the breadth of adversarial training, automated red-teaming, and self-play alignment literature.
- Figure 2 reports training curves and diversity-intensity ablation without error bars, making it difficult to assess whether differences between k values are meaningfully larger than training variance.
- No ablation on iteration count T or the specific step schedule (200 defense / 400 attack)—both are critical hyperparameters whose sensitivity is entirely unexplored.

---

## Nice-to-Haves

- **Qualitative prompt evolution examples.** The "natural curriculum of increasing complexity" is a central motivational claim, but no qualitative evidence is provided. Showing how adversarial prompts evolve across game iterations—in terms of attack strategy, phrasing, and semantic coverage—would make this concrete and potentially reveal whether the agent discovers genuinely novel attack strategies or merely rephrase known patterns.

- **Embedding-space visualization of prompt distribution.** A t-SNE/UMAP plot of adversarial prompts at different training iterations, comparing GPO vs. GPO+Div, would directly substantiate the diversity reward's effect on prompt coverage—a cleaner demonstration than the scalar diversity metric alone.

- **Evaluation against qualitatively different attack strategies.** The paper tests on held-out prompt *datasets* (OOD in distribution), but does not evaluate whether the GPO-trained defender withstands qualitatively different attack *strategies* (e.g., GCG suffix attacks, multi-turn manipulation). This would more robustly support the "enhanced generalization" claim.

- **DPO as an alternative defense backbone.** Since DPO avoids RL entirely and hence sidesteps reward hacking, a brief exploration of whether the adversarial prompt generation framework can drive DPO-based defense updates (as foreshadowed in the conclusion) would broaden the paper's impact.

- **Evaluation on a non-safety task.** The conclusion explicitly acknowledges the intent to extend to helpfulness and reasoning; even a single experiment on GSM8K or similar would clarify whether the framework is safety-specific or genuinely general.

---

## Removed Points

*These points were flagged for removal; treat them with caution if revisiting.*

- **No variance estimates / single-run statistics (Harsh Critic).** At the scale of LLM RLHF training, single-run evaluation is standard in the field. Demanding confidence intervals or multiple seeds is not a standard expectation for this setting and does not constitute a meaningful weakness.

- **Weak scope criticism re: broader alignment tasks (Harsh Critic / Spark Finder).** The paper explicitly scopes to safety ("this work primarily focused on prototyping our idea using safety-related tasks") and lists non-safety extension as future work. Faulting the paper for not yet doing experiments in reasoning/helpfulness is scope creep. The overstated "optimal LLM alignments" in the title is noted in Minor above.

- **DPO/SimPO missing as alignment baselines (Spark Finder).** DPO is an off-policy, prompt-distribution-agnostic method. Integrating it as a defense agent is a different systems design requiring non-trivial work. Comparing defense-side performance against DPO as a drop-in is not standard practice here, and the paper acknowledges DPO as a future direction.

- **Missing related work demands.** Per the review instructions, claims about missing related works cannot be verified without external sources and are not included.

- **Formatting and venue-tag concerns.** Not relevant to scientific content.

- **Criticism that the diversity term in the shared payoff is "conceptually odd" for the defender (Harsh Critic).** Inspection of Eq. (3.1) shows that R_div(x) depends only on x, not y. The defender optimizes over y given the prompt x, so for the defender, -β_div R_div(x) is a constant and the effective objective is just maximizing E[r(x,y)]. There is no conceptual oddity.

- **Unfair comparison: RLHF adversary trained against a weaker opponent than GPO adversary (Harsh Critic).** This asymmetry favors RLHF (an easier training environment), not the proposed method. Comparing GPO-adversary (stronger opponent) vs. RLHF-adversary (weaker opponent) puts RLHF at an advantage, making GPO's superior attack transfer a *stronger* result, not a confounded one. This criticism is removed per the review instructions.

---

## Novel Insights

The most genuinely novel observation across the reviews—one not fully discussed in the paper itself—concerns the **interaction between diversity reward non-stationarity and training stability**. Because the diversity penalty in Eq. (3.5) grows with |X| throughout training while the safety reward is bounded, the effective balance between attack aggressiveness and diversity shifts systematically across iterations. This implicit curriculum—early rounds are more aggressiveness-driven, later rounds increasingly diversity-driven—may be a hidden mechanism behind the "moderate intensity is preferable" finding. If true, explicitly scheduling the diversity coefficient (rather than holding it fixed at k) could yield a more principled and effective training procedure. The paper presents this intensity effect empirically but does not recognize it as a consequence of the unbounded sum formulation. Addressing this could simultaneously resolve the normalization issue and improve practical performance.

---

## Suggestions

1. **Benchmark against MART.** This is the single most impactful experiment missing from the paper. Reuse the same training setup and evaluation datasets; report ASR and r_safe for MART's defender and ASR/diversity for MART's attacker.

2. **Fix or clarify Eq. (3.5).** Verify whether the implementation uses standard cosine similarity (divide by ‖φ(x)‖ · ‖φ(x′)‖) or the formula as written (divide by squared norms), and correct the equation accordingly. Include a note in the appendix on numerical behavior.

3. **Normalize or window the diversity reward sum.** Divide Eq. (3.5) by |X| or restrict the reference set to a sliding window of recent prompts to prevent reward magnitude from growing with training length. Report how this affects training dynamics.

4. **Reconcile unconditional vs. conditional adversary formulations.** Add a paragraph in Section 3.1 or Appendix B explicitly describing the conditional prompt-transformation setup used in experiments, and note which theoretical results still apply (or which assumptions they require).

5. **Report empirical NEGap during training.** Add a subplot to Figure 2 showing NEGap(π̂_T, μ̂_T) computed on a held-out set as a function of training steps. This bridges the theory-practice gap and gives concrete evidence that the algorithm is actually converging toward equilibrium in practice.

6. **Add a cross-classifier safety evaluation.** Pick 200–300 examples from the held-out Anthropic set and evaluate defensive agent outputs using a *different* classifier (e.g., if training used Llama-Guard, evaluate with a ToxiGen-based classifier, or with human raters via crowdsourcing). This would substantially strengthen the claim that safety gains are genuine rather than classifier-specific.

7. **Quantify computational overhead.** Report total GPU-hours, wall-clock training time, and peak memory for GPO+Div vs. standard RLHF on the same base model and dataset, so readers can assess practical viability.

---

**Evaluation summary:** The paper introduces a genuinely interesting and novel approach—treating the prompt distribution itself as a learnable adversarial policy co-trained against a defending LLM under an RL objective with diversity constraints. The convergence analysis is technically sound for its idealized scope. Empirically, results are consistent and the transfer-attack evaluation design is strong. However, the paper has meaningful technical issues (cosine similarity formula, diversity reward scaling), a critical missing baseline (MART), and an insufficiently analyzed theory-practice gap. In its current form, the contribution is solid but the evidence base needs to be shored up before it fully supports its claims. The paper's significance is **moderate-to-high** for the safety alignment community, its novelty is **genuine and specific** in the prompt-generation-as-game framing, its technical soundness is **moderate** (pending the formula correction and normalization), and its empirical support is **moderate** (strong design, but key baseline absent).

---

## EW6bNEqalF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.2/10)
- Match: YES

### Final Review

## Summary

This paper studies offline reinforcement learning in episodic Regular Decision Processes (RDPs), where environment dynamics are governed by a hidden finite-state automaton. Building on the ADACT-H framework (Cipollone et al., 2023), the authors propose two contributions: (1) a novel *language metric* $L_X$ grounded in the dot-depth hierarchy of formal language theory, which replaces the $L_\infty^p$ suffix-distance test with a structured family of language-membership events $\mathcal{X}_{i,j}$, and (2) a Count-Min-Sketch (CMS) approach to reduce the memory burden of storing empirical suffix distributions. Theorem 1 establishes an exponential separation between $L_\infty^\ell$-distinguishability and $L_{\mathcal{X}_{2,1}}$-distinguishability on the T-maze family, while Theorems 2 and 3 provide PAC sample complexity bounds for each variant. Experiments on five classical POMDP/RDP benchmarks support the theoretical results.

---

## Strengths

- **Formally grounded and genuinely novel language metric.** The paper introduces Definition 2 (language metric) as a unifying framework that captures $L_\infty$, $L_1$, and prefix distances as special cases of $L_X(p, p') = \max_{X \in \mathcal{X}} |p(X) - p'(X)|$. The two-dimensional hierarchy $\mathcal{X}_{i,j}$ built from basic patterns $\mathcal{G}_i$ and the $C_k^\ell$ operator is a principled and original construction, not a heuristic. This connection between the dot-depth hierarchy of formal languages and state-distinguishability in RDP learning is, to our knowledge, new.

- **Rigorous exponential separation (Theorem 1).** The T-maze family provides a concrete and complete proof that $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$. This is not just an illustration — it is a theorem establishing that sample complexity of prior methods can be exponential in horizon $H$ on a natural class of instances. The mechanism is transparent (the distinguishing event is "probability of positive reward upon North," a single-language membership check).

- **Identification and correction of a proof error in prior work.** The paper discovers an error in the proof of Cipollone et al. (2023), leading to a corrected bound with an additional $\sqrt{H}/\mu_0$ factor affecting both the prior and the present work. Surfacing and fixing such errors is a substantive scientific contribution.

- **Figure 2 provides clean, theorem-consistent empirical validation on T-maze scalability.** The linear vs. exponential scaling of running time and RDP size as corridor length increases is exactly what the theory predicts, and the experiment is quantitatively informative (log-scale plots, 20 runs, up to $N=100$).

- **Five-domain evaluation with a competitive external baseline.** The comparison against FlexFringe—a state-of-the-art algorithm for learning probabilistic-deterministic finite automata—is meaningful because FlexFringe uses domain-agnostic heuristics and sometimes uses cycles (which is favorable to FlexFringe), making the comparison conservative with respect to the proposed method.

---

## Weaknesses

### Fatal
None.

### Major

- **Headline sample efficiency claim is not empirically validated.** The paper's title and abstract center on "sample efficiency via language metrics," and Theorem 3 gives a bound that is polynomially better in $\log|\mathcal{X}|$ when $j$ is constant. However, *no experiment varies dataset size* to show that the language metric achieves better policy quality with fewer episodes. All experimental comparisons are at fixed dataset sizes. This is the single largest gap: the central theoretical contribution is not directly tested empirically, and without learning curves over $|\mathcal{D}|$ it is impossible to assess whether the sample complexity improvement is behaviorally meaningful in the tested domains.

- **No direct comparison to ADACT-H or RegORL.** The paper explicitly frames its contribution as improving over RegORL (Cipollone et al., 2023), yet RegORL/ADACT-H (exact, without CMS) does not appear as a baseline in any table or figure. The CMS variant partially fills this role (since it uses the same $L_\infty^p$ test), but the paper should either include the exact ADACT-H where feasible or explicitly argue why CMS is the appropriate stand-in. Without this, the claimed improvement over the precise algorithm being compared to is unverifiable experimentally.

- **No ablation over language hierarchy parameters $(i, j)$.** The hierarchy $\mathcal{X}_{i,j}$ is a key contribution, but only $\mathcal{X}_{3,1}$ is evaluated across all domains. The sensitivity of results to $i$ and $j$ is entirely uncharacterized. Without this, it is impossible to assess whether $\mathcal{X}_{3,1}$ is genuinely appropriate or whether it was selected post hoc, and whether the hierarchy design is well-motivated for domains other than T-maze.

- **Model selection for $(i, j)$ is unresolved and practically critical.** Assumption 1 requires that the behavior policy ensures $L_{\mathcal{X}_{i,j}}$-distinguishability $\geq \mu_0 > 0$ for a *known* $\mathcal{X}_{i,j}$ that is an input to the algorithm. In practice, the learner does not know which $(i,j)$ yields non-trivial distinguishability without oracle knowledge of the RDP. No heuristic, model selection procedure, or sensitivity analysis is provided. This is a genuine practical limitation that the paper should at minimum characterize empirically (e.g., what happens when the wrong $(i,j)$ is chosen on T-maze?).

### Minor

- **The $1/d_m^*$ term can itself be exponentially large in $H$, partially undermining the "overcoming exponential dependence" narrative.** The paper acknowledges this ("The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach") and discusses it after Theorem 3, but this acknowledgment is buried. The narrative in the abstract and introduction emphasizes avoiding exponential dependence on $H$, while the bound retains this exponential dependence through $d_m^*$. A more calibrated framing in the abstract and introduction would better reflect the scope of the improvement.

- **Notation imprecision in the estimator definition.** The paper defines $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$ (Section 4.1), but $\mathcal{X}_{i,j}$ is a *set of languages*, not a single language. The indicator $\mathbb{I}(e \in \mathcal{X}_{i,j})$ does not type-check: $e$ is a trace, $\mathcal{X}_{i,j}$ is a collection of languages. Presumably one computes $\hat{p}_1(X) = \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$ for each $X \in \mathcal{X}_{i,j}$, then maximizes. Since Theorem 3's threshold involves $|\mathcal{X}|$ via a union bound, this is not cosmetic.

- **Hierarchy definition contains a clear typographic error.** Section 4.1 reads: "parameterised by $j$ for the granularity of the atomic symbols, and by $j$ for the sequential composition." One dimension should be $i$. In a section whose contribution hinges on a two-dimensional hierarchy, this confuses the exposition.

- **The corrected proof error produces a bound that is strictly weaker than originally claimed in Cipollone et al. (2023)**, with an additional $\sqrt{H}/\mu_0$ factor in both bounds. The paper does not analyze how this additional factor changes the comparison with the lower bound from the prior work (which depends on $L_1$-distinguishability). The current gap between the upper and lower bounds is wider than previously thought.

### Tiny

- **Notation inconsistency between $L_\infty^\circ$ (used in Section 2.2's distinguishability definition) and $L_\infty^p$ (used in Section 3 and Theorem 2).** These appear to be the same metric under different notation; the paper should reconcile them.

- **The $\varepsilon$-optimality target in Section 2.3** ("finding $\hat{\pi}$ satisfying $V_0^*(h) - V_{\hat{\pi}}^*(h) \leq \varepsilon$ for each $h \in \mathcal{H}_0$") is stated as a per-history guarantee, which is stronger than the expectation-over-$h_0$ definition given earlier in the section. The paper should clarify whether the theorems prove the weaker or stronger form.

---

## Nice-to-Haves

- **Computational complexity of the language-metric test.** The paper emphasizes "tractability" but only formalizes sample complexity. A rough analysis of the per-iteration cost of evaluating $L_X$ in terms of $|\mathcal{X}_{i,j}|$, episode length, and dataset size would make the tractability claim more complete, especially for large action/observation spaces where $|\mathcal{G}_i|$ grows.

- **Characterization of RDP classes well-served by low-$(i,j)$ families.** Theorem 1 establishes the exponential gain for T-maze with $\mathcal{X}_{2,1}$. A broader characterization of *which* RDP structures or behavior policies lead to a bounded $j$ being sufficient would significantly strengthen the paper's scope claim and help practitioners choose $(i,j)$.

- **Evaluation on a domain where $L_X \approx L_\infty^p$ (no structure to exploit).** Showing that the language metric does not regress relative to CMS in unstructured domains would demonstrate robustness and complete the empirical picture.

- **Provide per-domain $\mu_0$ values** under both $L_\infty^p$ and $L_{\mathcal{X}_{3,1}}$, and $d_m^*$ estimates, to ground the theoretical bounds with concrete numbers rather than asymptotic statements alone.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED — parser artifact] Example 2 notation $C_1^0(\mathcal{G})$ vs. 10-letter strings.** The critic flags inconsistency between $C_1^0$ and strings of length 10. Given the plain-text rendering artifacts visible throughout the document (e.g., malformed figure captions), the superscript is almost certainly a rendering error (likely $C_1^{10}$). This is not a paper error.

- **[REMOVED — scope creep] Missing comparison to sequence-model-based offline RL or latent-state RL.** These approaches are mentioned in the introduction but lack sample complexity guarantees for the RDP class; demanding engagement with the dominant empirical paradigm goes beyond this paper's scope.

- **[REMOVED — scope creep] Insufficient engagement with POMDP/PSR literature at a technical level.** The related work discusses PSRs and POMDPs and explains why their bounds don't directly apply. Demanding further technical development of these connections exceeds the paper's stated scope.

- **[REMOVED — non-standard expectation] Demand for formal runtime complexity theorem.** The paper demonstrates tractability empirically (Figure 2) and via informal analysis. Requiring a formal complexity theorem for a systems/algorithmic paper is non-standard at ICLR when runtime is demonstrated empirically.

- **[REMOVED — unfair baseline criticism] FlexFringe comparison disadvantages the proposed method.** FlexFringe uses cycles and performance-optimizing heuristics not available to ADACT-H variants, making the comparison asymmetrically harder for the proposed method. Criticisms that the comparison is "not directly comparable" because FlexFringe sometimes outperforms are not valid weaknesses—this makes the proposed method's wins more impressive and its losses expected.

- **[REMOVED — factual misread] CMS width parameter $w = \lceil \varepsilon/\delta_c \rceil$ claimed unusual.** The critic compares to standard CMS parameterizations using different variable conventions. The paper's CMS description is internally consistent and the proof of Theorem 2 depends on it; without access to the appendix proof, there is no basis to claim incorrectness.

- **[REMOVED — generic] "The paper should more explicitly situate itself relative to mainstream offline RL concerns (coverage mismatch, pessimism)."** The paper adopts the concentrability framework standard in offline RL. Demanding broader situating in offline RL is generic advice not specific to this paper's weaknesses.

---

## Novel Insights

The most significant novel insight—which neither the authors nor reviewers fully develop—is the observation that the language metric $L_X$ implicitly performs *soft aggregation over exponentially many suffix events* by collapsing them onto a polynomial-cardinality family of language-membership events. This is why the testing complexity drops from $O((AOR)^H)$ to $O(|\mathcal{G}_i|^j)$: the language hierarchy replaces exact suffix probabilities with marginal event probabilities, trading discriminative power for statistical efficiency. Whether this aggregation principle can be made adaptive (choosing $(i,j)$ based on data without oracle knowledge) seems like the key open problem, and connects to model selection in automata learning more broadly. A data-driven variant—e.g., incrementally increasing $j$ until the distinguishability test stabilizes—would resolve the main practical limitation and would be a natural extension worth highlighting explicitly.

---

## Suggestions

1. **Add sample efficiency experiments**: Plot policy quality (reward) vs. dataset size $|\mathcal{D}|$ for at least T-maze and one other domain, for both CMS and language metric variants. This is the most direct test of the paper's central claim and is entirely absent.

2. **Add ADACT-H (exact) as a baseline** where computationally feasible (short horizons), or explicitly argue in the text why CMS is the appropriate computational proxy for the original method.

3. **Run ablation over $(i,j)$**: At minimum, test $j \in \{1, 2\}$ on T-maze and Cheese; report reward, automaton size, and runtime. Include a case where a suboptimal $(i,j)$ is chosen (e.g., $\mathcal{X}_{1,1}$ on T-maze) to characterize sensitivity.

4. **Provide a heuristic or discussion for model selection of $(i,j)$**: Even an informal procedure (e.g., validate on a held-out portion of $\mathcal{D}$, or increase $j$ until learned automaton size stabilizes) would substantially improve practical applicability.

5. **Fix the estimator notation**: Define per-language estimators $\hat{p}_1(X) = \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$ for each $X \in \mathcal{X}_{i,j}$, then state $L_X(\hat{p}_1, \hat{p}_2) = \max_{X \in \mathcal{X}_{i,j}} |\hat{p}_1(X) - \hat{p}_2(X)|$.

6. **Recalibrate the abstract and introduction** to clarify that the exponential improvement is (a) instance-dependent on favorable language structure and (b) present when $1/d_m^*$ does not itself dominate.

---

**Evaluation along key axes:**

- **Novelty**: High. The language metric and its connection to the dot-depth hierarchy is a genuinely original idea; the unification of $L_\infty$ and $L_1$ metrics as special cases of $L_X$ is elegant and previously unexplored in the RDP learning context.
- **Technical soundness**: Good. The theoretical framework is well-constructed; the corrected bounds are credible; notation issues and the estimator imprecision are refinements rather than errors.
- **Empirical support**: Moderate. The T-maze scaling experiment is convincing and theorem-consistent. However, the absence of sample efficiency experiments—the paper's core claim—and the lack of ablation over the hierarchy leave the empirical case incomplete.
- **Significance**: Moderate-to-high for the RDP/non-Markovian RL subfield. The language metric idea is likely to influence follow-up work; the paper moves RDP learning meaningfully closer to practical applicability for structured domains.
- **Clarity**: Moderate. The mathematical development is generally rigorous and well-structured; the motivating examples are effective. Several notation inconsistencies and the two typographic errors in the central definitions detract from the exposition of the paper's most important sections.

---

## F6s7OApF0n

- GT: Reject (avg 4.8)
- Predicted: Accept (6.2/10)
- Match: NO

### Final Review

Now I have enough context to write the consolidated review. Let me analyze the paper carefully against all three sub-reviews.---

## Summary

CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) addresses the under-explored problem of optimizing a user-defined utility function over the *trade-off between BO cost (iterations spent) and validation performance*, rather than purely asymptotic performance at a fixed budget. The method contributes three tightly integrated components: (1) a utility-based acquisition function (Eq. 2) that selects configurations and dynamically chooses a variable evaluation horizon to maximize expected utility improvement; (2) a stopping criterion that blends normalized regret and probability-of-improvement; and (3) PFN-based LC extrapolation trained via a novel cross-dataset/cross-configuration mixup strategy. Empirical results across LCBench, TaskSet, PD1, and an author-collected object-detection dataset consistently outperform a strong set of multi-fidelity and transfer-BO baselines, with margins that widen under stronger cost penalties.

---

## Strengths

- **Utility-aware acquisition with variable horizon (Eq. 2):** Rather than evaluating expected improvement of final-epoch performance, the acquisition function scores each configuration by the best expected improvement in utility achievable at *any* future BO step Δt. This is a concrete and non-trivial departure from prior freeze-thaw acquisitions (DyHPO: greedy one-step; iFBO: random future epoch PI), and is well-motivated by the stated objective.

- **Gain magnitude grows monotonically with cost penalty:** Table 1 and Figure 5 show a clear and consistent pattern — at α=0 gains over the next-best transfer baseline (FSBO) are moderate, while at α=2e-4 CMBO achieves roughly 3–6× lower normalized regret. This is the strongest piece of evidence that the utility-aware framework captures something qualitatively different from standard methods, rather than just a surrogate improvement.

- **Mechanistic acquisition analysis (Fig. 7a–c):** The paper shows that configurations selected by CMBO have lower *achievable* future regret (7a), that the optimal horizon Δt/T transitions from large (non-greedy exploration) to small (greedy exploitation) as the BO proceeds (7b), and that under strong cost penalties CMBO concentrates evaluations on a small subset of configurations while baselines over-explore (7c). These analyses directly corroborate the intended behavior of Eq. 2.

- **LC mixup is simple and demonstrably effective:** The two-stage mixup (across datasets with shared λ₁, then across configurations) generates effectively infinite training examples from a finite LC dataset, reducing PFN overfitting (Fig. 6a) and producing measurable BO improvement (Fig. 6b). The key design insight — applying the *same* λ₁ to all configurations to preserve inter-configuration correlation — is a non-obvious implementation choice that the paper justifies.

- **Breadth of experiments:** Three standard multi-fidelity HPO benchmarks with different characteristics (tabular MLPs, NLP tasks, large-scale vision/bioinformatics) plus an original real-world object-detection dataset (500 LCs from RoboFlow100 across 30 tasks) provide unusually broad empirical coverage for an HPO paper.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 3 ablation has duplicate/incorrect rows.** Rows 3 and 4 both display p_b=✓, Acq=✓, T=✓ yet produce very different results (α=2e-4: 4.4 vs 0.9). The text claims "performance improves *sequentially* as each component is added," implying rows 3 and 4 differ in some component. The most likely explanation is that row 3 uses T=✓ *without* mixup and row 4 uses T=✓ *with* mixup, but the T column does not distinguish this. As presented, the ablation is uninterpretable: a reader cannot tell which component drives the 4.4 → 0.9 improvement at high cost penalty. This must be corrected, as it undermines the claimed component attribution.

- **Table 1 mislabels DPL as iFBO.** The table contains two rows labeled "iFBO": one citing "Kadav et al., 2023" and one citing "Kadavourian et al., 2024." However, Kadra et al. 2023 is the DPL paper, not iFBO (which is Rakotoarison et al., 2024). Since DPL appears correctly labeled in Tables 2 and 4, but is absent as a named row in Table 1, the first "iFBO" row is almost certainly a mislabeling of DPL. This makes the primary cost-sensitive comparison table misleading and must be fixed.

- **Preference learning remains largely synthetic and under-analyzed.** The paper positions utility estimation from user preference data as a key enabler, but (a) Fig. 2 and §B only demonstrate recovery under clean, dense synthetic labels; (b) the "Estimated" experiment in Table 2 constructs preferences by assuming "the user wants better tradeoff than iFBO" — a programmatic construction, not actual human preference elicitation; (c) there is no analysis of how many preference comparisons are needed, how sensitive the utility estimate is to noisy labels, or what happens when the estimated utility is misspecified. Given that preference learning is presented as a first-class contribution (abstract, §3.1, contribution bullet), the supporting evidence is insufficient.

### Minor

- **Algorithm 1, line 4 contains a notation error.** Line 4 reads n* ← argmax_{n ∈ C} A(n), where C = {(x,t,y)} is the *set of collected observations*, not the configuration pool. The paper's own text (§3.1) specifies "dynamically select…x_{n*} with n* ∈ [N]", confirming the argmax should be over [N] or X. As written, the algorithm only considers configurations already observed at least once, which would mean new configurations are never explored after the first round — clearly not the intended behavior.

- **Notation inconsistency between ỹ_b and ȳ_b.** Section 3.1 defines BO performance as ỹ_b (best validation performance so far), but Algorithm 1 (line 10) and the stopping criterion (Eq. 3, Eq. 5) use ȳ_b for the same quantity. This is not merely cosmetic — a reader trying to reconcile the stopping rule with the formal definitions is forced to guess which symbol is authoritative.

- **PFN pre-training cost is never discussed.** The paper proposes cost-sensitive HPO yet omits any discussion of the significant offline cost of training the PFN surrogate on LC datasets. For practitioners comparing CMBO to baselines that require no offline meta-training (e.g., BOHB, DPL), this is a real consideration. The paper should at minimum characterize the amortized cost or discuss on which scales the offline training is economically justified.

- **Uniform per-step cost assumption is unmotivated relative to the stated motivation.** The introduction motivates cost-sensitivity with examples from cloud credit usage and Slurm time allocation, where different configurations naturally have different per-epoch costs (varying model sizes, batch sizes, etc.). CMBO's utility U(b, ỹ_b) treats all BO steps as equally costly. The experimental setup uses tabular benchmarks where this is approximately true, but the paper never explicitly flags this as a limitation, leaving a gap between the rich motivating scenario and the implemented formalism. This should be stated as a scope boundary.

- **Acquisition function's variable horizon (Δt) is scored but not executed.** Equation 2 selects the best Δt to maximize expected utility improvement, but Algorithm 1 always advances by exactly one epoch per step. The paper is silent on why the scored Δt is an appropriate surrogate objective for the one-step action. Figure 7b corroborates the intended behavior empirically, but a brief clarification in the text would prevent misreading the acquisition as a commitment to Δt steps.

### Tiny

- **No systematic hindsight evaluation of stopping quality.** Figure 5 shows cherry-picked stopping trajectories; §H provides all tasks but only as a supplementary dump. A summary table of (average stopping step, fraction of runs where stopping utility is within ε of the true optimal utility) would make the stopping criterion's practical reliability concrete rather than relying on aggregate regret numbers.

- **The β sensitivity analysis (Fig. 7d) correctly covers all three benchmarks** (LCBench, TaskSet, PD1 are all plotted), confirming the optimal β ≈ e⁻¹ is consistent. However, this tuning is done over the same test benchmarks that the full method is evaluated on. The paper should clarify whether β=e⁻¹ was selected on held-out validation tasks or whether test benchmarks informed this choice.

---

## Nice-to-Haves

- **Add a trivial cost-weighted baseline.** A one-line modification to iFBO or DyHPO using EI/cost or EI − α·Δb would directly test whether the entire CMBO framework is necessary or whether cost-awareness alone (without the utility-aware acquisition and stopping) achieves similar utility gains. The absence of this baseline makes it harder to isolate the contribution of the framework architecture.

- **Predicted vs. actual LC visualization.** The entire framework depends on LC extrapolation quality (acquisition, stopping, and transfer all rely on it), yet no figure directly compares predicted and actual future LCs. A calibration plot or representative overlays would substantially strengthen confidence in the method.

- **Report actual wall-clock savings alongside epoch counts.** The paper uses "Total Epochs Spent" as the cost axis, which is appropriate for the stated metric. However, showing actual time savings in at least one real-world experiment (Table 4) would make the cost benefits tangible for practitioners who care about GPU-hours, not epoch counts.

- **Test with non-uniform per-configuration costs.** Even a simple synthetic experiment where different architectures or batch sizes induce different per-epoch wall-times would establish whether CMBO's step-based utility can be adapted to heterogeneous-cost HPO, which is the paper's primary motivating scenario.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Zero standard deviations (±0.0) are suspicious"** (Harsh Critic). The paper explicitly states it runs 5 runs for most methods and 30 for high-variance baselines. Transfer methods (FSBO, Quick-Tune†, ESBO) are essentially deterministic given a fixed tabular benchmark — their ±0.0 entries are expected and correctly reported. This is not a bug.

- **"β sensitivity shown only on PD1"** (Spark Finder). Figure 7d plots LCBench, TaskSet, and PD1 as three separate curves in a single subplot, along with their average. The analysis covers all three benchmarks. This criticism is factually incorrect.

- **"Comparison is unfair because baselines don't have utility-aware acquisition"** (Harsh Critic). The baselines are equipped with the regret-based stopping rule from Eq. 3; only the PI component (Eq. 5) is withheld because it is native to CMBO's acquisition. Crucially, the paper's claim is precisely that optimizing for utility gives better utility — evaluating both CMBO and baselines on the utility metric is the right comparison. The asymmetry in acquisition is the thing being tested, not an unfair handicap.

- **"FSBO outperforming multi-fidelity methods raises questions about when freeze-thaw is worth it"** (Harsh Critic). CMBO ultimately outperforms FSBO substantially (Table 1, Fig. 4). FSBO's strong performance at α=0 reflects the value of transfer in the conventional setting; the paper uses this as motivation for combining freeze-thaw *with* transfer. The finding is not contradictory.

- **Scope-creep criticism about lack of continuous search space support.** The paper explicitly operates in the tabular/finite-pool setting, which is a standard HPO evaluation protocol. Requesting extension to continuous BO as a weakness evaluates the paper against an unstated scope.

- **Demand for theoretical guarantees for the stopping criterion.** CMBO is an empirical systems paper; demanding convergence proofs or regret bounds for a heuristic stopping rule is not standard in this community.

---

## Novel Insights

The most genuinely novel insight beyond standard multi-fidelity BO is the **variable-horizon acquisition function** (Eq. 2): by scoring each configuration at the *best* future step for utility improvement rather than at a fixed target epoch, CMBO naturally transitions from non-greedy (large Δt) to greedy (Δt ≈ 0) as performance saturates under cost pressure, without any explicit schedule or annealing. This produces a qualitatively different BO trajectory — one that concentrates resources on fewer configurations as cost becomes dominant — and Fig. 7a–c provide the first mechanistic decomposition of how cost-awareness reshapes the selection policy step-by-step in freeze-thaw BO. A secondary insight is that LC mixup with a *shared* interpolation coefficient λ₁ across all configurations is a principled way to preserve inter-configuration correlations during data augmentation: interpolating datasets uniformly maintains relative performance rankings, whereas per-configuration λ₁ would destroy them. This distinction between dataset-level and configuration-level interpolation is understated in the paper but is a non-trivial design choice with potential applicability beyond this work.

---

## Suggestions

1. **Fix Table 3 immediately.** Add a T=✓(no mixup) row to make the ablation a proper 5-row factorial: ✗✗✗ → ✗✓✗ → ✗✓✓(no mixup) → ✗✓✓(mixup) → ✓✓✓(mixup). The current two identical-looking rows at the bottom make the strongest numerical claim (0.9 vs 5.8 at α=2e-4) uninterpretable.

2. **Fix the DPL/iFBO mislabeling in Table 1.** The row citing "Kadav et al., 2023" should be labeled DPL, consistent with its labeling in Tables 2 and 4.

3. **Fix Algorithm 1 line 4**: Change argmax_{n ∈ C} to argmax_{n ∈ [N]} (or argmax_{x_n ∈ X}) and clarify initialization for configurations with no observed LC yet.

4. **Unify notation** (ỹ_b vs ȳ_b throughout Sections 3.1–3.2 and Algorithm 1). Choose one symbol and use it consistently.

5. **Add a brief analysis of preference learning robustness.** Even a synthetic sweep over noise level in Bradley-Terry labels (e.g., label flip probability 0%, 10%, 20%) would substantially strengthen the preference-learning claim. Report the number of comparisons used in §B.

6. **Add EI/cost modification of iFBO as a baseline** in at least Table 1 or Table 2. This single experiment would definitively answer whether the full CMBO framework or just cost-weighting is responsible for the gains.

7. **State the uniform-cost assumption explicitly** as a boundary condition in §3.1 and §5, and note it as a direction for future extension, particularly since the motivating examples involve heterogeneous evaluation costs.

---

**Summary evaluation:**
The paper makes a **genuinely novel contribution** to multi-fidelity HPO with a well-motivated and practically relevant problem formulation. The utility-aware acquisition and the LC mixup are both concrete technical advances. Empirical support is **strong and consistent**, with especially compelling evidence from the scaling of gains with cost penalty. The preference-learning pipeline and the uniform-cost assumption are the main areas where the paper's framing exceeds its validation. The table errors (Table 1 mislabeling, Table 3 duplicate rows, Algorithm 1 notation) are more than cosmetic: they materially impair the interpretability of the two most important pieces of evidence (the main comparison and the ablation). Fixing these is essential, but the underlying experimental results appear sound. Overall this is a **solid, above-average ICLR submission** whose core contributions are well-supported, pending the corrections above.

---

## HozsY9Gdcl

- GT: Reject (avg 5.0)
- Predicted: Accept (5.8/10)
- Match: NO

### Final Review

## Summary

This paper introduces **Set-MI**, a method that aggregates individual Membership Inference (MI) scores across sets of documents that, by virtue of shared metadata (e.g., creation date, license, source dataset), are expected to be either entirely present or absent from a language model's training data. The authors construct five diverse benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrate that applying Set-MI on top of four existing Individual-MI methods yields a mean AUROC gain of 0.14, with additional ablations characterizing the effects of model size, deduplication, document length, set size, and aggregation strategy under simulated membership noise.

---

## Strengths

- **Principled and natural reframing of the MI problem.** Rather than inventing a new scoring function, the paper identifies a real structural property of training data curation (all-or-none inclusion by metadata category) and shows that exploiting it can markedly improve any loss-based MI method. The insight that data pipelines select by inclusion criteria is well-grounded with concrete examples (e.g., DOLMA containing Reddit data only through March 2023).

- **First set-based benchmark suite for LM-MI.** The five benchmarks span genuinely different structural types of set assumption—temporal (Wikipedia, Arxiv), categorical (Languages, License), and dataset-of-origin (Instructions). Constructing these is non-trivial and the resulting suite is a real contribution to the community beyond the method itself.

- **The deduplication finding is a novel and practically important result.** Figure 3 (right) shows that Set-MI's advantage is substantially larger on models trained on *duplicated* data than deduplicated data, and that this gap is much bigger for Set-MI than Individual-MI. This asymmetric sensitivity is a new and informative observation: deduplication disproportionately destroys the signal that set aggregation exploits, which has direct implications for auditing modern models.

- **Robustness section is proactive about the method's own vulnerability.** Section 6 explicitly attacks the set assumption by simulating noise. The comparison of MAX/MIN/FULL aggregation under different noise scenarios is practically useful. The fact that all three aggregation variants substantially outperform Individual-MI even at 50% noise ratio is a meaningful empirical guarantee.

- **Strong document-length ablation.** Figure 4 (left) cleanly shows that Set-MI's advantage over Individual-MI scales with the length of sampled tokens, with Set-MI exhibiting faster saturation—this characterizes a concrete design choice practitioners can make.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Benchmark statistics are inconsistent between Table 1 and the text.** This is a serious reproducibility concern. Table 1 reports Wikipedia and Arxiv as 1,000 sets / 100,000 documents, but the corresponding text sections each say "we subsample 100 sets with 100 documents per set" (= 10,000 docs). For Languages, the text says "resulting in 130 sets" while Table 1 reports 200 sets; for License, the text again says "resulting in 130 sets" while Table 1 reports 190. (Note: internally, Table 1 is self-consistent at 100 docs/set, so the table numbers are likely correct and the text contains copy-paste errors. The "130" in both Language and License sections appears to be copied from the Instructions section.) These discrepancies must be corrected, because reported AUROCs are averages over these datasets and the evaluation scale needs to be unambiguous.

- **Missing random-grouping control.** The paper never tests whether randomly composed groups of the same size also improve AUROC over Individual-MI. Without this baseline, it is impossible to determine whether gains come from the set assumption specifically (the paper's core claim) or simply from statistical variance reduction via averaging MI scores over any N documents. If random groups yield similar improvements, the contribution reduces to "averaging noise away," not "leveraging shared membership." This experiment is essential to validate the central hypothesis. Authors should run this on at least Wikipedia and Arxiv.

- **Perfect and near-perfect scores on Languages and Instructions are unexplained and potentially confounded.** Languages/LiRA achieves 1.000 AUROC with Set-MI, and Instructions/Min-K% Prob achieves 1.000 for *both* Individual-MI and Set-MI. These results may primarily reflect domain/style distribution shift between included and excluded language groups or instruction datasets—not genuine membership inference from model loss signals. If a model was simply never trained on, say, Swahili Wikipedia, any reasonable metric will detect that. The paper should diagnose whether these "easy" cases are MI successes or distribution-shift detections, since including them in the average AUROC gain inflates the headline number. Domain-restricted averages (e.g., over just Wikipedia and Arxiv) would give a more honest picture of the method's value in hard cases.

- **Ground-truth membership is a proxy, not a verified label.** For Wikipedia and Arxiv, the paper labels membership based on whether a document's creation date is before the Pile's data-collection cutoff. However, creation date ≠ inclusion in the Pile due to crawl incompleteness, filtering, formatting failures, and source-level truncations. The paper partially addresses this in Section 6 by using 13-gram overlap against the actual Pile as a "clean" version—but this validation is only used in the robustness section on Pythia-2.8B-dedup, not to validate the main Table 2 labels. The paper should report what fraction of "member" documents (by date proxy) actually have 13-gram overlap with the Pile, and discuss how large this gap is. If it is small, the concern is minor; if it is large, the Table 2 AUROCs may be measuring something other than MI.

### Minor

- **No variance estimates in Table 2.** Table 2 reports single-point AUROC values (some averaged over multiple models, but without standard errors). For a paper whose central claim is a +0.14 average improvement, some measure of variability—at minimum a standard deviation over the multiple models used for Wikipedia, Arxiv, and License—is needed. The zlib/Instructions result (0.458→0.429, a *drop*) is given one sentence without analysis. This deserves at least a diagnostic: does Set-MI hurt when the base Individual-MI score is below chance?

- **Deduplication and model-size ablations are narrow in scope.** The deduplication analysis uses only Loss Attack on Wikipedia; the model-size analysis uses only Wikipedia. Both are presented as general findings about Set-MI, but demonstrating these effects on a second domain (e.g., Arxiv) would substantially strengthen the generalizability claims.

- **The 30% threshold for MAX/MIN aggregation is not motivated.** Section 6 uses "top/bottom 30%" for MAX and MIN without explanation or sensitivity analysis. Since the paper recommends these variants for practical use, users need to know whether 10%, 20%, or 50% would work equally well. A brief sensitivity sweep is warranted.

### Tiny

- **Correlation of 0.824 between Individual-MI and Set-MI performance (Section 5.1) adds limited insight.** It confirms the intuitive observation that Set-MI inherits the quality of its base method. More useful would be identifying conditions under which Set-MI *fails to improve* Individual-MI, or a scatterplot of gain vs. base AUROC.

- **Recommendation to "select the best aggregation based on prior knowledge about the noise" (Section 6)** is circular: in practice, a user auditing a closed model will not know which class of noise dominates. Even a simple heuristic (e.g., "default to FULL unless you have specific evidence of one-sided noise") would be more actionable.

---

## Nice-to-Haves

- **Score distribution visualizations.** Overlaid histograms of member vs. non-member scores before and after aggregation would clarify whether Set-MI primarily increases mean separation, reduces variance, or reshapes the tails—all of which have different implications for threshold-based use.

- **Threshold-level analysis (precision/recall at fixed FPR).** AUROC improvements of 0.14 are encouraging, but applications like contamination detection or copyright auditing require high precision at low false-positive rates. A calibration plot or precision-recall curve at practically relevant operating points would help practitioners assess whether the gains are meaningful at deployment thresholds.

- **Discussion of soft/inferred sets.** The conclusion acknowledges that metadata availability is a limiting assumption but defers relaxation to future work. Even a brief sketch of whether clustering or semantic similarity could construct "soft sets" would help readers assess the method's scope.

- **Set-level AUROC alongside document-level AUROC.** Because Set-MI assigns identical scores to all documents within a set, the effective number of independent predictions is the number of sets, not documents. Reporting set-level AUROC would be conceptually cleaner and would complement the document-level figures.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Document-level vs. set-level inference confusion" (Harsh Critic, Section 1 and 2).** The paper is transparent that the method assigns a set-level score to all documents in the set for comparability with Individual-MI. This is clearly stated in the method section ("we assign the aggregated score from the set to every element within the set, so that the score is directly comparable with previous methods"). This is a design choice, not a conceptual error.

- **"Black-box assumption is inconsistent because token-level probabilities may not be available" (Harsh Critic, Section 2).** The paper explicitly focuses on the setting where "only the loss score of the target model is available," and all four Individual-MI methods it builds on use token-level log-probabilities, which are standardly exposed by the APIs and open models the paper targets. This is not a contradiction.

- **"Comparisons may be unfair because base methods were not tuned comparably" (Harsh Critic, Section 5.1).** The Set-MI aggregation is a wrapper applied identically to all base methods; there is no differential tuning.

- **"Why only four MI baselines?" (Harsh Critic, Section 2).** Loss, LiRA, Min-K% Prob, and zlib are the standard baselines in LM-MI literature. Demanding more without identifying specific methods that should have been included is scope creep.

- **Demanding theoretical variance-reduction analysis or formal proofs** (Harsh Critic). This is an empirical systems paper and formal analysis is not standard for this type of contribution in the field.

- **"Privacy / ethics section is too thin" (Harsh Critic, Ethical Considerations).** The paper's ethics section is brief but appropriate for a paper whose primary contribution is an auditing tool, not a privacy attack. The concern about group privacy is noted as potentially valid, but not a scientific flaw.

- **"Impracticality for truly closed-source models" (Spark Finder).** All experiments require AUROC evaluation, which requires ground-truth membership labels. Evaluating on a model with completely unknown training data is not possible in a rigorous setting. The paper's choice of models with known (but publicly undisclosed during evaluation) training data is the correct experimental design.

- **"Larger dataset sizes needed"** — The benchmarks cover up to 100,000 documents with multiple models. Dataset size is not a meaningful weakness here.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the **asymmetric interaction between deduplication and aggregation-based MI**: deduplication disproportionately suppresses the signal that set-level averaging exploits, more so than it suppresses individual-document signals. This implies that MI via averaging is sensitive to a specific form of data preprocessing that is increasingly common in modern pipelines, and suggests that the gap between theoretical recoverability and practical MI may widen as deduplication becomes the norm. A complementary insight from the reviews is that *trivially easy* benchmarks (Languages at 1.000, Instructions/Min-K% at 1.000) may be measuring near-domain detection rather than memorization—and the field would benefit from a sharper distinction between these two phenomena in benchmark design.

---

## Suggestions

1. **Run a random-grouping control immediately.** Form groups of the same sizes as the real sets but with randomly assigned documents, and compute AUROC under the same averaging scheme. Report this as a baseline in Table 2. This single experiment either validates the paper's central claim or substantially changes what is being claimed.

2. **Reconcile all numbers between Table 1 and the body text.** The "100 sets vs. 1,000 sets" discrepancy for Wikipedia/Arxiv and the "130 sets" copy-paste error for Languages and License must be corrected with explicit footnotes explaining what was actually run.

3. **Add a diagnostic for the 1.000-AUROC results.** For Languages/LiRA and Instructions/Min-K%, provide perplexity-gap analysis or cross-entropy gap between included vs. excluded categories at the *zero-shot* level to determine whether the separation is membership-driven or purely distributional.

4. **Report 13-gram overlap validation fraction for the main Wikipedia/Arxiv benchmarks,** not just for Section 6's "clean" subset. Even one sentence quantifying label quality (e.g., "X% of date-proxied member documents have ≥1 13-gram overlap with the Pile") would substantially increase confidence in the main results.

5. **Add a sensitivity table for MAX/MIN threshold** (10%, 20%, 30%, 50%) in Section 6 or the Appendix, to allow users to calibrate their choice without guessing.

---

**Overall assessment:** The paper has a simple but well-motivated idea, a valuable benchmark contribution, and empirically solid main results. The headline finding (average +0.14 AUROC) is encouraging, but the absence of a random-grouping control, combined with the presence of trivially-solved benchmarks in the average, means the true effect size in genuinely hard cases is unclear. The benchmark statistic inconsistencies are a fixable but significant reproducibility issue. The paper is in solid shape for ICLR but requires these specific corrections to make the empirical claims fully defensible. On the axes of evaluation: **novelty** is moderate (the insight is clear but the method is simple); **technical soundness** is adequate with the noted gaps; **empirical support** is broad but has the identified holes; **significance** is good for the auditing/governance community; **clarity** is mostly strong, with the exception of the stat inconsistencies and the 1.000-AUROC cases.

---

## lNuGCXxvkn

- GT: Reject (avg 5.2)
- Predicted: Accept (5.5/10)
- Match: NO

### Final Review

## Summary

This paper develops a theoretical framework for kernel ridge and ridgeless regression applied to linear inverse problems governed by elliptic PDEs. The central contribution is showing that the PDE operator (with spectral decay exponent $p < 0$) inherently stabilizes variance, enabling benign overfitting even in **fixed input dimension**, a setting where standard regression yields only tempered or catastrophic overfitting. The authors additionally characterize how inductive bias from Kernel Sobolev Space (KSS) norms affects convergence rates, showing that sufficiently smooth inductive bias renders the rate independent of the specific smoothness parameter $\beta$, with a threshold that surprisingly matches conditions derived in the Bayesian inverse problem literature.

---

## Strengths

- **Benign overfitting in fixed dimension via PDE structure**: The central insight—that the PDE operator acts as a spectral smoother ($p < 0$ attenuates high-frequency variance), enabling benign overfitting where standard regression cannot—is conceptually clean and mechanistically specific. This is not a recycled result; it is a qualitative departure from existing benign overfitting theory, which typically requires high-dimensional or data-dimension-growing settings.

- **Unified non-asymptotic framework spanning both estimator types**: The same bias-variance machinery (Theorems 3.6–3.7) handles both regularized and interpolating estimators. For the regularized case, the paper recovers the minimax optimal rate from Lu et al. (2022), which provides a non-trivial sanity check on tightness; for the interpolating case, it yields the first rigorous upper bound in this inverse-problem setting, covering benign, tempered, and catastrophic regimes.

- **Surprising connection to Bayesian smoothness threshold**: The smoothness threshold on $\beta$ derived from the minimax rate analysis ($\lambda\beta \geq \frac{\lambda r}{2} - p$) matches the condition derived in the Bayesian inverse problem literature (Knapik et al., 2011; Szabó et al., 2013). This cross-paradigm correspondence is a genuinely non-obvious finding that anchors the frequentist result in established statistical theory.

- **Clear operator-theoretic mechanism**: The noise stabilization effect through $\Delta^{-1}$ (the Green's function acts as a smoothing kernel suppressing high-frequency error) is explained clearly in §5, connecting the abstract spectral analysis to an interpretable physical mechanism.

---

## Weaknesses

### Fatal
None.

### Major

- **Benign overfitting is conditional on an assumption not in the main theorem body.** Theorem 4.2's headline claim—that physics-informed interpolation achieves benign overfitting—depends critically on $\rho_{k,n}$ being bounded. This requires sub-Gaussian features, a condition stated only in Remark 6, not in the theorem hypotheses. In the worst case (noted in the same remark), $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$, which can entirely cancel the variance stabilization effect. The paper does not characterize *when* sub-Gaussian behavior vs. the worst case applies for the spectrally transformed kernel $\tilde{K}$, nor does it provide a corollary with explicit sufficient conditions guaranteeing risk convergence to zero. For a paper whose headline is "benign overfitting in fixed dimension," leaving the central conclusion dependent on an uncontrolled quantity is a significant gap. The paper should provide a standalone corollary making the conditions for benign overfitting fully explicit, or characterize precisely when $\rho_{k,n} = \Theta(1)$ holds for the PDE-transformed kernel.

- **Co-diagonalization assumption is strong and insufficiently analyzed.** Assumption 2.2(d)—that $\mathcal{A}$ and the kernel covariance operator $\Sigma$ share the same eigenbasis—is the linchpin of all spectral analysis. Remark 2 acknowledges this and cites the torus + shift-invariant kernel case (justified by Bochner's theorem). However, the paper does not discuss what happens when this alignment breaks down even approximately: since the variance stabilization mechanism ($p < 0$ in the transformed spectrum $\tilde{\Sigma} = \mathcal{A}^2\Sigma^\beta$) relies entirely on the product of the two spectra being well-ordered, any significant misalignment could destroy the effect. Most practical PDE domains (e.g., irregular geometries, non-uniform sampling, Dirichlet boundary conditions on non-toroidal domains) do not satisfy this assumption. The contribution's practical reach is much narrower than the paper's framing implies without this discussion.

- **Experimental scope is insufficient for the paper's claims.** All experiments use a single PDE (2D Poisson equation, $p = -1$), one domain, and one ground truth function. The paper's central claim involves the parameter $p$, yet $p$ is never varied empirically. Similarly, no experiment directly tests the spectral parameter $\beta$ in the kernel setting, and no log-log convergence rate plot is provided to verify whether the predicted rates ($n^{-\lambda(\beta'-r)/(2p+\lambda\beta+1)}$) are reflected in practice. The activation-smoothness experiment is only a coarse proxy for $\beta$ and validates qualitative monotonicity, not the quantitative rate. At minimum, the paper needs: (1) at least one direct kernel-level experiment validating the bounds in their native setting (the entire theory is for kernel estimators, not neural networks), and (2) experiments varying the PDE order $p$ to support the mechanism claim.

- **Theory-to-experiment gap: kernel theory validated only by neural network experiments.** The experiments use PINNs (neural networks), while the theorems cover kernel estimators. The paper frames this as validating findings "beyond kernel methods," but without establishing an NTK-type equivalence or providing any kernel experiment, the experiments do not validate the theoretical guarantees themselves. This is a structural problem: a reviewer cannot assess whether the theory is tight from these experiments.

### Minor

- **The role of $\rho_{k,n}$ in worst-case behavior is underdeveloped.** Even granting Remark 6, the paper does not show that the worst case $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$ is avoidable in practice, nor does it identify structural properties of $\tilde{K}$ that preclude it. Without this, Theorem 4.2 cannot be used as a reliable guide to when benign overfitting holds.

- **No matching lower bounds for the interpolating regime.** Without lower bounds for the min-norm interpolator, it is unknown whether the variance bound in Theorem 4.2 is sharp or a loose upper bound. For the regularized case, the match with Lu et al. (2022) provides external validation of tightness; for the interpolating case, no such validation exists.

- **Notation is very dense and some operator relationships are underexplained.** The paper introduces many simultaneous operators ($S, S^*, \hat{S}_n, \Sigma, \mathcal{L}, \phi, \psi, \Lambda_{\mathcal{XY}}, \tilde{\Sigma}, \tilde{K}$) without a compact summary of their relationships. The relationship between $\phi$ (mapping to $\ell_2^\infty$ using $\sqrt{\lambda_i}$) and $\psi$ (without the eigenvalue scaling) is confusing because both map from $\mathcal{H}$ to sequences but serve different roles in the operator expressions. A diagram or summary table of operators in an appendix would substantially aid comprehension of Theorems 3.6–3.7.

- **The practical guidance on activation smoothness is heuristic.** §4.3 draws a prescription that "higher-order PDEs require smoother activation functions." While intuitive, the bridge from KSS theory (infinite-dimensional kernel) to finite-width PINN activations is not established theoretically—it requires an NTK-type argument that is not provided. This should be explicitly labeled as a heuristic/empirical conjecture rather than a theorem-derived prescription.

### Tiny

- **Assumption 2.2(a) (bounded outputs) and the Gaussian noise model in §3 are technically incompatible.** Almost surely bounded $y$ and $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ cannot simultaneously hold. This is a minor inconsistency common in the literature and does not affect the main results, but it should be acknowledged (e.g., sub-Gaussian noise suffices).

---

## Nice-to-Haves

- A phase diagram plotting regions of benign/tempered/catastrophic overfitting in $(p, \beta)$ space based on the exponents in Theorem 4.2 would make the three-regime coverage concrete and practically useful.
- An experiment with varying PDE order (e.g., Poisson $p=-1$, Biharmonic $p=-2$) showing increasing noise stability as $p$ becomes more negative would directly demonstrate the core mechanism.
- A log-log plot of empirical risk vs. sample size $n$ for a kernel estimator, overlaid with the predicted rate, would provide quantitative validation of the bounds.
- A brief analysis of perturbation robustness for the co-diagonalization assumption (e.g., what is the rate degradation if $\mathcal{A}$ and $\Sigma$ are only approximately jointly diagonalizable?) would significantly broaden the paper's claimed applicability.
- Adding the Bayesian-frequentist threshold correspondence as a formal proposition (rather than a remark) would strengthen one of the paper's most striking findings.

---

## Removed Points

*These points are flagged for removal; treat them with caution — they may reflect reviewer misreading rather than genuine paper flaws.*

- **[REMOVED — parser artifact] Smoothness threshold inconsistency between §1.1 and Remark 5.** The harsh critic claims the threshold in §1.1 ($\lambda\beta \geq \lambda^r/\lambda^p - p$) conflicts with Remark 5 ($\lambda\beta \geq \lambda r/2 - p$). Reading the paper, the §1.1 formula is almost certainly a PDF-to-text parsing corruption of the same mathematical expression. Remark 5, §4.3, and Table 1 all consistently present $\lambda\beta \geq \frac{\lambda r}{2} - p$ as the threshold. This is not a genuine inconsistency.

- **[REMOVED — formatting/parser] Equation (1) vs. Equation (3) norm vs. squared norm.** The critic claims (1) uses $\gamma_n\|f\|_{\mathcal{H}^\beta}$ (non-squared) while (3) uses $\gamma_n\|f\|_{\mathcal{H}^\beta}^2$ (squared), calling this a substantive discrepancy. Reading the paper, equation (1) in §1 (the informal problem statement) appears to be a presentation shorthand/parser issue. The formal problem statement in Lemma 3.1 (equation 3) uses the squared norm, which is standard for ridge regression. No legitimate inconsistency.

- **[REMOVED — misread] Criticism that "variance is not truly independent of $\beta$."** The critic notes $\tilde{\Sigma}$ involves $\beta$. However, the paper explicitly makes the more nuanced claim in §4.3: the variance *bound's exponent* is independent of $\beta$ (it depends on $2p + \lambda\beta'$, where $\beta'$ is the evaluation norm, not the regularization norm). The statement is precise; the critic misidentifies $\beta$ (regularization) and $\beta'$ (evaluation) as the same parameter.

- **[REMOVED — scope creep / non-standard requirement] Demanding theoretical proofs for neural network experiments.** The demand for an NTK-equivalence theorem to justify using neural networks as validation of kernel theory goes beyond standard expectations for this type of paper. The neural network experiments are explicitly framed as going "beyond kernel methods." The absence of a formal equivalence is a limitation worth noting (kept as a minor weakness), but the absence of a proof is not a fatal flaw.

- **[REMOVED — misread] Criticism that the paper claims to show all three overfitting regimes but only proves benign overfitting.** Theorem 4.2 provides upper bounds whose exponents can be positive, zero, or negative depending on the parameter regime, explicitly covering all three cases. The paper's claim of covering all three regimes is accurate; the critic expected a separate theorem for each regime, which is not the standard form in this literature.

---

## Novel Insights

The most genuinely novel observation synthesized from the reviews—partially surfaced by the Spark Finder but not fully articulated in the paper itself—is the following: the variance stabilization mechanism of the inverse problem ($p < 0$) and the inductive bias smoothness threshold ($\lambda\beta \geq \lambda r/2 - p$) are **coupled** in a specific way. As $|p|$ increases (higher-order PDEs), variance stabilization strengthens *and simultaneously* the smoothness requirement on $\beta$ becomes stricter. This creates a design principle for physics-informed learning: the benefit of using a PDE-constrained model comes with an obligation to match the inductive bias smoothness to the PDE order. The matching condition—which the paper shows recovers the Bayesian threshold from Knapik et al. (2011)—is therefore not merely a curiosity but a design law linking PDE order, kernel smoothness, and generalizability. This connection between the frequentist rate and the Bayesian posterior contraction condition across both the regularized and interpolating regimes is the paper's most surprising and underemphasized contribution.

---

## Suggestions

1. **Add a standalone corollary to Theorem 4.2** that states explicit sufficient conditions (including on $\rho_{k,n}$) under which benign overfitting holds (risk $\to 0$). This is the paper's headline claim and deserves a self-contained formal statement, not a theorem-plus-remark reconstruction.

2. **Include at least one kernel-level experiment** (not a neural network) directly comparing different $\beta$ values on a benchmark inverse problem with known spectral structure. Even a synthetic 1D example with a Matérn kernel and a Laplacian operator would validate the bound at the correct level of abstraction.

3. **Add an experiment varying PDE order $p$** (e.g., $\Delta$ vs. $\Delta^2$) to empirically demonstrate that variance stabilization strengthens as $|p|$ grows, directly testing the mechanism the theory predicts.

4. **Characterize when $\rho_{k,n} = \Theta(1)$ vs. grows** for the PDE-transformed kernel $\tilde{K}$. This is the single most important open question raised by the paper's own analysis; at minimum, provide a proposition for the sub-Gaussian feature case bounding $\rho_{k,n}$ under the paper's main assumptions.

5. **Add an operator/notation summary** (a table or diagram) in the main body showing the relationship among $S, \Sigma, \tilde{\Sigma}, \tilde{K}, \phi, \psi$. This would make Theorems 3.6–3.7 parseable without reading the appendix.

6. **Discuss the co-diagonalization assumption more carefully**: identify which practical PDE-kernel pairs beyond the torus + shift-invariant kernel satisfy it, and provide at least an informal discussion of what the results would look like under approximate diagonalization.

---

**Overall assessment:** The paper makes a genuine and non-trivial contribution to the theoretical understanding of physics-informed learning by identifying the spectral mechanism behind variance stabilization in fixed-dimensional inverse problems. The novelty is high and the core technical approach is sound. However, the paper's central benign-overfitting claim currently rests on an assumption ($\rho_{k,n} = \Theta(1)$) that is not in the theorem statement and is not characterized under the paper's main assumptions; this must be resolved. The experiments are too narrow to validate the theory at its own level of abstraction—no kernel experiment is provided despite the theory being exclusively about kernel estimators. Technical soundness is good at the level of proof structure, but the presentation of Theorems 3.6–3.7 is too abstract for the key mechanism to be clearly readable. Empirical support is weak for the paper's ambitions. Significance is potentially high if the main gap (benign overfitting conditionality) is properly resolved and the experimental grounding is strengthened.

---

## cojJ2s1e35

- GT: Reject (avg 5.5)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

Now I have everything I need. Let me synthesize the three reviews against the actual paper.

---

## Summary
WLA (World modeling through Lie Action) proposes an unsupervised framework that models environment transitions as Lie group actions—specifically block-diagonal rotation+scaling matrices—on a slot-attention latent space. The simulator is first trained without action labels via inverse dynamics across multiple environments, then a small adaptation network maps external action signals to Lie algebra parameters to solve the Controller Interface Problem (CIP). Results on ProcGen (8 games, seen setting) and a real-world Android robot dataset show clear improvements over Genie in temporal consistency (FVD, Δ_t PSNR), though with modest absolute action accuracy.

---

## Strengths

- **Principled algebraic inductive bias for cross-environment generalization.** Enforcing latent transitions as elements of a rotation+scaling Lie group is a concrete, non-trivial structural choice that most continuous latent world models (SSMs, Koopman, diffusion) do not make. This directly guarantees compositionality ($\mathcal{F}(h \cdot g) = \mathcal{F}(g)\mathcal{F}(h)$) and continuity ($\lim_{\delta\to 0}\mathcal{F}(g_{t,\delta})=I$) by construction (Eq. 3), unlike black-box approaches that require ad-hoc identity-preservation tricks (e.g., the noise augmentation in Valevski et al., 2024).

- **Single shared model across all ProcGen environments.** WLA trains and evaluates one model across all 8 ProcGen games simultaneously, which is a notably harder problem than the per-environment training used by Genie in its original form. The consistent per-game wins in Table 2 (e.g., PSNR: 11.30→22.10 on coinrun, Δ_t PSNR: 0.48→9.03) under this harder regime are meaningful.

- **Large FVD gap on real-world robot video.** Table 3 shows FVD of 131.02 (WLA) vs. 393.85 (Genie), a ~3× improvement, on the 1X Android dataset which contains continuous robot actions in diverse 3D settings. This directly supports the claim that the structured latent dynamics improve temporal coherence beyond per-frame fidelity.

- **Least action slot alignment.** The proposed slot permutation heuristic—choosing the permutation $\sigma$ minimizing $\|A_{n\to\sigma(n)}\|^2$ via a linear assignment solver—is a concrete and novel mechanism to maintain object-slot consistency over time, and the ablation in Table 1 shows it contributes meaningfully (MSE unseen: 0.675→0.602).

---

## Weaknesses

### Fatal
*None identified that fully invalidates the contribution, but several major issues cumulatively weaken confidence in the central claims.*

### Major

- **The commutativity assumption is central to the math but buried in the Conclusion.** Equation (4), $z(t) = \exp(\int_0^t A(s)ds)z(0)$, is valid only when $A(s_1)$ and $A(s_2)$ commute for all $s_1, s_2$, or under a time-ordered exponential (which is not used). This same approximation propagates to the training objectives in Eq. (9), where $\exp(\Delta \sum_\ell A[\ell])$ is used. The paper acknowledges this only in the final paragraph of Section 7 ("we assume a priori that transitions in the environment commute with each other"), without any analysis of how often or how severely this assumption is violated in ProcGen or the Android dataset, or how large the resulting modeling error is. Given that this assumption underlies both the theoretical justification and the training procedure, its practical impact must be analyzed, not just mentioned.

- **No quantitative evaluation on unseen environments.** The paper's primary motivation is cross-environment generalization, and the abstract explicitly claims "quick adaptation to new environments with novel action sets." Yet Table 2 reports only *seen* environment results. The ablation Table 1 includes an unseen MSE column, but there is no unseen counterpart to Table 2 reporting PSNR/Δ_t PSNR/LPIPS across held-out environments. The claim of inter-environmental generalization is thus empirically unsupported at the headline metric level.

- **Only one baseline (Genie) throughout.** For a paper proposing a new modeling paradigm at ICLR, comparing against a single system is insufficient. Relevant missing comparisons include: (a) a continuous latent dynamics model without Lie structure (to isolate the group-theoretic contribution), (b) an object-centric video predictor without Lie structure (to isolate the object-centric contribution), and (c) any existing structured latent world model such as RSSM/Dreamer or STORM. The ablations in Table 1 do probe rotation vs. no-rotation and slot alignment, but they do not compare against a flat (non-slot) latent baseline, so the contribution of object-centricity per se remains unquantified.

- **Abstract claims "minimal or no action labels" and "novel action sets," but no experiment tests this.** There is no label-efficiency experiment (e.g., performance as a function of the number of labeled adaptation trajectories), and no held-out environment with a genuinely new action vocabulary. The claim is stated in both the abstract and introduction but is never operationalized in the experimental section.

- **Incomplete ablation: object-centric and Lie-structure contributions are not separately disentangled.** The ablation in Table 1 only removes (i) the rotation component and (ii) least action alignment. There is no ablation removing slot structure entirely (using a flat latent), nor an ablation replacing Lie-structured transitions with a generic linear transition network. Without these, it is unclear whether gains come from the Lie structure, from object-centricity, or simply from the multi-environment training setup itself.

### Minor

- **Low absolute ActionACC (21.07% seen, 14.62% unseen) lacks contextualization.** WLA outperforms Genie (10.25%/8.30%), but the absolute values are low. The paper does not report the number of action classes in ProcGen to help interpret these figures, nor does it provide a downstream task evaluation (e.g., an RL score using the controller interface) to show whether these accuracy levels are sufficient for the CIP goal.

- **The Eq. (3) IDM is an anti-homomorphism, not a homomorphism.** The paper writes $\mathcal{F}_{\Phi,\Psi}(h \cdot g) = \mathcal{F}_{\Phi,\Psi}(g) \cdot \mathcal{F}_{\Phi,\Psi}(h)$, reversing the standard homomorphism order. This is formally an anti-homomorphism and may be correct under specific left/right action conventions, but the paper does not state the convention explicitly. In a theory-grounded paper, this should be clarified.

- **Eq. (6) notation is ambiguous.** Both arrows in the composition diagram appear annotated with $\mathcal{F}_{\Phi,\Psi}^{-1}$, but the first step is performed by $\text{Ctrl}_{\text{adapt}}$ and the second by the fixed $\mathcal{F}_{\Phi,\Psi}^{-1}$. As written, the equation is misleading.

- **Key architectural hyperparameters ($N$, $J$) deferred entirely to an appendix** that is not provided in the reviewed submission, hampering reproducibility assessment.

### Tiny

- The $\Delta_t$ PSNR formula as typeset is missing a closing parenthesis, making the expression ambiguous. This should be corrected for clarity.

- The conclusion's claim of being "the first of its kind as a generative interactive framework that is based on a state-space model" is too strong given the SSM-based video prediction literature (e.g., RSSM).

---

## Nice-to-Haves

- **Long-horizon rollout evaluation.** Figures 5 and 6 show 16-frame sequences; quantifying rollout quality at 32/64 steps (e.g., via FVD over longer windows) would strengthen claims about temporal coherence.

- **Visualization of slot assignments over time.** A qualitative figure showing which objects each slot tracks would validate whether the architecture truly decomposes dynamics object-wise as claimed.

- **Per-environment WLA vs. joint WLA comparison.** Showing that joint multi-environment training improves per-environment performance (vs. training separate models per environment) would directly validate the "inter-environmental" premise.

- **Discussion of failure cases.** Where does WLA break down—scenes with many objects exceeding slot capacity, fast dynamics where the commutative approximation fails, occlusion-heavy scenarios? A failure analysis would calibrate the method's operating envelope.

- **Extension to non-commutative (non-abelian) groups.** The current restriction to abelian rotation+scaling excludes important cases (e.g., 3D rotations). Discussing concrete paths toward non-abelian Lie groups would improve the paper's significance.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic: Fairness of Genie comparison.** The critic objects that Genie was given 0.4M iterations vs. its default 0.2M, and was adapted for multi-environment. However, this modification gives Genie *more* compute and a fairer regime. The asymmetry benefits the baseline, not the authors' method, making this a non-issue. REMOVED per rule on comparisons that favor the baseline.

- **Harsh Critic: Human analogy is scientifically loose.** Using human skill transfer as motivation is standard rhetoric in ML papers and does not constitute a scientific error. REMOVED as stylistic criticism.

- **Harsh Critic: "Unstructured vs. structured CIP formalism contributes little."** The CIP formalism provides useful conceptual scaffolding for positioning the problem against prior work and is not harmful. Whether it adds "operational" novelty is debatable, but it does help frame the paper. REMOVED as opinion rather than substantive flaw.

- **Harsh Critic: Lack of no-action identity guarantee in diffusion-based models hurts them unfairly.** The critic notes this point is presented without evidence. However, the paper correctly identifies this as a real theoretical property difference: Lie group structure enforces identity as a group axiom, whereas black-box models do not. The connection is principled even if the empirical consequence is not further demonstrated. REMOVED as unfair characterization.

- **Harsh Critic: Requesting theoretical proofs for why Lie structure is necessary.** Demanding formal necessity proofs for an empirical systems paper is not standard for ICLR. REMOVED.

- **Harsh Critic: Training cost and optimization details missing.** Lack of optimization details in the main text (batch size, learning rate schedule, training time) is a reproducibility concern, but appendix placement is standard and acceptable. Moved to nice-to-have context only.

- **Harsh Critic: "No broader impact discussion."** Not standard for ICLR 2025 submission requirements; absence of broader impact section is not a technical weakness. REMOVED.

---

## Novel Insights

The most underappreciated insight in the paper is the connection between the "no-action = identity" problem in black-box world models and the Lie group axioms. Because $M(e) = I$ is guaranteed by group structure, and $\lim_{\delta\to 0} M_{t,\delta} = I$ follows from the continuity of the Lie group action (Eq. 3), WLA avoids the identity-corruption problem that plagues diffusion-based controllers (Valevski et al., 2024) without any heuristic noise augmentation. This is a structural advantage that deserves stronger emphasis in the paper, as it constitutes a concrete and verifiable theoretical advantage over black-box baselines rather than a mere inductive bias.

---

## Suggestions

1. **Add a held-out environment evaluation table** (parallel to Table 2 but for unseen ProcGen environments) with full PSNR/Δ_t PSNR/LPIPS metrics. This is the single highest-priority missing experiment given the paper's motivation.

2. **Conduct a label-efficiency ablation**: report ActionACC and Δ_t PSNR on a held-out environment as a function of the number of labeled adaptation trajectories (e.g., 10, 50, 200, 1000). This directly tests the "minimal action labels" claim.

3. **Acknowledge and analyze the commutativity assumption earlier** (at Eq. 4, not only in the conclusion), and add an empirical measurement of how much the commutative approximation error grows with rollout length or environment complexity.

4. **Add a flat-latent (no-slot) and a generic-linear-transition ablation row to Table 1** to separately quantify the object-centric and Lie-structure contributions.

5. **Provide N, J, and latent dimensions in the main text**, even as a table, rather than deferring entirely to an appendix.

6. **Contextualize ActionACC** by stating the number of action classes and adding a chance-level baseline. If possible, add a downstream policy evaluation metric to show whether the latent actions are sufficient for actual control utility.

---

**Overall evaluation:** The paper introduces a technically interesting and well-motivated framework with a clear structural advantage over discrete autoregressive baselines. The empirical improvements on ProcGen and the Android dataset are real and non-trivial. However, the central experimental claim of cross-environment generalization is not adequately tested (no unseen quantitative results), the comparison landscape is too narrow (Genie only), and a critical mathematical assumption (commutativity) is acknowledged only in the conclusion without impact analysis. In its current form, the paper reads as a promising and partially validated contribution rather than a fully substantiated ICLR result. Closing the unseen-evaluation and label-efficiency gaps would substantially strengthen the submission.

---

## TH4gKbZS1E

- GT: Reject (avg 2.5)
- Predicted: Reject (3.2/10)
- Match: YES

### Final Review

## Summary

This paper empirically compares Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on ten synthetic 1D functions spanning six regularity categories: regular (smooth), continuous but non-differentiable, jump-discontinuous, singular, coherently oscillatory, and noisy. The authors match parameter counts between architectures and vary training sample sizes, reporting that KAN outperforms MLP on regular and severe-singularity/oscillation functions while MLP is generally superior on locally irregular (non-differentiable, jump) functions. The paper also finds that KAN is orders of magnitude slower than MLP in wall-clock time and that noise generally obscures the locally irregular features from both architectures.

---

## Strengths

- **Function-type taxonomy as a diagnostic lens.** The explicit categorization of test functions into six regularity classes (Table 1) provides a structured framework for understanding architecture-specific failure modes, which is more informative than a single aggregated benchmark.

- **Optimizer and wall-clock time analysis.** The inclusion of Tables 3 and 4, comparing Adam and L-BFGS for both architectures with actual training times, is a concrete practical contribution. Showing that KAN with L-BFGS is up to 70× slower than MLP — even when convergence in epochs favors KAN — is a result that practitioners need to know and that is often omitted from KAN studies.

- **Differentiated noise analysis.** Separating the noisy-function analysis into regular, localized-irregularity, and severe-discontinuity sub-categories, and finding that noise has little additional effect on already-difficult singularity/oscillation functions, is a non-obvious and practically meaningful observation.

---

## Weaknesses

### Fatal

*(None that individually invalidate the entire paper, but the combination of Major weaknesses below substantially undermines the reliability of the reported conclusions.)*

### Major

- **Critical text–figure inconsistency for jump functions (f₅, f₆).** Section 3.3 states unambiguously: "Results show that the MLP outperforms the KAN." The Figure 3 caption, however, states the opposite: "In all cases, KAN (red dashed line) fits the target function much better than MLP." These two claims are directly contradictory on the same experiment. This is not a minor labeling slip — the jump-function result is one of the paper's four main comparative findings. Without knowing which is correct, readers cannot trust either the text or the figures for this category. Numerical tables reporting final test loss are needed to settle this, and the paper has none.

- **Optimizer confounder for coherent oscillation functions.** For f₉ in Figure 8, KAN is evaluated with L-BFGS while MLP is evaluated with Adam (the per-model best), and the conclusion is drawn that "KAN consistently surpasses MLP." This is a best-of-each comparison, not a clean architectural comparison. Presenting it as evidence of architectural superiority without clearly labeling it as a practical best-system comparison conflates optimizer-architecture interaction with architectural capability. The conclusion in Section 5 ("KAN exhibits superior performance over MLP for regular functions or functions with severe discontinuities") inherits this confound.

- **Dangling cross-reference to non-existent "section D."** Section 3.5 reads: "taking a similar approach as described in section D." No such section exists anywhere in the paper. This indicates the paper is incomplete or was not proofread, and raises doubts about whether the experimental protocol for Sections 3.5 and 4.3 is fully described.

- **Noise model never formally defined.** Section 4 uses "noise level 10" and SNR values (SNR=10, SNR=0, SNR=4) interchangeably without ever specifying: (a) additive vs. multiplicative noise, (b) distribution (Gaussian, uniform?), (c) what "noise level 10" means dimensionally, (d) whether test loss is evaluated against the clean target or the noisy observations. These choices fundamentally affect the interpretation of all Section 4 results and are a reproducibility failure.

- **All results are single runs.** No variance across random seeds or noise realizations is reported. For shallow neural networks fitted on 1D toy functions, optimization noise and initialization can materially affect outcomes. Conclusions such as "KAN achieves a lower test loss with low noise levels but performs worse under high noise conditions" are drawn from what appears to be single-realization evidence. This is inadequate for any quantitative claim at ICLR.

### Minor

- **Exclusive focus on 1D univariate functions.** The Kolmogorov-Arnold theorem is fundamentally a statement about *multivariate* function representation: KAN's theoretical motivation is the decomposition of f: [0,1]ⁿ → ℝ into combinations of univariate functions. Testing only 1D functions reduces KAN to a spline approximator and bypasses the architectural regime where KAN's structure should theoretically matter. The paper briefly acknowledges that multiplication nodes in KAN 2.0 matter "minimally" for the tested functions, but this observation actually underscores the limitation rather than addressing it.

- **Wall-clock inefficiency downplayed in the conclusion.** The conclusion emphasizes that "KAN exhibits a faster convergence rate than MLP across all tested functions" (measured in epochs) without any corresponding acknowledgment of the 10–70× wall-clock overhead. A reader taking the conclusion at face value would have a seriously misleading impression of KAN's practical utility.

- **Training loss vs. test loss not disentangled.** Only test loss is reported in the convergence curves. For functions where KAN performs worse (f₃–f₆), it is impossible to determine whether the cause is poor optimization, underfitting, or overfitting. This distinction matters for understanding the architectural inductive bias and for suggesting remedies.

- **KAN grid resolution fixed without ablation.** Grid=3, k=3 is used throughout. This is KAN's most direct capacity control. For singular and oscillatory functions — exactly where spline resolution should matter most — no ablation is presented. The reader cannot tell whether KAN's failures are architectural or a result of under-specified hyperparameters.

### Tiny

- The paper contains no formal experimental setup section. Sampling domain, test-set construction, MSE vs. other loss definitions, and stopping criteria are scattered across subsections or omitted entirely.
- Section 3.3 defines jump locations as "x = ±0.5" but Table 1 defines f₅ as {1 if |x| < 0.5, else 0}, which is symmetric — this is not an error but the language in 3.3 could cause confusion.

---

## Nice-to-Haves

- **Multivariate test functions (2D/3D).** Adding at least one 2D benchmark would engage KAN's actual theoretical regime and make the comparison meaningfully broader.
- **Training-time-normalized convergence curves.** Plotting test loss against wall-clock time (in addition to epochs) would give an honest picture of the efficiency-accuracy tradeoff.
- **Mechanistic analysis of learned KAN activations.** Visualizing KAN's learned spline activations for representative functions — especially where it fails (f₃–f₆) versus succeeds (f₇–f₁₀) — would reveal whether the splines are adapting meaningfully and provide genuine insight beyond the empirical tally.
- **Bias–variance or train/test decomposition.** Separate curves showing training loss and test loss would help readers understand whether KAN's disadvantage on irregular functions stems from optimization difficulty or overfitting.
- **Theoretical hypothesis paragraph.** A brief discussion hypothesizing *why* smooth B-spline activations (KAN) might struggle at cusps and jump discontinuities while piecewise-linear activations (ReLU MLP) adapt more easily would substantially increase the paper's depth.

---

## Removed Points

*These points are flagged for removal — treat with caution.*

- **Critic: "The function definitions are potentially confusing — f₅ uses threshold x < 0.5 suggesting one-sided behavior."** Table 1 clearly defines f₅ = {1 if |x| < 0.5, 0 otherwise}, which is symmetric. This is a misread by the critic.

- **Critic: Requesting SIREN / Fourier-feature MLP baselines.** The paper's stated scope is a KAN-vs-MLP comparison. Adding sinusoidal baselines would be useful context but is outside the paper's framing and represents scope creep rather than a genuine weakness of the paper's contribution.

- **Critic: "The functions are too easy for Section 3.1."** f₁ and f₂ serve as calibration/sanity checks for a category labeled "regular." Their simplicity is the point. Demanding harder regular functions misunderstands the section's role.

- **Critic: "Severe discontinuities is inaccurate terminology."** The paper uses this as a collective label for its own subcategory of singular/oscillatory functions across restricted domains. This is a taxonomic style choice, not a factual error.

- **Critic: "Claims about KAN 2.0 are not operationalized."** The paper explicitly states that multiplication nodes matter "minimally for the functions used in this paper" and that lower versions of PyKAN are acceptable. This is addressed.

- **Positive reviewer Strength: "Timely topic / KANs were recently introduced."** This is generic and applies to any contemporaneous benchmarking paper.

- **Positive reviewer Strength: "Controlled experimental setup (matching parameter counts)."** While this is a genuine effort, it is standard practice in architecture comparison papers and does not distinguish this paper specifically, especially given the optimizer confound noted above.

---

## Novel Insights

The observation that noise has comparatively little *additional* effect on the test loss for already-difficult singularity and coherent oscillation functions (Section 4.3) is the paper's most genuinely non-obvious finding. The intuition — that approximation error already dominates before any noise is added — is plausible and, if confirmed with proper statistical rigor, would be a useful empirical result for practitioners applying KAN or MLP to highly irregular scientific data. However, this insight is currently stated qualitatively without the statistical support needed to be trusted.

---

## Suggestions

1. **Resolve the text–Figure 3 contradiction immediately.** Add a numerical table reporting mean test loss at convergence for each model on f₅ and f₆. This is the highest-priority fix.

2. **Define the noise model precisely.** Specify in a single paragraph: distribution, parameterization, how "noise level" maps to that parameter, whether test loss targets clean or noisy observations, and whether results are averaged over multiple noise draws.

3. **Run each experiment with ≥5 random seeds.** Report mean ± std on all test-loss comparisons. For the noisy experiments, additionally average over noise realizations.

4. **Fix or remove the "section D" reference** and ensure the experimental protocol for Sections 3.5 and 4.3 is self-contained.

5. **Clearly separate "best optimizer per architecture" comparisons from "fixed optimizer" comparisons.** Figure 8 is a best-of-each comparison; label it as such and present a fixed-optimizer parallel plot as an ablation.

6. **Provide an ablation over KAN grid sizes** (e.g., grid ∈ {3, 5, 10, 20}) for at least one function from each regularity category to establish that grid=3 is not artificially handicapping KAN.

7. **Rewrite the conclusion's convergence claim** to prominently note that "faster convergence in epochs" comes with 10–70× higher wall-clock cost, and quantify the tradeoff explicitly.

---

**Evaluation axes:**

- **Novelty:** Low-to-moderate. The function taxonomy is a structured contribution, but the overall experimental setup is straightforward benchmarking with no architectural or theoretical innovation.
- **Technical soundness:** Weak. The combination of a text–figure contradiction, an optimizer confounder for a key function class, a dangling section reference, and an undefined noise model constitutes a set of methodological gaps that are difficult to overlook.
- **Empirical support:** Weak. Single runs, no confidence intervals, qualitative conclusions drawn from visual inspection of plots, and unresolved inconsistencies undermine confidence in all reported results.
- **Significance:** Limited. All conclusions come from 1D toy functions; the primary regime where KAN's theory would differentiate it from MLP (multivariate decomposition) is entirely untested. The practical utility of the findings is unclear.
- **Clarity:** Below acceptable. The missing section reference, undefined noise model, and text–figure contradiction suggest the paper was not carefully proofread prior to submission.

---

## QKBu1BOAwd

- GT: Accept (Oral) (avg 7.3)
- Predicted: Accept (5.5/10)
- Match: YES

### Final Review

## Summary
DRAFT proposes an iterative framework for refining tool documentation for LLMs via self-driven trial-and-error interaction. The method decomposes documentation improvement into three phases—Experience Gathering (Explorer), Learning from Experience (Analyzer), and Documentation Rewriting (Rewriter)—augmented by a diversity-promoting exploration strategy (similarity constraint + self-reflection) and a tool-adaptive termination mechanism (BLEU + embedding similarity). Experiments on ToolBench and RestBench across three LLMs show consistent gains in Correct Path Rate and Win Rate, with supporting evidence from tool retrieval and human evaluations.

---

## Strengths

- **Genuine novelty in the application of execution feedback to documentation refinement.** Prior work (e.g., EasyTool) rewrites documentation one-shot using LLMs without grounding in actual tool execution traces. DRAFT uniquely grounds each revision in real tool responses (e.g., actual parameter errors, return field structures), enabling documentation that reflects actual tool behavior rather than a paraphrase of the original. This is a meaningful and underexplored direction.

- **Diversity-promoting exploration is a principled design.** The combination of embedding-based similarity constraint (Eq. 2) with self-reflection for regeneration is a thoughtful mechanism that addresses coverage failure in naive iterative exploration. The ablation in Table 2 confirms it matters: removing it drops CP% from 88 to 84 and Win% from 71 to 69 on TMDB with GPT-4o.

- **Tool-adaptive termination addresses a genuine over-iteration failure mode.** Figure 6 shows a non-monotonic performance curve across iterations, confirming that running too many refinements degrades performance. The mechanism is ablated in Table 2 (drop to 80% CP / 68% Win without it), giving direct empirical backing rather than just a theoretical claim.

- **Multi-stage evaluation is distinctive.** The paper validates improvements not only on downstream tool-use task performance (Table 1) but also on tool retrieval quality (Table 3) and human comprehension (Table 4). The retrieval analysis is particularly compelling: it shows that rewritten documentation is semantically improved even independent of any generation task, reducing concerns that gains are task-specific artifacts.

- **Cross-model transfer shows documentation improvements are model-agnostic.** DRAFT documentation refined with GPT-4o improves performance of GPT-4o-mini and Llama-3-70B (Table 1). On ToolBench, GPT-4o-mini + DRAFT (47% CP) even surpasses GPT-4o without DRAFT (37% CP), suggesting that documentation quality is a meaningful bottleneck worth targeting.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Missing single-pass-with-feedback ablation undermines the core claim about iteration.** The paper's central claim is that *iterative* trial-and-error refinement is necessary. However, the ablation in Table 2 only removes the diversity or termination mechanisms—it never compares DRAFT against a single-pass rewrite in which the Rewriter is given a batch of tool execution results at once without iteration. Without this, it is impossible to determine whether the iterative loop adds value beyond simply exposing the model to tool output once. This is the most critical missing experiment.

- **Win% evaluation is underspecified.** Win% is the primary metric for three out of six experimental columns in Table 1, yet the paper describes it only as computed by a "ChatGPT-based evaluator" (Section 3.1) with no disclosure of: which specific model was used, the prompt template, whether evaluation was blinded to method identity, whether position/order of outputs was randomized, or any measure of evaluator consistency. This is a major reproducibility and validity concern.

- **Algorithm 1 has a structural issue and confusing termination logic.** As written, when the break condition fires at Line 16 (`if Δ > τ then Break`), execution jumps out of the loop *before* Line 19 executes (`D̃ ← D̃ ∪ t_i`). This means the final converged documentation `t_i` is never added to the output set. The algorithm as presented returns all *pre-convergence* versions but not the converged one—which is the opposite of the intended behavior. Additionally, naming Δ the "degree of change" when it is computed as a similarity metric (BLEU + cosine sim, where higher = more similar = *less* change) is internally inconsistent. The paper says "we consider the iterative process to have converged when there is minimal change," which matches the logic (high Δ = converge), but the terminology inverts intuition. This should be corrected and clarified.

- **No computational cost analysis.** DRAFT makes multiple sequential LLM calls per tool (Explorer → Analyzer → Rewriter) across up to 5 iterations. No API call counts, token consumption, wall-clock times, or cost estimates are reported. Given that gains on ToolBench with GPT-4o are +6% CP and +1% Win over EasyTool, the cost-benefit tradeoff is a meaningful practical question. This is especially important because EasyTool is described as a single-pass rewrite, which would be substantially cheaper.

### Minor

- **Cross-model generalization claim is stronger than the evidence.** The only experiment with an alternative backbone is Figure 7, using Llama-3-70B on RestBench-TMDB only. The paper concludes from this that the approach generalizes across models, but one dataset and one alternative backbone are insufficient to substantiate "robust cross-model generalization capabilities." Testing on at least RestBench-Spotify and ToolBench would meaningfully strengthen this claim.

- **Ablation covers mechanisms but not modules.** The ablation (Table 2) validates the diversity and termination mechanisms, but there is no ablation isolating the Analyzer module. A Rewriter-only variant (using `(t_{i-1}, e_i, r_i)` directly without the Analyzer's intermediate suggestions `s_i`) would clarify whether the Analyzer is adding value beyond the information already available. Similarly, the exploration directions `d_i` produced by the Rewriter are not independently evaluated.

- **Retrieval results are not uniformly positive.** Contriever on Spotify @10 slightly decreases from 49.6 to 49.2 with DRAFT, and BM25 on Spotify @1 is unchanged (43.9 vs 43.9). The paper does not discuss these non-improvements, which suggests the benefits may be tool/dataset-dependent.

- **Human evaluation is small-scale and lacks agreement statistics.** Only 3 annotators evaluate 50 cases, with no inter-annotator agreement (e.g., Fleiss' κ) reported. For RestBench accuracy, 70% of cases are labeled "Equal," which suggests either the task is difficult to judge or many cases show no clear improvement—this deserves discussion rather than being passed over.

### Tiny

- **EasyTool backbone model mismatch.** The paper uses GPT-4o as DRAFT's backbone and compares against EasyTool, which uses "ChatGPT" (presumably GPT-3.5 or earlier). This comparison is not apples-to-apples in terms of the rewriting model's capability, which inflates the apparent advantage. Rerunning EasyTool with GPT-4o would give a cleaner comparison.

- **ToolBench subset claim scope.** Since only the I3-Instruction subset is used, claims about ToolBench performance should be scoped accordingly.

---

## Nice-to-Haves

- **Compare against in-context few-shot demonstrations from explored examples.** A natural alternative to rewriting documentation is to provide the Explorer's gathered (query, result) pairs directly as few-shot demonstrations during inference. This would not modify the documentation and is simpler to deploy. If DRAFT significantly outperforms this, it provides a strong practical argument for the documentation-rewriting approach.

- **Task-completion metric beyond path correctness.** CP% measures whether the model's tool-call sequence contains the ground-truth subsequence, not whether the final user query is answered correctly. A model could follow the right tool path but misuse the returned results. Adding a task-completion or answer-quality metric would strengthen the claim of practical benefit.

- **Sensitivity analysis for termination threshold τ.** The paper sets τ = 0.75 without discussion of how sensitive results are to this value. A small sweep would validate the chosen value and inform practitioners deploying DRAFT on new tool sets.

- **Trajectory visualization across iterations.** The paper shows a single before/after snapshot (Figure 2 right panel). Showing how the documentation evolves iteration-by-iteration for a concrete example would directly substantiate the claim that each iteration adds meaningful signal rather than noise.

- **Documentation length and completeness/conciseness tradeoff analysis.** The paper claims DRAFT produces documentation that is simultaneously more complete and more concise. These can be in tension. Tracking average documentation length across iterations and its relationship to performance would clarify whether gains come from enriched content, streamlined content, or both.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Figure 1(c) labeling concern (Harsh Critic).** The critic flagged that Figure 1(c) appears to show raw documentation winning 92.5% of the time, which contradicts the paper's claims. This is almost certainly a parsing artifact from the text extraction of the figure—the column headers (Raw vs. Improved) were likely swapped in parsing. The paper's narrative, its Table 1 results, and the figure caption all consistently claim that DRAFT-improved documentation is preferred. This should not be treated as a genuine paper flaw.

- **"Causal claim under-justified" / other bottlenecks not separated (Harsh Critic).** The critic argues the introduction must first prove documentation is the *dominant* bottleneck over planning errors, schema complexity, etc. This is scope creep: the paper's contribution is to show that documentation quality *is* a meaningful bottleneck worth addressing, not that it is the only one. Showing consistent downstream gains is sufficient justification. Removed.

- **Section 2.5 "marketing-oriented" (Harsh Critic).** Critique of the tone of a summary section—this is a style/formatting nitpick. Removed.

- **"No constraints on hallucinated rewrites" (Harsh Critic).** Speculative concern about the Rewriter fabricating tool behavior not observed in exploration. While theoretically valid, the paper does not claim to prevent hallucination and this falls outside stated scope. Additionally, the Rewriter is always conditioned on actual tool execution traces (r_i), which grounds its rewrites. Removed as a separate weakness; the human evaluation (Table 4, accuracy dimension) provides partial empirical evidence against systematic hallucination.

- **"Natural-language feedback is asserted better than scalar feedback without direct evidence" (Harsh Critic).** The paper discusses this design choice in the context of the broader LLM feedback literature (Section 4, Learning from Feedback) and provides sufficient conceptual justification. Not requiring an ablation of scalar vs. NL feedback is reasonable scope given the paper's focus. Removed.

- **"Fully automated / dynamically maintaining / explainability are over-claims" (Harsh Critic).** These are summary statements about the system's properties, not empirical claims being advanced as contributions. Critiquing them as insufficiently proven is a style nitpick on how system features are described. Removed.

- **Generic "missing limitations section" as a standalone weakness.** The absence of a dedicated limitations section is a presentation preference; the paper discusses several limitations inline (e.g., performance degradation from over-iteration, cost savings from termination). Removed as a standalone weakness, though the cost/safety concerns are kept in the main weaknesses.

- **"Related work is not sufficiently comparative" (Harsh Critic).** Vague and not specific enough to act on. Removed per instructions (no missing related works).

---

## Novel Insights

The spark finder's observation about providing explored examples as few-shot in-context demonstrations at test time (rather than baking them into documentation) is the most actionable novel insight from the synthesis. This baseline is not in the paper, and if it performs comparably to DRAFT, the justification for the more expensive documentation-rewriting pipeline weakens significantly. Conversely, if DRAFT outperforms it, that result would be one of the strongest possible empirical arguments for the paper's approach—since it would show that reformulating knowledge into persistent documentation is better than retaining raw interaction traces. This experiment is missing and is the single highest-leverage addition the authors could make to the paper. A secondary insight is that the retrieval gain analysis (Table 3) implicitly validates that DRAFT's documentation improvements are not specific to any generation model or prompt style, because retrieval is a model-agnostic downstream task—this is a stronger form of generalization evidence than the cross-model generation experiments, and the paper undersells it.

---

## Suggestions

1. **Add a single-pass-with-feedback baseline.** Run the Rewriter once using a batch of N Explorer traces (where N = average iterations in DRAFT) without the iterative loop. This is the minimal ablation to justify the iterative architecture and should be straightforward to implement.

2. **Add in-context few-shot demonstration baseline.** At inference time, prepend the (query, parameters, result) triples collected by DRAFT's Explorer to the prompt without modifying the documentation. Compare performance against DRAFT documentation. This tests whether documentation rewriting provides persistent, generalizable benefit over ephemeral context augmentation.

3. **Fully specify the Win% evaluator.** Publish the exact model name (version), evaluation prompt, position randomization protocol, and any cross-run consistency check. Without these details, Win% results cannot be reproduced or verified.

4. **Fix Algorithm 1 output logic.** Move Line 19 (`D̃ ← D̃ ∪ t_i`) to before the break check, or add explicit handling to output `t_{i-1}` when convergence is triggered, and rename Δ to something like "stability score" or "convergence score" to avoid the "degree of change" terminology inversion.

5. **Report cost per tool.** Add a table showing average API calls, tokens consumed, and approximate monetary cost per tool for DRAFT vs. EasyTool. Even rough estimates would allow practitioners to make an informed tradeoff decision.

6. **Rerun EasyTool with GPT-4o backbone.** Use the same backbone for EasyTool to make the comparison model-controlled. If DRAFT still outperforms, the advantage is attributable to iteration and feedback rather than model capability.

7. **Extend Figure 7 to all datasets.** Using Llama-3-70B as backbone on TMDB, Spotify, and ToolBench (rather than just TMDB) would meaningfully substantiate the cross-model generalization claim.

---

**Overall assessment:** DRAFT addresses a real and underappreciated problem—that human-written API documentation is often misaligned with LLM comprehension—and the core idea of using actual tool execution feedback to iteratively improve documentation is genuinely novel and practically significant. Empirical results are consistent across models and benchmarks, and the multi-dimensional evaluation (task performance + retrieval + human judgment) is a notable strength. The paper is held back primarily by the absence of a critical ablation (single-pass-with-feedback), underspecification of the Win% evaluator, and an algorithmic presentation issue in Algorithm 1. These are fixable gaps, and if addressed, the paper would stand on substantially stronger footing. As submitted, the work is promising and above the workshop tier, but the missing ablation is a genuine threat to the core narrative and should be resolved before the contribution can be considered fully established.

---

## BZQmpsuW7D

- GT: Reject (avg 5.8)
- Predicted: Reject (4.1/10)
- Match: YES

### Final Review

## Summary
SPARK proposes a physics-guided vector-quantized memory bank for augmenting dynamical system training data, combined with a Fourier-enhanced graph ODE for long-horizon forecasting. The central idea is to pre-train a discrete codebook enriched with boundary conditions and physical parameters, then use nearest-neighbor retrieval in the codebook to create physics-consistent augmented samples, thereby improving robustness to data scarcity and distribution shift. The method is evaluated across five PDE/weather benchmarks and shows strong improvement over neural-operator and vision-backbone baselines.

---

## Strengths

- **Physics-guided discrete augmentation is a genuinely novel framing.** While VQ-VAEs and physics-informed neural networks both exist, fusing boundary positional encodings and physical-parameter channel attention into a shared discrete codebook, then using that codebook specifically for latent-space data augmentation, is a distinct contribution not present in prior operator-learning or scientific ML literature. The motivation—that discrete prototypes over physics conditions provide a structured interpolation space for OOD generalization—is concrete and falsifiable.

- **Demonstrated plugin utility across multiple backbones.** Figure 1 (ERA5 radar chart across ViT, CNO, U-Net, SwinT, NMO) and Table 3 (SimVP, PredRNN, Earthfarseer + SPARK on SEVIR transfer) provide direct evidence that SPARK improves diverse backbone architectures without architectural modification. This partially substantiates the plugin claim and sets SPARK apart from papers that only propose end-to-end models.

- **Breadth and difficulty of benchmark coverage.** Five heterogeneous datasets (Prometheus CFD, ERA5 atmospheric, Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) plus a challenging sea ice transfer task represent a genuinely demanding evaluation regime for dynamical system modeling. Including both synthetic PDE and real-world meteorological data strengthens the generality claim.

- **Near-zero OOD degradation on ERA5.** SPARK's degradation from in-distribution to OOD on ERA5 (0.0398 → 0.0401, Table 4) is remarkably small compared to competing OOD-specific methods (LEADS: 0.2367 → 0.4233; CODA: 0.1233 → 0.2367). If the OOD protocol is appropriately challenging, this result is a compelling empirical signal.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **No ablation studies.** The method combines at least four distinct components: (1) boundary positional encoding, (2) physical-parameter channel attention, (3) VQ memory bank augmentation, and (4) Fourier-enhanced graph ODE. No ablation table appears in the paper. This means the improvement observed in Table 1 cannot be attributed to any specific component. It is equally consistent with the result that the Fourier-enhanced graph ODE alone explains nearly all gains and the augmentation plugin contributes little. This is the single most important missing experiment for an ICLR submission about a plugin augmentation method.

- **OOD evaluation protocol is undefined.** The paper reports "w/ OOD" and "w/o OOD" across all tables but never specifies what is shifted: which physical parameters are out-of-range, by how much, whether shifts are interpolative or extrapolative, and whether OOD test environments contain unseen boundary geometries, unseen parameter values, or unseen time windows. Since OOD robustness is the central empirical claim, this omission makes the results difficult to interpret or reproduce, and prevents assessing whether the challenges are actually challenging.

- **Numerical inconsistency across tables undermines reproducibility.** SPARK's Prometheus MSE is reported as 0.0294/0.0308 (w/o OOD / w/ OOD) in Table 1 but as 0.0323/0.0328 in Table 4. Similarly, SPARK's ERA5 numbers differ between Table 1 (0.0322/0.0321) and Table 4 (0.0398/0.0401). These are not within rounding error and are unexplained. Either different model variants, data splits, or hyperparameter settings are used—none of which are disclosed. This raises questions about selective reporting and significantly weakens confidence in both tables.

- **The augmentation target semantics are undefined.** Equation 7 defines augmented latent inputs v_i by mixing node embeddings with codebook entries. The paper states these augmented samples are added to the training set, but never specifies the corresponding prediction target Y_i. If the original target is reused unchanged, this requires a justification that the augmented latent is label-preserving under physics-guided interpolation. Without this, the theoretical rationale for why augmented samples improve generalization rather than corrupt it is missing.

- **Suspicious ERA5 baseline performance.** Table 1 shows FNO (0.7233/0.9821), UNO (0.6652/0.7621), and CNO (0.5243/0.7821) having dramatically higher MSE on ERA5 than NMO (0.0432/0.0563) or the authors' method (0.0322/0.0321). These order-of-magnitude gaps for well-known methods on a standard atmospheric dataset—without any explanation of hyperparameter tuning, input normalization, or task adaptation—suggest potential misconfiguration of baselines. If these methods were not properly adapted to the ERA5 task (e.g., different input resolution, normalization, or rollout horizon), the reported improvement is inflated.

### Minor

- **Equation 7 normalization is inconsistent with K.** The augmented representation is $v_i = \lambda h_i + (1-\lambda)\sum_{n=1}^K e_n$. As written, the sum of K codebook entries is not normalized by K, so the scale of the second term grows with K. This is likely a typo (should be $\frac{1}{K}\sum$), but as stated it makes the interpolation invalid for K > 1.

- **Equation 8 notation is under-specified.** The symbol δ in $q_i = \frac{1}{T_0}\sum_t \delta(\alpha_i^t \cdot v_t^i)$ is never defined (activation function? identity?), and the attention scores $\alpha_i^t$ lack normalization, making it unclear how the weighted average is computed. In Equation 9, $H^l$ appears in the ODE derivative but is not defined in this section; it is unclear whether it refers to layer-wise hidden states or historical observation embeddings.

- **Sea ice section lacks comparative quantitative evaluation.** Section 4.3 shows SPARK's training convergence curves (Figure 5) but does not report a comparison table of SPARK vs FNO vs U-Net on the sea ice task in terms of MSE, SSIM, or PSNR. The qualitative Figure 4 is suggestive but insufficient for a quantitative claim of superiority on this challenging task.

- **Theoretical analysis is generic.** Theorems 1 and 2 are a standard mutual-information generalization bound and a PAC-Bayesian bound. Neither theorem depends on vector quantization, the specific augmentation formula (Eq. 7), the Fourier-enhanced ODE, or any property of the proposed architecture. The key step—showing that SPARK specifically reduces $I(\theta; \mathcal{D} | \mathcal{P})$ or $\text{KL}(Q\|P)$—is asserted rather than proved. As written, the theory justifies "any physics-informed prior helps" rather than "SPARK's design helps."

### Tiny

- The paper title says "Quantitative Augmentation" while the abstract and body consistently use "Quantized Augmentation." The former is an incorrect description of the method.
- The scalability claim (Table 2) shows ERA5 MSE rising from 0.0302 to 0.0391 when model size drops from 24.56MB to 2.18MB—a 29% increase. Describing this as "stable" performance is overstated; the degradation is monotonic and non-negligible.

---

## Nice-to-Haves

- **Sensitivity analysis on λ, K, and memory bank size M.** These three hyperparameters directly control the augmentation behavior, yet no analysis of sensitivity is provided. Even a small grid search over λ ∈ {0.1, 0.3, 0.5, 0.7} and K ∈ {1, 5, 10} would indicate whether the method is robust or tightly tuned.
- **Memory bank interpretability.** A t-SNE/UMAP of learned codebook entries colored by physical parameters (e.g., viscosity bins, boundary condition type) would directly test whether the discrete codes correspond to physically meaningful regimes—this would strengthen the "physics-guided" framing.
- **Controlled data scarcity experiments on primary benchmarks.** The transfer experiment (Table 3) tests data scarcity in a cross-domain setting. Systematically training with 5%, 10%, 20%, 50% of Prometheus or Navier-Stokes data and comparing with and without SPARK would more directly validate the data-scarcity claim.
- **Computational cost comparison.** The method stacks a VQ-VAE pretraining stage on top of a Fourier graph ODE. A table comparing training/inference time and GPU memory against FNO/NMO baselines would clarify whether the improved accuracy justifies the added cost.
- **Augmentation visualization.** Showing a decoded augmented sample (decoded from $v_i$) alongside the original and the nearest codebook neighbor would make the augmentation mechanism interpretable and would help readers assess physical plausibility of the generated samples.
- **Physical constraint metrics.** Reporting conservation error (e.g., mass or energy conservation residuals) alongside MSE would strengthen the "physical consistency" claim beyond the visual energy-spectrum comparison.

---

## Removed Points
*These points are flagged for removal—treat them with caution.*

- **"First to propose" claim (Harsh Critic).** While the claim is poorly supported, removing it from the paper is a style fix, not a substantive weakness. The actual novelty of the combination stands on its own without this phrase. Not a reviewable weakness.
- **KNN graph vs. physical topology (Harsh Critic).** The paper follows prior work on graph-based spatial modeling (Fan et al., 2019) in using KNN. This is standard practice in the GNN-for-PDEs literature. Criticizing KNN without evidence that mesh adjacency would perform better is scope creep.
- **Physical parameter conditioning may be too weak (Harsh Critic §3.2).** The channel attention in Eq. (3) is a standard and well-motivated design (following Takamoto et al., 2023). Claiming it cannot capture higher-order interactions is speculative.
- **Potential train-test information leakage (Harsh Critic).** The concern about physical parameters/boundary conditions at test time is reasonable in principle, but without evidence that the datasets actually hide this information at test time, this is conjecture rather than an identified flaw.
- **Unfair comparison due to SPARK using more side information (Harsh Critic, Reviewer 2).** The baselines do not use boundary/parameter conditioning in Table 1; however, the asymmetry disadvantages SPARK's competitors, not SPARK—this makes SPARK's advantage conservative rather than inflated. Per the rules, this should be removed as a weakness.
- **Confidence intervals on single-run results (Harsh Critic §4.2).** For large-scale PDE and weather benchmarks (ERA5, Prometheus), single-run evaluation is standard. This is a nice-to-have at best.
- **Demand for theoretical proof of ODE stability under quantization (Harsh Critic, Reviewer 2).** Requesting a theoretical bound on how VQ error propagates through ODE integration is beyond standard expectations for an empirical systems paper at ICLR.
- **Broader impact discussion (Harsh Critic).** This is a formatting/completeness issue rather than a scientific weakness.
- **Why is transfer fine-tuning sometimes worse for baselines (Harsh Critic §4.5)?** The observation that SimVP without SPARK slightly degrades at higher SEVIR data fractions (Table 3) is a known phenomenon (negative transfer) and does not imply baseline misconfiguration.

---

## Novel Insights

The most genuinely novel conceptual observation—under-discussed in the sub-reviews—is that the VQ discretization step serves a dual purpose: it compresses physics-rich representations for efficiency, and it implicitly defines a manifold of physically plausible states as the codebook. Augmentation then amounts to interpolating on this physics-constrained manifold rather than in unconstrained input space, which is a principled mechanism for generating physically plausible synthetic samples. This is a more interesting idea than standard latent-space mixup because the codebook geometry is shaped by physical priors. However, the paper does not articulate or test this interpretation explicitly—whether the VQ codebook actually organizes by physics modes (rather than arbitrary clusters) remains unvalidated. Demonstrating this (via codebook visualization colored by physical parameters) would substantially elevate the conceptual contribution.

---

## Suggestions

1. **Add a full ablation table** as the highest priority. At minimum: (a) full SPARK, (b) SPARK without augmentation (just the Fourier ODE), (c) SPARK without Fourier ODE (just augmentation + standard predictor), (d) SPARK without boundary encoding, (e) SPARK without VQ (continuous latent). This directly answers what drives the improvement.

2. **Define the OOD protocol precisely** for each dataset—what parameter/condition is shifted, what range constitutes in-distribution vs. out-of-distribution, and whether test-time physical parameters are available. Add this as a table in the appendix.

3. **Reconcile and explain the number discrepancies** between Table 1 and Table 4 (Prometheus and ERA5). If the model configurations differ, state which variant is used in each table and why.

4. **Specify the augmentation target:** explicitly state that Y_i is held fixed during augmentation of X_i, and provide a justification (e.g., augmented and original samples share the same physics regime, so the same future trajectory is a valid target under the VQ interpolation).

5. **Fix Equation 7** to normalize by K: $v_i = \lambda h_i + \frac{1-\lambda}{K}\sum_{n=1}^K e_n$, or explicitly justify the un-normalized form.

6. **Audit ERA5 baselines.** Reproduce FNO/UNO/CNO on ERA5 with standard hyperparameters from their respective papers and report the configuration used. The order-of-magnitude error gap vs. NMO is suspicious and needs explanation.

7. **Define δ in Eq. 8** and clarify the attention mechanism (is $\alpha_i^t$ normalized? softmax over t?). Define H^l in Eq. 9 and clarify what dimension the DFT operates over for graph node features.

---

**Evaluation axes:**

- **Novelty:** Moderate-to-good. The specific combination of physics-informed VQ augmentation for dynamical system OOD robustness is a concrete and original contribution. Individual components (VQ-VAE, boundary GNN, Fourier ODE) are established, but the integration and motivation are non-trivial.
- **Technical soundness:** Weak. Multiple equations are under-specified, no ablations exist, and the theoretical analysis is disconnected from the proposed design. The method as written is not fully reproducible.
- **Empirical support:** Mixed. The breadth of benchmarks and the plugin validation across backbones are genuine strengths. However, undefined OOD protocols, unexplained number inconsistencies across tables, and suspicious baseline results on ERA5 significantly weaken the evidential strength.
- **Significance:** Moderate. The problem (OOD-robust PDE modeling under data scarcity) is important and practically relevant. If the ablations confirm that the augmentation plugin independently drives improvement, the significance is high; currently this is unestablished.
- **Clarity:** Below acceptable ICLR standard. Key symbolic definitions are missing or inconsistent (δ, H^l, z_i as index vs. embedding), the augmentation procedure is not fully specified, and the absence of ablations leaves the contribution ambiguous.

---

## OclHGmt2ZM

- GT: Reject (avg 5.2)
- Predicted: Reject (4.7/10)
- Match: YES

### Final Review

---

## Summary

CMamba is a Mamba-based model for multivariate time series forecasting that addresses vanilla Mamba's lack of cross-channel dependency modeling. It introduces three components: (1) M-Mamba, a modified Mamba block tailored for time-series patches that removes the convolution branch, uses a feature-independent transition matrix A, and makes the skip connection D data-dependent; (2) GDD-MLP, a global data-dependent MLP for efficient cross-channel mixing; and (3) Channel Mixup, a training-time augmentation that linearly combines channels within a sample to reduce overfitting in channel-dependent models. Experiments across seven standard forecasting benchmarks show competitive performance with significantly lower computational overhead than attention-based channel-mixing alternatives.

---

## Strengths

- **Demonstrated computational efficiency of GDD-MLP.** Table 5 provides direct FLOPs measurements showing a 0.35–1.35% overhead increase even for the 862-channel Traffic dataset. This concretely supports the efficiency claim over self-attention-based channel mixing, and is a specific, quantified contribution rather than a qualitative assertion.

- **Modular versatility validated across diverse architectures.** Table 4 shows consistent improvement when GDD-MLP and Channel Mixup are inserted into four architecturally distinct models — iTransformer (CD/Transformer), PatchTST (CI/Transformer), RLinear (CI/Linear), and TimesNet (CD/Conv) — with an average ~5% gain. The especially large gains on CI models (PatchTST: 17.8% MSE improvement on Electricity) suggest the modules address a genuine gap in CI architectures. This cross-model portability is a meaningful contribution that distinguishes the paper from pure CMamba-architecture engineering.

- **Insightful diagnosis of Mamba component suitability for time-series patching.** The ablation in Table 2 reveals that vanilla Mamba's convolution branch is indeed redundant in a patch-based regime, and that making the skip-connection D data-dependent provides incremental benefit. While the ablation is narrow (Weather only), the findings are conceptually coherent: patching already provides local aggregation, making conv redundant, and D-dependence addresses within-patch variation across channels. This is a non-obvious, empirically-grounded contribution that differs from simply transplanting Mamba to a new domain.

- **Clear explanation and concrete ablation of Channel Mixup's role.** Table 3 directly demonstrates that GDD-MLP alone degrades performance on Traffic (MSE 0.479 → 0.525) while GDD-MLP + Channel Mixup recovers and surpasses the baseline (0.444). This is an honest and useful empirical finding, and the optimization loss curves in Fig. 4 visually corroborate the overfitting story.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of Mamba-based baselines in Table 1.** The paper explicitly discusses S-Mamba, Bi-Mamba+, and Time-SSM in the Related Work section and even criticizes S-Mamba for "large computational overhead." Yet none of these appear in the main comparison table. For a paper whose core claim is a *better adaptation of Mamba* for multivariate forecasting, the absence of the closest competitors leaves the most important performance claim (superiority within the SSM family) entirely unsubstantiated.

- **Table 1 highlighting errors that misrepresent the state-of-the-art claim.** From the text-extracted table, CMamba's results are highlighted as best (red) in settings where they are not:
  - ETTh2 MSE: CMamba 0.273 vs. ModernTCN 0.228 — ModernTCN is clearly better, yet CMamba is highlighted red.
  - ETTm2 MSE: CMamba 0.468 vs. TimesNet 0.412 / RLinear 0.422 — CMamba is markedly worse, yet highlighted red.
  The paper claims "top 1 in 65 out of 70 settings" — this count needs careful reexamination. These errors directly undermine the paper's primary empirical contribution claim.

- **Ambiguity in GDD-MLP's cross-channel interaction mechanism.** Eq. 5 applies MLP_1 and MLP_2 to `Pooling(H_t)`, where pooling over the embedding dimension gives descriptors F^l ∈ R^{V×N}. The critical question — whether the MLP processes the full V×N descriptor *jointly* (enabling cross-channel information flow) or operates *independently per (v,n)* entry (reducing it to SE-style per-channel dynamic scaling with no actual cross-channel dependency) — is never made explicit in the paper. If the latter, GDD-MLP does not capture cross-channel dependencies at all, only performs dynamic per-channel modulation, which contradicts the central contribution claim. The paper must clarify the operating dimension of MLP_1/MLP_2 and provide evidence that cross-channel information flows through the weight generation process.

- **Severe GDD-MLP failure on Traffic is under-analyzed.** Using GDD-MLP alone degrades MSE on Traffic from 0.479 to 0.525 (a ~10% regression). The paper attributes this to CD overfitting but provides no further analysis: no train/validation gap comparison, no exploration of whether the failure correlates with channel count (Traffic has 862 channels), and no discussion of when one can expect GDD-MLP to help vs. hurt. This failure mode is significant enough that practitioners cannot safely apply GDD-MLP standalone without Channel Mixup, and the conditions under which it is safe are unknown.

### Minor

- **M-Mamba ablation is limited to a single dataset (Weather).** The three design choices (remove conv, feature-independent A, data-dependent D) are core architectural contributions of M-Mamba. Validating them only on Weather — where effect sizes are tiny (0.240→0.237 MSE) — is insufficient to support the strong architectural claim that "convolution operation and gated z-branch are redundant for time series forecasting." Multi-dataset validation is needed.

- **GDD-MLP equation block (Eq. 11) suggests a sequential layout inconsistent with Fig. 2's parallel illustration.** Eq. 11 feeds M-Mamba output directly into GDD-MLP with only a post-GDD-MLP residual, while Fig. 2(b) depicts them as parallel branches. The actual data flow needs to be unambiguously stated to ensure reproducibility.

- **A ∈ R^3 for M-Mamba is stated but not explained.** The paper says "A ∈ R^3 in M-Mamba" without clarifying whether this means the state size S=3 or a different parameterization. If S=3, this is an unusually small state and the rationale for this choice is missing.

- **No end-to-end runtime or memory comparison with key competitors.** While Table 5 reports FLOPs for GDD-MLP in isolation, the paper never reports end-to-end training/inference time or memory footprint against iTransformer or S-Mamba. Since efficiency is a primary motivation for GDD-MLP over self-attention, this comparison should be direct and system-level.

### Tiny

- **Channel Mixup hyperparameter σ is not discussed in the main paper.** The distribution of λ is Gaussian with mean 0 and std σ, but σ's value and its sensitivity are deferred entirely to the appendix. Even a single sentence noting the default value and its robustness would help assess reproducibility from the main text.

- **Notation inconsistency in Section 3.1.** The paper writes "X_{v,v}" for the sequence of channel v, which should presumably be "X_{:,v}." This is a minor typo but affects readability of the preliminary setup.

---

## Nice-to-Haves

- **Controlled experiment isolating data-dependence vs. global receptive field in GDD-MLP.** The paper argues MLP fails because it lacks *both* data-dependence *and* global receptive field. A 2×2 ablation (local/global × static/dynamic) would directly validate which factor drives performance and would substantially strengthen the paper's explanatory contribution.

- **Sensitivity analysis for Channel Mixup σ.** Since the λ ~ N(0, σ²) can generate negative values and arbitrarily large perturbations, understanding how σ affects performance would help practitioners configure the method and clarify whether it is functioning as true channel mixup or as noise injection.

- **Analysis of GDD-MLP channel dependency patterns.** Weight heatmaps from GDD-MLP across correlated vs. uncorrelated channel pairs (e.g., in ETT where HULL/MULL relationships are explicitly cited as motivation) would provide interpretable validation that the module captures the claimed type of cross-channel dependency.

- **Discussion of limitations in the conclusion.** The paper should acknowledge: (a) semantic risks of Channel Mixup for heterogeneous physical variables; (b) conditions under which GDD-MLP alone may degrade performance; (c) scope of efficiency claims (per-module FLOPs vs. end-to-end cost).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Eq. 8 contradicts Algorithm 1" (Reviewer 2).** Eq. 8 describes the formula for one virtual channel X' ∈ R^{L×1} resulting from mixing channel i and j. Algorithm 1 implements the vectorized version that applies this to all V channels simultaneously (X' = X + λ * X[:,perm]), yielding shape L×V. These are not contradictory — one is a per-channel formula, the other the vectorized implementation.

- **"Table Traffic MAE shows CMamba=0.645 vs iTransformer=0.262, CMamba highlighted as best."** This is a PDF extraction artifact producing garbled column alignment on the Traffic row. The paper's text explicitly states CMamba performs well on Traffic ("as well as or better than iTransformer"), and the claim about 65/70 top-1 rankings — while needing verification — is likely based on the actual formatted table. This should be verified against the original PDF, not penalized as an authorial error.

- **"GDD-MLP resemblance to SE-blocks overstates novelty" (Reviewer 2).** While the resemblance is real and the paper should acknowledge it, SE-blocks in CNNs operate on spatial feature maps and are not typically applied as cross-sequence channel mixers in temporal/sequence modeling. The application context and specific adaptation (avg+max pooling over patch×embedding for sequence-structured data) are different enough that this is not a fatal novelty concern. It is a reasonable acknowledgment to add, not a critical weakness.

- **"Semantic invalidity of Channel Mixup for heterogeneous channels" (Reviewer 1) — weakened.** The paper explicitly addresses this: channels share temporal characteristics (CI strategy effectiveness is cited), and λ ~ N(0,σ²) with mean 0 ensures the channel's own characteristics are preserved on average. The empirical ablation (Table 3) shows it consistently helps across heterogeneous datasets like Traffic and Weather. The concern is legitimate as a limitation to acknowledge, but does not invalidate the method.

- **"Requesting confidence intervals/variance statistics as a fatal weakness."** Across seven datasets and 70 settings with three runs, requesting standard deviations in the main table is desirable but not standard practice in the MTSF literature at current norms. The lack of variance reporting is a minor concern (especially given the small effect sizes in Table 2), not a major one.

- **"Using baseline numbers from iTransformer weakens comparability"** — weakened. The paper explicitly reruns the three baselines (MICN, TimeMixer, ModernTCN) whose experimental settings differed from iTransformer's. Reusing others from iTransformer under confirmed-identical settings is acceptable practice in the MTSF literature.

---

## Novel Insights

The most genuinely novel observation in this paper — beyond its own stated contributions — is the empirical finding that **vanilla Mamba's convolution branch becomes redundant *specifically because of patching***: patching already provides local temporal aggregation, making the conv1d redundant (and potentially harmful by creating conflicting inductive biases). This insight about the interaction between tokenization strategy and architectural redundancy has implications beyond CMamba — it suggests that researchers transplanting Mamba-style SSMs to other structured sequence domains should revisit which internal components are load-bearing given their specific tokenization. The Channel Mixup failure-then-recovery on Traffic (Table 3: 0.479 → 0.525 with GDD-MLP alone → 0.444 with both modules) also surfaced a useful empirical principle: channel-dependent methods appear to require explicit regularization at high channel counts, and within-sample channel augmentation is one effective form.

---

## Suggestions

1. **Add S-Mamba and Bi-Mamba+ to Table 1.** This is the most critical gap. At minimum, report CMamba vs. S-Mamba on a representative subset of datasets to substantiate the claim of improved Mamba adaptation.

2. **Audit and correct Table 1 highlighting.** Verify the ETTh2 MSE and ETTm2 MSE rankings in the camera-ready version; if the numbers extracted here are accurate (ModernTCN 0.228 < CMamba 0.273 on ETTh2, TimesNet 0.412 < CMamba 0.468 on ETTm2), the highlighting must be corrected and the 65/70 count revised.

3. **Clarify the operating dimension of MLP_1/MLP_2 in GDD-MLP.** Add one sentence (and ideally a shape annotation in an equation) specifying whether the MLP processes the V×N descriptor jointly across channels or per-patch/per-channel. If the intention is joint processing, make this explicit as the source of cross-channel information flow.

4. **Extend the M-Mamba design ablation to at least 3 additional datasets** (e.g., ETTh1, Electricity, Traffic) to validate that removing the conv branch generalizes beyond Weather.

5. **Specify σ for Channel Mixup in the main paper** and include at least a brief sensitivity plot or range, as this is a key hyperparameter of a core contribution.

6. **Explicitly discuss the relationship to SE-Net/Squeeze-and-Excitation** in the related work or methodology, and articulate why GDD-MLP's formulation (avg+max pooling over N×E for temporal sequences) is better suited to the MTSF setting than a direct SE-block application.

---

**Evaluation summary:** The paper addresses a real and timely problem with a computationally efficient design that shows genuine empirical promise. The modular portability result (Table 4) and the efficiency result (Table 5) are its cleanest contributions. However, the paper is held back by: missing Mamba-baseline comparisons that are essential to its core claim, apparent table-highlighting errors that call its performance numbers into question, and insufficient clarity on whether GDD-MLP actually achieves cross-channel information mixing versus per-channel dynamic scaling. Novelty is moderate; technical soundness is adequate but with reproducibility gaps; empirical support is broad in dataset coverage but has notable methodological gaps (narrow ablations, missing baselines). As submitted, it falls short of a confident acceptance at ICLR but has the elements of a solid paper with targeted revisions.

---

## AnPEfzBstD

- GT: Reject (avg 3.5)
- Predicted: Reject (4.0/10)
- Match: YES

### Final Review

## Summary

This paper presents a large-scale empirical benchmark—over 2,100 runs across four datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS)—comparing 1D, 2D, and 3D molecular representations as feature extractors within Bayesian optimization (BO) loops for materials discovery. The benchmark combines MPNN (2D), Equiformer v2 (3D), and MolFormer (1D LLM) feature extractors with GP and linearized Laplace approximation (LLA) surrogates, evaluating performance across data regimes and transfer-learning settings. The central finding is that 1D/2D representations consistently match or outperform 3D representations in BO, with the pretrained 1D LLM (MolFormer) being the strongest performer on most datasets.

---

## Strengths

- **Scale and statistical robustness of the benchmark:** 35 configurations per dataset, over 2,100 total runs averaged across 15 seeds with reported standard errors, covering four chemically distinct datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS). This scale is uncommon in representation-learning benchmarks and provides a meaningful empirical foundation.

- **Sample-complexity characterization of equivariant models:** Section 5.2 provides direct empirical validation that 3D equivariant models require substantially more training data to match 2D baselines, aligning with and extending the theoretical result of Elesedy & Zaidi (2021) into the sequential decision-making setting. This is a specific, actionable finding for practitioners.

- **Inclusion of both GP and LLA surrogates:** Testing two qualitatively different approaches to uncertainty quantification (kernel-based GP on frozen embeddings vs. Laplace-linearized BNNs) makes the comparison more robust and reveals that the dimension-ordering result is consistent across surrogate families.

- **Anonymous code release:** A reproducible codebase is provided, adhering to open-science standards.

---

## Weaknesses

### Fatal

None that fully invalidate the paper, but the Major weaknesses below collectively undermine the central interpretive claim.

---

### Major

- **Uncontrolled pretraining confound—the paper's central framing is misleading.** MolFormer is a large transformer pretrained on millions of molecules from large-scale chemical databases, whereas MPNN and Equiformer v2 are constrained to ~1.5M parameters and appear to be trained from scratch or only on QM9 (Section 4: "the GNN feature extractors are constrained to similar sizes, with each containing approximately 1.5 million parameters"). The headline conclusion—"1D beats 2D/3D"—is therefore primarily a statement about *large pretrained foundation model beats small task-trained models*, not about representational dimensionality. This confound is never disentangled anywhere in the paper. Without a controlled comparison (e.g., a pretrained 2D/3D GNN at comparable data exposure, or an unpretrained/fine-tuned LLM), the dimensionality framing is unjustified. This is the most significant flaw because it affects the interpretation of essentially every result in the paper.

- **Internal inconsistency between abstract/conclusion and Section 5.1.** The abstract states: "LLM methods consistently outperform," and the conclusion opens with: "Across all datasets examined LLMs consistently outperformed both 2D and 3D models." However, Section 5.1 explicitly states for QM9: "Contrary to all other datasets, LLMs performed worse than 2D and 3D models." This is a factual internal contradiction. QM9 is not a marginal dataset—it is the dataset on which encoders are pretrained. An inconsistency of this kind in the central claim is a serious presentation failure.

- **Task selection is systematically biased against 3D representations, but the title draws a general conclusion.** All target properties—atomization energy (QM7), HOMO-LUMO gap (QM9), and absolute energy (GEOM)—are largely determined by molecular connectivity (2D topology) rather than precise 3D geometry. The paper acknowledges in the conclusion that "future research should focus on… tasks where 3D information might be more important, e.g. protein docking," but still titles the paper "Is 3D A Step Too Far For Optimizing Molecules?" and draws general conclusions. The experimental scope only justifies a conclusion about quantum scalar property optimization on small organic molecules under equilibrium geometry—not molecular optimization broadly. The negative 3D result may simply reflect that the chosen tasks do not require 3D information, not that 3D is unhelpful in principle.

- **Conformer handling for 3D models is entirely unspecified.** GEOM provides multiple conformers per molecule, yet the paper gives no information on whether the lowest-energy conformer, a random conformer, or some ensemble is used in 3D experiments. This is critical: using a suboptimal conformer could entirely account for 3D's poor showing relative to 2D, and readers have no way to assess or reproduce the 3D results without this information.

- **No computational cost measurements despite central efficiency claims.** The abstract and aggregated results discussion both frame the contribution as evaluating the "trade-off between computational cost and predictive accuracy." However, no wall-clock time, FLOPs, memory usage, or conformer-generation time is reported anywhere. The claim that "computational overhead of 3D models often outweighed their predictive performance" (Section 5.1) is stated as a fact but has no empirical support in the paper.

---

### Minor

- **Acquisition function and BO implementation details absent.** Section 2.1 introduces BO abstractly but does not specify which acquisition function is used in experiments, whether evaluation is sequential or batched, or whether there is observation noise. For a benchmarking paper at ICLR, these are reproducibility requirements, not minor details.

- **Transfer-learning section is too narrow to support its conclusions.** Figure 5 covers only QM7 and QM9 (two out of four datasets), and the LLM is excluded from the direct comparison (only transfer learning perspective). The conclusion that "foundation models prove a good tool to leverage in molecular optimization" is drawn from a comparison between 2D and 3D models alone on two datasets, making it speculative.

- **Fig. 2 aggregation methodology is unexplained and potentially misleading.** The paper aggregates all 1D models (including the large pretrained MolFormer) into a single curve. Since MolFormer dominates most individual results, this curve primarily reflects MolFormer's performance rather than the average 1D behavior. The weighting across datasets of different sizes and property types is not described. The dramatic gap between 1D and 2D/3D in Fig. 2 should be decomposed by model type.

- **No statistical significance testing.** Results are reported with standard error over 15 seeds, but no formal tests (e.g., bootstrap, Wilcoxon) are performed. Given overlapping error bars visible in some figures, it is unclear whether reported differences between 2D and 3D are significant.

---

### Tiny

- The LLA covariance notation in Section 2.2 writes $\mathcal{N}(\theta_*, \Sigma_*^{-1})$ where $\Sigma_*^{-1}$ is described as "determined by the inverse Hessian," conflating the Hessian (precision) with its inverse (covariance). This is a minor notational inconsistency with no impact on the experimental results.

---

## Nice-to-Haves

- **Control for pretraining data scale:** Even a brief analysis using a pretrained 2D GNN (e.g., a graph foundation model) or a version of MolFormer fine-tuned from scratch on QM9-scale data would help isolate dimensionality from pretraining effects. This would substantially strengthen the core claim.

- **At least one 3D-critical task:** Including even one conformer-sensitive property (e.g., internal energy at 298K, stereoselectivity, or a conformer-ranking task from GEOM) would make the scope of the negative result much more defensible and informative.

- **Uncertainty calibration analysis:** Calibration plots (reliability diagrams, or rank correlation between predicted uncertainty and error) per representation-surrogate combination would clarify whether 3D's BO underperformance stems from worse point predictions or from poorly calibrated uncertainty—two root causes with very different remedies.

- **Per-dataset decomposition of aggregated Figure 2:** Showing the 1D/2D/3D curves decomposed by model class (e.g., separating MolFormer from fingerprint GP in the 1D category) would make the aggregated result more interpretable and honest.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's complaint about Section 2.3 being encyclopedic rather than benchmark-specific:** This is a style/writing nitpick. The background section is appropriate for a paper with broad audience (chemists and ML practitioners).

- **Critic's complaint about LLA not being justified over ensembles or MC dropout:** The paper uses two standard surrogate families (GP and LLA); this is a reasonable and defensible scope. Demanding coverage of all BNN variants is scope creep.

- **Critic's complaint about the Laplace approximation notation being "inconsistent":** The $\Sigma_*^{-1}$ notation is sloppy but standard shorthand in the Laplace literature. It is a minor notation issue, not a scientific error.

- **Critic's complaint about "why do nobody use them" phrasing being imprecise:** This is a stylistic/rhetorical point about the figure caption, not a scientific issue.

- **Critic's complaint that BO barely beats random search on DRUGS as "potentially troubling":** This is actually reported transparently in the paper ("GP regression and random search performed similarly"), and the paper acknowledges this rather than hiding it. It is not a fabricated result; it is an honest finding about a hard dataset.

- **Demand for theoretical proofs or user studies:** Not standard for an empirical systems/benchmarking paper.

- **Demand for confidence intervals on large-scale benchmarks where single-run evaluation is norm:** Already above norm with 15 seeds; formal statistical testing is nice-to-have, not required.

---

## Novel Insights

The most genuinely novel empirical finding in this paper—beyond recapitulating known supervised-learning results—is the direct measurement of data hunger in equivariant models within the BO loop itself: 3D equivariant models require training sets exceeding ~10,000 molecules to approach 2D MPNN performance, even in the sequential optimization setting where labeled data accumulates gradually. This operationalizes the theoretical prediction of Elesedy & Zaidi (2021) in a realistic discovery context and provides concrete guidance for practitioners: when operating in typical drug/materials discovery library sizes (thousands, not hundreds of thousands), investing in 3D featurization is unlikely to pay off relative to a well-pretrained 2D or 1D model, at least for the class of scalar quantum-chemical properties studied here. However, this insight is currently entangled with the pretraining confound, and its full value cannot be realized until that confound is controlled.

---

## Suggestions

1. **Deconfound pretraining from dimensionality (essential).** Either (a) use a pretrained 2D GNN (e.g., a graph foundation model trained on comparable data to MolFormer) as the 2D representative, or (b) train MolFormer from scratch on QM9-scale data to match the 2D/3D training conditions, or (c) report separate curves for pretrained-MolFormer vs. from-scratch GNNs. Without this, the paper's central framing is not scientifically defensible.

2. **Document conformer handling explicitly.** State in Section 4 which conformer(s) from GEOM are used (lowest-energy, random, ensemble-averaged) and, if possible, include an ablation over conformer choice. This is critical for reproducibility and for interpreting the 3D negative result.

3. **Quantify computational costs.** Add a simple table reporting wall-clock time per BO iteration (conformer generation + forward pass + surrogate update) for each feature extractor. This directly substantiates the paper's core cost-accuracy trade-off claim.

4. **Fix the abstract/conclusion inconsistency on QM9.** Change "LLM methods consistently outperform" to an accurate summary that acknowledges the QM9 exception.

5. **Separate MolFormer from the "1D" aggregate in Figure 2.** Show curves for (a) MolFormer, (b) fingerprint GP/LLA, and (c) SMILES GP separately so readers can see what drives the 1D aggregate.

6. **Scope the title and conclusion appropriately.** The claim "3D is a step too far for optimizing molecules" is only supported for scalar quantum-property optimization on small organics under equilibrium geometry. A scoped title (e.g., "…for quantum property optimization in closed-library BO") would be more accurate and still interesting.

---

**Overall assessment:** The paper addresses a practically relevant question with commendable experimental scale. However, it is currently **moderately weak** in its present form. The pretraining confound is not a subtle concern—it affects the interpretation of the headline result throughout. The internal inconsistency in the central LLM claim, the unspecified conformer handling, and the absence of any cost measurements are additional gaps. The paper would need to resolve these issues—especially the pretraining confound—before its conclusions could be trusted at the level required for an ICLR publication.

---

## 6S4WQD1LZR

- GT: Accept (Poster) (avg 6.7)
- Predicted: Accept (7.0/10)
- Match: YES

### Final Review

## Summary

This paper establishes universal approximation theorems for deep transformers operating as "in-context learners" over an arbitrary (even infinite) number of tokens. The authors model token contexts as probability measures equipped with the Wasserstein distance and prove that, for any fixed approximation precision ε, a single transformer with **fixed embedding dimension** (d + 3d') and **fixed number of heads** (proportional only to the output dimension d') can uniformly approximate any continuous in-context mapping over the space of measures. Two main results are proved: Theorem 1 for the unmasked (bidirectional) setting via an elegant Stone–Weierstrass argument using a generalized Laplace-like transform, and Theorem 2 for the masked (causal) setting, which requires additional Lipschitz-context and identifiability assumptions handled via a space-time lifting.

---

## Strengths

- **Fixed embedding dimension independent of context length and precision.** Unlike Yun et al. (2019), which requires the embedding dimension to grow with the number of tokens, and unlike other related universality results that require width to scale with approximation precision, the paper establishes that a single fixed-width transformer is expressive enough for all context sizes simultaneously. This directly addresses a well-identified gap in transformer expressivity theory and is stated precisely in Theorem 1.

- **Elegant and technically non-trivial proof via a measure-valued Laplace transform.** The point-separation step in Proposition 1 reduces to showing injectivity of a novel generalized Laplace-like transform L(μ) (Eq. 16). This is a mathematically clean argument and the algebraic structure exploited—products of elementary in-context functions via depth and MLP approximation of componentwise multiplication (Lemma 3)—is a genuine technical contribution that explains why depth is essential in a way not seen in standard MLP universality proofs.

- **Space-time lifting for the causal/masked case.** The introduction of time as an auxiliary dimension to restore permutation invariance in causal attention (Section 2.3, Eq. 12–13) is a natural and original idea that resolves the structural obstruction of causality in a clean way. The formalism is self-consistent: Lemma 12 shows compositions of causal identifiable maps stay in that class, enabling the parallel with the unmasked proof.

- **Sharpness analysis of the identifiability assumption.** The paper does not merely impose identifiability for technical convenience but proves (Lemma 13) that uniform approximability by masked transformers *forces* identifiability of the target map, making Definition 3 both necessary and sufficient for the setting. This tightness result meaningfully characterizes the limits of causal transformer expressivity.

- **Unified formalism spanning finite and infinite contexts.** The measure-theoretic reformulation (Eq. 9) captures finite empirical measures and continuous measures under the same framework, providing a principled "mean-field" view of transformers that could serve as a foundation for future convergence and optimization analyses.

---

## Weaknesses

### Fatal
None.

### Major

- **Non-quantitative result: no bound on depth or parameter growth.** The paper explicitly acknowledges in Section 3.1 that there is "no explicit control over the dependency of the number of MLP parameters ξ_ℓ on ε," and no bound on how many layers L are needed. Likewise, token magnitudes may grow unboundedly across layers ("our construction does not provide any a priori bound on how the magnitude of the tokens grows through the layers"), which also means the MLP approximation of the squaring operator in Lemma 3 is applied over a domain that is not a priori controlled. This is more than a minor limitation: without any depth-ε or width-ε trade-off, the result is a pure existence theorem that cannot be used to reason about model scaling, approximation efficiency, or practical construction. The paper positions itself as a step toward understanding transformer capabilities, but a universality result with no complexity bound provides very limited information about whether the architecture is efficient or the construction feasible.

- **The H = d' heads constraint, each with d_head = 1, is an architecturally unusual outcome.** The theorem guarantees a "fixed number of heads" but this number scales linearly with the *output* dimension d'. For high-dimensional outputs, this could mean a large number of scalar-output heads, which is far from standard multi-head attention configurations. The claim that "embedding dimension and number of heads are independent of precision" is technically accurate but requires the qualification that they grow with target dimension—a qualification that should appear prominently in the abstract and contributions section, not only in Section 3.1.

### Minor

- **Masked setting restrictions substantially narrow practical scope.** Theorem 2 requires (a) Lipschitz contexts (Definition 1)—with a Lipschitz constant C that blows up as min token time gaps δ shrink—making the theorem non-uniform over sequences of growing length with denser timestamps; (b) causal identifiability; and (c) the atom-at-zero condition $\bar\mu(\{0\}) \geq \sigma$ (which excludes density-valued time marginals, as acknowledged in Remark 2). While the paper addresses (c) via Remark 2's fixed-marginal variant and (b) via Remark 1's sharpness argument, the combination of these restrictions makes the masked universality theorem substantially weaker in scope than its unmasked counterpart. The contrast between the two settings is not sufficiently flagged in the introduction and contributions summary.

- **No coverage of modern positional encodings (RoPE) in the causal setting.** As the paper notes, RoPE is excluded from the current formulation. Since virtually all deployed causal language models use relative or rotary positional encodings, and since the masked theorem's practical relevance depends on encoding positional information faithfully, this is a genuine limitation—not merely a technicality. The paper appropriately labels it as future work, but its importance warrants more discussion than a single sentence in the Conclusion.

- **Injectivity of L(μ) (Lemma 1) receives no main-text intuition.** The entire point-separation argument—the most novel and critical ingredient enabling Stone–Weierstrass—is delegated entirely to Appendix B.1. Since this is the crux of why the transform separates measures (presumably via a connection to Cramér–Wold/Radon-type identifiability), an ICLR audience deserves at least a sentence explaining the key idea. Without this, the central density argument appears to work "by magic."

### Tiny

- **Proposition 1 continuity explanation is slightly imprecise.** The denominator ∫e^{c(⟨x,a⟩+b)(⟨z,a⟩+b)}dμ(z) is not merely "not always zero"—it is strictly positive for all μ ∈ 𝒫(Ω) and x ∈ Ω because the exponential integrand is everywhere positive and μ is a probability measure. The argument is correct but stated loosely.

- **Wasserstein motivation vs. weak\* theorem statement.** The introduction and abstract emphasize Wasserstein continuity as the natural notion of smoothness, but Theorem 1 is stated using weak\* topology. On compact domains these topologies coincide for probability measures, and the paper does note this in the notation section, but the relationship should be stated explicitly near Theorem 1 to avoid confusion.

---

## Nice-to-Haves

- Even a coarse informal discussion of how depth L might scale with ε (e.g., by analogy to MLP approximation of the squaring function) would help readers calibrate whether the construction is exponential or polynomial in 1/ε. The paper hints that the MLP approximation of squaring "should be well-behaved," but this remains unsubstantiated.

- A concrete worked example—such as approximating the mean map μ ↦ ∫y dμ(y) or in-context linear regression—would make the construction tangible and reveal whether token magnitudes stay controlled in practice for simple cases.

- A comparison table contrasting the hypotheses and conclusions of Theorem 1 vs. Theorem 2 would make the distinction between the two settings immediately legible and highlight the price paid for causality.

- Extending or providing further discussion on whether the Lipschitz-context assumption for the masked setting can be weakened when timestamps have a fixed regular structure (e.g., uniform spacing), and how the constant C depends on the minimum time separation δ as sequence length grows.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic: Title overselling** — The abstract is precise and immediately frames universality as an approximation-theoretic result over continuous measure-valued contexts. The title is a fair high-level summary for ICLR.

- **Harsh Critic: Novelty relative to existing work is "diffuse"** — The paper cleanly identifies what it achieves over Yun et al. (2019): fixed embedding dimension, arbitrary context length, fixed heads. This gap is stated precisely.

- **Harsh Critic: Empirical measure loses "multiplicity/order information"** — This is inherent to the measure-theoretic setup and intended. The paper is explicit that permutation invariance is the appropriate structure in the unmasked case; the formalism is not a defect but the point.

- **Harsh Critic: Normalization layers omitted** — The paper states this is "for simplicity," which is standard in transformer theory papers (Yun et al. also omit normalization). Not a defect for a universality result at this level of generality.

- **Harsh Critic: Practical mismatch of construction is a fatal flaw** — This is a pure expressivity paper; demanding a practical construction or connection to training dynamics is out of scope. The Conclusion explicitly and appropriately separates expressivity from learnability.

- **Spark Finder: Requires numerical experiments benchmarking against Yun et al.** — This is a theory paper. No experiments are expected or standard for such results at ICLR theory tracks. Moving to nice-to-have is appropriate.

- **Harsh Critic / Spark Finder: Demanding theoretical bounds as prerequisite for publication** — Non-quantitative universality theorems are published routinely; quantitative bounds would be a significant additional contribution, but their absence does not invalidate the result. Retained as a major weakness rather than a fatal flaw.

---

## Novel Insights

The most genuinely novel insight in this paper—beyond proving the universality theorem itself—is the identification of a **generalized Laplace-like transform** L(μ) (Eq. 16) that is injective on 𝒫(Ω) and can be realized as the output of a single attention head. This provides a new mechanism for measure identification via attention that is distinct from all prior approaches (which relied on fixed-size token representations). Combined with the observation that **products of elementary in-context functions can be realized via depth** (since attention cannot directly multiply), this yields a novel interplay between depth and approximation power specific to the transformer architecture—explaining why depth, not width, is the essential resource in this setting. The space-time lifting for causal attention, while technically natural, also provides a clean framework that may be of independent interest for studying other sequential architectures over continuous-time processes.

---

## Suggestions

1. **Provide main-text intuition for Lemma 1.** Even one sentence explaining why L(μ) injects measures (e.g., via moment-generating-function uniqueness or connection to Cramér–Wold) would substantially improve the paper's accessibility and trust in the core argument.

2. **Quantify or bound the depth–precision relationship informally.** Trace the MLP approximation of x ↦ x² through the squaring error rate (e.g., Yarotsky-type bounds) to give at least a heuristic depth–ε trade-off.

3. **Clarify the H = d' / d_head = 1 constraint prominently.** Add a direct comparison to standard architectures (e.g., H = 8, d_head = 64) to help readers gauge whether the theorem's architectural parameters are within or far from practice.

4. **Discuss uniform-in-n scope of Theorem 2.** Explicitly state that as sequence length grows with dense timestamps (small δ), the Lipschitz constant C in the masked theorem grows as Radius(Ω)/δ, potentially making the set of approximable contexts shrink with n. This clarification is necessary for the "arbitrary number of tokens" claim to be properly qualified in the masked case.

5. **Add a compact summary of the masked-setting assumptions.** A brief itemized list of what "Lipschitz context + identifiability + support condition" means operationally (e.g., "no abrupt distributional jumps in the prefix, no dependence on absolute position when prefix distribution coincides") would help practitioners assess applicability.

---

**Overall character of the paper:** This is a technically sophisticated and original theory paper making a genuine advance over prior transformer universality results. The unmasked theorem (Theorem 1) is strong and the proof technique is elegant and non-trivial. The masked theorem (Theorem 2) is a meaningful extension but carries heavier assumptions that limit its practical reach. The paper's primary limitations—non-quantitative bounds and the absence of standard positional encodings in the causal setting—are acknowledged by the authors. The result is clearly above the threshold of theoretical interest for ICLR, though the non-quantitative nature and the masked-setting restrictions prevent it from being a definitive account of causal transformer expressivity.

- **Novelty:** High. The fixed-dimension universality over measure spaces with the Laplace-transform injection mechanism is new.
- **Technical soundness:** High. The proof strategy is rigorous and the sharpness analysis (Lemma 13) adds credibility.
- **Empirical support:** N/A (pure theory paper; none expected).
- **Significance:** Moderate to high. A foundational result for transformer theory, with the non-quantitative nature somewhat limiting immediate impact.
- **Clarity:** Good overall, with specific gaps around Lemma 1's motivation and the relative scope of the two theorems.

---

## K2jOacHUlO

- GT: Accept (Spotlight) (avg 7.2)
- Predicted: Accept (6.5/10)
- Match: YES

### Final Review

## Summary
This paper introduces **situated faithfulness** — the ability of LLMs to dynamically calibrate trust between internal parametric knowledge and external contexts when the two conflict, as arises in RAG pipelines. To benchmark this, the authors evaluate several QA datasets paired with correct and incorrect contexts, contributing **RedditQA**, a new dataset featuring human-written real-world misinformation from Reddit. They propose two method classes — Self-Guided Confidence Reasoning (SCR) and Rule-Based Confidence Reasoning (RCR) — and a fine-tuning approach, **CR-DPO**, which trains Llama-3-8B via preference optimization over self-sampled confidence-reasoning traces, yielding an average +8.9% gain on situated faithfulness.

---

## Strengths

- **RedditQA fills a genuine benchmark gap.** Unlike all prior work (ClashEval, FaithEval, DynamicQA), which relies on synthetically perturbed incorrect contexts, RedditQA sources incorrect contexts from naturally occurring Reddit posts, providing a qualitatively different test of model robustness to misinformation-as-it-appears-in-the-wild. This is a substantive contribution that no concurrent benchmark offers.

- **The insight that calibration quality ≠ better decision-making is non-obvious and empirically substantiated.** Section 5.3 / Table 3 demonstrates that improving confidence calibration (via isotonic regression, threshold tuning, percentile correction, or self-consistency) does not reliably improve situated faithfulness. The explanation — that a well-calibrated confidence score can still be misaligned with the accuracy-maximizing decision rule — is conceptually precise and not a generic observation. This analysis is one of the most valuable contributions in the paper.

- **The SCR vs. RCR split across model capability levels provides actionable practical guidance.** The finding that strong models (GPT-4o, GPT-4o-mini) benefit from end-to-end SCR while weaker models (Llama-3-8B) are better served by RCR — especially InternalConf via sequence probability — offers concrete guidance for practitioners choosing methods. Tables 1 and 2 clearly support this pattern.

- **CR-DPO ablations are unusually informative.** Table 4 isolates the contributions of CoT, DPO vs. SFT, training task diversity, and trace source (self-sampled vs. GPT-4o). The finding that self-sampled reasoning paths outperform GPT-4o-sourced traces — because confidence reasoning is grounded in the model's own knowledge, which a stronger model cannot share — is a surprising and theoretically motivated result.

- **The RCR signal-rule misalignment taxonomy is a useful contribution.** The paper identifies three distinct failure modes for RCR: (1) flawed/biased rules, (2) noisy/biased confidence signals, and (3) structural misalignment between signal and rule objective. This structured diagnosis goes beyond simply reporting that RCR underperforms.

---

## Weaknesses

### Fatal
None identified.

### Major

- **The central model-capability finding rests on only three models, two of which are closely related GPT-4o variants.** The claim that "stronger reasoning models benefit more from SCR while weaker models benefit from RCR" is a key insight of the paper, but it is supported by exactly three data points: GPT-4o, GPT-4o-mini, and Llama-3-8B. GPT-4o and GPT-4o-mini are not independently diverse architectures. This is far too narrow a base to draw a general conclusion about model capability and method fit. Without additional models spanning a capability spectrum (e.g., Llama-3-70B, Mistral, Gemma), this finding remains suggestive, not established. The paper's framing should be substantially qualified.

- **CR-DPO, the paper's central training contribution, is evaluated on a single model (Llama-3-8B).** All claims about CR-DPO's generality — to unseen tasks, to varying context types — are derived from one architecture and scale. Without evidence across even one additional open-source model, the method cannot be characterized as a general approach to improving SCR in smaller LLMs.

- **No measurement of general capability degradation after CR-DPO.** The paper does not evaluate whether CR-DPO hurts performance on standard benchmarks (e.g., MMLU, reasoning tasks). A model that improves situated faithfulness by learning to systematically distrust external contexts could be practically harmful if it simultaneously becomes overconfident in its parametric knowledge in general QA settings. This omission is a significant gap for any claim of practical utility.

- **Heavy GPT-4o involvement in dataset construction creates potential evaluation circularity.** GPT-4o participates in claim filtering, question generation, context modification, and context verification for several datasets. GPT-4o and GPT-4o-mini are then among the primary evaluated models. This introduces a plausible stylistic or inferential alignment between training data generation and evaluation that is not analyzed. The paper does not quantify or mitigate this risk.

- **The benchmark uses binary correct/incorrect contexts, limiting practical relevance.** All experiments pair each question with a completely correct or completely incorrect context. Real RAG pipelines encounter contexts that are partially correct, partially outdated, or partially relevant. The core claim of "situated faithfulness" in realistic deployment conditions is untested under this most natural setting. This is acknowledged nowhere as a limitation.

### Minor

- **No analysis of source-selection behavior.** The paper measures only final answer accuracy, yet its central framing is about *trust* and *confidence reasoning*. Without a breakdown of how often each method selects the internal vs. external answer across the four quadrants (internal correct/wrong × context correct/wrong), it is impossible to determine whether SCR succeeds by reasoning carefully or simply by defaulting to internal knowledge. A confusion-matrix-style analysis would make the "confidence reasoning" claim more than interpretively asserted.

- **The Figure 2 CR-DPO example contains factual errors in reasoning.** The CR-DPO output states "Richard M. Daley served as the mayor of Chicago from 1955 to 1976" — this conflates Richard J. Daley (father, 1955–1976) with Richard M. Daley (son, 1989–2011). The final answer (Chicago) is correct, but the reasoning contains clear factual inaccuracies. As the paper's only qualitative success case for CR-DPO, this is concerning: it suggests the model learns to *argue against* misleading contexts more forcefully, not necessarily to reason from better-organized knowledge.

- **RedditQA is multiple-choice while other datasets use open-ended QA.** Answer format can substantially affect model behavior under conflicting context (e.g., guessing probability, distractor salience). The "Total" metrics aggregate these into a single number without discussing whether this conflation is appropriate.

- **Dataset statistics for RedditQA are not reported in the main text.** Final dataset size, topic distribution, inter-annotator agreement, and filtering rates are deferred to an appendix. For a benchmark contribution at a venue like ICLR, these statistics should be prominent in the paper body.

- **TACS(LR) is a substantially weakened approximation of the original TACS method.** The paper substitutes the original hidden-state classifier with an LLM prompting approach because hidden states are inaccessible for proprietary models. Conclusions drawn from TACS(LR) performing poorly (e.g., that preprocessing approaches fail) are conflated with conclusions about the original method. This should be framed much more carefully: TACS(LR) failing does not mean TACS fails.

### Tiny

- The SF metric equally weights Acc_t and Acc_f, but in many real deployments, correct contexts are far more frequent than incorrect ones. This makes SF overweight robustness relative to utility. This is acceptable as a benchmark stress-test metric but is sometimes written as if it were a deployment objective.

- The conclusion that "SCR operates more effectively in text space" is a plausible interpretation but is asserted mechanistically rather than demonstrated. It could equally be explained by end-to-end prompting avoiding brittle intermediate decomposition errors, which is a simpler explanation.

---

## Nice-to-Haves

- A scaling plot of model capability (e.g., MMLU score or benchmark-derived proxy) vs. SCR–RCR performance gap across more models would be highly impactful and provide genuine support for the capability-conditioned recommendation.

- An evaluation with partially correct or mixed-quality contexts would stress-test situated faithfulness under more realistic RAG conditions and substantially strengthen the practical relevance claim.

- A general capability evaluation (MMLU or similar) before and after CR-DPO would significantly increase confidence in the method's safety and deployability.

- Evaluating CR-DPO on one additional open-source model (e.g., Llama-3-70B or Mistral-7B) would establish whether the training approach generalizes across architectures and scales.

- A confusion matrix decomposing source-selection behavior per method (internal vs. external choice, conditioned on which source is correct) would directly validate or challenge the "confidence reasoning" interpretation of SCR's success.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"DIA is too weak a baseline to justify the +24.2% claim" (Harsh Critic):** DIA is an intentionally naive upper bound on standard RAG systems. The paper also compares against TACS and a full suite of RCR methods. Comparing against DIA to demonstrate the magnitude of the vulnerability is appropriate; the RCR methods serve as the more principled comparison. The asymmetry here is not unfair — it favors the baseline to make a stronger point about the problem severity.

- **"No confidence intervals or statistical significance" (Harsh Critic / Spark Finder):** Single-run evaluation is the standard norm for large-scale QA benchmarks with API models, and many of the reported differences are large (>5 percentage points). This does not meet the bar for a substantive weakness in this research community's standards.

- **"The upper bound of Acc_f is not valid without stronger assumptions" (Harsh Critic):** The paper's claim that "wrong contexts can't help a model answer questions it cannot answer" is a reasonable and approximately correct assumption given the experimental design, where false contexts are specifically constructed to point to wrong answers. While a false context could theoretically contain tangential helpful information, this is an edge case that does not undermine the benchmark design.

- **"The related work does not deeply engage with calibration/uncertainty literature" (Harsh Critic):** The related work is selective but adequate for the paper's scope. Requiring an extensive survey of calibration literature imposes scope creep on a paper whose contribution is empirical and applied, not a theoretical contribution to calibration.

- **"Concurrent work is not deeply compared" (Harsh Critic, multiple reviewers):** The paper cannot be penalized for not comparing against methods whose specifics are not established by an external reviewer; the paper positions against the concurrent work's benchmarks (which it uses in evaluation) and is explicit about what is and is not compared.

- **Requests for missing related works:** Per review instructions, not included, as these cannot be verified without external sources.

---

## Novel Insights

The most genuinely novel insight in this paper — largely surfaced by the spark finder but empirically supported in Section 5.3 / Table 3 — is the **structural misalignment between confidence calibration and accuracy maximization in rule-based systems**. Improving a model's calibration (via isotonic regression, threshold tuning, etc.) does not reliably improve situated faithfulness because a well-calibrated confidence score need not track the binary source-selection decision optimally. This is a theoretically clean insight: calibration optimizes expected calibration error, while the situated faithfulness objective optimizes accuracy — and these are different optimization targets. This finding has implications beyond this paper for any system that uses predicted confidence as a proxy for decision-making in retrieval-augmented settings.

---

## Suggestions

1. **Expand model coverage with at least one intermediate-capability open-source model** (e.g., Llama-3-70B or Mistral-7B) to substantiate the capability-conditioned SCR vs. RCR recommendation, and apply CR-DPO to at least one additional model beyond Llama-3-8B.

2. **Add a general capability evaluation** (e.g., MMLU, ARC, or similar) before and after CR-DPO training to demonstrate the method does not degrade parametric knowledge or broad reasoning — this is essential for any practical deployment claim.

3. **Add a source-selection confusion matrix** breaking down how often each method selects internal vs. external answers conditioned on (internal correct × context correct/wrong), to provide evidence for the "confidence reasoning" interpretation rather than a "default to internal" alternative explanation.

4. **Explicitly discuss the GPT-4o data generation circularity** in the main paper limitations section, and ideally provide a disaggregated analysis of GPT-4o performance on RedditQA vs. non-GPT-4o-generated datasets.

5. **Report RedditQA's key statistics** (final N, domain distribution, inter-annotator agreement, discard rate) in the main body rather than the appendix, given this is a benchmark contribution.

6. **Qualify the Figure 2 CR-DPO example**: either correct the factual dates in the reasoning or acknowledge that CR-DPO teaches adversarial argumentation rather than improved knowledge organization, as this distinction matters for interpreting what the model has learned.

7. **Include at least one experiment with mixed-quality contexts** (partially correct, partially misleading) to probe the robustness of situated faithfulness methods in more realistic RAG conditions.

---

**Evaluation axes:**
- **Novelty:** Moderate-to-good. RedditQA and the SCR/RCR taxonomy are meaningful contributions. CR-DPO is technically a task-specific DPO application, not an algorithmic advance, but is a practical contribution. The calibration-vs-accuracy insight is genuinely novel.
- **Technical soundness:** Moderate. The formulation is clean, but the experimental scope is narrow (3 models, 1 training model, binary context setup), and several core claims are under-supported by evidence.
- **Empirical support:** Adequate for the narrower claims; insufficient for the broader generalization claims (capability-SCR/RCR relationship, CR-DPO generality).
- **Significance:** Good. Knowledge conflicts in RAG are a real, growing problem. The datasets, methods, and analysis fill a genuine gap in the literature.
- **Clarity:** Good. The paper is well-organized, tables are informative, and the SCR/RCR taxonomy is easy to follow. Key implementation details are too heavily deferred to appendices.

---

## Acvo2RGSCy

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.5/10)
- Match: YES

### Final Review

## Summary
DeLLMa (Decision-making Large Language Model assistant) is a framework that scaffolds LLM decision-making under uncertainty using classical decision theory. It decomposes the process into four steps: state enumeration (identifying latent factors via LLM), state forecasting (verbalized probabilities over factor values), utility function elicitation (pairwise LLM rankings fitted via Bradley-Terry), and expected utility maximization (Monte Carlo). The authors evaluate on two real-world domains—agricultural crop planning (USDA data) and stock selection—and demonstrate consistent improvements over prompting baselines and OpenAI o1 (zero-shot), while exposing interpretable intermediate reasoning traces.

---

## Strengths

- **Novel integration of decision theory with LLM inference-time scaling.** Rather than applying CoT or self-consistency generically, the paper constructs a theoretically grounded pipeline around the expected utility principle. This is a specific and actionable framing that is distinct from prior inference-time compute work, and fills a real gap between generic reasoning benchmarks and structured uncertainty quantification.

- **Bradley-Terry utility elicitation from LLM pairwise rankings is an inventive contribution.** Using overlapping minibatch rankings fed into a classic preference estimation model is a concrete and novel procedure that avoids asking the LLM for cardinal utilities directly (which is brittle). The variance-reduction trick of pairing the same sampled states with all actions is a well-designed engineering choice that meaningfully reduces noise.

- **Consistent multi-model and multi-domain improvements.** DeLLMa yields accuracy gains across GPT-4, Claude 3, and Gemini 1.5, and across two structurally different domains (natural language reports vs. tabular price data). Robustness across model families strengthens the generality claim.

- **DeLLMa substantially outperforms o1-preview at comparable cost.** Table 3 shows DeLLMa (at $0.09–$0.37/instance) achieving 73.3% vs. 33.3% on Agriculture and 64.2% vs. 35.0% on Stocks against o1-preview (zero-shot, ~$0.21/instance). Even granting the asymmetry of the comparison (see weaknesses), this is a strong empirical finding suggesting that specialized decomposition adds value over general-purpose inference-time reasoning.

- **Human-LLM utility agreement matches inter-annotator agreement.** Table 4 shows 65–68% human-LLM agreement versus 67.0% ± 6.3% inter-annotator agreement. Rather than interpreting this negatively, this is the right benchmark: if the task is genuinely hard for humans too, the elicited utility is performing as well as a human annotator could.

- **Baseline failure modes are diagnosed.** The paper provides concrete analysis (Appendix C.4, Figures 12/17/18) of why baselines fail—sentiment echoing, inability to reason counterfactually—which is a genuine contribution beyond just outperforming baselines.

---

## Weaknesses

### Fatal
None. The core empirical contributions stand and the methodological approximations, while imperfect, do not undermine the experimental claims.

### Major

- **Independence assumption over latent factors is unvalidated and potentially significant.** Section 3.2 factorizes the state distribution as $\prod_i \pi_i(\cdot|\mathcal{C})$, but factors like climate, yield, and price in agriculture (or economic conditions and individual stock growth in finance) are obviously correlated. The ablation in Table 2 compares DeLLMa to uniform/under/overspecified forecasts, but does *not* compare against a joint specification that accounts for correlations. Since the full expected utility computation depends on the correctness of this distribution, this is a central assumption that remains unquantified in its impact. The paper acknowledges it as a simplification but provides no sensitivity analysis.

- **Benchmark diversity and independence are limited.** Both environments use exactly 7 items and generate 120 instances by enumerating all subsets. These 120 instances are not independent—a model with a fixed preference ranking over the 7 fruits/stocks will have correlated performance across all subsets that include its preferred item. This inflates the effective sample count and limits what can be concluded about generalization. The paper's generalization claims would be substantially strengthened with additional domains or randomly sampled (non-exhaustive) decision instances.

- **Baseline comparisons are incomplete, particularly against o1.** The comparison to o1 is zero-shot only (Table 3). A structured prompt giving o1 the same decomposition scaffold (enumerate states, estimate probabilities, rank state-action pairs, compute expected utility) would test whether DeLLMa's gains stem from the decision-theoretic structure or simply from spending more compute tokens in a particular way. Without this, the claim that "specialized inference-time reasoning outperforms general-purpose reasoning models" is not fully established. Similarly, no baseline directly prompts the LLM to estimate expected utility numerically (a much simpler structured baseline than DeLLMa), making it hard to attribute gains to specific components.

- **Stock evaluation ground truth is noisy and potentially contaminated.** The target is a single month (December 2023) of returns for well-known tickers (NVDA, META, GME, etc.). A single month of returns for highly salient stocks is vulnerable both to stochastic noise (single-month returns are largely unpredictable) and to pretraining contamination (models likely have strong priors about these companies' trajectories). The paper acknowledges the leakage concern but does not resolve it. Multi-period aggregation or less salient assets would substantially improve the validity of this evaluation.

### Minor

- **Bradley-Terry scores used as cardinal utilities lack full justification.** Algorithm 2 returns Bradley-Terry preference strengths, which are ordinally grounded but not necessarily on a ratio scale. Averaging them in Eq. (3) as expected utilities assumes interval-scale properties. This is a reasonable heuristic that works empirically, but the paper would benefit from an explicit acknowledgment that this is an approximation and from discussing whether the results are sensitive to the scale of the elicited scores.

- **Verbalized probability mapping $\mathcal{V}$ is unjustified.** The mapping from {very likely, ..., very unlikely} to numeric values is applied throughout but its sensitivity is never analyzed. Different plausible mappings could shift the state distribution and thus the final decision. A brief sensitivity analysis or justification of the chosen mapping would substantially improve confidence in the forecasting component.

- **State forecasting ground-truth annotation is underspecified.** Table 1 reports ECE and NLL for forecast quality, but the latent factors are free-form strings generated by the LLM. The paper says ground truths were "manually annotated" but does not describe how annotators resolved cases where the LLM's factor definitions are abstract or action-entangled (e.g., "apple yield" as a state factor, which conflates state and action). This makes Table 1 difficult to interpret rigorously.

- **No confidence intervals or significance tests on primary results.** With 120 instances per environment, bootstrap confidence intervals are straightforward and would help distinguish genuine improvements from noise, particularly for the moderate gains seen in the state forecasting ablation (Table 2) and for the stock domain.

### Tiny

- **"Scaling laws" is an overstatement.** The linear trends in Figure 3 with respect to sample size and overlap percentage are informative hyperparameter studies, not scaling laws in any technical sense. This is a minor terminology issue.

- **Human evaluation includes paper authors as annotators.** The 5 external volunteers plus the paper authors constitute a small and potentially biased panel. The result is still useful, but the limitation should be noted more prominently.

---

## Nice-to-Haves

- **Factor correlation modeling.** Even a simple approach—asking the LLM to jointly assign probabilities over combinations of factor values for a small number of high-dependency factor pairs—could improve the fidelity of the state distribution without requiring full joint specification over all $\ell^k$ states.

- **Structured prompt for o1 comparison.** Providing o1 the same decision-theoretic scaffold and comparing its decomposed output to DeLLMa would tightly isolate whether the gains come from the pipeline structure vs. the underlying model, and would give a stronger argument for DeLLMa's design choices.

- **Failure mode analysis.** Characterizing instances where DeLLMa still fails—whether due to bad state enumeration, miscalibrated forecasts, or poor utility rankings—would provide actionable insight for future improvements and strengthen the paper's analytical depth.

- **Compute-vs-performance Pareto frontier.** An explicit plot of accuracy against total API cost across different sample sizes and overlap percentages would help practitioners calibrate DeLLMa for their budget and time constraints.

- **Sensitivity of results to the verbalized probability mapping.** Testing 2–3 reasonable alternative mappings $\mathcal{V}$ would either show robustness (positive) or reveal a hyperparameter that deserves more principled treatment.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Missing related works in preference learning/reward modeling"** (Critic): The paper cites the utility elicitation literature (Farquhar 1984) and LLM-for-ranking work (Lee et al., 2024; Qin et al., 2023). The missing-related-work criticism could invoke non-existent references and is excluded per guidelines.

- **High-stakes framing vs. narrow experiments** (Critic): The paper explicitly scopes itself to a proof-of-concept in its conclusion ("avails many possibilities") and defers portfolio construction to future work. Criticizing the absence of sequential/continuous-action domains beyond the paper's explicit scope is scope creep. The narrow evaluation is already captured as a major weakness about benchmark diversity, which is in scope.

- **"No end-to-end worked example"** (Spark Finder): Figure 3 (right) and Figure 4 (right) show complete decision trees with enumerated states, sampled latent factors, weights (posterior probabilities), and utilities for Agriculture and Stocks respectively. The criticism is factually incorrect for this paper.

- **"Unfair comparison where asymmetry favors the baseline" (o1 zero-shot)**: The o1 zero-shot comparison is *asymmetric in favor of DeLLMa* (more compute, structured pipeline), which is noted as a separate weakness about missing structured-prompt baselines. The criticism of "unfair comparison where asymmetry favors the baseline" does not apply here—the baseline is disadvantaged, not the authors' method, so comparing in this direction proves a stronger point.

- **Circular reliance on LLM for utility elicitation** (Spark Finder): The paper's key insight is that decomposing the task into explicit state-action pairwise comparisons makes the LLM's ranking task tractable even if direct decision-making is not. The empirical results support this design. The "circularity" concern conflates the hardness of the ranking sub-task with the hardness of the end-to-end decision. The concern is philosophically interesting but the empirical gains demonstrate it is not a practical failure mode.

- **"Contribution is diffuse"** (Critic): For a systems/empirical paper at ICLR, a clear framework + empirical validation across multiple models and domains constitutes a well-defined contribution. The claim is unfounded as a standalone criticism.

---

## Novel Insights

The most genuinely novel observation—partially surfaced by the Spark Finder—is that *the independence assumption in state forecasting may matter less than the utility elicitation module* for the final decision quality: Table 2 shows that Uniform, Underspecified, and Overspecified state priors perform similarly to calibrated DeLLMa forecasts on GPT-4 and Gemini 1.5, while DeLLMa-Naive (weaker utility elicitation) is consistently worse than DeLLMa-Pairs/Top1. This suggests that the Bradley-Terry–based utility elicitation—not the probabilistic forecasting of states—is the primary driver of performance, a conclusion the paper does not explicitly draw but which has important implications for where future work should focus: improving the utility elicitation mechanism (e.g., more robust ranking, cardinal alignment) may yield greater returns than refining the state forecasting pipeline.

---

## Suggestions

1. **Decompose the source of gains explicitly.** Run an oracle-state version of DeLLMa (replace LLM state forecasts with realized ground-truth states) and an oracle-utility version (replace Bradley-Terry with ground-truth utility rankings) to quantify each module's contribution. This would definitively settle whether improvements come from better uncertainty modeling or better preference elicitation.

2. **Test on at least one additional domain** with a different action structure—e.g., multi-attribute choices, where simple ranking is harder—to validate the framework beyond "pick one from k options."

3. **Add a structured-prompt o1 baseline.** Prompt o1 with explicit instructions to enumerate states, estimate probabilities, rank state-action pairs, and compute expected utility. This is the natural control for the Table 3 comparison.

4. **Analyze verbalized probability mapping sensitivity.** Try 2–3 alternative scale mappings (e.g., {0.9, 0.7, 0.55, 0.45, 0.3, 0.1} vs. the current mapping) and report whether results are stable. This is a cheap experiment that would substantially increase confidence in the forecasting step.

5. **Clarify the Bradley-Terry / cardinal utility relationship.** Add a brief discussion noting that BT scores are used as a proxy for cardinal utilities and that the validity of expected utility averaging depends on the monotone relationship between BT strengths and true utilities. Acknowledge this as a modeling assumption rather than a rigorous derivation.

6. **Report normalized utility (already in Appendix B) as a co-primary metric.** Accuracy against a single "optimal action" penalizes rational risk-averse choices and conflates decision quality with outcome luck. Normalized utility is a better reflection of decision quality under uncertainty and should appear in the main text.

---

