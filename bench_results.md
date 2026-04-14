# ICLR Benchmark Results

Date: 2026-04-14 02:59
Critic/Merger: openai/gpt-5.4 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## SI6zocV2SS

- GT: Reject (avg 1.5)
- Predicted: Reject (2.1/10)
- Match: YES

### Final Review

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

---

## E6kQ51yfAj

- GT: Reject (avg 4.5)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

## Summary

This paper proposes Game-Theoretical Preference Optimization (GPO), which frames LLM alignment as a two-player zero-sum game between an adversarial prompt-generating agent and a defensive response-generating agent, trained iteratively with PPO. A diversity mechanism (SelfBLEU + sentence-embedding novelty rewards) prevents adversarial mode collapse. The authors prove an O(T⁻¹/²) Nash gap bound for an idealized variant of the algorithm and evaluate on three safety datasets, showing improvements in both defensive robustness and adversarial red-teaming capability.

---

## Strengths

- **Novel joint optimization framing distinguishing it from prior work.** Unlike MART (iterative red-team + SFT hardening) or self-play methods that fix both agents' prompt sets, GPO jointly trains attacker and defender via PPO under a shared game-theoretic objective. The difference from MART—using RL rather than SFT for iteration, and treating both agents symmetrically—is explicitly articulated in the related-work section and is a genuine architectural contribution.

- **Diversity mechanism with demonstrated effectiveness.** The two-component diversity reward (SelfBLEU + embedding-based novelty) concretely solves a mode-collapse problem. Table 2 shows that RLHF without diversity collapses to near-uniform attack patterns (diversity ~0.49–0.52), while GPO+Div recovers diversity to ~0.70–0.86 *while simultaneously increasing attack strength* — a tradeoff that RLHF+Div alone fails to achieve. This result is specific and non-obvious.

- **Bilateral evaluation of both agents.** The paper evaluates both the defensive and adversarial sides, the latter tested as a red-teamer against three *held-out* third-party target models (Llama-2-7b-chat, Vicuna-7b-v1.5, an RLHF model). This is notably more complete than safety alignment papers that only report the defensive agent's performance, and the transfer results in Table 2 support genuine generalization of the adversary.

- **Jailbreak OOD generalization experiment.** Training on some jailbreak methods and testing on withheld "less common" attack types (Table 3) constitutes a meaningful generalization test beyond the main safety datasets, and the gains (ASR 16.67 → 10.42) are consistent with the general safety results.

- **GPO+Div improves instruction-following quality over RLHF.** MT-Bench results (Table 4) show GPO+Div (6.22) outperforms RLHF (6.11) and SFT (5.82), suggesting the safety gains do not come at a helpfulness penalty for the full method — a result that runs counter to common assumptions in safety alignment.

---

## Weaknesses

### Fatal
None.

### Major

- **Theory-practice gap is significant and incompletely disclosed.** Theorem 3.2 guarantees an O(T⁻¹/²) Nash gap for an *idealized algorithm* (Algorithm 2 in appendix) that assumes uniform initialization, exact optimization (PPO error ignored), and returns *average* policies. The deployed system (Algorithm 1) returns the *last iterate*, uses PPO with neural networks, and does not satisfy these assumptions. The paper does acknowledge these changes in Section 3.3 ("we change our practical algorithm a bit"), but then repeatedly states that "the system reaches a Nash Equilibrium" and that "Algorithm 1 can find an approximate Nash equilibrium" as if the theorem applies to the practical procedure. Stronger statements appear in the abstract ("iterative RL optimization converges to a Nash Equilibrium") without qualification. This misrepresentation of the theorem's scope is a substantive issue: the convergence guarantee is the paper's central theoretical claim, and it provably does not apply to what is actually run.

- **Classifier-reward and evaluation confound.** Both the training signal and the evaluation metrics (ASR and r_safe) are defined by the same or closely related toxicity classifier. If the trained models are learning to fool the classifier rather than genuinely becoming safer, the reported improvements would be illusory. The paper provides no evaluation with a different safety judge, no human annotation, and no analysis of whether outputs that score well on the classifier might remain subtly harmful. This is especially important because the adversarial agent is specifically trained against this same classifier reward, creating a shared-oracle confound. Without cross-judge validation, the safety improvements should be interpreted cautiously.

- **Missing iterative red-teaming baseline.** The paper positions itself against MART (iterative SFT-based method) and standard RLHF, but does not include a natural baseline: simply alternating PPO-based attacker and defender training *without* the game-theoretic formulation or diversity constraints. Without this, it is impossible to determine whether the gains of GPO over RLHF stem from (a) iterative co-adaptation, (b) the game-theoretic joint objective, (c) diversity rewards, or (d) some combination. MART is discussed in related work as a direct predecessor but is not used as a comparison point in any table.

- **Adversarial agent formulation inconsistency between theory and practice.** In Equation (3.1), µ_φ is an *unconditional* distribution over prompts. But the experimental pipeline describes the adversary as "transforming original harmful prompts into similar but more harmful variations" — a *conditional* rewriting model µ_φ(x′ | x_seed). This is a fundamental difference in the action space, and the theoretical analysis (which treats µ_φ as a distribution over the full prompt space) does not straightforwardly cover conditional prompt rewriting. The paper never reconciles this discrepancy.

### Minor

- **No variance or multi-seed reporting.** RL-based training can have substantial run-to-run variance. No confidence intervals or multi-seed statistics are reported in any table. Even reporting results across 2–3 seeds with standard deviations would substantially increase confidence in the numerical claims. This is particularly important for smaller improvements, e.g., GPO vs. RLHF in Table 1 (9.27 vs. 10.89 ASR on Anthropic).

- **Plain GPO degrades MT-Bench quality below RLHF, unexplained.** Table 4 shows GPO (6.02) scores *lower* than RLHF (6.11), while GPO+Div (6.22) recovers. The paper discusses GPO+Div's positive result but does not explain the quality degradation in plain GPO. If the adversary without diversity constraints pushes the defender toward over-refusal or other degenerate behavior, this should be explicitly analyzed rather than passed over.

- **Compute budget is not matched or reported.** GPO trains two LLM agents iteratively, while RLHF trains one. No training time, GPU hours, or token budget comparison is provided. It is possible that GPO's gains are partly attributable to greater total compute rather than the game-theoretic structure.

- **Embedding diversity formula notation.** Equation (3.5) uses ||φ(x)||²||φ(x')||² in the denominator while the surrounding text describes cosine similarity, whose denominator should be ||φ(x)||·||φ(x')||. If this is a notation error rather than a rendering artifact, it describes a different (non-cosine) similarity metric that may not match the intended implementation.

### Tiny

- The average adversarial policy in Theorem 3.2's discussion is written as µ_{θ_t} where it should be µ_{φ_t}, inconsistent with the paper's own notation convention.
- The scalar "Diversity" metric in Table 2 is not explicitly defined in the main text (presumably a combination of SelfBLEU and embedding novelty per Eq. 3.4–3.5), and its normalization is not specified. Raw Data's diversity values (0.91 on Anthropic, 0.56 on BeaverTails) differ substantially for unclear reasons.

---

## Nice-to-Haves

- **Ablation varying number of game iterations T.** The core narrative is convergence via iteration, yet no experiment varies T or tracks how safety/diversity metrics evolve across game rounds. Even a small-scale ablation (T = 1, 2, 4, 8) would directly substantiate the iterative convergence claim.
- **Qualitative examples of adversarial prompt evolution.** Showing how prompts change round-over-round would concretely demonstrate what the co-adaptation discovers and distinguish it from static red-teaming.
- **Empirical Nash gap curve.** Plotting NEGap over training iterations would bridge the theoretical claim and observed behavior, and is directly enabled by the metrics already defined in the paper.
- **Per-target-model breakdown in Table 2.** Currently only averaged over three targets; per-model results would reveal whether adversarial transfers broadly or mainly exploits the RLHF-trained co-trained opponent.
- **Ablation of diversity components.** An ablation comparing SelfBLEU-only, embedding-only, and the combined diversity reward would clarify the contribution of each component given that diversity is a central mechanism.
- **MART as an explicit baseline.** Including MART in the experimental tables would provide a cleaner comparison to the most relevant prior iterative red-teaming method.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Fully trains agents" is vague and unmeasured.** The harsh critic correctly notes this phrase lacks operational meaning, but it reads as informal narrative rather than a falsifiable claim; penalizing it as a scientific flaw is excessive.
- **The Paraphrase baseline is weak.** True, but the baseline is included primarily as an ablation rather than a state-of-the-art comparison; its inclusion does not mislead.
- **Demanding confidence intervals on large-scale benchmarks.** Single-run evaluation is standard for safety RLHF benchmarks in the community; requesting CIs has been noted as a minor concern but removed as a standalone weakness per community norms (retained as a minor weakness because RL variance *is* genuinely high in this specific setup).
- **Diversity ablation (SelfBLEU-only vs. embedding-only) as a major weakness.** Reasonable to request but not essential to validate the core claim; moved to nice-to-have.
- **Criticisms of the "Paraphrase" baseline being unfair to the authors' method.** The comparison is not favorable to the authors' method and is informative; no removal needed, but it is not a meaningful weakness of the paper.
- **Demanding formal notion of "coverage."** The paper uses "coverage" informally as intuition; demanding a formal definition is scope creep for an empirical paper.
- **Claim that mentioning MART without including it constitutes a missing related work.** Related work discussion is adequate; the absence in tables is captured in the major weaknesses.

---

## Novel Insights

The most interesting observation emerging from the three reviews, beyond the paper's own contributions, concerns the synergy between the game-theoretic structure and the diversity reward. RLHF+Div in Table 2 actually *decreases* ASR compared to RLHF (33.60 vs. 37.72 on Anthropic), suggesting that diversity alone makes the adversary *weaker* when facing a static opponent. Only when diversity is combined with iterative co-training (GPO+Div) does ASR increase above plain RLHF. This implies the game-theoretic structure is not merely additive: the adversary needs a *moving defensive target* to justify exploring diverse strategies. This interaction effect is under-analyzed in the paper and, if confirmed with proper controls, would constitute a substantive argument for the joint optimization framework beyond either component alone.

---

## Suggestions

1. **Rewrite all Nash convergence claims in the abstract, introduction, and main text to clearly specify that they apply to the theoretical Algorithm 2 (average policies, no optimization error), not to the deployed PPO system.** Replace "the system reaches a Nash Equilibrium" with "the idealized variant converges to an approximate Nash Equilibrium." This is a calibration fix, not a new experiment.

2. **Add one cross-judge evaluation run** using a safety evaluator different from the training classifier (e.g., GPT-4-based safety scoring or HarmBench) on a subset of outputs to test whether ASR improvements transfer beyond the training classifier.

3. **Include a simple iterative PPO baseline** — alternate training attacker (PPO, safety reward) and defender (PPO, safety reward) for the same number of total steps as GPO, without the game-theoretic joint objective or diversity constraints — to isolate the contribution of the game formulation.

4. **Explicitly reconcile the conditional vs. unconditional adversary.** Either reformulate the theory to cover conditional prompt rewriting (µ_φ(x′|x_seed)) or explain in the main text why the theoretical results extend to this setting.

5. **Report multi-seed results or at minimum error bars** derived from bootstrap resampling of the evaluation set, which is low-cost and would address variance concerns without re-running training.

6. **Analyze and explain the plain GPO quality regression in Table 4** (GPO 6.02 < RLHF 6.11). Is this over-refusal? Does it correlate with diversity collapse in the adversary?

---

## Evaluation Summary

| Axis | Assessment |
|---|---|
| **Originality** | Moderate-high. Joint PPO-based co-training of attacker and defender with diversity regularization in a game-theoretic framing is a meaningful contribution beyond prior iterative red-teaming. |
| **Importance of research question** | High. Adaptive prompt generation for safety alignment is a central open problem. |
| **Claims well supported** | Partially. Empirical gains are plausible and consistent across datasets, but the central convergence claim is substantially overstated relative to what is proved, and classifier-evaluation confound is unaddressed. |
| **Soundness of experiments** | Moderate. Dataset coverage is good; the bilateral adversarial/defensive evaluation is commendable. However, a critical iterative baseline is missing, no variance is reported, and all evaluation rests on a single classifier. |
| **Clarity of writing** | Moderate. The high-level idea is easy to follow, but the theory-practice boundary is blurred throughout and the conditional-vs.-unconditional adversary inconsistency is never resolved. |
| **Value to research community** | Moderate-high. The diversity mechanism, bilateral evaluation framework, and game-theoretic training procedure are genuinely useful contributions to safety alignment research. |
| **Contextualized relative to prior work** | Adequate but incomplete. MART is correctly identified as the closest prior work but excluded from comparison tables, and the key distinction (RL vs. SFT for iteration) needs empirical support. |

---

## slZZnzlITo

- GT: Reject (avg 4.8)
- Predicted: Accept (5.3/10)
- Match: NO

### Final Review

## Summary
This paper proposes Multimodal Open Set Recognition (MMOSR), extending OSR to multimodal data, and empirically diagnoses "fusion degradation"—the failure mode whereby naïvely applying OSR regularization to fused multimodal representations over-compresses the feature space and degrades both closed-set accuracy and unknown detection. To address this, the authors introduce the Multimodal Representation Reactivation Network (MRN), which combines bidirectional cross-attention (Mutually Enhanced Fusion) with a Mixture-of-Experts classifier (Adaptive Fusion). Experiments across four datasets covering image-text, audio-visual, and RGB-depth modalities show consistent, if sometimes modest, improvements over both unimodal OSR and multimodal fusion baselines.

---

## Strengths

- **Clean identification of a concrete failure mode.** Table 1 on Food-101 delivers a targeted diagnostic: Text-OSR achieves AUROC ≈ 90–92, Fusion improves closed-set ACC at a small AUROC cost, but Fusion-OSR drops both AUROC and ACC relative to Fusion alone (e.g., −2.25 AUROC / −5.23 ACC at the 10-class split). This controlled ablation—isolating modality, fusion, and OSR independently before combining them—is more careful than a typical systems paper and provides a useful warning to the community against naïve combination.

- **Dual-role design: standalone method and plug-in fusion module.** MRN is evaluated not only as a standalone MMOSR model but as a drop-in fusion backbone inside ARPL and CSRR (ARPL-MRN, CSRR-MRN). In both roles it consistently outperforms ADD/CAT/GQA fusion counterparts across all four datasets and both OSR loss families, demonstrating that the architecture is a genuinely reusable component rather than a one-shot solution.

- **Consistent positive gains as a plug-in.** In the plug-in setting (Table 2), MRN yields positive gains over the next-best fusion strategy in every single cell across both OSR methods and all four datasets. This consistency is meaningful, even if the margins are sometimes small.

- **Breadth of benchmark.** Four datasets spanning three heterogeneous modality pairs (image-text, audio-visual, RGB-depth) and varying numbers of known/unknown classes (5–80 in Table 3) provide meaningful diversity. All baselines are reproduced by the authors using the same encoder, ensuring fairer comparison than re-using published numbers.

---

## Weaknesses

### Fatal
None.

### Major

- **MoE ablation is entirely missing.** Table 4 ablates only the two cross-attention branches (C₁, C₂) while the Adaptive Fusion / MoE module is always present in every row. There is no experiment comparing MoE against a single MLP of matched capacity. Because MRN attributes part of its gain to "adaptive fusion capturing multiple informative representations," the absence of this ablation leaves one of the two central architectural claims without direct support. The observed gains could plausibly arise entirely from cross-attention, with MoE contributing nothing beyond a larger parameter count.

- **Fusion degradation diagnosis is limited to a single dataset.** The entire Section 3.2 motivation—including Table 1 and Figure 2—is built on Food-101 alone with a single simple addition-based Fusion-OSR baseline. There is no analogous diagnostic table or visualization for Flower-102, CREMA-D, or SUN RGB-D. Given that CREMA-D and SUN RGB-D have very different modality characteristics, the generality of "fusion degradation" as a universal MMOSR pathology remains unverified. This weakens the necessity argument for a new task and the targeted motivation for MRN's design.

- **No variance estimates; reported gains are often within plausible noise.** All results are single-point numbers with no standard deviations, confidence intervals, or multiple-seed averaging. In Table 2, several improvements are extremely small: SUN RGB-D +0.37 AUROC / +0.01 OSCR over the next-best fusion method, and CREMA-D is actually negative (−1.05 AUROC / −0.18 OSCR vs. MLA). Sub-1% improvements are uninformative without variance. For a paper that defines a new benchmark task, this is a significant reproducibility and credibility concern.

### Minor

- **MRN underperforms MLA on CREMA-D, yet the paper claims consistent superiority.** Table 2 shows MLA beats MRN on CREMA-D (67.83 vs. 66.78 AUROC; 57.50 vs. 57.32 OSCR). The paper's narrative that MRN "consistently demonstrates exceptional MMOSR performance" does not acknowledge this exception. The paper should be candid about where improvements hold and where they do not, and analyze why CREMA-D behaves differently.

- **Unknown rejection mechanism is entirely standard.** Section 4.3 uses maximum Softmax probability thresholded to retain 95% of known samples—a baseline known to be suboptimal in OSR/OOD literature (energy scoring, Mahalanobis distance, OpenMax, etc. all outperform it in many settings). No comparison is made against alternative scoring functions applied to the same MRN features. It is therefore unclear how much of any gain comes from the learned representation versus the scoring function, and whether a stronger scorer atop a weaker backbone could match MRN.

- **Fusion degradation is measured only qualitatively.** The claim that OSR regularization "over-compresses" fused representations is supported exclusively by t-SNE visualizations and downstream metric drops. t-SNE is a nonlinear embedding that does not reliably preserve global structure; it is not a measurement of representation compactness, rank, or discriminability. Quantitative diagnostics (e.g., feature covariance rank, effective dimensionality, CKA similarity between fused and unimodal representations with/without OSR loss) are needed to substantiate the stated mechanism.

- **OSCR metric is never formally defined.** Despite being used in all main tables and highlighted in the headline claim ("up to 5.23% on OSCR"), the OSCR formula does not appear anywhere in the paper's main text. Readers unfamiliar with the metric cannot verify whether the evaluation protocol matches the original definition.

- **Benchmark construction protocol underspecified.** The text does not state how many random seeds are averaged per result, how known/unknown class splits are drawn (fixed seed vs. average over random splits), or how the threshold τ is calibrated in practice (validation split? training split?). These details are necessary to reproduce any result and to assess whether the gains are stable across splits.

### Tiny

- **Equation (1) has ambiguous notation.** The expression `Softmax(W₁^Q z₁ z₂ W₁^K / √d)(W₁^V z₂)` is dimensionally unclear without explicit transposes and shape annotations. Standard cross-attention would require `z₂ᵀ` to form the inner product. The missing transpose makes it impossible to verify the operation from the equation alone.

- **Metric inconsistency across sections.** Table 1 / Table 4 use AUROC + ACC; Table 2 uses AUROC + OSCR; Table 3 uses both. If OSCR is the preferred MMOSR metric (it captures both open-set and closed-set performance jointly), it is odd that it is absent from the key necessity analysis (Table 1) and ablation (Table 4).

---

## Nice-to-Haves

- **Computational cost analysis.** Adding a table comparing parameter count, FLOPs, and inference latency between MRN and simpler fusion baselines (ADD, CAT, MLA) would help practitioners evaluate the performance-cost trade-off, especially for the robotic deployment scenarios mentioned in the introduction.

- **Fusion degradation visualization on other datasets.** A Figure-2-style t-SNE (or, better, quantitative analysis) for CREMA-D and SUN RGB-D would substantiate whether the phenomenon is general. Even a brief note explaining why CREMA-D behaves differently from Food-101 would be valuable.

- **Missing-modality robustness.** The paper's practical motivation (robots, unmanned systems) strongly implies sensors can fail. Testing MRN under one-modality-missing at inference time—and comparing it to unimodal fallback—would directly address the practical scenario and could reveal a natural advantage of the bidirectional cross-attention design (graceful degradation when one branch has no query signal).

- **Stronger fusion+OSR baselines.** The "Multimodal fusion with OSR" rows in Table 2 use ADD/CAT/GQA as the fusion component. Using TMC or MLA (the strongest fusion baselines) as the backbone for ARPL/CSRR would be a fairer stress test of MRN as a plug-in and would tighten the experimental claims.

- **Alternative OSR scoring functions.** Comparing energy scoring, Mahalanobis distance, or OpenMax on top of MRN features would disentangle representation quality from threshold selection and likely improve reported AUROC further.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Figure 2 (d) description contradicts the paper's narrative" (Harsh Critic).** The critic claims Figure 2d (Fusion-OSR) is described as having "dispersed unknowns," which would contradict the paper's narrative of over-compression. The alt-text used by the critic is an AI-generated image description and is likely unreliable. The quantitative evidence in Table 1 clearly and consistently shows Fusion-OSR degrades both AUROC and ACC relative to plain Fusion, supporting the paper's narrative. This criticism should not be trusted.

- **"GQA is not a canonical multimodal fusion baseline" (Harsh Critic).** GQA (Grouped Query Attention, EMNLP'23) is used as a fusion mechanism, not in its NLP efficiency role. Its use is unusual but not unreasonable as an attention-based fusion baseline. Per the rules, citing a published method is sufficient to assume it exists and can be used as a baseline.

- **"CLIP comparison is unfair" (Harsh Critic).** CLIP operates in a zero-shot or 16-shot fine-tune regime while MRN trains from scratch on known classes. The comparison is acknowledged to be across different data regimes and is presented as an additional reference point, not as the primary apples-to-apples comparison. The paper does not misrepresent the comparison. This is not a fairness problem—if anything, CLIP has a massive pretraining advantage, so MRN beating it strengthens rather than weakens the paper's claim.

- **"Lack of theoretical grounding for fusion degradation" (multiple reviewers).** This paper is an empirical systems contribution. Demanding theoretical proofs of a diagnosed failure mode is not a standard expectation for this type of work. Weakened to the quantitative diagnostic request already listed under Weaknesses/Minor.

- **"Missing related works on multimodal OOD / uncertainty estimation" (Harsh Critic).** Per instructions, potential missing related works are not included as we cannot verify their existence.

- **"The problem definition is too conventional to be a contribution" (Harsh Critic).** The task formulation is intentionally minimal; the contribution lies in the empirical diagnosis and the proposed method. Evaluating a paper negatively for scoping its formal definition narrowly is unfair.

---

## Novel Insights

The most genuinely useful insight this paper surfaces—and one that practitioners should take seriously—is that OSR regularization applied naïvely to multimodal fusion is not a "free lunch." Because OSR losses (e.g., ARPL's reciprocal-point loss) penalize feature spread to compress known-class representations, they work against the goal of multimodal fusion, which needs diverse, modality-specific information to be preserved. The result is that adding an OSR regularizer to a strong multimodal backbone can *hurt* both closed-set accuracy and open-set detection simultaneously, even though each component works well in isolation. This failure mode is subtle because it is non-obvious that the same inductive bias that helps in unimodal OSR (compactness of known classes) is harmful when applied to fused representations that need to retain cross-modal diversity. The design response—using cross-attention to force modality representations to actively inform each other before compression, and MoE to preserve representation diversity after—is a principled architectural answer to this diagnosed pathology, even if it draws on standard components.

---

## Suggestions

1. **Add MoE ablation.** Add a row to Table 4: "C₁ ✓, C₂ ✓, single MLP (same capacity)" vs. "C₁ ✓, C₂ ✓, MoE (E=15, K=4)" on both Food-101 and Flower-102. This is the single most important missing experiment and can be done with existing infrastructure.

2. **Report mean ± std over ≥ 3 random class splits** for all main results in Table 2 and Table 3. Given that some improvements are below 0.5%, single-run results are insufficient to substantiate the claims.

3. **Extend the fusion degradation diagnostic (Table 1 equivalent) to at least one additional dataset** (CREMA-D or SUN RGB-D), using the same Image-OSR / Text-OSR / Fusion / Fusion-OSR structure. This directly validates the generality of the phenomenon that motivates the paper.

4. **Formally define OSCR in the paper** (e.g., in Section 5.1 or an appendix), and ensure Table 4's ablation uses it so that all results are interpretable on the same evaluation axis.

5. **Add a quantitative representation-compactness measurement** (e.g., effective rank of the covariance matrix of class embeddings, or average pairwise cosine distance among class centroids) measured before and after applying OSR regularization to fusion representations. Even one such figure would transform the "fusion degradation" claim from qualitative to quantitative.

6. **Explicitly address the CREMA-D exception** in the results discussion: explain whether the underperformance relative to MLA is expected (e.g., audio-visual fusion behaves differently), or identify what aspect of CREMA-D makes it structurally different from the other datasets.

---

## Evaluation

**Originality:** Moderate. The task definition is a natural extension of OSR to multimodal settings—not a conceptual surprise—but the concrete diagnosis of fusion degradation and the targeted architectural response are genuine contributions beyond prior work. The individual components (cross-attention, MoE) are standard; their combination in this specific context is new and well-motivated.

**Importance of research question:** High. Open-set recognition in multimodal systems is a practically significant and underexplored problem. The finding that naïve fusion+OSR combinations can *degrade* performance is a practically important warning for system builders.

**Claims well-supported:** Partially. The existence of fusion degradation is convincingly shown on Food-101 but not generalized. MRN's effectiveness is supported in 3/4 standalone comparisons and all plug-in comparisons, but the absence of variance estimates and the incomplete ablation weaken confidence in the magnitude of the gains.

**Soundness of experiments:** Moderate. The experimental design (same encoders across methods, training from scratch on known classes only, diverse modality pairs) is sound in principle, but missing variance, an incomplete ablation, and a single-dataset motivation analysis are meaningful gaps.

**Clarity of writing:** Generally clear with a logical flow from problem to diagnosis to method. The equation notation in (1) is ambiguous and the OSCR metric is never defined, but these are addressable.

**Value to the research community:** Moderate to high. The paper creates a useful benchmark and identifies a reproducible failure mode that is likely to be of interest to anyone combining OSR with multimodal architectures.

**Contextualization relative to prior work:** Adequate for OSR and multimodal fusion. The connection to the broader multimodal uncertainty and OOD detection literatures could be developed further to sharpen the novelty claim, though this is not a blocking issue.

---

## EW6bNEqalF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (5.6/10)
- Match: YES

### Final Review

## Summary

This paper proposes two new techniques for offline RL in Regular Decision Processes (RDPs): (1) a novel language-based metric $L_X$ grounded in the dot-depth hierarchy of star-free regular languages, which replaces the $L_\infty^p$ test in the ADACT-H state-merging algorithm, and (2) a Count-Min-Sketch (CMS) approach to compactly store empirical suffix distributions. The central theoretical result (Theorem 1) proves an exponential gap between $L_\infty^\ell$-distinguishability and $L_{\mathcal{X}_{2,1}}$-distinguishability for the T-maze family of RDPs, and Theorem 3 derives a PAC sample complexity bound that depends on $\log |\mathcal{X}|$ rather than horizon-dependent suffix-space size. The authors also discover and correct an error in a prior proof, adding a $\sqrt{H}/\mu_0$ factor to the RegORL sample complexity.

---

## Strengths

- **Provable exponential gap via concrete family (Theorem 1).** The paper rigorously constructs a family of RDPs $(\mathbf{R}_N)$ and behavior policies $(\pi_N^b)$ for which $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$. The T-maze running example grounds this in an intuitive and widely-used benchmark domain; the argument is not merely asymptotic but is tied to a specific structural property (aggregating over the language of "North-action-then-positive-reward" traces).

- **Two-dimensional hierarchy that unifies common metrics.** The $\mathcal{X}_{i,j}$ hierarchy is a principled and elegant construction: $j=1$ on singleton patterns recovers $L_\infty^p$; taking $\mathcal{X} = 2^{\Gamma^\ell}$ recovers total variation ($L_1$). The interpolation via the operator $C_k^\ell$ and the basic pattern sets $\mathcal{G}_1, \mathcal{G}_2, \mathcal{G}_3$ provides a concrete and actionable family of tests, each with $|\mathcal{X}_{i,j}| \in \mathcal{O}((AOR)^j)$, growing polynomially in problem parameters for fixed $j$.

- **Identification and correction of a prior proof error.** The analysis in Theorem 2 uncovers a missing $\sqrt{H}/\mu_0$ multiplicative factor in the RegORL proof of Cipollone et al. (2023). Identifying and correcting such errors is a meaningful contribution to the reliability of the theoretical foundation in this area.

- **T-maze scaling experiment supports the core claim.** Figure 2 provides direct empirical evidence that the language metric approach scales polynomially in corridor length $N$ (both in runtime and automaton size), while CMS scales exponentially. The gap is observed consistently across 20 runs, with the language approach handling $N=100$ while CMS times out beyond $H=15$—precisely the behavior predicted by the theoretical analysis.

---

## Weaknesses

### Fatal
None.

### Major

- **No direct empirical comparison with the baseline being improved (ADACT-H / RegORL with $L_\infty^p$).** The paper's central empirical claim is that the language metric improves on existing RDP offline RL algorithms. However, the only comparator in experiments is FlexFringe, a grammar/PDFA learner with no PAC guarantees and different objectives (it sometimes learns cyclic automata). The original ADACT-H with $L_\infty^p$ is never included as a baseline in Table 1 or Figure 2. Without this comparison, there is no direct empirical validation that the proposed method yields better automata or policies than the prior approach on matched datasets—this is especially important given that the theoretical improvement is conditional (on $\mu_0$ and $d_m^*$).

- **No sample complexity experiments: reward vs. dataset size $K$ is never measured.** The core theoretical claim is improved PAC sample complexity (Theorem 3). Yet the experiments report only fixed-dataset reward, runtime, and automaton size. The natural validation—sweeping $K$ (number of offline episodes) and measuring policy quality as a function of $K$ for each method—is entirely absent. Without this, the empirical support for the paper's primary contribution is indirect at best.

- **No guidance for selecting $(i, j)$ in $\mathcal{X}_{i,j}$.** Assumption 1 presupposes that the chosen $\mathcal{X}_{i,j}$ satisfies $L_{\mathcal{X}_{i,j}}$-distinguishability $\geq \mu_0 > 0$. In practice, the user has no means to verify this from data, and the paper provides no adaptive selection, cross-validation criterion, or statistical test for sufficiency of $(i,j)$. If $(i,j)$ is chosen too small, $\mu_0 = 0$ and the algorithm silently fails. All experiments use $\mathcal{X}_{3,1}$ without justification. This gap significantly limits practical applicability.

### Minor

- **The $d_m^*$ term can independently be exponentially small in $H$, potentially negating the $\mu_0$ gain.** Theorems 2 and 3 depend on $d_m^* = \min_{u,a,o} d_t^*(u,a,o)$, the minimum occupancy of the optimal policy. The paper briefly notes that "$1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach," but provides no analysis of when this is well-behaved or how it interacts with the language family choice. In the worst case, the exponential improvement in $1/\mu_0$ is entirely offset by an exponential $1/d_m^*$, yet the paper presents the bounds without clearly quantifying this interaction.

- **$L_X$ is only a pseudometric, and the consequences for correctness are unaddressed.** Footnote 1 acknowledges that $L_X$ is only a pseudometric. This means distinct RDP states can have $L_X$ distance zero for a given $\mathcal{X}_{i,j}$, in which case the algorithm would incorrectly merge them. The paper does not analyze whether or under what conditions this is avoided, nor how it affects the model recovery guarantee beyond assuming $\mu_0 > 0$.

- **The CMS memory advantage is not measured empirically.** CMS is introduced with the stated benefit of reducing memory requirements (Theorem 2). However, Table 1 reports only runtime, automaton size, and reward—not memory. The theoretical memory benefit is never validated or quantified experimentally, leaving the CMS contribution empirically unsupported on its primary claim.

- **CMS width parameter $w = \lceil \varepsilon / \delta_c \rceil$ appears non-standard.** The standard CMS (Cormode & Muthukrishnan, 2005) sets width $w = \lceil e/\varepsilon \rceil$, which is inversely proportional to desired accuracy: smaller $\varepsilon$ requires larger $w$. The paper's formula $w = \lceil \varepsilon / \delta_c \rceil$ gives a smaller $w$ for smaller $\varepsilon$, which is the opposite of standard behavior. This may reflect a non-standard parameterization or a typesetting issue, but as presented it is inconsistent with the cited data structure and should be clarified. The resulting approximation property used in the proof of Theorem 2 is also not explicitly stated.

- **The correction to the prior proof ($\sqrt{H}/\mu_0$ factor) weakens the narrative without being properly contextualized.** The paper says both RegORL and its new bounds have an additional $\sqrt{H}/\mu_0$ factor due to a corrected prior proof. However, it does not clearly show whether the new bounds in Theorem 3 are still strictly better than the corrected RegORL bounds across the relevant parameter regimes. A direct corollary comparing the two in matched notation would resolve this.

### Tiny

- **The estimator definition for $L_X$ is ambiguous as written.** The paper states $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j}) / |\mathcal{Z}_1|$, but $\mathcal{X}_{i,j}$ is a *set of languages*, not a single language. The intended empirical estimator for $L_X$ should be $\hat{L}_X(\mathcal{Z}_1, \mathcal{Z}_2) = \max_{X \in \mathcal{X}} |\hat{p}_1(X) - \hat{p}_2(X)|$ where $\hat{p}_i(X) = \sum_{e \in \mathcal{Z}_i} \mathbb{I}(e \in X) / |\mathcal{Z}_i|$. The distinction matters for understanding the computational cost.

- **Notation instability for the reference metric ($L_\infty^\circ$, $L_\infty^p$, $L_\infty^\ell$).** These appear to refer to related but distinct metrics, yet the notation shifts across sections without a clear mapping. This adds friction when following the comparison between old and new bounds.

- **Monotonicity of $L_{\mathcal{X}_{i,j}}$ is stated but not proved or cited in the main text.** The claim $L_{\mathcal{X}_{i,j}} \leq \min(L_{\mathcal{X}_{i+1,j}}, L_{\mathcal{X}_{i,j+1}})$ presumably follows from $\mathcal{X}_{i,j} \subseteq \mathcal{X}_{i+1,j}$ and $\mathcal{X}_{i,j} \subseteq \mathcal{X}_{i,j+1}$, but this should be made explicit given its central role.

---

## Nice-to-Haves

- An ablation over $(i,j)$ (e.g., comparing $\mathcal{X}_{1,1}$, $\mathcal{X}_{2,1}$, $\mathcal{X}_{3,1}$ on T-maze) would directly illustrate the practical effect of language family choice and provide intuition for practitioners—particularly showing what happens when $\mathcal{X}_{i,j}$ is too small to distinguish relevant states.

- Visualizing the learned automata against ground-truth RDP structure (e.g., the two parallel $\top/\perp$ components in T-maze) for each method would give qualitative evidence that the language metric correctly recovers RDP structure.

- A discussion of when $d_m^*$ is expected to be well-behaved (e.g., structured RDPs with good coverage) and how it interacts with the $\mu_0$ improvement would make the bounds more interpretable.

- Extending experiments to domains with $H \geq 50$–$100$ where the exponential improvement in $\mu_0$ should manifest not just in runtime but also in policy quality with realistic dataset sizes.

- An end-to-end theorem connecting dataset size $K$ to $\varepsilon$-optimal policy return in the main paper (currently deferred to the appendix via RegORL), making the offline RL contribution self-contained.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's concern about practical relevance of the algorithm contributions vs. model reconstruction.** The paper clearly states its contribution is a tractable statistical test for ADACT-H, which is then incorporated into RegORL for policy learning. This framing is consistently maintained and is a legitimate, well-scoped contribution.

- **Criticism that algorithm pseudocode is in the appendix.** This is standard practice and not a weakness for an ICLR submission.

- **Criticism that broader impact is not discussed.** Not a standard ICLR requirement.

- **Criticism that sequence-model or recurrent-policy offline RL baselines are absent.** Such methods lack PAC guarantees and operate under different objectives than RDP model recovery. Their absence is within the paper's scope; including them could be a nice-to-have but is not a core flaw.

- **Criticism of "unfair comparison" with FlexFringe (FlexFringe uses heuristics that can hurt its performance).** The paper explicitly acknowledges that FlexFringe uses heuristics that do not preserve guarantees (Section 5). The asymmetry is disclosed and works *against* the authors' method (FlexFringe is freed from PAC constraints), making any win over it a stronger result.

- **Generic strength: "well-written" and "topic is important."** Removed per synthesis rules.

- **Critic concern about lack of broader impact section.** Removed.

---

## Novel Insights

The most genuinely novel insight in this paper—beyond presenting the $L_X$ framework—is the structural observation that the exponential blow-up of $L_\infty^p$ in RDPs like T-maze is not an intrinsic property of the problem's difficulty but an artifact of the metric: the *same* RDP states that are exponentially hard to separate by singleton-string probabilities can be $\Omega(1)$-separated by a single aggregated language event (the probability of "action North followed eventually by positive reward"). This shifts the conceptual frame from "the problem is hard" to "the test is the wrong granularity." The two-dimensional $\mathcal{X}_{i,j}$ hierarchy then provides a concrete realization of this insight, showing that the right level of aggregation can be extracted from the algebraic structure of episode traces (temporal ordering of basic action/observation/reward patterns) rather than from sufficiency-style or bisimulation-style notions. The connection to dot-depth hierarchy from formal language theory is a nontrivial cross-domain link that could inspire further work on complexity-theoretic characterizations of learnability in non-Markovian settings.

---

## Suggestions

1. **Add ADACT-H with $L_\infty^p$ as a direct experimental baseline.** On each domain, include the original ADACT-H variant as a third column in Table 1 using the same dataset. This is the method being improved and its absence is the most significant empirical gap.

2. **Add reward-vs-dataset-size curves.** For at least T-maze and one other domain, plot policy quality as a function of $K$ for the language metric and ADACT-H with $L_\infty^p$. This directly validates the sample complexity claims.

3. **Clarify or correct the CMS width formula $w = \lceil \varepsilon/\delta_c \rceil$.** If this is an intentional non-standard parameterization, state explicitly what approximation guarantee it provides and how it enters the proof of Theorem 2.

4. **Provide at least a heuristic for selecting $(i,j)$.** Even a monotone-search procedure ("start at $(1,1)$, increase until the statistical test is ever triggered") would substantially improve the paper's practical usability and address the most common practitioner objection.

5. **Add memory measurements to experiments.** Report peak memory usage for ADACT-H (language metric), ADACT-H (CMS), and FlexFringe. Without this, Theorem 2's stated advantage cannot be empirically verified.

6. **Fix the estimator notation in Section 4.1.** Define $\hat{p}(X)$ for each $X \in \mathcal{X}_{i,j}$ individually and write the empirical $\hat{L}_X$ as the max over $X$.

7. **Include a brief end-to-end corollary** combining Theorem 3 with the RegORL bound to state the dataset size needed for an $\varepsilon$-optimal policy, making the offline RL guarantee visible in the main text.

---

## Evaluation

**Originality:** High. The language metric framework and its connection to the dot-depth hierarchy are conceptually novel contributions not previously applied in RL or automata-based policy learning. The two-dimensional $\mathcal{X}_{i,j}$ hierarchy is a principled and non-trivial construction.

**Importance of research question:** Moderate. The exponential sample complexity barrier in offline RDP learning is a genuine and previously identified open problem. The broader RDP/non-Markovian RL setting is scientifically important, though it remains a specialized subfield.

**Claims well supported:** Partially. The theoretical claims (Theorems 1–3) are well-supported, and the T-maze scaling experiment is convincing for the tractability claim. However, the primary sample efficiency claim is not validated empirically (no reward-vs-K curves, no comparison with the exact baseline being improved), which is a significant gap between the paper's framing and its experimental evidence.

**Soundness of experiments:** Weak. Five domains are evaluated, but the baseline set is insufficient (FlexFringe is not an offline RL algorithm), no sample complexity experiments are conducted, CMS memory is not measured, and only a single $(i,j)$ is tested. The experiments support the tractability story but not the sample efficiency story.

**Clarity of writing:** Moderate. The high-level narrative is clear and the T-maze example threads the paper effectively. However, there are notation inconsistencies across sections, an ambiguous estimator definition, and a potentially erroneous CMS formula that create friction in the technical sections.

**Value to the research community:** Moderate-to-high contingent on revision. The language metric framework is a valuable conceptual contribution with potential applicability beyond offline RDPs (e.g., online learning, PSR identification). In its current form the empirical gaps reduce the paper's immediate utility.

**Contextualization relative to prior work:** Good. The paper positions itself clearly against RegORL and ADACT-H, and the related work section is appropriately broad. The identified error correction in prior work strengthens the theoretical foundation of the area.

---

## F6s7OApF0n

- GT: Reject (avg 4.8)
- Predicted: Accept (6.5/10)
- Match: NO

### Final Review

## Summary
CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) reformulates HPO as maximizing a user-specified utility function that trades off BO step cost against best-so-far performance, rather than optimizing asymptotic validation accuracy. The paper introduces a utility-based acquisition function with dynamic lookahead, an adaptive stopping criterion that blends regret-based and probability-of-improvement signals, and a novel two-stage LC mixup strategy for training PFN surrogates on existing learning curve datasets. Across four benchmarks (LCBench, TaskSet, PD1, RoboFlow) and eight baselines, CMBO consistently achieves the best normalized regret under various cost-sensitive settings.

---

## Strengths

- **Genuinely novel utility-based framing for freeze-thaw BO.** Prior freeze-thaw methods (DyHPO, iFBO, DPL) target asymptotic performance or a fixed final epoch; CMBO is the first to treat the joint (cost, performance) trade-off as the primary objective, and this reformulation drives both acquisition design and endogenous stopping in a unified way.

- **Two-stage LC mixup preserves cross-configuration structure.** The key insight—applying the same λ₁ across all configurations in the first mixup step to preserve correlation structure before individual interpolation in the second step—is a principled and non-obvious design choice for augmenting learning curve datasets for PFN training. Fig. 6 shows it measurably reduces test loss and downstream regret on PD1.

- **Acquisition function analysis confirms intended behavior.** Figs. 7a–c show that the selected configurations initially have large optimal Δt (exploratory, non-greedy) but progressively shift toward Δt≈0 (exploitative) as cost dominates, and the method concentrates on fewer configurations under higher cost penalties. This directly confirms that Eq. (2) functions as designed.

- **Comprehensive ablation isolates contributions.** Table 3 cleanly attributes gains to three distinct components (stopping criterion, acquisition, transfer learning), with monotone improvement as each is added—particularly dramatic under strong cost penalties (α = 2e−4).

- **Strong performance across diverse benchmarks and utility forms.** Rank 1.0 across all conditions in Table 2 (various utility functions on PD1) and consistent top rank in Tables 1 and 4 demonstrate that the method's advantage is not confined to a narrow setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Cost is modeled as BO step count, not actual compute.** The utility U(b, ỹ_b) penalizes the integer step index b, but the paper explicitly motivates cloud credits, wall-clock time, and Slurm quotas—all of which are non-uniform across configurations and architectures. In the RoboFlow experiment, three architectures (ResNet-50, HRNet, MobileNetv2) have very different per-epoch compute. The current formulation equates all BO steps as equal-cost, which directly undermines the "cost-sensitive" framing. This is the single largest gap between the paper's motivation and its technical contribution; the utility function and stopping criterion need the cost axis to be meaningfully calibrated for real-world applicability.

- **Table 3 ablation has unexplained duplicate rows.** The last two rows are both labeled p_b ✓, Acq. ✓, T. ✓ but report substantially different results (e.g., 4.4 vs. 0.9 for α=2e−4). Since the paper describes only three binary ablation factors, there is an implicit fourth varying component that is never labeled. As presented, this renders the bottom two rows uninterpretable and weakens the ablation evidence.

- **Transfer learning contributes disproportionately to gains, but cross-benchmark ablation is missing.** The mixup ablation (Fig. 6) is presented only on PD1. Given that transfer learning is one of three claimed core contributions, an ablation on at least one additional benchmark is necessary to confirm generalization of this finding.

### Minor

- **Utility elicitation from user preferences is not validated end-to-end.** All main experiments use analytically specified utilities (linear, quadratic, staircase). The "Estimated" condition in Table 2 is constructed synthetically by assuming the user wants a better trade-off than iFBO—this is a method-relative construction, not a real user preference. The Bradley-Terry preference learning is demonstrated only in isolation (Fig. 2, 1,000 synthetic queries). Since this component is presented as a substantive contribution, its absence from the closed-loop evaluation (estimate utility → run BO → measure outcome) is a notable gap. Appendix §B discusses fewer datapoints but still uses synthetic queries.

- **Algorithm 1 notation bug.** Line 4 reads n* ← argmax_{n ∈ C} A(n), but C is a set of (x, t, y) triples, not a set of configuration indices. This should be n ∈ [N]. The intent is clear from context, but the formal definition is incorrect and should be corrected.

- **β sweet spot varies by dataset but a single value is used globally.** Fig. 7d shows that the optimal β differs for LCBench, TaskSet, and PD1, yet β=e⁻¹ is applied uniformly across all experiments. While the average performance at β=e⁻¹ appears good, there is no discussion of how sensitive results are to this choice and how a practitioner would select β without access to the benchmark. The paper should include a brief sensitivity analysis or criterion for choosing β in new settings.

- **Zero standard deviations for FSBO and Quick-Tune† are not explained.** Several entries in Tables 1–4 for these methods show ±0.0. While FSBO and Quick-Tune† may be deterministic, ±0.0 at the reported precision should be explicitly justified (e.g., confirmed deterministic, or variance rounds below 0.05).

### Tiny

- **Notation inconsistency between ỹ_b (§3.1) and ȳ_b (Algorithm 1).** Both appear to denote best-so-far BO performance but use different symbols. Line 10 of Alg. 1 defines ȳ_b while §3.1 introduces ỹ_b for the same quantity. Similarly, Eq. (5) uses ȳ_{b+Δt} in the indicator while prior sections use ỹ.

- **The U_prev update at line 11 of Alg. 1 clarifies a key design choice, but interaction with the acquisition is not analyzed.** The paper explains that U_prev is not the global best utility but the most recent one (justified by cost irreversibility). However, if utility dips repeatedly, the acquisition threshold decreases, which could allow continued exploration of configurations with modest expected improvement. The paper's discussion is qualitative; a brief quantitative analysis would strengthen the argument.

---

## Nice-to-Haves

- Extend U(b, ỹ_b) to accept a per-step cost function c(x_n, t_n) so that BO steps differing in wall-clock cost are weighted appropriately; this would make the "cloud credits" and "Slurm wall-time" motivations technically consistent with the formulation.
- Provide wall-clock overhead comparison of the PFN surrogate vs. lighter alternatives (GP, deep kernel GP), including PFN pretraining cost, so readers can assess total efficiency.
- End-to-end validation with real user preference queries (even a small user study) to test whether ~100–300 pairwise comparisons suffice for downstream BO quality.
- An oracle stopping comparison scatter plot—actual b* chosen by CMBO vs. the oracle optimal stopping step—would be the most direct visualization validating the stopping criterion.
- Ablation of the BetaCDF stopping form in Eq. (4): compare against simpler monotone squashing functions (logistic, piecewise linear) to assess whether the specific family matters or whether any monotone transform of p_b works similarly.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Eq. (3) stopping rule is directionally odd."** The critic claims the rule stops when normalized regret is large. This is correct and intentional: the LHS (Û_max − U_prev)/(Û_max − Û_min) is large precisely when U_prev has declined far below the best seen utility, signaling the BO has passed its optimum. This is the semantically correct stopping signal. The critic misread the criterion's intent.

- **"Using U_prev rather than best utility so far is unusual and under-analyzed."** The paper explicitly justifies this on page 5: "the cost of BO that has previously been incurred is not reversible." This is a deliberate and sound modeling choice. The stopping criterion compensates for cases where utility has fallen and continues falling.

- **"Stopping rule for baselines is not optimized fairly."** Footnote 2 explicitly explains that the PI-based component of the stopping criterion (Eq. 5) depends on the utility-aware acquisition function, which baselines do not have. The paper correctly gives baselines the best-available regret-only stopping threshold (δ_b = 0.2), and notes this was found to perform well for them. This is a reasonable constraint, not a fairness failure.

- **"The evaluation metric favors the proposed formulation by construction."** Optimizing and being evaluated on the same criterion is expected for any method with a well-defined objective. The paper also reports complementary metrics (average rank, Fig. 5 trajectories, Fig. 7 analyses), providing multiple lines of evidence.

- **"Potential data leakage through task-level interpolation."** The paper uses distinct train/test task splits; linear interpolation between training tasks is a standard augmentation. Without evidence of leakage, this is speculative.

- **"FSBO outperforms multi-fidelity methods is under-analyzed."** The paper does provide a clear explanation: transfer learning substantially improves sample efficiency, making FSBO (which uses the same LC datasets) competitive despite its black-box nature. Quick-Tune† underperforms FSBO due to a greedy acquisition and no data augmentation.

- **"The conclusion is overly confident."** The empirical evidence is comprehensive (4 benchmarks, 8 baselines, ablations, real-world data). The conclusion language is somewhat strong but proportionate to the experimental scope.

- Missing related work critiques (removed per review instructions).

---

## Novel Insights

The paper surfaces an underappreciated failure mode of multi-fidelity BO: even methods that efficiently allocate epochs *within* the optimization loop systematically over-explore because they ignore the cost of the optimization process itself. The insight that cost-sensitivity should act on the meta-level (when to stop the entire BO) rather than only at the object-level (which configuration to evaluate next) is productive and broadly applicable. The finding in Fig. 4 that FSBO—a black-box method that cannot switch configurations mid-run—outperforms all multi-fidelity baselines on most benchmarks is striking: it implies that the sample efficiency gains from dynamic configuration switching are currently smaller than the gains from strong transfer surrogates. This suggests that the field may be somewhat overfocused on sophisticated acquisition designs relative to the surrogate quality, a message the CMBO results reinforce since the transfer component drives the largest share of improvement in Table 3.

---

## Suggestions

- **Fix the Table 3 presentation:** label or describe all four ablation conditions, including the implicit component distinguishing the last two rows. Consider introducing a "T. (no mixup)" vs. "T. (with mixup)" distinction explicitly.
- **State the equal-cost-per-step assumption explicitly** in §3.1 and acknowledge it as a limitation in §5, given that the motivation heavily invokes non-uniform compute scenarios.
- **Expand the mixup ablation to at least one additional benchmark** (LCBench or TaskSet) to confirm that the results in Fig. 6 generalize.
- **Clarify zero-variance entries:** add a footnote or table note explaining which methods are run deterministically and why their variance is below rounding threshold.
- **Address the Algorithm 1 notation bug** (line 4: n ∈ C → n ∈ [N]).
- **Provide a brief discussion of β selection** for practitioners: does the chosen β=e⁻¹ correspond to a principled heuristic, or should users tune it, and if so, how?

---

## Evaluation on Key Axes

**Originality:** High. The utility-based objective for freeze-thaw BO and the two-stage LC mixup are both novel contributions not present in prior work. The adaptive stopping criterion combining normalized regret with probability of improvement is also a new design.

**Importance of research question:** High. Cost-sensitivity in HPO is practically critical and structurally underrepresented in the multi-fidelity BO literature. The shift from "maximize terminal accuracy" to "maximize utility over (cost, performance)" is a meaningful and well-motivated reformulation.

**Claims well-supported:** Mostly. Empirical results are comprehensive across four benchmarks, and the ablation isolates contributions at the component level. However, the utility learning-from-preferences claim—arguably the most novel aspect of the framing—is not validated in the closed-loop BO setting with real user data.

**Soundness of experiments:** Good overall, with one significant gap: the ablation table has an unresolved ambiguity (duplicate rows), and the mixup ablation is restricted to one benchmark. The evaluation metric and baseline stopping setup are well-documented.

**Clarity of writing:** Good. The method is clearly described and the algorithm is easy to follow. Minor notation inconsistencies (ỹ vs. ȳ, C indexing in Alg. 1) should be cleaned up.

**Value to the research community:** High. The framework, transfer surrogate, and stopping criterion are modular and could benefit practitioners running HPO under real budget constraints. The real-world RoboFlow experiment adds credibility beyond tabular benchmarks.

**Contextualization relative to prior work:** Good. The paper accurately positions CMBO against the freeze-thaw BO literature and provides a principled distinction between greedy and non-greedy multi-fidelity methods, black-box transfer BO, and the new cost-sensitive setting.

---

## AnPEfzBstD

- GT: Reject (avg 3.5)
- Predicted: Reject (4.2/10)
- Match: YES

### Final Review

## Summary

This paper presents a large-scale benchmark (2100+ runs across 4 datasets and 35 setups) comparing 1D, 2D, and 3D molecular representations within a Bayesian optimization (BO) framework for materials discovery. Using MolFormer (1D/LLM), MPNN (2D), and Equiformer v2 (3D) combined with GP and Laplace-approximation surrogates, the paper finds that simpler representations generally match or outperform 3D equivariant GNNs, that 3D models require substantially more training data to be competitive, and that transfer learning is a viable strategy across modalities. The practical takeaway is that 1D and 2D representations offer a better cost-performance tradeoff for the surveyed tasks.

---

## Strengths

- **Sample-complexity analysis (Section 5.2) is a concrete and novel contribution.** Systematically varying training-set size from 500 to 50,000 and demonstrating that 3D models consistently lag 2D models at low-data regimes—while the gap narrows at ≥10,000 samples—directly operationalizes a mechanistic reason for 3D's underperformance in BO settings and advances beyond prior benchmarks that simply report end-to-end BO curves.

- **Benchmark breadth and statistical rigor are above average for this class of paper.** Four datasets spanning three orders of magnitude in size (QM7 ~7K to GEOM DRUGS ~318K), two surrogate families, and 15 random seeds with reported standard errors represent genuine experimental investment. The use of the normalized GAP metric to enable cross-dataset aggregation is sensible.

- **Inclusion of a transfer-learning condition alongside single-property prediction** adds a useful and underexplored dimension to BO benchmarking, showing that multitask pretraining can approach task-specific training quality—a practically relevant finding for practitioners who cannot afford extensive labeled data for every target property.

---

## Weaknesses

### Fatal
None.

### Major

- **Unmatched model scale between MolFormer and the GNNs fundamentally confounds the 1D-vs-3D conclusion.** The paper constrains MPNN and Equiformer v2 to ~1.5M parameters trained on QM9, while MolFormer is a pretrained masked language model trained on orders-of-magnitude more data. The headline finding "LLMs consistently outperform 2D and 3D models" cannot be attributed to representation dimensionality; it may simply reflect that a heavily pretrained large model beats smaller, less-pretrained models. Without a comparably pretrained 3D foundation model (e.g., Uni-Mol, pretrained SchNet/DimeNet) or a size-matched non-pretrained SMILES model as ablation, the 1D-vs-higher-dimensional comparison is uninterpretable for the paper's stated research question.

- **The chosen target properties are largely topology-determined, making the claim that "3D features are not useful" severely underscoped.** Atomization energy (QM7), HOMO-LUMO gap (QM9), and absolute energy (GEOM) are properties well-predicted from 2D graph topology. Tasks where 3D is genuinely differentiating—stereo-isomer discrimination, protein–ligand binding affinity, conformer-dependent solvation energy, or reaction selectivity—are absent. The paper's general-sounding conclusion ("3D is not useful") only holds for the specific properties tested, and the text does not adequately qualify this. The practical recommendation to chemists is therefore misleading for any workflow where 3D geometry is intrinsically necessary.

- **Acquisition function is never specified, yet is a critical BO hyperparameter.** Sections 2.1 and 4 introduce BO and describe the BO loop (including an "acquisition function" box in Fig. 1) but never state whether EI, UCB, Thompson sampling, or greedy selection was used. Different acquisition functions impose different exploration–exploitation regimes and may interact asymmetrically with representation quality and uncertainty calibration. This omission makes the results neither fully reproducible nor interpretable from the manuscript alone.

- **Cost-benefit claims are stated without any empirical cost data.** The paper repeatedly claims that 3D's "computational overhead outweighs predictive performance" and uses this as a primary reason for practitioners to avoid 3D. However, no wall-clock times, GPU hours, inference latency, or memory measurements are reported anywhere. Without these numbers, the cost-benefit argument is assertion, not evidence.

- **The conformer selection protocol for 3D inputs is undescribed.** GEOM provides multiple conformers per molecule; the paper never states which conformer is selected, how (lowest energy, random, RDKit-generated), or whether quality was checked. Since 3D model performance is known to be sensitive to conformer quality, using suboptimal conformers would artificially degrade 3D performance—directly undercutting the validity of the comparison. This is especially important for GEOM DRUGS, which the paper cites as a dataset emphasizing conformational flexibility.

- **Internal inconsistency between Section 5.1 and the Conclusion regarding QM9/LLM performance.** Section 5.1 explicitly states: "Contrary to all other datasets, LLMs performed worse than 2D and 3D models" for QM9. The Conclusion states: "Across all datasets examined LLMs consistently outperformed both 2D and 3D models." These are mutually contradictory. The Conclusion also contains a likely typo: the QM9 explanation reads "the task may have been the most dependent on information not captured by 2D and 3D representations," which should read "not captured by 1D representations" given the context. These inconsistencies undermine the reliability of the narrative synthesis.

### Minor

- **Section 5.2 (sample complexity) omits the LLM/1D baseline.** The 2D-vs-3D comparison is shown for varying training sizes, but since the paper's headline claim involves 1D/LLM dominance, excluding MolFormer from this analysis leaves unanswered whether the LLM's advantage is robust at low data regimes or only emerges once it has sufficient fine-tuning data. This is a material gap given the stated research question.

- **Transfer learning results are restricted to QM7 and QM9** (as confirmed by Fig. 5's caption), omitting MoleculeNet and GEOM DRUGS—the larger, more complex datasets where generalization across properties is presumably more valuable. The claim "foundation models prove a good tool to leverage in molecular optimization" is unsupported at the scale where it would most matter.

- **Training convergence of 3D models is not verified.** Equivariant GNNs like Equiformer v2 are known to be more difficult to optimize than MPNNs, and constraining both to ~1.5M parameters does not guarantee comparable training quality. Training loss curves or validation metrics confirming convergence for all modalities at each data regime are absent, leaving open whether 3D gaps reflect representation limits or optimization failures.

- **The experimental setup section is ambiguous about whether feature extractors are trained separately per dataset.** The text says "The models were trained on QM9," but BO is also run on QM7, MoleculeNet, and DRUGS. It is unclear whether 3D/2D models trained solely on QM9 are applied to other datasets (introducing domain shift) or whether separate training occurs per dataset. The current wording is inconsistent, and this matters for interpreting cross-dataset comparisons.

### Tiny

- The GAP metric is defined using both $y^*$ and $y_*$ inconsistently in the notation.
- The paper lacks a dedicated Limitations section, which would clarify the offline virtual-library setting versus true closed-loop experimental BO and the scope of the 3D conclusions.
- The sentence in Section 5.1 (QM9 paragraph) citing "information not captured by 2D and 3D representations" when the intended referent is clearly "1D representations" needs correction.

---

## Nice-to-Haves

- **Uncertainty calibration analysis (reliability diagrams) per model type.** BO acquisition quality depends jointly on predictive accuracy and calibrated uncertainty. Demonstrating whether 3D models produce well-calibrated uncertainty would clarify whether the 3D gap is fixable via better calibration methods or reflects a deeper representational mismatch.
- **Ablation separating pretraining from architecture for the LLM.** A non-pretrained transformer operating on SMILES (same architecture as MolFormer but trained only on QM9) would allow the paper to decompose "1D representation advantage" from "massive pretraining advantage."
- **Low-budget BO regime analysis (first 50–100 steps highlighted).** For expensive-oracle settings that motivate BO in the first place, the early optimization phase is most relevant; the main figures currently emphasize 1000-step convergence.
- **Correlation of 3D model relative performance with molecular characteristics** (conformational flexibility, rotatable bond count) to make the "when does 3D help" question actionable rather than purely dataset-dependent.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED] Criticism that the related work section insufficiently covers 3D GNN property prediction literature and foundation model pretraining scale comparisons.** Per instructions, missing related work claims are not evaluated since external sources cannot be verified.
- **[REMOVED] Criticism demanding theoretical/mechanistic causal analysis of why 3D underperforms.** The paper explicitly scopes to an empirical benchmark; requiring theoretical proofs imposes standards not expected for empirical systems papers at ICLR.
- **[REMOVED] Criticism about "real-world chemistry datasets" vs. offline benchmarks.** This is a minor stylistic/framing nitpick; offline virtual-library BO is a standard and accepted evaluation paradigm in the community.
- **[REMOVED] Criticism about LLA choice vs. deep ensembles, MC dropout, sparse GPs, etc.** LLA is an established, principled Bayesian NN approximation with prior BO-specific validation (Kristiadi et al., 2023/2024); critiquing the surrogate choice is outside the paper's stated scope of comparing representations. The paper cites Li et al. 2024 as motivation for the two-hidden-layer LLA architecture.
- **[REMOVED] Criticism about "ensuring best 10 observations remain in virtual library" being a "problematic design choice."** This is a standard BO benchmarking technique to ensure the global optimum is always reachable during the BO loop; it is not a flaw.
- **[REMOVED] Criticism that "multiple conformers per molecule" should be tested.** Testing single-conformer vs. multi-conformer inputs is outside the stated scope and would be a distinct methodological contribution. (The missing description of *which* conformer is selected—flagged under Weaknesses—is distinct from this.)
- **[REMOVED] Strength: "The paper is well-written / the topic is important / the benchmark is extensive."** These are generic and apply to any paper in the area.

---

## Novel Insights

The most genuinely novel observation that emerges from cross-reading the reviews and paper is the **interaction between representation dimensionality and data regime as a structured, quantified phenomenon in BO** (Section 5.2). Prior work had noted qualitatively that equivariant models can be data-hungry, but grounding this within the BO loop—where the surrogate is incrementally updated—reveals a compounding effect: not only do 3D models need more data to reach parity with 2D models in supervised learning, but the BO acquisition step must compensate for a worse-calibrated surrogate early in optimization. The sample-complexity crossover observed above 10,000 training observations provides a concrete threshold that practitioners can use. This finding, if validated with proper conformer controls and cost measurements, has the potential to be a durable empirical result. The remaining insights (LLM dominance, transfer learning viability) are unfortunately confounded or underdeveloped as detailed above.

---

## Suggestions

1. **Include a pretrained 3D baseline (e.g., Uni-Mol) or a non-pretrained 1D transformer** to isolate representation dimensionality from pretraining scale in the LLM comparisons. This is the single highest-priority revision.
2. **Add at least one conformer-sensitive task** (stereo-isomer property discrimination, docking score, conformer-dependent solvation) so the scope of "3D is not needed" can be properly bounded.
3. **State the acquisition function explicitly** and, if multiple were tested, report sensitivity to acquisition function choice.
4. **Report wall-clock time per BO iteration** (or at minimum per feature-extraction call) for each modality to make the cost-benefit claim empirical rather than qualitative.
5. **Describe the conformer selection protocol** (tool, energy criterion, number of conformers retained) and ideally include a brief ablation or discussion of how conformer quality affects 3D performance.
6. **Harmonize the QM9/LLM finding** between Section 5.1 and the Conclusion, and fix the misidentified representation in the QM9 explanation.
7. **Extend transfer learning experiments to MoleculeNet and GEOM DRUGS** to support the "foundation model" framing.
8. **Include LLM/1D in the sample-complexity figure** (Section 5.2) to complete the picture of all modalities' data efficiency.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. No new algorithm or methodology; the benchmark itself is the contribution. Including 3D representations in BO benchmarking fills a genuine gap, but the experimental design leaves the most interesting comparisons confounded. |
| **Importance of research question** | High. Representation choice in molecular BO is a concrete, practically significant question with direct implications for computational chemistry pipelines. |
| **Claims well-supported** | Partially. The "2D ≥ 3D" finding is reasonably supported across datasets and settings, but the "1D/LLM dominates" claim is critically confounded by pretraining scale. The cost-benefit claim lacks empirical cost data entirely. |
| **Soundness of experiments** | Moderate. The scale and seed count are commendable, but missing protocol details (acquisition function, conformer selection) and the confound between model scale and representation type reduce confidence in the conclusions. |
| **Clarity of writing** | Below expectations for ICLR. Internal inconsistencies (QM9/LLM contradiction between Section 5.1 and Conclusion), missing method details, and ambiguous training setup descriptions impair reproducibility and interpretation. |
| **Value to the research community** | Moderate. The sample-complexity finding and transfer learning analysis are actionable. However, the headline takeaway as currently presented ("3D is not useful") risks being misapplied by practitioners working on tasks where 3D geometry is intrinsically necessary, because no such tasks are tested. |
| **Contextualized relative to prior work** | Adequate. The paper correctly identifies that existing BO benchmarks skip 3D representations and positions itself accordingly. The connection to broader supervised-learning literature on 3D vs. 2D molecular modeling is thin. |

---

## HozsY9Gdcl

- GT: Reject (avg 5.0)
- Predicted: Accept (5.4/10)
- Match: NO

### Final Review

## Summary
Set-MI is a framework for Membership Inference (MI) in Language Models that improves upon individual-document MI by aggregating per-document scoring signals over groups of documents that share a natural membership property (the "set assumption": documents grouped by creation date, language, license, or instruction dataset source are either all in or all out of the training set). Built on top of four existing Individual-MI methods, Set-MI is evaluated on five newly constructed benchmarks spanning Wikipedia, Arxiv, language identity, licensing, and instruction-tuning data, reporting an average AUROC improvement of 0.14. Ablations study the role of model size, deduplication, document length, set size, and aggregation strategy under simulated label noise.

---

## Strengths

- **Novel and diverse benchmark construction.** The paper constructs five domain-spanning benchmarks (Wikipedia, Arxiv, language identity, license category, instruction-tuning datasets) that did not previously exist as set-structured MI testbeds. This is a tangible, reusable contribution beyond the method itself.

- **Practically informative ablations.** The set-size ablation (Figure 4 right) demonstrates that even 3 documents per set yields meaningful gains over individual inference — a specific, actionable result for practitioners. Similarly, the finding that document length saturates around 256–512 tokens gives concrete guidance.

- **Novel finding on deduplication.** The analysis in Section 5.3 reveals that deduplication in training data impacts Set-MI substantially more than Individual-MI, a finding that carries real implications for understanding the relationship between training data processing and memorization.

- **Robustness analysis with verified labels.** Section 6 uses 13-gram overlap to verify actual membership against the Pile, providing a ground-truth reference point that is stronger than the proxy labels used in the main experiments. The controlled noise simulation comparing MAX/MIN/FULL aggregation under different noise configurations is a useful practical guide.

- **Modular and lightweight design.** The framework is plug-and-play on top of any existing Individual-MI method and requires no retraining or model access beyond document-level loss — a practically attractive property.

---

## Weaknesses

### Fatal
None.

### Major

- **Temporal distributional shift confound for Wikipedia and Arxiv (unaddressed).** The set assumption for these two benchmarks is defined by document creation date, which is the *same* attribute used to assign ground-truth labels. Lower model loss on pre-cutoff documents could reflect temporal language/topic shift — documents from 2010 have different statistical properties from documents from 2022 regardless of memorization. The paper never controls for this: there is no experiment comparing a model trained on a different temporal slice of the same data, nor any analysis separating distribution shift from memorization. Since Wikipedia and Arxiv are the primary benchmarks (1,000 sets each, featuring prominently in the abstract's 0.14 average AUROC claim), an uncontrolled confound here substantially weakens the attribution of improvement to MI signal aggregation.

- **Missing metadata-only baseline.** If the set grouping attribute (date, language, license) itself is sufficiently predictive of training membership — independent of any model loss signal — then aggregating over sets could trivially improve AUROC by exploiting metadata structure rather than a better MI signal. The paper never tests a baseline that predicts membership purely from set metadata without querying the model. Without this control, it is impossible to distinguish genuine MI signal amplification from set-structure information leakage. This is a fundamental gap in experimental design.

- **Proxy membership labels in main experiments.** For Wikipedia and Arxiv, ground-truth membership is assigned by creation date relative to the Pile's collection date, not by verifying whether each document actually appears in the Pile. The clean Wikipedia experiment in Section 6 — using 13-gram overlap to verify membership — achieves AUROC near 1.0 at zero noise, substantially higher than the proxy-label Wikipedia results in Table 2 (e.g., 0.575 for Loss Attack). This gap suggests that proxy labels may introduce substantial noise that bounds headline performance, and that the main results reflect label quality as much as MI difficulty. The main experiments would be substantially more convincing if at least one model/domain setting used verified labels.

- **Inconsistencies between Table 1 and benchmark descriptions.** The text for Wikipedia states "We subsample 100 sets with 100 documents per set," but Table 1 reports 1,000 sets and 100,000 documents. Arxiv has the same discrepancy. For Language, the text says "resulting in 130 sets" whereas 20 languages × 10 subsets = 200 sets (matching Table 1). For License, the text again says "resulting in 130 sets" while Table 1 reports 190 sets. These are not trivial discrepancies — they concern the fundamental scale of evaluation and are directly relevant to reproducibility.

### Minor

- **Document-level AUROC conflates set-level decisions.** The paper assigns the aggregated set score to every document in the set and evaluates AUROC over documents. Since all documents in a set receive identical scores, the effective number of independent decision points is the number of sets (e.g., 1,000 for Wikipedia), not the number of documents (100,000). Reporting set-level AUROC alongside document-level AUROC is needed for proper statistical interpretation, particularly for smaller benchmarks (Languages: 200 sets, Instructions: 130 sets).

- **Narrow scope of key ablations.** The deduplication ablation (Section 5.3) uses only Loss Attack on Wikipedia; the document-length ablation (Section 5.4) uses only LiRA on Wikipedia. Broader coverage across at least two domains and two MI methods would be needed to assert these findings generalize. As stated, these are findings specific to one benchmark × one method combination.

- **Robustness analysis restricted to a single setting.** Section 6 uses only Pythia 2.8B dedup, Wikipedia, and Loss Attack. The robustness claims in the abstract ("robust under practical settings") generalize from a single controlled configuration.

### Tiny

- Notation in Section 3: The formal set partition writes "$S_i, S_j \in \mathcal{D}$" but sets are not elements of $\mathcal{D}$ (documents are). Lower-case $s_i$ is also used inconsistently with upper-case $S_i$ in the same block.

- The abstract says Set-MI "enhances prior MI methods"; a small qualification is needed given that zlib entropy on Instructions decreases from 0.458 to 0.429 (Table 2). The text of Section 5.1 handles this correctly but the abstract does not.

---

## Nice-to-Haves

- **Adaptive aggregation selection**: Since practitioners typically lack prior knowledge of which sets are noisier (member or non-member), an automatic or heuristic strategy for selecting MAX/MIN/FULL without oracle knowledge would significantly increase practical utility.

- **Embedding-based set construction**: When explicit metadata is unavailable, clustering documents by semantic similarity to infer sets is a natural extension. A small experiment or discussion of this would help practitioners facing unstructured corpora.

- **Stratified analysis by set distance from membership boundary**: For date-based benchmarks, sets close to the cutoff date are much harder than sets far from it. Reporting AUROC stratified by temporal distance from the cutoff would reveal whether gains concentrate on easily separable sets, which would qualify the practical utility of the headline improvement figure.

- **Set-level AUROC as a complementary metric**: Reporting both document-level and set-level AUROC throughout would make the statistical interpretation of all tables and figures cleaner.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

**Removed criticisms:**

- *LiRA implementation is incorrect*: The paper adapts LiRA to a simplified token-probability ratio form for the black-box loss-only setting. This is a deliberate design choice given the access assumptions, not an implementation error. The paper is internally consistent about what access is assumed.

- *The "any method" framing is fundamentally misleading*: One counterexample (zlib/Instructions) exists and is visible in Table 2. The paper discusses in Section 5.1 that poor Individual-MI can lead to worse Set-MI. This is not a fatal framing issue but a minor qualification already partially handled.

- *Larger dataset sizes / more models in benchmark*: The dataset sizes (100,000 docs for Wikipedia/Arxiv, 20,000 for Languages) and model zoo (Pythia family, GPT-Neo, BLOOM, SILO, Tulu) are adequate for the evaluation claims. Demanding more is generic and not specific to a flaw in this paper.

- *Closed/commercial model evaluation (GPT-4, Claude)*: Testing on models with unknown ground-truth training sets would require speculative membership labels, which could not validate claims rigorously. This is scope creep rather than a weakness.

- *Theoretical proofs of convergence or noise tolerance*: This is an empirical systems paper. Demanding theoretical analysis is not standard practice in this area and goes beyond the paper's stated scope.

- *Confidence intervals for every table*: For benchmarks with 1,000 sets (Wikipedia, Arxiv), single-run AUROC evaluation is standard. The smaller benchmarks (Languages, License, Instructions) do warrant careful interpretation, which is better addressed by the set-level AUROC point above.

- *Unfair baseline comparisons*: No cases of comparisons that are unfairly asymmetric in favor of the baseline were identified.

**Removed generic strengths:**

- "The paper is well-written and clearly structured" — applies to any competently written paper and is not specific to this contribution.

---

## Novel Insights

The most substantive insight from the combined reviews — not explicitly acknowledged in the paper itself — is the dual confound problem for date-based benchmarks: (1) temporal distributional shift (older text has different statistical properties than newer text, independent of memorization) and (2) metadata structure exploitation (aggregating over date buckets may partially reconstruct the membership signal from metadata alone, without the model's loss contributing anything). These two issues operate in the same direction: both would inflate the apparent AUROC improvement of Set-MI on Wikipedia and Arxiv. The paper's Section 6 experiment provides indirect reassurance — verified 13-gram labels with a known training corpus still show high AUROC, suggesting some genuine MI signal — but since Section 6 is not the main experimental setup, the confounds remain live concerns for the headline results. Resolving them would either strongly validate or meaningfully revise the core contribution.

---

## Suggestions

1. **Add a metadata-only baseline**: for each benchmark, report the AUROC of a classifier that predicts membership from set metadata alone (e.g., logistic regression on date, language label, license category) without any model queries. This single experiment would resolve the most important methodological ambiguity in the paper.

2. **Add a distributional shift control for Wikipedia/Arxiv**: evaluate a model whose training cutoff *differs* from the Pile cutoff (but trained on the same general data distribution) on the date-structured Wikipedia/Arxiv benchmarks. If Set-MI still correctly identifies the *actual* training cutoff, this rules out pure distributional shift as the explanation.

3. **Extend verified-label experiments to main results**: run the 13-gram-overlap membership verification (as in Section 6) for at least Pythia 12B on Wikipedia as a main result, not just a robustness check. This would directly address the proxy-label concern for the paper's most prominent benchmark.

4. **Reconcile Table 1 with the benchmark construction text**: clarify whether Wikipedia/Arxiv comprise 100 or 1,000 sets, and whether Language/License/Instructions comprise 130 or 200/190/130 sets.

5. **Report set-level AUROC alongside document-level AUROC** in all main result tables, to avoid conflating set-level statistical power with document-level sample size.

6. **Expand deduplication and document-length ablations** to at least one additional domain and one additional MI method to support the generality claims made in the abstract.

---

## Paper Quality Assessment

**Originality**: Moderate-to-high. The conceptual shift from individual-document MI to set-level aggregation is a clear and well-motivated contribution. Simple aggregation has appeared in clinical NLP contexts, but applying it to web-scale pretraining data using natural metadata is novel and the benchmark construction is a distinct contribution.

**Importance of research question**: High. Training data auditing, evaluation contamination, and copyright analysis are urgent practical problems for LLM deployment and governance.

**Whether claims are well supported**: Moderate. The empirical improvement in Table 2 is clearly demonstrated across diverse settings. However, the two unaddressed confounds (distributional shift, metadata leakage) and the proxy-label issue for main experiments mean the attribution of improvement to MI signal aggregation is not fully established. The headline number (0.14 AUROC average) is directionally reliable but its mechanistic interpretation is uncertain.

**Soundness of experiments**: Moderate. The benchmark variety, model coverage, and ablation breadth are commendable. The narrow scope of individual ablations, conflation of document- and set-level AUROC, and absence of the missing-baseline experiment are meaningful methodological gaps.

**Clarity of writing**: Mostly good, but the inconsistencies between Table 1 and section text are a notable lapse in precision for a paper whose contribution partly rests on benchmark construction.

**Value to the research community**: Moderate-to-high, conditional on benchmark release. The benchmarks and the practical ablation findings (set size, document length, deduplication impact) would be genuinely useful; the core Set-MI method is easy to adopt.

**Contextualization relative to prior work**: Adequate. The differentiation from Jagannatha et al. (2021) and the Individual-MI baselines are well-handled. The paper appropriately scopes to the black-box loss-only setting.

---

## TH4gKbZS1E

- GT: Reject (avg 2.5)
- Predicted: Reject (3.5/10)
- Match: YES

### Final Review

## Summary

This paper presents a controlled empirical benchmark comparing Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on ten hand-curated 1D functions spanning six regularity classes: regular/smooth, continuous-but-non-differentiable, jump-discontinuous, singular, coherently-oscillatory, and noisy variants thereof. The authors match parameter counts between architectures and study the effects of training sample size and optimizer choice (Adam vs. L-BFGS). The central finding is that KAN does not uniformly outperform MLP: KAN appears superior on regular and singular/oscillatory functions, while MLP prevails on cusp and jump-discontinuity functions. The paper is an explicit extension of the authors' prior work (Shen et al., 2024) on KAN noise sensitivity.

---

## Strengths

- **Structured function taxonomy with controlled parameter matching:** The six-category classification is practically motivated, and Table 2 shows near-exact parameter parity (118 vs. 120, and 238 vs. 240 parameters). Matching parameters rather than relying on ad hoc choices is a deliberate and methodologically sound choice that is missing from many KAN vs. MLP comparisons in the literature.

- **Concrete optimizer sensitivity findings with runtime data:** Tables 3 and 4 provide quantitative evidence that KAN with L-BFGS incurs 30–70× wall-clock overhead versus MLP (e.g., 588s vs. 8.3s for f₇). This is a specific, practically consequential finding that practitioners comparing architectures need but seldom find explicitly reported.

- **Non-trivial asymmetric performance pattern:** The empirical result that KAN is worse on cusp/jump functions (f₃–f₆) but better on singular/oscillatory functions (f₇–f₁₀) is not an obvious or foregone conclusion. It distinguishes between function types that other comparative studies treat uniformly and offers a structurally interesting—if underanalyzed—observation about spline-based inductive biases.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Figure 3 caption directly contradicts Section 3.3 text.** The text (Section 3.3) states: "Results show that the MLP outperforms the KAN" for jump functions f₅ and f₆. The alt-text/caption of Figure 3 states: "In all cases, KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line)." These claims are logically incompatible. This contradiction is not a parser artifact—both appear explicitly in the extracted document. It creates fundamental ambiguity about what the actual experimental result is for one of the paper's most important function categories, and directly undermines the reliability of the reported findings.

- **Optimizer mismatch confounds the architectural comparison for f₉.** Figure 8's caption confirms that for f₉, the comparison is MLP (Adam) vs. KAN (L-BFGS). The paper then concludes "KAN consistently surpasses MLP" on coherent oscillations. Since L-BFGS is the better-performing optimizer for KAN on f₉ (per Figure 7), this comparison attributes to *architecture* an advantage that is at least partly attributable to *optimizer*. The same mismatch appears in Figure 11(g) for the noisy f₉ comparison. Claims about architectural superiority on oscillatory functions are therefore not cleanly supported.

- **No statistical reliability analysis.** All results appear to be single training runs with no reporting of variance across seeds, initializations, or data subsamples. Neural network training is stochastic, and the performance gaps in many of the figures are moderate. Without multiple runs and summary statistics, it is not possible to distinguish genuine architectural differences from initialization noise. This is a baseline methodological requirement for empirical comparisons at ICLR.

- **Limited novelty relative to stated prior work.** The paper explicitly states: "This research continues directly and naturally from our recent study on the efficacy of KANs in fitting noisy functions (Shen et al., 2024)." The additional contribution—extending from noisy regular functions to a broader taxonomy of irregular functions—is incremental, and the paper does not articulate a clear mechanistic advance or conceptual insight that goes beyond cataloging performance across function types. The absence of any explanatory analysis (see below) makes this feel like a dataset-of-experiments extension rather than a standalone contribution.

- **No mechanistic explanation for observed performance patterns.** The paper documents that KAN underperforms on cusps and jumps but does not explain why. Is it the smooth B-spline basis that cannot represent discontinuities efficiently? Grid resolution? Optimization landscape geometry near non-smooth targets? Without any mechanistic analysis—even informal—the findings remain a list of empirical observations rather than actionable insights. ICLR expects understanding, not just measurements.

### Minor

- **Shallow and narrow architecture search.** Only single-hidden-layer MLPs ([1,39,1] and [1,79,1]) are evaluated. Multi-layer MLPs are the dominant baseline in modern deep learning and are known to behave differently on approximation tasks. The results may not generalize to the architectures practitioners actually use.

- **Computational cost is measured but underemphasized in conclusions.** Tables 3 and 4 clearly show that KAN with L-BFGS is orders of magnitude slower than MLP. The conclusion's summary of findings does not foreground this tradeoff. For a paper whose audience includes practitioners deciding between architectures, the efficiency cost is as important as the accuracy result.

- **Noise model is insufficiently specified.** Section 4 introduces noise levels of 0, 2, 4, and 10, and Figure 11 uses SNR=0, 4, 10, but the main text never formally defines the noise distribution (Gaussian? uniform?), how SNR is computed relative to the signal, or whether noise is applied to inputs or outputs only. This makes the noisy-function experiments difficult to reproduce or interpret precisely.

- **"Severe discontinuities" is a misleading category label.** In Section 4, singularities (f₇, f₈) and coherently oscillatory functions (f₉, f₁₀) are grouped under "severe discontinuities." These are qualitatively distinct: singularities have well-defined limits that diverge, while coherent oscillations have no limit. Conflating them under one label may obscure differences in behavior.

### Tiny

- The section header "KOMOGOROV-ARNOLD THEOREM" in Section 2 contains a typo ("Komogorov" missing one 'l').

---

## Nice-to-Haves

- **Hyperparameter ablation for KAN (grid size, spline order k).** Only grid=3, k=3 is tested throughout. Since grid resolution directly controls the B-spline's ability to represent sharp features, ablating this setting would clarify whether the observed weaknesses of KAN on jumps and cusps are fundamental to the architecture or an artifact of the chosen configuration.

- **Compute-budget-matched comparisons.** Parameter count matching is a first step, but given the 10–70× runtime gap shown in Tables 3–4, a comparison at equal wall-clock time or equal number of function evaluations would give a more practically relevant picture of the accuracy–efficiency tradeoff.

- **Extension to multivariate functions.** The Kolmogorov-Arnold theorem and KAN's theoretical motivation are specifically about multivariate compositional structure. Including even a few bivariate test cases (e.g., f(x₁,x₂) = sin(x₁)/x₂ with a singularity at x₂=0) would strengthen the paper's connection to KAN's theoretical basis and expand the scope of conclusions.

- **Zoomed visualizations near irregular points.** Global function-fitting plots make it difficult to assess whether either architecture is capturing local behavior near cusps, jump locations, or singularities. Inset zoomed views near x=0 (for f₃, f₇, f₉) and x=±0.5 (for f₅, f₆) would provide clearer diagnostic evidence.

- **Visualization of learned KAN activation functions.** A claimed advantage of KAN is interpretability. Showing the learned univariate activations for a few cases (e.g., whether KAN learns a 1/x-shaped activation for f₇) would both validate or challenge the interpretability claim and provide mechanistic insight into how KAN represents different function types.

---

## Removed Points

*These points were raised by sub-reviewers but are removed or substantially weakened here for the stated reasons — treat them with caution.*

- **Criticism: The paper should evaluate on high-dimensional or real-world datasets.** Removed as scope creep. The paper explicitly scopes its contribution to controlled function regularity benchmarks. Evaluating whether KAN does X well should not be penalized for not also doing Y. Multivariate extension is listed as a nice-to-have but is not a core flaw given the stated scope.

- **Criticism: Formal contributions are not enumerated in the introduction.** Removed as a pure presentation nitpick. The paper's structure makes the contributions inferrable even without a bullet-pointed list.

- **Criticism: The regularity taxonomy is not grounded in formal approximation theory (Sobolev, Hölder, BV).** Removed as an unjustified rigor demand for an empirical paper. The informal categorization is serviceable for the experiments and is pedagogically clear. A formal measure-theoretic taxonomy is not standard practice in this subfield.

- **Criticism: Comparisons where the worse optimizer is used for MLP are "unfair."** Removed — using the weaker optimizer for the *stronger-performing* architecture (MLP) would only strengthen the paper's claims where MLP wins. The cases where the concern is real (optimizer mismatch favoring KAN) are already captured as a genuine Major weakness above.

- **Criticism: References to Shen et al. 2024 and other cited works may not exist.** Removed per instruction — if cited, assumed to exist.

- **Criticism: The title is too narrow.** Removed as a formatting/style concern.

- **Weakness claimed: "KAN exhibits faster convergence across all tested functions" is unsupported.** Partially mitigated — the paper text in Section 5 does make this claim, and the figure descriptions are largely consistent with it (Figure 4: "KAN consistently achieves lower loss than MLP"). The convergence advantage appears consistently observed, even if it must be tempered by the lack of statistical confidence.

---

## Novel Insights

The most interesting and underexploited observation in this paper is the asymmetry between the two main failure modes: KAN underperforms on functions with localized, bounded irregularities (cusps and jumps), yet outperforms on functions with globally extreme behavior (singularities and densely oscillatory near unreachable points). This asymmetry is not explained anywhere in the paper but hints at a genuine inductive bias story: spline-based activations may be well-suited for capturing globally steep or monotone-local behavior but ill-suited for capturing bounded, localized transitions. If the paper could articulate and test this mechanistic hypothesis — e.g., by examining whether increasing grid density helps on cusps but not jumps, or whether the spline coefficients exhibit pathological behavior near discontinuities — it would transform a catalog of observations into an explanatory contribution.

---

## Suggestions

1. **Immediately resolve the Figure 3 vs. Section 3.3 contradiction.** Check whether the figures show KAN or MLP winning on jump functions and align caption, figure, and text to reflect the actual data. This is the single most urgent correction.

2. **Fix the optimizer mismatch for f₉.** Either run MLP and KAN under the same best optimizer for each (with clear reporting) or present a full 2×2 optimizer × architecture factorial design for the affected functions, so architectural and optimizer effects can be disentangled.

3. **Run each experiment with at least 5 random seeds and report mean ± std.** This applies to all learning curves and final test loss values. Even condensed to a supplementary table, this would substantially increase the evidential value of the comparisons.

4. **Add a mechanistic analysis section.** For the most striking results (KAN's failure on jump functions and success on singularities), provide at least an ablation: vary grid size for KAN on f₅/f₆ and f₇/f₈, and report whether increasing grid resolution closes the gap on jumps. This would directly test the hypothesis that spline resolution (rather than architecture fundamentals) drives the observed differences.

5. **Define the noise model precisely** in Section 4, specifying distribution, parameterization of noise levels, and whether test loss is evaluated against clean or noisy labels.

6. **Foreground the computational cost tradeoff** in the abstract and conclusion. The 10–70× runtime overhead of KAN is a practically critical result that deserves equal prominence to the accuracy comparisons.

---

## Evaluation

| Axis | Assessment |
|------|------------|
| **Originality** | Low-to-moderate. The function taxonomy is useful, but the paper is explicitly incremental relative to the authors' prior work (Shen et al., 2024), and the empirical KAN-vs-MLP comparison space is now crowded. No new methodology, theoretical insight, or analytical framework is introduced. |
| **Importance of research question** | Moderate. Knowing when KAN helps vs. hurts relative to MLP is practically relevant for the many researchers now considering KAN as an alternative. |
| **Claim support** | Weak. The Figure 3 contradiction, optimizer mismatch for f₉, and absence of multi-seed statistics mean several headline claims cannot be taken at face value without correction. |
| **Soundness of experiments** | Weak. Parameter matching is careful, but the absence of statistical analysis, the shallow-architecture restriction, and the optimizer mismatch undermine the controlled-comparison framing. |
| **Clarity of writing** | Adequate. The structure is easy to follow, but the Figure 3 contradiction, imprecise noise model, and misleading category label ("severe discontinuities") are genuine clarity failures. |
| **Value to research community** | Limited in current form. The asymmetric performance pattern across regularity classes is potentially useful, but the methodological issues mean practitioners cannot fully trust the results without replication. |
| **Contextualization relative to prior work** | Adequate acknowledgment of the negative KAN literature; the explicit framing as an extension of Shen et al. 2024 is honest, though it weakens the novelty claim. |

---

## lNuGCXxvkn

- GT: Reject (avg 5.2)
- Predicted: Accept (5.5/10)
- Match: NO

### Final Review

## Summary

This paper derives non-asymptotic Sobolev-norm learning curves for kernel ridge and ridgeless regression applied to elliptic linear inverse problems. The central finding is that the PDE operator—because its eigenvalues *grow* with index (negative $p$)—dampen high-frequency variance sufficiently to enable benign overfitting in fixed spatial dimension, contrasting with standard regression where this requires high-dimensional asymptotics. A secondary contribution quantifies how the smoothness of the inductive bias ($\beta$) affects convergence: rates are independent of $\beta$ once a threshold $\lambda\beta \ge \frac{\lambda r}{2} - p$ is met, a condition that surprisingly matches Bayesian inverse-problem literature.

---

## Strengths

- **Benign overfitting in fixed spatial dimension via inverse-problem structure.** The specific mechanism—eigenvalues of $\mathcal{A}$ growing with index ($p < 0$) causing the spectrally transformed covariance $\tilde{\Sigma} = \mathcal{A}^2 \Sigma^\beta$ to have a steeper effective decay than a pure kernel covariance—is a concrete and non-obvious explanation distinguishing inverse problems from regression. This is not a generic claim but is precisely tracked through Theorem 4.2 and Remark 7.

- **Unified bias-variance framework covering both ridge and ridgeless estimators.** The same spectral decomposition apparatus (Theorems 3.6, 3.7) and the same concentration coefficient $\rho_{k,n}$ yield bounds for both settings by changing how $k$ is chosen, and the regularized case provably recovers the known minimax rates of Lu et al. (2022), providing internal consistency validation.

- **Cross-paradigm agreement of the smoothness threshold.** The condition $\lambda\beta \ge \frac{\lambda r}{2} - p$ derived from a frequentist upper-bound analysis independently reproduces the smoothness condition from Bayesian inverse-problem theory (Knapik et al., 2011; Szabó et al., 2013). This unexpected alignment strengthens confidence in the correctness of the bound.

- **Empirical confirmation that sufficiently smooth activations give rate-independent convergence.** Figure 1 (Left) shows convergence curves for ReLU through ReLU$^4$; the ReLU$^3$ and ReLU$^4$ curves are near-indistinguishable, directly illustrating the theoretical threshold prediction. This is a clean qualitative test even if not a quantitative rate validation.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Bounded-output assumption (2.2a) is inconsistent with the Gaussian noise model used throughout the analysis.** Assumption 2.2(a) states $y$ is bounded almost surely by $M$, yet Section 3.2 and Theorem 4.2 assume $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, which is unbounded. Because the concentration inequalities in Theorem 3.5 rely on boundedness, this is not a cosmetic inconsistency—it is a gap in the proof foundations. The paper must either replace boundedness with a sub-Gaussian assumption throughout, or verify that the specific Gaussian noise case is handled separately. This needs to be corrected before the theory can be trusted as stated.

- **No kernel experiments despite all theory being about kernel estimators.** The theoretical contributions concern kernel ridge/ridgeless regression (Lemma 3.1, Theorems 3.6, 3.7, 4.1, 4.2), yet every experiment uses neural networks. There is no direct empirical test of the kernel estimator, no comparison of ridged vs. ridgeless kernel behavior at matched settings, no verification of the predicted polynomial rate exponents (e.g., $n^{\lambda(\beta'-r)/(2p+\lambda\beta+1)}$), and no ablation over $p$ (PDE order) or $\beta$ (Sobolev regularization exponent) in the kernel regime. This creates a fundamental mismatch between the paper's theoretical core and its empirical support.

- **The bias bound carries an explicit $1/\delta$ factor, rendering the high-probability statement unusually weak.** Theorem 3.7 bounds bias as $\lesssim \rho_{k,n}^3 \cdot \frac{1}{\delta} \cdot [\ldots]$, and this factor propagates to Theorem 4.2. A bias bound that blows up as $\delta \to 0$ means the stated rates are only achievable at constant confidence levels, not in the usual high-probability sense. Neither the paper nor the appendix discusses whether this $1/\delta$ is removable (e.g., by a union bound or Markov inequality argument) or is a fundamental artifact of the proof technique. This undermines the precision of the main learning-curve claims.

- **The benign overfitting conclusion depends critically on $\rho_{k,n} = \Theta(1)$, which is only established under sub-Gaussian features.** Remark 6 acknowledges that in the worst case $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$; since the variance and bias bounds scale as $\rho_{k,n}^2$ and $\rho_{k,n}^3$ respectively, benign overfitting in the general case is not established—it relies on the sub-Gaussian assumption that the paper simultaneously claims to avoid. The main text does not adequately foreground this conditionality; Section 4.2's prose implies a cleaner conclusion than the theorem actually delivers.

### Minor

- **Only a single PDE (2D Poisson) with one synthetic ground truth is tested.** While the Poisson equation is illustrative, the paper claims general applicability to elliptic inverse problems. The Schrödinger equation in Example 2.3 is introduced as a motivating case but never experimentally tested. A second PDE—especially a higher-order one to test the $p$-dependence of variance stabilization—would considerably strengthen the empirical story.

- **Neural network experiments do not verify proximity to the kernel/NTK regime.** The theoretical–experimental bridge depends on networks operating in or near the lazy training regime. No diagnostic (e.g., relative parameter change, NTK alignment) is provided to justify using kernel theory to explain the neural network results. Without this, the connection is speculative.

- **No comparison between regularized and interpolating estimators in experiments.** One of the paper's primary contributions is unifying ridge and ridgeless estimation, yet Figure 1 only shows the PINN interpolator vs. a plain NN interpolator. A direct ridged vs. ridgeless comparison under matched conditions would validate the regime transition claim.

- **The simultaneous diagonalization assumption (Assumption 2.2d) is strong and its limitations are underexplored.** The paper correctly cites its prevalence in prior work and notes that it holds for shift-invariant kernels with the Laplacian on the torus. However, it does not discuss what happens to the rates under perturbative misalignment, nor whether the spectral framework degrades gracefully or catastrophically when the assumption is violated. A brief analysis of sensitivity would significantly improve practical credibility.

- **The closed-form solution (Lemma 3.1) uses the population covariance $\Sigma^{\beta-1}$, not an estimable empirical quantity.** The paper mentions the semi-supervised analogue in passing but does not analyze the practically realizable estimator. The gap between the oracle estimator studied and a computable implementation should be stated explicitly as a limitation, especially given the practical framing of the introduction.

### Tiny

- **Smoothness threshold notation is inconsistent between Section 1.1 and Remark 5/Section 4.3.** Section 1.1 writes the threshold as $\lambda\beta \ge \frac{\lambda^r}{\lambda^p} - p$ whereas Remark 5 and Section 4.3 write it as $\lambda\beta \ge \frac{\lambda r}{2} - p$. The latter version is consistent with the surrounding analysis; the former appears to be a formatting artifact. Since this threshold is a core advertised contribution, the inconsistency should be corrected and a single canonical statement should appear prominently.

- **Abstract overstates the generality of the rate-independence claim.** The statement "the convergence rate is actually independent to the choice of (smooth enough) inductive bias" omits that this holds above a smoothness threshold on $\beta$ that itself depends on problem parameters; below the threshold the rate is suboptimal. A brief qualification in the abstract would be more accurate.

---

## Nice-to-Haves

- **Log-log convergence rate plots with predicted theoretical slopes overlaid.** Figure 1(Left) uses only 5 sample sizes on a linear scale. Log-log plots with slope annotations would allow readers to assess whether the predicted exponents (e.g., $n^{\lambda(\beta'-r)/(2p+\lambda\beta+1)}$) are empirically supported.

- **Spectral visualization of variance stabilization.** Plotting the effective eigenvalues $\lambda_i^\beta p_i^2$ vs. $i$ for several values of $p$ (including $p=0$ for comparison) would directly illustrate the paper's core mechanism and make the theoretical insight more accessible to practitioners.

- **A perturbation analysis or discussion of near-commutativity.** Even an informal argument about how rates degrade when $\mathcal{A}$ and the kernel covariance operator nearly (but not exactly) share eigenvectors would address the most frequently raised practical concern.

- **Direct verification that the kernel method's ridge vs. ridgeless transition matches Theorems 4.1 and 4.2.** A controlled kernel experiment varying $n$, $\beta$, and PDE order would close the gap between theory and experiment in the cleanest possible way.

---

## Removed Points

*These points were flagged for removal. Treat them with caution; they may reflect misreadings.*

- **"Fixed dimension" is not formalized.** The critic argued that "fixed dimension" is never defined. However, the paper uses polynomial eigendecay $\lambda_i \propto i^{-\lambda}$ with $\lambda > 1$, which is the standard capacity condition holding in any fixed spatial dimension $d$; Remark 2 explicitly ties $\lambda$ to Matérn/Sobolev kernels on $\mathbb{T}^d$. The contrast with high-dimensional benign overfitting (which typically requires intrinsic dimension diverging) is made via citation. The framing is adequate, not missing.

- **Neural-network experiments compare "PINN vs. NN" unfairly.** The critic implied this comparison favors the PINN and is therefore meaningless. However, the asymmetry is intentional: the purpose is to show that incorporating inverse-problem structure (PDE operator) changes the noise sensitivity of an otherwise identical interpolating architecture. This is exactly the comparison needed to validate the benign overfitting claim, and the asymmetry favors the baseline (plain NN), not the author's method.

- **Claims that the discussed kernel estimator is too abstract without worked examples.** The paper provides Example 2.3 (Schrödinger equation on the torus), Table 1 with parameter semantics, and Remark 2 connecting co-diagonalization to Fourier modes. While more worked-out rate instantiations would help (addressed in Nice-to-Haves), the abstraction level is acceptable for a theory paper targeting kernel inverse-problem experts.

- **Claims about unfairness of comparisons with Barzilai & Shamir (2023).** The paper explicitly recovers Barzilai & Shamir's result as a special case ($p=0, \beta=1, \beta'=0$), which is a genuine consistency check, not an unfair comparison.

- **Requests for theoretical proofs bridging to neural networks.** The paper explicitly frames the neural network section as validation "beyond kernel estimators" and does not claim the theory directly covers neural networks. Demanding formal NTK-regime proofs is outside the paper's stated scope.

---

## Novel Insights

The most genuinely novel insight in this work—one not merely restating the paper's claims—is the realization that the spectral structure of the *inverse* operator, rather than high intrinsic dimensionality, can serve as the mechanism for variance self-regularization. In standard regression, benign overfitting requires the effective rank of the feature covariance to grow with $n$; here, the PDE operator shifts the effective eigenvalue spectrum so that the transformed covariance $\tilde{\Sigma} = \mathcal{A}^2 \Sigma^\beta$ satisfies the necessary spectral-decay conditions for benign overfitting even when the ambient (spatial) dimension is fixed. A direct corollary is that the two axes traditionally conflated—data dimension and statistical complexity—can be decoupled in the inverse-problem setting, with physical operator order substituting for dimensionality. The surprising agreement between the frequentist smoothness threshold and the Bayesian posterior contraction condition (Knapik et al., 2011) reinforces this and hints at a deeper structural reason why both paradigms converge on the same regularity requirement for inverse problems.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | High. The link between PDE operator order and benign overfitting in fixed dimension is novel; the unified kernel Sobolev framework for ridge and ridgeless inverse-problem estimators has not appeared in this form before. |
| **Importance of research question** | Substantial. Understanding generalization of overparameterized PDE solvers is an open and practically relevant question. |
| **Claims well-supported** | Partially. The regularized case is well-supported and recovers prior rates. The interpolation case has meaningful gaps: the $1/\delta$ bias factor and the $\rho_{k,n}$ conditionality mean the benign overfitting claim is established only under additional assumptions that are not always foregrounded clearly. |
| **Soundness of experiments** | Weak. The experiments are exclusively neural-network-based despite an entirely kernel-theoretic paper; there are no quantitative rate validations; and the setting is limited to a single 2D PDE. |
| **Clarity of writing** | Moderate. The high-level narrative is clear and Sections 1–2 are readable, but Theorems 3.5–3.7 are not digestible in the main text without more interpretive corollaries, and there are notational inconsistencies. |
| **Value to the research community** | Meaningful, particularly for the physics-informed ML theory community. The variance-stabilization insight and the smoothness threshold are actionable results. |
| **Contextualization relative to prior work** | Mostly adequate. Comparisons to Lu et al. (2022), Barzilai & Shamir (2023), and Haas et al. (2024) are present, though the Lu et al. comparison could be made more explicit (a direct proposition mapping theorem parameters would help). |

Overall, the paper contains a real and interesting theoretical contribution. However, the inconsistency between the boundedness assumption and Gaussian noise model, the absence of any kernel experiments, the $1/\delta$ bias issue, and the under-qualified dependence on $\rho_{k,n}$ in the main benign-overfitting claim represent substantive issues that need to be addressed before the work fully delivers on its promises.

---

## cojJ2s1e35

- GT: Reject (avg 5.5)
- Predicted: Accept (5.7/10)
- Match: NO

### Final Review

## Summary
WLA (World modeling through Lie Action) is an unsupervised framework that models environment transitions as Lie group actions operating linearly on object-centric slot-based latent representations. Rather than learning a separate world model per environment, WLA trains a single cross-environment simulator that captures continuous and compositional dynamics; it then solves the Controller Interface Problem (CIP) by learning a lightweight adapter (`Ctrl_adapt`) from labeled action sequences to the learned Lie algebra parameters. The framework is evaluated on Phyre (qualitative), 8 ProcGen game environments, and a real-world Android robotics dataset, showing large improvements over Genie in controllability-specific metrics.

---

## Strengths

- **Single unified model across 8 diverse ProcGen environments.** Most world model work trains per-game; WLA learns one model jointly. The fact that this outperforms a per-environment baseline (Genie) in all games on Δ_t PSNR (from near-zero or negative for Genie to clearly positive for WLA, e.g., coinrun: 0.48 → 9.03; ninja: 0.05 → 4.06; bigfish: −0.09 → 1.26) is a non-trivial result and a concrete empirical confirmation of the cross-environment benefit.

- **Least Action Principle for slot alignment is a creative and effective contribution.** The ablation (Table 1) shows a meaningful performance drop when it is removed (MSE 0.675 → 0.602 on unseen), and it addresses a known failure mode of slot-attention in temporal settings without requiring extra supervision.

- **Two-stage design cleanly separates unsupervised world modeling from supervised controller adaptation.** The pre-trained `(Φ, Ψ)` transfers to a new environment requiring only a small labeled dataset for `Ctrl_adapt`, which is a practically attractive property for the robotics setting.

- **Android FVD result (393.85 → 131.02) demonstrates real-world temporal coherence.** FVD captures distribution-level video quality and long-range consistency; this improvement is substantially larger than might be explained by per-frame reconstruction differences, suggesting the Lie-structured latent model genuinely improves temporal dynamics on unstructured robot video.

---

## Weaknesses

### Fatal
None.

### Major

- **Only Genie is used as a baseline, which prevents attributing gains to specific design choices.** The paper's core hypothesis is that Lie-structured continuous latent actions outperform discrete or unstructured continuous latents. To test this, the comparison needs at least one continuous latent dynamics model without Lie structure (e.g., a slot-attention model with standard recurrent latent transitions, or DreamerV3 adapted for video prediction). Without this control, the observed gains could plausibly come from the object-centric architecture or the state-space formulation alone, rather than the Lie group structure specifically. The two included ablations (w/o rotation, w/o least action) are internal and do not address this.

- **The "minimal or no action labels" claim in the abstract is misleading.** The abstract states WLA "can be trained using only video frames and, with minimal or no action labels, can quickly adapt to new environments with novel action sets." Section 4.3 clarifies that `Ctrl_adapt` requires a labeled dataset `{(x[t], a[t])}`. The "no labels" story applies only to pretraining `(Φ, Ψ)`, not to the full pipeline that actually enables controllable interaction. Crucially, the paper never measures how many labeled trajectories are needed — there is no label-efficiency experiment, so "minimal" is unquantified and the adaptation claim is unverified. This is a central promise of the work and it is unsupported.

- **Cross-environment generalization is not clearly demonstrated.** The paper trains on all 8 ProcGen games jointly and calls unseen evaluation "out-of-domain," but does not clearly specify what "unseen" means: new procedurally generated levels of the same games, or held-out game types? If the former, this is in-distribution generalization, not the cross-environment transfer that motivates the paper. There is no experiment that trains on a subset of games and tests on held-out games, which would directly validate the inter-environmental generalization claim.

### Minor

- **The commutativity assumption (A(s) commute over time) is acknowledged in Section 7 but not empirically characterized.** This assumption underpins the closed-form solution in Eq. (4) and the rollout in Eq. (9), and it is not merely a modeling simplification — it rules out sequences where action order matters (e.g., "jump then run" ≠ "run then jump"). The paper provides no analysis of how much degradation this causes in non-commutative settings, nor any indication of which environments satisfy it approximately.

- **Phyre evaluation is entirely qualitative with no baselines.** Phyre is used as a "sanity check" for interpolation and composition, but no quantitative metrics are reported and no baseline model is compared. This makes it impossible to assess whether the demonstrated behaviors arise from the Lie group structure specifically or from any reasonable generative model with a smooth latent space.

- **Implementation is underspecified for reproducibility.** Critical details are missing: number of slots N, number of Lie actions J, latent dimensionality, learning rate, batch size, rollout length during training, number of training steps, whether hyperparameters are shared across all ProcGen games and Android, and specifically how the Android architecture was "slightly adapted." The footnote in Section 4.2 ("these parameters are not to be stored as parts of the model") hints at an unusual optimization scheme (per-trajectory per-timestep trainable λ, θ), but the relationship between these optimized values and the IDM outputs is not clearly explained. This raises the question of whether the IDM performs amortized inference or is merely post-hoc supervised on memorized trajectory codes.

- **The learned (λ, θ) parameters are not analyzed semantically.** The paper claims WLA learns compositional action primitives, but never demonstrates that individual (λ, θ) dimensions correspond to interpretable actions (e.g., consistent horizontal motion, rotation, etc.) across environments or even within one. Without this, the Lie group parameterization could just be a flexible coordinate system that fits training data without capturing the claimed compositional structure.

- **Notation in Eq. (6) overloads F^{-1}_{Φ,Ψ}.** In Section 3.1, F_{Φ,Ψ} is defined as the IDM mapping observation-space transitions to latent matrix representations (g → M(g)). Its inverse should map latent operators back to observation transitions. But in Eq. (6), F^{-1}_{Φ,Ψ} is used to map Lie algebra parameters (λ, θ) to transition matrices M_{t,δ} via matrix exponentiation — a different operation. These are distinct mathematical maps and the overloading without explanation creates genuine confusion.

- **The commutativity assumption should appear explicitly in Section 3, not only in Section 7.** Eq. (4)'s closed form z(t) = exp(∫A(s)ds)z(0) and Eq. (9)'s rollout both assume A(s) commute across time. This is a non-trivial restriction that shapes the entire framework, but the paper introduces it as a limitation only in the conclusion rather than as a modeling assumption in the technical presentation.

### Tiny

- **Slot alignment description contains a notation ambiguity.** Section 4.4 states: "we choose the permutation σ to the slots so that the transition z_n[t+1] → z_σ(n)[t+1] in the latent space is minimal." This appears to describe matching future-frame slots to other future-frame slots, whereas the intended operation should relate current-frame slots z_n[t] to future-frame slots z_σ(n)[t+1]. The description should be clarified.

- **Eq. (3) order of composition.** The Fact states F_{Φ,Ψ}(h·g) = F_{Φ,Ψ}(g)·F_{Φ,Ψ}(h), which is anti-homomorphic unless a right-action convention is being used. The paper does not explain the convention, which will cause confusion for mathematically oriented readers. A sentence clarifying the action convention would resolve this.

---

## Nice-to-Haves

- A label-efficiency experiment plotting adaptation performance as a function of the number of labeled trajectories (1, 5, 10, 50, ...) would directly validate the "minimal labels" claim and is probably the single most impactful addition.
- An experiment with true cross-game generalization: train on 6 ProcGen games, adapt to 2 held-out games with few labels.
- A continuous latent dynamics baseline (e.g., slot-attention model without Lie constraints, or RSSM-style model) to isolate the contribution of the Lie group structure from the object-centric architecture.
- Analysis of rollout degradation over longer horizons — the Lie group structure theoretically promotes stability and this should be measurable.
- Visualization of slot assignments across frames, verifying that slots track objects consistently over time.
- Hyperparameter sensitivity of N (slots) and J (Lie actions), since these are user-specified and practitioners need guidance.
- Timing/computational cost comparison with Genie, especially for the linear assignment problem solver in slot alignment.
- Failure case analysis showing where WLA breaks (strongly non-commutative dynamics, objects appearing/disappearing, stochastic environments).
- A qualitative visualization showing which (λ, θ) dimensions correspond to which actions, to provide evidence for the "compositional primitives" claim.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **[REMOVED] Concern about Genie baseline unfairness due to increased training iterations.** The paper gave Genie 2× training iterations (0.4M vs. 0.2M default) to accommodate the multi-environment setting. This asymmetry favors the baseline, not the proposed method — it is therefore a more conservative comparison, not an unfair one.

- **[REMOVED] Demand for confidence intervals and multi-seed statistics.** Single-run evaluation is standard for ProcGen-scale benchmarks in world modeling. Requesting this as a weakness imposes a non-standard rigor requirement.

- **[REMOVED] Requests for specific missing related works.** Per review policy, we do not flag missing related works when external sources cannot be verified.

- **[REMOVED] Criticism that the introduction "doesn't pin down novelty sharply."** The paper provides a clear contribution statement and the combination of Lie group structure, object-centric encoding, and cross-environment training is distinguishable from prior work. Demanding a more formal contribution list is stylistic.

- **[REMOVED] Framing that "Valevski et al. lack identity mapping under no action" is too broad.** This is a specific technical characterization the paper makes in related work, not a misrepresentation of Valevski et al.'s goals.

- **[REMOVED] Criticism of no downstream robotic task metric on Android.** The paper is a world model for video prediction; evaluating it by downstream task success would impose an out-of-scope requirement. The video prediction metrics (FVD, PSNR, Δt PSNR) are appropriate for the stated contribution.

- **[REMOVED] General claims that the paper "overclaims"** without specific grounding — several sub-claims the harsh critic calls overclaims are either qualified by the paper or do have experimental support (the large Δt PSNR gains specifically validate controllability improvements relative to Genie).

---

## Novel Insights

The juxtaposition of the three reviews reveals one genuinely important insight beyond the paper's own contributions: the commutativity assumption is the most fundamental constraint in WLA's design, yet it is the least empirically characterized. The paper assumes transitions commute, implements rollout under this assumption (Eq. 9), and notes it as a limitation only in the conclusion — but it never tests what happens in environments where commutativity provably fails. If this assumption fails silently on real game dynamics (where "jump then move" ≠ "move then jump"), it could explain why the model has near-zero performance degradation in Table 1's ablations while still leaving headroom for non-abelian extensions. A targeted commutativity stress-test would not only validate the current scope but also motivate the future work the authors themselves propose.

---

## Suggestions

1. **Run a labeled-data ablation (1 / 5 / 10 / 50 / 100 labeled trajectories for `Ctrl_adapt`).** This is the most important addition — it directly validates the "minimal labels" claim that currently lacks any quantitative support.

2. **Clarify the "unseen" setting precisely.** Specify whether "unseen" in Table 1 means (a) held-out procedurally generated levels of the same games, or (b) held-out game types. If (a), add an experiment with held-out game types to demonstrate true inter-game transfer.

3. **Add one continuous-latent baseline.** A slot-attention model with standard diagonal SSM transitions (no Lie group structure, same architecture otherwise) would directly test the hypothesis that the Lie structure drives the gains.

4. **Add design a commutativity test.** Identify at least one ProcGen environment where action order provably matters, and measure WLA's performance there relative to the in-domain results. This would honestly characterize the scope of the model.

5. **Clarify the IDM/per-trajectory optimization pipeline.** Explicitly state whether {λ_{nj}[t], θ_{nj}[t]} are stored per-trajectory (transductive) or predicted by the IDM (amortized), and explain how the IDM is supervised in the former case.

6. **Fix the Eq. (6) / F^{-1} notation.** Introduce a distinct symbol (e.g., exp_G or π) for the map from Lie algebra parameters to the matrix group element, separate from F^{-1}_{Φ,Ψ} which should denote the observation-space map.

7. **Move the commutativity assumption to Section 3**, adjacent to Eq. (4), and frame it as a modeling assumption rather than a limitation discovered in hindsight.

8. **Include full hyperparameter table in the appendix** covering N, J, latent dimension, training steps, optimizer, learning rate, batch size, rollout length, and any Android-specific modifications.

---

## Evaluation on Key Axes

**Originality:** *High.* The combination of Lie-structured linear latent dynamics, object-centric slot attention, and joint cross-environment world modeling is a distinct contribution. The Least Action Principle for slot alignment is a creative addition not seen in prior work. While individual components (equivariant autoencoders, slot attention, state-space dynamics) are borrowed, their integration and application to multi-environment world modeling is novel.

**Importance of research question:** *High.* Generalizable, controllable world models are a foundational problem for planning and policy learning. Tackling cross-environment generalization without per-environment training is both ambitious and practically motivated.

**Claims well supported:** *Moderate.* The controllability gains over Genie on ProcGen are large and consistent across 8 games, giving good support to the core claim that Lie-structured latents improve action responsiveness. However, the "minimal labels" and "cross-environment generalization" claims are not quantitatively supported and rely on an ambiguous experimental setup.

**Soundness of experiments:** *Moderate.* Results are reported without variance, the unseen/out-of-play protocol is underspecified, only one external baseline is used, and the Phyre section is purely qualitative. The ablations cover two components but do not isolate the Lie structure from the object-centric design.

**Clarity of writing:** *Moderate.* The high-level idea is clearly communicated. However, the mathematical sections have meaningful notation issues (Eq. (3) ordering, Eq. (6) overloading), the role of per-trajectory optimized parameters is ambiguous, and the implementation section lacks sufficient detail for reproduction. The paper explicitly defers rigorous formalism to the appendix (which was not available for review).

**Value to the research community:** *Moderate-to-high.* If the results hold under stronger baselines, WLA would represent a meaningful advance in structured world modeling. The code is implied but not released; the reproducibility gaps currently limit immediate uptake.

**Contextualization relative to prior work:** *Good.* The paper engages substantively with Genie, LAPO, NFT/Koyama et al., and VPT, and clearly articulates how WLA differs from each. The framing around CIP is novel and useful. Some relevant continuous-latent dynamics methods (Koopman operators, DreamerV3) are mentioned only briefly.

---

## BZQmpsuW7D

- GT: Reject (avg 5.8)
- Predicted: Reject (4.6/10)
- Match: YES

### Final Review

## Summary
SPARK is a physics-guided augmentation framework for dynamical system modeling that addresses data scarcity and distribution shift. It builds a vector-quantized discrete memory bank conditioned on boundary information and physical parameters, augments training samples by mixing latent representations with retrieved codebook entries, and employs a Fourier-enhanced graph ODE for long-horizon prediction. Experiments span five benchmarks (Prometheus, ERA5, Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) and include OOD and cross-domain transfer evaluations.

---

## Strengths

- **SPARK genuinely functions as a plugin, supported by concrete evidence.** Figure 1 shows backbone+SPARK outperforming backbone across ViT, CNO, U-Net, and NMO on ERA5. Table 3 demonstrates SPARK applied to three distinct backbones (SimVP, PredRNN, Earthfarseer) with consistent improvements on SEVIR under varying data fractions. This is specific, multi-backbone validation that most augmentation papers do not provide.

- **Energy spectrum evaluation goes beyond standard MSE.** Figure 6 compares SPARK, Swin-T, and FNO on power spectra for Navier-Stokes, Spherical-SWE, and 3D Reaction-Diffusion. This is a physically meaningful diagnostic—important for the SciML community—and SPARK visibly recovers high-frequency structure that FNO and Swin-T miss.

- **Cross-domain transfer experiment with controlled data fractions is a strong evaluation design.** Table 3 systematically varies target-domain data from 20% to 100% while transferring from ERA5 to SEVIR, and shows that SPARK+backbone consistently outperforms backbone alone (whereas vanilla backbone transfer can actually degrade performance at high data fractions). This directly tests the stated data-scarcity motivation.

- **Benchmark diversity is above average.** Five datasets spanning synthetic PDEs (Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) and real meteorological data (ERA5, SEVIR), with both OOD and non-OOD splits, is a thorough empirical scope for a single paper.

---

## Weaknesses

### Fatal
*None identified, but the combination of Major weaknesses below would significantly undermine confidence in the results without revisions.*

### Major

- **Complete absence of ablation study.** The method has four distinct components: boundary/parameter injection (§3.2), the VQ memory bank (Eq. 5–6), the memory-bank augmentation (Eq. 7), and the Fourier-enhanced graph ODE (§3.3). Table 1 only reports "Ours + SPARK" as a monolithic system; there is no table removing one component at a time. It is therefore impossible to determine whether the gains come from the augmentation mechanism (the paper's core claim), the bespoke downstream predictor, or some combination. This is the single most critical omission for an ICLR submission making component-level claims.

- **Unexplained ERA5 baseline discrepancy undermines comparison fairness.** In Table 1, FNO, UNO, and CNO report MSE of 0.7233, 0.6652, and 0.5243 on ERA5 (w/o OOD), while ViT achieves 0.0762 and NMO achieves 0.0432. Neural operators are specifically designed for PDE problems and typically outperform generic vision backbones on such tasks; a 20× gap versus vision models is anomalous and unexplained. Possible causes (unsuitable preprocessing, incompatible resolution, poor tuning) are never discussed. Until clarified, the ERA5 results cannot be trusted, which affects the headline comparison.

- **Augmented sample label preservation is conceptually unjustified.** Eq. (7) produces augmented inputs by interpolating a sample's latent representation with Top-K codebook entries from potentially different physical environments. The corresponding output label $\mathcal{Y}_i$ is the original future state, implicitly assumed to be preserved under this mixing. However, if the retrieved codes represent different boundary conditions or parameter regimes, the future dynamics associated with the mixed input could differ substantially from the original label. The paper does not address this, and it is a fundamental question for the validity of the augmentation scheme.

- **OOD split definitions are absent for all datasets.** Table 1 reports w/o OOD and w/ OOD numbers, and Table 4 repeats this, but the paper never specifies what constitutes the OOD condition for each benchmark—whether it is an unseen parameter range, unseen boundary condition, unseen temporal period, or some combination. The near-zero OOD degradation for SPARK on ERA5 (0.0322 → 0.0321) raises the question of whether the OOD split is sufficiently challenging. Without defining the shift, the OOD claim cannot be evaluated.

- **No data scarcity experiments on primary benchmarks.** The abstract and introduction prominently motivate data scarcity, yet Table 1 uses full training data for all five benchmarks. The only scarcity experiment is Table 3, which conflates limited data with cross-domain transfer. A direct evaluation varying training data fraction (e.g., 10%, 25%, 50%) on Prometheus or Navier-Stokes is needed to validate the stated motivation.

### Minor

- **Notation conflicts impede reproducibility.** The symbol $\delta$ denotes physical parameters in Eq. (2) but appears as an activation/transform in Eq. (8) without redefinition. Similarly, Eq. (5) assigns $z_i$ via argmin (making it an index), yet it is described in the text as "the nearest neighbor code embedding." These inconsistencies are not merely cosmetic; they make it difficult to implement the method correctly.

- **No VQ codebook diagnostics.** VQ-VAE methods are known to be prone to codebook collapse where only a small fraction of codes are ever used. The paper employs the standard straight-through estimator (sg[·]) but reports no codebook utilization, perplexity, or dead-code percentage. If the memory bank is degenerate, the augmentation mechanism would not function as claimed.

- **Figure 5 (sea ice RQ2) is self-referential.** The quantitative evidence for RQ2 consists of SPARK's own training loss, SSIM, and PSNR curves over 80 epochs. These curves demonstrate that the model trains successfully but provide no comparison against baselines on the sea ice task. The qualitative Figure 4 compares against FNO and U-Net, but not against NMO (the strongest baseline in Table 1).

- **Physical parameters are ill-defined for ERA5.** §3.2 conditions channel attention on "physical parameters $\delta$" (scalars like viscosity). For ERA5, the paper says it uses u, v, and humidity as "forcing terms"—these are dynamic spatiotemporal fields, not scalar parameters in the same sense as viscosity or diffusion coefficient. How these are projected into a single parameter vector for Eq. (2) is not explained.

- **"Quantitative" vs. "quantized" title inconsistency.** The paper title says "quantitative augmentation" while the abstract, methodology, and conclusion consistently use "quantized augmentation" (referring to VQ-VAE). These are different concepts. The title should match the method.

### Tiny

- The theoretical section (§3.4) presents standard information-theoretic and PAC-Bayes generalization bounds with physical prior $\mathcal{P}$ substituted in. The conclusion that "physical priors reduce $I(\theta; \mathcal{D} | \mathcal{P})$" is stated as an implication of the theorem, but it is actually an assumption. The theorems do not analyze VQ discretization, the augmentation rule, or the Fourier graph ODE, and are not connected to any measurable quantity in the experiments. The theory as written does not add analytical insight specific to SPARK.

---

## Nice-to-Haves

- **Compare augmentation against simple baselines (MixUp, noise injection).** The physics-guided memory bank mixing is the central novelty over standard augmentation; a direct comparison would strengthen the claim.

- **Sensitivity analysis for key hyperparameters** ($\lambda$, $K$, memory bank size $M$). These are central to the augmentation behavior and their robustness across datasets is unknown.

- **Training and inference time alongside Table 2.** Table 2 explores model size but omits wall-clock training/inference time and memory overhead. Since the method adds VQ pretraining, GNN encoding, and ODE solving on top of any backbone, cost transparency would help practitioners.

- **Codebook visualization (t-SNE/UMAP colored by physical parameter).** Visualizing whether the learned codes organize by physically meaningful axes would provide qualitative evidence that the memory bank captures physics rather than memorizing training samples.

- **Controlled OOD severity analysis.** Stratifying OOD splits by shift magnitude (e.g., small vs. large viscosity extrapolation) would make the robustness claims more precise and informative.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Plugin claim is unvalidated"** (Harsh Critic, strong form): Removed. Figure 1 explicitly shows backbone+SPARK vs. backbone for multiple architectures on ERA5, and Table 3 applies SPARK to SimVP, PredRNN, and Earthfarseer. The plugin is validated, though the main Table 1 does not isolate it — which is a fair ablation concern but different from the plugin claim itself.

- **Missing error bars / no statistical tests** (Harsh Critic): Removed. Single-run evaluation is the norm for large-scale benchmarks in this community. Multiple-run statistics are not standard practice for neural operator and dynamical systems benchmarks at ICLR scale.

- **Ethics statement is perfunctory** (Harsh Critic): Removed. This is a style/formatting concern and ICLR does not mandate extensive societal impact statements for applied ML papers.

- **Requests for missing related work citations** (Harsh Critic): Removed per instructions; external references cannot be verified.

- **Title does not signal technical novelty precisely enough** (Harsh Critic): Removed as a pure style nitpick. The title issue kept is the substantive "quantitative" vs. "quantized" semantic confusion.

- **Strength: "comprehensive empirical evaluation" / "well-written"** (Reviewer 2): Removed as generic. The specific strength retained is the breadth and design of the transfer experiment (Table 3), not generically "extensive experiments."

---

## Novel Insights

The most underappreciated aspect of the paper is the interaction between the physics-aware discrete memory bank and transfer learning. Table 3 reveals a striking asymmetry: vanilla backbone transfer from ERA5 to SEVIR **hurts** performance at higher data fractions (e.g., SimVP degrades 15.79% at 100% SEVIR, PredRNN degrades 8.70% at 100%), while SPARK+backbone transfer consistently helps. This suggests that SPARK's physics-conditioned quantization may act as a domain-invariant regularizer—filtering out ERA5-specific distributional artifacts that would otherwise cause negative transfer—rather than simply providing more training signal. This mechanism is not analyzed in the paper but is worth investigating: if true, it would explain why SPARK's benefit is largest in low-data regimes and implies a specific use case beyond generic augmentation.

---

## Suggestions

1. **Add an ablation table** removing each of the four components (boundary encoding, parameter channel attention, VQ memory bank, Fourier graph ODE) on at least Prometheus and Navier-Stokes. This is the highest-priority revision.

2. **Explain or fix the ERA5 FNO/UNO/CNO results.** If these operators genuinely perform poorly on this ERA5 formulation (e.g., because the task is on irregular grids), explain why and note that the comparison is one-sided in SPARK's favor for those methods — this actually makes SPARK's advantage over NMO more meaningful.

3. **Define OOD splits precisely** in the experimental section or appendix: for each dataset, state what physical parameter ranges, boundary conditions, or temporal windows are held out, and quantify the shift magnitude.

4. **Report codebook diagnostics** (% active codes, assignment entropy, average nearest-neighbor distance in codebook space) to validate that the VQ memory bank is functioning as intended rather than collapsing.

5. **Add at least one direct data scarcity curve** (training fraction vs. MSE) on a primary benchmark (e.g., Prometheus) to directly substantiate the data-scarcity claim from the abstract.

6. **Justify label preservation** in the augmentation (Eq. 7): either argue theoretically that interpolation in the learned physics-conditioned latent space preserves output labels, or test empirically that augmented samples have lower prediction error than purely random latent interpolations.

7. **Fix the $\delta$ notation conflict** between §3.2 (physical parameters) and Eq. (8) (activation function), and reconcile the $z_i$ index vs. embedding ambiguity between Eq. (5) and the surrounding text.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. The combination of VQ-VAE memory bank with physics-conditioned attention and graph ODE for augmentation in dynamical systems is novel in its integration, though each individual component (VQ, GNN, Neural ODE, Fourier operator) is well-established. The augmentation-for-OOD framing in physical systems is a fresh angle. |
| **Importance of research question** | High. Robustness to distribution shift and data scarcity in physical system modeling are genuine bottlenecks in scientific ML with real deployment consequences. |
| **Claims well supported** | Weak to moderate. The headline performance gains (Table 1) are plausible but tainted by unexplained baseline discrepancies (ERA5 operators) and complete absence of ablations. The plugin claim is supported. The data scarcity claim is not directly demonstrated on primary benchmarks. |
| **Soundness of experiments** | Moderate concern. The five-benchmark comparison is broad, and the transfer experiment is well-designed. However, missing ablations, undefined OOD splits, and the ERA5 operator anomaly prevent confident interpretation. |
| **Clarity of writing** | Adequate but with specific notation conflicts that impede reproduction (δ overloading, z_i as index vs. embedding, unclear spectral convolution definition on irregular graphs). |
| **Value to the research community** | Moderate to high if ablations confirm that the VQ augmentation is the driver of gains. The transfer learning finding (SPARK mitigates negative transfer) is potentially the most valuable and least discussed result. |
| **Contextualized relative to prior work** | Adequate at a high level; the paper covers the right related work categories. More discriminative positioning against retrieval-augmented and memory-based scientific ML would strengthen the novelty argument. |

---

## QKBu1BOAwd

- GT: Accept (Oral) (avg 7.3)
- Predicted: Accept (5.6/10)
- Match: YES

### Final Review

## Summary

DRAFT is a framework that iteratively refines tool documentation for LLMs through a three-phase self-driven loop: an Explorer probes the tool with diverse queries and captures execution results, an Analyzer identifies discrepancies between documentation and observed behavior and proposes targeted revision suggestions, and a Rewriter synthesizes these inputs to produce an updated documentation version plus directions for the next exploration round. The process is governed by a diversity-promoting exploration strategy (cosine similarity constraint + self-reflection) and a tool-adaptive termination mechanism (BLEU + embedding cosine similarity convergence). Experiments on ToolBench and RestBench with three LLMs (GPT-4o, GPT-4o-mini, Llama-3-70B) consistently outperform static documentation rewriting baselines, with secondary benefits demonstrated on tool retrieval.

---

## Strengths

- **Consistent, non-trivial gains across heterogeneous LLMs and datasets.** On ToolBench (I3-Instruction), CP% improvements over EasyTool reach 5–7 points across all three tested LLMs (Table 1), and on RestBench-TMDB the gains are even larger (e.g., GPT-4o: 79→88 CP%). The fact that gains transfer to GPT-4o-mini and Llama-3-70B using documentation refined with GPT-4o specifically demonstrates that the refined documentation captures model-agnostic usability improvements, not just idiosyncrasies of the backbone.

- **Cross-model generalization with a weaker backbone (Figure 7).** The paper tests Llama-3-70B as a refinement backbone—a weaker, open-source model—and shows that its refined documentation still benefits all three evaluation models on RestBench-TMDB. This is a practically important finding: organisations without GPT-4o access can still benefit from DRAFT.

- **Secondary retrieval benefit (Table 3).** Showing that DRAFT-refined documentation improves both sparse (BM25) and dense (Contriever) tool retrieval is a meaningful bonus, demonstrating that the improvements reflect genuine semantic enrichment rather than being a narrow inference-time artifact.

- **A notable cross-class result (Table 1, ToolBench).** GPT-4o-mini + DRAFT achieves 47 CP% on ToolBench, exceeding the raw GPT-4o at 37 CP%. This is a striking concrete demonstration of the practical value of documentation quality.

- **Human evaluation confirms improved completeness and accuracy.** Table 4 shows strong human preference for DRAFT documentation, especially on ToolBench (68% DRAFT vs 4% raw for completeness, 56% vs 0% for accuracy). The zero raw-preferred accuracy on ToolBench is notable.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing single-pass rewrite with execution traces baseline.** The paper's central thesis is that *iterative* trial-and-error is essential for documentation quality. However, the critical missing comparison is: give the backbone LLM the tool's original documentation plus a set of real execution traces (i.e., the same observations DRAFT generates), and ask it to rewrite the documentation in a single pass. Without this ablation, the paper cannot substantiate the claim that iteration—rather than simply having execution feedback at all—drives the improvement. EasyTool is a static rewrite without execution feedback, but it is not a single-shot rewrite *with* execution feedback. This gap directly undermines the necessity of the iterative framework.

- **No ablation removing the Analyzer module.** DRAFT's modular decomposition (Explorer → Analyzer → Rewriter) is presented as a contribution. However, there is no ablation comparing against simply feeding the Explorer's execution traces directly to a Rewriter without the intermediate Analyzer pass. Without this ablation, it is unclear whether the Analyzer's structured suggestion layer adds value over an end-to-end "observe and rewrite" prompt.

- **Algorithm 1 contains a pseudocode bug.** In Lines 15–18, when the convergence condition (Δ > τ) is satisfied at iteration *i*, the algorithm breaks *before* reaching Line 19, meaning t_i (the converged version) is never added to the output set D̃. The last version added would be t_{i-1}. Conversely, if no convergence is detected and all *I* rounds complete, D̃ accumulates t_1 through t_I—multiple intermediate versions per tool—rather than a single refined documentation. The intended semantics (use only the final version per tool) are not implemented as written, and the paper should clarify or correct this.

### Minor

- **Termination criterion is unvalidated.** The claim that BLEU + cosine similarity convergence "prevents overfitting" is not empirically confirmed. Figure 6 shows performance peaks and then declines, but the paper does not demonstrate that the adaptive termination mechanism actually halts at or near the peak iteration for each tool. A concrete analysis—e.g., showing the round at which the mechanism fires vs. the round of peak performance—would substantiate this claim.

- **Query diversity ≠ behavioral coverage.** The similarity constraint (Eq. 2) enforces semantic dissimilarity between generated *queries*, not between *API parameter configurations* or *response regimes*. Two semantically distinct queries may invoke identical tool behavior, while two semantically similar queries may exercise different parameter paths. For tools where important edge cases arise from parameter combinations, boundary values, or authorization conditions, the proposed diversity mechanism may leave meaningful behavioral regions unexplored. This is not acknowledged.

- **No cost or efficiency reporting.** DRAFT requires multiple LLM calls (Explorer + Analyzer + Rewriter) per iteration per tool, potentially up to five rounds. The paper provides no reporting on average API call counts, token usage, wall-clock time, or total refinement cost per tool. For a practically motivated system, this absence makes it impossible to assess the cost–benefit trade-off relative to one-shot methods like EasyTool.

- **Cross-model generalization evidence is narrower than claimed.** The abstract states "robust cross-model generalization capabilities." In reality, the main experiments use three models from two provider families (GPT-4o, GPT-4o-mini, Llama-3-70B), and the backbone generalization experiment (Figure 7) covers only one additional backbone (Llama-3-70B) on one dataset (RestBench-TMDB). The word "robust" is not warranted by this evidence. The explanation offered (shared transformer structure, pretraining corpora) is speculative and untested.

- **Retrieval results are mixed.** Table 3 shows gains on TMDB with both retrievers but Contriever @10 on Spotify slightly *decreases* (49.6→49.2). The paper interprets this section positively without acknowledging the mixed pattern; the claim should be softened to "can improve retrieval in most settings."

- **Ablation scope is limited.** Table 2 reports ablations for only one dataset (TMDB) and one model (GPT-4o) in the main paper. The performance drops for w/o diversity and w/o adaptive are modest (88→84 and 88→80 CP%), and without variance, it is unclear whether these are robust. The paper states appendix results show similar trends, which partially addresses this.

- **No limitations section.** The paper does not discuss when DRAFT may fail or should not be used: tools with rare dangerous failure modes, stateful or rate-limited APIs, tools that cannot be safely probed, or cases where exploration quality is poor because initial documentation is nearly empty. These are important practical caveats.

### Tiny

- **§2.5 contains unsubstantiated claims.** "Dynamically maintaining an accurate and up-to-date representation of evolving features" is listed as a strength (§2.5) but is never tested—no experiment with temporally evolving tools is presented. This should be framed as a potential benefit or future direction rather than an established strength.

- **High "Equal" rates in human evaluation are not discussed.** Table 4 shows Equal rates of 44–70% on several criteria for RestBench. While DRAFT is still clearly preferred, the substantial "Equal" proportion implies improvements are real but not dramatic in many cases; acknowledging this candidly would strengthen the paper's credibility.

- **The termination threshold τ and similarity threshold φ are not sensitivity-tested.** The paper uses τ=0.75 and φ=0.9 without showing how performance changes with alternative values, making it unclear how sensitive results are to these choices.

---

## Nice-to-Haves

- **Documentation length control.** Longer documentation can trivially provide more information. Reporting average length before and after refinement, and optionally including a length-matched baseline, would confirm that DRAFT improves information *density* rather than just quantity.

- **Iteration trajectory visualization.** A case study showing what concretely changes in the documentation across iterations—what is added, removed, or restructured—would make the improvement mechanism transparent and provide intuition beyond aggregate metrics.

- **Failure case analysis.** Win% is not 100%, meaning DRAFT occasionally produces worse documentation. Characterising when and why refinement degrades performance would increase trust in deployment.

- **Hyperparameter sensitivity study.** A sensitivity analysis for τ (e.g., 0.6, 0.75, 0.9) and φ (e.g., 0.8, 0.9, 0.95) would support reproducibility across different tool domains.

- **Testing a more distinct model family** (e.g., Mistral, Qwen) for cross-model generalization would substantially strengthen that finding.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **"No statistical significance / confidence intervals"** (harsh critic, major): WEAKENED to minor. Single-run evaluation with expensive API-based LLM benchmarks is the norm in this community. The lack of variance reporting is a limitation, but it does not invalidate results where gains are consistently 5–10+ CP points across multiple datasets and models.

- **"Analyzer and Rewriter use the same LLM is a problem"** (harsh critic): REMOVED. Using the same backbone for all roles is standard practice in LLM agent papers. The paper need not justify this choice.

- **"text-embedding-ada-002 is dated / proprietary"** (harsh critic): REMOVED. Using an available, well-established embedding API is a reasonable engineering choice. Criticising the specific embedding model as a weakness is not substantive.

- **"No formal objective is defined"** (harsh critic): REMOVED. This is an empirical systems paper; formal objectives are not expected.

- **"The method reads as advocacy in §2.5"** (harsh critic): REMOVED as style nitpick; the substance (evolving tools not tested) is captured under Tiny weaknesses.

- **"Figure 1(c) is potentially confusing"** (harsh critic): REMOVED. The confusion appears to be a PDF-to-text rendering artifact, not a genuine paper flaw; the caption clearly states the figure highlights that DRAFT documentation is more favored.

- **"EasyTool comparison is unfair because DRAFT uses more compute"** (harsh critic): WEAKENED to Minor (subsumed into cost reporting weakness). The comparison itself is intentionally asymmetric in favour of EasyTool (simpler, cheaper); the authors' stronger claim is that even with a harder comparison, iterative feedback adds value. This is an acceptable scientific choice.

- **"The problem statement mixes multiple failure modes"** (harsh critic): REMOVED. Categorising documentation failure modes (incompleteness, redundancy, inaccuracy) is appropriate motivational framing, not a methodological flaw.

- **"DRAFT cannot guarantee correctness for unobserved tool behaviors / overconfident docs"** (harsh critic safety concern): REMOVED as outside the paper's stated contribution scope; not expected for this type of paper.

---

## Novel Insights

The most genuinely novel observation in this paper—partially surfaced by reviewers but underemphasised by the authors—is the **information-asymmetry gap between execution feedback and documentation text**. Human-authored documentation fails not because of bad writing, but because its authors cannot exhaustively probe tools in the way an automated framework can. DRAFT demonstrates that systematically inducing diverse tool executions and using the gap between observed outputs and documented behaviour as a revision signal is more effective than documentation rewriting guided purely by linguistic analysis. The secondary finding that documentation refined by a weaker open-source backbone (Llama-3-70B) still generalises to improve stronger models suggests that the informational content captured through tool interaction is largely model-agnostic—an insight with implications for building shared, model-independent tool interfaces. The key open question the paper leaves unaddressed is whether the **iterative** structure is necessary or whether a single, well-designed batch of diverse execution traces is sufficient for a one-shot rewrite to achieve equivalent gains.

---

## Suggestions

1. **Add a single-shot baseline with execution traces.** Run one round of DRAFT's exploration to collect N diverse execution examples, then ask the backbone to rewrite the documentation once using all examples simultaneously. Compare this to the iterative DRAFT. This is the most important missing experiment and should be in the main paper.

2. **Add an Analyzer ablation.** Compare DRAFT against a version where the Rewriter is given execution traces directly (no Analyzer intermediate step). This tests whether the structured suggestion layer contributes.

3. **Fix Algorithm 1.** Either move Line 19 (̃D update) so it fires before the convergence break (and only the final version per tool is retained), or explicitly specify that D̃ stores only the last accepted documentation per tool and clarify the pseudocode accordingly.

4. **Report average iteration counts and token/API cost per tool** in a table alongside performance, so readers can assess practical efficiency.

5. **Add a figure or table** showing the round at which adaptive termination fires per tool vs. the round of peak performance, to validate the termination mechanism's effectiveness.

6. **Add a limitations section** covering: poor initial documentation stalling exploration, stateful/rate-limited APIs, tools that cannot be safely probed, and cost at scale.

---

## Evaluation on Key Axes

**Originality:** Moderate. The application of iterative self-refinement to *tool documentation* via *actual tool execution feedback* is a genuinely novel angle; the individual components (self-refinement, similarity-based diversity, convergence detection) are not new. The integration is the contribution, and it is well-motivated.

**Importance of research question:** High. Documentation quality is a genuine bottleneck for LLM agents, and the problem is practically significant at scale.

**Claims well supported:** Partially. The claim that DRAFT improves tool-use performance is well supported. The claim that the *iterative* framework is necessary (vs. a single-shot rewrite with execution feedback) is not supported due to the missing baseline.

**Soundness of experiments:** Moderate. Multiple datasets, multiple models, and a secondary retrieval analysis are solid. However, the missing single-shot baseline, limited ablation scope, and absent variance reporting reduce confidence in the sufficiency of the experimental validation.

**Clarity of writing:** Good. The high-level idea and staged decomposition are clear. Method details (algorithm correctness, evaluation criteria for the Analyzer) are under-specified relative to ICLR standards.

**Value to the research community:** Moderate-to-good. A practical, plug-and-play documentation improvement pipeline that benefits both retrieval and execution is immediately deployable. The cross-model generalization finding enhances this value.

**Contextualization relative to prior work:** Adequate. The distinction from EasyTool (no execution feedback) is clear; the paper could more sharply position against broader self-improvement and active probing literature.

---

## 6S4WQD1LZR

- GT: Accept (Poster) (avg 6.7)
- Predicted: Accept (6.8/10)
- Match: YES

### Final Review

## Summary

This paper establishes that deep transformers are universal approximators for continuous in-context mappings when contexts are modeled as probability measures over token embeddings. The key contribution is handling an arbitrary (even infinite) number of tokens with a single architecture of **fixed embedding dimension and fixed number of heads (proportional to the output dimension, independent of both precision ε and token count n)**. Results are proved for both unmasked (e.g., ViT-type) and masked causal (e.g., autoregressive NLP) settings, with the latter requiring a novel "space-time lifting" and additional regularity assumptions on contexts.

---

## Strengths

- **Genuinely new separation of architectural hyperparameters from precision and token count.** Prior work (Yun et al., 2019) requires embedding dimension to grow with token count n; this paper achieves universality with fixed width by moving to a measure-theoretic formulation. The contrast is explicit and clearly articulated, and the difference is substantial rather than incremental.

- **Space-time lifting for causal attention (Section 2.3).** Encoding token order via a time coordinate and restoring permutation invariance in the measure-theoretic domain is a non-obvious and elegant device. The result that the discrete causal formula (Eq. 3) is exactly recovered from the space-time empirical measure formulation directly validates the construction.

- **Injectivity of the Laplace-like transform as the proof's technical lynchpin (Lemma 1).** The separation argument in Proposition 1—reducing density (Stone-Weierstrass) to injectivity of L(μ)(a,c) = ∫ e^{c⟨a,y⟩}⟨a,y⟩ / ∫ e^{c⟨a,z⟩}dμ dμ(y)—is a nontrivial and interesting technical contribution, not a routine application of standard machinery.

- **Proof strategy for approximating products via depth (Lemmas 2–3).** Attention layers do not form a multiplicative algebra; the paper's workaround—building an algebra of "cylindrical functions" via elementary single-head attention units, then approximating componentwise multiplication via MLP depth—is conceptually clean and honestly described with explicit architectural bounds (d_tok(θ_ℓ) ≤ d + 3d', H(θ_ℓ) ≤ d').

- **Honest and precise accounting of limitations.** The paper identifies, in the main body, that (i) the result is non-quantitative, (ii) head count grows with output dimension, (iii) token norm growth through layers is not bounded, and (iv) the masked case requires identifiability and Lipschitz-in-time contexts—with Remark 1 proving identifiability is *sharp* and not improvable.

---

## Weaknesses

### Fatal
None.

### Major

- **Non-quantitative approximation bounds.** There is no control on the depth L or MLP parameter count as a function of ε. The paper explicitly states this (Section 3.1: "we have no explicit control over the dependency of the number of MLP parameters ξ_ℓ on ε") and defers quantitative bounds to future work. For a universality theorem at ICLR, this is a genuine gap: without rates, one cannot distinguish a vacuously true existence result from a practically useful approximation guarantee. The paper argues that MLP squaring approximation "should behave well," but provides no bound, and there is no analysis of whether token norms stay bounded through the construction, creating the possibility of a numerically unstable construction. This gap is particularly salient given that depth L is the free variable being used to achieve approximation.

- **Identifiability condition for the masked case is sharp but not connected to practice.** Theorem 2 requires the target map to be "identifiable" (Definition 3), which the paper proves is tight. However, the paper provides no analysis of whether standard ICL tasks—next-token prediction, in-context regression, or sequence completion—actually produce identifiable maps in the measure-theoretic sense. Without at least one worked example or structural result showing that natural tasks satisfy identifiability, the masked result risks being a theorem about a carefully circumscribed class with unclear intersection with practical settings.

### Minor

- **Incomplete continuity justification in Proposition 1 (point 1).** The proof sketch states that γ_λ is continuous "because the denominator...is not always zero." Non-vanishing is necessary but not sufficient; continuity of the ratio jointly in (μ, x) under the weak* × ℓ² topology also requires that numerator and denominator vary continuously—which follows from weak*-continuity of integration against bounded continuous functions on compact Ω, but this step is not stated. Since the entire theorem hinges on Proposition 1, this argument should be completed in the main text, not left implicit.

- **Architectural gap: normalization layers are omitted.** The paper states upfront (Section 2) that normalization is omitted "for simplicity," but does not discuss whether the universality results extend when normalization is included or whether its omission is essential for the proof machinery. LayerNorm is a core component of practical transformers and changes the representational geometry; the paper should at minimum argue why the omission is harmless or identify it more explicitly as a scope limitation.

- **"Slight adjustments" for RoPE likely understated.** The conclusion says extending to RoPE requires "slight adjustments." RoPE modifies the attention kernel in a position-dependent way that changes the form of the inner products ⟨Q^h x, K^h y⟩ used throughout the measure-theoretic formulation. Whether the injectivity of the Laplace-like transform and the algebra structure are preserved under RoPE-modified kernels is not obvious and may require non-trivial new arguments. The characterization as "slight" should be hedged.

- **Central proof ingredients too compressed for expert verification.** Lemma 1 (injectivity of L) and Lemma 5 (compactness of X_σ^σ) are the two results on which the unmasked and masked theorems respectively hinge, yet both are deferred to appendices with only sketch-level intuition in the main text. The paper would benefit from at least one additional key step of the injectivity argument appearing in the main body.

### Tiny

- **Notation inconsistency in Section 4.** Definition 1 uses C for the Lipschitz constant and σ for the mass-at-0 threshold, introducing Lip_C^σ. The reduced space is then written X_σ^σ, which silently sets C = σ. This conflation should be made explicit (e.g., X_{C,σ}^σ with C = σ as a specific choice).

- **Equation (7) uses ○ where ◇ seems intended.** The in-context composition operator ◇ is defined in Eq. (5)–(6) specifically to track how context updates propagate. Eq. (7) reverts to ○ for some compositions, which is either an overloading or a notational inconsistency that should be clarified.

---

## Nice-to-Haves

- **Even a single synthetic empirical illustration.** The paper mentions in Appendix D that the framework covers in-context regression. Showing a concrete transformer construction reproducing known behavior (e.g., linear regression in context, following Akyürek et al. 2022 / von Oswald et al. 2023) would bridge theory and practice and make the abstract claim about "performing regression within context" concrete.

- **Theorem-level comparison table with prior universality results.** A summary comparing Yun et al. (2019), Nath et al. (2024), Alberti et al. (2023), and this work across: number of heads, embedding dimension vs. n, topology of approximation, function class approximated, and whether masking is handled—would make the novelty immediately parseable.

- **Discussion of which practical tasks satisfy identifiability.** Even an informal argument that standard autoregressive prediction tasks satisfy Definition 3 would substantially increase the impact of Theorem 2.

- **Explicit corollary for ICL regression in the main text.** Appendix D shows transformer universality for regression operators; promoting this as a corollary to the main text would make the connection to in-context learning concrete for ICLR readers.

- **Discussion of trainability gap.** The paper's conclusion briefly notes the connection to Chizat & Bach (2018) but does not elaborate. A short paragraph on whether the specific constructions produced by the proof are gradient-accessible would be valuable, even if speculative.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **[REMOVED – intended design, not a flaw]** Harsh Critic: "unmasked theorem does not address order-sensitive behaviors." The permutation-equivariant unmasked setting is precisely the stated scope (ViT-type models). The paper correctly and explicitly states this; it is not a gap.

- **[REMOVED – likely OCR/parsing artifact]** Harsh Critic: "d_cok notation vs. d_tok." The paper clearly establishes d_tok as the token dimension; the apparent "d_cok" in Lemmas 2 and 3 is almost certainly a parsing error, not an authorial inconsistency.

- **[REMOVED – generic, applies to all theory papers]** Spark Finder: "no comparison experiment with Yun et al.'s construction." Empirical comparisons between universal approximation constructions are not a standard expectation for theory papers in this setting.

- **[REMOVED – paper correctly states this]** Reviewer 2: "abstract claims 'fixed number of heads' but heads scale with output dimension d'." The abstract reads "a fixed number of heads (proportional to the dimension)"—this is accurate. The head count does not scale with ε or n; it scales with the output dimension d', which is a fixed property of the target task, not of the approximation regime. The abstract is correctly calibrated.

- **[REMOVED – asymmetric comparison favorable to baseline is intentional]** No specific instance here, but any critique of omitting normalization as making the result "stronger than the real model" is moot: the omission, if anything, weakens the guarantee relative to practice.

---

## Novel Insights

The most genuinely novel technical insight is the use of a generalized Laplace-like transform L(μ)(a,c) = ∫ e^{c⟨a,y⟩}⟨a,y⟩ / ∫ e^{c⟨a,z⟩}dμ dμ(y) to separate probability measures, which converts the hard problem of point-separation on an infinite-dimensional space P(Ω) into a question about injectivity of a moment-generating-function-type map. This is cleanly different from moment-matching arguments used in other universality proofs. The second notable structural insight is that the *lack* of a multiplicative algebra structure in shallow attention—a limitation relative to MLP universality—is precisely what forces depth: the paper is the first to make this architectural necessity explicit and to show that depth (rather than width) is the correct resource for compensating for this algebraic deficiency. The space-time lifting technique also merits attention as a general method for handling causality in measure-theoretic settings beyond transformers.

---

## Suggestions

1. **Prove at least a coarse depth bound.** Even an exponential bound L = O(exp(1/ε)) would distinguish the result from a vacuous existence claim and significantly increase its value for the theory community.

2. **Complete the continuity argument in Proposition 1.** Explicitly state that weak*-continuity of ∫ f dμ in μ (for bounded continuous f on compact Ω) is being used in the denominator and numerator of γ_λ, and that this implies joint continuity in (μ, x).

3. **Promote Appendix D's regression result to a main-text corollary.** This is the clearest connection to in-context learning as studied empirically, and it is currently invisible from the main body.

4. **Address the identifiability condition concretely.** Show that at least one standard autoregressive task (e.g., stationary process next-token prediction) satisfies Definition 3, or explain why the condition may generically fail and what that implies for the practical scope of Theorem 2.

5. **Add a brief note on whether the normalization omission is essential or incidental.** If LayerNorm can be absorbed into the MLP blocks without breaking the proof architecture, state this explicitly; if not, identify it clearly as a technical limitation requiring future work.

---

## Evaluation

| Axis | Assessment |
|------|-----------|
| **Originality** | High. The measure-theoretic formulation for arbitrary token count, the Laplace-transform injectivity argument, and the space-time lifting for causal attention are all non-routine contributions that distinguish this work from prior universality results. |
| **Importance of research question** | High. Expressivity of transformers under arbitrary context length is a foundational open question; this paper makes meaningful progress. |
| **Claims well supported** | Moderate. The mathematical structure is sound and the proof strategy is clearly laid out, but the key lemmas (injectivity, compactness) are deferred, the continuity argument in Proposition 1 is incomplete as stated, and the lack of quantitative bounds limits the strength of the supporting evidence. |
| **Soundness of approach** | Good. The Stone-Weierstrass strategy is well-chosen and the algebraic structure of the argument is clean. The identifiability sharpness result (Lemma 13) is notably rigorous. |
| **Clarity of writing** | Good for the unmasked section; the masked section (Section 4) is denser and suffers from notational overloading that makes it harder to follow. |
| **Value to the research community** | Solid. The unmasked theorem is a genuine theoretical advance and the proof techniques (especially Laplace transform separation and depth-for-multiplication) may find broader use. The masked result is narrower but still contributes. |
| **Contextualization relative to prior work** | Adequate in the body; a comparison table at the theorem level would make the advances more immediately legible. |

---

## Acvo2RGSCy

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (5.8/10)
- Match: YES

### Final Review

## Summary
DeLLMa (Decision-making Large Language Model assistant) proposes a four-step inference-time framework for LLM-based decision making under uncertainty: (1) latent state enumeration, (2) verbalized state forecasting, (3) utility elicitation via LLM-ranked pairwise preferences fit with a Bradley–Terry model, and (4) Monte Carlo expected-utility maximization. The framework is evaluated on 120-instance benchmarks in agricultural planning and stock selection, demonstrating consistent accuracy improvements over zero-shot, CoT, and self-consistency baselines across GPT-4, Claude 3, and Gemini 1.5, and outperforming o1-preview zero-shot by a wide margin.

---

## Strengths

- **Novel decision-theoretic scaffold for inference-time LLM reasoning.** While prior work scales inference compute via CoT, ToT, or self-consistency, DeLLMa is the first to explicitly decompose inference into state forecasting and utility elicitation guided by expected-utility maximization—a genuinely distinct contribution over reasoning-trace-based methods.

- **Consistent cross-model improvement.** DeLLMa yields accuracy gains across all three evaluated model families (GPT-4, Claude 3, Gemini 1.5), demonstrating the generality of the scaffolding rather than a quirk of one model's tendencies.

- **Inference-time compute scaling with meaningful signal.** Figure 3 shows monotonically increasing accuracy with both sample size and overlap percentage—an empirically grounded scaling property that is directly relevant to the inference-time compute literature.

- **Competitive with o1-preview at similar cost.** Table 3 shows that DeLLMa (GPT-4, $n=64$) achieves 73.3% and 64.2% vs. o1-preview's 33.3% and 35.0% on agriculture and stocks, at comparable per-instance cost. This is a striking empirical result that motivates task-specialized pipelines over general reasoning models.

- **Human evaluation of utility elicitation.** Rather than simply reporting end-to-end accuracy, the paper also evaluates the intermediate utility elicitation step via a 412-sample human agreement study, which is a meaningful attempt at component-level validation not common in prompting-framework papers.

---

## Weaknesses

### Fatal
None.

### Major

- **Evaluation conflates decision quality with realized outcome, undermining the theoretical framing.** The paper's primary metric is accuracy against the *ex-post* optimal action—the fruit or stock that actually turned out best. Decision theory, however, evaluates decisions against the *expected utility* given information available at decision time, not realized outcomes. A method that makes a well-reasoned decision with good beliefs can easily "fail" this metric if a low-probability event occurs, while a method that gets lucky with poor beliefs can "succeed." This is not a minor framing issue—it is a direct contradiction of the paper's own decision-theoretic justification. The normalized utility in Appendix B partially addresses this but is relegated to a secondary result. The paper should either (a) construct an evaluation where ground-truth expected utility under the true data-generating distribution is known, or (b) explicitly and prominently acknowledge that accuracy here measures *forecast quality* rather than *decision rationality*, and reframe claims accordingly.

- **Utility elicitation contribution is not isolated.** Both domains have analytically computable utilities from forecasted outcomes (price×yield, monthly return). A critical missing baseline is "forecast-and-pick": use the LLM-forecasted state distribution to compute expected value per action directly, without any Bradley–Terry ranking or utility elicitation. Without this, it is impossible to determine whether DeLLMa's gains come from the decision-theoretic scaffolding as a whole, from state forecasting alone, or specifically from the utility elicitation module presented as a core contribution. Table 2's ablations address state forecasting variants but not the removal of the elicitation step entirely.

### Minor

- **Independence assumption in state forecasting is unquantified.** The paper explicitly posits factor independence "for computational simplicity" (Algorithm 1). In both domains, the violated correlations are substantial (climate ↔ yield in agriculture; macroeconomic health ↔ individual stock growth in stocks). While the paper acknowledges this, there is no experiment or analysis showing how much this distorts downstream expected-utility estimates. Table 2 ablates forecast quality but not the factorization structure itself.

- **State forecasting ablation (Table 2) undermines the module's apparent importance.** For GPT-4 and Gemini 1.5, the uniform, underspecified, and overspecified forecast variants remain within 1–3% of full DeLLMa, while still substantially beating baselines. This raises the question of whether accurate state forecasting is doing meaningful work for these models, or whether the utility elicitation step largely compensates. The paper attributes this to "robustness," but an alternative interpretation—that the state forecasting step contributes little to GPT-4/Gemini performance—deserves explicit engagement.

- **State enumeration quality is not validated.** The entire pipeline's correctness depends on the latent factors $(f_1,\ldots,f_k)$ generated in §3.1 being relevant, non-redundant, and reasonably comprehensive. Yet this step receives no empirical validation. There is no measurement of factor relevance, coverage, or sensitivity to prompt phrasing. Because state space size is $\ell^k$, poor factor selection can compound combinatorially.

- **Calibration evaluation in Table 1 is underspecified.** The paper says "we manually annotate a set of ground truth values for states," but does not report how many forecast points were evaluated, who the annotators were, what constitutes a ground-truth value for a qualitative latent factor such as "climate," or whether annotations were made before or after seeing outcomes. These omissions make Table 1 difficult to interpret as reliable calibration evidence.

- **No confidence intervals or statistical significance reported.** Main results in Figures 2 and 4 report point estimates without variance. With 120 problem instances and stochastic LLM outputs, confidence intervals are warranted and standard.

- **Human evaluation does not strongly validate utility elicitation accuracy.** Table 4 shows LLM–human agreement of ~65–68%, while inter-annotator agreement is 67.0% ± 6.3%. The LLM matches human noise levels but does not demonstrably exceed them. This is encouraging as a lower bound but should not be presented as strong validation; it mainly confirms that the task is inherently ambiguous.

### Tiny

- **The verbalized-to-numeric probability mapping $\mathcal{V}$ is an important modeling choice** (the entire forecast posterior depends on it), but its exact values are deferred to the appendix and not discussed in the main text. The mapping should at minimum be summarized in §3.2, as readers need to know what "likely" maps to in order to evaluate the method.

- **Notation inconsistency between Eq. (1) and Eq. (3):** $U_{\mathcal{C}}(a)$ vs. $U_C(a)$.

- **Bradley–Terry scores are used as cardinal utilities**, but BT recovers a latent preference scale only up to monotone transformation, not as calibrated cardinal values. Since expected utility in Eq. (3) averages these scores across states, the cardinalness assumption matters and should be noted.

---

## Nice-to-Haves

- **Testing on a domain with multi-attribute or non-linear utilities.** Both current domains effectively reduce to "maximize a single numeric quantity." The utility elicitation module would be far more differentiated and informative in a domain involving risk aversion, competing objectives, or qualitative tradeoffs (e.g., medical triage with side-effect penalties), where a simple "expected value" strategy is demonstrably suboptimal.

- **Scalability analysis for larger action spaces.** The paper evaluates up to 7 actions. A discussion or experiment on how API cost and accuracy scale to 20–50 actions (e.g., a diversified portfolio) would help practitioners assess deployment feasibility. The authors do defer continuous/portfolio actions to future work, which is reasonable, but a cost-scaling analysis for discrete sets would be immediately useful.

- **Comparison with a richer o1 baseline.** The current o1 comparison uses zero-shot prompting. A version where o1 receives the same structured decision-theory chain (without full DeLLMa automation) would better isolate whether the gains come from the framework design versus DeLLMa's specific implementation details.

- **Per-factor calibration breakdown.** Table 1 reports aggregate ECE/NLL. Per-factor calibration plots would reveal which latent factors are systematically mis-forecasted and whether those failures drive downstream decision errors.

- **Correlation-aware forecasting experiment.** Even a brief experiment with joint prompting or a simple copula approximation for correlated factors would address the independence assumption more concretely and guide future framework extensions.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Critique that Bradley–Terry / utility elicitation is not novel because of RLHF/preference learning overlap.** The related work section does not need to more deeply engage with the RLHF literature to establish novelty; the contribution is the *integration* of BT-based elicitation into a decision-theoretic inference-time framework, not BT itself.

- **Criticism that "human-auditable" is not rigorously evaluated as a property.** The paper uses "human auditable" to refer to the transparency afforded by explicit intermediate artifacts (decision trees with states, weights, utilities), not a formally evaluated cognitive property. The claim is descriptive of mechanism, not a falsifiable empirical assertion. The concern is reasonable but asking for a formal HCI evaluation of auditability is out of scope for this systems paper.

- **Criticism that the o1 comparison is "unfair" because o1 is used zero-shot.** Per the review instructions, comparisons that are asymmetric in favor of the baseline (o1 is a more powerful model; using it zero-shot still arguably favors it given its built-in reasoning capabilities) should be discounted as a weakness. The comparison is informative and intentional.

- **Criticism that baselines underperforming random guessing undermines the evaluation.** This is actually an interesting finding that the paper discusses as a failure mode of prompting-based methods, not an evaluation flaw.

- **Critique that Figure 3's "linear performance trends" is too strong a claim for a "scaling law."** The paper uses "linear" to describe a qualitative shape observed in the figure, not to assert a formal scaling law. This is a language nitpick.

- **Critique of limited limitations section.** While a more explicit limitations section would improve the paper, this is a formatting/organization concern rather than a scientific flaw, especially given that the paper explicitly acknowledges the independence assumption, defers sequential and portfolio settings to future work, and scopes its contribution to single-step discrete-action problems.

---

## Novel Insights

The most genuinely novel insight surfaced across the three reviews—and not fully developed in the paper itself—is the **internal contradiction between the decision-theoretic framework and the evaluation protocol**. The paper frames DeLLMa as an expected-utility maximizer, yet measures performance against ex-post optimal outcomes. This is not merely a metric choice: it implies that the benchmarks are actually testing *forecast accuracy over a very narrow, retrospective outcome distribution*, not *decision quality under uncertainty* as claimed. A deeper insight follows: if both domains have ground-truth utility functions (price×yield, stock return), it becomes possible to construct a principled evaluation using historical empirical outcome distributions as the true posterior. That would allow direct measurement of how well DeLLMa approximates Bayes-optimal decisions, rather than how often it is lucky enough to pick the realized winner—and would make the decision-theoretic framing internally consistent. None of the reviews develop this constructively, but it points toward a more rigorous version of the benchmark.

---

## Suggestions

1. **Add a "forecast-and-pick" baseline** that computes expected value from the state forecast distribution without any Bradley–Terry ranking. This is straightforward to implement and is the minimal experiment needed to isolate the utility elicitation module's contribution.

2. **Reframe or relocate the evaluation metric discussion.** In the main text, explicitly acknowledge that "accuracy" measures whether the predicted decision matches the realized optimal, and discuss in what sense this is a proxy for (rather than a direct measure of) decision quality under uncertainty. Move normalized utility from the appendix to a co-equal main result.

3. **Report confidence intervals** (e.g., bootstrap 95% CIs) for all main accuracy figures, given the 120-instance evaluation size and stochastic outputs.

4. **Specify Table 1 calibration details in the main text**: number of annotated forecast points, annotation protocol, annotator identities (or that they are the authors), and how ground-truth values for qualitative factors were determined.

5. **Add a brief analysis of the independence assumption's impact**, e.g., by comparing forecast distributions from factorized sampling against joint prompting for a small number of instances, to quantify how much the approximation distorts the state distribution.

6. **State the verbalized-to-numeric mapping** $\mathcal{V}$ explicitly in the main text (even as a small table) for reproducibility, since it directly determines the shape of the forecast posterior.

---

## Evaluation on Key Axes

- **Originality:** Moderately high. Applying classical expected-utility maximization as an inference-time scaffold for LLMs is a genuinely novel framing, distinct from all prior reasoning methods. The specific combination of verbalized forecasting + Bradley–Terry utility elicitation + Monte Carlo EU maximization is original. However, each individual component is borrowed from existing work, and the novelty is in assembly rather than invention.

- **Importance of research question:** High. Decision making under uncertainty is a critical use case for LLMs, and the paper addresses a genuine gap: current inference-time methods are designed for deterministic reasoning, not EU maximization under a user-aligned utility.

- **Whether claims are well-supported:** Moderate. The main accuracy claims are supported for the two evaluated domains and multiple LLM backbones. However, the core theoretical claim—that DeLLMa maximizes expected utility—rests on an approximation chain (factorized independence, verbalized probabilities, BT cardinality) whose combined effect is not analyzed. The evaluation conflating decision quality with realized outcomes weakens the claim that the framework achieves rational decision making.

- **Soundness of experiments:** Moderate. The benchmarks are author-constructed and small (120 instances each, retrospective, narrow domain). Baselines are appropriately chosen for the comparison claims, but the missing forecast-only baseline is a significant gap. Statistical uncertainty is not reported. The stock experiment is particularly fragile given its single-month evaluation window and small, known-ticker action set.

- **Clarity of writing:** Good. The paper is well-organized, the four-step structure is easy to follow, and the algorithms are readable. Some overstatement in framing ("high-stakes," "human-auditable") slightly oversells the empirical grounding, but the technical exposition is clear.

- **Value to the research community:** Moderate-to-high. The framework provides a reusable scaffold for practitioners deploying LLMs in decision-support contexts, and the inference-time compute scaling results are directly actionable. The publicly available code and decision-tree visualization further increase practical value.

- **Contextualization relative to prior work:** Adequate but could be stronger. The paper appropriately situates itself relative to CoT/ToT/self-consistency and forecasting literature. The connection to utility elicitation and preference learning is acknowledged but not deeply developed. The paper is appropriately modest about not comparing to all possible agentic or tool-using pipelines.

---

## EW6bNEqalF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (5.6/10)
- Match: YES

### Final Review

## Summary

This paper proposes two improvements to offline RL in Regular Decision Processes (RDPs): (1) a novel language metric $L_X$ grounded in formal language theory that replaces $L_\infty^p$-based distinguishability with a structured language-family test, and (2) a Count-Min-Sketch (CMS) approach to reduce memory requirements. The language metric is shown (Theorem 1) to achieve $\Omega(1)$ distinguishability in the T-maze family where $L_\infty$-based methods suffer $\mathcal{O}(2^{-N})$ distinguishability, and PAC sample complexity bounds are proven for both variants. The authors also identify and correct a mistake in prior work (Cipollone et al., 2023).

---

## Strengths

- **Exponential separation result (Theorem 1) is clean, concrete, and compelling.** The paper constructs an explicit family of RDPs where $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability remains $\Omega(1)$. This directly justifies the paper's motivating question and is not a generic argument—it pinpoints precisely which structural property makes the language family work (the probability of seeing a reward under action *North*).

- **T-maze scaling experiment (Figure 2) is the strongest piece of empirical evidence and directly matches the theory.** It shows linear vs. exponential scaling of time and RDP state count for language metric vs. CMS as corridor length grows, which precisely mirrors the Theorem 1 separation. The experiment runs to corridor length 100, providing ample quantitative evidence.

- **The two-dimensional hierarchy $\mathcal{X}_{i,j}$ is a creative bridge between formal language theory and RL distinguishability.** The $C_k^\ell$ operator combined with atomic symbol families $\mathcal{G}_1, \mathcal{G}_2, \mathcal{G}_3$ yields a principled and structured function class for an IPM-style distance, rather than an ad hoc construction. The unifying perspective in Definition 2 that recovers $L_\infty$, TV, and prefix metrics as special cases is elegant.

- **Correction of an error in Cipollone et al. (2023) demonstrates thoroughness.** The identification of an additional $\sqrt{H}/\mu_0$ multiplicative factor in the sample complexity is a substantive contribution to the theoretical foundations of the field.

- **Practical results in Table 1 are convincing on hard domains.** On T-maze(c), Cheese, and Mini-hall—domains requiring long-term memory—the language metric approach achieves substantially better policies than both FlexFringe and CMS, with smaller automata and faster runtime. FlexFringe fails entirely on T-maze(c) (reward 0.0 vs. 4.0).

---

## Weaknesses

### Fatal
None.

### Major

- **End-to-end offline RL evaluation is absent.** The paper's stated contribution is offline RL, yet all experiments evaluate only automaton learning quality (number of states, runtime) and the reward of the *derived* policy—without ever running the full offline RL pipeline (ADACT-H → RegORL planning → policy evaluation against a proper offline RL baseline). There is no comparison against RegORL with the original $L_\infty^p$ statistical test, which is the direct theoretical predecessor the paper claims to improve. Without this, the empirical offline-RL claim rests entirely on Table 1 reward numbers, which cannot be attributed to sample efficiency improvements.

- **No experiments validating the sample efficiency (PAC) claim.** The central theoretical contribution is improved sample complexity. Yet every experiment uses a fixed $K = 100$ episodes dataset. There are no success-probability-vs-dataset-size curves, no ablations over dataset size, and no empirical demonstration that the language metric requires fewer samples to recover the correct automaton or achieve a target reward. The sample efficiency improvement is entirely theoretical and empirically unsubstantiated.

- **Model-selection problem for $(i, j)$ is unresolved and could undermine applicability.** Assumption 1 requires the behavior policy to guarantee $L_{\mathcal{X}_{i,j}}$-distinguishability $\mu_0 > 0$ for the *chosen* language family. However, the practitioner does not know the hidden RDP and therefore cannot verify which $(i,j)$ satisfies this for a given dataset. The experiments always use $\mathcal{X}_{3,1}$, but no justification—theoretical or empirical—is provided for why this choice is safe across domains. The paper provides no characterization of when small $(i,j)$ fails to distinguish states, no adaptive search procedure, and no failure-mode analysis. This is not a minor reproducibility concern; if the chosen family is too coarse, distinct RDP states will be incorrectly merged and the resulting policy may be arbitrarily suboptimal.

- **No ablation over the hierarchy parameters $(i, j)$.** Since the central technical contribution is the hierarchy $\mathcal{X}_{i,j}$, the paper should empirically demonstrate the tradeoff between expressiveness and cost by varying both $i \in \{1,2,3\}$ and $j$. As it stands, only $\mathcal{X}_{3,1}$ is tested, making it impossible to assess whether the hierarchy design is necessary or whether $\mathcal{X}_{1,1}$ would suffice, or when larger $j$ matters.

### Minor

- **The $d_m^*$ dependence can be exponential in $H$, potentially negating the sample complexity gains.** Theorem 3's bound is $\tilde{\mathcal{O}}(C_\mathbf{R}^* \log(1/\delta) \log|\mathcal{X}| / (d_m^* \mu_0^2))$. The paper acknowledges that $1/d_m^*$ can be exponential in $H$ if some optimal-policy state is very hard to reach. This means the claimed exponential improvement via $\mu_0$ can be offset by an exponential penalty in $1/d_m^*$. The paper does not analyze whether these terms can simultaneously be favorable, which leaves the claimed exponential benefit ambiguous for general structured RDPs beyond T-maze.

- **The correction to Cipollone et al. (2023) is stated but not made self-contained.** The paper says "both their and our sample complexity has an additional multiplicative term $\sqrt{H}/\mu_0$," but this factor is not visibly present in the theorem statements shown in the main text (Theorems 2 and 3 show $\mu_0^2$ in the denominator, not $\mu_0^3$ or an explicit $\sqrt{H}$). The $\tilde{\mathcal{O}}$ notation hides only poly-logarithmic terms per the notation section, so a polynomial $\sqrt{H}$ factor should appear explicitly. This inconsistency needs clarification.

- **The offline RL objective is stated in two inconsistent forms.** Section 2 defines $\varepsilon$-optimality in expectation over $h_0$, but Section 2.3 states the goal as $V_\circ^*(h) - V_{\hat\pi}^*(h) \leq \varepsilon$ for each $h \in \mathcal{H}_0$. The notation $V_{\hat\pi}^*(h)$ (value of $\hat\pi$ with a star superscript) is also potentially misleading; presumably this is $V^{\hat\pi}$.

- **Theorem 2 does not specify the CMS parameters in the theorem statement itself.** The CMS approximation quality depends on $\delta_c$ and $\varepsilon$, but these are not enumerated in the theorem body, making the guarantee incomplete as stated in the main text.

- **No memory measurements despite CMS being explicitly motivated by memory reduction.** The paper introduces CMS to address memory requirements, but Table 1 reports only runtime and state counts. Peak memory usage is not reported for any algorithm, making it impossible to verify the memory efficiency claim empirically.

- **Typo in the two-dimensional hierarchy description.** Section 4.1 reads: "It is parameterised by $j$ for the granularity of the atomic symbols, and by $j$ for the sequential composition." Both dimensions are labeled $j$; one should clearly be $i$.

### Tiny

- The conclusion says the language approach "remov[es] the dependency on $L_\infty^p$-distinguishability parameters," which is slightly inaccurate—it replaces that dependency with $L_X$-distinguishability, which may be much more favorable but is still a distinguishability requirement.

---

## Nice-to-Haves

- **Discuss practical strategy for selecting $(i,j)$.** A heuristic such as iterating from $(1,1)$ upward until state counts stabilize, or using a held-out validation split, would greatly improve practical utility.
- **Robustness analysis when Assumption 1 is violated.** An empirical demonstration of graceful degradation (e.g., reverting toward $L_\infty$ behavior or harmless state over-splitting) would reassure practitioners.
- **Include comparison against neural offline RL methods (e.g., CQL/IQL with RNN history embeddings).** The paper's formal guarantee setting means this comparison is not required, but it would help contextualize practical performance relative to the broader ICLR community's typical baselines.
- **Provide a more general theorem characterizing when small-$(j)$ families suffice.** Theorem 1 is an existence result; a structural characterization (e.g., "whenever the distinguishing event depends only on $j$ elementary patterns in the trace") would strengthen the contribution.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[Removed — scope creep]** Harsh Critic's suggestion that the paper should compare against "sequence-model-based offline RL / latent-state learning." The paper explicitly targets provably correct algorithms, and local optimization methods are acknowledged to lack formal guarantees. Evaluating this paper on whether it competes with deep learning approaches without guarantees is scope creep.
- **[Removed — scope creep]** Requests for comparison against Transformer/RNN-based policies from Review 2. Same reasoning as above; the paper's contribution is formal and theoretical, not a systems benchmark.
- **[Removed — methodology norms]** Requests for confidence intervals on all Table 1 quantities. Standard deviations are provided where results vary across runs; exact values are given where the outcome is deterministic. This is acceptable practice.
- **[Removed — cannot verify]** All criticisms about missing related works, as external reference availability cannot be confirmed.
- **[Removed — overly picky]** Harsh Critic's claim that the paper "relies too heavily on appendices for pseudocode." The pseudocode of ADACT-H is a standard prior algorithm; it is reasonable to place it in the appendix with a clear pointer. The novel parts of the algorithm (statistical test modification) are described in the main text.

---

## Novel Insights

The most genuinely novel conceptual insight in this paper—one not fully appreciated by the sub-reviewers—is that the exponential blowup in $L_\infty^p$-based RDP learning is not a fundamental barrier but an artifact of choosing too fine-grained a function class for the distributional comparison. By framing state distinguishability as an integral probability metric (IPM) over a structured language class, the paper shows that the right level of granularity—coarser than individual suffix strings but finer than total variation—can be both (a) computationally tractable via simple membership counting and (b) statistically powerful via $\Omega(1)$ separation in structured domains. This reframing connects automata learning with IPM theory in a way that may be useful beyond RDPs, e.g., in any setting where state-merging algorithms must compare distributions over structured sequential data.

---

## Evaluation Summary

| Axis | Assessment |
|---|---|
| **Originality** | High — the language metric hierarchy and its connection to the dot-depth hierarchy is a novel and principled idea, not a routine extension |
| **Importance of research question** | Solid — non-Markovian offline RL is genuinely important and the exponential sample complexity barrier is a real obstacle |
| **Claims well supported** | Partially — the theoretical claims are proven, and the T-maze scaling experiment is compelling; but the sample efficiency claim lacks direct empirical support |
| **Soundness of experiments** | Weak — experiments evaluate automaton quality and policy reward, but not the core sample efficiency claim; no comparison to ADACT-H with original test |
| **Clarity of writing** | Adequate — the narrative arc is coherent, but the notation for $L_\infty^\circ$, $L_\infty^p$, $L_1^\circ$, $L_1^p$, $L_X$ is easy to confuse, and Definition 1 / the hierarchy need more intuition |
| **Value to research community** | Moderate-to-high for the subfield; the IPM reframing and correction of prior error are useful contributions |
| **Contextualized relative to prior work** | Good — the paper clearly identifies what it improves over RegORL and positions itself correctly in the non-Markovian RL literature |

---

## Suggestions

1. **Add a direct ablation comparing ADACT-H (original $L_\infty^p$), ADACT-H+CMS, and ADACT-H+$L_X$ under identical datasets.** This is essential to empirically validate the sample efficiency claim and should be the primary baseline, not FlexFringe.
2. **Add at least one dataset-size sweep** (e.g., vary $K$ from 10 to 1000 on T-maze or Corridor) showing recovery probability vs. dataset size for each method.
3. **Provide an ablation over $(i, j)$** on at least two domains to justify the $\mathcal{X}_{3,1}$ choice and characterize when larger $j$ or $i$ is needed.
4. **Clarify the $\sqrt{H}/\mu_0$ correction** either by making it explicit in the theorem statement or explaining why it is absorbed into the $\tilde{\mathcal{O}}$ notation.
5. **Add peak memory measurements** to Table 1 to substantiate the CMS memory efficiency claim.
6. **Fix the notation typo** in the hierarchy parameterization description (both dimensions labeled $j$).
7. **Discuss Assumption 1 robustness** — what happens empirically when $\mathcal{X}_{i,j}$ is too coarse? Does ADACT-H over-merge states or under-merge?

MY FINAL SCORE: <pineapple>5.8</pineapple>

---

## SI6zocV2SS

- GT: Reject (avg 1.5)
- Predicted: Reject (2.2/10)
- Match: YES

### Final Review

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

---

## HozsY9Gdcl

- GT: Reject (avg 5.0)
- Predicted: Accept (5.9/10)
- Match: NO

### Final Review

## Summary
This paper introduces **Set-MI**, a method that improves membership inference (MI) for language models by exploiting a *set assumption*: documents sharing metadata (e.g., publication date, license, language) are either collectively present in or collectively absent from a model's training data. Set-MI aggregates individual document-level MI scores within each set and broadcasts the aggregated score back to each member. The authors construct five diverse benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrate an average AUROC gain of 0.14 over four Individual-MI baselines. Additional analyses study the effect of model size, deduplication, document length, set size, and noise robustness.

---

## Strengths

- **Operationalizing a principled insight about pretraining data curation.** The observation that LM pretraining corpora are curated by *inclusion criteria* (date cutoffs, license categories, dataset sources) rather than per-document decisions is well-motivated and directly reflected in the set assumption. This is not a trivially recycled idea—it is a specific and accurate structural observation about how large pretraining datasets are actually assembled (e.g., DOLMA's Reddit cutoff), and it is used to design a concrete method.

- **Diverse benchmark suite filling a real gap.** The five benchmarks span temporal, linguistic, licensing, and instruction-tuning dimensions. Constructing MI benchmarks for LMs with known ground truth is non-trivial, and a multi-domain suite with varied notions of "set" is a concrete contribution the community can build on, provided the statistics inconsistencies noted below are corrected.

- **Meaningful robustness analysis.** Section 6's controlled noise injection study—comparing FULL/MAX/MIN aggregation under member-set noise, non-member-set noise, and both—provides genuine practical guidance. The finding that all three aggregators substantially outperform Individual-MI even under high noise ratios (up to 50% flipped labels) is informative, and the qualitative recommendation (MAX when member sets are noisy, MIN when non-member sets are noisy) is actionable.

- **Scaling and deduplication findings.** The systematic study linking larger model size to larger Set-MI gains (and deduplication to reduced gains) connects to prior memorization literature and adds new set-level evidence that directly informs practical deployment choices.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Benchmark statistics inconsistency between Table 1 and the main text.** Table 1 reports Wikipedia: 1,000 sets / 100,000 docs and Arxiv: 1,000 sets / 100,000 docs. However, Section 4 explicitly states "we subsample 100 sets with 100 documents per set," yielding 10,000 docs—a 10× discrepancy. For License, Table 1 reports 190 sets / 19,000 docs, but Section 4 text says "resulting in 130 sets with 100 documents per set" (13,000 docs). These are not typographic issues: the actual experimental scale determines statistical reliability and reproducibility. If experiments truly used 100 sets, the main claims rest on much smaller samples than implied by Table 1. This must be clarified and corrected.

- **Document-level AUROC is not the right primary metric for a set-based method.** Set-MI assigns an identical score to every document in a set, then reports AUROC at the document level. Because all documents within a set share the same predicted score and the same ground-truth label, they are statistically dependent—effectively contributing the same "vote" |s| times rather than once. This inflates the effective sample size and can make document-level AUROC appear stronger than the method's actual discriminative power over distinct membership decisions. A set-level AUROC (one observation per set) should be the primary reported metric, with document-level AUROC as a secondary figure for compatibility with prior work.

- **Language and Instructions benchmarks may reflect domain separability rather than membership inference.** LiRA + Set-MI achieves **1.000 AUROC** on Languages, and Min-K% Prob achieves **1.000** on Instructions even at the Individual-MI level. These near-perfect scores raise the concern that the model is exploiting distributional differences between languages BLOOM was or was not trained on (e.g., perplexity differences due to absent language coverage) rather than subtle membership signals in the MI sense. Similarly, instruction-tuning datasets may have stylistic markers that separate them from non-member corpora without true memorization. The paper should analyze these benchmarks' difficulty more carefully and clarify whether the 0.14 average improvement is driven by genuinely hard domains or partly by inflated performance on trivially separable ones.

- **Missing pseudo-set control.** The core claim is that the *set assumption*—shared membership—is what drives improvement, not merely variance reduction from averaging more noisy signals. Without a control experiment where documents are randomly assigned to pseudo-sets (destroying the set assumption while keeping set size and averaging constant), it is impossible to separate these two explanations. This is a fundamental validation gap for the paper's stated contribution.

### Minor

- **The negative result (zlib on Instructions: 0.458 → 0.429) is noted but not analyzed.** This is the only case where Set-MI hurts performance, and understanding it is important for practitioners. The paper attributes it qualitatively to poor base signal, but does not investigate whether it is the set assumption being violated, averaging of systematic biases, or something specific to zlib's behavior on instruction-formatted text.

- **Robustness study uses only Loss Attack and one domain (Wikipedia).** Section 6's noise experiment is informative but uses a single base method. Since the main paper emphasizes LiRA and Min-K% Prob as the strongest base methods, showing robustness properties for at least one of those would substantially strengthen the conclusions.

- **Ground-truth labeling method is inconsistent across the paper.** Section 5 uses date cutoffs relative to Pile collection dates as ground truth for Wikipedia and Arxiv. Section 6 explicitly uses 13-gram overlap with the Pile as a "correct" ground truth. It is unclear whether main experiments (Table 2) use the date heuristic or n-gram verification, and to what degree date-based labels introduce false positives/negatives (e.g., documents that existed before the cutoff but were not actually ingested due to filtering).

- **The failure mode where Set-MI underperforms Individual-MI when base methods are below chance** is identified (Section 5.2) but no mitigation is proposed. A practical diagnostic or guard (e.g., check Individual-MI AUROC on a small calibration set before applying Set-MI) would make the method more deployable.

### Tiny

- The zlib formulation uses the ratio of *LM loss* to *zlib entropy*, which is standard, but the notation is slightly inconsistent between the averaging notation for Loss Attack (token probabilities) and the zlib formula. A clarifying note distinguishing log-probabilities from raw probabilities would aid reproducibility.

- Figure 4 (Left) shows performance saturating around 256 tokens yet 1,024 is used in main experiments. A brief justification for this choice would be useful.

---

## Nice-to-Haves

- **Evaluation on a model with a verifiable but unpublished training cutoff** (e.g., using Llama 2's known cutoff date for Arxiv) would demonstrate Set-MI's utility in a setting closer to real auditing use cases, where the training set is not fully public.
- **Set-level calibration or precision-recall analysis** in addition to AUROC would help characterize false accusation rates in applications like contamination detection.
- **Testing non-trivial aggregators** (weighted average by document confidence, robust estimators like trimmed mean) as an optional extension to complement the MAX/MIN/FULL comparison in Section 6.
- **Pseudo-metadata clustering** (e.g., by topic or style) as a strategy for scenarios where explicit metadata is unavailable—could substantially broaden applicability.
- **Expanded deduplication analysis**: the current study uses only Loss Attack on Wikipedia. Extending to LiRA or Min-K% Prob on Arxiv would better characterize when deduplication breaks Set-MI's advantage.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Critic: "Loss Attack averages raw probabilities, which is unusual."** LM(t_i) in this literature is standardly interpreted as log-probability (negative loss), and averaging log-probabilities to form a document score is the conventional Loss Attack formulation. The notation is informal shorthand, not an error.

- **Critic: "LiRA formula is not the standard formulation."** The paper presents a simplified notation for the ratio-based score. The actual LiRA implementation is deferred to Appendix A, which is a reasonable choice for a methods overview; the formula presented is a correct conceptual summary.

- **Critic: "Disjoint-set assumption is restrictive; should allow multiple overlapping grouping axes."** Requiring the method to also handle non-disjoint multi-axis groupings is outside the paper's stated scope. The method is presented as applicable to any partition satisfying the assumption; extending to joint multiple-axis inference is future work, not a flaw.

- **Critic: "Comparison with Set-MI is unfair because it uses extra side information."** The paper is explicit that Set-MI exploits available metadata as an additional input. This is not hidden; comparing against Individual-MI (which does not use metadata) is intentional and makes the contribution clear: metadata enables this gain. The comparison favors the baseline (which uses less information) and thus proves a stronger point for the proposed method.

- **Critic: "The abstract overstates practical reliability."** The conclusion language ("brings up the limit of MI to a practically robust level") is mildly optimistic but not egregiously misleading given the results. The robustness analysis in Section 6 does provide noise-tolerance evidence, and the scope is appropriately constrained to settings where metadata is available.

- **Critic: Requests hypothesis tests and confidence intervals for all results.** Single-run AUROC evaluation is the prevailing norm in LM MI literature. Requiring confidence intervals across all 20+ table cells would be non-standard for this subfield, though the point about variance across random token segment choices is worth a brief mention.

- **Critic: "Ethical discussion is too brief."** The dual-use discussion is concise but covers the key points appropriate for a research paper. Expanding it to a full treatment is a style preference.

---

## Novel Insights

The most genuinely novel observation—surfaced primarily by the spark finder and partially by the harsh critic—is that the paper does not clearly separate two distinct sources of improvement: **(1) the semantic coherence of set-assumption-satisfying groups** (shared metadata implies shared membership, which implies correlated loss signals) versus **(2) generic variance reduction from averaging any set of documents**. The pseudo-set control experiment (randomly assigned groups of equivalent size) is conspicuously absent. If simple averaging over random groups produces similar AUROC gains, the contribution would reduce to "more data per decision point helps," which is less interesting than the set assumption. If the gains are substantially larger with metadata-defined sets than with random pseudo-sets, this would be strong evidence for the set assumption's specific value. The paper as written cannot distinguish between these two explanations, and this gap is the most important open question the work raises.

---

## Evaluation by Axis

**Originality:** Moderate-to-good. The set assumption idea is intuitive and has precedent in clinical MI work (Jagannatha et al.), but its application to web-scale LM pretraining data with natural metadata structures is a meaningful and specific contribution. The benchmark construction adds additional originality.

**Importance of research question:** High. Data transparency, copyright auditing, and contamination detection are pressing concerns for the ML community, and improving MI performance from near-random to substantially above chance is practically relevant.

**Claim support:** Mixed. The main empirical gains are real and well-replicated across models and domains. However, the inflated AUROC on trivially easy benchmarks (Languages, Instructions), the missing pseudo-set control, and the document-vs-set metric issue mean the evidence is less clean than it appears.

**Soundness of experiments:** Moderate. The experimental design is broad and includes good ablations. The benchmark statistics inconsistency and absent set-level metric are real methodological issues that need resolution before the empirical claims can be fully trusted.

**Clarity of writing:** Good. The paper is readable and the central idea is easy to follow. The main clarity issues are in precise notation and benchmark construction reproducibility details.

**Value to the research community:** Moderate-to-good. The benchmark suite, if properly documented and with statistics clarified, would be a useful resource. The main method, though simple, provides actionable guidance.

**Contextualization relative to prior work:** Adequate. The paper covers the main Individual-MI baselines and correctly situates its contribution as orthogonal and additive.

---

## Suggestions

1. **Fix benchmark statistics inconsistency**: Reconcile Table 1 with the text descriptions for Wikipedia, Arxiv, and License. If the experiments used 100 sets × 100 docs (10,000 total for Wikipedia/Arxiv), update Table 1 accordingly. If they used 1,000 sets × 100 docs, update the text.

2. **Report set-level AUROC as primary metric**: Compute AUROC where each set contributes one observation (its aggregated score vs. its set-level ground truth), and compare this to document-level AUROC to demonstrate the difference. Reframe Table 2 around set-level AUROC.

3. **Add pseudo-set control**: Run Set-MI with randomly constituted sets of the same sizes as the metadata-defined sets. Report AUROC relative to the metadata-defined Sets to show the set assumption's marginal contribution.

4. **Analyze Language and Instructions benchmarks for confounding**: Report per-language AUROC for BLOOM to check whether gains track with language coverage in BLOOM's training, and whether held-out languages have systematically higher per-token loss for reasons unrelated to MI. Acknowledge if these benchmarks primarily test "did the model see this language at all" rather than nuanced membership inference.

5. **Clarify ground truth construction**: Explicitly state in Section 5 for each benchmark whether ground truth uses date/metadata heuristics alone or is verified via n-gram overlap, and estimate label noise where applicable.

6. **Analyze the zlib/Instructions failure case**: Check whether the set assumption is violated for those instruction datasets (e.g., partial inclusion from Tulu mixes), or whether zlib scores are anti-correlated with membership for instruction-formatted text, and briefly discuss in the paper.

MY FINAL SCORE: <pineapple>5.8</pineapple>

---

## slZZnzlITo

- GT: Reject (avg 4.8)
- Predicted: Accept (5.2/10)
- Match: NO

### Final Review

## Summary

This paper proposes Multimodal Open Set Recognition (MMOSR), a new task extending OSR to multimodal settings. The authors identify "fusion degradation" — a phenomenon where OSR regularization over-compresses fused representations and suppresses modality-specific information — and propose the Multimodal Representation Reactivation Network (MRN), which uses bidirectional cross-attention (mutually enhanced fusion) and a Mixture-of-Experts head (adaptive fusion) to recover richer representations. Experiments on four datasets spanning image-text, audio-visual, and RGB-depth settings show that MRN generally outperforms existing multimodal fusion methods, with gains up to +5.23 OSCR on Flower-102.

---

## Strengths

- **First systematic study of MMOSR across genuinely diverse modality pairings.** The experimental scope covers image-text (Food-101, Flower-102), audio-visual (CREMA-D), and RGB-depth (SUN RGB-D) — a breadth of modality types that most multimodal papers in this space do not achieve, and which meaningfully supports the claim that the challenge is not dataset-specific.

- **MRN as a fusion backbone consistently improves OSR methods across all datasets.** Table 2 shows that ARPL-MRN and CSRR-MRN outperform ARPL/CSRR with every other fusion strategy (ADD, CAT, GQA) on all four datasets, including CREMA-D where standalone MRN does not win. This asymmetry — the backbone always helps even when the standalone method does not dominate — provides a more robust signal of architectural value than any single headline number.

- **Fusion degradation identification is empirically grounded.** Table 1 shows a clear pattern: combining additive fusion with OSR regularization (Fusion-OSR) causes AUROC to drop below either single-modal OSR or plain fusion across 5/10/20-class splits on Food-101. This observation is genuine and non-obvious, even if the mechanism is not formally characterized.

- **Competitive with large pretrained models without pretraining.** Table 3 demonstrates that scratch-trained MRN outperforms zero-shot CLIP and 16-shot CoOp/MaPLe across all class-ratio settings on Food-101, despite having no access to the large-scale pretraining used by those models. Since any unfairness in this comparison favors the pretrained baselines, the MRN result is a stronger empirical claim by construction.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing the most natural baseline: per-modality OSR score ensemble.** The most straightforward approach to multimodal OSR is to run OSR independently on each modality and combine the rejection scores (e.g., by averaging MSP or taking the maximum). Without this baseline, it is impossible to determine whether MRN's gains arise from fusion architecture improvements or simply from aggregating two OSR signals. This is a critical omission for a paper whose central claim is that multimodal-specific fusion design is necessary.

- **No OSR-specific training objective.** Section 4.2.3 shows that MRN is trained with only a standard classification loss and a load-balancing loss. The fusion degradation narrative is about OSR regularization harming representations — yet MRN itself does not apply any OSR regularization at training time. Unknown rejection is then performed via standard MSP thresholding. The method is therefore a better multimodal classifier evaluated with post-hoc OOD scoring, not an OSR method in the usual sense. This mismatch between the problem framing and the actual training objective should be explicitly acknowledged. It is not fatal, but it significantly narrows the methodological contribution relative to what is claimed.

- **Inconsistent performance undermines "consistent superiority" claim.** On CREMA-D (Table 2), standalone MRN achieves 66.78 AUROC and 57.32 OSCR, which is worse than MLA's 67.83 AUROC and 57.50 OSCR. The paper's text in Section 5.2 states MRN "consistently demonstrates exceptional MMOSR performance," which is inaccurate. For a paper proposing a method for MMOSR broadly, this inconsistency requires either a more honest characterization or an analysis of why CREMA-D's audio-visual structure is a distinct failure mode.

- **No variance estimates for marginal gains.** Open-set class splits are randomly sampled. Several reported improvements are very small (SUN RGB-D: +0.37 AUROC, +0.01 OSCR; Food-101: +0.72 AUROC, +1.38 OSCR). Without standard deviations across multiple random splits, there is no basis to distinguish these from split-to-split variance. This is particularly important for ICLR where marginal gains without significance are insufficient support for broad empirical claims.

- **Ablation does not isolate the MoE module.** Table 4 ablates only the cross-attention directions (C1 and C2), with the first row defined as "encoders + adaptive fusion (MoE)." There is no comparison of MoE vs. a single MLP head, or MoE vs. simple concatenation + linear, which would be necessary to evaluate whether the MoE component provides benefit beyond a larger/deeper prediction head. Since adaptive fusion is presented as a co-equal contribution, this omission is significant.

### Minor

- **Threshold selection protocol underspecified.** Section 4.3 states the threshold τ is set to "ensure 95% of the known samples are correctly classified" but does not specify on which data split. If this uses test-known samples, it would contaminate evaluation. The paper should explicitly state that the threshold is tuned on a held-out validation set.

- **Fusion degradation diagnosis is restricted to one dataset and one baseline.** The necessity analysis in Section 3.2 is based entirely on Food-101 with a simple additive Fusion-OSR. While Section 5 broadens the comparison, the motivating failure demonstration would be substantially stronger if it included the stronger multimodal baselines (TMC, MLA) and additional datasets, especially given that the central argument relies on this failure mode being general.

- **No computational cost reporting.** MRN introduces 15 experts with top-4 gating plus bidirectional cross-attention on top of encoders. Parameter counts, FLOPs, or inference time relative to simpler baselines are not reported, making practical trade-off assessment impossible.

- **Cross-attention equation (Eq. 1) is notationaly unclear.** As written, `Softmax(W_Q z1 · z2 W_K / sqrt(d)) (W_V z2)` mixes matrix-vector conventions ambiguously. The shapes of z1 and z2 (pooled vectors vs. token sequences) are never stated, making the exact computation difficult to reproduce for different modality structures (e.g., text token sequences vs. pooled audio embeddings).

### Tiny

- **t-SNE and Grad-CAM visualizations are suggestive but not rigorous.** Figures 6 and 7 are illustrative, but t-SNE can create apparent cluster separation that does not reflect actual decision boundaries, and Grad-CAM examples may be cherry-picked. Quantitative proxies (e.g., intra/inter-class distance ratios, feature effective rank) would support the "reactivation" narrative more convincingly.

- **The ablation asymmetry in cross-attention direction** (C2 > C1 in Table 4) is noted but the explanation that "images serve as queries, leveraging richer visual information" is speculative and not tested. It would be more informative to check whether this direction asymmetry holds across modalities where text vs. image dominance differs.

---

## Nice-to-Haves

- **Pretrained encoders as MRN backbone.** Training from scratch with ResNet34 is a standard setup for controlled comparison, but evaluating MRN with CLIP's pretrained encoders (or similar) would demonstrate practical relevance at modern capability levels and directly answer whether the architectural contributions hold in the pretrained regime.

- **Quantitative measure of fusion degradation.** Adding metrics such as representation effective rank, class separation ratio, or cosine similarity between known clusters before/after OSR training would make the fusion degradation claim rigorous rather than relying on t-SNE plots.

- **Expert specialization visualization.** Showing which experts activate for which modality combinations or class types (e.g., per-expert routing frequency by modality) would validate whether MoE actually captures diverse, complementary representations or simply functions as a wider MLP.

- **Missing-modality robustness evaluation.** Since the practical motivation includes sensor-failure scenarios (robots, unmanned systems), even a brief analysis of MRN behavior when one modality is zeroed out or replaced with noise would inform practical deployment considerations.

- **Extension to more than two modalities.** The paper notes pairwise cross-attention can be extended but does not demonstrate this. An experiment with three modalities (or even a discussion of the quadratic scaling trade-off) would substantiate the generality claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Missing related works (multimodal OOD detection, selective prediction, open vocabulary multimodal recognition).** Per review policy, criticisms about missing citations are removed since we cannot confirm existence from external sources. The related work section covers the directly relevant OSR and multimodal fusion literature adequately.

- **GQA is not a canonical multimodal fusion baseline.** GQA (Ainslie et al., 2023) is described in the paper as an approach for multimodal learning efficiency and is included as a fusion baseline used in their setting. Since the paper cites and uses it, criticism of its relevance as a citation is not warranted.

- **Training all models from scratch unfairly disadvantages strong multimodal baselines.** This comparison is intentionally asymmetric in favor of pretrained baselines (CLIP, CoOp, MaPLe), not against them. Per review rules, comparisons where unfairness benefits the baseline are not legitimate weaknesses — they actually strengthen the claim when MRN still wins. REMOVED.

- **Lack of formal theoretical justification (theorems, proofs) for fusion degradation.** This is an empirical systems paper at ICLR; demanding theoretical proof of an empirically observed phenomenon imposes non-standard rigor requirements for this setting. REMOVED.

- **Requesting confidence intervals for all large-scale benchmark results.** Single-run reporting is the norm for large-scale benchmarks. The variance concern is legitimately kept only for the marginal-gain cases in Table 2. The broad demand for confidence intervals everywhere is REMOVED as a blanket expectation.

- **Style criticisms** (abstract phrasing strength, conclusion tone): REMOVED as pure framing/style issues.

---

## Novel Insights

The most substantive novel observation across the three reviews — one not directly made explicit in the paper itself — is the asymmetric evidence structure in Tables 2–3: standalone MRN is not universally best (it loses on CREMA-D), but MRN-as-backbone (ARPL-MRN, CSRR-MRN) consistently and universally improves over alternative fusion strategies including on CREMA-D. This suggests the paper's true contribution may be more precisely characterized as a robust multimodal feature extractor that reduces sensitivity to OSR regularization, rather than a complete MMOSR system. This reframing actually makes the contribution cleaner: the architecture reactivates suppressed representations regardless of the downstream OSR objective, which is a more defensible and reproducible claim than "MRN is the best MMOSR method."

---

## Suggestions

1. **Add a per-modality OSR score ensemble baseline** (e.g., average MSP from independent unimodal OSR models) as the simplest multimodal OSR competitor. This single baseline would either validate or substantially weaken the necessity of fusion-level design.

2. **Restate the main claim more precisely:** distinguish between (a) MRN as a standalone method and (b) MRN as a fusion backbone for existing OSR objectives. The fusion-backbone framing is more consistently supported and would align better with the asymmetric evidence in Table 2.

3. **Add variance across random class splits.** Run each configuration across at least 5 random known/unknown splits and report mean ± std. Focus this effort on the datasets where gains are smallest (SUN RGB-D, CREMA-D) to determine whether the results are statistically meaningful.

4. **Specify the threshold protocol.** Confirm explicitly that τ is chosen on a held-out validation set of known classes, not on the test distribution, and describe the split construction.

5. **Add one ablation row: MoE replaced by single MLP** with equivalent parameter count. This is the minimum needed to attribute any gain to the expert-diversity mechanism vs. simply more parameters.

6. **Broaden the Section 3.2 motivation** to include at least one non-food dataset and one stronger fusion baseline (MLA or GQA), to show fusion degradation is not specific to Food-101 + additive fusion.

---

## Evaluation on Key Axes

- **Originality**: Moderate. The MMOSR task formulation and fusion degradation observation are genuinely novel framing contributions. The MRN architecture composes known elements (bidirectional cross-attention, sparse MoE) without introducing new mechanisms; the novelty lies in their application to a new problem rather than the components themselves.

- **Importance of research question**: High. Multimodal OSR is a practically critical and underexplored problem. The paper is timely.

- **Claims well-supported**: Partially. The strongest claims (MRN as fusion backbone consistently helps) are well-supported. The "consistent superiority" of standalone MRN is not supported (CREMA-D). The fusion degradation mechanism is plausible and empirically motivated but not rigorously characterized.

- **Soundness of experiments**: Moderate. Four datasets with diverse modalities and many baselines are a genuine strength. Missing the per-modality ensemble baseline, absence of variance estimates, and incomplete MoE ablation are meaningful gaps.

- **Clarity of writing**: Adequate. The high-level idea is well-communicated. Technical under-specification (attention tensor shapes, threshold protocol) limits reproducibility.

- **Value to research community**: Moderate-to-good. Establishing the MMOSR benchmark and demonstrating that naive combination fails provides a useful starting point. The contribution would be considerably higher if the per-modality baseline were included and the evaluation protocol tightened.

- **Contextualized relative to prior work**: Adequate. The paper covers the directly relevant OSR and multimodal fusion literature and does not overclaim novelty relative to them.

---

## F6s7OApF0n

- GT: Reject (avg 4.8)
- Predicted: Accept (6.4/10)
- Match: NO

### Final Review

## Summary

CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) proposes a framework for HPO in which a user-defined utility function encodes the trade-off between BO cost (in epochs) and validation performance, rather than optimizing asymptotic performance alone. The method introduces (i) a utility-aware acquisition that maximizes expected utility improvement over a dynamically chosen future horizon Δt, (ii) an adaptive stopping criterion combining normalized regret with a probability-of-improvement (PI) threshold, and (iii) a PFN surrogate trained via a two-stage LC mixup strategy for transfer learning across task families. Extensive experiments on LCBench, TaskSet, PD1, and a real-world object-detection dataset demonstrate consistent improvements over multi-fidelity and transfer-BO baselines under various utility functions and cost penalties.

---

## Strengths

- **Genuinely novel acquisition formulation.** Eq. (2) extends expected improvement from performance to utility, with a joint optimisation over the future continuation length Δt per configuration. This is a non-trivial departure from prior freeze-thaw acquisitions (DyHPO, DPL, iFBO), which either use single-step greedy extensions or maximise performance at the last epoch or randomly chosen epochs. The shift from exploration to exploitation as cost dominates utility (Fig. 7b, showing Δt/T shrinking over BO steps) is an insightful and empirically verified consequence of the formulation.

- **Two-stage LC mixup for PFN transfer learning.** The proposal to apply a shared λ₁ across configurations within a dataset before mixing across configurations is a subtle but important detail: it preserves inter-configuration correlations while generating effectively unlimited synthetic training tasks. Fig. 6a directly shows that mixup reduces overfitting in the surrogate; Fig. 6b shows a downstream BO benefit. This is a principled and practical contribution beyond generic data augmentation.

- **Comprehensive and consistent empirical validation.** The paper evaluates on three standard HPO benchmarks (LCBench: 35 tabular datasets, TaskSet: 9 NLP tasks, PD1: 7 tasks with modern architectures) plus a real-world object-detection dataset with 30 tasks from heterogeneous architectures. Results are consistent across benchmarks, cost penalties, and utility function families (linear, quadratic, square-root, staircase), and ablations in Table 3 cleanly decompose contributions from the stopping criterion, acquisition function, and transfer learning.

- **Empirical demonstration that strong transfer priors can dominate multi-fidelity mechanics.** The observation that FSBO (a black-box transfer-BO that evaluates only at the last epoch) outperforms most multi-fidelity baselines (Fig. 4) is a sharp and practically important finding, underscoring that sample efficiency from transfer learning can outweigh the benefit of multi-fidelity decisions when the surrogate prior is weak.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison with cost-aware BO baselines.** The paper's central claim is that explicitly representing cost-performance utility improves over conventional BO. However, the baseline set contains only methods that ignore cost (multi-fidelity BO methods optimizing asymptotic performance) or that use a simpler fixed budget. There is no comparison with methods that incorporate cost into the acquisition function (e.g., EI-per-unit-cost, cost-cooled EI, BOCA, or multi-fidelity acquisitions that weight improvement by evaluation cost). Without this, it is impossible to determine whether CMBO's gains come from the principled utility formulation or simply from the superior surrogate and stopping logic. This is a critical omission given the paper's framing.

- **No comparison with simple heuristic stopping baselines.** Practitioners facing limited compute often just "run BO for K epochs." Without a comparison to "fixed shorter budget" or "stop after K steps with no improvement" baselines, it is unclear whether the complex utility + stopping machinery is necessary, or whether one would obtain similar cost-performance trade-offs by simply truncating a standard multi-fidelity BO at a shorter total budget.

- **Stopping criterion is heuristic and under-justified.** Eqs. (3)–(5) combine an approximate normalized-regret estimate (using hand-constructed Û_max and Û_min) with a PI-based threshold modulated by BetaCDF^γ. While the idea is intuitive, the paper provides no theoretical justification for this specific construction, no analysis of when the regret proxy is well-calibrated (it can be severely distorted if U_min is a loose lower bound), and no ablation demonstrating that β=e^{-1} transfers across utility families. The β ablation in Fig. 7d covers only PD1; its stability on LCBench and TaskSet is untested, and there is no principled guidance for setting β on a new problem.

- **Utility estimation from preferences is lightly validated.** The paper presents preference learning as a key contribution (Abstract, §3.1, §4.7), yet experiments almost entirely use hand-specified utility functions. The single "Estimated" condition in Table 2 constructs preferences synthetically by anchoring to iFBO's trajectory—not from actual human annotators. Fig. 2 shows one synthetic recovery example. Critical questions remain unanswered: (a) how many pairwise comparisons are needed in practice, (b) sensitivity to preference noise or misspecification of the utility family, (c) effect of utility estimation error on stopping and acquisition decisions. The "Estimated" condition in Table 4 is described as preference-based but no details of the elicitation protocol are provided.

### Minor

- **Table 3 presentation error.** Rows 3 and 4 of Table 3 have identical checkmarks (p_b=✓, Acq.=✓, T.=✓) but report dramatically different regrets (4.4 vs. 0.9 for α=2e-4). The intended distinction (presumably T.=✗ vs T.=✓ for the last row) is lost in the formatting, making the ablation hard to interpret. This should be corrected.

- **LC mixup over heterogeneous configuration spaces.** The second-stage mixup in §3.3 linearly interpolates configuration–curve pairs: (x'', l'') = λ₂(x_n, l'_n) + (1−λ₂)(x_{n'}, l'_{n'}). For benchmarks with categorical or log-scaled hyperparameters, linear interpolation of raw x-vectors does not correspond to any meaningful intermediate configuration. The paper does not address how x is represented or whether mixup is restricted to continuous embeddings. Since the surrogate quality drives most of CMBO's advantage, this assumption deserves explicit justification or qualification.

- **Algorithm 1, Line 4 notation issue.** Line 4 reads `n* ← arg max_{n ∈ C} A(n)`, but C is defined as a set of (x, t, y) triples (the history), not a set of configuration indices. At the first BO step, C is empty, so the argmax is over an empty set. The intended range is presumably n ∈ [N] or x_n ∈ X. This is likely a notation bug but affects reproducibility.

- **Uniform per-epoch cost assumption.** The method uses BO steps (epochs) as the cost axis. This assumes equal wall-clock cost per epoch across all configurations and tasks. Yet the object-detection dataset trains ResNet-50, HR-Net, and MobileNetv2—architectures with very different per-epoch costs. The paper acknowledges this implicitly via the Quick-Tune† modification (which removes cost-weighting for non-uniform wall-time), but does not discuss what happens to utility optimality when epoch cost is heterogeneous.

- **Computational overhead not reported.** CMBO's acquisition requires MC sampling of full future trajectories for all N configurations at each BO step, plus maximisation over Δt. The paper does not report surrogate training time, per-step inference cost, or comparison with the wall-clock overhead of baselines. For a method that claims cost-sensitivity, this omission is notable.

### Tiny

- Fig. 5 is explicitly labeled "cherry-picked examples." While the aggregate tables (Tables 1–4) are the primary evidence, the paper should at least note that per-task breakdowns appear in the appendix to make it clear that the cherry-picked examples are representative of aggregate trends, not chosen to hide failures.

- The paper says Û_min is approximated by decaying y^{worst}_1 across B steps. The rationale for this specific approximation and its sensitivity are not discussed, even briefly.

---

## Nice-to-Haves

- A controlled robustness study on utility misspecification: inject varying levels of noise into the utility function parameters and measure the degradation in stopping quality and final regret. This would directly validate the preference-learning pipeline's practical limits.

- Visualization of LC extrapolation quality at early BO stages (e.g., after 1, 3, 5 observations) to provide direct evidence that the transfer-learned surrogate is reliable when it matters most for stopping decisions.

- At least one experiment with heterogeneous per-epoch wall-clock costs, where b in the utility function is replaced by cumulative compute time, to demonstrate applicability to the motivating cloud/Slurm scenario.

- A per-task win/tie/loss breakdown or effect-size analysis beyond average rank, to assess whether CMBO's advantage is broad and consistent or driven by a subset of tasks.

- Sensitivity analysis for γ (the exponent in BetaCDF^γ) to complement the β ablation, since γ sets the baseline stopping threshold in the regret-only limit.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Zero standard deviations for deterministic methods (Harsh Critic).** The critic flags "exactly 0.0 standard deviation" as suspicious. The paper explains that FSBO and Quick-Tune† are deterministic methods and that task averaging eliminates all stochasticity for such methods. This is a legitimate and clearly explained design choice; no problem exists here.

- **"Dramatic improvement" wording in the conclusion (Harsh Critic).** This is a stylistic/tone concern with no substantive technical content. The empirical results do support strong improvements; the word choice is a matter of writing style and does not constitute a weakness.

- **U_prev not being the best-so-far utility creates potential for oscillation (Harsh Critic).** The paper explicitly discusses and defends this in §3.2: "the cost of BO that has previously been incurred is not reversible … U_prev can either increase or decrease during the BO, and we need to stop the BO when U_prev starts decreasing monotonically." The design choice is clearly motivated and the acquisition is specifically formulated to handle it. While a deeper theoretical analysis would be welcome, the absence of one does not invalidate the approach.

- **Optimization target max_{n,t} y_{n,t} unusual (Harsh Critic).** BO performance ỹ_b is the best validation seen up to step b. With α=0 this matches standard multi-fidelity HPO. With α>0, it is a natural consequence of optimizing a trajectory-based utility. The paper is clear about this definition; this is not a flaw.

- **Preference learning literature is under-covered in related work (Harsh Critic).** Appendix §A explicitly defers related work on cost-sensitive HPO and BO with user preference. Criticizing related work coverage without access to the appendix is unreliable.

- **Comparison with FSBO as unfair to CMBO (multi-fidelity vs. black-box) (Harsh Critic).** The comparison is intentionally informative: if a simpler black-box transfer-BO can match or exceed complex multi-fidelity methods, it shows the importance of priors. The "unfairness" here favors the baseline (FSBO), not CMBO, making the result a stronger point for CMBO when it wins and an honest admission when it doesn't. This should not be counted as a weakness.

---

## Novel Insights

The most insightful observation arising from this work—beyond the expected contributions—is that **the dominant source of performance improvement in modern multi-fidelity BO appears to be surrogate quality (transfer learning) rather than multi-fidelity decision-making per se**. The finding that FSBO, a black-box method that never exploits intermediate epoch information, outperforms all multi-fidelity baselines on LCBench and nearly so on PD1 (Fig. 4) is a sharp empirical result with important implications: years of algorithmic development in freeze-thaw BO and Hyperband variants may have been bottlenecked primarily by weak priors, not by suboptimal fidelity-selection policies. This motivates a rethinking of the standard narrative that multi-fidelity exploration is the key lever for HPO efficiency. The CMBO results then show that once a strong transfer prior is in place, the utility-aware acquisition and stopping criterion provide substantial additional gains—particularly for aggressive cost penalties—suggesting these two axes (prior quality and cost-sensitivity) are complementary and largely orthogonal.

---

## Suggestions

1. **Add cost-aware BO comparison.** Include at least one established cost-aware baseline (e.g., EI/cost or a cost-adjusted version of iFBO) on the same benchmarks. Even a simple acquisition that divides EI by expected epoch cost would clarify whether the utility framing provides structured gains beyond naive cost-weighting.

2. **Fix Table 3 row labeling.** Correct the duplicate row headers in Table 3 (rows 3 and 4 both showing p_b=✓, Acq.=✓, T.=✓) to reflect the intended ablation difference.

3. **Add a simple-budget baseline.** Compare CMBO against "run iFBO for K epochs, then stop" where K is chosen to match CMBO's average stopping budget. This isolates whether the utility-aware machinery adds value beyond reduced total budget.

4. **Provide computational overhead measurements.** Report per-step inference time for CMBO and representative baselines (iFBO, DPL) on a common hardware setting. This is essential for a cost-sensitive method.

5. **Clarify hyperparameter representation in LC mixup.** Specify how categorical or log-scaled hyperparameters are represented when computing the configuration mixup (x'', l'') = λ₂(x_n, l'_n) + ..., and whether the mixup applies to raw parameter vectors or continuous embeddings.

6. **Extend β ablation to LCBench and TaskSet.** The current ablation (Fig. 7d) only covers PD1. At minimum, a brief table showing robustness of the β=e^{-1} choice across all three benchmarks would substantially strengthen the stopping criterion's credibility.

---

## Evaluation on Key Axes

- **Originality:** High. The utility-aware acquisition with dynamic Δt selection and the combined regret+PI stopping criterion are genuinely novel formulations for multi-fidelity BO. The LC mixup strategy is a creative and practical adaptation of mixup for PFN transfer learning. The overall problem formulation (explicitly optimizing a user-defined cost-performance trajectory utility) is a meaningful departure from the asymptotic-performance focus of prior work.

- **Importance of research question:** High. Cost-sensitive HPO is directly relevant to practitioners operating under cloud budgets or cluster allocations, and the observation that specifying a cost-performance trade-off is easier than specifying a target budget is practically well-motivated.

- **Claims well-supported:** Moderate-to-high. The empirical results are broad, consistent, and supported by ablations. The acquisition analysis (Fig. 7a–c) provides mechanistic insight. The main unsubstantiated claim is the preference-learning pipeline, which is presented as a key contribution but validated only superficially.

- **Soundness of experiments:** Moderate. The benchmarks and baseline set are appropriate and broad. However, the absence of cost-aware BO baselines is a notable gap given the paper's framing, and the stopping criterion comparison is inherently disadvantageous to baselines (who cannot use the PI-based component). The Table 3 error undermines confidence in the ablation.

- **Clarity of writing:** Good overall, with intuitive explanations and well-designed figures. The Algorithm 1 notation issue and the Table 3 presentation error are concrete clarity failures that should be fixed.

- **Value to the research community:** Solid. The empirical finding about transfer learning dominating multi-fidelity mechanics has implications beyond this paper, and the utility framework and surrogate training recipe are directly usable by practitioners.

- **Contextualization relative to prior work:** Adequate in the main text; the paper positions itself clearly against freeze-thaw BO and transfer-BO literature. The positioning against cost-aware BO is underdeveloped, though the appendix apparently contains additional discussion.

MY FINAL SCORE: <pineapple>6.4</pineapple>

---

