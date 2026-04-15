Now let me search for calibration papers.Now I have enough information to write the final consolidated review.

---

## Summary

The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation (TTA) method for single-sample settings. Given an ensemble of independently pre-trained classifiers, IEL identifies the majority-voted class at each test step, selects the ensemble member with the highest predicted probability for that class as a soft-target "teacher," and minimizes the cross-entropy between all member model outputs and that teacher via backpropagation. The method is positioned as filling a gap in the single-sample TTA literature, where batch-statistics-dependent methods like TENT are ineffective. Experiments on CIFAR-10C, CIFAR-100C, and ImageNet-C report accuracy improvements over a static ensemble baseline for most corruption types.

---

## Strengths

- **Practically motivated setting**: Single-sample TTA is a genuinely under-served regime, and the paper correctly identifies that many existing TTA methods (e.g., TENT) degrade in this regime because batch normalization statistics cannot be reliably estimated.
- **Architecturally diverse model zoo**: Experiments use five heterogeneous architectures (ResNets, VGG, MobileNet, ShuffleNet, RepVGG) rather than ablating a single family, which adds credibility that the method is not architecture-specific.
- **Strong results on ImageNet-C**: Table 3 shows positive gains across all 19 ImageNet-C corruption types with non-trivial magnitudes (e.g., +17.71% on glass blur), suggesting a real signal rather than marginal improvements.
- **Honest reporting of negative results**: The paper candidly reports catastrophic forgetting on 3/15 CIFAR-10C corruption types (Gaussian, Shot, and Impulse Noise) and 4/15 CIFAR-100C types, which strengthens credibility.
- **Simple mechanism**: The method is easy to implement and relies on no auxiliary architecture, augmentation, or access to source data, which are practical advantages.

---

## Weaknesses

### Fatal
*None that completely invalidate the work as a paper, but two Major issues together make acceptance very difficult in current form.*

### Major

- **No comparison to any TTA baseline**: The paper positions itself squarely in the TTA literature, devotes §2.2 to TENT, EATA, CoTTA, and ROID, and frames its contribution explicitly as a TTA approach. Yet every experiment compares only against the *static (non-adapted) ensemble*. There is no comparison to single-model entropy minimization applied independently to each member, EATA (which handles single-sample settings), ROID (which explicitly uses an ensemble of past model states), CoTTA, or any other adapted baseline. Demonstrating that some adaptation beats no adaptation is well-established and does not establish IEL's value as a TTA method. Without these comparisons, the reader cannot determine whether IEL's gains come from the peer-distillation mechanism specifically or from any self-training scheme. This is the most consequential gap.

- **Best-epoch reporting on the held-out evaluation split compromises the generalization claim**: Tables 1–3 all state "Highest accuracy improvements (%) **over all epochs**" and show numbers for both the tuning set *and* the evaluation set. This means the reported evaluation-set numbers are the best achieved across all training epochs, not a fixed-point result. While the evaluation set receives no gradient updates, using it to determine which epoch to report de facto incorporates it into model selection. Given that the paper itself (§3.1, Fig. 3) explicitly shows that performance peaks around epoch 4–5 and subsequently declines or collapses toward baseline—making epoch choice material—this cherry-picking weakens the headline claim that "IEL can improve generalization error." A per-epoch table or consistent final-epoch reporting alongside peak accuracy would be necessary to assess the true magnitude of generalization gains.

### Minor

- **No ablation isolating peer distillation from entropy minimization**: The paper acknowledges (§1, §3) that minimizing cross-entropy to a low-entropy teacher is mathematically equivalent to entropy minimization for all members, and that this is a "side effect." But it never tests a baseline of independent per-model entropy minimization, nor a variant where the teacher softmax is replaced with the hard majority-vote label. Without these ablations, the contribution of *cross-model knowledge transfer* versus simple *entropy reduction* per model is unverified.

- **Confusing conceptual framing**: The contributions list "diversity as a new optimization signal," but the method explicitly minimizes diversity (as the paper acknowledges: "we minimize the diversity of the ensemble"). What the paper means is that it uses the *cross-entropy between models* (a diversity measure) as the optimization objective. This should be stated clearly rather than claiming diversity is the signal—it will confuse readers familiar with diversity-promotion methods in ensemble learning.

- **Unrealistic evaluation protocol (weight reset per corruption)**: Model weights are reset before each corruption type, implying knowledge of distribution boundaries. This assumption is unrealistic for continuous deployment and prevents evaluation of catastrophic forgetting in realistic non-stationary streams.

- **Unsupported batch-size-1 claim**: The paper states "Using only one sample per batch was empirically found to produce identical performance gains as using multiple samples per batch" but presents no data for this. Results across batch sizes would be needed to support this claim.

### Trivial

- The regularization constant description ("α = 10e⁻¹¹") is ambiguous notation, and no justification or sweep is given for the learning rate. The role of α in the loss is not explained.
- No run-to-run variance or confidence intervals are reported.

---

## Nice-to-Haves

- Provide per-epoch accuracy curves for representative failing and succeeding corruption types (beyond Fig. 2/3) to give a complete picture of convergence and stability.
- Analyze majority-vote error rates on noise-type corruptions to understand *why* catastrophic forgetting specifically occurs there (is it that the ensemble is collectively wrong more often?).
- Vary the number of ensemble members (2, 3, 5, 10) to show how performance scales—this is central to ensemble method evaluation and is acknowledged as important but deferred to future work.
- Quantify computational overhead (wall-clock time, FLOPs, GPU memory) vs. static inference and vs. single-model TTA, given that IEL backpropagates through all models per sample.

---

## Removed Points

*These points are flagged as removed; treat them with caution as they did not survive scrutiny.*

- **Harsh Critic – "diversity as optimization signal is conceptually invalid"**: REMOVED as overstated. The paper explicitly acknowledges it minimizes diversity and provides a clear (if awkward) framing. It is kept as a Minor weakness (confusing framing) rather than a structural flaw.
- **Harsh Critic – "mechanism is purely entropy minimization, making IEL's contribution invalid"**: PARTIALLY REMOVED. The paper acknowledges the mathematical relationship to entropy minimization. The entanglement is real and worth noting (kept as a Minor weakness on ablations), but it does not render the method trivially invalid—the peer selection mechanism based on majority voting is a non-trivial choice with potential benefits independent of entropy alone.
- **Human Finder – "unfair comparison due to multiple adaptation epochs"**: REMOVED. The static baseline is genuinely non-adapted. Complaining that IEL adapts while the baseline does not is simply criticizing the fact that IEL is a learning method, which is the entire point. This asymmetry favors the baseline (doing nothing has zero compute cost), so the comparison is not unfair to IEL.
- **Harsh Critic – "Eq. (1) includes selected teacher in the loss, confounding mechanisms"**: REMOVED as a strawman. The paper explicitly discusses this (§3): the teacher's self-cross-entropy is just its own entropy, which is near-zero for a high-confidence prediction, making the teacher term a negligible contributor. This is a feature, not a flaw.

---

## Novel Insights

The most insightful observation across all reviewers—confirmed against the paper—is that the evaluation methodology itself (best-epoch reporting on the held-out evaluation split) is structurally inconsistent with the paper's central generalization claim. Taken together with the complete absence of any TTA baseline comparison, the paper currently demonstrates only that "adapting an ensemble to a stationary stream beats not adapting," which is an uncontroversial result. The core mechanism—using ensemble majority-vote confidence to dynamically select a peer teacher—is a genuinely novel and practically lightweight idea for single-sample adaptation; however, the evidence required to establish it as a *better* TTA method than existing alternatives is simply missing.

---

## Suggestions

1. **Add at minimum two TTA baselines**: (a) per-model independent entropy minimization with the same learning rate and epochs, and (b) EATA or ROID applied to each ensemble member individually. This is essential for positioning the contribution.
2. **Fix evaluation reporting**: Report per-epoch accuracy on the evaluation split (or final-epoch results alongside peak), so the reader can assess generalization behavior honestly without cherry-picking.
3. **Add the peer-distillation ablation**: Replace the soft teacher target with the hard majority-vote label to isolate the contribution of soft targets; run single-model entropy minimization per member to isolate the cross-model signal.
4. **Provide sequential corruption evaluation**: Remove the weight-reset-per-corruption assumption and evaluate on sequentially changing corruptions to demonstrate practical robustness.

---

## Score and Decision

**Calibration anchors reviewed:**
- *4wk2eOKGvh* (TTE via Linear Mode Connectivity, **accepted**, avg 6.5): A closely related ensemble-for-TTA paper. It compares against established TTA methods (TENT, EATA, etc.), provides ablations and theoretical grounding, and demonstrates consistent improvements over prior art—not just over a static baseline. This is what a publishable ensemble-TTA paper looks like.
- *PxL35zAxvT* (TTA with Auxiliary Tasks, **rejected**, avg ~4.7): Compares against some TTA baselines, but suffers from narrative gaps and missing analysis. Still more experimentally rigorous than the paper under review.
- *vhazhSm6I0* (TTA for GNNs, **rejected**, avg ~3.75): Has baseline comparisons but incremental contribution.

The paper under review sits below PxL35zAxvT: it lacks *any* TTA baseline comparison, has a questionable evaluation protocol (best-epoch on eval split), and offers no ablations. The large positive results on ImageNet-C are encouraging and show the idea has genuine merit, but they cannot be properly interpreted without knowing whether a simpler entropy minimization baseline would achieve similar gains. The paper is significantly weaker than the accepted TTE paper, which sets the 6.5 bar for this type of work.

**Assessment on key axes:**
- *Originality*: Moderate — peer distillation for single-sample TTA is a fresh angle.
- *Importance of research question*: Good — single-sample TTA is a real gap.
- *Claims supported by evidence*: Poor — core claims require TTA baseline comparisons and honest epoch reporting.
- *Soundness of experiments*: Poor — best-epoch cherry-picking on eval split, no ablations, no baselines.
- *Clarity of writing*: Adequate — the method is understandable but the contribution framing is muddled.
- *Value to community*: Low in current form; higher if experimental gaps are addressed.

**Score: 3.5 / 10 — Reject**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>