Now I have enough calibration data. Let me synthesize my final review.

## Summary

The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method that adapts an ensemble of independently pre-trained classifiers by dynamically selecting, per sample, the model with highest confidence for the majority-voted class as a soft target, then minimizing cross-entropy between all models and this target. Each optimization step uses a single sample (batch size 1). Experiments on CIFAR-10C, CIFAR-100C, and ImageNet-C show improvements over static ensemble baselines on most corruption types, with documented catastrophic forgetting on noise-type corruptions.

## Strengths

- **Simple and well-specified algorithm.** The core idea — majority-vote-based dynamic teacher selection with cross-entropy distillation to all models — is clearly described and easy to implement. The algorithm is presented in pseudocode (Algorithm 1) and the loss function (Eq. 1) is straightforward.

- **Comprehensive per-corruption, per-model reporting.** Tables 1–3 cover 15 corruption types across 3 datasets, reporting improvements for each individual model and the majority vote ensemble, including both tuning and evaluation splits. The paper honestly reports negative results (e.g., Gaussian Noise, Shot Noise, Impulse Noise degradations), rather than cherry-picking only positive outcomes.

- **Proper tuning/evaluation split.** The 90/10 split of target-domain data into adaptation vs. generalization evaluation is good methodological practice that avoids trivial overfitting claims.

- **Batch-norm freezing and batch-size-1 setting.** By freezing BN parameters and using batch size 1, the paper ensures gains are not artifacts of re-estimated batch statistics, strengthening attribution of improvements to IEL itself.

- **Demonstrates that multi-model joint adaptation can work.** The experiments convincingly show that adapting multiple models simultaneously via consensus-driven distillation can yield substantial accuracy gains on many corruption types (e.g., +20% on Glass Blur for ResNet50 on ImageNet-C), establishing that the phenomenon is real.

## Weaknesses

### Fatal
None.

### Major

- **No comparison to any TTA baselines.** The paper positions itself squarely within the TTA literature (discussing TENT, EATA, ROID, CoTTA in Section 2.2) and makes claims about IEL being "a solid step forward" for TTA, yet the experiments compare only against a *static ensemble* (no adaptation). Without baselines such as TENT applied to individual models, simple per-model entropy minimization, or single-sample TTA methods like EATA/ROID, it is impossible to assess whether IEL's gains come from the proposed ensemble mechanism or simply from the fact that any gradient-based adaptation on individual models would improve over a static baseline. This is a structural evaluation gap that undermines the central empirical claim. (Even a single baseline of entropy minimization on each model independently would be highly informative.)

- **Misleading "diversity as optimization signal" framing.** The contributions list (Section 1) claims "diversity as a new optimization signal," but the actual loss (Eq. 1) *minimizes* cross-entropy between members and the majority-voted teacher, explicitly *reducing* diversity. The paper acknowledges this in the text ("by minimizing the cross-entropy distances … we minimize the diversity of the ensemble"), but the contributions and human-collaboration narrative sell this as leveraging diversity. The real optimization signal is *consensus/distillation*, not diversity. This overclaims the conceptual novelty: IEL is essentially online mutual distillation to a dynamic majority-voted teacher — a reasonable idea, but not a new diversity principle.

- **"Highest accuracy over all epochs" reporting inflates results.** Tables 1–3 report the best accuracy achieved at any epoch for each corruption. Figure 3 clearly shows that IEL accuracy peaks around epoch 4–5 and then degrades, in some cases falling back to or below the static baseline by later epochs. Selecting the peak epoch in hindsight gives an optimistically biased picture of IEL's practical performance, since at deployment time there is no oracle for choosing the right stopping point. Without reporting accuracies at a fixed epoch, showing full trajectories, or establishing an early-stopping criterion, the reported numbers are not representative of achievable performance.

### Minor

- **Single-sample TTA claim is somewhat misleading.** The paper repeatedly emphasizes the "single sample" setting as a key contribution. While technically each optimization step uses batch size 1, the method is evaluated with multi-epoch adaptation over thousands of samples from the *same stationary corruption type* (10,000 samples for CIFAR-C, 7,000 for ImageNet-C with weights reset between corruptions). This is closer to unsupervised domain adaptation with a known stationary target than to the truly online, single-sample-per-distribution TTA scenario implied by the framing.

- **No ablation on ensemble size, architecture diversity, or loss function alternatives.** The paper uses fixed sets of 5 models (CIFAR) or 4 models (ImageNet) and cross-entropy loss. Key ablations are missing: (a) How does IEL scale with 2, 3, or 10+ models? (b) Does architecture diversity matter, or would homogeneous ensembles work? (c) The paper itself notes (Section 3) that using KL divergence (which yields 0 when inputs match) might be better than cross-entropy, but no comparison is provided.

- **Catastrophic forgetting on noise corruptions is reported but not analyzed.** Tables 1–2 show severe degradation on Gaussian Noise, Shot Noise, and Impulse Noise (e.g., −15% to −31% for individual models). The paper mentions this as a limitation but provides no analysis of *why* these corruptions cause failure or when practitioners should expect IEL to help vs. harm. Understanding failure modes is critical for practical adoption.

### Trivial

- Minor notation ambiguity in $H(\mathbf{x})$: it is described as "the softmax output of the model" but could be read as a model index, since the argmax in the definition selects over models, not softmax vectors. The usage in Eq. (1) clarifies this, but could be cleaner.

## Nice-to-Haves

- Report per-sample inference time, memory, and FLOPs for IEL vs. single-model TTA methods, since IEL requires M forward/backward passes per sample.
- Test on non-stationary TTA protocols (e.g., sequential corruption types without weight resets) to assess robustness in realistic deployment scenarios.
- Track and report ensemble diversity (e.g., disagreement metrics) over epochs to empirically validate or refute the diversity-reduction claims.
- Consider an early-stopping or diversity-preserving mechanism to address the degradation observed in later epochs (Figure 3).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"IEL is too computationally expensive for practical deployment."** While valid conceptually, the paper explicitly acknowledges this limitation (Section 4.1, "Limitations and Future Work") and argues it is "a cost worth paying" for the ability to adapt all parameters. This is a known trade-off, not a hidden flaw. Downgraded to nice-to-have.

- **"No comparison to online mutual distillation / co-training literature."** The paper does discuss knowledge distillation (Section 2.3) and connects IEL to the KD framework. While it could discuss Deep Mutual Learning and co-training more explicitly, demanding exhaustive coverage of all related work falls into the "missing related works" category, which should not be penalized per the rules.

- **"No evaluation on real-world distribution shifts or segmentation/detection tasks."** The paper scopes its evaluation to CIFAR-C and ImageNet-C, which are standard TTA benchmarks. Requesting additional domains goes beyond the stated scope and is a generic "do more experiments" request.

- **"The 'single-sample' setting is not truly single-sample because they use many samples across epochs."** While the framing is somewhat misleading (moved to minor), the method *does* operate on one sample at a time — the batch size is 1 and each step requires a single forward/backward pass. The distinction between batch-size-1 and having access to a stream of samples from a stationary distribution is a real methodological concern, but it is not a misrepresentation of what the method technically requires.

- **"No confidence intervals or multiple runs reported."** Single-run evaluation without variance estimates is standard practice in the TTA literature on these benchmarks. Requesting confidence intervals for large-scale benchmarks where single-run reporting is the norm is an unreasonable demand for this field.

## Novel Insights

The most interesting novel observation in the reviews and the paper is the tension between IEL's "diversity" narrative and its actual mechanism: IEL exploits pre-existing diversity (different models making different predictions) to select a teacher, then systematically destroys that diversity through cross-entropy minimization. This creates a temporal dynamic — early in adaptation, diversity is high and the majority vote is informative; over time, models homogenize, diminishing the information in the vote. This explains both why IEL peaks and then degrades (Figure 3) and why noise corruptions (where pre-existing accuracy is very low, making majority votes unreliable) cause catastrophic failure. The method's success and failure modes are thus structurally linked to the same mechanism, which is an insight the paper itself does not articulate but the experimental evidence supports.

## Suggestions

1. **Add at minimum a per-model entropy minimization baseline** (same learning rate, same single-sample setting, same epochs) to determine whether the ensemble soft target mechanism is necessary for the observed gains or whether simple adaptation of individual models suffices.
2. **Report results at a fixed epoch** (e.g., epoch 1 or epoch 3) rather than best-over-all-epochs, or show full accuracy trajectories for representative corruptions in the main paper.
3. **Reframe the contribution**: Replace "diversity as a new optimization signal" with "ensemble consensus as a dynamic soft target for test-time mutual distillation." The method is interesting as an online distillation procedure; overselling it as a diversity signal weakens credibility.

## Score and Decision

**Calibration:**
- TTE (4wk2eOKGvh.md): scores 6/8/6/6, accepted poster — had proper baselines, integration with existing TTA methods, clear methodology.
- DeYO (9w3iw8wDuE.md): scores 8/6/6/8, accepted spotlight — identified a real problem, had strong baselines and thorough analysis.
- Intransigent Teachers (Chq4OQ3p18.md): scores 5/3/8/5, rejected — missing baselines, limited novelty, overclaimed.
- FEATHER (6yJuDK1DsK.md): scores 5/5/3/5, rejected — limited evaluation, weak methodological contribution.

IEL falls below TTE and DeYO (which had proper baselines and thorough evaluation) but is somewhat above papers that have fundamental methodological flaws. The core idea is reasonable (online ensemble distillation for TTA), the method is clearly described, and improvements over static baselines are substantial. However, the lack of *any* TTA baselines, the misleading "diversity" framing, and the best-over-epochs reporting are significant weaknesses that prevent confident assessment of IEL's contribution relative to the state of the art. The paper shows a promising phenomenon but does not adequately establish its value in context.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>