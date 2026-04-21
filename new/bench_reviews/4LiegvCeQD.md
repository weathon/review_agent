Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method where multiple independently pre-trained classifiers mutually learn from each other by minimizing cross-entropy between each model's output and the softmax of the model with the highest predicted probability for the majority-voted class. IEL operates one sample at a time with frozen batch normalization, making it applicable in the restrictive single-sample TTA setting. Experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C show improvements over a static (un-adapted) ensemble.

## Strengths

- **Novel and intuitive core idea**: Dynamic mutual distillation within an ensemble, where the teacher changes per sample based on majority vote, is a clean and conceptually appealing approach. The algorithm is clearly specified (Eq. 1, Algorithm 1), and the self-consistency property—where the selected model's loss contribution is just its own entropy, which is low by construction—is a sound design insight (Section 3).

- **Strong improvements on ImageNet-C**: Table 3 shows positive improvements on all 15 corruption types for majority vote accuracy, with several large gains (e.g., glass blur +17.78/+17.71, zoom blur +15.37/+17.86, motion blur +13.11/+13.71). This is the paper's strongest evidence that IEL delivers on its core claim.

- **Operates in single-sample, frozen-BN setting**: By冻结 all batch normalization parameters and using batch size 1 (Section 4), IEL addresses a genuinely restrictive setting where methods like TENT become ineffective. This is a practically relevant niche that many TTA methods cannot handle.

- **Honest reporting of failure cases**: The paper transparently reports catastrophic forgetting on noise-type corruptions in CIFAR-10-C and CIFAR-100-C (Tables 1–2), with negative accuracy deltas shown rather than omitted.

- **Thorough breadth of evaluation**: Three datasets × 15 corruption types, with tuning/evaluation splits to assess generalization within corruption types.

## Weaknesses

### Fatal
None.

### Major

- **Oracle epoch selection inflates reported results.** Tables 1–3 explicitly report "Highest accuracy improvements (%) over all epochs," meaning each number reflects the best epoch selected retrospectively. In the TTA setting with no ground-truth labels, there is no oracle to identify the best epoch, and the paper's own Figure 3 shows that accuracy can peak and then degrade. Without a practical stopping criterion or fixed-epoch results, the actual achievable accuracy is unknown and likely substantially lower than reported. The paper acknowledges this as future work (Section 3.1: "In future work, we would like to terminate IEL once all model predictions agree with each other"), but the method as evaluated cannot be deployed as described. This is a serious evaluation gap that inflates the headline numbers.

- **No comparison to any existing TTA method.** The paper positions IEL as a TTA contribution and discusses TENT, EATA, COTTA, and ROID in related work (Section 2.2), yet the only experimental baseline is a static (un-adapted) ensemble. There is no comparison showing that IEL outperforms simply applying existing single-model TTA (e.g., TENT with a small batch, or EATA in single-sample mode) to the individual ensemble members or to the ensemble as a whole. Without this comparison, it is impossible to determine whether IEL provides any advantage over existing TTA approaches—even in the single-sample setting IEL targets. The paper argues TENT is ineffective at batch size 1 but provides no experiment demonstrating this claim.

### Minor

- **Systematic failure on noise-type corruptions on CIFAR lacks characterization.** On CIFAR-10-C and CIFAR-100-C, IEL *degrades* majority-vote accuracy by up to −20.56% (Gaussian Noise, Table 1) and −15.67% (Impulse Noise, Table 2). These failures are systematic (all three noise types) rather than random. The paper dismisses this as "only 3 of 15 corruption types" (Section 4.1), but does not explain *why* majority-vote consensus fails precisely on noise corruptions, nor provide any quantitative predictor (e.g., agreement rate, ensemble confidence) for when IEL helps vs. hurts. Notably, ImageNet-C shows improvements on noise types (Table 3: Gaussian Noise +8.19, Impulse Noise +7.97), suggesting the failure is scale-dependent, but this contrast is unexplained.

- **The claimed "single-sample advantage" is asserted but not demonstrated against alternatives.** The paper claims single-sample adaptation as a key distinguishing advantage (Abstract, Section 1), but never compares IEL against a batch-based method like TENT under identical conditions. To validate the single-sample advantage, one must either show that batch-based methods fail at batch size 1 while IEL succeeds, or that IEL at batch size 1 outperforms batch-based methods at their preferred batch size. Neither demonstration is provided.

### Trivial
None.

## Nice-to-Haves

- Per-epoch accuracy curves for all corruption types (not just Figures 2–3) to expose peak-and-degrade behavior and inform a practical stopping criterion.
- A simple stopping rule (e.g., entropy- or agreement-based) evaluated without oracle epoch selection, to show realistic achievable performance.
- Experiments on non-stationary shifts (sequential or mixed corruption types), which are the more realistic TTA setting and may trigger the catastrophic forgetting the authors acknowledge.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **Harsh critic's claim that the cross-entropy self-term actively degrades the strongest model**: The critic argued that the gradient of the entropy of the selected model's own output "pushes that model toward even more confident (possibly wrong) predictions." While technically the gradient is nonzero, the paper correctly notes this contribution is small (since the selected model already has low entropy). The paper also explicitly acknowledges this risk and proposes KL divergence for future work (Section 3). This is a minor theoretical nuance, not a major flaw. Moved to context in Minor weaknesses above as part of the noise-failure characterization gap.

- **Harsh critic's complaint about the regularization constant α = 10e−11 being "effectively zero"**: The paper itself states this "effectively makes our learning rate even smaller" (Section 4). This is transparent, not a hidden problem. Trivial observation.

- **Harsh critic's concern about the 90/10 split not reflecting real TTA**: The split is within a corruption type to measure generalization, not to simulate the domain shift itself. This is a reasonable experimental choice for what it aims to measure, and the paper's main distribution-shift claim rests on the corruption types themselves, not the split.

- **Harsh critic's concern about "no comparison to TTA baselines" phrased as if this makes the paper not a TTA contribution at all**: The paper clearly operates in the TTA space and has a valid TTA setup (single-sample, no labels, distribution shift). The gap is in evaluation completeness (missing baselines), not in the method's identity as a TTA approach.

- **Strength finder's claim about "entropy minimization emerges as a validated side effect"**: This is partially supported by Figure 1 showing correlation, but the correlation between loss and entropy does not validate that entropy minimization is causally beneficial or that it avoids pitfalls of direct entropy minimization. Dropped as overly strong.

- **Strength finder's claim about "beneficial self-consistency property" as a separate strength**: This is already captured in the main strengths and is a design property of the loss function, not an empirical finding. Merged into the core idea strength.

## Novel Insights

The most revealing observation from the reviews is the tension between what IEL claims to be and what it actually demonstrates. IEL is positioned as a TTA method for the single-sample setting, and its strongest empirical evidence (ImageNet-C, Table 3) shows substantial and universal improvements. Yet the oracle epoch selection means these improvements are upper bounds rather than achievable performance, and the lack of TTA baselines means we cannot determine whether even these upper bounds beat simple alternatives. The core idea—ensemble mutual distillation—has genuine appeal, but the paper evaluates it as an optimization procedure rather than as a deployable TTA method, leaving the key question unanswered: can IEL be used in practice without an oracle, and if so, how does it compare to the simplest alternatives?

## Suggestions

- Report accuracy at specific fixed epochs (e.g., epoch 1, epoch 3, final epoch) alongside the oracle "highest over all epochs" numbers, so readers can assess realistic achievable performance.
- Add at least one comparison to an existing TTA method (e.g., TENT applied to individual models) even under the paper's preferred single-sample, frozen-BN conditions, to contextualize IEL's contribution within the TTA landscape.
- Analyze the ensemble agreement rate or confidence as a potential trigger for when to apply IEL vs. when to skip it, especially to address the systematic noise-corruption failures on CIFAR.

## Score and Decision

### Calibration Anchors

| Paper | Score | Comparison |
|-------|-------|------------|
| READ (TPZRq4FALB) | 8.0 | Multi-modal TTA with proper baselines, new benchmarks, comprehensive evaluation. Far stronger than this paper. |
| DeYO (9w3iw8wDuE) | 7.0 | Single-sample TTA with proper comparisons to existing methods. Demonstrates advantage over baselines directly. Stronger evaluation than this paper. |
| HyRe (8HQS1X2AK4) | 5.33 | Ensemble reweighting at test time with missing comparisons to prior work. Similar profile: interesting ensemble idea but missing critical baselines. This paper has the additional oracle-selection problem, making it weaker. |
| ROSITA (lF9QXpfNHm) | 4.67 | TTA paper with overclaimed novelty and missing baselines. Very similar weakness profile to this paper. |
| DART (xqxG5WogN6) | 5.67 | TTA method with novelty concerns and limited evaluation rigor. |
| Test-time prompt tuning (pdzHpQbGrn) | 2.5 | Very weak novelty and evaluation. This paper is clearly stronger. |
| Benchmarking paper (FaL6aTuXod) | 1.5 | Fundamentally flawed evaluation. This paper is far stronger. |

This paper shares the same core weakness pattern as ROSITA (4.67) and HyRe (5.33): an interesting idea with incomplete evaluation, missing baselines against directly comparable TTA methods, and overclaimed results. It additionally suffers from oracle epoch selection, which inflates all headline numbers. The core idea has genuine merit and the ImageNet-C results are compelling even as upper bounds, but the two major weaknesses—no TTA baselines and oracle selection—significantly undermine the contribution claims. I place it slightly below HyRe (which at least compared against some baselines) and around ROSITA's range.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>