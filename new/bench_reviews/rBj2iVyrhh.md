Now I have a thorough understanding of both the paper and the calibration landscape. Let me compose my final review.</think>## Summary

The paper proposes Classifier-Constrained Alternating Training (CCAT) to address modality imbalance in multimodal learning. The key insight is that while alternating training methods reduce encoder-level interference, they fail to prevent the shared classifier from developing bias toward faster-converging modalities. Drawing an analogy to class imbalance remedies, CCAT pretrains an "unbiased" classifier with bidirectional cross-attention and contribution-balancing regularization, then freezes it during alternating modality training while using LoRA adapters for modality-specific adaptation. A sample-level secondary update mechanism targets severely imbalanced samples. Experiments on three bimodal benchmarks show consistent improvements over prior SOTA.

## Strengths

- **Clear and well-motivated problem identification.** Figure 1 provides empirical evidence that alternating training (MLA) reduces but does not eliminate modality contribution disparity (1.00→0.92), supporting the claim that encoder-level interventions alone are insufficient. This is a meaningful and previously underappreciated observation about alternating training methods.

- **Systematic and coherent framework.** Each component addresses a specific aspect of the identified problem: pretraining with regularization produces a balanced classifier, freezing prevents dominance drift, LoRA enables modality-specific adaptation without disrupting the shared decision boundary, and secondary updates target sample-level imbalance. The ablation study (Table 2) confirms all components contribute positively on CREMA-D.

- **Strong empirical results on Kinetic-Sound.** The +6.76% improvement on Kinetic-Sound over prior SOTA (MMPareto: 72.53% → CCAT: 79.29%) is substantial and notable, demonstrating effectiveness on a dataset where modality imbalance is severe.

- **Consistent improvements across all three benchmarks.** The method achieves improvements on every dataset (CREMA-D: +2.27%, KS: +6.76%, MVSA: +1.92%), with particularly strong gains in weaker modalities (e.g., video accuracy on CREMA-D: 73.79% vs. next best 68.01%).

## Weaknesses

### Major:

- **The theoretical "framework" in Section 3.1 is informal and overclaimed.** The paper advertises "a new theoretical framework" and "profound theoretical isomorphism" between class and modality imbalance, but the actual content consists of standard cross-entropy gradient expressions (Eq. 1), a trivial approximation for minority classes (Eq. 2), and the observation that if γ₁≫γ₂ then gradients are dominated by the stronger modality (Eq. 3). This last observation is essentially tautological and does not: (a) distinguish encoder vs. classifier parameters; (b) model γ dynamics under different training schemes; (c) derive any testable prediction about when the fixed-classifier remedy should work or fail; or (d) formally justify why freezing the classifier specifically addresses the dynamics described. The "framework" functions as a narrative analogy rather than a theoretical contribution, which makes the claim of providing "a new theoretical framework" an overclaim. This matters because the paper explicitly lists this as contribution (i).

- **The MI-based contribution measure is underspecified and its role across stages is inconsistent.** The paper uses an InfoNCE-style mutual information estimator (Eq. 5) as the basis for both pretraining regularization and sample selection, but: (a) implementation details (negatives, temperature, batch-size dependence) are not discussed in the main text; (b) during pretraining, contributions are computed on cross-attention fused features f, while during alternating training they are computed on "decision-level" unimodal outputs — yet both are called "modality contribution" as if they measure the same thing; (c) the regularizer L_reg = (1/N)Σ|c¹−c²| forces equal MI-derived contribution by design during pretraining, and then the second-stage contributions are measured with a classifier already trained under this equality-enforcing objective, creating a circularity concern where the method measures what it shapes. No validation (e.g., synthetic settings, correlation with error analysis, comparison to simpler proxies like gradient norms) is provided.

- **Ablation design does not sufficiently disentangle component contributions.** CCAT combines five interventions (cross-attention fusion, MI-regularized pretraining, classifier freezing, LoRA, secondary updates). Table 2 toggles Fix/Alt/Sec/LoRA but ambiguously initializes the classifier: the "Fix ✗" row still uses MI-regularized pretraining, yet no baseline tests a "jointly trained classifier with the same cross-attention architecture and regularization, no alternation, no freezing, no LoRA." Without that baseline, it is unclear whether the gains primarily come from better initialization via MI-regularized cross-attention pretraining, or from the classifier-constrained alternating training per se. This matters for attributing the improvements to the paper's central thesis.

- **No results reported for standard deviations or statistical significance.** All results in Table 1 are single-run values. Given that improvements on CREMA-D (+2.27% over MLA, +1.35% over the claimed SOTA LFM) and MVSA (+1.92%) are modest relative to typical run-to-run variance on these small datasets, the significance of these improvements cannot be assessed. This is a community norm issue—many papers in this space report single runs, but the small improvements on 2 of 3 datasets make it more concerning here.

### Minor:

- **Limited evaluation scope.** Only three bimodal datasets with small backbones (ResNet-18/50 + BERT) are tested. The method's applicability to trimodal settings, larger architectures, or transformer-based models is untested, though the paper makes broad claims ("balanced, robust multimodal representations"). This is a minor concern as the scope is clearly defined, but the generality claims are stronger than the evidence warrants.

- **Pretraining stage details are insufficient.** The paper does not specify: how many epochs are allocated to pretraining vs. alternating training within 150 total; whether encoders are trained or frozen during pretraining; and the "initialize {Enc_m}" in Algorithm 1 suggests encoders may restart from random weights after pretraining, which is a potentially important design choice warranting discussion.

- **Hyperparameter sensitivity is partially addressed but gaps remain.** Grid search over LoRA rank r and threshold β is reported (Table 3, Figure 4), but the regularization coefficient λ=0.001 receives no sensitivity analysis despite being conceptually central to the "unbiased classifier" claim. Optimal configurations vary substantially across datasets (β from 0.05 to 0.30, r from 2 to 8), suggesting dataset-dependent tuning.

### Trivial:

- The LFM baseline is missing results on MVSA (shown as "-"), making the SOTA claim on that dataset incomplete.

## Nice-to-Haves

- Plot CCAT's modality contribution trajectories alongside Figure 1's MLA curves to directly demonstrate that classifier freezing resolves the "entrenched bias" the paper identifies as the core problem.
- Provide a "jointly trained with same cross-attention + MI regularization" baseline to cleanly isolate the contribution of the alternating + frozen-classifier mechanism.
- Test on at least one trimodal or larger-scale dataset to support the generalization claims.
- Report computational overhead (wall-clock time, FLOPs) for the two-stage pipeline with secondary updates relative to single-stage baselines like MLA.

## Removed Points

- **"Missing baseline SMSL (Zhou et al., 2025b)":** The human finder flagged this, but SMSL is cited in the method section for defining the MI-based contribution metric, not necessarily as a direct baseline competitor. It may be an evaluation-method paper rather than a multimodal learning method, and without external verification of its applicability as a baseline on these exact benchmarks, this point is speculative.

- **"Novelty of bidirectional cross-attention is limited":** The paper does not claim bidirectional cross-attention as a novel contribution; it cites Cheng et al. (2024) for BiCross. This criticism misattributes novelty claims.

- **"No comparison with frozen random classifier":** This is an interesting experiment but goes beyond what's standardly required. The paper does ablate frozen vs. unfrozen pretrained classifier, which already addresses the core comparison. Testing a random freeze is a nice-to-have, not a required baseline.

- **"Theoretical analogy is imperfect because class imbalance involves sample quantity":** While true, the paper explicitly acknowledges this distinction by noting that class and modality imbalance share "early-dominance-triggered bias" dynamics. The analogy is about the optimization dynamics, not a claim that they are identical problems. This is a valid caveat but not a fatal flaw.

- **"Feature distribution mismatch P(z^m|y) ≠ P(f|y)":** The paper acknowledges this mismatch in Section 3.3 and proposes LoRA specifically to address it. While more analysis of how severe this is would strengthen the paper, the criticism that it is "not rigorously analyzed" is demanding work outside the paper's core scope—LoRA is a well-established technique for distribution adaptation.

- **"Harsh critic's demand for formal convergence analysis or conditions for when fixed classifier hurts":** This demands theoretical rigor above what's standard in empirical multimodal learning papers at this venue level. The paper provides gradient dynamics intuition and empirical validation; formal convergence guarantees would be a significant extension beyond its scope.

## Novel Insights

The observation that alternating training resolves encoder-level gradient interference but leaves residual classifier bias — which is structurally analogous to class imbalance in that early dominance creates a path-dependent entrenchment — is genuinely insightful, even though the formal development of this analogy is thin. The idea of importing fixed-classifier strategies from the class imbalance literature into multimodal learning is a creative mapping. However, the circularity concern around the MI-based contribution measure (where the method both enforces and evaluates balance) is an important limitation that the paper does not address.

## Suggestions

- Run 3-5 seeds on all datasets and report mean ± std. This is the single most important improvement for credibility, given the modest margins on CREMA-D and MVSA.
- Add one baseline: same cross-attention architecture + MI regularization + joint training (no alternation, no freezing). This cleanly tests whether the frozen-classifier alternating regime is necessary beyond better initialization.
- Plot CCAT's modality contribution curves alongside MLA's from Figure 1 to directly validate the central claim about reducing classifier bias.
- Provide sensitivity analysis for λ (the regularization coefficient) given its conceptual importance to the "unbiased classifier" claim.

## Score and Decision

**Calibration:** I compared against:
- **MLA** (alternating unimodal adaptation, same topic): scores 3-5, rejected — concerns about limited novelty, scalability, missing baselines
- **MMPareto** (multimodal imbalance, gradient conflict): scores 3-5 with one 8, rejected — concerns about theoretical rigor, limited experiments, computation omitted
- **BalancedMLA** (similar domain): scores 1-5, rejected — limited novelty, limited evaluation
- **DMI** (information-theoretic multimodal, strong theory): scores 5-6, rejected — marginal improvements, limited scope
- **AVSAC** (bidirectional audio-visual): scores 3-6, rejected — novelty concerns about bidirectional fusion

CCAT is stronger than MLA/BalancedMLA in empirical results (+6.76% on KS is notable) and has a more complete ablation. It is comparable in novelty to MMPareto (combining existing ideas into a coherent framework with moderate theoretical motivation). However, it shares similar weaknesses: informal theory that overclaims, limited dataset scope, and insufficient ablation isolation. The theoretical contribution is oversold relative to its depth.

The paper sits around the range of these comparable works (scores 3-6). The substantial empirical gain on KS is a positive differentiator, but the informal theory, MI circularity concern, ablation ambiguity, and lack of statistical significance testing are material weaknesses. I place this slightly above the weakest comparable papers but below the acceptance threshold.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>