## Summary
This paper proposes that grokking is prevented by numerical instability (Softmax Collapse - SC) caused by optimization dynamics (Naïve Loss Minimization - NLM). It introduces StableMax, a numerically stable activation that enables grokking without weight decay, and ⊥Grad, an optimizer that removes the NLM component to eliminate the generalization delay. The paper provides empirical evidence across MLPs and Transformers on modular arithmetic tasks.

## Strengths
- **Concrete mechanistic explanation for training failure:** The identification of Softmax Collapse as floating-point absorption errors (Definition 3, Sec. 3.1) provides a specific, testable mechanism for why learning stops. Figure 2 demonstrates that test accuracy plateaus precisely when SC begins, with earlier collapse under lower precision (float16 vs float32 vs float64), distinguishing this from purely statistical explanations.
- **StableMax enables grokking without weight decay:** Figure 4 (left) shows models reaching 100% test accuracy on modular addition (40% split) using StCE loss without regularization, whereas standard Softmax fails completely in the identical setting (Fig. 2a). This is a practical, computationally cheap intervention validated across multiple tasks.
- **Empirical evidence of gradient alignment with NLM direction:** Figure 5 shows cosine similarity between weights and gradients spiking to ~0.9 across MLPs and Transformers immediately after 100% training accuracy, providing concrete empirical support for the NLM hypothesis across architectures.
- **Unified explanation for existing grokking interventions:** Section 5.2 offers a cohesive framework explaining why weight decay, MSE loss on shallow networks, and other methods work—by mitigating NLM and avoiding SC. This synthesizes prior observations under a single mechanism.

## Weaknesses

### Fatal
None

### Major
- **Theory-experiment mismatch for Transformers:** The core theoretical argument for NLM relies on positive homogeneity (Definition 6, Sec. 4.2), proving that weight scaling is an NLM direction for homogeneous networks. However, the Transformer experiments (Fig. 5c, Fig. 6a) use architectures that are not positively homogeneous due to bias terms, layer norms, and skip connections. The paper acknowledges this ("approximately homogeneous," Sec. 4.2) but relies solely on empirical cosine similarity rather than theoretical justification for these cases. Since the Transformer results are central to claiming NLM is a pervasive cause of grokking, the disconnect between the homogeneity-dependent theory and non-homogeneous experiments undermines the theoretical contribution. The Limitations section (Sec. 7) notes this but does not resolve it.
- **Incomplete mechanism for ⊥Grad acceleration:** The paper claims NLM causes the delay in generalization and ⊥Grad removes this delay (Sec. 5). However, the explanation for why removing the NLM component accelerates learning is incomplete. If the gradient is dominated by NLM (cosine similarity ≈ 0.9, Fig. 5), the orthogonal component is small. Simply projecting out the NLM component should prevent weight scaling but not necessarily make learning faster. The paper hints at an interaction with adaptive optimizer mechanics (Sec. 5.2 discusses weight decay balance) but does not explicitly analyze or verify why ⊥Grad leads to faster generalization (Fig. 6) rather than just preventing SC. Without this, the claim that NLM causes the delay (rather than co-occurring) is not fully supported.

### Minor
- **"Without regularization" claim is overstated:** The abstract and introduction claim grokking is achieved "without regularization" using StableMax. However, StableMax (Definition 4) fundamentally alters the loss landscape by replacing the exponential penalty with a linear/rational one, which functionally acts as an implicit regularizer on weight norm and confidence. While technically "no weight decay," the claim that regularization is "not necessary" (Sec. 3.3) is misleading without discussing how StableMax's inductive bias compares to weight decay. The evidence shows StableMax enables grokking but does not establish it does so without imposing comparable constraints on the solution space.
- **Limited robustness analysis:** Grokking is known to be initialization-sensitive, but the key results (Fig. 4, Fig. 6) appear to be single-run plots without error bars or multiple seeds. This does not demonstrate the robustness of StableMax or ⊥Grad across different initializations, which is important for establishing these interventions as reliable solutions.

### Trivial
None

## Nice-to-Haves
- Plot the evolution of Adam's second-moment estimate with and without ⊥Grad to verify the hypothesis that NLM inflates the denominator, damping the useful gradient signal.
- Show the distribution of logits during training for Softmax vs. StableMax to visually demonstrate how StableMax prevents extreme values leading to SC.
- Compare final weight norms and solution properties of models trained with StableMax (no WD) vs. Softmax (with WD) to substantiate claims about inductive bias.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Harsh Critic: "Theoretical Framework Does Not Match Experimental Setup"** — This is a legitimate concern and is retained in Major weaknesses.
- **Harsh Critic: "Mechanism of ⊥Grad Acceleration is Under-Specified"** — This is a legitimate concern and is retained in Major weaknesses.
- **Harsh Critic: "'Without Regularization' Claim is Semantically Contested"** — This is a legitimate concern and is retained in Minor weaknesses.
- **Harsh Critic: "Learning rate and batch size details are sparse"** — Removed per hard rules about reproducibility nitpicks; the paper mentions "full batch setting" (Sec. 2.2) and specific LR values would be implementation details.
- **Harsh Critic: "Fig 7 lacks contour density or confidence regions"** — Removed as a presentation nitpick; the figure shows trajectories clearly.
- **Harsh Critic: "Seed Variance Analysis" missing** — Retained as Minor weakness (limited robustness analysis).
- **Harsh Critic: "Non-Homogeneous NLM Theory" needed** — This is the theory-experiment mismatch, retained in Major.
- **Harsh Critic: "Orthogonal Signal Magnitude" analysis** — Related to ⊥Grad mechanism, covered in Major.
- **Harsh Critic: "Logit Distribution" visualization** — Moved to Nice-to-Have.
- **Harsh Critic: "⊥Grad Trajectory" for ⊥SGD** — The paper does include ⊥SGD results (Fig. 7); this criticism misreads the figure.
- **Harsh Critic: "Large-Scale Validation"** — Scope creep; the paper explicitly focuses on grokking tasks. Moved to Nice-to-Have.
- **Harsh Critic: "Theoretical Extension for LayerNorm and Bias"** — This is the theory-experiment mismatch, covered in Major.
- **Strength Finder: "Empirical validation that floating-point precision limits grokking"** — Retained as Strength 1.
- **Strength Finder: "StableMax enables grokking without regularization"** — Retained as Strength 2.
- **Strength Finder: "⊥Grad eliminates the generalization delay"** — Retained as evidence in Strength 2.
- **Strength Finder: "Quantitative evidence of gradient alignment"** — Retained as Strength 3.
- **Strength Finder: "Unified explanation for disparate grokking interventions"** — Retained as Strength 4.

## Novel Insights
The paper's identification of Softmax Collapse as a floating-point absorption error (rather than overflow) is a genuinely novel observation that corrects a blind spot in the grokking literature. Most prior work on numerical stability in deep learning focuses on overflow in exponentials, but this paper shows that absorption errors in the sum—when the correct logit dominates so completely that smaller terms vanish—are what stop learning. This distinction is important because it explains why the standard "numerically stable" Softmax formulation (subtracting max) does not prevent the problem. The connection between NLM dynamics and SC provides a mechanistic bridge between optimization behavior and numerical failure that was previously missing.

## Suggestions
- Temper the theoretical claims about NLM for non-homogeneous models. The empirical evidence (Fig. 5) is compelling, but the paper should frame the Transformer results as empirical observations rather than theoretically guaranteed phenomena. Consider adding a proposition or discussion about why quasi-homogeneous models might exhibit similar behavior without claiming proof.
- Clarify the ⊥Grad acceleration mechanism. Add analysis of how removing the NLM component affects the effective gradient magnitude and optimizer dynamics (particularly for Adam). A plot showing the orthogonal gradient component magnitude over time would strengthen the explanation.
- Qualify the "without regularization" claim. Acknowledge that StableMax introduces its own inductive bias and discuss how this compares to weight decay. The claim should be that weight decay is not necessary, not that regularization in any form is unnecessary.
- Add multiple seeds or error bars for key results (Fig. 4, Fig. 6) to demonstrate robustness across initializations.

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| /home/wg25r/review_agent/human_reviews_2026/wCnHeql3ow.md (Egalitarian GD for grokking) | 6.00 | Similar empirical grokking intervention with theory; this paper has stronger numerical stability insight but weaker theoretical guarantees for Transformers. |
| /home/wg25r/review_agent/human_reviews_2026/CXlsqTAf1E.md (Preconditioned GD for grokking) | 5.00 | Similar optimizer intervention; this paper has more concrete mechanistic explanation (SC). |
| /home/wg25r/review_agent/human_reviews_2026/blfwRondjY.md (Grokking in LLM Pretraining) | 5.50 | Empirical grokking study; this paper has more novel mechanism identification. |
| /home/wg25r/review_agent/human_reviews_2026/sLX5P7FTfT.md (Neural Collapse for grokking) | 4.67 | Theory-experiment mismatch similar to this paper; this paper has cleaner empirical validation. |
| /home/wg25r/review_agent/human_reviews_2026/Y5kuPDwLJB.md (Geometry of Grokking) | 2.50 | Theoretical grokking paper rejected for weak empirical support; this paper has stronger experiments. |
| /home/wg25r/review_agent/human_reviews_2026/0jHyEKHDyx.md (Flash Attention numerical stability) | 6.50 | Numerical stability paper with mechanistic explanation; this paper is similar in approach but less comprehensive validation. |
| /home/wg25r/review_agent/human_reviews_2026/3kTb7Iortf.md (Theory overclaims) | 2.50 | Paper rejected for theory-experiment mismatch; this paper has better empirical grounding. |

**Scoring rationale:** This paper sits between the 5.0-6.0 range anchors. It has stronger empirical validation than the rejected theory papers (Y5kuPDwLJB, 3kTb7Iortf) and provides a more concrete mechanistic explanation than CXlsqTAf1E (5.0). However, it has notable theory-experiment gaps for Transformers that prevent it from reaching the 6.0+ range of wCnHeql3ow or 0jHyEKHDyx. The SC identification is genuinely novel and valuable (comparable to the Flash Attention paper's contribution), but the NLM theory is less rigorous. The paper's empirical contributions (StableMax, gradient alignment evidence) are solid and practically useful, warranting acceptance despite the theoretical limitations.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>