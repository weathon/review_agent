Now let me look at the calibration papers referenced in the human finder review.Now I have sufficient grounding in the paper's actual content and the calibration space. Let me produce the final consolidated review.

---

## Summary

This paper proposes a framework for studying feature learning in neural networks through the lens of representational geometry. The core idea is to use *manifold capacity*—a representation-based measure of task-relevant manifold separability ("packability")—to quantify the degree of richness in feature learning, supplemented by geometric measures (radius, dimension, center/axis alignment) that delineate distinct learning strategies and stages. The framework is applied to both feedforward DNNs (CIFAR-10/100) and RNNs trained on neuroscience tasks (perceptual decision making), with an additional section on out-of-distribution generalization.

---

## Strengths

- **Novel bridging contribution.** Applying manifold capacity theory (originally from computational neuroscience) to the lazy-versus-rich feature learning question is a genuine and well-motivated innovation. The paper correctly identifies that weight/kernel-centric measures are inaccessible in biological settings and that existing measures collapse a richer phenomenology into a binary dichotomy.

- **Non-trivial theoretical extension.** Theorem 1 extends the prior results of Ba et al. (2022) from a regression setting to a classification setting via margin analysis, requiring separate technical machinery. The connection between one-step gradient updates and monotone capacity growth is analytically established, not merely assumed.

- **Geometric decomposition provides actionable mechanistic descriptors.** The decomposition of capacity dynamics into radius, dimension, and alignment changes (Figure 4) goes beyond scalar tracking. The identification of distinct learning strategies (radius compression vs. dimension compression) and four qualitative learning stages in VGG-11 training is concrete and interpretable.

- **Breadth of empirical coverage.** The paper covers 2-layer synthetic networks, VGG-11/ResNet-18 on CIFAR-10, and RNNs on neuroscience tasks. The finding that networks with different initial weight rank converge to the same final capacity but via geometrically distinct trajectories (Section 5.1) is a non-trivial observation invisible to weight-change measures.

- **Representation-based analysis is genuinely superior for neuroscience.** The motivation for representation-over-weights analysis in biological settings is valid, and the RNN application demonstrates that the framework can recover the findings of Liu et al. (2024) while adding geometric subtlety.

---

## Weaknesses

### Fatal
*None identified. The paper makes a real contribution and its claims, while overstated in places, are grounded in meaningful evidence.*

### Major

- **Narrow theorem, broad empirical claims.** Theorem 1 applies *only* to a one-step gradient descent update in a 2-layer teacher-student model with fixed readout weights and proportional asymptotics. The paper itself acknowledges this (footnote 6): "the key Gaussian equivalence step might not hold for more steps." Yet the abstract states the paper shows "both theoretically and empirically that task-relevant manifolds untangle during rich learning, and that manifold capacity quantifies the degree of richness"—a claim whose theoretical component is far narrower than that phrasing suggests. The gap between the one-step 2-layer theory and the multi-step VGG-11/ResNet-18/RNN experiments is substantial and unaddressed.

- **The OOD generalization section is conceptually misframed.** Section 5.2 explicitly states "we focus on the case where the label set in D_test is different from that in D_train," and the experiment evaluates a linear probe trained on CIFAR-100 *labels* after pretraining on CIFAR-10 *labels*. This is zero-shot cross-task transfer, not OOD generalization in the standard sense (same task, different input distribution). A drop in CIFAR-100 linear-probe accuracy in the ultra-rich regime could simply reflect overspecialization of features to CIFAR-10 category boundaries, which is a categorically different phenomenon from OOD generalization failure. The abstract's claim that the framework provides "geometric insights into out-of-distribution generalization" is thus overstated for what the experiment actually tests. The paper acknowledges this as a "future direction" only in passing.

### Minor

- **"Capacity is better than conventional measures" claim is visual and anecdotal.** Section 3.2 and Figure 3 conclude that capacity is superior to weight changes, NTK alignment, and representation-label alignment at distinguishing richness regimes. However, there is no quantitative evaluation criterion, no rank-correlation against an independently validated target, and no variance estimates across seeds. The comparison is based on visual ordering in 2-layer synthetic experiments only—a narrow basis for a headline claim (Section 1.1: "capacity is better than conventional measures").

- **All analyses restricted to last-layer representations.** Section 2.3 states this explicitly. For a framework claiming to quantify "feature learning"—which by definition concerns the internal transformation of representations—analysing only the final layer is a significant interpretive limitation. The last layer is closest to the label space and is the layer where the least autonomous feature restructuring occurs. The paper does not discuss whether the findings change at intermediate layers.

- **No statistical uncertainty quantification anywhere.** No error bars, no confidence intervals, no reports of the number of random seeds appear in any figure. The learning stages in Figure 4c, the capacity trajectories in Figures 2b and 5c, and the geometric comparisons in Figure 5d are all single-run or unreported-averaging. For results about training dynamics that can be seed-sensitive (e.g., stage transitions), this omission weakens the claims.

- **Disconnect between feedforward theory and RNN applications.** Theorem 1 is derived for a feedforward 2-layer network. Section 5.1 applies the framework to RNNs, which have fundamentally different training dynamics (recurrent state, temporal structure, BPTT). The paper offers no argument for why conclusions drawn in feedforward theory should transfer to the recurrent setting.

- **RNN finding is correlational; dynamic regime confound unaddressed.** Varying the initial recurrent weight rank changes not only the structural bias but potentially the initial dynamic regime (stable vs. chaotic). The paper does not control for or discuss this confound, so the geometric differences observed across initial weight ranks could reflect differences in learning stability rather than structural inductive biases per se.

### Trivial

- **Learning stage labels are post-hoc.** The four stages in Figure 4c ("clustering," "structuring," "separating," "stabilizing") are informal labels attached to visual inspection of smoothed heatmaps from a single VGG-11 run. No formal stage-detection criterion is provided, and no replication across architectures or seeds is shown. This undermines confidence that the stages reflect a robust phenomenon.

- **Row-normalized heatmaps in Figure 6c** make cross-dataset and cross-regime effect-size comparisons difficult, though this is a visualization choice rather than a substantive flaw.

---

## Nice-to-Haves

- **Intermediate layer analysis.** Showing that capacity tracks feature learning in earlier layers (not just the last) would substantially strengthen the claim that the framework captures "feature learning" broadly, not just last-layer label alignment.

- **Formal stage detection or cross-architecture replication.** Even a simple consistency check of the four learning stages across ResNet-18 or across seeds would increase confidence in this finding.

- **Comparison to representation-based feature learning measures beyond CKA.** SVCCA or spectral methods would make the "capacity is better" claim more compelling against representation-level baselines, not just weight-level ones.

- **Computational cost discussion.** Manifold capacity estimation via quadratic programming (Algorithm 1) may be expensive for large $P$ or $N$. A brief scaling analysis or practical guidance would be helpful for practitioners.

- **Failure modes.** A single case where capacity does *not* track feature learning (e.g., severely non-convex manifolds, very few samples per class) would increase trust in the framework by bounding its scope of validity.

- **Extension to transformers or modern architectures.** VGG-11 and ResNet-18 are somewhat dated; ViTs or MLP-Mixers have different feature learning dynamics and would be natural tests of generality.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Harsh Critic — "comparisons to baselines are unfair":** The critique that weight change, NTK alignment, and accuracy are "weakly matched competitors" is partly fair but partly scope creep. These are exactly the measures used in the lazy/rich literature that the paper targets; the comparison is appropriate to the paper's stated purpose. The concern about lack of statistical rigor is retained separately.

- **Harsh Critic / Neutral Reviewer — claims about "overclaiming without foundation":** Several such criticisms reduce to the narrow-theorem point already captured in the Major Weaknesses, and the remaining framing accuses the paper of misrepresentation when the paper is in fact transparent about its scope (e.g., footnote 6, the "future direction" language in Section 5.2).

- **Neutral Reviewer — garbled / illegible text:** A large portion of the neutral review (from "Clarity: Generally good..." onward) consists of corrupted, unintelligible text and was entirely discarded.

- **Human Finder — "insufficient exploration of activation function effects":** The paper does not study activation function effects, which is a reasonable omission given its focus on lazy/rich dynamics. Demanding it would be scope creep; retained only as a nice-to-have.

---

## Novel Insights

The most genuinely novel observation in this work is the distinction between *learning strategies*—the trade-off between radius compression and dimension compression as a function of regime richness—and *learning stages*—the non-monotone geometric trajectory during training even when accuracy has saturated. The finding in Section 5.1 that RNNs with different initial ranks converge to the same final capacity via geometrically distinct paths is not predictable from weight-change measures alone and opens a new representational lens on structural inductive biases in recurrent circuits. The analytical connection between the geometric approximation $\alpha_\text{mf} \approx (1 + R_\text{mf}^{-2})/D_\text{mf}$ and the trade-off contours in Figure 4 is an elegant mechanistic descriptor that could become a useful diagnostic tool in both ML and neuroscience.

---

## Suggestions

1. **Reframe the OOD section** as "zero-shot cross-task transfer" or "cross-distribution linear probing," and qualify the abstract claim accordingly. This is not a scientific flaw—the experiment is interesting—but the framing currently misrepresents the phenomenon being studied.

2. **Add error bars / report number of seeds** for all training dynamics figures, especially Figure 4c (learning stages) and Figure 5c/d (RNN capacity trajectories).

3. **Soften "capacity is better than conventional measures"** to "capacity provides complementary and, in these settings, more sensitive information than weight-based measures." Run a rank-correlation against the scale parameter $\bar{\eta}$ as a quantitative criterion.

4. **One intermediate-layer analysis** (e.g., penultimate vs. last layer in VGG-11) would directly address the scope limitation of last-layer-only analysis.

5. **Acknowledge the feedforward-to-recurrent gap** in Section 5.1 with a brief discussion of what additional assumptions or evidence would be needed to apply the Theorem 1 reasoning to RNNs.

---

## Score and Decision

**Calibration comparison:**
- *slSmYGc8ee* ("How connectivity structure shapes rich/lazy learning in RNNs"): Accept, scores 8/6/8/5 ≈ 6.75. Closer experimental alignment between theory and setting, but narrower empirical scope.
- *vt5mnLVIVo* ("Grokking as lazy-to-rich transition"): Accept, scores 8/8/3/5 ≈ 6. Also narrow theorem + broader empirical claim, arguably cleaner single-phenomenon focus.
- *k9t8dQ30kU* ("Task structure + nonlinearity → representational geometry"): Accept, scores 5/6/8/8 ≈ 6.75. Purely empirical, broader architecture coverage.
- *KJFyOwAnLR* ("Emergent geometry in neural representations"): Reject, scores 3/6/3 ≈ 4. Overclaims without adequate experimental controls or theory.
- *ByCV9xWfNK* ("Intermediate layers for OOD"): Accept, scores 6/6/6/8/6/6 ≈ 6.3. Careful execution, moderate claim scope.

This paper sits between the accepted calibration papers and the rejected one. It is clearly above KJFyOwAnLR (which had no theoretical anchor and broader claims with fewer experimental controls). It falls somewhat short of the cleanest accepted papers (vt5mnLVIVo, slSmYGc8ee) due to the OOD framing issue, the lack of statistical rigor, and the overstated superiority claim. The core contribution—a principled, representation-based, theoretically grounded framework for lazy/rich regime characterization with mechanistic geometric descriptors—is real and useful to the community. The weaknesses are correctable without changing the paper's fundamental findings.

**Assessment axes:**
- *Originality*: Good. Bringing manifold capacity to the lazy/rich debate is novel.
- *Importance of research question*: High. Feature learning quantification matters for both ML and neuroscience.
- *Claims vs. support*: Partially mismatched. Theory is narrower than claimed; OOD framing is off.
- *Soundness of experiments*: Moderate. Visually compelling but lacks statistical rigor.
- *Clarity*: Good in the main text (aside from the conceptual misframing of OOD).
- *Value to community*: Moderate-to-high, particularly for neuroscience community.

**Final Score: 5.5** — borderline, marginally below acceptance in its current form, but addressable with relatively targeted revisions (OOD reframing, error bars, softening superiority claim).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>