Now I have enough information to write the consolidated meta-review. Let me verify the OOD claim language specifically.Now I have enough to write the consolidated review. Let me synthesize carefully.

---

## Summary

This paper proposes a representational geometry framework for studying feature learning in neural networks through the lens of task-relevant manifold untangling. Using manifold capacity (MCT) from prior neuroscience work, the authors: (1) provide theoretical and empirical justification that capacity quantifies the lazy/rich learning dichotomy; (2) use auxiliary geometric measures (radius, dimension, alignment variants) to identify learning stages and strategies; and (3) apply the framework to structural inductive biases in RNNs and geometric correlates of OOD generalization failure.

---

## Strengths

- **Representation-based alternative to weight/kernel metrics for neuroscience applicability.** The paper correctly identifies a genuine gap: weight matrices and NTK are inaccessible in real neural circuit recordings, yet most lazy/rich analyses rely on them. Grounding the analysis in neural representations is a meaningful design choice that opens the framework to neuroscience settings — as demonstrated in Section 5.1 where the RNN structural bias analysis is performed from hidden-state representations rather than weight matrices.

- **Theorem 1 provides a rigorous monotonicity result in a non-trivial classification setting.** The paper extends Ba et al. (2022) from regression to classification — requiring the additional step of analyzing the Gaussian equivalent margin via Montanari et al. (2019) — and proves that capacity monotonically tracks the learning rate η (richness proxy) and is linked to prediction accuracy via an increasing function. This is a technically non-trivial extension even within the acknowledged scope of one gradient step in the proportional asymptotic limit.

- **Using the inverse scale factor $\bar{\eta}$ as control is a legitimate independent ground truth.** The harsh critic questions whether capacity is merely validated against a "regime-inducing hyperparameter." But $\bar{\eta}$ from Chizat et al. (2019) is precisely the accepted independent control in the lazy/rich literature — it directly sets the relative magnitude of hidden vs. readout learning rates. Validating capacity's monotone tracking of $\bar{\eta}$ is a meaningful empirical confirmation, not a circular one.

- **The wealthy/poor vs. rich/lazy conceptual decomposition is a genuine contribution.** By distinguishing initial representational quality (wealthy/poor, driven by initialization) from training dynamics (rich/lazy, driven by the learning process), the paper reveals that final capacity can be the same for networks following different trajectories — a genuinely novel observation in Section 5.1 that could not be seen with weight-based or NTK-based diagnostics.

- **Geometric trajectories in radius-dimension space reveal qualitatively distinct behaviors.** The contour plot analysis (Fig. 4a,b) — showing that some regimes compress radius while preserving dimension, while others trade radius for dimension — is a specific, concrete finding that is invisible to scalar metrics like accuracy or weight-change norms. This is a substantive insight, not a generic observation.

---

## Weaknesses

### Fatal
*None. The paper's claims, though overclaimed in places, are not fundamentally wrong.*

### Major

- **Theory-experiment gap: Theorem 1 covers one gradient step; experiments span full training.** The paper explicitly acknowledges in footnote 6 that "the key Gaussian equivalence step might not hold for more steps." Yet the entire empirical program — VGG-11 trained for $10^5$ epochs, RNNs trained for 10,000 iterations, learning stages observed over full training — relies on capacity being a principled measure throughout training. Theorem 1 provides only a one-step justification. This is not an unfair criticism invented by reviewers: the paper's own footnote 6 confirms the limitation. Without either (a) extending the theory or (b) providing explicit empirical evidence that the one-step monotonicity result actually generalizes to multi-step training (e.g., by checking whether capacity remains monotone in $\bar{\eta}$ at each checkpoint), the theoretical-to-empirical extrapolation is unjustified. This is the paper's most significant gap.

- **OOD section uses "explain" where only "correlate" is warranted.** Figure 6's caption states that "the expansion of manifold radius and the increase of center-axis alignment *explain* the failure of OOD generalization in the ultra-rich regime." The setup is: CIFAR-10 pretrained model → linear probe trained on CIFAR-100 → CIFAR-100 test accuracy drops in ultra-rich regime. The geometric changes and accuracy drop co-occur but no intervention isolates whether the geometric quantities are causal, mediating, or simply coincident with overfitting. The paper's own conclusion defers to "future direction," revealing the authors are aware the evidence is exploratory. The word "explain" in the figure caption, abstract, and Section 1.1 directly overstates what the evidence supports and should be changed to "correlate with."

### Minor

- **Learning stages (Section 4.2, Fig. 4c) are observed in a single VGG-11 run with no reproducibility evidence.** The four-stage segmentation (clustering, structuring, separating, stabilizing) is presented from one training run on CIFAR-10. There is no change-point analysis, no multi-seed validation, and no evidence that these stages appear in other architectures. The caption itself only claims "at least four stages," suggesting the authors are aware this is descriptive. Presented as a case study this is reasonable; presented as a discovered property of feature learning dynamics it requires more support.

- **Analyses are restricted to last-layer representations only.** Stated explicitly in Section 2.3: "All analyses were performed on the test data representations in the last layer." Since rich feature learning is a hierarchical process that unfolds across layers, restricting to the last layer may miss or distort the phenomena of interest. This limitation is not discussed in the main text.

- **The comparison to "conventional measures" lacks a rigorous criterion for "better."** Section 3.2 and Fig. 3 show that capacity tracks $\bar{\eta}$ while accuracy, weight changes, NTK alignment, and representation-label alignment do not consistently distinguish settings. However, the comparison is qualitative and visually argued. No rank correlation with $\bar{\eta}$, no statistical testing across seeds, and no discriminative accuracy metric is provided.

- **RNN application section lacks variance across seeds.** Section 5.1 shows final-epoch capacity values and geometric measures for RNNs with different initial weight ranks, but no error bars or seed variability are reported. RNNs trained with SGD on cognitive tasks are notoriously sensitive to initialization and random seed; without reproducibility evidence, the precision of the geometric comparisons in Fig. 5d is unclear.

### Trivial

- **"Wealthy vs. poor" terminology partially overlaps with "rich vs. lazy"** causing some readability friction. The conceptual distinction is meaningful but the linguistic proximity (wealthy/rich) is confusing at first read.

---

## Nice-to-Haves

- A multi-step extension of the theory, even informal: show empirically that capacity remains monotone in $\bar{\eta}$ at each training checkpoint and relate this to why the one-step result generalizes.
- Random-label or shuffled-label controls to verify that capacity increases specifically track task-relevant (not spurious) representation changes.
- Layer-by-layer capacity evolution plots to test whether the findings generalize beyond the last layer.
- Controlled intervention for the OOD claim: regularize manifold radius during training (e.g., a radius penalty) and show whether OOD performance is preserved in the ultra-rich regime.
- Multi-seed validation of the four learning stages across architectures (e.g., ResNet-18, a smaller transformer).

---

## Removed Points

*These points are flagged as removed — treat them with caution.*

- **"Capacity is merely validating against a regime-inducing hyperparameter rather than an independent measure of feature learning"** (Harsh Critic): As verified above, $\bar{\eta}$ from Chizat et al. (2019) is the accepted, well-characterized independent control for the lazy/rich regime. Validating capacity against it is standard and legitimate, not circular.

- **"The 'distinct learning strategies' are continuous trajectory variations rather than discrete modes"** (Harsh Critic): The paper never claims the strategies are discrete or statistically clustered — it presents trajectory visualizations in radius-dimension space and interprets them qualitatively. The harsh critic's objection is a strawman.

- **"Manifold geometry may simply co-vary with overfitting rather than revealing a novel failure mode"** (Harsh Critic, OOD section): This is a legitimate theoretical possibility, but it is not a disproof of the paper's findings. The finding that radius expansion co-occurs with OOD failure remains informative regardless of whether overfitting is the underlying cause. This does not warrant removal of the OOD section, only the "explain" language.

- **"No comparison with CCA, SVCCA, or CKA on class-conditional structure"** (Spark): The paper compares against the standard measures used in the lazy/rich literature (NTK alignment, representation-label alignment, weight changes). CCA/SVCCA are different-purpose measures from the representational similarity literature and are not the relevant competitors here.

- **"The neuroscience application lacks real neural data"** (Human Finder): The paper explicitly scopes to "RNNs trained on common neuroscience tasks" as a way to study structural inductive biases from representations — not as a claim about fitting real neural recordings. The motivation for representation-based methods is the inaccessibility of weight data in neuroscience, not a claim that the authors have neural data. The framing is forward-looking, and the application section is clearly labeled accordingly.

- **"Novel concerns about novelty since MCT is adopted from prior work"** (Human Finder): Applying an existing methodological framework to a new scientific question is a valid contribution. The specific combination — manifold capacity × lazy/rich regime × training dynamics × geometric strategies — is not in prior work. This is not a novelty gap.

- **Scalability concern for Algorithm 1** (Neutral Reviewer): The paper explicitly scopes to CIFAR-scale and neuroscience-scale problems and does not claim to scale to ImageNet or LLMs. Demanding scalability analysis for a framework paper that never claims to address large-scale models is out of scope.

- **Pure formatting/style nitpicks** from all reviewers.

---

## Novel Insights

The most genuinely novel conceptual insight in the paper is the **wealthy/poor vs. rich/lazy decomposition**: the observation that final capacity can converge to the same value across different network structures, while the geometric realizations (radius, dimension, alignment) remain systematically distinct. This means "how much" feature learning occurs (measured by final capacity) and "how" it occurs (measured by geometric trajectory) are separable quantities — a finding that cannot be recovered from weight-norm or NTK-based diagnostics. This decomposition has direct practical implications for neuroscience: observing the same representational quality at the end of learning does not imply the same learning mechanism or structural bias was operative. The identification of this level of geometric detail below the capacity scalar is the paper's most distinctive contribution.

---

## Evaluation on Key Axes

- **Novelty**: Moderate-to-good. Applying MCT to the lazy/rich regime is new; the wealthy/poor decomposition and geometric trajectory analysis are original observations. The theorem is a non-trivial extension of Ba et al. (2022). The OOD and RNN applications are genuinely new territory for this framework.
- **Technical soundness**: Moderate. Theorem 1 is technically sound but narrowly scoped. Empirical analyses are reasonable but the theory-experiment gap is real and the OOD language overclaims.
- **Empirical support**: Moderate. CIFAR-scale experiments with standard architectures; results are consistent across the tested settings but limited in scope (one architecture per experiment, single seeds for key visualizations).
- **Significance**: Moderate. The framework offers a genuinely different lens on feature learning with clear neuroscience applicability. The OOD correlate result opens an interesting direction, though it is premature to call it an explanation.
- **Clarity**: Good. The paper is well-organized, the progression from theory → mechanisms → applications is logical, and Fig. 1 is an effective overview. The "wealthy/poor" terminology is slightly confusing.

---

## Score and Decision

**Calibration against past reviews:**

- **mMPaQzgzAN (6.5, Accept):** JumpReLU SAE paper. Strong theoretical contribution (KDE-STE interpretation), broad empirical validation on Gemma 2 9B across 9 configurations, practically impactful. Theory is sound and directly supported by experiments.
- **D0Cdljktp2 (4.0, Reject):** Memformer paper. Proof sketches rather than proofs, toy-scale experiments (d=5, n=20), training-data comparison in a key figure.

This paper sits **between these two**. It has a real theorem (unlike the Memformer) but a significant theory-experiment gap that the JumpReLU paper avoids. Its empirical scope is substantially more serious than d=5 toy tasks, but considerably narrower than the JumpReLU paper's Gemma 2 9B multi-site evaluation. The OOD overclaiming and single-seed learning stages push it toward the lower end of acceptance, while the genuine geometric insights and the principled theoretical grounding push it above the midpoint.

Placement: **above D0Cdljktp2 (4.0), below mMPaQzgzAN (6.5)**, scoring in the borderline range.

The paper makes a genuine and coherent contribution to understanding feature learning through representation geometry. Its weaknesses — the theory-experiment gap being primary — are real but not fatal, and the interesting geometric decompositions are substantive enough to warrant publication with revisions. However, the overclaiming in the OOD section and the single-run nature of key observations mean it is not ready as-is.

**Score: 5.5 — Borderline Accept**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>