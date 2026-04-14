=== CALIBRATION EXAMPLE 39 ===

# Final Consolidated Review
## Summary

AKOrN (Artificial Kuramoto Oscillatory Neurons) replaces standard threshold units in deep networks with vector-valued Kuramoto oscillator dynamics, implemented via convolutional and self-attentive connectivity. Oscillators are represented as unit-norm vectors on a hypersphere and updated iteratively according to a generalized Kuramoto differential equation, with a norm-based readout that achieves phase invariance. The paper demonstrates that this single architectural primitive improves performance across three qualitatively distinct settings: unsupervised object discovery (first synchrony-based model competitive with slot-based approaches and scaling to natural images), combinatorial reasoning (Sudoku, with emergent test-time extension and energy-based voting), and adversarial robustness/calibration (non-zero adversarial accuracy and near-perfect calibration without adversarial training).

---

## Strengths

- **First synchrony-based model to scale to natural image object discovery without pre-trained SSL features:** AKOrN trained from scratch on ImageNet outperforms DINO, MoCoV3, and MAE on PascalVOC (52.0 / 60.3 MBO_i/c vs. DINO's 47.2 / 53.5), and is competitive with DINO+slot-based hybrids. Prior synchrony models required large pre-trained SSL backbones even for limited performance. This is a concrete, benchmark-verified advance over the prior art.

- **First distributed-representation model competitive with slot-based approaches on complex CLEVRTex:** AKOrN achieves 88.5 FG-ARI and generalizes strongly on OOD (87.7) and CAMO (77.0) variants, which slot-based models largely fail to report on. The claim of competitiveness with slot-based approaches — rather than superiority — is accurate and the advance over prior synchrony models (which could not scale to CLEVRTex) is substantial.

- **Emergent adversarial robustness and calibration without adversarial training:** AKOrN achieves non-zero AutoAttack+EoT accuracy (58.91% / 51.56%) where ResNet-18 and ViT both collapse to 0.00%. Simultaneously, it achieves ECE of 1.3–1.4% without any calibration procedure, with near-perfect confidence-accuracy correlation (Figure 9). Achieving both simultaneously as emergent properties — without any specialized training — is a striking and unexpected empirical finding.

- **Test-time step extension as OOD generalization mechanism:** Increasing Kuramoto steps at inference (from 16 to 128) improves OOD Sudoku accuracy from ~17% to ~52% while the analogous iterative self-attention baseline (TrSA) peaks early and degrades. This is a genuinely novel and practically valuable emergent property distinguishing AKOrN's dynamical inductive bias from parameter-shared transformers.

- **Cross-task empirical breadth from a single architectural primitive:** Demonstrating that a single neuron-level change meaningfully improves three qualitatively different task families — feature binding, combinatorial reasoning, and adversarial robustness — is a rare and significant result that strengthens the argument for the general utility of the oscillatory inductive bias.

---

## Weaknesses

### Fatal
None. The core empirical claims are well-supported across multiple task domains and the main concerns are revision-level.

### Major

- **Theoretical-empirical consistency gap around the energy function:** The Lyapunov guarantee (Eq. 3) requires scalar symmetric J (J_ij = J_ji ∈ ℝ), shared Ω, and Ωc_i = 0 (Section 3). Yet the paper explicitly abandons all these constraints for better empirical performance (Section 3, last paragraph). The paper acknowledges this for the attentive case ("it is unclear whether the energy defined in Eq. (3) is proper"), but the convolutional case with asymmetric J is equally unguaranteed. The energy function then becomes the central mechanism for energy-based voting (Section 6.2) and is used to explain correctness estimation and calibration (Section 6.3). The justification for using energy as a confidence metric is empirical ("the energy value decreases relatively stably"), not theoretical. This is not a fatal flaw — the empirical observations are real — but the paper should more explicitly separate the theoretical properties of the symmetric idealization from the practical design, and ground all energy-based applications in empirical evidence rather than theoretical justification from Eq. (3). The current framing may mislead readers into believing the energy properties are theoretically guaranteed in the deployed model when they are not.

- **Computational cost entirely unquantified:** AKOrN requires T iterative Kuramoto updates per layer across L blocks. No FLOPs, parameter counts, or wall-clock timing comparisons are provided across any of the three experimental settings. This omission makes it impossible to evaluate whether AKOrN's improvements hold under compute-matched comparisons. The problem is especially acute for Sudoku, where energy-based voting uses 4096 random samples to reach ~90% accuracy (Figure 7) — a massive inference-time cost that is never compared against equivalent compute given to baselines (e.g., beam search or ensemble voting with R-Transformer or IRED). The paper cannot fairly claim improved reasoning performance without accounting for this asymmetry in inference budget.

- **Adversarial robustness claim requires gradient obfuscation verification:** AKOrN's iterative oscillator dynamics with the normalizing operator Π can create gradient discontinuities or poor gradient flow, which are known sources of gradient masking that can produce misleadingly high adversarial accuracy. AutoAttack with EoT (used in the paper) mitigates but does not conclusively rule out this effect. Given the extraordinary claim that a standard-trained model achieves non-zero adversarial accuracy while ResNet-18 and ViT fail completely, BPDA (backward pass differentiable approximation) attacks or gradient norm analysis across iterations is needed to substantiate genuine robustness rather than gradient obfuscation. This is a standard verification step for non-standard architectures making robustness claims at ICLR.

### Minor

- **Missing standard deviations in Table 2 (PascalVOC / COCO2017):** Results are reported as single numbers. Given that evaluation involves clustering with random initializations, multi-seed variance is necessary. All results in Tables 1 and 3 include standard deviations — the omission in Table 2 is inconsistent.

- **Ablations on key hyperparameters are sparse:** The number of Kuramoto steps T and rotating dimensions N are critical hyperparameters whose sensitivity is not analyzed for object discovery or robustness (Figure 8 provides a component ablation for robustness but not T or N sensitivity). The paper mentions "models perform better with asymmetric J" but provides no quantitative ablation of symmetric vs. asymmetric J except indirectly. Given T is set to 16 for Sudoku, it is unclear how T was chosen for other tasks.

- **Readout module design not ablated:** The norm-based readout (Eq. 6) is motivated by phase invariance but not compared against alternatives (e.g., cosine similarities between oscillators, phase difference features). The role of function g (identity vs. MLP) is also not analyzed. This matters because the readout is the primary interface between oscillatory dynamics and downstream prediction.

- **Accuracy-robustness tradeoff not analyzed:** AKOrN's clean accuracy (88.91% / 91.23%) is notably lower than ResNet-18 (94.41%) and Diffenderfer et al. (96.56%). While the paper frames this as "robustness by design," the implicit tradeoff with clean accuracy is not characterized. A comparison at matched clean accuracy, or an analysis of where along the accuracy-robustness frontier AKOrN sits, would sharpen the claim.

- **Up-tiling not uniformly disclosed in Table 2:** The up-tiling method introduces additional inference-time computation. It is not explicitly stated which results in Table 2 use up-tiling, and whether baselines benefit from the same augmentation.

- **Calibration mechanism speculation needs contextualization:** The paper speculates that AKOrN's energy "roughly approximates the likelihood of input examples," which is unverified. Excellent ECE could also arise from uniformly softer predictions (lower average confidence) rather than accurate uncertainty estimation. A comparison against a temperature-scaled ResNet-18 would clarify whether AKOrN offers intrinsically better calibration or simply happens to output lower-confidence predictions that happen to be well-calibrated on this benchmark.

### Tiny

- Step size γ details are incomplete: whether γ is per-layer, per-oscillator, or global, and how it is initialized, is not specified despite its likely importance for training stability.

- The ISA-TS gap (92.9 vs. 88.5 FG-ARI on CLEVRTex) is somewhat underemphasized in the narrative. The claim of being "competitive" with slot-based models is defensible, but the 4.4-point gap at the top of the leaderboard, particularly since ISA-TS is also a learned discrete/continuous hybrid rather than a purely slot-based model, warrants more explicit discussion.

---

## Nice-to-Haves

- **Additional reasoning benchmark beyond Sudoku:** The reasoning claim is made broadly, but only Sudoku (a single constraint-satisfaction domain) is tested. One additional task (e.g., a graph reasoning or logical deduction benchmark) would substantially strengthen the claim.

- **Symmetric vs. asymmetric J ablation:** A systematic quantitative analysis of the performance-theory tradeoff (symmetric J giving theoretical guarantees but worse performance vs. asymmetric J being better empirically) would ground the design choice and help future practitioners.

- **Early stopping / convergence criterion:** An adaptive convergence criterion for the Kuramoto steps per sample would reduce fixed compute costs and make the method more practical for variable-complexity inputs.

- **Discussion of connections to deep equilibrium models (DEQ) and unrolled optimization:** AKOrN's iterative update structure with shared weights is structurally related to these frameworks. Engagement with this line of work would situate AKOrN within the broader landscape of implicit-depth networks and inform the discussion of convergence and test-time step extension.

- **Oscillator trajectory visualization in trained models:** Figure 1 shows untrained oscillators; the supplementary includes trained model visualizations, but main-paper trajectory plots confirming genuine oscillatory vs. static-fixed-point behavior of trained models would validate the dynamical representation narrative.

---

## Removed Points

*These points were flagged for removal — treat with caution.*

- **"Misleading abstract" due to accuracy-robustness tradeoff:** The harsh critic claims the abstract is misleading. However, the abstract says "performance improvements across a wide spectrum of tasks," and Table 4 clearly shows the tradeoff. The paper does not hide it. Kept as a minor weakness (tradeoff not explicitly analyzed) rather than an accusation of dishonesty.

- **Kuramoto-SAT motivation is "invalid" because references are physics analog solvers:** The paper uses these as motivation for why oscillatory dynamics can in principle reason; the demonstration is empirical (Sudoku). Demanding that motivation references be learned neural networks conflates inspiration with proof. Removed.

- **Test-time extension comparison with TrSA is "unfair" due to shared parameters:** The harsh critic argues TrSA naturally degrades when steps exceed training distribution. Both models use shared parameters; both are tested beyond their training step count. AKOrN improving while TrSA degrades is precisely the claim. The comparison is symmetric and the asymmetry favors the baseline by using an identical setup. Removed.

- **Comparison with Bartoldson et al. is misleading:** Bartoldson et al. uses adversarial training; AKOrN does not. The table clearly reflects this, and comparing against adversarially trained models to show what "robustness by design" can approach (rather than exceed) is a legitimate and conservative framing. Not a weakness. Removed.

- **Demand for ImageNet top-1 classification benchmark:** The paper's contribution is not a general classification backbone; it is validated on object discovery (ImageNet pretrained, natural image evaluated), reasoning, and robustness. Demanding ImageNet top-1 accuracy is scope creep. Removed.

- **Demand for theoretical convergence proofs:** This is an empirical systems paper. Requiring Lyapunov guarantees for the asymmetric case would be a non-standard demand. The theoretical gap is kept as a framing concern (Major weakness), not a demand for proof. The proof-demand is removed.

- **Missing related works:** Per instructions, all criticisms about missing related works are removed as they cannot be verified without external sources.

---

## Novel Insights

The most substantive novel insight emerging from cross-reading the reviews against the paper is the following unresolved tension: AKOrN's energy function (Eq. 3) is theoretically justified only under symmetric conditions that the practical model explicitly violates, yet empirically the energy (a) decreases stably during Kuramoto iteration, (b) correlates with prediction correctness in Sudoku, and (c) appears to produce near-perfect calibration in classification. This is not merely an acknowledged limitation but a genuinely interesting empirical phenomenon: why do asymmetric, non-reciprocal Kuramoto dynamics with data-dependent attentive connectivity still produce a meaningful and stable energy landscape that correlates with task-level correctness? One possibility is that the asymmetric J, despite violating the Lyapunov symmetry condition, still induces an approximate gradient flow on some implicit energy surface — analogous to results in non-Hermitian physics or non-equilibrium statistical mechanics. If this behavior could be theoretically grounded (even approximately), it would transform AKOrN from an empirically-motivated departure from the theory into a principled contribution with both theoretical and practical legs. The authors should consider this gap not merely as a limitation to acknowledge but as the most intellectually interesting open question raised by their work.

---

## Suggestions

1. **Reframe the energy function usage:** In Sections 6.2 and 6.3, explicitly state that the energy-based applications rest on empirical observation rather than the theoretical Lyapunov guarantee of Section 3. Add a brief paragraph discussing why asymmetric J might still produce stable energy trajectories (e.g., citing non-equilibrium dynamical systems or noting empirical energy plots from the supplementary).

2. **Add a computational cost table:** Report FLOPs per sample and/or wall-clock inference time for AKOrN vs. ResNet-18 / ViT baselines across all three task settings. For Sudoku energy-based voting, report the total compute for 4096-sample voting vs. a compute-matched baseline.

3. **Conduct BPDA or gradient norm analysis for adversarial robustness:** Run BPDA attacks or report gradient norms across Kuramoto steps to rule out gradient masking as an explanation for robustness. This is essential for the robustness claim to be credible at ICLR.

4. **Add T and N sensitivity plots for at least one non-robustness task:** A performance-vs.-T curve for object discovery (or Sudoku OOD without voting) and a performance-vs.-N curve would significantly inform practitioners and validate the dynamical computation narrative.

5. **Report multi-seed statistics for Table 2** (PascalVOC / COCO2017) for consistency with the rest of the paper.

6. **Compare calibration against temperature scaling:** Add a temperature-scaled ResNet-18 baseline to Figure 9 to demonstrate that AKOrN's calibration is not replicable via a trivial post-hoc procedure.

---

**Novelty:** High — integrating Kuramoto dynamics as a general-purpose neural primitive is a genuinely original contribution, distinct from prior oscillatory NN work which was limited to synthetic/small-scale settings or required pre-trained backbones.

**Technical soundness:** Moderate — the empirical methodology is careful and multi-faceted, but the theoretical-empirical inconsistency around the energy function is a real gap, and the adversarial robustness claim needs stronger verification against gradient masking.

**Empirical support:** Moderate-high — three diverse task families with consistent, reproducible improvements, though key controls are missing (compute comparisons, gradient obfuscation checks, missing standard deviations).

**Significance:** High — a single architectural primitive that improves feature binding, reasoning, and robustness simultaneously could meaningfully influence future neural architecture design, particularly as the field rethinks basic computational units beyond threshold neurons.

**Clarity:** High — the mathematical formulation is precise and the paper's narrative arc from motivation to experiments is coherent and well-organized.

# Actual Human Scores
Individual reviewer scores: [10.0, 10.0, 8.0, 8.0]
Average score: 9.0
Binary outcome: Accept
