Now I have enough of the paper. Let me do calibration searches.Now I have enough information to write the final consolidated review. Let me synthesize the findings.Now I have everything I need. Let me write the final consolidated review.

---

## Summary

TULiP is a post-hoc uncertainty estimator for OOD detection that grounds its design in linearized training dynamics (NTK theory). The core theoretical contribution is Theorem 3.1, which bounds the effect of a hypothetical pre-convergence perturbation on a trained network's prediction by the NTK-distance between the test point and the training set. Proposition 3.3 then eliminates the need for training data by upper-bounding this quantity using only the trained parameters. Practically, TULiP constructs surrogate posterior samples via layer-wise-scaled weight perturbation and derives an entropy-based uncertainty score; it achieves particularly strong near-OOD detection on CIFAR-10 and ImageNet-200 in the OpenOOD benchmark.

---

## Strengths

- **Theorem 3.1 is a genuine theoretical result** (Section 3.3, Eq. 5): the bound on training fluctuations is dominated by the Frobenius-norm distance between the test-point Jacobian and the nearest training-point Jacobian, yielding a clean NTK-distance interpretation (Eq. 6). The proof idea — triangulating via a pivot in the training set and leveraging near-perfect convergence (A4) — is non-trivial and well-motivated.
- **Synthetic validation is convincing** (Figure 2): using exact NTK computation via `neural-tangents`, the bound (Eq. 5) correctly encapsulates the ground-truth ensemble variance on Splines regression, and the Two-Moons classification map (right panel) closely matches the simulated prediction variance (middle panel), directly validating Theorem 3.1 under ideal conditions.
- **Strong near-OOD empirical results** (Table 1): On CIFAR-10, TULiP achieves FPR@95 of 33.80 and AUROC 89.67, compared to the best non-training-data baseline (GEN: 53.67 FPR@95, 88.20 AUROC) — roughly a 20 percentage-point FPR improvement. On ImageNet-200, TULiP achieves the best near-AUROC of 83.84. These are genuinely state-of-the-art for methods that do not use training data.
- **Cross-architecture validation** (Figure 3): TULiP consistently outperforms MLS and ODIN across MobileNet V3, VGG 16, and RegNet Y 16GF on ImageNet-1K, suggesting the method is not narrowly tuned to ResNet-specific behavior.
- **Computationally efficient**: The method requires only O(M) forward passes with no backward passes, making it practical for large-scale evaluation. The paper notes it evaluates ~3× faster than ViM on large benchmarks.
- **Transparent about limitations**: The paper explicitly acknowledges the heuristic layer-wise scaling (Section 4.1: *"such scaling is highly heuristic"*), the near/far OOD tradeoff (Figure 4, Section 6), and poor transformer performance (Appendix C.4).

---

## Weaknesses

### Fatal
None.

### Major

- **Theory-to-practice gap undermines the "theoretically-driven" framing.** The abstract and introduction claim TULiP is a *theoretically-driven* method with "direct theoretical justifications regarding the training process," but Algorithm 1 departs from the theoretical framework at multiple key design decisions, each acknowledged in the paper:
  - The layer-wise scaling (Eq. 12, $\Gamma_l = (1/\sqrt{|\theta_l|})\cdot I$) is explicitly called "highly heuristic" in Section 4.1. Its empirical justification (Figure 1b) holds at Epoch 20 but the paper itself notes this relationship "disappears at t=200" (Figure 1c), so it does not capture the full training trajectory.
  - The hyperparameter $\lambda$ is substituted for the theoretically determined constant $K$ from Lemma 3.2 (Section 4.2), without a principled connection between the two.
  - The term $\mathbb{E}_\mathbf{x}[\Theta(\mathbf{x},\mathbf{x})]$ is dropped from Eq. 9 (Section 4.3) on the grounds of tractability and being "irrelevant to $\mathbf{z}$" — acceptable for ranking but it means the theoretical bound (Eq. 9) and the implemented score $S$ differ by an unquantified dataset-dependent constant.
  - $\theta_{t_s} \leftarrow \mathbf{0}$ (Algorithm 1, line 1) approximates the initialization mean rather than an actual initialization draw, as the theory requires.
  
  Individually each departure is acknowledged and reasonable; cumulatively they mean Algorithm 1 is more accurately described as *theoretically inspired* than *theoretically derived*. The paper's identity claim ("theoretically-driven") would be better supported if it quantified how much each approximation costs, or if it showed that the score $S$ correlates monotonically with the theoretical bound in practice. The claim should be scoped accordingly.

- **"Consistently improves previous state-of-the-art" is overstated.** The abstract (and Introduction bullet iii) state TULiP *consistently* improves prior methods. Table 1 contradicts this for far-OOD on ImageNet-1K: TULiP AUROC 88.03 vs ASH AUROC 95.74 (a 7.7-point gap), and FPR@95 48.01 vs ASH 19.49 (a 28-point gap). The paper offers a post-hoc explanation (ResNet-50 redundant representations) in Section 5.2, but no theory-driven prediction of when TULiP should or should not outperform ASH. The near-OOD results *are* consistently strong; the headline claim should be scoped to that regime.

### Minor

- **Closeness assumption (Eq. 8) is only validated on ID data** (Figure 1d): The experiment uses 256 ID samples from ImageNet-1K and 128 OOD samples per dataset, but validation is presented as confirming the assumption holds by "a large margin." However, Figure 1d appears to check whether the assumption holds generally, not whether it holds *more tightly for ID vs. near-OOD inputs* — which is the critical question given that near-OOD is the use case. If the assumption barely holds for the hardest near-OOD inputs, the bound may not be informative there.

- **Exponential constant in Theorem 3.1 may render the bound vacuous for practical settings.** The constant $C = \frac{\alpha \bar{\Theta}_X^{1/2}}{\lambda_{\max}}(e^{(T-t_s)L\lambda_{\max}} - 1)$ grows exponentially in $(T-t_s)L\lambda_{\max}$. At $t_s = 0$ (the implementation choice), $C$ is exponentially large in the full training horizon times $L\lambda_{\max}$. The synthetic validation (Figure 2) uses infinite-wide networks in the exact lazy regime, which may have very small $\lambda_{\max}$. The paper does not discuss whether the bound is informative in the practical finite-width setting, or provide numerical estimates of $C$.

- **Finite-difference step size is far from the theoretical limit.** Equation 13 approximates a Jacobian-vector product with $\delta \rightarrow 0$, but in practice $\delta \in \{2, 5, 8\}$ — large values that introduce $O(\delta)$ bias in the approximation. This affects $D$ in line 12 of Algorithm 1, which feeds directly into the uncertainty score $S$. The sensitivity of results to $\delta$ is partially explored (ablation search) but the bias introduced relative to the theoretical quantity is not characterized.

### Trivial

- The ablation (Figure 4) varies $\lambda$ and $\epsilon$ but never compares against a condition using the raw perturbed predictions ($\tilde{f}_i^{\text{raw}}$) without the $\gamma$-scaling step (i.e., no surrogate ensemble construction). This would isolate whether TULiP's gains come from the bound-matching construction or simply from raw weight perturbation as a variance-estimation heuristic (analogous to MC-Dropout without dropout layers).

---

## Nice-to-Haves

- A principled extension of the layer-wise scaling (Eq. 12) to transformer architectures (attention layers and layer norms), which the paper acknowledges as future work but would meaningfully broaden TULiP's applicability given transformers' dominance in large-scale vision.
- An experiment directly plotting the score $S$ (lines 11–13 of Algorithm 1) against held-out OOD-ness across inputs, to empirically establish whether the theoretical scoring mechanism (not just the surrogate samples) correlates with OOD detection on real networks.
- Sensitivity analysis of the layer-wise scaling across different training setups (e.g., Adam vs. SGD, different learning-rate schedules) to assess how broadly the Epoch-20 Jacobian ratio justification (Figure 1b) generalizes.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they may reflect reviewer misreadings.*

- **"K is arbitrary in Lemma 3.2"** (Harsh Critic): The paper explicitly says λ is "a proxy to the constant K in Lemma 3.2" (Section 4.2). This is a known approximation the authors acknowledge, not a silent substitution. Removed as strawman.
- **"Surrogate samples have no valid probabilistic interpretation"** (Harsh Critic): The paper consistently calls these "surrogates to posterior samples," never claiming exact equivalence. Equation 10 justifies variance matching to an upper bound, and the paper is explicit that significant simplifications were made. The claim is appropriately hedged throughout. Removed as strawman.
- **"Setting θ_{t_s}←0 conflates initialization mean with actual initialization"** (Harsh Critic): The paper explicitly says "we take $t_s = 0$ and substitute $\theta_{t_s}$ with $\mathbb{E}[\theta_0] = \mathbf{0}$" (Section 4), presenting it as an acknowledged practical approximation. Removed as strawman.
- **"ASH outperforms TULiP — unfair comparison"**: This is not an unfair comparison asymmetry favoring baselines; ASH genuinely outperforms TULiP on far-OOD ImageNet-1K. Kept as a real weakness (Major section).
- **Hyperparameter asymmetry vs. parameter-free baselines** (Harsh Critic): The paper follows OpenOOD protocol with a small validation set search, which is standard for the benchmark. EBO and MLS are not strictly "parameter-free" relative to score computation (threshold tuning is standard). Removed as not substantive enough to change assessment.
- **"The bound's constant is exponentially large — synthetic validation is in a different regime"** — partially valid; retained as Minor weakness but weakened from the Harsh Critic's "structural" framing.
- **"Closeness assumption validated only on ID data"** — kept as Minor but weakened from the Harsh Critic's characterization (the paper does validate it on 5 OOD datasets, just not specifically showing the margin difference for near-OOD).
- **Strength Finder: "Theory-predicted near-OOD advantage is empirically confirmed"** — dropped from Strengths because the closeness assumption is only validated on ID/OOD data but not specifically on the mechanism driving the near/far tradeoff; the near-OOD advantage is confirmed empirically but the theoretical prediction is post-hoc.

---

## Novel Insights

The most genuinely novel observation in this paper — one that partially survives the theory-to-practice criticisms — is the connection between the *parameter count of a layer* and *how much it deviates from the NTK at early vs. late training* (Figure 1a–b). The empirical finding that layers with more parameters train slower and that their Jacobian ratio correlates with $|\theta_l|^{-1/2}$ at Epoch 20 suggests a structured relationship between parameter count, training speed, and NTK trajectory that could have applications beyond TULiP's specific use case. This is a concrete, potentially reusable empirical finding about training dynamics that the paper uses as a heuristic but which could underpin more principled future methods.

---

## Suggestions

1. **Reframe the theory-practice connection honestly**: Replace "theoretically-driven" with "theoretically-motivated" or "theoretically-inspired," and add a summary table in the paper (or appendix) enumerating each approximation from theory to Algorithm 1, with a sentence on its impact. This would make the paper's actual contribution clearer and more defensible.
2. **Quantify the exponential constant**: For the practical settings used in experiments, compute numerical estimates of $C$ in Theorem 3.1 to establish whether the bound is informative or merely qualitative.
3. **Add a near/far closeness assumption check**: In Figure 1d, overlay the closeness margin specifically for near-OOD vs. far-OOD subsets to test whether the theory predicts the empirically observed near-OOD advantage.
4. **Scope the "consistently improves" claim**: Change the abstract and Introduction bullet iii to specify that TULiP "achieves state-of-the-art or competitive near-OOD detection performance" across all datasets — consistent with what Table 1 actually shows.
5. **Add raw-perturbation ablation**: Include an experimental condition using variance of $\tilde{f}_i^{\text{raw}}$ directly (without γ-scaling to match the bound), to separate the contribution of the theoretical scoring from the weight-perturbation ensemble.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Decision | Comparison |
|---|---|---|---|---|
| NECO: Neural Collapse-based OOD | `/human_reviews/9ROuKblmi7.md` | 5.75 | Accept (poster) | Most similar profile: post-hoc OOD + theoretical explanation + competitive benchmark results; TULiP has stronger near-OOD margins but weaker theory-practice connection |
| SCALE: Activation Shaping for OOD | `/human_reviews/RDSTjtnqCg.md` | 6.25 | Accept (poster) | Post-hoc OOD with mechanism analysis + SoTA; simpler method than TULiP but broader SoTA claim support |
| DNNs Tend to Extrapolate Predictably | `/human_reviews/ljwoQ3cvQh.md` | 7.00 | Accept (poster) | OOD/uncertainty with novel theoretical insight + strong empirical; more cohesive theory-experiment alignment than TULiP |
| Streamlining Prediction in BDL | `/human_reviews/pW387D5OUN.md` | 7.00 | Accept (poster) | Bayesian DL with linearization + strong empirical; tighter theory-practice link than TULiP |
| Unified Uncertainty Estimation | `/human_reviews/56jIlazr6a.md` | 5.25 | Reject | Uncertainty estimation framework that also conflates epistemic/aleatoric concerns |
| Exact Path Kernels for OOD | `/human_reviews/gZRfDWLlGY.md` | 4.33 | Reject | OOD via kernel path decomposition — weaker empirical results and more fragmented theory than TULiP |
| Pre-trained Networks Detect Familiar OOD | `/human_reviews/Pb9PIECnNF.md` | 4.00 | Withdrawn | OOD with pre-trained models — incremental, limited scope, weaker than TULiP |
| TTA for OOD (HAct-like) | `/human_reviews/H65sp7ztys.md` | 3.67 | Withdrawn | Modest OOD contribution with limited theoretical backing — clearly weaker than TULiP |

TULiP sits between NECO (5.75, accepted) and SCALE (6.25, accepted). It has a more ambitious theoretical framework than NECO and stronger near-OOD margins, but its theory-practice gap is wider and its SoTA claims are less fully supported. The most natural score is **5.5**, with a borderline-Accept decision. The paper's near-OOD results are genuinely strong (especially CIFAR-10), the theoretical framework is a real contribution, and the paper is honest about limitations — all of which merit acceptance. The major weaknesses (overclaimed framing, heuristic-heavy implementation) do not invalidate the contribution but should be addressed in revision.

**Originality**: Moderate-to-good. Connecting NTK training fluctuations to post-hoc OOD detection is a novel framing.  
**Importance of research question**: High. Post-hoc OOD detection with theoretical grounding is an important open problem.  
**Claims supported**: Partially. Near-OOD claims are well-supported; far-OOD SoTA claims are not.  
**Soundness of experiments**: Good. OpenOOD benchmark is comprehensive; ablations are present but limited.  
**Clarity of writing**: Good. Paper is clearly written and honest about limitations.  
**Value to the research community**: Moderate-to-high. Practical method with genuine near-OOD improvements and a reusable theoretical framework.

**Score: 5.5** — borderline accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>