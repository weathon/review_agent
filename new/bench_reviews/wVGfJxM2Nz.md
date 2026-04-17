## Summary

The paper argues that encoding geometric inductive biases—symmetric positive definiteness for dissipative systems and symplectic structure for conservative systems—enables smaller models to achieve more robust generalization than larger, structure-naive alternatives. This is demonstrated on two systems: a 2D heat transfer problem identified via Riemannian optimization on the SPD manifold, and an 18D Fermi-Pasta-Ulam-Tsingou system modeled with symplectic Hamiltonian neural networks (SHNNs). In both cases, structure-preserving models achieve better stability with far fewer parameters than baselines.

## Strengths

- **Clear and well-motivated thesis**: The central argument—that structure-aware inductive biases reduce model size dependency—is clearly articulated and timely, addressing a practical concern in physics-informed ML.

- **Strong empirical evidence for the conservative case**: The SHNN results on FPUT are striking (Table 2): a 1,441-parameter SHNN achieves better rollout MSE and orders-of-magnitude lower energy drift than a 97,074-parameter LSTM, with compelling phase-space visualizations (Figure 4) that intuitively show energy drift in structure-naive models.

- **Insightful energy drift visualization**: The 2D projected phase portraits with time-evolving energy contours (Figure 2 and Figure 4) effectively convey *why* structure-preserving models generalize better, making the geometric intuition concrete and pedagogically valuable.

- **Demonstrates practical benefit of Riemannian optimization**: RieOpt converges faster (Figure 7 vs. 8) and achieves lower out-of-distribution MSE than EucOpt (Table 1), illustrating the value of respecting manifold geometry during optimization.

## Weaknesses

### Major

- **The dissipative case is trivially simple, making the comparison mostly a statement about problem structure rather than the method.** The system is a 2-state linear model where the LSSM has only 3–6 free parameters to learn from 8,759 hourly data points. Almost any reasonable parametric approach would succeed here. The comparison against generic black-box models (RF, XGBoost, LSTM) conflates "linear parametric model vs. nonlinear black-box" with "structure-preserving vs. structure-naive." A fairer comparison would be against classical system identification methods (e.g., subspace identification) or against an unconstrained linear model of similar parametric complexity. The core issue is that the paper's headline claim—"small structure-aware models outperform large naive ones"—is hard to credit when the "small" model is essentially a known physics-based parameterization with a handful of scalar parameters and abundant data.

- **Missing structure-preserving baselines for the conservative case isolates the wrong variable.** The paper compares SHNNs against LSTM and NeuralODE but not against the most natural alternatives: standard HNNs (Greydanus et al., 2019, cited in the paper) or SympNets (Jin et al., 2020, also cited). Without these baselines, it is impossible to determine whether the observed gains come from the Hamiltonian parameterization alone, the symplectic discretizer, or their combination. This directly undermines the paper's ability to make claims about *which* structural inductive biases matter and *how much* each contributes.

- **The energy drift evaluation metric inherently favors SHNN.** Energy drift (driftRMS) measures deviation from the true Hamiltonian—an invariant that SHNN is architecturally designed to conserve by construction. Using this as a primary evaluation metric makes the comparison partially tautological for the question "does structure-preserving design help with structure preservation?" The rollout MSE is a fairer metric, and there SHNN still performs well, but the paper leans heavily on driftRMS as evidence (Figures 3–4, Table 2 discussion) in a way that is circular.

- **The SPD–stability connection is presented in an overgeneral and potentially misleading way.** The paper frames optimization on the SPD manifold as ensuring stability of the discrete-time dynamics (Section 2.1.1). While this happens to be correct for the specific heat transfer system (where continuous-time eigenvalues are real and negative, so exp(Aτ) has eigenvalues in (0,1)), SPD in general only guarantees eigenvalues > 0, not |λ| < 1. A symmetric positive definite matrix could have an eigenvalue of 10 and still be on the SPD manifold—but the corresponding discrete-time system would be unstable. The paper's language ("Φ_A belonging to the SPD manifold Sym+ₙ which is a non-Euclidean space... of symmetric but stable discrete-time dynamical systems") conflates positive definiteness with stability. This does not invalidate the experiments for this specific system, but it misrepresents the theoretical connection and could mislead readers applying the approach to other systems.

### Minor

- **Incomplete LSTM hyperparameter sweep.** Table 2 shows missing entries ("– – –") for several LSTM depths (L=2,4,8). The paper's strongest benchmarking statement—that a 1,441-parameter SHNN beats the best 97,074-parameter LSTM—is based on an incomplete search over LSTM architectures. Shallower LSTMs with widths between 18–72 might achieve comparable or better efficiency but were not evaluated.

- **Single-run results with no variance estimation.** All results appear to be from single training runs with no standard deviations across random seeds. Given the nonconvex optimization involved, and the fact that NeuralODE driftRMS varies by more than an order of magnitude across configurations (1.2 to 1803 in Table 2), this is a meaningful gap. (Note: single-run evaluation is common in this subfield, so this is a minor rather than major concern.)

- **Typo in the loss function.** Equation (7) defines the loss as ∑‖Φ_A T_i + Φ_B T_i − T_{i+1}‖², but per Eq. (4), the second term should be Φ_B U_i, not Φ_B T_i. This appears to be a rendering error but could cause confusion.

### Trivial

— (none remaining)

## Nice-to-Haves

- **Add HNN and SympNet baselines** for the conservative case to isolate whether gains come from Hamiltonian parameterization, symplectic integration, or both. This would substantially clarify the contribution.

- **Test on a higher-dimensional dissipative system** (e.g., m ≥ 10 states) to demonstrate that the SPD manifold approach scales beyond the trivial 2D case, and to make the comparison more meaningful.

- **Report the learned Φ_A matrices** for both RieOpt and EucOpt alongside the physics-based initial guess, to reveal whether Riemannian optimization yields physically interpretable corrections.

- **Perform data-scaling experiments** (vary training set size) to directly test the core claim that structure preservation reduces data dependency. Currently this claim is asserted in the introduction but never measured experimentally.

- **Train on multiple trajectories** at different energy levels for the FPUT system to systematically assess generalization across the phase space.

- **Evaluate the Cholesky parameterization** (Φ_A = LL^T) mentioned in Section 2.1.2 as a simpler alternative SPD-enforcing method, to establish whether Riemannian optimization is necessary or whether simpler constraint-enforcement suffices.

## Removed Points

- **"Comparison against structurally-naive baselines is not fair" (harsh critic, structural)**: The harsh critic argued the entire experimental framing is fundamentally unfair because structurally-naive models don't have access to comparable priors. This misreads the paper's intent: the paper's thesis IS that structure-preserving inductive biases give you something you cannot easily get from naive models. The asymmetry is the point, not a flaw. However, the specific objection that the dissipative case conflates "parametric physics model vs. generic forecaster" is retained above. The broader claim of unfairness is removed because it misunderstands the paper's comparative purpose.

- **"Energy drift metric is tautological — remove entirely"**: Retained in weakened form. Using driftRMS as *one* metric is legitimate; the problem is the paper's heavy reliance on it as primary evidence for structure-preservation claims. Removed the claim that this invalidates the entire evaluation; retained as a minor-major concern about over-reliance on a metric that structurally favors the proposed method.

- **"Conflation of SPD with stability is a fatal theoretical error" (harsh critic)**: Downgraded from Fatal to Major. For the specific 2D heat transfer system studied, the continuous-time A matrix does have negative real eigenvalues, making exp(Aτ) SPD with eigenvalues in (0,1)—i.e., the SPD constraint does enforce stability in this particular case. The error is one of *overgeneralization in the framing*, not of the actual experiments being wrong. Reclassified as a presentation/accuracy issue rather than fatal.

- **"Reproducibility concerns about hyperparameters and missing implementation details"**: Removed as nitpick per instructions (undisclosed hyperparameters and implementation details are standard).

- **"Formatting/style issues"**: Removed per instructions.

- **"No comparison with subspace identification methods (N4SID, etc.)"**: The paper's stated scope is comparing structure-preserving vs. structure-naive ML approaches, not benchmarking against classical SID. Removed as scope creep, though it is retained as a nice-to-have.

- **"Computational cost not analyzed"**: While relevant, this is not a standard requirement for empirical comparison papers in this field and the methods are well-established. Moved to minor consideration within the trivial-dimensionality weakness.

## Novel Insights

The paper's most insightful contribution is the visual and quantitative demonstration that structure-naive models suffer from qualitatively different failure modes (energy drift across invariant level sets) that cannot be fixed by scaling model size—this is not just a matter of accuracy but of fundamentally wrong asymptotic behavior. However, the insight is somewhat diminished by the fact that this is predictable from the theory (symplectic integration conserves energy by construction, generic models don't) and the paper doesn't systematically analyze the boundary conditions under which structure preservation stops being necessary (e.g., richer training data, different architectural priors). The dissipative case adds less insight than the conservative one, since the trivial dimensionality and known-linear physics make the result largely expected.

## Suggestions

1. **Add HNN and SympNet baselines** to the conservative experiments; this single addition would dramatically strengthen the paper by isolating the contribution of symplectic integration versus Hamiltonian parameterization.
2. **Replace or augment the dissipative case** with a system of dimension ≥ 5 where the LSSM approach is not trivially parameterized, to make the "small model" claim more credible.
3. **Correct the SPD–stability framing**: Add a clarifying remark that while SPD holds for the specific heat transfer system (because the continuous-time eigenvalues are real and negative), the constraint SPD ≠ stability in general, and that the approach should be applied with appropriate stability constraints for other systems.
4. **Report quantitative rollout metrics on perturbed initial conditions** for the FPUT system (not just qualitative phase portraits) to substantiate the generalization claims.

## Score and Decision

**Calibration papers examined:**
- **RoN6M3i7gJ** (Riemannian Framework for Lagrangian Dynamics): Scores 5–6, Accept Poster. Similar application of Riemannian optimization for structure-preserving dynamics with limited novelty and scalability concerns.
- **0Y26tFG3WF** (Lagrangian NNs for Chaotic Systems): Scores 3–5, Reject. Shares overly simple experiments and missing baselines.
- **U1DjXQeJRx** (Poisson-Dirac NNs): Scores 5–8, Accept Poster. Similar structure-preserving approach with concerns about missing classical baselines.
- **AZGIwqCyYY** (Cross-Domain Hamiltonian Generalization): Scores 5–6, Accept Poster. Shares novelty concerns and limited evaluation but clean execution.
- **XqDM97DtMf** (Learning Chaotic Dynamics): Scores 3–8 (withdrawn/reject). Shares dissipative structure-preserving learning with limited experiments.

This paper sits at a similar level to the Riemannian Framework paper (RoN6M3i7gJ, avg ~5.5) and below Poisson-Dirac NNs (U1DjXQeJRx, avg ~6.5). The conservative case results are compelling, but the dissipative case is too simple to carry weight, missing structure-preserving baselines limits the scientific insight, and the SPD/stability framing is misleading. The paper's contribution is primarily demonstrating (rather than advancing) existing methods. This places it slightly below papers with comparable scope but cleaner execution.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>