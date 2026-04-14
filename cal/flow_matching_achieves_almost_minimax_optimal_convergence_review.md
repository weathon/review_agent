=== CALIBRATION EXAMPLE 57 ===

# Final Consolidated Review
## Summary

This paper establishes the first almost minimax optimal convergence rate for Flow Matching (FM) under the $p$-Wasserstein distance ($1 \leq p \leq 2$), matching the known optimal rate of $n^{-(s+1)/(2s+d)}$ (up to arbitrarily small $\delta > 0$ and $\text{poly}(\log n)$ factors) for target densities in Besov spaces. The analysis generalises the framework of Oko et al. (2023) beyond diffusion models by treating a broader family of mean/variance schedule parameters $(\sigma_t, m_t)$, and crucially identifies $\kappa = 1/2$ (i.e., $\sigma_t \sim \sqrt{t}$) as the necessary condition on the variance decay rate to achieve optimality. The technique for relating the Wasserstein distance to the $L_2$ regression risk is entirely new for the ODE setting, using the Alekseev–Gröbner lemma in place of Girsanov's theorem.

---

## Strengths

- **First minimax rate result for FM.** Prior works (Albergo & Vanden-Eijnden 2023; Benton et al. 2023b) only showed convergence without deriving rates; this is the first rate that is provably nearly optimal, placing FM on the same theoretical footing as diffusion models.

- **Extension from $W_1$ to $W_p$ ($p \leq 2$) via the Alekseev–Gröbner lemma.** Oko et al. (2023) obtained the Wasserstein bound only for $W_1$ via Girsanov/SDE tools that do not apply to ODEs. The new proof strategy—bounding the displacement via the variational equation for an ODE—yields $W_r$ for $1 \leq r \leq 2$ and appears applicable more broadly in the ODE generative model literature.

- **Identification of $\kappa = 1/2$ as the critical threshold.** The paper rigorously characterises how the variance schedule determines the convergence rate (Theorem 9, Eq. 24): for $\kappa = 1/2$ the rate is almost optimal; for $\kappa > 1/2$ it is strictly suboptimal. This is an actionable design principle, not a tautology, and the derivation pinpoints exactly why: the integral $\int (\sigma'_t)^2 dt$ diverges for $\kappa < 1/2$ and is $O(\log N)$ for $\kappa = 1/2$, which is the boundary of manageability.

- **Coupling-independence of the bound.** The paper explicitly notes (Section 2.2) that the theoretical guarantees hold regardless of whether the $(x_{[0]}, x_{[1]})$ pair is drawn independently or via optimal transport. This is a non-obvious point that clarifies the role of OT-CFM from a statistical-theory perspective.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Inconsistency between the formal Theorem 9 (Eq. 22) and the proof-sketch conclusion (Eq. 24), with $Q_0$ left undefined.** Eq. 22 states the rate as $n^{-\frac{s + (Q_0)^{-1} - 1 - \delta}{2s+d}}$, while the explicit calculation in the proof sketch yields Eq. 24: $\tilde{O}(n^{-\frac{s+(2\kappa)^{-1}-\delta/2}{2s+d}})$. For $\kappa = 1/2$ these differ by $1/(2s+d)$ in the exponent — exactly the margin between almost-optimal and suboptimal. The informal Theorem 1 (Eq. 10) compounds the problem by displaying $(2\kappa)\kappa - 1$ (likely a PDF-parsing artifact for $(2\kappa)^{-1}$), and the symbol $Q_0$ is never defined anywhere in the main text. The proof sketch itself appears internally sound and consistent with the claim; the issue is purely in the statement of the formal theorem. However, as written, the key theorem of the paper is literally uninterpretable and numerically inconsistent with the derived bound. This must be corrected before publication: the authors should (i) define $Q_0$ explicitly (presumably $Q_0 = 2\kappa$), (ii) remove the spurious "$-1$" from Eq. 22 if it is a typo, and (iii) harmonise Eqs. 10, 22, and 24.

- **Assumption (A1) requires boundary smoothness $\tilde{s} > \max\{6s-1, 1\}$, an extremely restrictive condition for moderate $s$.** For $s = 2$, boundary smoothness beyond order 11 is required; for $s = 3$, beyond order 17. The paper attributes this to "a technical reason to compensate for the nondifferentiability of $p_0$ at the boundary," but does not explain whether the factor $6s-1$ is tight or an artefact of a loose intermediate bound. Because this assumption gates the entire result, its severity limits the claimed generality—a density with $s = 3$ interior smoothness but only $\tilde{s} = 10$ boundary smoothness is excluded. The authors should either (a) argue why $6s-1$ is unavoidable, (b) verify whether the analysis of Oko et al. (2023) faces the same constraint (if so, it is an inherited limitation that should be stated as such), or (c) indicate whether a weaker condition suffices.

- **The almost-optimal rate requires $O(\log n)$ separate neural networks, one per dyadic time interval.** This time-partition is acknowledged as a limitation in Section 4.4, but the paper does not quantify the gap. In practice, a single time-conditioned network is trained end-to-end. Without the time partition, the paper's own analysis yields only $\tilde{O}(n^{-1/(2s+d)})$ — much worse than optimal for $s > 0$. The theoretical result therefore describes an algorithm that shares the FM objective but not the FM architecture. The authors should at minimum discuss whether a single sufficiently expressive network could implicitly implement the required adaptive complexity, or whether the partition is a fundamental proof artifact.

- **The tension between $\kappa = 1/2$ optimality and the popularity of affine / linear-path FM ($\kappa = 1$) is underexplored.** For the affine path ($\sigma_{[\tau]} = 1-\tau$, i.e., $\kappa = 1$ in reverse time), Theorem 9 gives a strictly suboptimal rate $n^{-(s+0.5-\delta)/(2s+d)}$. This affects OT-CFM and rectified flow — the most widely used FM variants. The paper mentions this in passing but does not discuss whether the suboptimality is an artefact of the proof, whether the constant factor partially compensates, or whether the theoretical gap has empirically observable consequences. A more substantive discussion here would significantly strengthen the paper's practical relevance.

### Minor

- **Assumption (A3) requires *exact* power-law behavior** $\sigma_t = b_0 t^\kappa$ near $t = 0$. Real implementations use cosine schedules, learned schedules, or schedules that only *approximately* satisfy this near $t = 0$. The paper does not discuss robustness to perturbations of the power law, leaving it unclear whether the result is "structurally stable."

- **The abstract does not disclose that almost-optimality requires $\kappa = 1/2$.** Readers may come away believing the result is unconditional, when the main insight is precisely the identification of this schedule condition. A single phrase in the abstract would prevent misunderstanding.

- **The notation shift from $[\tau]$ (forward time, Section 2) to $t$ (reverse time, Section 4) is signalled but the $[\cdot]$ subscript convention is not consistently maintained** in the formal theorems. This causes momentary confusion when cross-referencing the informal Theorem 1 (which uses $[\tau]$) with Theorem 9 (which uses $t$).

### Tiny

- The ERM (exact empirical risk minimiser) assumption is standard in statistical learning theory but not achievable with gradient-based optimisation. This is industry-standard in this literature and should be acknowledged but not held against the paper.

---

## Nice-to-Haves

- **Empirical convergence-rate verification** on synthetic low-dimensional Besov densities: a log-log plot of $W_2$ error vs. $n$ for $\kappa = 1/2$ vs. $\kappa = 1$ would test whether the theoretically predicted slope difference is observable at practical sample sizes. This is not required of a theory paper but would bridge the theory–practice gap.

- **Discussion of the constant-factor dependence** hidden in the $\tilde{O}(\cdot)$ notation with respect to $d$ and $s$. Two methods that share the same asymptotic exponent may differ enormously in sample complexity at finite $n$ if their hidden constants scale differently with $d$. Even a qualitative comparison of the hidden constants for FM vs. DM would be informative.

- **Analysis of whether a single expressive time-conditioned network can approximate the time-partitioned ensemble**, possibly under an additional universality assumption. If such an equivalence can be argued, the practical relevance of Theorem 9 would increase substantially.

- **Incorporating discretisation error** from finite ODE solver steps into the bound. The current guarantee is for the continuously-solved ODE. Jiao et al. (2024) address discretisation but not smoothness; combining both would yield an end-to-end implementable guarantee.

- **Verifying Assumption (A4) from (A3)** for specific schedules. The paper claims (A4) is plausible for the diffusion path ($\sigma_t = t^{1/2}$, giving $(\sigma'_t)^2 \sim 1/t$ and $\int_{T_0} (\sigma'_t)^2 dt = O(\log N)$), but this verification is not carried out. A brief appendix derivation would make (A4) less ad hoc.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **[REMOVED] Abstract omits the κ = 1/2 parameterization (harsh critic).** While it would be clearer to mention it, the abstract appropriately notes that "specific conditions necessary to attain almost optimal rates" are identified, and the informal Theorem 1 explicitly lists $\kappa \geq 1/2$. This is not a real weakness.

- **[REMOVED] Criticism of the $e^{2\int_s^t e^{L_u} du}$ factor in Theorem 3 being "extremely large."** The paper explicitly resolves this via the dyadic partition $t_j = 2t_{j-1}$, which bounds $e^{\int_t^{t_j} (C/u)du} = (t_j/t)^C \leq 2^C$. The harsh critic acknowledges this resolution is "correct" while requesting more explicit derivation — this is a minor exposition request, not a mathematical concern.

- **[REMOVED] Request for missing related-work citations.** Per review policy, external references cannot be confirmed, and such criticisms risk being erroneous. Not included.

- **[REMOVED] Criticism that the conclusion overstates the FM–DM comparison.** The paper states "both FM and DM can attain the same almost minimax optimal convergence rate" — this is accurate, since DM achieves this only for its natural schedule ($\sigma_t \sim \sqrt{t}$) and FM achieves it only for $\kappa = 1/2$, which *is* the same schedule. The claim is not overblown.

- **[REMOVED] Discretization error is not addressed (harsh critic, spark finder).** The paper's stated scope is sample complexity; the authors explicitly contrast their contribution with Jiao et al. (2024) which handles discretization. Criticizing absence of discretization analysis is scope creep.

- **[REMOVED] No TV distance result (spark finder).** The paper achieves $W_p$ for $p \leq 2$ and acknowledges TV as an open problem. Demanding TV is outside the paper's stated scope.

---

## Novel Insights

The most striking observation across all three reviews is that the paper's optimality condition ($\kappa = 1/2$, i.e., $\sigma_t \sim \sqrt{t}$) is *precisely* the variance schedule of a diffusion model. The result therefore does not merely show that FM can match diffusion models in theory — it shows that FM *becomes* a diffusion model (in the probability-flow ODE sense) precisely at the point where it achieves statistical optimality. Affine and linear-path variants that are popular in practice (OT-CFM, rectified flow, with $\kappa = 1$) sacrifice a factor of $n^{-1/(2(2s+d))}$ in the minimax rate. Whether this asymptotic penalty is irrelevant at practical $n$ (because the constants or trajectory straightness dominate) or genuinely harmful is an open empirical and theoretical question that this paper raises but does not resolve — and it is arguably the most important actionable implication of the results.

---

## Suggestions

1. **Fix the formal statement of Theorem 9 (Eq. 22):** define $Q_0$ explicitly (likely $Q_0 = 2\kappa$), remove the apparent spurious "$-1$" from the exponent, and ensure Eqs. 10, 22, and 24 are mutually consistent. This is the most critical revision.

2. **Clarify the $\tilde{s} > 6s-1$ requirement** in Assumption (A1): state whether it is inherited from Oko et al. (2023) unchanged, explain the proof step that requires it, and if possible tighten the bound or discuss its necessity.

3. **Add a focused discussion in Section 5** on the $\kappa = 1$ suboptimality: why is the rate $n^{-(s+0.5)/(2s+d)}$ instead of $n^{-(s+1)/(2s+d)}$ for affine paths, and what does this say about the practical relevance of the bound for OT-CFM and rectified flow?

4. **Clarify time-partitioning's theoretical vs. architectural status** (Section 4.4): is the partition strictly necessary for the current proof technique, or could it be replaced by a single network with adaptive internal structure? A brief discussion would help readers interpret the applicability of the result.

5. **Harmonise forward $[\tau]$ and reverse $t$ notation** throughout the formal theorems so that Theorem 1 (informal) and Theorem 9 (formal) can be directly compared without a change-of-variable translation.

---

**Evaluation summary:** The paper makes a genuine, technically non-trivial theoretical contribution that is significant for the FM and generative modelling communities. The core mathematical argument appears sound. However, the formal theorem statement (Eq. 22) as presented is inconsistent with the derived result (Eq. 24) and contains an undefined symbol ($Q_0$) — a flaw that must be corrected. The boundary-smoothness assumption (A1) is also unusually restrictive and deserves explicit justification. Contingent on these being resolved, the contribution merits publication at a top venue.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
