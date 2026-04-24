Now let me search for calibration papers.Now I have enough information to write the final review. Let me compile it.

---

## Summary

The paper proposes Derivative Learning (DERL), a method to train neural networks by directly matching partial derivatives of the target function (plus IC/BC anchoring), rather than using a PDE residual or direct function-value supervision. The authors provide theoretical guarantees (Theorems 2.1–2.3 via Sobolev space analysis) that zero-loss implies convergence to the true solution, validate DERL on five PDE/ODE settings, and introduce derivative distillation — transferring physical knowledge across models by matching teacher model derivatives, including Hessians (HESL). The paper's core claim is that pure derivative targets avoid the optimization difficulty of PDE-residual entanglement, and that DERL outperforms PINNs and other baselines.

---

## Strengths

- **Clear mechanistic distinction from PINNs and Sobolev learning**: DERL uses a *pure* derivative objective (Eq. 3) with no PDE residual term and no function-value term in the interior. This is a meaningful distinction from Sobolev training (Czarnecki et al., 2017), which adds derivative terms *on top* of function-value supervision. The framing is clean and reproducible.

- **Theoretical guarantees (Theorems 2.1–2.3)**: The paper proves formally that minimizing the DERL loss controls the full W^{1,2} norm of the error, using classical Sobolev space tools (Poincaré inequality + trace theorem applied to the NN setting). The bound ‖û − u‖_{W^{1,2}} ≤ 2(C+1)ε when L(û,u) ≤ ε (Theorem 2.2) is concrete and the analysis is rigorous.

- **Breadth of experiments**: Five different systems are evaluated — an ODE (damped pendulum), a time-independent PDE (Allen-Cahn), a time-dependent PDE (continuity equation), a system of PDEs (Navier-Stokes), and two distillation tasks (KdV, NCL). This breadth demonstrates generality beyond a niche setting.

- **Strong results on physical consistency metrics**: DERL consistently achieves better PDE residuals than OUTL and PINN. On Navier-Stokes (Table 5), DERL reduces the momentum residual to 0.36 vs. PINN's 6.96 — a ~20× improvement. On Allen-Cahn (Table 3), DERL achieves L²=0.010380 vs. PINN's 0.030950 and OUTL's 0.018174.

- **Novel and concrete distillation finding**: The HESL result in Table 6 — that matching Hessians of the teacher rather than gradients reduces PDE residual from 0.32480 (DERL) to 0.19153 (HESL), while BC loss drops by over an order of magnitude (teacher: 0.33532, HESL student: 0.014220) — is specific, reproducible, and novel. Higher-order derivative distillation is an underexplored mechanism for physical knowledge transfer.

- **Outperforms physics-specialized HNN/LNN on the pendulum**: Despite HNN and LNN being specifically designed for conservative systems and requiring external ODE solvers, DERL achieves field error 0.28006 vs. 0.44699 (LNN) and 0.44277 (HNN) (Figure 2b). This is a notable result given HNN/LNN's structural inductive bias.

---

## Weaknesses

### Fatal
None.

### Major

- **Abstract performance claim is contradicted by the paper's own tables.** The Abstract and Section 1 state that DERL "outperforms PINNs and other state-of-the-art approaches." However: in Table 4 (Continuity), OUTL achieves L²=0.027932 vs. DERL's 0.028827; in Table 5 (Navier-Stokes), OUTL achieves L²=0.011950 vs. DERL's 0.021687 (~81% worse), and SOB beats DERL on incompressibility (E4.I) residual (0.29979 vs. 0.30337). DERL wins on PDE residual metrics in most cases but does not win on the most fundamental metric — L² distance from the ground truth — in two of the four main PDE experiments. The paper should clearly stratify its claims: DERL is better at *physical consistency* (PDE residuals) than OUTL and PINN, but OUTL often achieves better raw solution accuracy. This conflation of two distinct metrics in the headline claim is misleading and will undermine the paper's credibility.

- **Missing neural operator baselines.** The paper's experiments operate in a dense-data regime (fine grids with Δx=Δy=Δt=0.01) for Continuity, Navier-Stokes, and KdV. In this regime, the relevant comparison class is Fourier Neural Operators (FNO, Li et al., 2021) and DeepONet, which the related work section explicitly acknowledges as "powerful and find many applications to real-world problems." None of these are included as baselines. Without comparison to FNO-class methods — which are the current practical standard for surrogate PDE modeling with simulation data — the paper cannot credibly claim to outperform "state-of-the-art approaches" in these settings. This is the most significant gap in the experimental evaluation.

### Minor

- **The "without seeing interior data" framing is misleading.** The paper repeatedly states that DERL "achieves these results without having access to the solution in the interior" (Section 4.4) or "without ever seeing any data for t > 0" (Section 4.3). However, DERL's derivative targets are computed via finite differences from the full dense numerical simulation (Δx=Δy=Δt=0.01 for Continuity; same for Navier-Stokes), which requires evaluating u(x,y,t) at interior grid points across the full time domain. The distinction the paper makes — that the *loss function* does not regress to function values directly, only to their derivatives — is legitimate and important, but the framing should be more precise: DERL does not directly supervise function values at interior points, but its training targets are computed from those values. The claim "without access to the interior solution" is technically incorrect.

- **Theoretical results explain convergence, not advantage.** Theorems 2.1–2.3 prove that when the DERL loss → 0, û → u in W^{1,2}. This is a standard Sobolev space argument (Poincaré + trace theorem). The theorems do not explain *why* DERL's training objective is easier to optimize than PINNs' or OUTL's, nor do they provide finite-sample or convergence-rate guarantees. The theoretical section is rigorous but proves a weaker claim than the paper implies, leaving the key empirical advantage unexplained theoretically.

- **Single-run results for close comparisons.** All tables report single-run values. For pairs where the performance difference is small (e.g., DERL 0.028827 vs. OUTL 0.027932 in Table 4, a difference of ~3%; DERL 0.38331 vs. PINN teacher 0.037171 in Table 6), it is impossible to know if differences are meaningful without variance estimates across seeds. Multiple seeds are especially important for the distillation experiments, where the claim rests on the student achieving nearly teacher-level L² accuracy.

- **Distillation claim slightly overstated.** The paper states that "distilling a PINN can lead to performance improvements" (Section 4.5). In Table 6, HESL achieves L²=0.037380 vs. the teacher's 0.037171 — the student is marginally *worse* in solution accuracy. The genuine improvement is specifically in BC adherence and PDE residual, which is still a valuable finding but should be stated more precisely.

### Trivial
- None worth noting after filtering parser artifacts.

---

## Nice-to-Haves

- **Training time/computational cost comparison**: DERL uses AD to match derivative targets, PINNs use AD for PDE residuals, OUTL requires no AD interior passes. A wall-clock comparison would clarify the practical trade-offs, especially since DERL loses to OUTL on solution accuracy in two experiments despite potentially higher computational cost.

- **Loss convergence curves**: Plotting training loss vs. epochs for all methods would clarify whether DERL's PDE residual advantage comes from faster convergence, better final optima, or both — directly testing the hypothesis that avoiding PDE-residual entanglement improves optimization.

- **Evaluation at different data densities**: DERL is evaluated only in dense-data regimes. Testing across data densities would reveal the method's actual niche relative to PINNs (which are designed for sparse data) and neural operators (which require dense data).

---

## Removed Points

*These points were flagged for removal; treat with caution.*

1. **Harsh Critic Issue 1 (Data-requirement paradox undermining the central framing)**: The critic argues that DERL is "not learning without knowledge of equations" because derivatives are computed from the solution. However, the paper's claim is specifically that DERL doesn't need the *functional form of the PDE* (i.e., the operator F in F(u, Du)=0), only the derivatives of the solution. This is a legitimate distinction from PINNs. The claim is somewhat overstated in parts (see Minor weakness above), but the structural paradox the critic identifies is partly a misreading. The paper accurately describes DERL as equation-free at the level of the loss formulation. REMOVED as a fatal/major issue; a narrower version of this criticism is captured in the Minor weakness about the "interior data" framing.

2. **Harsh Critic: "Distillation claim unsupported because student doesn't beat teacher on L²"**: The paper's distillation claim is that distilling *improves physical consistency* (PDE residual, BC adherence), not that it improves raw solution accuracy. The abstract says "distillation of higher-order derivatives improves physical consistency," which is accurate per Table 6. The critic strawmanned this claim. REMOVED.

3. **Strength Finder: "Consistent empirical superiority over PINNs across diverse PDEs"**: Partially valid — DERL consistently beats PINN, but not all baselines (OUTL wins on L² in two experiments). WEAKENED: moved to partial strength.

4. **Strength Finder: "Applicability to empirical derivatives confirms practical applicability"**: This is a restatement of the theoretical result from Appendix C and not a separate novel empirical finding. Dropped as a generic supporting strength without independent evidential value.

---

## Novel Insights

The most genuinely novel observation in this paper is the Hessian-level distillation result (HESL): training a student network to match second-order derivatives (Hessians) of a teacher PINN — rather than first-order gradients or output values — reduces boundary condition loss by over an order of magnitude and brings PDE residual close to teacher-level without sacrificing solution accuracy. This suggests that higher-order derivative supervision carries disproportionately more physical constraint information than gradient-level supervision, which has not been systematically demonstrated in prior work on knowledge distillation for physical systems.

---

## Suggestions

1. Rewrite the Abstract's performance claim to accurately reflect that DERL outperforms on *physical consistency metrics* (PDE residuals), while results on solution accuracy (L²) are mixed vs. OUTL. This is actually a more interesting and defensible claim — DERL achieves physical consistency closer to PINNs but with OUTL-level solution accuracy.

2. Add at least one neural operator baseline (FNO or DeepONet) on Continuity, Navier-Stokes, or KdV to properly contextualize the "state-of-the-art" claim.

3. Report results across multiple random seeds (at least 3–5) for all tables, with standard deviations, especially for close comparisons.

4. Revise phrasing around "without having access to the interior solution" to accurately reflect that derivative targets are computed from interior solution values via finite differences, even though function values are not directly supervised.

5. Add a section or figure analyzing *why* DERL's training objective is easier to minimize than PINNs' — e.g., conditioning analysis, loss landscape geometry, or gradient variance comparisons. This would substantially strengthen the theoretical contribution from "correctness when loss=0" to "explanation of empirical advantage."

---

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Path | Avg Score | Decision | Comparison |
|---|---|---|---|---|
| cd-PINN (continuous dependence extension) | 7xJgPtLHfm.md | 5.00 | Reject | Incremental PINN extension, limited experiments; DERL is broader |
| Physics-informed NNs transformed geometries | kIZcruKmBg.md | 3.25 | Reject/Withdrawn | Narrow scope, weak experiments; less substantial than DERL |
| Sobolev acceleration for NNs | YhT1ZemZow.md | 4.50 | Reject | Very similar type of contribution (derivative training theory), limited experiments; DERL is broader |
| Operator preconditioning for PINNs | WWlxFtR5sV.md | 6.33 | Accept (poster) | Strong theory explaining optimization difficulty; DERL has broader experiments but weaker theoretical explanations of its advantage |
| PINeCONes (neural ODE + PINN) | TB5THwq1sq.md | 3.60 | Reject | Missing key baselines, limited to 1D; DERL is more substantive |
| PhyMPGN (physics-encoded GNN) | fU8H4lzkIm.md | 8.00 | Accept (Spotlight) | Much stronger: irregular mesh, small data, clear baselines; well above DERL |
| Refined generalization for PINNs | vsLohTBH4h.md | 4.50 | Reject | Theory-only, no experimental contribution; DERL stronger empirically |

**Assessment**: DERL is substantially stronger than the low-end anchors (PINeCONes at 3.6, cd-PINN at 5.0, Sobolev acceleration at 4.5): it has broader experiments, covers ODEs through third-order PDEs, introduces the novel HESL distillation finding, and has clear theoretical backing. However, it falls short of the operator preconditioning paper (6.33) primarily because: (a) that paper has clean, consistent results in its setting; (b) its theory actually *explains* the training difficulty whereas DERL's theory only proves correctness at zero loss; (c) DERL has a significant missing-baseline problem (no FNO) and overclaims in the abstract. DERL also falls well below the PhyMPGN spotlight (8.0), which had far stronger empirical comparisons and a more original architecture.

The paper is above the 4.5 floor (it has more breadth and a specific novel finding in HESL distillation), but the missing FNO baselines and overclaimed headline result are real deductions. I place this at **5.0**: the paper has a real but incremental contribution (pure derivative supervision for physical systems, Sobolev-theory backing, novel HESL) with experimental gaps that would need to be addressed before the claims are fully established.

**Originality**: Moderate. The "pure derivative" approach distinguishes from Sobolev training but is a relatively small step. Hessian distillation is more novel.  
**Importance**: Moderate. Physical ML is important, but the actual regime where DERL uniquely excels (vs. OUTL and FNO) is unclear from the presented experiments.  
**Claim support**: Mixed. Well-supported for Allen-Cahn and pendulum, partially supported for Navier-Stokes and Continuity, unsupported for the "state-of-the-art" framing.  
**Experimental soundness**: Incomplete (missing FNO baselines, single-run results).  
**Clarity**: Generally good, with some imprecise framing around "interior data."  
**Community value**: Moderate; the distillation idea could be genuinely useful if empirically validated more thoroughly.

**Score: 5.0 | Decision: Reject** (borderline; revision could elevate to accept with FNO baselines, corrected abstract claims, and multi-seed results)

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>