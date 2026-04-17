---
job_id: 14caf303-93e7-4c81-bf58-18cfc5afa8cb
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: d2pUyiXwcm.pdf
paper: Physics-Informed Inference Time Scaling for Solving High-Dimensional Partial Differential Equations via Defect Correction
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely in scientific machine learning / physics-informed ML, focusing on solving high-dimensional PDEs with a hybrid surrogate–Monte Carlo method, which is fully aligned with ICLR topics (representation learning, hybrid AI systems, applications to physical sciences, optimization, and learning theory).

## Minimum Quality
Pass ✅.  
The paper is in English, has a standard structure (Abstract, Introduction, Methodology, Experiments, Results, Conclusion, plus an extensive appendix), presents nontrivial new ideas and theory, and backs them with substantial empirical evidence. I do not see fatal methodological, theoretical, or experimental flaws that would warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to LLM reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces Simulation-Calibrated Scientific Machine Learning (SCaSML), an inference-time correction framework for high-dimensional PDE solvers. Given a trained surrogate solution \( \hat u \) (e.g., PINN or GP), the authors derive a “Structural-preserving Law of Defect”, a new semi-linear PDE governing the error \( \tilde u = u - \hat u \), and solve this defect PDE at inference time with Multilevel Picard (MLP) Monte Carlo.  

They prove that, under regularity and surrogate-accuracy assumptions, the SCaSML error is bounded by the product of the surrogate error and the Monte Carlo error and derive an improved scaling law \(O(m^{-\gamma - 1/2 + o(1)})\). Experiments on several PDEs up to 160 dimensions show 20–80% error reductions over PINN/GP surrogates and improvements over naive MLP solvers.

---

## Strengths

1. **Conceptual clarity and structural insight.**  
   The key idea of formulating the surrogate error as the solution of another semi-linear PDE that preserves the structure of the original (Fact 2.3 and Lemma D.9) is clean and technically insightful. Unlike vague “residual correction” heuristics, the defect PDE is derived by explicit subtraction and retains the same semi-linear form as Equation (1), which is crucial for re-using existing high-dimensional Monte Carlo solvers.

2. **Link between surrogate accuracy and simulation complexity.**  
   The analysis around Assumption 2.4 / E.2 and Theorem 2.5 / E.6 is a strong point: the authors show that the nonlinearity driving the defect PDE, \(\tilde F\), has its “source term” at the origin equal to the residual \(\epsilon\) (Definition D.8, Remark D.10). Lemma E.5 then bounds key MLP variance/complexity quantities linearly in \(e(\hat u)\), leading to the global \(L^2\) error bound (Eq. (9) and Eq. (54)) where the total error is \(E(M,N) \cdot C_F e(\hat u)\). This gives a principled explanation for why better surrogates make the correction cheaper and more accurate.

3. **Improved scaling law is both theoretical and empirical.**  
   Corollary 2.6 / E.8 carefully analyzes the joint training–inference budget and shows that if the surrogate error scales as \(e(\hat u)\sim m^{-\gamma}\), then SCaSML achieves \(O(m^{-\gamma - 1/2 + o(1)})\) when allocating another \(m\) samples to inference-time simulation. The scaling derivation via Lambert–W-like manipulations is quite nontrivial but internally consistent. Figure 4(b–d) nicely corroborates this: in the viscous Burgers experiments, the log–log slopes for SCaSML are clearly steeper than those for the base GP surrogate across dimensions, supporting the claimed acceleration.

4. **Nontrivial high-dimensional experiments and clear quantitative gains.**  
   The experimental suite is unusually broad for this type of paper: linear convection–diffusion, viscous Burgers, LQG/HJB, and diffusion-reaction with oscillatory solutions, up to 160 dimensions. Table 1 is particularly informative: across all problem families and dimensionalities, SCaSML almost always has the lowest \(L^2\), \(L^\infty\), and \(L^1\) errors. For example, on LQG at 160d, the PINN surrogate has \(L^2\) error \(1.12\mathrm{e}{-1}\) vs SCaSML’s \(9.94\mathrm{e}{-2}\), and the naive MLP is catastrophically bad (\(5.27\mathrm{e}{+0}\)). Similar 20–80% reductions hold across VB-PINN, VB-GP, and DR. The authors also provide repeated-run statistics (Tables 2–6) with confidence intervals and paired \(t\)-tests, which strengthens empirical credibility.

5. **Figures effectively convey key behaviors.**  
   - **Figure 3a** (violin plots of absolute error) clearly shows that SCaSML shrinks both the mean and spread of errors relative to surrogates and MLP, across multiple problems.  
   - **Figure 3b** (inference-time scaling) demonstrates monotonically decreasing errors as Monte Carlo samples increase, showing that additional compute is indeed well used.  
   - **Figure 4a** illustrates the “product of errors” viewpoint and compares surrogate-only vs MC-only vs SCaSML scaling regimes in a way that is easily digestible.  
   - The numerous violin plots in Appendix G (e.g., Figure 5–9) and pointwise error maps (Figures 23–27) confirm that improvements are not driven by a few outliers but are broadly distributed across test points.

6. **Algorithmic details and reproducibility.**  
   Appendix B and C provide significant detail about both PINN/GP surrogates and the MLP implementation. Algorithm 1 is explicit about how MLP_Law_of_Defect recursively calls itself over levels and time samples, and Algorithms 2–3 articulate stabilization techniques (thresholding and Hutchinson estimator). This level of detail is uncommon and makes the work much more reproducible.

7. **Interesting practical angle: “elastic compute” and small-vs-large PINN tradeoff.**  
   The fixed-budget analyses in Appendix G.7–G.8 (e.g., Figure 28–29) are compelling: for a given wall-clock budget, using a smaller surrogate + SCaSML often beats training a larger PINN or pure MLP. This concretely illustrates the inference-time scaling narrative and is likely to be useful for practitioners in high-dimensional PDEs.

---

## Weaknesses

1. **Some mathematical definitions and core equations have inconsistencies/typos that need fixing.**  
   - In Definition 2.1, the terminal condition of the defect PDE (Eq. (4)) is written as \(\tilde u(T,\mathbf{y}) = g(\mathbf{y}) - \tilde u(T,\mathbf{y})\), which is dimensionally wrong; it should be \(\tilde u(T,\mathbf{y}) = g(\mathbf{y}) - \hat u(T,\mathbf{y})\), consistent with the text and Eq. (5).  
   - There is notational fluctuation between \(\tilde u\), \(\breve u\), \(\bar u\) and \(\hat u\) for the defect throughout Sections 2 and Appendix D (e.g., Eq. (5) suddenly uses \(\breve u\), Def. D.8 uses \(\bar u\), and A.4 mistakenly redefines \(\hat u\) as the defect), which makes the derivations harder to follow and creates potential for hidden errors.  
   - Equation (7) has a typo: the PDE is written as \(\frac{\partial\hat{u}}{\partial r} + \mathcal{L}\breve{u} + \tilde F(\cdot)=0\), but the time derivative should be \(\partial \breve u / \partial r\).  
   While these are likely fixable, they occur at the core of the method and should be cleaned up to avoid confusion, especially for readers trying to re-derive the law of defect.

2. **Regularity and surrogate-accuracy assumptions are strong but not well justified in realistic high-dimensional settings.**  
   Assumption E.2 / F.1 requires that the true defect \(\tilde u\) has uniformly bounded \(W^{1,\infty}\) norm and, for quadrature MLP, Gevrey-class bounds on all powers of \(\left(\partial_t + \frac{\sigma^2}{2}\Delta_x\right)^k \tilde u\) (Assumption F.1, Item 3). Moreover, the surrogate error is summarized in a scalar \(e(\hat u)\) that simultaneously controls the \(L^\infty\) residual and the defect’s Sobolev norms. In practice, for high-dimensional PINNs and GPs, such strong regularity and tight residual bounds on the learned solution are rarely verified; indeed, surrogate residuals can be highly non-smooth and localized. The paper would benefit from at least some discussion or numerical evidence that these assumptions are not wildly violated for the considered experiments (e.g., empirical norms of \(\epsilon\) vs. training size, or counterexamples where SCaSML misbehaves).

3. **Positioning w.r.t. existing error-control / hybrid ML–simulation work could be deeper.**  
   The discussion in Section 2.2 distinguishes classical defect-correction for FEM and Newton-type iterative solvers, which is useful, but the paper omits several directly relevant strands:
   - Recent works on high-dimensional PINNs explicitly aiming at the curse of dimensionality, such as Hu et al. (2023, “Tackling the Curse of Dimensionality with PINNs”), which offer alternative inference-time or training-time strategies for scaling and might serve as baselines.  
   - Broader model reduction / hybrid schemes where ML surrogates are corrected or coupled with simulators (e.g., Meuris 2023, “Model Reduction for Dynamical Systems: Machine Learning and Memory”).  
   Currently, the narrative somewhat suggests that using surrogates as control variates in stochastic solvers is largely new, but there is prior variance-reduction and debiasing literature in PDE Monte Carlo that should be acknowledged and contrasted.

4. **Baseline coverage, while nontrivial, is still limited in several dimensions.**  
   Table 1 compares SR (PINN or GP), naive MLP, and SCaSML. However:
   - For PDEs where other high-dimensional solvers exist (e.g., deep BSDE, Deep Ritz, other MLP variants), those are not included as baselines. In particular for HJB/LQG, there are alternative deep control solvers; and for the oscillatory diffusion–reaction problem, more standard Monte Carlo or variance-reduced schemes could serve as references.  
   - The MLP hyperparameters are held fairly fixed (most experiments use 2 levels and \(M=10\)), making it unclear how well-tuned the naive MLP baseline is. For instance, in Table 1 the LQG MLP errors are catastrophically large (\(5.27\mathrm{e}{+0}\)–\(5.63\mathrm{e}{+0}\) in \(L^2\)), while SCaSML works well. It is plausible that better clipping, different time-sampling, or more levels would rescue MLP to some extent; at minimum, a sensitivity study of MLP hyperparameters would make the comparison more convincing.

5. **Some aspects of the algorithmic design are heavily heuristic and not theoretically tied back.**  
   The method introduces several heuristic choices that are important in practice but mostly absent from the theory:  
   - Clipping thresholds are chosen very differently for MLP vs SCaSML (e.g., LQG uses 10 vs 0.1; diffusion–reaction uses 10 vs 0.01). These clearly influence stability and bias, but there is no systematic procedure or sensitivity study; the text only notes “smaller magnitude of the defect” as informal justification.  
   - Hutchinson-based Laplacian estimation is used for some high-dimensional problems (HJB) but not for others (diffusion–reaction); the trade-off between approximation bias and variance is not analyzed.  
   - The choice between quadrature vs full-history MLP is empirical; although Appendices E and F give separate analyses, the main text does not clearly explain when each is preferable or how to choose parameters such as \(\alpha\) in the time-sampling density.  
   Connecting these knobs to theory or at least showing robustness experiments would strengthen the practical side.

6. **Clarity and redundancy issues in exposition.**  
   While the paper is rich in detail, there are several clarity issues:  
   - Section 2.3 in the main text partially repeats definitions and explanations that are then fully reintroduced in Appendix B.2, yet the main text omits explicit forms for \(\Phi\) and instead refers to Eq. (28) “in the appendix”, which interrupts the logical flow for readers not jumping back and forth.  
   - Some references are clearly duplicated or corrupted (e.g., the Hu et al. reference repeated multiple times in the reference list, Page 12) which hints at possible copy-paste issues.  
   - There are occasional small errors in Algorithm 1 (e.g., “gQ,[s,T](t)” instead of \(q^{Q,[s,T]}(t)\), and inconsistent use of indices), again making it harder to implement directly.

7. **Scope of theory vs experiments could be better aligned.**  
   The main theoretical results (Theorem 2.5, E.6, F.5) are proved for semi-linear heat equations with \(\mu=0\) and \(\sigma=s I_d\), and under fairly restrictive assumptions. In contrast, experiments use more general drifts and nonlinearities (e.g., Burgers with nonlinear convection term, HJB with \(-\|\nabla u\|^2\)), and the defect PDE in Eq. (7) is written in generic semi-linear form. While it is standard to prove results in a simplified setting, the paper sometimes slides from “for simplicity we present results for \(\mu=0, \sigma\propto I\)” to broader claims about “semi-linear parabolic PDEs” without clearly stating which experiments are actually covered by the assumptions. A more explicit statement of what is and is not covered by the theory would avoid overclaiming.

---

## Potentially Missing Related Work

1. **Hu et al., “Tackling the Curse of Dimensionality with Physics-Informed Neural Networks,” 2023.**  
   This work directly addresses high-dimensional PDEs with PINNs via stochastic dimension gradient descent; it is a strong point of comparison for the LQG and diffusion–reaction benchmarks and also conceptually relevant, since it is another way to mitigate dimensionality issues. It should be discussed in the Introduction / related work context around high-dimensional SciML solvers, and ideally used as an additional baseline or at least qualitatively compared in Section 3.3 and 3.4.

2. **Meuris, “Model Reduction for Dynamical Systems: Machine Learning and Memory,” 2023 (PhD thesis).**  
   This dissertation surveys and develops ML-based model reduction for dynamical systems, including hybrid simulation–ML methods. While not directly implementing a defect-PDE correction, it is thematically very close to SCaSML’s idea of combining surrogates and simulation. It would be appropriate to cite in Section 1–2 when discussing hybrid scientific computing and to situate SCaSML among broader model reduction and surrogate-enhancement approaches.

Depending on space, the authors might also reference other Monte Carlo variance-reduction/control-variate techniques in PDE contexts, as several of their arguments (e.g., using the surrogate as a control variate for the stochastic estimator) resonate strongly with that literature.

---

## Questions

1. **On the strength and realism of Assumption E.2 / F.1.**  
   Can the authors provide empirical evidence that the surrogate residuals \(\epsilon\) and defect norms \(\|\tilde u\|_{W^{1,\infty}}\) behave in a way compatible with Assumption E.2 for their experiments? For example, plotting empirical \(\sup |\epsilon|\) vs. training size \(m\) on Burgers/HJB would give some comfort that \(e(\hat u)\) is not an arbitrary quantity.

2. **Terminal-condition typo and notation consistency.**  
   Please confirm that Eq. (4) should read \(\tilde u(T,y) = g(y)-\hat u(T,y)\) and Eq. (7) should have \(\partial \breve u / \partial r\) rather than \(\partial \hat u / \partial r\). If so, could you clean up the notation (choosing one symbol, say \(\tilde u\), throughout) and verify that all subsequent derivations use the corrected forms?

3. **Sensitivity to clipping thresholds and MLP hyperparameters.**  
   How sensitive is SCaSML’s performance to the choice of clipping threshold and number of levels/samples \(N,M\)? For instance, in Table 1 for LQG, could naive MLP be brought closer to SCaSML by better-tuned clipping, or is the difference robust? A small ablation, perhaps included in the appendix, would help clarify whether SCaSML’s advantages are structural or mostly coming from more conservative parameter settings.

4. **Range of PDEs covered by the current theory.**  
   Many experiments involve general drifts, non-quadratic nonlinearities, and nontrivial terminal data. To what extent do the proofs in Appendices E–F extend beyond the \(\mu=0, \sigma=s I_d\) semi-linear heat setting, and what would be required to extend them to Burgers or HJB-type nonlinearities? Clarifying this would help readers understand how much of the empirical success is inside or outside the provable regime.

5. **On the choice between quadrature and full-history MLP.**  
   The paper empirically uses both quadrature and full-history MLP (e.g., Figures 10–15), and the appendices give separate analyses. Can the authors summarize, perhaps in Section 2.3, a practical guideline for which variant to use when, and whether the improved scaling law (Corollary 2.6 / E.8) is expected to hold similarly for quadrature MLP in realistic settings?

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The central derivations (defect PDE, Feynman–Kac representation, MLP error bounds) are well grounded in existing theory, and the product-form error bound is plausible and supported experimentally. There are some typographical/notation issues and strong assumptions whose realism is not fully justified, but I did not find fatal logical errors.

---

## Presentation Rating

3: good.  
The paper is generally well written, with clear motivation, helpful figures (e.g., Figures 1–4), and extensive appendices. However, notation inconsistencies, some errors in key equations/algorithms, and duplicated references hurt clarity, especially for readers trying to re-derive the math.

---

## Contribution Rating

3: good.  
The idea of a structurally exact defect PDE for surrogate error, combined with high-dimensional MLP at inference time, is a meaningful conceptual and practical contribution. The improved scaling result and substantial high-dimensional experiments are valuable to the SciML community, even though related work coverage and theoretical assumptions could be strengthened.

---

## Overall Rating

8: Accept, good paper (poster).  
The paper presents a well-motivated and technically solid framework that bridges surrogate-based SciML and stochastic simulation for high-dimensional PDEs, backed by both theory and a strong empirical study. Some details (notation, assumptions, baselines, and parameter sensitivity) need refinement, but the core idea and results are sufficiently strong and interesting to merit acceptance.

---

## Reviewer Confidence

4: confident.  
I am familiar with PDE-based SciML, Feynman–Kac / BSDE methods, and Monte Carlo variance reduction, and I checked the main derivations and experiments in detail. Some of the more intricate constant-tracking in the appendices was not re-derived line by line, but the overall reasoning appears consistent.