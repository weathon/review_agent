## Summary

This paper proposes Randomized Asymmetric Chain of LoRA (RAC-LoRA), a parameter-efficient fine-tuning method that iterates low-rank updates by fixing one sketch matrix per chain step and optimizing the other. The authors reformulate each step as a projected gradient update on the full parameter matrix and prove non-asymptotic convergence rates for smooth non-convex objectives and linear rates under the Polyak-Łojasiewicz condition, extending the analysis to SGD, random reshuffling, and federated settings.

## Strengths

- **Novel theoretical lens on LoRA chaining.** The derivation showing that fixing a random sketch matrix turns the RAC-LoRA update into a Lipschitz-smooth projected gradient step on $W$ (Eq. 3–4) is conceptually fresh and provides a clean analytical handle that prior LoRA/COLA analyses lacked. This directly addresses the loss of smoothness in standard LoRA reparameterization noted by Sun et al. (2024).
- **Broad convergence coverage.** The paper proves an $O(1/T)$ stationary-point rate for smooth non-convex problems (Theorem 5.3) and a linear rate under PL (Theorem 5.5), and it extends these results across GD, SGD, random reshuffling, and federated random reshuffling (Table 1). The federated extension (Fed-RAC-LoRA) is a nice addition that leverages the single-trainable-matrix property for communication efficiency.
- **Controlled validation in convex settings.** The linear regression experiments (Figure 2) cleanly validate the predicted dependence of the convergence speed on the rank ratio $r/n$, and the logistic regression results (Appendix A) provide additional alignment between theory and practice.
- **Empirical parameter efficiency in low-capacity regimes.** On the MLP MNIST task with rank $r=1$ (Table 3), RAC-LoRA approaches COLA performance while training only one matrix per block (133 or 912 parameters vs. COLA’s 1K), supporting the claim that asymmetric chaining can reduce trainable parameters.

## Weaknesses

### Fatal
None.

### Major
- **Abstract and introduction overstate FPFT equivalence.** The abstract claims the method provides “provable guarantees of convergence to the same solution as FPFT” without qualification. However, Theorem 5.3 only guarantees convergence to *a* stationary point in the general non-convex regime, not the *same* stationary point reached by full-parameter fine-tuning. Theorem 5.5 guarantees convergence to the global optimum only under the Polyak-Łojasiewicz condition, which is neither established for the neural network benchmarks nor flagged in the abstract framing. This misrepresents the theoretical contribution.
- **Neural network experiments test an unanalyzed optimizer.** The non-convex experiments on RoBERTa (Section 6.2.1) and MLP (Section 6.2.2) use AdamW—with adaptive preconditioning, momentum, and weight decay—to solve the subproblem in Algorithm 1. The convergence analysis explicitly covers Gradient Descent, Random Reshuffling, and SGD (Table 1), and makes no provision for adaptive inner solvers. Because the theorems do not apply to AdamW, the non-convex empirical results cannot substantiate the repeated claim that experiments “validate our theoretical results” in non-convex settings.

### Minor
- **Section 3 counterexample conflates representational and algorithmic limitations.** The quadratic problem in Eq. (2) has a full-rank optimal solution, and rank-$r=1$ methods cannot represent it. While the divergence of LoRA/COLA at step size $1/L$ is a valid algorithmic issue (due to loss of Lipschitz smoothness), convergence to a suboptimal stationary point is partly an inevitable representational limitation. The paper would benefit from distinguishing approximation error from optimization error more sharply.
- **Disconnect between Algorithm 1 and the analyzed update.** Algorithm 1 allows “any iterative solver” for the subproblem in Step 4, whereas Theorems 5.3–5.5 assume the specific GD updates in Eq. 3–4. It is unclear whether the theorems assume exact minimization of the local quadratic model (yielding Eq. 3) or a finite number of inner-loop steps, and how the inner-loop cost enters the rate.
- **Missing experimental details.** Table 3 omits the chain length $T$ and total training budget, making it impossible to assess whether RAC-LoRA and COLA are compared at equal compute or total parameter updates.

### Trivial
- Imprecise phrasing in Section 4: randomization does not “prevent optimization within a restricted subspace”; each step is still confined to the column/row space of the sketch matrix, and the method cycles through randomly drawn subspaces.

## Nice-to-Haves
- A non-convex benchmark where the subproblem is solved with ordinary GD or SGD (as analyzed), to test whether the theory maps directly to practice without the confounding factor of AdamW.
- Tasks where FPFT substantially outperforms single LoRA, to demonstrate that RAC-LoRA closes a meaningful capacity gap rather than operating in a regime where all low-rank methods perform similarly.
- An analysis of the accumulated update rank (e.g., singular-value spectrum of $\sum_t \Delta W^t$) to verify empirically that chaining builds higher-rank adaptations.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Computational cost of Eq. 3 comparable to FPFT:** This criticism misunderstands parameter efficiency. Computing $\nabla f(W^t)$ is inherent to any backpropagation-based method; RAC-LoRA saves memory and optimizer state by training only one small matrix per step, not by skipping the forward/backward pass.
- **Purely representational counterexample:** While partially valid, this is overstated. COLA is explicitly designed to overcome rank limitations through chaining, so its failure in the example (including divergence) is not purely representational. RAC-LoRA’s success with the same rank shows the randomization mechanism matters.
- **Baseline tuning concerns on GLUE:** The paper uses its own experimental setup; differences from Hu et al. (2021) do not necessarily indicate incorrect tuning, and the paper is transparent about its methodology.
- **Typos, grammar, and formatting issues:** These are parser artifacts from the PDF extraction, not author errors.
- **Missing appendix or proofs:** The parser strips appendix sections; they exist in the original submission.

## Novel Insights

The paper’s reformulation of chained asymmetric LoRA as randomized subspace descent is genuinely novel and provides a clean analytical handle on a widely used empirical technique. The observation that fixing one sketch matrix turns the update into a projected gradient step on the full parameter matrix (Eq. 3–4) is particularly insightful, because it allows the authors to import classical smooth-optimization and sketching theory into the PEFT domain. If the authors align their empirical validation with their theoretical assumptions and tighten their claims to what is actually proven, this framework could become a standard reference for optimization analysis of low-rank adaptation.

## Suggestions
1. Revise the abstract and introduction to qualify the FPFT-matching claim: state that matching FPFT holds under the PL condition (or convexity), and present the general non-convex result as convergence to a stationary point with an $O(n/r)$ slowdown.
2. Include at least one non-convex experiment where the subproblem is solved with ordinary GD or SGD as analyzed, or explicitly frame the neural network experiments as heuristic evaluations and acknowledge that the theory does not currently cover AdamW.
3. Clarify in Algorithm 1 and Section 5.1 whether Step 4 is implemented via the closed-form update (Eq. 3) or an iterative solver, and specify the number of inner steps if the latter.

## Score and Decision

**Score: 5.0 (Reject)**

**Calibration anchors and comparison:**

- **LoRA-RITE (VpWki1v2P8, avg 8.67, Accept Oral):** Strong theory with clear practical improvements and aligned experiments. The current paper is well below this because it lacks the experimental validation needed to back its central claims.
- **SD-LoRA (5U1rlpX68A, avg 7.50, Accept Oral):** Novel decomposition with supportive theory and strong empirical results across benchmarks. The current paper has a comparably interesting theoretical perspective but weaker empirical alignment.
- **GoLore (udtrtwkvk5, avg 5.25, Reject):** Similar structure—insightful theory about random projections fixing convergence issues, but reviewers found the experiments misaligned (GoLore used only for final 20% of training) and gains marginal. The current paper has more comprehensive theory coverage but a more severe central overclaim in the abstract, placing it roughly at the same level.
- **Manifold-LoRA (c2OtbtZXFC, avg 4.75, Withdrawn):** Theory was deemed correct but the application to LoRA was poorly motivated and experiments showed no advantage. The current paper’s theory is more directly relevant to LoRA and the MLP results show clearer differences, so it sits slightly above this anchor.
- **ACSS (uu2CorJCUi, avg 4.80, Withdrawn):** A reviewer proved the main linear-convergence theorem wrong. The current paper’s core theorems appear mathematically sound, so it is comfortably above this anchor.

Relative to these anchors, the paper under review has a genuinely novel theoretical framework but suffers from a central overclaim in the abstract and a theory–experiment disconnect in the non-convex evaluation. These issues place it in the borderline-to-reject range, comparable to GoLore but slightly below due to the abstract’s misrepresentation of the general non-convex guarantee. A score of 5.0 reflects solid intellectual contribution undermined by presentation and alignment flaws that a revision could resolve.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>