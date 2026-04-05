=== CALIBRATION EXAMPLE 1 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  The title suggests a method for “combining gradients of different orders,” which broadly matches the paper’s intent, but it is somewhat vague and does not reflect the paper’s actual technical object: a step-size/update rule built from repeated Hessian-vector interactions in a quadratic setting. Since the method is not presented in a general gradient-order framework with clear formalism, the title is more aspirational than precise.

- **Does the abstract clearly state the problem, method, and key results?**  
  No. The abstract is extremely underspecified. It does not define the optimization problem, the proposed algorithm, or what “products of Hessian matrices of different orders” means. It also does not report any concrete experimental results, complexity claims, or conditions under which the method works. For ICLR standards, the abstract is far too vague to support the claimed contribution.

- **Are any claims in the abstract unsupported by the paper?**  
  Yes. The abstract claims “third-order and even higher-order” methods with “faster convergence rates,” but the paper does not provide a rigorous theory establishing higher-order convergence, nor does it compare rates in a standard sense. The empirical section only reports iteration counts on a very narrow quadratic test and does not substantiate a general convergence-rate claim.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  The paper does motivate the study of step-size methods for quadratic minimization and cites steepest descent, Barzilai–Borwein, and CBB. However, the gap is not clearly articulated. It is not explained why existing line-search or spectral step-size methods are insufficient, nor why a “higher-order” combination rule is needed beyond existing acceleration ideas. The motivation stays at an intuitive level and does not identify a crisp technical limitation in prior work.

- **Are the contributions clearly stated and accurate?**  
  Only partially. The paper claims to construct “a new descent method by combining the gradient with products of the Hessian matrix of different orders,” but the actual algorithm is never presented in a clean, general form. The main contribution seems to be the GOC update rule in Eq. (22–24), yet the derivation is hard to follow and the statement of novelty is not precise enough for an ICLR paper.

- **Does the introduction over-claim or under-sell?**  
  It over-claims. Statements like “Whave developed third-order and even higher-order, which offer faster convergence rates” go beyond what is established. At the same time, it under-sells by failing to clearly specify the mathematical object being optimized or the exact relationship to known spectral gradient methods.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  No. The method is not reproducible from the current description. The derivation from SD/CBB to GOC is opaque, and key definitions are inconsistent or missing. In particular:
  - Eq. (8) defines \(r_k\), but the algebra and indexing are not cleanly explained.
  - Eqs. (19–24) appear to introduce a sequence of quantities involving \(\mu_k\), \(r_k\), and repeated Hessian-free operations, but the formulas are not consistently derived.
  - Algorithm 1 is not a complete algorithmic specification: it lacks clear initialization, termination details, and unambiguous update equations.
  
  For ICLR, reproducibility is a major concern, and this method is not described at the level expected.

- **Are key assumptions stated and justified?**  
  Only partly. The paper focuses on convex quadratic objectives with SPD Hessian, but then implicitly suggests broader applicability (“higher-order methods”) without showing how the method extends beyond this case. There is also an implicit assumption that repeated Hessian-vector products can be obtained cheaply, but the computational cost is not analyzed. The choice of a fixed step size \(d\) in the implementation of Hessian-free iterations is introduced without justification.

- **Are there logical gaps in the derivation or reasoning?**  
  Yes, several.
  - The paper asserts that SD is “first-order” and CBB is “second-order” based on how \(r_k\) evolves, but this is not a standard order notion and is not formally defined.
  - The leap from observing patterns in eigenvalue components to proposing a polynomial-like update in Eq. (22–24) is not rigorously derived.
  - The explanation that “by updating once in the negative gradient direction and once in the positive gradient direction with a fixed step size \(d\), we can calculate the final updated point” is not fully justified mathematically; it is unclear how this yields the claimed polynomial in \(A\).
  - The paper also appears to conflate finite-difference approximations of Hessian-vector products with exact Hessian operations, but does not discuss approximation error.

- **Are there edge cases or failure modes not discussed?**  
  Yes. The method seems tailored to diagonalizable convex quadratics with well-behaved eigenstructure. There is no discussion of:
  - nonquadratic objectives,
  - nonconvex landscapes,
  - noisy gradients,
  - ill-conditioned cases where repeated Hessian-vector products may amplify numerical error,
  - cases where \(r_k\) may fall near an eigenvalue and cause instability in the polynomial update,
  - whether the update can become non-descent in general.

- **For theoretical claims: are proofs correct and complete?**  
  There are no real proofs supporting the core claims. The paper makes convergence assertions and order interpretations, but these are not established with the rigor expected at ICLR. The statement citing Raydan and Svaiter about CBB convergence is borrowed from prior work, but the proposed GOC method lacks proof of convergence, monotonicity, or rate. The theoretical section is therefore incomplete.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Only very weakly. The paper claims a faster higher-order method, but the experiments are limited to a synthetic diagonal quadratic with chosen eigenvalues. This does not test:
  - generality beyond diagonal quadratics,
  - robustness to rotations/non-diagonal Hessians,
  - performance on real optimization problems,
  - behavior on nonquadratic objectives,
  - scalability.

  So the experiments do not adequately test the core claim of a broadly useful optimization method.

- **Are baselines appropriate and fairly compared?**  
  The baselines BB and CBB are relevant, but the comparison is incomplete. There is no comparison to standard gradient descent with exact line search, conjugate gradient on quadratics, spectral gradient variants beyond BB/CBB, or any modern first-order optimizer. Since the method is framed as an optimization advancement, ICLR would expect stronger and broader baselines.

- **Are there missing ablations that would materially change conclusions?**  
  Yes. A crucial ablation would be the effect of the claimed “order” parameter \(m\): the paper mentions higher-order methods, but experiments only show one GOC variant without systematically varying \(m\). Another important ablation would be computational cost versus iteration count, since the method appears to use multiple gradient evaluations per update. Without wall-clock or gradient-evaluation accounting, the iteration-count advantage may not translate to efficiency.

- **Are error bars / statistical significance reported?**  
  No. The paper reports deterministic iteration counts on a synthetic problem and one random initialization scheme, but there are no repeated trials, variability measures, or significance tests. Given that the method appears sensitive to initialization, this is a notable gap.

- **Do the results support the claims made, or are they cherry-picked?**  
  The results are suggestive but not convincing. The examples in Figure 3 compare iteration counts on one fixed diagonal problem and one random initialization experiment. The method appears better in those settings, but the scope is too narrow to support broad claims of superiority or “higher-order” convergence. The fact that BB “could not satisfy the stop condition” in one case is also unusual and may reflect the chosen setup rather than a general limitation of BB.

- **Are datasets and evaluation metrics appropriate?**  
  The evaluation metric is iteration count or stopping-condition count on a toy quadratic problem. This is not sufficient for an ICLR-level optimization paper unless the paper is purely theoretical and proves a new convergence theorem, which it does not. No real datasets or standard benchmarks are used.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  Yes, extensively. The derivation in Sections 2 and 3 is hard to follow and often mathematically unclear. The paper appears to use \(r_k\), \(\mu_k\), and eigenvalue arguments in a way that is not consistently defined. The algorithm description is especially difficult to parse because the update equations are not cleanly presented and the meaning of intermediate quantities is ambiguous.

- **Are figures and tables clear and informative?**  
  The figures are referenced but not sufficiently explained. Figure 1 is meant to illustrate SD and CBB geometrically, but the narrative does not make the geometry precise enough. Figure 2 claims to show how \(\mu\) varies with \(x\), yet the connection to the method is not transparent. Figure 3 is potentially informative, but because the experiment design is underspecified, the figure does not fully validate the method. There are no tables summarizing key quantitative results, which would have helped.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  No. The paper does not clearly acknowledge that the method is derived and tested only for convex quadratic objectives. It also does not discuss the computational cost of extra gradient/Hessian-vector evaluations.

- **Are there fundamental limitations they missed?**  
  Yes:
  - The method seems specialized to SPD quadratic objectives.
  - It may require repeated gradient evaluations and/or Hessian-vector products, which can be costly.
  - There is no convergence proof outside the toy case.
  - The claimed “higher-order” structure may not survive in non-diagonal or non-quadratic settings.
  - The algorithm may be sensitive to the choice of fixed step size \(d\).

- **Are there failure modes or negative societal impacts not discussed?**  
  There are no obvious direct societal harms from the method itself. The main issue is methodological limitation rather than social impact. However, over-claiming an optimization method without adequate validation could mislead practitioners about its practical usefulness.

### Overall Assessment
This paper has an interesting ambition: to reinterpret spectral gradient methods through eigenvalue-dependent “order” and to build a new update rule from repeated Hessian-vector interactions. However, at ICLR standards, the submission falls well short in clarity, rigor, and empirical validation. The main method is not precisely specified, the theoretical claims are not proven, and the experiments are too narrow to support the broad performance claims. The work may contain a potentially interesting idea for the quadratic SPD setting, but in its current form it does not meet ICLR’s acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes GOC, a new gradient-based optimization method intended to combine “different orders” of gradient/Hessian information, positioned as a higher-order extension of steepest descent (SD) and Cauchy-Barzilai-Borwein (CBB). The main idea is to interpret the step-size dynamics through a spectral/eigenvalue viewpoint and then construct an update using repeated Hessian-free products so that multiple powers of the Hessian acting on the gradient are combined in one iteration.

### Strengths
1. **Attempts to unify several classical gradient methods under a common spectral perspective.**  
   The paper explicitly relates SD, BB, and CBB through the quantity \(r_k\) and discusses their behavior in terms of eigenvalue directions. This is conceptually aligned with what ICLR reviewers often value: a clear lens that connects algorithm design to known optimization dynamics.

2. **Tries to propose a computationally cheap way to obtain higher-order information.**  
   The method is motivated by Hessian-free products rather than explicit Hessian formation, which is a reasonable direction if the goal is to obtain higher-order effects with low overhead.

3. **Includes some empirical comparison against BB and CBB.**  
   The paper reports iteration counts on a quadratic example and claims GOC reaches the stopping criterion in fewer iterations than BB and CBB in the presented settings. Even though the evaluation is limited, the authors do provide at least one numerical comparison rather than only theory.

4. **The manuscript has an identifiable algorithmic proposal.**  
   Despite presentation issues, Algorithm 1 communicates a concrete iterative scheme with input/output structure, which makes the contribution more than a purely qualitative discussion.

### Weaknesses
1. **The theoretical formulation is unclear and in parts mathematically inconsistent.**  
   The paper’s central claims about “first-order,” “second-order,” and “higher-order” methods are not rigorously established. The derivations around \(r_k\), \(\mu_i\), and the polynomial update appear informal and in several places ambiguous, making it difficult to verify what the method actually computes.

2. **The novelty relative to existing gradient/CBB/BB methods is not convincingly demonstrated.**  
   The paper suggests GOC is a higher-order extension of SD/CBB, but it does not clearly isolate what is fundamentally new versus a reparameterization or heuristic combination of known gradient evaluations. For an ICLR submission, the bar is not just “some combination” but a clearly justified new idea with distinct algorithmic and empirical benefits.

3. **Reproducibility is weak.**  
   The experiments are not sufficiently specified: the exact problem dimension, conditioning, initialization distribution, stopping criteria implementation, step-size/d parameters, and whether results are averaged over multiple runs are not fully described. There is also no public code, no hyperparameter sensitivity study, and no details that would let another researcher reproduce the reported iteration counts reliably.

4. **Empirical evaluation is too narrow for ICLR standards.**  
   The paper evaluates on essentially one quadratic toy setting with a single family of examples. ICLR typically expects broader, more realistic benchmarks and stronger evidence that a method generalizes beyond a hand-crafted convex quadratic case.

5. **The claimed “higher-order” advantage is not convincingly supported by complexity or convergence analysis.**  
   The paper argues that using more Hessian-free multiplications yields higher order and faster descent, but it does not provide a formal convergence theorem, iteration complexity, or a proof that the order terminology corresponds to standard optimization notions. The improvement is asserted more than established.

6. **Presentation and exposition are not at the level expected by ICLR.**  
   Beyond parser artifacts, the paper’s organization and notation are difficult to follow. Key quantities are introduced without precise definitions, the algorithm description is hard to parse, and some statements appear more heuristic than scientific. This significantly lowers clarity and confidence.

### Novelty & Significance
**Novelty: low to moderate.** The idea of leveraging repeated Hessian-vector products and interpreting step sizes spectrally is not entirely new, and the paper does not clearly demonstrate a decisive conceptual advance over existing gradient methods such as SD, BB, CBB, or polynomial acceleration ideas in optimization.

**Significance: low.** As submitted, the contribution is limited by weak theoretical grounding and a very narrow experimental evaluation. Against ICLR’s acceptance standards, this would likely fall below the bar because the paper does not yet show convincing evidence of a broadly impactful or technically robust method.

**Clarity: low.** The paper is difficult to parse and lacks precise definitions and proofs.

**Reproducibility: low.** Insufficient experimental and algorithmic details are provided.

### Suggestions for Improvement
1. **Provide a rigorous mathematical definition of GOC.**  
   Clearly state the update rule, assumptions, and exactly how the “order” is defined. If the method is a polynomial in the Hessian applied to the gradient, write this explicitly and prove the equivalence.

2. **Add formal convergence analysis.**  
   Include theorem statements for quadratic objectives, assumptions on \(A\), and a proof of convergence rate. If “higher order” is a key claim, quantify it using standard optimization metrics rather than informal spectral intuition.

3. **Strengthen the experimental section substantially.**  
   Evaluate on multiple problems, including ill-conditioned quadratics and non-quadratic smooth objectives. Compare not only against BB/CBB but also strong modern baselines relevant to the claimed setting.

4. **Report full reproducibility details.**  
   Specify dimensions, conditioning, initialization distributions, stopping conditions, parameter settings, and whether results are averaged over seeds. Release code and add ablations on the role of the repeated Hessian-vector products.

5. **Clarify the algorithmic cost.**  
   Since the method uses multiple gradient/Hessian-free evaluations per iteration, compare wall-clock time and total gradient/Hessian-vector products, not just iteration counts. A method that reduces iterations but increases per-iteration cost may not actually be more efficient.

6. **Improve the exposition and notation.**  
   Define every variable precisely once, use consistent notation for gradients, Hessians, and spectral quantities, and rewrite the derivations in a structured theorem/lemma style. This would substantially improve credibility and readability.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add experiments on standard benchmark suites beyond one synthetic quadratic, such as CUTEst quadratic/nonquadratic problems or the ICLR-relevant optimization benchmarks used for gradient-method papers. Without diverse problems, the claim that GOC is a generally better “higher-order” method is not believable.
2. Compare against strong and relevant baselines: SD, BB1/BB2, CBB, Yuan, Dai, Hager-Zhang, and other modern spectral gradient variants under the same stopping criteria and same function/gradient evaluation budget. The current comparison is too narrow to support any efficiency claim.
3. Report wall-clock time and total oracle cost, not just iteration counts. GOC appears to require extra gradient evaluations per step, so fewer iterations do not necessarily mean better efficiency.
4. Add ablations on the order \(m\) and on the fixed step size \(d\). The paper claims a higher-order mechanism, but it does not show whether the benefit persists across orders or is just a tuned special case.
5. Test sensitivity to conditioning, dimension, and initialization. The method is motivated by eigenvalue structure, so it must be shown whether performance collapses or improves as the condition number changes.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a correct derivation of the update rule from first principles, including exact definitions of \(r_k\), \(\mu_k\), and how the \(A^j g_k\) terms are obtained. As written, the mathematical connection between the proposed algorithm and the claimed higher-order combination is not rigorous enough to trust.
2. Prove descent and convergence guarantees. ICLR reviewers will expect at least monotonic decrease or global convergence conditions, plus clear assumptions under which the method is valid.
3. Analyze computational complexity per iteration in terms of gradient/Hessian-vector products and total oracle calls. The core contribution is “efficient,” but the paper never establishes whether the claimed gain survives the added computation.
4. Clarify what “order” means mathematically and why the method is genuinely higher-order rather than just repeated spectral updates. The current argument is heuristic and does not justify the central conceptual claim.
5. Quantify robustness to non-quadratic objectives. The analysis is entirely quadratic/eigenvalue-based, so the paper needs to show what carries over, what breaks, and why.

### Visualizations & Case Studies
1. Add convergence plots of objective gap, gradient norm, and cumulative oracle calls on multiple benchmarks, not just a single illustrative quadratic. This would expose whether GOC truly converges faster or only appears better under one setup.
2. Include per-iteration plots of \(r_k\) and the proposed higher-order coefficients on problems with different condition numbers. The key claim is that GOC exploits eigenvalue structure; these plots would show whether that actually happens.
3. Provide a case study where GOC fails or becomes unstable, especially on ill-conditioned or non-quadratic objectives. A method-level failure analysis is needed to judge whether the approach is robust or brittle.
4. Visualize trajectories in low-dimensional problems to compare zig-zagging, step-length behavior, and progress along principal curvature directions. This would directly test the paper’s geometric story about SD, CBB, and GOC.
5. Show convergence versus evaluation budget with fair normalization across methods. Without this, iteration-count plots can be misleading because GOC uses extra inner computations.

### Obvious Next Steps
1. Formalize the method as a general framework for arbitrary order \(m\), then derive a practical algorithm for choosing \(m\) adaptively. The paper currently presents a heuristic construction, not a reusable method.
2. Extend the theory and experiments to nonconvex objectives. For an ICLR submission, restricting the claim to a special convex quadratic case is too narrow to support the broader optimization contribution.
3. Derive an implementable version that does not rely on fragile hand-crafted algebraic manipulations or exact quadratic structure. If the method cannot be applied generally, its contribution is limited.
4. Release a complete algorithm with precise stopping criteria, parameter choices, and computational cost accounting. The current presentation is not reproducible enough for an ICLR standard.


# Final Consolidated Review
## Summary
This paper proposes GOC, an optimization update intended to combine information from repeated Hessian-vector interactions and gradient steps, and frames SD, BB, and CBB as increasingly higher “orders” within a shared spectral viewpoint. The core claim is that this construction yields a faster descent method on convex quadratic objectives, with a simple numerical comparison suggesting fewer iterations than BB and CBB on the toy problems studied.

## Strengths
- The paper tries to connect classical spectral gradient methods through a unifying eigenvalue-based lens, using the quantity \(r_k\) to interpret SD/CBB behavior. This is a potentially useful conceptual framing, and it is one of the few parts of the paper that reflects a coherent motivation.
- It does at least present a concrete algorithmic proposal rather than remaining purely descriptive. Algorithm 1 and the quadratic examples make the intended update rule identifiable, and the experiments do show the method outperforming BB/CBB on the specific synthetic setups reported.

## Weaknesses
- The main mathematical development is not rigorous enough to support the central claim. The paper repeatedly refers to SD as “first-order,” CBB as “second-order,” and GOC as “higher-order,” but this notion of order is not defined in any standard optimization sense, nor is the claimed polynomial/Hessian-free derivation made precise. As written, the update formulas are difficult to verify and the connection between the heuristic spectral story and the actual algorithm remains unclear.
- Reproducibility and evaluation are both very weak. The experiments are limited to a single synthetic convex quadratic family with ad hoc initialization choices, and the paper reports only iteration counts. There is no wall-clock accounting, no systematic study of the claimed order parameter \(m\), no sensitivity analysis, and no testing on nontrivial benchmarks. This makes the efficiency claim unconvincing.
- The paper does not establish convergence or even basic descent guarantees for the proposed method. It borrows intuition from SD/CBB and cites prior convergence results for CBB, but GOC itself is not analyzed with a theorem or proof. For an optimization method paper, that is a major omission.

## Nice-to-Haves
- A clearer and more standard presentation of the update rule, ideally rewritten as an explicit polynomial in \(A\) applied to the gradient with all intermediate quantities defined once.
- Broader experiments on additional quadratic instances with varying conditioning, and at least one nonquadratic smooth benchmark to show whether the method survives beyond the toy setting.

## Novel Insights
The paper’s most interesting idea is not the specific formula, but the attempt to reinterpret step-size methods as manipulating the spectrum of the quadratic through repeated Hessian-vector interactions. That viewpoint could, in principle, motivate a family of polynomially filtered gradient methods. However, the submission does not yet turn that intuition into a clean, general, or verifiable optimization method; instead, it stays at the level of heuristic spectral storytelling with a narrow demonstration.

## Potentially Missed Related Work
- Polynomial acceleration and spectral gradient methods beyond BB/CBB — relevant because the paper’s “higher-order” construction is closest in spirit to this line of work.
- CUTEst-style optimization benchmarks — relevant as a standard evaluation suite for gradient-method papers.
- Yuan / Dai / Hager-Zhang style step-size methods — relevant because the paper positions itself among spectral and alternate step gradient methods.

## Suggestions
- Rewrite the method section from first principles: define \(r_k\), \(\mu_k\), and the claimed higher-order update precisely, and state explicitly what is being computed at each step.
- Add at least one theorem proving descent or convergence for the quadratic case, and be explicit about the assumptions under which it holds.
- Evaluate GOC under a fair oracle budget, comparing total gradient/Hessian-vector products and wall-clock time against BB, CBB, and other strong spectral-gradient baselines.
- Include an ablation on the order \(m\) and on the fixed step size \(d\), since the current paper does not show that the purported higher-order benefit is robust rather than tuned.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject
