---
job_id: a2f7e1a4-b90c-4f5b-a4b8-d8f0dbab7a2b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: mHRuCmc9lo.pdf
paper: Robust Decision Making With Partially Calibrated Forecasts
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about calibration, robust decision-making, and learning theory, all core ICLR topics (uncertainty quantification, robust optimization, and theory for ML-powered decisions).

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Methodology/Theory, Experiments, Results, Conclusion). The work is technically nontrivial, the math is coherent, and the empirical section, while limited, is sound. There are no obvious fatal methodological or theoretical errors, nor signs of data leakage or similar issues.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The paper text contains no instructions targeting automated reviewers or other manipulative content.

---

# Expected Review Outcome:

## Summary

The paper studies how a risk-neutral decision maker should act on predictions that satisfy *partial* (rather than full) calibration guarantees. It defines an ambiguity set of conditional expectations consistent with a given class of $\mathcal{H}$-calibration constraints, then formulates robust decision making as a minimax problem over this set. The authors derive a dual characterization of the minimax-optimal policy for finite-dimensional $\mathcal{H}$, show that decision calibration (and any strictly stronger $\mathcal{H}$ containing its tests) collapses the robust policy to the plug-in best response, and instantiate the framework for two practically common sources of partial calibration (self-orthogonality from squared loss and bin-wise calibration), with a small empirical study on regression datasets.

## Strengths

1. **Clear and principled robust decision-theoretic framework.**  
   The formulation of $\mathcal{H}$-calibration in Equation (2)/(3) and the corresponding ambiguity set $\mathcal{Q}$ in Equation (4) provide a very clean way to represent “all realities consistent with the calibration guarantees.” The minimax robust policy in Equation (5) is a natural optimization problem, and the “interpolating property” between minimax-conservative and plug-in-aggressive policies, nicely illustrated in **Figure 1**, gives an intuitive picture that will likely be influential for subsequent theoretical work in calibration and decision making.

2. **Technically solid dual characterization of robust policies.**  
   **Theorem 3.1** gives a useful structural result: for finite-dimensional $\mathcal{H}=\mathrm{span}\{h_i\}$, the worst-case map $q^\star$ has a pointwise form obtained by solving a small convex problem in $p$, with dual multipliers $\lambda_i^\star\in\mathbb{R}^d$. The proof in Appendix A (Pages 13–14) uses Sion’s minimax theorem and convex duality in a careful and internally consistent way. The explicit Lagrangian in the proof,
   \[
   L(q,\lambda)=\mathbb{E}\big[\operatorname{val}(q(f(X)))\big]+\sum_i\lambda_i\cdot\mathbb{E}[h_i(f(X))(q(f(X))-f(X))]
   \]
   and the resulting dual $G(\lambda)$ make the derivation transparent and check out mathematically.

3. **Sharp and conceptually important “collapse” at decision calibration.**  
   The transition result in **Section 4.1**, formalized as **Theorem 4.1** and **Theorem 4.2**, is particularly compelling: once $\mathcal{H}$ contains the $|\mathcal{A}|$ decision-region indicators $\mathbf{1}_{R_a}$, the adversarial tilt disappears ($q^\star(v)=v$ a.e.) and the minimax-optimal robust rule becomes exactly the plug-in best response. **Figure 2** nicely visualizes this “sharp transition” from fully conservative ($\mathcal{H}=\emptyset$) to fully trusting ($\mathcal{H}\supseteq \mathcal{H}_{\text{dec}}$). This gives a precise decision-theoretic semantics to decision calibration that is significantly stronger than the previously known swap-regret interpretation.

4. **Practical instantiations from standard training pipelines.**  
   **Proposition 4.4** shows that any regression model with a linear last layer trained to stationarity under squared loss satisfies a “self-orthogonality” moment condition, which is exactly an $\mathcal{H}$-calibration condition for $\mathcal{H}=\{h_j(v)=e_j^\top v\}$. This is a very common setting in practice (linear regression, MLP with linear head), and the paper demonstrates how Theorem 3.1 then yields an explicit one-dimensional (for $d=1$) or low-dimensional convex program in $p$ and a scalar/matrix dual variable $\lambda$ or $\Lambda$, which is computationally tractable.

5. **Closed-form robust policies for bin-wise calibration.**  
   **Proposition 4.5** (Page 9) gives an especially clean result: with bin-wise calibration $\mathcal{H}_{\text{bin}}=\{\mathbf{1}_{B_j}\}$ and linear utilities over a finite action set, the worst-case belief $q^\star$ is piecewise constant, taking the bin mean $m_j$ on each bin, and the robust policy simply best-responds to $m_j$. This provides a nice interpretation of standard post-hoc calibration techniques (histogram binning, isotonic regression) as directly yielding robust decision rules with no additional optimization.

6. **Experiments validating qualitative theoretical predictions.**  
   The empirical evaluation in **Section 5** uses two real regression datasets (Bike Sharing and California Housing) with a two-layer MLP trained under squared loss (hence approximately satisfying the self-orthogonality condition). **Table 1** directly compares mean utilities of the plug-in vs robust policies under i.i.d. and two adversarial distributions. The patterns clearly match the theory: under adversaries tuned to harm the plug-in rule, the robust policy improves utility (e.g., Bike Sharing: 0.393 vs 0.412; California Housing: 0.155 vs 0.166), and under the robust-tuned adversary the robust policy is never worse. The nominal i.i.d. performance is slightly better for plug-in, as expected. Even though the experiments are limited in scope, **Table 1** effectively illustrates the minimax gain vs nominal cost tradeoff.

7. **Careful extension to approximate calibration.**  
   Appendix B develops the $\varepsilon$-approximate $\mathcal{H}$-calibration setting with norm-bounded slack in the moment constraints. **Theorem B.1** shows that the same structural characterization holds, but with a penalized dual objective $G(\lambda)-\varepsilon\sum_i\|\lambda_i\|_2$. **Theorem B.2** and **Proposition B.3** bound the suboptimality of the plug-in policy and the value degradation under approximate decision and bin-wise calibration. These results make the theory more realistic and practically relevant, since exact calibration is rarely achieved.

## Weaknesses

1. **Strong reliance on linear-in-outcome utilities and finite action sets.**  
   The central Assumption 2.1 restricts utilities to be linear in the outcome vector $v$ for each action, which is standard in expected-utility formulations but excludes risk-averse or variance-sensitive utilities that depend nonlinearly on $v$ (the authors briefly acknowledge this in Section 6). Many decision makers in practice are exactly interested in dispersion-sensitive criteria, CVaR, or other nonlinear functionals. While the authors note that some nonlinear utilities can be linearized over sufficiently rich bases (end of Section 6, referencing Gopalan et al. 2024b; Lu et al. 2025), they do not formalize how their dual characterization would scale in such settings or what happens when the resulting basis is high-dimensional or infinite-dimensional. At minimum, the paper should discuss more concretely how the framework would generalize when $u(a,\cdot)$ is convex or concave but not linear, and whether analogues of Theorem 3.1 exist under additional structure.

2. **Limited empirical scope and absence of competing robust-decision baselines.**  
   The experiments are restricted to two 1D regression tasks (Bike Sharing and California Housing), with a very simple 3-action decision set and a single parametric utility form $u(a,y)=\alpha ay - C(a)$. While **Table 1** is consistent with the theory, there is no comparison to alternative robust decision-making approaches (e.g., distributionally robust optimization with Wasserstein balls, simple confidence-interval–based worst-case rules, or heuristic “shrink the forecast toward the mean” policies). This weakens the empirical claim that the proposed minimax policy is practically useful beyond theoretical interest. For instance, a baseline that robustifies via a simple additive or multiplicative shrinkage of $f(X)$ could be easily constructed and might perform comparably; without such baselines, it is unclear whether the dual-based robust policy is meaningfully better than simpler heuristics in realistic finite-sample settings.

3. **Operationalization and estimation of expectations in finite samples is under-specified.**  
   The robust policies depend on expectations such as $\mathbb{E}[f(X)^2]$ and the dual objective $G(\lambda)$ defined in Section 4.2 and Appendix B. The authors state in Section 5 that “We use the calibration data to substitute any population level expectation that is needed,” but do not provide details on how sensitive the resulting $\lambda^\star$ and $q^\star$ are to estimation error, nor any finite-sample calibration guarantees. For example, in the self-orthogonality-based robust policy (Section 4.2), the dual objective for $d=1$,
   \[
   G(\lambda) = \mathbb{E}\left[\min_{p\in[0,1]}\{\mathrm{val}(p)+\lambda f(X)p\}\right]-\lambda\,\mathbb{E}[f(X)^2],
   \]
   is maximized over $\lambda$ using empirical expectations; there is no discussion of regularization, variance, or stability. In practical deployments with moderate $n$, this could lead to overfitting in the dual space, but the paper does not explore this or provide robustness guarantees under sampling noise.

4. **Decision calibration results hinge on task-specific calibration that may be unrealistic in many deployments.**  
   The sharp transition result in **Section 4.1** is conditioned on $f$ satisfying decision calibration *for the specific downstream utility $u$ and action set $\mathcal{A}$*. While **Corollary 4.3** notes that one can unify requirements for multiple (finitely many) tasks via a union test class $\mathcal{H}_{\mathrm{dec}}^{\mathrm{all}}$, this still assumes that all relevant downstream utilities are known at training time. In many real-world ML-as-a-service settings, downstream tasks and utilities are unknown or evolve after model deployment. The paper briefly notes this advantage only in the “if you can influence the forecaster” regime (Section 4.2), but does not concretely address the more difficult setting where $u$ is unknown, which is exactly what prior work on omniprediction and swap-regret–based guarantees tries to handle. A clearer comparison between the present minimax-optimality guarantees and the “unknown agent” or omnipredictor formulations (e.g., Kleinberg et al. 2023; Gopalan et al. 2024b; Okoroafor et al. 2025) would strengthen the positioning.

5. **Some key arguments for the sharp transition could be more transparent, especially regarding measurability and tie-breaking.**  
   The proof of **Theorem 4.1** (Page 14) introduces regions $R_a$ based on a measurable tie-broken best-response policy $a_{\mathrm{BR}}$. It uses Jensen’s inequality over each region and conditional expectations $\mu_a = \mathbb{E}[f(X)\mid f(X)\in R_a]$. While correct in spirit, the proof glosses over edge cases such as regions with measure zero and the requirement that $\mu_a\in R_a$ (stated using convexity of $R_a$ but not justified for arbitrary tie-breaking). The assumption that $\mathrm{val}$ is convex is induced by linear utilities over a finite action set, but this is invoked somewhat implicitly in several places (e.g., in the “collapse” argument after Theorem 4.2). These details matter because the entire decision-calibration story hinges on the formal equivalence
   \[
   \mathbb{E}[u(a_{\mathrm{BR}}(f(X)),q(f(X)))] = \mathbb{E}[u(a_{\mathrm{BR}}(f(X)),f(X))]
   \]
   for all $q\in\mathcal{Q}$. The paper would benefit from a more explicit statement of the assumptions under which $\mu_a\in R_a$ and from a brief discussion of how degenerate cases (e.g., overlapping boundaries or flat regions of $u$) are handled.

6. **Scope of $\mathcal{H}$-calibration classes actually achievable in practice is not thoroughly discussed.**  
   The paper heavily uses the abstract family of linear test classes $\mathcal{H}=\mathrm{span}\{h_i\}$ but sidesteps how challenging it is to enforce a given $\mathcal{H}$. While this is partially by design (the paper’s goal is decision-making *given* an $\mathcal{H}$-calibrated forecaster), the claims about practical upshots (e.g., “decision calibration is tractable in high dimensions” in Section 1.2, or “zero-bias and bin-wise calibration can be obtained cheaply” in Section 4.2) are not accompanied by discussion of sample complexity, computational cost, or degradation in predictive accuracy compared to unconstrained training. This abstraction is fine at the theory level, but it leaves readers with an incomplete picture of the overall pipeline cost to achieve the desired calibration guarantees.

7. **Experiments restricted to 1D outcomes and simple discrete actions; no exploration of higher-dimensional outcomes or continuous action spaces.**  
   Although the theory accommodates general $d$ and arbitrary (albeit finite) $\mathcal{A}$, the experiments only consider $d=1$ and $\mathcal{A}$ of size 3. This makes the dual problem trivial and does not stress-test the more general convex programs described in Section 4.2 and Appendix B. There is no indication of the computational scaling of computing $a_{\mathrm{robust}}$ in higher-dimensional or larger-action settings, which is crucial for assessing the practical viability of the dual-based approach.

8. **Some notational and typographical issues slightly hinder readability.**  
   There are a number of small but noticeable glitches in the references and text (e.g., inconsistent citation formatting on Page 11–12 such as “- K. Kuhn et al. (2019) Daniel Kuhn …” and garbled author lines like “F. George Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie.”). In **Theorem 4.2**, the equation for $a_{\text{robust}}$ is typeset with unusual spacing (“a _ {\text {r o b u s t} }”), making the notation a bit hard to parse. While these are minor, they detract from an otherwise well-polished theoretical paper.

## Potentially Missing Related Work

1. **Hu, Tan, Zou (2026), “Conformal Robustness Control: A New Strategy for Robust Decision.”**  
   This work develops a robust decision-making framework based on conformal prediction sets, directly addressing decision-making under distributional uncertainty with calibration-like guarantees. It appears closely related conceptually to the minimax-robust lens in this paper and should be discussed in the Related Work section (Section 1.2) and possibly contrasted in the introduction of Section 2 when explaining how robust decisions are derived from uncertainty sets $\mathcal{Q}$.

2. **Yeh, Christianson, Wu (2025), “End-to-End Conformal Calibration for Optimization Under Uncertainty.”**  
   This paper studies end-to-end learning of uncertainty sets for downstream optimization problems, fusing calibration and robust decision making. It is directly relevant to the theme of learning predictions that are useful for robust decisions. It would be appropriate to cite and compare in Section 1.2 and in the concluding discussion (Section 6), clarifying similarities and differences between conformal-uncertainty–set–based and $\mathcal{H}$-calibration–based ambiguity sets.

3. **Fang, Ke (2025), “Information Seeking for Robust Decision Making under Partial Observability.”**  
   While more on the planning / RL side, this work proposes robust decision making under uncertainty and partial observability, with explicit information-seeking components. It is relevant to the broader robust decision-making literature referenced in Section 1.2 and could usefully broaden the discussion there, emphasizing complementary approaches to robustness (planning and information gathering vs. calibration-based ambiguity sets).

4. **Noorani, Kiyani, Pappas (2025), “Human-AI Collaborative Uncertainty Quantification.”**  
   This paper considers collaborative uncertainty quantification between humans and AI, aimed at robust decision making under uncertainty. Given the overlapping authorship and thematic connection to trustworthy predictions and decisions, it should be explicitly mentioned in Section 1.2 and possibly in the conclusion to situate the present work within the broader program of uncertainty-aware decision making.

5. **TMLR 2025: “Learning Robust Penetration Testing Policies under Partial Observability: A systematic evaluation.”**  
   Although domain specific (cybersecurity / penetration testing), this work investigates robust sequential decision policies under partial observability. It is tangentially related and could be briefly mentioned as an application-side example in the robust decision-making literature surveyed in Section 1.2, to signal awareness of robust decision making under uncertainty in various domains.

6. **NeurIPS 2025 Workshop: “Calibrated surrogate losses for robust classification with a reject option.”**  
   This paper studies calibration of surrogate losses for robust classification with an explicit reject option, linking calibration to robust decisions in a different way. It is directly relevant to the interplay between calibration and decision strategies and should be cited near the discussion of decision calibration (Section 4.1) and/or in the Related Work section.

## Questions

1. **Practical enforcement of decision calibration.**  
   For realistic multiclass problems and nontrivial action sets, how do the authors envision enforcing decision calibration in practice without severe sample complexity or accuracy loss? Can they provide a concrete algorithmic strategy and complexity estimate (beyond citing prior decision-calibration work) for training a decision-calibrated predictor for a given $u$ and $\mathcal{A}$?

2. **Behavior under nonlinear utilities.**  
   Suppose a decision maker is risk-averse with a utility of the form $u(a,v)=\mathbb{E}[U(a,Y)]-\beta\operatorname{Var}[U(a,Y)]$ or uses CVaR-type objectives. Do the authors believe that a variant of Theorem 3.1 can be obtained if $u(a,\cdot)$ is convex (or concave) but not linear? If so, what are the main obstacles, and is there a natural way to define an $\mathcal{H}$-calibration condition that pins down not just means but also variance or tail information?

3. **Finite-sample robustness of the dual-based robust policy.**  
   In the self-orthogonality example (Section 4.2), how sensitive is the calculated $\lambda^\star$ (and thus $q^\star$ and $a_{\mathrm{robust}}$) to sampling error when the expectations in $G(\lambda)$ are approximated empirically? Could the authors provide either a finite-sample bound or at least an empirical sensitivity analysis (e.g., bootstrap variability of $\lambda^\star$ and the resulting utilities) to demonstrate that the robust policy is not overly brittle?

4. **Comparisons to simpler heuristics or alternative robust methods.**  
   In the experiments (Section 5), how does the proposed robust policy compare to more naive approaches, such as shrinking $f(X)$ toward the global mean or using a simple interval-based worst-case choice over a confidence band around $f(X)$? Adding such baselines in **Table 1** or an additional table would help clarify whether the dual-based robust policy provides tangible benefits in moderate-sample regimes.

5. **Computational scaling and higher-dimensional experiments.**  
   Could the authors comment quantitatively on the computational cost of computing $a_{\mathrm{robust}}$ in higher dimensions, say $d=10$ or $d=100$, with a moderate number of actions? Have they tried even small synthetic experiments in such settings to verify that the convex programs remain tractable and that the theory behaves as expected?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The theoretical developments are internally consistent, the duality arguments are carefully laid out, and the experiments, while limited, correctly implement the proposed robust policy. The main limitations are scope assumptions (linearity in $v$, finite $\mathcal{A}$) and the lack of more comprehensive empirical validation or finite-sample analysis, not obvious errors.

## Presentation Rating

3: good.  
The paper is generally well written, clearly structured, and mathematically precise. Figures (**Figure 1** and **Figure 2**) and **Table 1** are informative and support key conceptual claims. Some notation glitches and minor typographical issues in the references slightly reduce polish but do not impede understanding.

## Contribution Rating

3: good.  
The paper makes a meaningful conceptual and technical contribution: a robust decision-theoretic framing for partially calibrated forecasts, a nontrivial dual characterization, and a sharp identification of decision calibration as the threshold at which plug-in best response is minimax-optimal. The empirical section is modest, which limits the practical impact somewhat, but the theoretical insights are likely valuable to the calibration and robust decision-making communities.

## Overall Rating

8: Accept, good paper (poster).  
The work presents a solid and conceptually clean framework for robust decision making with partially calibrated forecasts, backed by rigorous theory and a particularly nice result on the collapse at decision calibration. While empirical validation is limited and some assumptions are restrictive, the theoretical contribution and clarity of the framework merit acceptance.

## Reviewer Confidence

4: confident.  
I am comfortable with the learning-theoretic, calibration, and robust optimization aspects, and I carefully checked the main derivations and proofs. Some application-side implications and finite-sample issues could benefit from further discussion, but they do not affect my assessment of the core contributions.