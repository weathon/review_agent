# MOSIC: Model-Agnostic Optimal Subgroup Identification with Multi-Constraint for Improved Reliability

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Current subgroup identification methods typically follows a two-step approach: first estimate conditional average treatment effects (CATEs) and then apply thresholding or rule-based procedures to define subgroups. While intuitive, this decoupled approach fails to incorporate key constraints essential for real-world clinical decision-making—such as subgroup size and propensity overlap. These constraints operate on fundamentally different axes than CATE estimation and are not naturally accommodated within existing frameworks, thereby limiting the practical applicability of these methods. We propose a unified optimization framework that directly solves the primal constrained optimization problem to identify optimal subgroups. Our key innovation is a reformulation of the constrained primal problem as an unconstrained differentiable min-max objective, solved via a gradient descent-ascent algorithm. We theoretically establish that our solution converges to a feasible and locally optimal solution. Unlike threshold-based CATE methods that apply constraints as post-hoc filters, our approach enforces them directly during optimization. The framework is model-agnostic, compatible with a wide range of CATE estimators, and extensible to additional constraints like cost limits or fairness criteria. Extensive experiments on synthetic and real-world datasets demonstrate its effectiveness in identifying high-benefit subgroups while maintaining better satisfaction of constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MOSIC, a model-agnostic optimization framework for treatment subgroup identification under multiple real world constraints, including minimum subgroup size, propensity score overlap, and optionally safety or fairness constraints. Unlike standard two-stage approaches that first estimate CATE and then apply post-hoc thresholding or filtering, MOSIC directly formulates subgroup identification as a constrained optimization problem. Experiments on both synthetic datasets and real data show that MOSIC consistently achieves higher subgroup treatment effects while satisfying the specified constraints.

### Strengths
**Good Motivation**

The paper transfers the subgroup identification problem from the standard two-stage CATE estimation and filtering into an optimization problem with constraints. The idea is elegant, and this kind of shift is decision-focused and real-world aware, aligning with the emerging trend in causal machine learning.

**Theoretical Proof**

I like that the paper provides several theoretical lemmas to support their algorithms, e.g., the strict local min–max point in Lemma 2. I didn't check the full details of the proof, but they seem correct (although I could be mistaken). The authors also provide experiments on both synthetic and real-world data to show the proposed method’s performance compared with baselines.

### Weaknesses
My main concern is about the practical utility of the proposed methods. The reason why the standard two-stage CATE methods are widely accepted is because they are easy to understand, easy to use, and easy to interpret. While I appreciate the theoretical and empirical results that MOSIC presents in the paper, the key question is: is the gain worth the additional complexity?

**Easy to understand?**

The idea is straightforward: shifting from estimation to optimization. However, the transition from Problem I to Problem IV sometimes feels reactive rather than intentional. Section 4 in particular reads like a narrated debugging process (“we tried X but it was unstable, so we added Y...”), rather than a clean forward design narrative. This hurts clarity, especially for theoretical readers.

**Easy to use?**

There is no discussion of computational cost or efficiency. The two real-world datasets used have sample sizes of 13,361 and 6,516, but there is no reporting of runtime or resource overhead relative to simpler two-stage baselines. This omission limits the reader’s ability to judge whether the method is practical for real-time or large-scale deployment. And for many clinicians who may need to use this method, will it still be practical if they do not have sufficient computational resources?

**Easy to interpret?**

While the method supports decision-tree backbones in Section 5.3, the paper never shows a concrete learned subgroup rule. For example, a clinician might prefer an interpretable patient subgroup like: “patients aged > 60 should not take the drug.” But this kind of explicit rule is missing from the paper. For deployment and clinical trust, this is a significant gap.

### Questions
1. Line 156: The paper claims that the method naturally accommodates more general linear and ratio-form constraints. Could the authors clarify what kinds of constraints can be naturally supported by MOSIC, and what kinds cannot? Additionally, how does the computational or optimization complexity scale as more constraints are added, does it grow linearly, or is there a risk of exponential blow-up? I also wonder would there be a finite-sample guarantees with constraints? 

2. I understand that strict local min–max is a standard target for nonconvex optimization. However, since we are dealing with patient data, robustness is a critical concern. You mention running experiments over 100 random train/test splits, but do you also evaluate stability across different random initialization seeds of the optimizer (not just different data splits)? In other words, do different seeds lead to meaningfully different subgroups being selected, even if the average ATE is similar? Is the identified subgroup S(x) stable across initializations?

3. As mentioned in the weaknesses section, providing runtime and interpretability analysis would significantly strengthen the paper, particularly for clinical decision. Even an approximate runtime comparison against two-stage baselines, and at least one example of a human-readable decision-tree subgroup rule, would be very helpful.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MOSIC, a unified optimization framework for identifying optimal treatment subgroups under multiple practical constraints such as subgroup size and propensity overlap. Unlike conventional two-step CATE-based approaches, MOSIC formulates the task as a constrained min–max optimization and solves it using a modified Gradient Descent–Ascent algorithm. The authors establish theoretical guarantees for feasibility and local optimality and validate the approach on both synthetic and ICU datasets (eICU, MIMIC-IV). The framework is model-agnostic, compatible with various CATE estimators and model architectures, and can incorporate additional constraints such as fairness and safety.

### Strengths
- Introduces an optimization-based framework that integrates multiple constraints directly into subgroup identification, addressing the limitations of post-hoc filtering methods.  
- Provides theoretical guarantees for convergence, feasibility, and approximate satisfaction of nonconvex constraints.  
- Empirical evaluations show improvements in subgroup ATE, size control, and covariate balance on both synthetic and real-world datasets.  
- Offers an open-source implementation with transparent algorithmic details, enhancing reproducibility.

### Weaknesses
- The method’s scalability in high-dimensional settings is not fully addressed, especially given the use of $\gamma$-GDA with many constraints.  
- Limited exploration is provided on uncertainty quantification or confidence intervals for subgroup ATEs, which is critical for decision-making in clinical applications.
- Numerical stability concerns of $\gamma$-GDA are only discussed theoretically; empirical results showing training dynamics (e.g., convergence plots, constraint violation over iterations) would substantiate the stability claim.

### Questions
- The imbalance criterion should be SMD > 0.1 rather than 0.2 according to Austin (2009).  
- The abbreviation “LR” in line 079 likely refers to Logistic Regression, which should be clearly defined when it first appears.
- What is the computational complexity and scalability of $\gamma$-GDA when applied to high-dimensional data or a large number of constraints?
- Can the authors discuss how subgroup interpretability is maintained when using black-box backbones such as MLPs and whether any post-hoc explainability tools were explored?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes MOSIC, a model-agnostic framework for identifying treatment-benefit subgroups under multiple constraints such as minimum size and overlap. It formulates a constrained optimization with ReLU-gated Lagrangian penalties and provides local min–max guarantees. Experiments on synthetic data and two ICU datasets show that MOSIC achieves higher subgroup treatment effects with fewer imbalance violations than prior subgroup discovery methods.

### Strengths
The paper proposes a clear optimization framework for subgroup identification under multiple constraints. The formulation is well defined and the paper is clearly written. The experiments are consistent across datasets and show some improvement in subgroup ATE and balance, though mainly within controlled and limited settings

### Weaknesses
1. The largest concern is the limited baselines: the paper includes CAPITAL, causal trees, causal forests, and outcome-weighted learning, but exclude stronger HTE estimators such as DR-learner, R-learner, BART, and locally centered causal forests. These are standard in healthcare and could change the ranking. Overlap-weighted estimators are also missing and align directly with the overlap constraint. It would also be helpful to discuss recent constrained policy methods from off-policy evaluation, safe RL, and constrained contextual bandits that optimize under budgets or risk

2. Experimental setup lacks robustness and diagnostic depth.
The evaluation relies on AIPTW without clear cross-fitting, which can bias subgroup ATEs when selection and estimation share nuisance models. The reliability metric is only the count of covariates with standardized mean difference>0.2 after IPTW, which is coarse and does not measure magnitude or joint imbalance. Overlap enforcement is only evaluated during training, not validated on test sets. 

3. Causal validity in ICU cohorts is not well supported.
The treatment window (10 hours before to 24 hours after ICU admission) risks immortal time bias if early deaths are excluded. Confounding from indication severity is possible, especially for steroids, yet no sensitivity analysis or negative control outcomes are reported. The added safety constraint on GCS is useful, but without external validation or clinician input, it’s unclear whether the selected subgroups are clinically meaningful or plausible.

### Questions
1. How do you ensure unbiased AIPTW estimates if subgroup selection and nuisance estimation share data? Was any cross-fitting or sample splitting applied?
2. Can you evaluate overlap on test data, e.g., showing the distribution of estimated propensities or proportion outside [0.05, 0.95] within selected subgroups?
3. Why not compare against overlap-weighted baselines or modern CATE learners like DR-/R-learner or BART?
4. Could you report uncertainty for subgroup ATEs (e.g., influence-function or bootstrap CIs) and include a sensitivity analysis for unmeasured confounding in the ICU data?

### Soundness
2

### Presentation
3

### Contribution
2
