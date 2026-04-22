# Accelerated Mirror Descent Method through Variable and Operator Splitting

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Accelerated Mirror Descent (Acc-MD) is derived from a discretization of an accelerated mirror ODE system using a variable--operator splitting framework. A new Cauchy--Schwarz type inequality enables the first proof of linear accelerated convergence for mirror descent on a broad class of problems. Unlike prior methods based on the triangle scaling exponent (TSE), Acc-MD achieves acceleration in some cases where TSE fails. Experiments on smooth and composite optimization tasks show that Acc-MD consistently outperforms existing accelerated variants, both theoretically and empirically.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose an Accelerated Mirror Descent (Acc-MD) framework derived from discretization of accelerated mirror ODE systems and a variable-operator splitting framework. 
By introducing a new Cauchy–Schwarz type inequality based assumption, Acc-MD achieves acceleration under relative strong convexity and relative smoothness while TSE-based methods may fail. Experimental results show that Acc-MD consistently outperforms baseline methods, achieving strong performance in non-Euclidean settings.

### Strengths
1. The authors provide rigorous theoretical results based on Lyapunov functions and Bregman methods, adapting them to the paper’s assumptions.

2. The authors provide scenarios where assumption A2 enables acceleration for Acc-MD while TSE-based methods fail. These observations are interesting and impressive.

### Weaknesses
1. The Acc-MD framework builds directly on the VOS framework and on discretization of continuous-time systems. It is a natural combination of existing methods, but the innovation is limited.

2. It is non-trivial to verify the assumption A2, which is the foundation of most analysis in this paper. Both Theorem 3.5 and Theorem 3.6 depend on smoothness of $f$ and strong convexity of $\phi$, which makes A2 unrealistic.

3. The authors present only one example where TSE fails while A2 holds. This is insufficient to claim superiority over TSE-based methods.

4. The authors claim that "we give the first accelerated linear convergence guarantee under relative strong convexity...", which is misleading. This result relies on assumption A2, which is neither proven to be commonly guaranteed nor straightforward to verify.

5. Since this work proposes an acceleration framework, its experiments only report loss v.s. steps, with no wall-clock time or FLOPs. This evaluation is inadequate because each step includes an inner argmin loop. Moreover, the experiments focus solely on synthetic toy problems, which are useful for intuition but not sufficient for supporting the conclusion. The authors should also benchmark their method on large-scale, real-world datasets such as those in [1].

Reference:

[1] Šehić, Kenan, et al. (2022) "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso".

### Questions
See mentioned weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes an accelerated mirror-descent (Acc-MD) flow obtained by a variable-splitting continuous dynamics and an implicit–explicit discretization. The authors claim four main contributions: (i) the new split Acc-MD flow and a simple discretization; (ii) an accelerated linear convergence guarantee under relative strong convexity with a rate expressed via a constant $C_{f,\phi}$ that stems from a Cauchy–Schwarz style inequality (Assumption A2); (iii) an extension to the convex case via a perturbation + homotopy argument recovering $O\left(\frac{C_{f,\phi}}{k^2}\right)$ and (iv) extensions to composite/constrained problems and numerical benchmarks showing empirical improvements.

### Strengths
- The paper gives a clear derivation of the split flow, a semi-implicit discretization, and an implementable Algorithm 1. The derivation and algorithmic steps are presented in a self-contained way.
- The authors provide Lyapunov functions and detailed proofs that support the claimed decay and linear/sublinear rates under stated assumptions. The perturbation/homotopy path to handle is presented with the associated bounds.
- The paper includes numerical experiments (entropic MD, quartic objective, constrained quadratic on simplex, LASSO) and reports iteration/time comparisons versus several baselines. The results indicate that Acc-MD converges faster for these problems.

### Weaknesses
- The paper refers to the continuous-time accelerated/variational literature (Wibisono et al., Krichene et al., Nesterov) and positions Acc-MD as a new discretization. However, the manuscript does not clearly separate what is genuinely new vs. what is a modest variant of existing accelerated mirror flows and discretizations. The related work and motivation sections cite the prior art, but the paper lacks a precise theorem-level comparison that isolates the distinct mathematical mechanism or performance gain of Acc-MD beyond those prior frameworks.
- The accelerated linear rate under relative strong convexity is contingent on the relative Cauchy–Schwarz inequality and the constant $C_{f,\phi}$. However, (A2) is nonstandard and its applicability/interpretation is not convincingly justified. Computing or estimating $C_{f,\phi}$ in realistic problems could be difficult. The paper gives some sufficient conditions (Theorem 3.6) and examples where (A2) holds, but practical guidance and a discussion of the assumption’s restrictiveness are limited. 
- The method and proofs require knowledge/estimates of $\mu$ and $C_{f,\phi}$. The authors themselves note this as a limitation (highlighting the need for adaptive schemes). The experiments use fixed choices and show empirical gains, but there is no study of sensitivity to mis-estimated constants or proposed adaptive rules. 
- The paper shows Acc-MD beats ABPG/BPG/NAG on these instances, but (i) some examples appear tailored to highlight relative-smoothness advantages, and (ii) there is no evaluation on a larger scale or diverse real-world problems to demonstrate generality. Wall-clock comparisons are given, but implementation and hyperparameter details (step size selection, how $C_{f,\phi}$  was chosen) should be reported for reproducibility.
- The authors claim Acc-MD can achieve acceleration in cases where ABPG does not (e.g., $\gamma=1$), but the paper provides limited theoretical or experimental diagnostics to explain precisely why Acc-MD avoids that limitation and whether this is broadly applicable or only in crafted examples.

### Questions
- Please produce a concise table that explicitly states which prior accelerated mirror-descent results are subsumed or improved by your analysis, and which aspects are genuinely new. Cite the matching equations/theorems. This will enable the readers to place the contributions in perspective. 
- Provide practical diagnostics and algorithms to estimate $C_{f,\phi}$ for typical problems (e.g., entropic objectives, LASSO). How sensitive is convergence to over- or under-estimating this constant? Can you supply experiments where $C_{f,\phi}$ is mis-specified? 
- The paper highlights the limitations of relying on accurate $\mu$ and $C_{f,\phi}$. Can you propose and test an adaptive rule (line search, local estimates) that does not require these constants yet preserves the accelerated behavior, or explain why this is difficult?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an accelerated mirror descent (Acc-MD) method and, under relative smoothness and relative strong convexity, proves an accelerated linear rate (Theorem 3.4). The proposed Acc-MD is also able to achieve acceleration even in cases where TSE fails.

### Strengths
The paper is well written, and the contributions are clear. I enjoy reading the section on "Discussion on assumption (A2)," which explains when the main assumption is satisfied and why it is valid (however, as explained below, this section could simply be examples of A2 - just a paragraph with bullet points - rather than theorems with lengthy presentations).

The numerical experiments verify the theoretical claims.

### Weaknesses
I find the presentation in several parts of the paper a bit confusing, introducing update rules and notations without properly presenting them. For example, the mirror descent update in the introduction of the paper is unnecessary and potentially confusing, as this is not a clear update rule (and does not serve a purpose to be there). The proper update rule was later given in section 2 where it was explained how the value of $x_{k+1}$ is actually updated using previous information.

The paper has essentially one main theoretical result: Theorem 3.4. The rest of the theory part of the paper focuses on simply providing examples of why the A2 is satisfied. However, this could have been just a proposition with the details in the appendix. The way the theory section of the paper is presented is written in such a way that it could be confusing, devoting almost 2 pages to discussion for A2, which are not really theorems in terms of convergence guarantees, but examples that satisfy the assumption (which of course require proof but this, to my understanding, is not really a highlight of the paper - could be part of the appendix without affecting the outcome).

There are papers that already provide accelerated mirror descent for the relative smoothness and relative strong convexity. These depend on the triangle scaling exponent (TSE), which is not the case for this method. However, I am not sure how this can be of general interest for the optimization or ML community. How can the results be used in other scenarios or applications?

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose Accelerated Mirror Descent (Acc-MD), a first-order optimization algorithm based on semi-implicit discretization of mirror descent flows.

The method uses a variable-splitting formulation and a Cauchy-Schwarz-type assumption to handle relative smoothness and strong convexity.

They evaluate their approach on entropic and quartic optimization problems, comparing convergence rates against ABPG, NAMD, and classical mirror descent.

### Strengths
The paper is clearly written and tries to address an important gap in accelerated mirror descent optimization 

The proposed Accelerated Mirror Descent (Acc-MD) is nice and simple and achieves linear convergence without relying on triangle-scaling assumptions as most prior methods do.

Some experiments on entropic and quartic objectives show that Acc-MD achieves good results compared to existing methods

### Weaknesses
It is not clear how sensitive Acc-MD is to inaccurate estimates of the relative smoothness and strong convexity constants (which is a very likely case in a real-life problem) as they control the step sizes and can make things break. Ablation studies is needed here

The experiments are limited to small synthetic convex problems (entropic and quartic objectives), with no large-scale or real-world benchmarks to show practical scalability or wall-clock gains. 

The authors need to test on datasets like ImageNet,  with models like ResNet and LLMs like BERT on GLUE Benchmarks. The authors need to show the practical use of this method on real life applications  and large scale benchmarks since this is a machine learning conference. For example, Adam,while the theory was on convex problems, it was heavily empirically tests on non-convex problems.

There is no clear ablation or sensitivity analysis on discretization choices, mirror functions, or inexact proximal updates, which makes it hard to tell how robust the method is in practice.

Without large-scale experimental results, my score will be a reject.

### Questions
How sensitive is the proposed Acc-MD method to inaccurate estimates of mu and strong convexity constants?​ and could an adaptive or line-search variant maintain the same accelerated guarantees?

Can the authors provide results on larger or real-world datasets (like logistic regression or constrained ML problems) to show the method’s scalability and robustness beyond synthetic settings?

How does Acc-MD behave when the Cauchy–Schwarz–type assumption (A2) is violated or only approximately holds? does the convergence degrade gracefully, or does it break down?

### Soundness
3

### Presentation
3

### Contribution
3
