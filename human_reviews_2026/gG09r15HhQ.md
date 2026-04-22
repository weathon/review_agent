# Elastic Optimal Transport: Theory, Application, and Empirical Evaluation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
The classical optimal transport such as Kantorovich's optimal transport and partial optimal transport could be too restrictive in applications due to the full-mass or fixed-mass preservation constraints. To remedy this limitation, we propose elastic optimal transport (ELOT) which is distinctive from the classical optimal transport in its ability of adaptive-mass preserving. It aims to answer the problem of how to transport the probability mass adaptively between probability distributions, which is a fundamental topic in various areas of artificial intelligence. The strength of elastic optimal transport is its capability to transport adaptive-mass in the light of the geometry structure of the problem itself. As an application example in machine learning, we apply elastic optimal transport to both unsupervised domain adaptation and partial domain adaptation tasks. It adaptively transports masses from source domain to target domain by taking domain shift into consideration and respecting the ubiquity of noises or outliers in the data, in order to improve the generalization performance. The experiment results on the benchmarks show that ELOT significantly outperforms the state-of-the-art methods. As a powerful distribution matching tool, elastic optimal transport might be of interests to the broad areas such as artificial intelligence, healthcare, physics, operations research, urban science, etc. The source code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers the mathematical aspects of optimal transport (OT), where a variant of existing OT formulation is proposed. Specifically, the authors focus on the strict equality or inequality constraints on transport mass, and try to propose a new mechanism to achieve a relaxation on the strict hard constraints, i.e., adaptive mass. Empirical validations are conducted on standard domain adaptation datasets compared with other OT counterparts.

### Strengths
S1. A new OT formulation with elastic constraints and theoretical analysis.

S2. The organization is clear and the writing is easy to follow.

S3. The experimental performance is superior to other standard OT variants.

### Weaknesses
W1. The rigor of the claim and theoretical analysis should be improved.

W2. The empirical validation could be improved, where the SOTA comparison methods should be considered.

### Questions
Q1. The elastic optimal transport (ELOT) is not rigorously defined and analyzed. Specifically, the notation $\mathbb{R}_{±}$ in Eq. (6) for the cost function is not defined. If it implies that the cost function could be both positive and negative, the ELOT could not be well-defined, e.g., does it still satisfy the metric property? A rigorous and complete theoretical analysis is necessary to ensure the validity of ELOT.

Q2. The ELOT seems to be an incremental work of partial OT. Specifically, the basic conclusions and technical details (i.e., Thm. 1 and its proofs) of ELOT are almost the same as partial OT [27, Prop. 1], while the proofs in line 54-56 manuscript seem to be problematic, i.e., there are no rigorous proofs to ensure that the submatrix of augmented OT plan is equivalent to the OT plan of ELOT.

Q3. Moreover, the formulation of ELOT in Eq. (6)-(7) still cannot ensure the adaptive learning of mass value $s$. Specifically, the technique based on dummy variables still induces the parameter $\sigma$ in Eq. (8), which is basically equivalent to adjusting $s$.

Q4. In the claim below Eq. (8), the authors set $\sigma$ as 0 in the experiment. However, in such a formulation, the ELOT problem seems to be exactly the original Kantorovich OT problem, as the augmented cost matrix is block-diagonal where the $n\times m$ block and $1\times 1$ block are independent. Based on this problem, it would be hard to understand the performance improvement of ELOT in experiments.

Q5. The comparison experiment with SOTA methods is not significant. This experiment is conducted on 3 DA datasets, where the performance is significantly lower than the SOTA methods in recent years. Some further verifications on the latest proposed datasets and advanced baselines are necessary, e.g., DomainNet dataset and ViT backbone.

Q6. The quality of OT plan learned by ELOT is not sufficiently validated. For example, if ELOT can adaptively adjust the mass of plan, a quantitative/qualitative comparison between the OT plans of different OT formulations should be considered.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Elastic Optimal Transport (ELOT), an optimal transport (OT) formulation designed to address practical limitations of classical OT, partial OT, and unbalanced OT. In standard OT, all mass from the source distribution must be transported to the target distribution. Partial OT transports only a fixed budget $s$ of mass, which the user must choose. Unbalanced OT relaxes the marginal constraints but requires tuning divergence penalties. ELOT instead allows the optimal plan to decide how much mass, depending only on the cost. ELOT explicitly allows the cost matrix $C$ to include negative entries, and thus adaptively determines the transported mass. The authors apply ELOT to domain adaptation and compare it with OT-based approaches.

### Strengths
- The motivation is strong. The authors aim to adaptively match datapoints without hand-tuning transport mass, which is really useful for practical applications if realized.
- The authors consider a signed cost to realize the adaptive amount of transported mass, which is reasonable.
- The computation is transformed into an OT formulation, and thus can be solved by existing tools.
- The paper is easy to read.

### Weaknesses
- Novelty relative to partial OT / unbalanced/robust OT needs to be strengthened.  Partial OT already allows transporting only part of the mass. In classical formulations, one can introduce a Lagrange multiplier that effectively shifts the cost matrix and induces an “automatic” choice of how much mass to transport. Prior work (the paper cites Caffarelli & McCann, 2010) can create a situation where negative effective cost encourages matching only for “good pairs,” and the transported mass adapts as a function of that shift. 
ELOT is described as more general because it directly allows mixed-sign costs and only imposes marginal $\le$ constraints, and it claims to “automatically finds the optimal mass to be transferred without setting a priori budget.” However, ELOT can still look like “partial OT with a specific cost shaping and slack embedding.” The paper should give a crisper mathematical argument for why ELOT is fundamentally different and not just a repackaging.

- Calling $W(\mu,\nu)$ a ‘distance’ may be misleading. Because $C$ can be non-symmetric and can contain negative values, the induced objective $W(\mu,\nu)$ is not guaranteed to be a metric or even a divergence: non-negativity, symmetry, and triangle inequality can fail.
The paper uses “optimal transport distance” language, which is standard OT terminology, but here it risks overstating the geometric meaning. This should be toned down or clarified. Meanwhile, the commonly used costs, e.g., L1 L2, are not suitable. How could ELOT be applied to partial transport under such widely utilized costs?

- The paper aims to avoid hand-tuning transport mass, a hyperparameter. However, the costs in the domain adaptation task and the partial domain adaptation task involve hyperparameters, to which the performance is sensitive, as shown in the experiments. So I feel like the authors avoid tuning one parameter but instead tune several other parameters. For DA, the parameters are set to the values in the compared methods. But for other applications, how these parameters are determined.

### Questions
- Could the authors explain how to define the cost for a general task, as the commonly utilized costs are often non-negative in most tasks?
- Can the private class data be detected and omitted by ELOT in partial DA?
- The compared methods are not soTA. Could the authors include more recent SOTA? For example, MOT[1], ARPM[2], both are OT-based.

[1] Mot: Masked optimal transport for partial domain adaptation

[2]Adversarial Reweighting with α-Power Maximization for Domain Adaptation

### Soundness
2

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
4

### Summary
The paper introduces Elastic Optimal Transport (ELOT), a novel formulation of optimal transport that relaxes the full-mass or fixed-mass constraints present in classical OT, partial OT, and unbalanced OT. ELOT allows for adaptive-mass preservation and supports mixed-sign cost matrices, making it more flexible for real-world applications where noise, outliers, or distribution shifts are present. The authors provide theoretical analysis, an equivalent reformulation solvable with standard OT solvers, and apply ELOT to unsupervised and partial domain adaptation tasks. Experimental results on standard benchmarks (VisDA, Office-31, Office-Home) demonstrate that ELOT outperforms several state-of-the-art OT and non-OT baselines.

### Strengths
1.	Novel Formulation: The idea of adaptive-mass transport is well-motivated and addresses a clear limitation of existing OT methods. The introduction of a mixed-sign cost matrix enhances its applicability to real-world problems.
2.	Strong Empirical Performance: The paper provides extensive and convincing experiments on multiple domain adaptation benchmarks, showing consistent and significant improvements over a range of strong baselines.
3.	Theoretical-Practical Bridge: The work offers valuable theoretical insights and a practical reformulation that enables the use of standard OT solvers, facilitating wider adoption.

### Weaknesses
Proofs of Theorems 1 and 2 Lack Rigor: 
The proofs of Theorems 1 and 2, while intuitively appealing, lack rigor in their current form. In particular, the argument that two transport plans are equivalent because they minimize the same cost ignores the issue of non-uniqueness of solutions in ELOT—unlike in classical OT where uniqueness often holds under mild conditions. The authors should revisit the proofs to account for potential multiple optima. The theorems could be rephrased to state that there exists an optimal plan for the reformulated problem that matches an optimal plan of the original ELOT, rather than implying equality of all such plans.

### Questions
1.  Potential Degeneracy When Transport Mass is Zero: 
The formulation of ELOT allows for the possibility that the total transported mass s=0, which occurs when all ground costs are positive. While this may be desirable in some outlier-rich scenarios, it could also lead to degenerate solutions in applications where some degree of alignment is necessary.
Suggestion: The authors should discuss this property and its implications for the general applicability of ELOT. A discussion on whether this poses a limitation in practice and if potential modifications or safeguards could mitigate this issue would strengthen the paper.
2. Ambiguity in Theoretical Relationship Between ELOT and Unbalanced Optimal Transport: 
In the current formulation, ELOT is presented as distinct from unbalanced OT. However, in unbalanced OT, the marginal divergence penalties act as soft constraints. when τ1,τ2→0 the penalties vanish, allowing relaxed inequalities γ1_m ≤μ,γ^T 〖1〗_n≤ν —which closely resembles the ELOT formulation. The paper currently does not analyze this limiting behavior, making it unclear whether ELOT can be regarded as a limiting case of unbalanced OT.
Suggestion:
To further strengthen the theoretical contribution, the authors may consider discussing the limiting behavior of unbalanced OT as the penalty coefficients approach zero, and clarifying how ELOT relates to this case. This would help position ELOT more clearly within the broader OT framework.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Elastic Optimal Transport (ELOT), a novel formulation using marginal inequalities and a mixed-sign ground cost. The authors present two key theoretical results: an equivalence to an equality-constrained problem whose plan is invariant to a parameter σ (Thm. 1), and a "mass transport mechanism" (Thm. 2) showing that mass flows only on negative-cost entries, which enables automatic outlier filtering. Empirically, the method demonstrates consistent improvements over domain adaptation baselines on VisDA, Office-31, and Office-Home.

### Strengths
In general the method is interesting and promising, avoiding limitations of the current OT solvers. Theoretically justified formulation with consistent gains in UDA and partial DA using a uniform backbone and setup.

### Weaknesses
While the formulation is clean and intuitive, my primary concern is the unclear formal relationship to Unbalanced Optimal Transport (UOT). The authors mentioned UOT in the background but did not consider these methods in more detail. Although UOT is a well-known approach to a similar problem to that stated in the paper, a more detailed analysis of its relationship to existing solvers is necessary. The  The authors assert a connection, but do not analyse it sufficiently. Experimental comparison to UOT is completely ignored. Why? To strengthen the contribution, please consider addressing the following questions:

### Questions
**Q1**: Under what specific assumptions on the cost C or marginals (μ,ν) do ELOT and UOT provably yield different couplings?

**Q2**: Does elastic solver can be considered in entropy-based regularized settings? Given that negative costs can cause instability in entropic solvers (e.g., Sinkhorn's exp(−C/ε)), what stabilization techniques are used? Does the plan invariance (Thm. 1) hold exactly under this entropic regularization?

**Q3**: The paper defines the resulted W and calls it an “optimal transport distance,” but fails to discuss its metric properties as identity, symmetry. Does the resulted solution value actually provide some sort of Wasserstein distance?  I the cost matrix C has no negative entries, the zero plan (γ=0) is feasible and optimal, yielding W(μ,ν)=0 by construction. This seems to violate the identity property of a distance and is consistent with Theorem 2, which states mass only flows on negative-cost entries.

minors:
Add runtime/memory measurements or complexity analysis to justify statements.

### Soundness
3

### Presentation
3

### Contribution
2
