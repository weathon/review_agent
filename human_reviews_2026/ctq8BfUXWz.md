# Branched Schrödinger Bridge Matching

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Predicting the intermediate trajectories between an initial and target distribution is a central problem in generative modeling. Existing approaches, such as flow matching and Schrödinger bridge matching, effectively learn mappings between two distributions by modeling a single stochastic path. However, these methods are inherently limited to unimodal transitions and cannot capture *branched* or *divergent* evolution from a common origin to multiple distinct modes. To address this, we introduce **Branched Schrödinger Bridge Matching (BranchSBM)**, a novel framework that learns branched Schrödinger bridges. BranchSBM parameterizes multiple time-dependent velocity fields and growth processes, enabling the representation of population-level divergence into multiple terminal distributions. We show that BranchSBM is not only more expressive but also essential for tasks involving multi-path surface navigation, modeling cell fate bifurcations from homogeneous progenitor states, and simulating diverging cellular responses to perturbations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Branched Schrödinger Bridge Matching (BranchSBM), which solves a branched generalized SB by decomposing it into a sum of unbalanced conditional SOC objectives. It learns branch-specific drifts and growth rates with mass-conservation constraints via a multi-stage procedure (interpolant → flow matching → growth → joint). On LiDAR navigation, cell differentiation, and drug-perturbation gene expression, BranchSBM improves endpoint/intermediate reconstruction over single-branch SBM.

### Strengths
1.  Frames branched GSB as a sum of Unbalanced CondSOC tasks; gives branch-wise drifts and growth with mass conservation. 
2.  Consistent W1/W2/MMD improvements vs. single-branch SBM; scales to ≥3 branches in perturbation tasks.

### Weaknesses
1. Requires branch priors/endpoint clustering. Training assumes access to clustered terminal distributions $\pi_{1,k}$ (and uses them directly in losses), rather than discovering branches from snapshots.
2. The work decomposes branchGSB into a sum of unbalanced CondSOC subproblems with branch-specific drifts/growth and a task-specific state cost guiding intermediate trajectories. I think several components are not entirely new. In per branch, the formulation considered here is close to unbalanced SB/RUOT except for the task-specific state cost. The neural interpolant function considered here and used for flow matching is also presented in (Neklyudov et al.,  ICML 2024). And I believe the paper misses some references. The paper discusses the connections with other methods mainly in the appendix, but it would benefit from a more explicit discussion of some key references in the main text.
3. Not totally simulation-free in training. Although drift learning avoids state-SDE simulation (Stages 1–2), growth learning still requires time integration of $g_{t,k}$ to obtain the weight.
4. Training complexity and the contribution of each stage remain unclear. The paper reports Stage-3 vs. Stage-4 loss trends (Table 6) but does not present ablation on metrics (e.g., W/MMD) to quantify each stage’s effect on final performance, stability, or sample quality. More granular ablations could clarify the necessity/sufficiency of each stage.
5. Baselines are limited. Comparisons are primarily to a single-branch SBM baseline; no comparisons to other unbalanced SB/RUOT (e.g., DeepRUOT), or flow-matching methods (e.g., OT-CFM, SF2M, MetricFM) are provided, which weakens the case that BranchSBM is uniquely required.
6. Pipeline choices may inject bias. Sensitivity to the number of branches K, and to endpoint clustering protocols, is not analyzed. 
7. Dependence on a task-specific state costs $V_t$. The approach follows intermediate trajectories “governed by a state cost”; results highlight energy/mass evolution but do not systematically study how $V_t$ choices affect accuracy, stability, or branching time.  
8. Perturbation experiments are conducted on top-50 PCs but metrics on top-2 PCs for W distances; it is unclear why the authors choose to do so.
9. Scalability with many branches/memory. The authors note higher space complexity (multiple branch networks), though time is said to be comparable, and inference is cheaper. Empirical scaling with large K or long horizons is not demonstrated. Also lacks analysis of training time or memory scaling with dimensionality or number of branches.

### Questions
1. Could the method be extended to infer branching structure automatically from snapshot data? If not, please discuss how sensitive training is to incorrect or coarse endpoint clustering.
2. The four-stage pipeline (interpolant → flow matching → growth → joint) is complex, and the contribution of each stage to the final outcome remains unclear. Could the authors report ablations on key metrics (e.g., W₁/W₂/MMD) to quantify how each stage affects trajectory fidelity and stability?
3. The evaluation mainly contrasts with a single-branch SBM. Including other competitive baselines (e.g., DeepRUOT, SF2M, OT-CFM, MetricFM) would better contextualize where BranchSBM provides concrete advantages.
4. Since $V_t$ guides intermediate trajectories, could the authors explore how varying its definition affects results? Currently, its influence on branching time or trajectory stability is not quantified.
5. The perturbation experiments rely on PCA (50, 100, 150 PCs) and evaluate W distances only on the top-2 PCs. Could the authors test whether performance holds in the full PCA space to validate?
6. The paper notes increased space complexity due to per-branch networks, but no measurements of training time or memory usage with respect to dimensionality or number of branches are reported. Including empirical scaling results (e.g., runtime vs. K or dimension) would clarify practical feasibility.
7. Since BranchSBM explicitly learns branch-specific growth rates $g_{t,k}$, it would be informative to visualize these trajectories on biological datasets—e.g., the mouse hematopoiesis dataset. Showing how growth evolves along different branches could clarify the biological interpretability and verify whether the model captures expected proliferation or decay patterns.

References
1. Neklyudov, Kirill, et al. "A computational framework for solving wasserstein lagrangian flows." ICML 2024.
2. Zhang, Zhenyi et al. "Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport." ICLR.
3. Tong, Alexander, et al. "Improving and generalizing flow-based generative models with minibatch optimal transport." TMLR.
4. Tong, Alexander, et al. "Simulation-free Schrödinger bridges via score and flow matching." AISTATS.
5. Kapusniak, Kacper, et al. "Metric flow matching for smooth interpolations on the data manifold. NeurIPS 2024.

I would be happy to hear the authors’ thoughts and clarifications in the rebuttal and may adjust my score accordingly.

The reviewer wrote this review. An LLM was used only for language refinement.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents Branched Schrödinger Bridge Matching (BranchSBM) for learning in a new problem setting: branched Schrödinger bridges. This framework is more general than the standard approach and has the potential to be applied to certain tasks involving multiple paths. Theoretically, the problem originates from branched conditional stochastic optimal control, where there can be $K$ joint couplings ($K+1$ multi-directional branches) to represent a holistic solution from the initial condition. The authors present a bridge matching algorithm and show its efficacy on cell dynamics data.

### Strengths
Based on my reading, the method appears to have the following strengths.
* The paper is relatively straightforward to understand, even though the problem itself and the surrounding materials are complex.
* The proposed method seems to be sound (I did not check the entire appendix).
* The authors performed experiments on biological data and showed practical aspects of BranchSBM .

### Weaknesses
I believe this work, in its current form, lacks clarity in many regards.

* It seems that the branched Schrödinger bridge problem is a subset of the well-established multi-marginal Schrödinger bridge (optimal transport) problem. The claim seems to be that BranchSBM clearly specifies the notion of a source and branches, thus enabling more efficient algorithms, such as interpolant optimization. However, the manuscript is written in a way that does not clearly reveal this technical motivation and high level implications. Therefore, I think the corresponding contribution is somewhat vaguely written.
* The overall experiments are limited, and it is not clear why single-cell data provides sufficient verification for the proposed framework. The authors are encouraged to put more effort into experiments on other high-dimensional data.
* There is room for theoretical proof or analysis to demonstrate why the proposed scheme is fundamentally better or more efficient than single-branch SBM.

### Questions
* Could you summarize how BranchSBM is fundamentally different from single-branch SBM with multi-modal marginals?
* What is the general training time for single-branch SBM vs. BranchSBM for challenging problems?

### Soundness
3

### Presentation
2

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
This paper introduces Branched Schrödinger Bridge Matching (BranchSBM), a generative framework designed to model transport from a single initial distribution to multiple distinct terminal distributions. Current models, like standard Schrödinger Bridge Matching (SBM) and flow matching, are limited to modeling single, continuous paths between a source and target. The goal of BranchSBM is to learn branched trajectories to guide from a single source to multiple target by parameterizing separate dynamics for each path. The method also extends the Generalized SBM (GSB) framework by including a non-linear state cost, $V_t(X_t)$, which acts as a potential term to ensure that the learned trajectories remain on the data manifold.

The proposed BranchSBM algorithm solves the "Branched GSB problem" by parameterizing separate velocity fields (drifts) $u_{t,k}$ and growth rate fields $g_{t,k}$ for each of the $K$ branches with neural networks. This formulation allows the model to handle "unbalanced" transport, where the mass of each branch can grow or shrink. The model is trained using a four-stage algorithm. Stage I, trains a neural interpolant to find the optimal, energy-minimizing paths between the source and each target endpoint. Stage II then uses conditional flow matching to train the branch-specific drift networks $u_{t,k}$ to replicate the velocities from Stage 1. Stage III freezes the drift networks and trains the growth networks $g_{t,k}$ to match the known target mass of each branch. Stage IV unfreezes all parameters and jointly fine-tunes all networks to minimize a combined objective that includes energy, mass matching, and distribution reconstruction losses.

### Strengths
The paper tackles an interesting and well-motivated problem in machine learning of how can we model population-level dynamics that diverge from a single source to multiple distinct outcomes. This is common scenario in fields like single-cell genomics, where we are interested in how a single, source population of pluripotent cells evolved into several distinct cell types, and standard generative models that assume a single, unimodal transition are insufficient. The authors' core idea of extending Schrödinger bridges to explicitly handle branched paths is a relevant contribution.

The theoretical framework for decomposing this complex problem also appears sound. The authors provide proofs for their core propositions, notably Proposition 1, which tractably reframes the "Unbalanced GSB" problem into a solvable "Unbalanced CondSOC" objective, and Proposition 2, which then justifies modeling the full "Branched GSB" problem as a sum of these individual objectives . This provides a formal grounding for the method's architecture, which separates the dynamics for each branch.

### Weaknesses
The proposed method's significant complexity raises concerns about its practical adoption, scalability, and generalizability. The algorithm is a four-stage sequential training pipeline (Algorithm 1) that involves training at least $2(K+1) + 1$ separate neural networks (an interpolant, $K+1$ drift fields, and $K+1$ growth fields). Furthermore, the overall objective is a carefully weighted sum of at least 6 different loss terms (trajectory, flow, energy, match, mass and reconstruction), which are balanced by many hyperparameters. This multi-stage, multi-loss design suggests a high sensitivity to tuning and raises questions about its robustness. Given this intricacy, the paper would be substantially strengthened by meticulous ablation studies to justify each component. For instance, a thorough analysis of the model's performance when stages (such as 1 or 4) are simplified or loss terms are removed would help distinguish critical components from those offering marginal gains and would better motivate the final design.

This substantial methodological complexity is not yet fully justified by the provided experimental validation. The cost of the method would be more compelling if benchmarked against a wider array of competitive alternatives and shown to have outstanding results. Currently, most of the results are compared almost exclusively against a single baseline (single-branch SBM). While this demonstrates that a branched model is better at modeling branches than a non-branched one, it doesn't contextualize the method within the broader field. Furthermore, Despite being motivated by cell differentiation, the paper notably omits a comparison against any of the many existing, often simpler, single-cell trajectory inference algorithms, some of which also explicitly designed to deal with branching. This makes it difficult to assess the practical utility of BranchSBM over established methods.

### Questions
The method's parameters and computation scale linearly with the number of branches, $K$. You've shown this for $K=2$ and $K=3$. Have you tested the method's stability and performance for a larger $K$, such as $K=10$ or $K=20$, which would be more representative of complex biological differentiation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Branched Schrödinger Bridge Matching (BranchSBM) to overcome the limitations of existing approaches, such as flow matching and Schrödinger Bridge Matching, which can only model a single unimodal path. BranchSBM learns multiple time-dependent velocity fields and growth processes, enabling it to capture branched or divergent evolution from a common origin to multiple distinct outcomes.

### Strengths
* This method is grounded in solid mathematical theory and is particularly useful for challenging tasks, such as multi-path surface navigation.
* The paper adopts an approach similar to OT-CFM, where optimal transport is used to define couplings that improve flow-matching vector fields. It formulates the **Branched Generalized Schrödinger Bridge Problem** and develops a theoretical foundation around it, enabling the computation of couplings for downstream CFM models to learn branched paths effectively. This is a common method, yet its deduction is non-trivial in this case.

### Weaknesses
There is room to improve the clarity of the technical sections by providing more detailed explanations. For example, could the authors clarify why it is necessary to consider the time-dependent weights and growth rates in Equation (5)? What is the motivation or scenario that calls for this addition, and how does it connect to the problem studied in the paper? Explaining why this is the appropriate formulation would help readers better understand both the method and the rationale behind it.

### Questions
Could this method be applied to tasks such as text-to-image generation or other common multimodality problems? If so, is there a reason why the paper does not explore these applications? 

Please also refer to the weaknesses section for additional context.

### Soundness
4

### Presentation
3

### Contribution
4
