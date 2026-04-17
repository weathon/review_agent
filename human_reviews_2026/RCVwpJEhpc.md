# Constrained Generative Optimization via Progressive Flow Adaptation

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Adapting generative foundation models, in particular diffusion and flow models, to optimize given reward functions (e.g., binding affinity) while satisfying constraints (e.g., molecular synthesizability) is fundamental for their adoption in real-world scientific discovery applications such as molecular design or protein engineering.
While recent works have introduced scalable methods for reward-guided fine-tuning of such models via reinforcement learning and control schemes, it remains an open problem how to algorithmically trade-off reward maximization and constraint satisfaction in a reliable and predictable manner.
Motivated by this challenge, we first present a rigorous framework for Constrained Generative Optimization, which brings an optimization viewpoint to the introduced adaptation problem and retrieves the relevant task of constrained generation as a sub-case. Then, we introduce Constrained Flow Optimization (CFO), an algorithm that automatically and provably balances reward maximization and constraint satisfaction by reducing the original problem to progressive fine-tuning via established, scalable methods.
We provide convergence guarantees for constrained generative optimization and constrained generation via CFO.
Ultimately, we present an experimental evaluation of CFO on both synthetic, yet illustrative, settings, and a molecular design task optimizing quantum-mechanical properties.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Constrained Flow Optimization (CFO) as a method for fine-tuning pretrained generative models subject to explicit expectation constraints. CFO formulates the problem as a constrained optimization and applies an augmented Lagrangian approach to solve the posed problem. CFO is evaluated on both synthetic and real molecule design tasks.

### Strengths
- The ablation of the FineTuningSolver algorithm in line 414 is helpful in being able to understand how the performance of CFO is affected by the FineTuningSolver.
- The manuscript is generally well-written and the proposed CFO algorithm is intuitive and easy to understand.

### Weaknesses
1. Assumption 5.1 is a relatively strong assumption that requires the FineTuningSolver to return near-globally-optimal policies, and Theorem 5.4 further assumes $\varepsilon_k\to 0$ as $k\to\infty$ (i.e., arbitrarily accurate inner solves). These are unrealistic assumptions that are almost certainly violated in real-world settings with practical deep networks and standard optimizers - even if this is noted to be a previously held assumption in prior work (line 271). While I appreciate that this limitation is acknowledged in line 296, this limitation essentially invalidates the relevance of much of the theoretical contributions and convergence guarantees to the empirical setup. The manuscript should either provide empirical evidence that these assumptions are valid (e.g., in the synthetic tasks assuming the global optima are known) or consider proving a version of Theorem 5.4 without making use of especially strong assumptions.
2. There are a number important ablation studies that are missing. More specifically, it is unclear how sensitive the proposed CFO method is with repect to (1) the initial penalty parameter $\rho_{\text{init}}$; (2) the penalty growth rate parameter $\eta$; (3) the contraction parameter $\tau$; and (4) the minimum Lagrant multiplier $\lambda_{\text{min}}$. These ablations are particularly important since Algorithm 1 seems to be heuristic-based.
3. I think it is also important to independently (1) ablate Step 4 in Algorithm 1 and set the Lagrange multiplier to a constant value independent of $k$ (and showing how CFO performance changes as a function of this constant value); and (2) ablate Step 5 in Algorithm 1 and set the penalty $\rho$ to a constant value independent of $k$ (also modulate the exact value of this constant in the ablation). I note that the first ablation in this comment is essentially the baseline approach mentioned by the authors in line 54 to include constraint(s) as manually weighted objective function(s); the goal would be to empirically demonstrate that CFO is indeed an improvement over this baseline.
4. REINFORCE and rejection sampling also seem to be relevant baselines that should be included to evaluate empirically against.
5. In Figure 4(b), it looks like the AM baseline has not converged yet - it's unclear if the AM method will eventually satisfy the constraint if given enough steps.
6. The optimization is formulated as an expectation constraint in Equation 5. That is a much weaker requirement than per-sample feasibility or even a PAC guarantee, and the specific application for molecule design would seem like it's especially important to achieve per-sample guarantees (e.g., toxicity, chemical validity).
7. I'm not sure why the manuscript specifically focuses on only synthetic and molecular design as the application areas of CFO. In principle, CFO seems to be broadly applicable to any generative optimization problem - and indeed, this is how it sounds from the title of this work. Especially since there are a number of hyperparameters involved with CFO, it would be important to show that CFO performance is robust across different tasks with the same hyperparameter configurations.

### Questions
8. In line 219, why is the initial Lagrange multiplier value set to 0 and not $\lambda_{\text{min}}$?
9. What if there are multiple constaints in Equation (5)? Importantly, what happens if the constraint(s) do not admit a feasible set?
10. It seems like [this paper](https://proceedings.mlr.press/v267/yao25b.html) considers an almost identical problem formulation (albeit different empirical evaluation setup) as in Equation (5) and manages to arrive at an exact solution for the problem that does not rely on heuristics as Algorithm 1 does. The only different seems to be the choice of constraint function and the definition of the pre-trained policy. How does the proposed approach vary from/improve upon this prior work?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents Constrained Flow Optimization (CFO), a framework for optimizing pre-trained generative flow models that simultaneously maximizes rewards while satisfying explicit constraints. Using an Augmented Lagrangian approach, CFO automatically handles the balance between reward optimization and constraint adherence without requiring manual parameter tuning. The authors establish theoretical convergence properties and validate the approach through synthetic experiments and molecular design applications, showing reliable constraint enforcement compared to baseline methods.

### Strengths
The authors clearly identify a gap in current generative fine-tuning approaches—namely, the lack of principled handling of hard constraints in optimization problems. Introducing the augmented Lagrangian method into this constrained optimization problem is well motivated and sound. And the paper is well organized, with detailed pseudocode, clear notations, and reproducibility notes. Appendices provide implementation details, dataset descriptions, and solver pseudocode.

### Weaknesses
My primary concern is that the technological contribution appears incremental. The proposed algorithm largely applies the classical augmented Lagrangian framework to fine-tune a flow-based model. The main difficulty lies in practical tuning rather than in introducing a fundamentally new theoretical concept. Although the authors provide a theoretical analysis, it does not substantially advance the state of the art from a methodological standpoint.

In addition, the selection of baselines is rather limited. The comparisons are primarily made against a pretrained model without fine-tuning and the AM method. This choice seems designed to highlight the method’s novelty rather than to demonstrate its practical utility. Including additional baselines—such as methods employing soft constraints, test-time optimization strategies, or other comparable approaches—would strengthen the empirical evaluation and better substantiate the practical value of the proposed technique.

### Questions
See weakness.

### Soundness
4

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
3

### Summary
This paper addresses the critical problem of Constrained Generative Optimization (CGO), where a generative model (specifically, a flow model) must be adapted to maximize an objective function while satisfying another hard constraint. Common fine-tuning based methods usually fail to obey hard constraints. The proposed CFO transforms the constrained problem into a sequence of unconstrained, KL-regularized fine-tuning subproblems. By progressively and adaptively tuning the Lagrange multiplier and penalty parameter, CFO automatically and provably balances the trade-off between reward maximization and constraint satisfaction. Experimental validations are provided.

### Strengths
1. The introduction of Constrained Flow Optimization (CFO), which leverages the powerful and robust Augmented Lagrangian scheme to solve the CGO problem, is novel and sound.
2. The paper provides formal convergence guarantees, which is a major advantage over heuristic methods.
3. The low-dimensional (2D) experiment part has good organization and is also illustrative (i.e., Fig 2 and 3). The authors have made the important message easy to follow, thanks to its clear presentation.

### Weaknesses
1. While Figure 4 presents results for the crucial molecular design task, the comparison is not abundant. A full evaluation should include results comparing the CFO against more SOTA baselines as well as common heuristic baselines. While I understand widely acknowledged baselines in the area of the CGO problem might be limited, one must-have heuristic baseline is the standard fixed-weight Lagrangian, which can also serve as an ablation study to verify that the progressive tuning effectively improves over a fixed-weight counterpart.
2. While formal theories about the adaptive Lagrangian optimization are provided, I found the writing has much space for improvement. For example, the entire Section 5 proceeds too quickly without sufficient clarification. Eq. 14, 15, and Theorem 5.4 are especially hard to parse with non-trivial notations. It is also not clear what the implications are for most of the content in the section, except the final outcome: CFO enjoys global optimality under certain conditions.

### Questions
1. How was the number of MC iterations for the constraint and reward estimations determined in the high-dimensional setting? Is it possible to provide a theoretical and/or logical analysis showing how increasing the number of MC iterations might reduce the small violations observed in the results?
2. Considering deviations, it seems Figure 4c does not really prove CFO's advantage over AM? Note AM does not need progressive optimizaiton so it consumes significantly less computation overheads. About this part it is also important to add computation analysis between CFO, PRE and AM because non-significant performance improvement becomes less promising at the price of costs.
3. It is highly recommended to improve theory section's writing to improve readability. Even not going too deep into details, it's important to have a quick take-away overview of what the notations are meaning and how they are connected to each other to finally prove global optimality,

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this work, the authors introduce Constrained Flow Optimization (CFO) which is a principled framework for fine-tuning generative flow or diffusion models under explicit constraints. The central problem addressed is how to adapt pre-trained generative foundation models (e.g., FlowMol, diffusion backbones) to maximize task-specific rewards (e.g., molecular property improvement) while ensuring constraint satisfaction (e.g., synthesizability or energy bounds).

The authors first formalize this setting as Constrained Generative Optimization (CGO), bridging ideas from optimization and generative modeling. They then derive CFO as a dual optimization algorithm based on the augmented Lagrangian method, which transforms the constrained objective into a series of standard KL-regularized fine-tuning problems. This approach allows progressive adaptation of generative models without manual tuning of reward–constraint trade-offs and provides theoretical guarantees for convergence and constraint satisfaction.

Empirically, CFO is evaluated on synthetic 2D settings and high-dimensional molecular design tasks, demonstrating that it can increase rewards (e.g., dipole moment) while enforcing stability constraints (e.g., xTB energy bounds). The authors further show CFO’s compatibility with differentiable and non-differentiable constraint functions, highlighting its generality for scientific discovery domains such as drug and material design.

### Strengths
The authors effectively demonstrate that Constrained Flow Optimization (CFO) scales to realistic, high-dimensional molecular design problems. Applying the method to a nontrivial chemical property optimization task of maximizing dipole moment under energetic stability constraints provides a credible and practically relevant benchmark.

### Weaknesses
With regards to the molecular design task, while CFO demonstrates effective scalar property optimization, the generated molecules show reduced chemical validity and unclear structural diversity, both of which are critical in molecular design and drug discovery. This raises concerns that the observed gains may stem from model bias exploitation or surrogate inaccuracies, rather than genuinely improving molecular quality. Although the authors briefly acknowledge this limitation, a more systematic analysis such as evaluating scaffold diversity, Lipinski or QED distributions, and functional group statistics would strengthen claims of chemical realism and practical relevance.

Furthermore, the paper would benefit from comparisons with diversity-preserving fine-tuning methods, such as Relative Trajectory Balance (RTB) (https://arxiv.org/abs/2503.06337, https://arxiv.org/pdf/2503.06337). RTB explicitly mitigates mode collapse and maintains distributional breadth during optimization, which is especially relevant for molecular generation tasks where balancing property improvement and diversity is crucial. Including such baselines would clarify whether CFO’s improvements come at the cost of diversity and robustness.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
