# Graph Logic Flows: Geometry-Driven, Certificate-Carrying Reasoning on Dynamic Graphs

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
We introduce Graph Logic Flows (GLF), a framework that replaces stacked message passing with a single implicit update: one Jordan Kinderlehrer Otto (JKO) step of a reflected Wasserstein flow on a learned transport geometry. Task and domain rules such as shortest path consistency, triangle inequalities, conservation, or temporal smoothness are compiled into convex barriers where applicable and smooth surrogate constraints, and a lightweight runtime judge enforces them with different activations at train and test time. Each prediction returns numerical certificates including energy descent, KKT residuals, and logic residuals, yielding certificate carrying outputs. Under standard convexity assumptions, we establish theory for EVI contraction ensuring stability, finite step barrier invariance with strict feasibility in the small step limit for convex barriers, tracking under metric drift with ODE grade rates, and a sensitivity bound against oversquashing governed by the learned geometry rather than network depth. GLF unifies nonlocal reasoning, logic enforcement, and label free test time adaptation within a single convex integration step. Empirical case studies on dynamic graph benchmarks demonstrate the framework in practice, highlighting audit trails and constraint monitoring even under challenging predictive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces Graph Logic Flows (GLF), a framework that replaces stacked message passing on dynamic graphs. This approach enforces logical constraints and has strengths such as nonlocal reasoning, label free test time adaptation, and alleviates oversquashing.

### Strengths
The paper is theoretically strong. The idea of combining geometry theory in graph machine learning is novel and valid. The framework unifies nonlocal reasoning, logic enforcement, and label free test time adaptation within a single convex integration step.

### Weaknesses
The writing is hard to follow. I suggest making the preliminary part easier to understand, and put mathematical heavy things to the appendix.

### Questions
How can I interpret the empirical results combing with the theoretical guarantees? For example, how can I see the alleviation on oversquashing from the empirical results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Graph Logic Flows (GLF), replacing stacked message-passing with a single implicit update on learned transportation geometry. Domain rules are compiled into convex barrier functions activated by a lightweight adjudicator. EVI contraction theory establishes stability and barrier invariance guarantees, with each prediction returning verifiable certificates. Demonstrates label-free test-time adaptation on dynamic graphs.

### Strengths
1. Theoretical rigor: Complete proofs for EVI contraction, barrier invariance, and geometry-controlled sensitivity bounds provide solid theoretical foundation

2. Verifiable outputs: Certificate-carrying predictions (energy descent, KKT residuals) enable post-hoc verification for safety-critical applications

3. Novel perspective: Recasting graph reasoning as transport flows with JKO schemes addresses over-squashing from a fresh angle

4. Test-time adaptation: Achieves zero-shot adaptation on dynamic graphs with logic violations dropping to near-zero

### Weaknesses
1. Insufficient experiments: Only two dynamic graph benchmarks; significantly underperforms popularity baseline on LastFM, undermining practical claims

2. Computational cost unknown: No runtime/memory comparisons; implicit solvers and barrier projections likely incur high overhead

3. Scalability concerns: No experiments beyond 10K nodes; feasibility of single implicit update on large-scale graphs unverified

4. Missing ablations: No analysis of barrier types, adjudicator design, or connection regularization; difficult to assess critical components

### Questions
1. Computational efficiency: What are wall-clock time and memory compared to standard GNNs with equivalent expressiveness? How does the implicit solver scale?

2. LastFM failure analysis: Why underperform baselines on LastFM? Is it heavy tails, fast drift, or connection geometry learning limitations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes GLF, a framework that replaces stacked message passing with one implicit JKO step (Jordan–Kinderlehrer–Otto) per snapshot to integrate a reflected Wasserstein flow on a learned transport geometry. “Logic” is compiled as convex barriers and enforced by a runtime judge. The theory claims EVI contraction, finite-step barrier invariance, tracking under metric drift, and geometry-controlled sensitivity. Empirically, on real-world data GLF beats simple popularity baselines.

### Strengths
- Clear, unified formulation: one implicit JKO step replaces many GNN layers, making the update rule easy to reason about.
- Built-in constraint handling: domain rules are encoded as convex barriers, so predictions can respect logical/physical constraints.
- Certificates out of the box: each prediction comes with energy/KKT/violation metrics for auditability.

### Weaknesses
- Benchmark breadth and strength. Evaluation is limited to two JODIE datasets with very simple non-learned baselines. There are no comparisons to strong dynamic GNN baselines (e.g., TGAT [1], and implicit models for dynamic graph [2], each snapshot has one GNN update). This makes it hard to gauge practical competitiveness.
- Compute cost & practicality. Each snapshot solves a convex program, which is heavier than constant-time heuristics and typical message passing.
- Anti-oversquashing claim strength. The sensitivity bound hinges on assumption 2 (valid learned geometry). As presented, it reads existential/conditional rather than a uniform guarantee across datasets.
- The paper’s central claim is that compiling task rules as convex barriers and enforcing them via the runtime judge improves reliability. However, the manuscript does not isolate or quantify the benefit of these barriers. Therefore, no direct evidence showing enforcement improves accuracy, calibration, robustness, or stability.


[1] Xu, Da, et al. "Inductive representation learning on temporal graphs." arXiv preprint arXiv:2002.07962 (2020).
[2] Zhong, Yongjian, et al. "Efficient and effective implicit dynamic graph neural network." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

### Questions
- Can you provide details on how does the model train, what is the loss used in your tasks, how to optimize the geometry and potential?

### Soundness
4

### Presentation
2

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
The authors propose Graph Logic Flows to replace stacked message passing with an implicit Jordan-Kinderlehrer-Otto step of a reflected Wasserstein flow on a learned transport geometry. The work compiles algorithmic constraints as convex barriers and enforces them with a runtime judge whose activations differ between train and test time with a certificate. The method is essentially learning an edge-wise positive connection between the nodes, to induce a graph wasserstein metric regularized by a discrete penalty. Experiments are provided on two user-item log style datasets from the JODIE suite.

### Strengths
1. Formulating the prediction as a single reflect JKO step over a learned optimal transport geometry is interesting. The method proposed by the authors combines ideas from propagation, logic enforcement and test time adaptation in one step. 

2. The authors provide formal theoretical results on EVI contraction, finite step barrier invariance, tracking drift and geometry sensitivity, which is principled and would be helpful for follow-up works in this space. 


3. Auditable graph reasoning does not seem to have been very well explored in prior works, so the overall contribution is relevant to the community.

### Weaknesses
1. There are issues with the empirical setup. The authors compare their method to only two baselines, popularity and recency-popularity which are too weak for comparisont. They do not use any neural dynamic graph or session based recommenders such as TGN, TGAT, SASRec, etc. In my opinion, this significantly weakens their claim of state of the art robustness and practical use. 


2. The runtime logic with convex barriers which is mentioned as the primary contribution by the authors is not correctly evaluated. The authors do not mention which barriers were activated on JODIE and what fraction of predictions violated or obeyed the rules. The authors mention that the barriers are off during training and only lightly used as evaluation (L805-806) which is an issue. 


3. Some results are unclear. The authors mention that the certificate with energy drop should be non-negative, however in Table 5 and 6 there are negative values as well. So it is either a mismatch or an energy increase as opposed to a decrease which contradicts their drop hypothesis.

### Questions
See Weaknesses section above

### Soundness
2

### Presentation
2

### Contribution
2
