# Optimal Dataset Design for Nurture-then-Nature Teaching

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Designing an optimal dataset to teach a target concept to a learner has been a well-studied problem in Machine Learning. Prior works have mostly focused on unconstrained single-phase teaching, where the learner learns solely under the guidance of a helpful teacher who can provide any number of examples. In this work, we introduce a more realistic two-phase framework called "Nurture-then-Nature" where the learner first learns under the guidance of a teacher in the 'Nurture' phase, followed by an i.i.d. learning phase from 'Nature'. Importantly, the teacher is constrained to provide a dataset of size up to $B$ and is required to minimize the final error of the learner. We study this problem in the 'instance-agnostic' and 'instance-aware' settings and provide efficient teaching algorithms for each of them. We provide theoretical guarantees and experimental results to support our findings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the machine teaching problem under a two phase framework, where the student first learns from a teacher (nurture), then without the teacher’s supervision on i.i.d. data (nature). The teacher is constrained to a limited dataset size with the goal os minimizing the final error of the student. The authors provide theoretical results for this framework and experimental setups to support these results.

### Strengths
The setup of the problem is interesting and similar to real world examples, where the teacher trains the student during the nurture phase e.g., pre-training to get a good initialization so that the student can effectively continue learning without the teacher's supervision after.

### Weaknesses
- Practicality of the Instance-Aware Setup. he instance-aware teaching framework assumes that the teacher has knowledge of all examples observed by the student during the nature phase which seems like a strong assumption.
- Clarity and Messaging in Figures. The figures are useful for conveying insights, but some are challenging to interpret. For example, in Fig. 2 and 5b, the key differences or takeaways are not immediately clear (the plots in Fig. 2 appear quite similar). Adding clearer annotations or captions could help readers more easily grasp the intended message. Fig 5b, shows that there are different points selected but it does not help with understanding why these particular points are optimal in each case.
- Further insights from the selected dataset. It could be interesting to visualize (e.g., with a toy vision dataset) the difference in data selected by the teacher in the nurture phase under the ntn setting vs a nature-only setting, under the same budget. Also how the selected examples change under more extreme budget constraints.

### Questions
- What are some scenarios where the teacher would be aware of what the student will see in the Nature phase?
- Given than the paper focuses on B < TD, do the selected points reflect something like importance or influence? How does the selected points change with lower values of B.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies “Nurture-then-Nature (NtN)” dataset design: a teacher first provides a small, curated labeled set (“nurture”), after which the learner continues on naturally sampled data (“nature”). The work analyzes two regimes: Instance-agnostic teacher: doesn’t know the future data distribution and aims to pick examples that shrink the remaining hypothesis space before the nature phase. and Instance-aware teacher: assumes a linear datamodel where final risk is exactly a linear function of example indicators

### Strengths
1. The paper is well written and motivated, and lays out the two-phase training and splits the nature stage into agnostic vs. aware cases.

### Weaknesses
1. The “optimality” proof for the agnostic teacher only optimizes an upper bound on risk (via VC dimension). The authors equate minimizing this bound with minimizing true post-nature error, which is not guaranteed.

2. The intro section could benefit from a comparison against curriculum learning, active learning, or modern data selection methods.

3. minor: the paper still shows the template title

### Questions
1. Could the authors discuss potential pathways for extending these ideas to larger-scale deep learning settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the Nurture then Nature (NtN) teaching framework: a teacher with limited budget B first provides a teaching dataset (Nurture) and then the learner receives i.i.d. data from the environment (Nature). The teacher’s objective is to minimize the learner’s final error after the Nature phase.

•	Two teacher knowledge regimes are studied: 
o	Instance agnostic: teacher does not know PX. The problem is reduced (via PAC guarantees) to minimizing the VC dimension (or other proxies) of the surviving version space; algorithms/constructive solutions or approximations are given for several hypothesis classes (finite binary classes → greedy 1−1/e approximation via budgeted max coverage; axis aligned rectangles → exact VC reductions per budget; homogeneous linear classifiers → kill orthogonal subspace to reduce VC from d to d−B+1; polynomial kernels → analogous feature space construction assuming preimages).
o	Instance aware: teacher knows PX. Using a linear datamodel (risk ≈ wP^T 1D) the expected final risk reduces to a weighted sum; the optimal B item teaching set is the B items with smallest weights wP,x (1−Px)^n (efficient selection).

Overall, the paper gives theoretical guarantees (optimality or approximation ratios), proof sketches, algorithms/pseudocode in the appendix, and synthetic experiments (linear classifiers, axis aligned rectangles, datamodel selection) showing improvements over no teach and a simulated random baseline.

### Strengths
•	Problem novelty and relevance: NtN formalizes a realistic two phase teaching scenario (limited guided teaching followed by natural i.i.d. learning).

•	Clear separation of settings: instance agnostic vs instance aware capture different practical knowledge regimes and motivate different techniques.

•	Theoretical contributions: nontrivial results for multiple hypothesis classes (exact VC reductions for linear class, approximation for finite classes, concrete budget→VC tables for rectangles).

•	Elegant reduction for instance aware case: datamodel linearization yields a simple, efficient selection rule with clear interpretation (weight scaled by (1−Px)^n).

•	Reproducibility support: proofs, pseudocode and experimental details (meta dataset construction, datamodel learning) are provided in the appendix.

•	Empirical validation: synthetic experiments verify expected behaviors (higher budget reduces final risk; proposed methods outperform naive baselines in the tested regimes).

### Weaknesses
•	Strong assumptions for instance aware solution: the linear datamodel assumption (risk exactly linear in dataset indicators) is strong; the paper lacks analysis of the effect of datamodel approximation/error on selection quality and final risk.

•	Limited scalability and realism of experiments: evaluations are on small, synthetic, low dimensional/discrete domains (e.g., 16 point circle, small grids). Methods that require enumerating version spaces or finding feature space preimages may not scale to high dimensional real data (images, large corpora).

•	Constructive/algorithmic gaps: for some claims (polynomial preimages, rectangle examples) the paper assumes existence or states constructions but provides limited practical algorithms or complexity analysis for finding these examples in constrained domains.

•	Strong learner assumptions: realizability and version space learner assumptions simplify analysis but reduce applicability under label noise, model misspecification, or when learners produce single hypotheses (not full VS).

•	Baselines: the simulated baseline (random simulated teaching sets) is weak; stronger heuristics (greedy VC reduction, information gain, uncertainty sampling) are not compared.

•	Computational cost of datamodel training: building the meta dataset requires training many base learners on many subsets; costs and required meta sample sizes are not discussed.

•	Lack of real world scenarios and reduced practical contribution: the paper provides only synthetic, small scale experiments and no demonstrations on real datasets or tasks. This reduces the perceived practical contribution and leaves unclear whether the methods (particularly datamodel learning and preimage constructions) work in realistic, high dimensional settings.

### Questions
Suggested reviewer questions for the authors
1.	Datamodel robustness: if the learned datamodel ŵ deviates from the true w (||ŵ − w|| large), can you bound how selection via ŵ affects the expected final risk? How accurate must the datamodel be in practice for the instance aware selection to be beneficial?

2.	Scalability and practicality: how do your algorithms scale when |X| is large and the hypothesis class or feature map is high dimensional? For the linear and polynomial constructions, how do you find the required vectors/preimages when only a constrained finite X is available?

3.	Relaxing realizability: how do your instance agnostic results change when realizability fails (label noise or h* not in H)? Can your VC reduction approach be adapted to agnostic or noisy settings?

4. Stronger baselines: have you compared OPT VC / OPT DM to greedy heuristics that approximate VC reduction or information gain selection? If not, can you run such comparisons?

5.	Datamodel training cost: how many meta subsets and base trainings are needed to obtain a usable datamodel in your synthetic experiments? Can you estimate computational requirements for larger problems and propose practical approximations?

6.	Feature preimage existence: for polynomial/kernel results you assume preimage existence. Which common kernels satisfy this, and what do you propose when preimages do not exist?

7.	Negative examples and reduction: the linear class analysis suggests negative labeled examples do not help reduce ambient dimensionality. Can you provide intuition or caveats when you cannot freely choose inputs (constrained X) or under noisy labels?

8.	Empirical sensitivity: can you show sensitivity analyses for OPT DM to (a) number of meta samples, (b) regularization λ in Lasso, and (c) errors in estimated PX (when PX is estimated rather than known)?

9.    Practical teaching scenarios (new suggestion): please consider evaluating or discussing more realistic practical teaching scenarios to demonstrate broader applicability. For example: Large pretrained models or LLMs teaching downstream agents (e.g., an LLM producing demonstrations or curricula for smaller RL agents or classifiers).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a new framework, "nurture-then-nature", which focuses on optimal dataset design for a first phase that comes from a teacher with a limited budget, and aims to minimize error after learning from a nature phase, whose distribution is either known or unknown by the teacher. It presents algorithms with guarantees for both settings, and provides experiments to help conceptually explain and support the theory.

### Strengths
- introduces and studies a new framework for budget-constrained teaching, inspired by practical situations
- provides experiments that aim to conceptually clarify the theoretical aspects
- i am less familiar with the theory, but sections 4 and 5 seem to make sense under the assumptions

### Weaknesses
- while there is practical motivation for the setting, there is less demonstrated practical applicability, especially in the experiments section
- the experiments are designed with toy datasets specifically with the framework in mind, but do not provide empirical justification for the strength of the algorithms introduced. this could potentially be improved with the use of real datasets and additional baselines
- assumptions made seem unlikely to hold in a realistic setting, which would require additional experiments to demonstrate practicality

### Questions
Most of my concerns are with the experimental section, as described in weaknesses. I would be happy to adjust my review if they are addressed, or if the authors could justify why the existing experiments are sufficient.

### Soundness
2

### Presentation
2

### Contribution
2
