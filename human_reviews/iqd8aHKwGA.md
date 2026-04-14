# Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs

- Decision: Reject
- Scores: 8, 6, 6, 6, 3, 5

## Abstract
Quadratic programming (QP) is the most widely applied category of problems in nonlinear programming. Many applications require real-time/fast solutions, though not necessarily with high precision. Existing methods either involve matrix decomposition or use the preconditioned conjugate gradient method. For relatively large instances, these methods cannot achieve the real-time requirement unless there is an effective preconditioner. Recently, graph neural networks (GNNs) opened new possibilities for QP. Some promising empirical studies of applying GNNs for QP tasks show that GNNs can capture key characteristics of an optimization instance and provide adaptive guidance accordingly to crucial configurations during the solving process, or directly provide an approximate solution. Despite notable empirical observations, theoretical foundations are still lacking.

In this work, we investigate the expressive or representative power of GNNs, a crucial aspect of neural network theory, specifically in the context of QP tasks, with both continuous and mixed-integer settings. We prove the existence of message-passing GNNs that can reliably represent key properties of quadratic programs, including feasibility, optimal objective value, and optimal solution. Our theory is validated by numerical results.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the expressive power of Graph Neural Networks (GNNs) for solving linearly constrained quadratic programming (LCQP) and its mixed-integer variant (MI-LCQP). The authors focus on a key question: Do GNNs possess sufficient expressive power to predict feasibility, optimal objective value, and an optimal solution? The paper provides an affirmative answer for general LCQPs but a negative one for general MI-LCQPs. However, the authors also identify specific subclasses of MI-LCQP for which the answer is affirmative.

### Strengths
This paper establishes a solid theoretical foundation for the application of GNNs to LCQPs and MI-LCQPs. The writing is clear.

### Weaknesses
1. In order to strengthen the motivation, I would suggest including a survey of previous empirical successes of GNNs in quadratic programming or providing 2-3 specific examples of recent empirical successes of GNNs in quadratic programming, along with brief descriptions of the key results and implications.
2. The paper lacks an analysis of GNN size and depth. These factors are crucial in practical applications.

### Questions
Refer to "Weakness"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work investigates the expressive power of graph neural networks (GNNs) in representing both continuous and discrete quadratic programs (QPs). Specifically, it validates three key capabilities related to the empirical performance of GNNs for QPs: accurately predicting feasibility, optimality, and optimal solutions. By answering these questions, the study bridges the gap between theoretical and empirical aspects, shedding light on the strong performance observed in prior research in this field. The authors also provide sufficient numerical results to support their claims.

### Strengths
1. Theoretical foundation: The paper successfully bridges the gap between theoretical predictions and empirical results by demonstrating the capacity of GNNs to represent QPs.
2. Solid theoratical proofs: The authors provide well-founded proofs to support their theoretical claims.
3. Impactful: The discoveries in this work pave the way for further exploration of GNN applications in accelerating solutions to QP problems, holding the potential for significant empirical advancements.
4. The paper is generally well-written and easy to follow.

### Weaknesses
The primary concern regarding this work is the specific type of MI-LCQPs that can be represented by GNNs. 
Providing examples of application scenarios for this type of problem would help strengthen the authors' claims, as mixed-integer problems with unique optimal solutions do not appear to be very common in practice.

### Questions
Please refer to "Weakness".

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work explores the expressive power of GNNs for LCQP and MI-LCQP, representing a significant contribution to the field. The authors provide a valuable framework and a theoretical foundation to demonstrate GNNs' capabilities in representing feasibility, optimal objective values, and solutions for LCQPs. Detailed proofs and well-structured analysis are commendable, and the organization of the paper is logical and flows well.

### Strengths
The paper offers a strong contribution to understanding the expressive power of Graph Neural Networks (GNNs) for Quadratic Programming (QP), particularly for linearly constrained (LCQP) and mixed-integer (MI-LCQP) problems. The strengths of the paper include:
1. The authors have innovatively connected GNNs with LCQPs and MI-LCQPs, an area that has received limited attention, thereby expanding the potential applications of GNNs in optimization.
2. The proofs are well-structured, and the reliance on the Weisfeiler-Lehman (WL) test to support the expressive power of GNNs is a thoughtful and sound approach. 
3. The paper is well-organized, with a logical flow from background concepts to theoretical contributions and then to experimental results. 
4. The significance of this work lies in its potential impact on optimization problems where approximate solutions are acceptable.

### Weaknesses
Several aspects could be improved for greater applicability and clarity:
1. The assumptions such as MP-tractability and unfoldability in Section 4, as well as the requirements for Q to be symmetric and the mappings to be continuous, might limit the broader applicability of the results in practical scenarios. 
2. Certain assumptions, such as the existence of an optimal solution for MI-LCQPs and continuity in mappings, are necessary for the proofs but lack a detailed discussion of their purpose and impact on the proof process. Without understanding why these assumptions are critical, readers might find it difficult to gauge the generalizability and limitations of the theoretical results.
3. The experimental validation relies on randomly generated LCQP and MI-LCQP instances, which may not fully capture the diversity and complexity of real-world problems. 
4. While embedding size is explored, there are few ablation studies to test other aspects of the model, such as alternative message-passing mechanisms or different GNN architectures.

### Questions
Here are several questions and suggestions for the authors:
1. Could the authors elaborate on the situations where MP-tractability and unfoldability assumptions might not hold? Are there practical examples where these conditions might fail, and how would that impact the use of GNNs in those cases?
2. The paper includes several assumptions, such as Q being symmetric and the mappings being continuous. Could the authors explain why these assumptions are necessary and how they influence the proof process? 
3. Beyond the testing on synthetic data, can the authors discuss the limitations or difficulties of applying for the GNN in practical LCQP and MI-LCQP problems, e.g., applications or optimizations in power system operations or financial portfolio? 
4. Could the authors provide more experimental insights into how different message-passing mechanisms or GNN architectures might affect performance? 
5. The WL test is central to the theoretical results but may be challenging for some readers. Can the authors provide more intuitive insights or examples to illustrate how the WL test is applied in this context?

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
4

### Summary
In this article the authors present new regarding the expressivity of GNNs for Quadratic Programming, and more specifically for Quadratic Programs (QPs) with linear constraints (continuous and mixed integer). In particular, they explicit a class of QPs (called MP-tractable) such that for each QP in the class, there exists a GNN that can provide accurate information about the feasability and the optimal value of the QP. The main results are stated in Theorem 3.3 and Theorem 3.4, and hold in probability, given a distribution on the considered class of QPs.

### Strengths
- The paper has complete introduction and clearly states the contribution.

- The authors combine results in Approximation Theory and properties of GNNs to answer fundamental questions about their expressivity in the framework of QPs.

- The authors present positive and negative results regarding this expressive power.

- The authors have added numerical experiments to illustrate and support their claims.

Overall, I find the motivations of the study and the results interesting. In particular the positive results in the positive semi-definite case interesting. However, I have some reservations as detailed next.

### Weaknesses
- The MP-tractable class is not very insightful or practical to work with. Can the authors elaborate if the criterion they propose is efficient to be tested? (as one of the initial motivations in the introduction was to improve efficiency in solving QPs?)

- The presentation of the technical parts could be shortened for clarity.  Cf. also my two questions below.

- The notations and statements in the proofs of the appendix are not very well chosen, making them more difficult to follow for ``superficial'' reasons. The authors do not include a lot of their proofs, very often coming from other papers.

### Questions
- Can the authors elaborate on the difference of second part of Theorem 3.3 and Theorem 3.4? Isn't it true that Theorem 3.4 implies the second part of Theorem 3.3 simply by continuity of the evaluation of the objective function?

- Same question applies to Th. 4.7 vs Th. 4.9.

- (Due to weakness 1) In which part exactly did the authors prove the separation property claimed page 5?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigates the expressive power of GNNs in representing key properties of quadratic programs (QP) and mixed-integer QP (MI-QP). It provides a theoretical foundation for using GNNs to approximate feasibility, optimal objective values, and solutions for QPs. The study includes both theoretical proofs and experimental results.

### Strengths
1. **Importance of GNN for QP**: The paper tackles an important problem by providing a theoretical foundation on the representational power of GNNs for quadratic programming, which is relevant to various practical applications.
2. **Clarity of Writing**: The paper is well-structured, making it easy to follow. The clear explanations and organized framework enhance readability and accessibility.

### Weaknesses
1. **Novelty and Limited Contributions**: While the paper aims to establish theoretical insights into GNNs' representation capabilities for QPs, much of its theoretical approach relies on Chen et al. (2023a, b), which studies the representation power of GNNs for LP and MILP. The paper’s theoretical contributions to the QP domain are thus incremental and may lack non-trivial novelty.

2. **Empirical Validation**: Although the paper includes some experimental validation, the scope is limited. A stronger empirical component using QP benchmark datasets is needed to substantiate the claims further. Additionally, the current experiments focus mainly on synthetic data, which may not sufficiently represent real-world QP applications.

3. **Handwaving Claims in Proofs**: Several proof sections lack rigor and rely on handwaving arguments. For example, statements like "xxx can be proved following the same lines as in the proof of Theorem xxx in Chen et al. (2023a), with trivial modifications to generalize results for LP-graphs to the LCQP setting," fail to provide a rigorous and self-contained argument, making it challenging for readers to verify the results.

### Questions
Compared to Chen et al. (2023a, b), what are the non-trivial contributions of the theoretical analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper studies whether GNN can represent key properties (such as feasibility, optimal value, and optimal solution) of LCQP (Linearly Constrained Quadratic Programming) and MI-LCQP (Mixed-Integer LCQP) problems. The study establishes the following theoretical results:

1. **LCQP**: GNNs can universally represent the feasibility of LCQPs. For **convex** LCQPs, GNN can also universally represent the optimal values and optimal solutions.

2. **MI-LCQP**: GNNs can NOT universally represent MI-LCQPs in terms of either feasibility, optimal values, or optimal solutions. The authors provide counter-examples to prove it. Then, the authors identify a subset of MI-LCQPs, on which the GNNs can universally represent MI-LCQPs in terms of all the three key properties.

The authors further provide small-scale experimental evidence to support the theoretical results.

### Strengths
1. The paper studies a crucial topic in the field of learning to optimize (L2O). GNNs have been extensively utilized for learning key properties—primarily optimal solutions—of optimization problems. By focusing on the representational power of GNNs in this context, the study offers theoretical insights and potential practical guidance.

2. In previous work, significant progress has been made in analyzing the representation power of GNNs in LP and MILP. Extending this analysis to nonlinear programming problems like LCQP and MI-LCQP is good. This paper made a commendable exploration of these extensions.

3. The paper is well-written and highly accessible, making it easy to follow.

### Weaknesses
My major concern lies in distinguishing the analysis in this paper from previous studies of GNN's representation power on LP/MILP (Chen et al. (2023a), Chen et al. (2023b)). Two significant similarities raise questions about the originality of the analysis technique:

1. To prove the universal representation power of LCQP,  the authors leverage the Stone-Weierstrass theorem to establish the GNN's approximation capabilities based on their separation capabilities. This approach appears identical to that used by Chen et al. (2023a) for LPs. In particular, the proof ideas of Theorems 3.3 and 3.4 (illustrated on Page 5) seem directly translatable to LPs, with minimal modification (i.e., replacing LCQPs with LPs). If this understanding is correct, the theoretical innovation for LCQPs might be limited.

2. The definition of unfoldable MI-LCQPs, which allows GNNs to achieve universal representation, closely mirrors the definition of unfoldable MILPs in Chen et al. (2023b). Furthermore, the proof techniques for establishing the representation power for unfoldable MI-LCQPs appear largely similar to those used for unfoldable MILPs in the prior work.

In summary, while the paper provides a comprehensive extension to LCQPs and MI-LCQPs, it is unclear whether its main proofs represent substantial innovations or are direct (albeit not completely trivial) extensions of the methodologies in Chen et al. (2023a, 2023b).

### Questions
I will consider changing my opinion if the authors can address my comments in "weakness".

### Soundness
3

### Presentation
3

### Contribution
2
