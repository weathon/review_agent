# Jacobian Descent for Multi-Objective Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 6

## Abstract
Many optimization problems require balancing multiple conflicting objectives.
As gradient descent is limited to single-objective optimization, we introduce its direct generalization: Jacobian descent (JD).
This algorithm iteratively updates parameters using the Jacobian matrix of a vector-valued objective function, in which each row is the gradient of an individual objective.
While several methods to combine gradients already exist in the literature, they are generally hindered when the objectives conflict.
In contrast, we propose projecting gradients to fully resolve conflict while ensuring that they preserve an influence proportional to their norm.
We prove significantly stronger convergence guarantees with this approach, supported by our empirical results.
Our method also enables instance-wise risk minimization (IWRM), a novel learning paradigm in which the loss of each training example is considered a separate objective.
Applied to simple image classification tasks, IWRM exhibits promising results compared to the direct minimization of the average loss.
Additionally, we outline an efficient implementation of JD using the Gramian of the Jacobian matrix to reduce time and memory requirements.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper first proposes using the Jacobian matrix to analyze the multi-objective algorithms and provides convergence guarantees. A new method called AUPGrad was proposed, which can project and resolve conflicts while preserving the relative influence of each target gradient (In fact, I am very skeptical about this). It also proposed reducing the time and memory complexity of the algorithm by using the Gramian of the Jacobian matrix.

### Strengths
A new algorithm has been proposed to reduce complexity using the Gramian of the Jacobian matrix

### Weaknesses
1. First of all, the writing of this paper is disastrous due to the lack of necessary introduction to multi-objective algorithms. For those unfamiliar with multi-objective algorithms, especially Pareto optimality, it is really difficult to understand this paper. The Pareto frontier, as an important concept in this paper, is not introduced at all. Additionally, there is a significant lack of explanation for various details. For example, in Figure 1, we are dealing with a multi-objective optimization problem, but why does Figure 1 only consider two objectives? (There are many such issues!)

2. Regarding the claim in the abstract about "completely resolving conflicts," I have serious doubts. As far as I know, when an algorithm reaches the Pareto frontier, no direction can make the objective function decrease simultaneously. If this algorithm resolves conflicts, how does it converge to the Pareto frontier?

3. In Figure 1, we observe that MGDA (Multi-Gradient Descent Algorithm) is also in the dual cone. I find it hard to understand what distinct advantages this paper has over MGDA. (It’s also hard to see some advantage of this paper compared to the SMG[1] algorithm.)

4. In fact, projection-based multi-objective algorithms have been researched in recent years (e.g., MoCo[2]), but this paper does not show any advantage over these works, nor does it compare with them.

Given these issues, I highly doubt the contribution of this paper. Studying algorithms from the perspective of the Jacobian matrix's properties seems like an interesting direction, but the writing problems and lack of novelty in this paper make it difficult for me to remain positive.

[1]Liu, Suyun, and Luis Nunes Vicente. "The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning." Annals of Operations Research 339.3 (2024): 1119-1148.
[2]Fernando, Heshan Devaka, et al. "Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach." The Eleventh International Conference on Learning Representations.

### Questions
The presentation of this paper needs significant revisions. Additionally, the article lacks innovation and requires a stronger comparison with similar algorithms.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces the Jacobian Descent (JD) algorithm, an extension of gradient descent designed for multi-objective optimization. Unlike traditional scalarized multi-objective methods, JD optimizes vector-valued objective functions by updating model parameters with the Jacobian matrix.  Additionally, the paper introduces Instance-wise Risk Minimization (IWRM), a new optimization paradigm that considers each training sample as an independent objective. This allows JD to optimize at an instance level, yielding promising results in empirical evaluations on image classification tasks.

### Strengths
Innovative Contribution: The JD method presents a clear and novel perspective to gradient descent for multi-objective optimization, an area of significant relevance to fields like multi-task learning and adversarial training.

Applicability to Emerging Areas: By addressing applications such as federated learning and distributed optimization, the paper suggests impactful future directions for JD and AUPGrad beyond standard optimization problems.

### Weaknesses
1. Not Very Well-Structured: The organization of the content in this article is  inconsiderate. For example, in the beginning of Section 2, the definition of the relationship between vectors u and v is actually related to the Pareto optimal condition. It would have been more appropriate to clearly introduce the definitions related to Pareto before presenting the definitions used in this paper. However, the paper  provides the Pareto optimal condition for the first time in Section 2.4. If the reader is not familiar with the multi-objective problem, they may struggle to understand the author's intent while reading the paper.


2.Limited Comparison to State-of-the-Art: Theoretically,  The paper didn't provide a specific convergence rate for JD  and avoided comparing it with the convergence rates of other gradient-based multi-objective optimization algorithms. This omission may limit the reader's ability to assess the efficiency and competitiveness of the proposed method against existing approaches in the field. 


3.Scalability Concerns: While the proposed Gramian-based approach aims to improve efficiency, the JD method remains computationally expensive for large-scale problems, which might hinder practical applications, especially in high-dimensional or real-time scenarios.

### Questions
1) This paper seems to deliberately avoid comparison and discussion with multi-objective algorithms based on Pareto stationary. However, in reality, whether it is the definition of 'non-conflicting' or the design of the AUPGrad algorithm, the paper merely shifts the perspective for solving the problem to the Jacobian matrix. The core idea is not fundamentally different from previous algorithms. Could you provide a detailed discussion on the differences between this algorithm and MGDA, as well as its stochastic variants?

2)  Theoretically, since this algorithm does not provide specific convergence results, how can we evaluate the performance of this algorithm compared to other multi-objective optimization algorithms?

3) This paper lacks visualizations of the algorithm's convergence trajectories to the Pareto front. Visualizing the optimization paths of multiple initial points is also a crucial aspect of presenting the optimization effects of multi-objective algorithms from an experimental perspective.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors extend the concept of gradient descent to JD by using the Jacobian matrix of a vector-valued objective function, with each row representing the gradient of a different objective. A key contribution is the AUPGrad (Averaged Unconflicting Projection of Gradients) aggregator, which is proposed to resolve conflicts between gradients in a way that preserves their relative influence. 

The paper provides theoretical convergence guarantees for JD with AUPGrad, especially in the context of smooth and convex functions, and demonstrates its effectiveness through experiments on image classification tasks. Additionally, the paper introduces Instance-wise Risk Minimization, a new learning paradigm that treats the loss of each training example as a distinct objective, and shows promising results compared to traditional loss minimization methods.

### Strengths
1.	Theory: The paper provides a solid theoretical foundation, including convergence proofs for JD with AUPGrad under certain conditions, which adds rigor to the proposed method.

2.	Reproduction: The source codes are provided, which is helpful to validate the realistic performance.

3.	Detailed introductions to baselines. Appendix B states the properties and computation steps of several compared algorithms, which helps to highlight and better understand the advantages of this work.

### Weaknesses
1.	Lack of mathematical backgrounds. It’s kindly suggested to add more detailed introductions to the investigated multi-objective learning and definitions on Pareto fronts.

2.	Theory. The theoretical guarantees provided for JD with AUPGrad rely on assumptions of smoothness and convexity, which may not hold in some practical scenarios, limiting the applicability of the results. More practical examples are required. Could you please further derive the convergence rate? Some computation complexity from the theoretical perspective is also crucial for new algorithms.

3.	Broader baselines and empirical settings. 
The experiment setting is limited, while comprehensive benchmarks are required to validate across a variety of problems, especially those with inherently conflicting objectives.

Besides, more empirical scenarios of classical multi-objective learning tasks [1] are also helpful.

[1] Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance

4.	Moreover, the authors are kindly suggested to adopt some accelerate techniques on approximating the Jacobian counterpart, instead of merely taking averages. As present in Table 7, the proposed A_{UPGrad} consumes more time for training.

### Questions
Please see the comments in Weakness.

I would be willing to raise my score if these questions concerned are well addressed.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose a general framework for multi-objective optimization (MOO), termed Jacobian Descent (JD), where the multi-objective optimization update direction is viewed as the output of an aggregator mapping. This mapping takes the Jacobian of the multi-objective vector and maps it to a vector in the dimension of the parameters to be updated. The authors introduce stochastic implementations of JD, termed Stochastic Jacobian Descent (SJD) and Stochastic Sub-Jacobian Descent (SSJD). The paper then proposes the Unconflicting Projection of Gradients aggregator, denoted as $A_{\text{UPGrad}}$, as a desirable aggregator for MOO updates. This aggregator has properties such as non-conflicting behavior (i.e., the update direction does not conflict with any individual gradient), linear scaling, and weighted combinations (the update direction is a weighted combination of the rows of the Jacobian). Theoretical results are provided for the convergence of MOO updates using $A_{\text{UPGrad}}$ to the Pareto front under certain assumptions. The authors further introduce Instance-wise Risk Minimization (IWRM) as an application of MOO for traditionally single-objective problems, where the loss at each data point is considered as an individual objective. They also discuss how some existing MOO aggregators fit within the JD framework. Experimental results are presented to demonstrate the applicability of $A_{\text{UPGrad}}$ in practical IWRM problems. Additionally, the paper outlines a more efficient implementation of JD, known as Gramian-based JD, to address the computational overhead associated with the proposed JD approach.

### Strengths
* The proposed JD framework generalizes many existing MOO approaches, providing insights into the similarities, differences, and behaviors of these approaches.
* The unconflicting projection of gradients aggregator, UPGrad, has provable convergence guarantees and demonstrates faster convergence emperically in practical applications compared to existing MOO methods.
* The paper is fairly easy to follow, the proposed methods are well motivated, and the build-up to the proposed methods are clearly explained.

### Weaknesses
* The theoretical convergence guarantees are only asymptotic, so it is unclear how JD compares with standard gradient descent in terms of theoretical convergence rate. 
* As the authors acknowledge, the proposed method is computationally intensive. It is also unclear, even with the outlined method for increasing computational efficiency, whether the computational overhead can be reduced (compared to, for example, standard SGD on the ERM objective) due to the need to compute the Jacobian. For example, even with a smaller batch size than that used for the Mean objective, applying JD for IWRM involves separate per-datapoint gradient calculations. On the other hand, applying the Mean aggregator for the ERM problem with a larger batch size might result in faster convergence with lower computational overhead. 
* While viewing ERM as an IWRM and adaptively optimizing the objective may result in faster convergence for the empirical loss, it is unclear how this kind of adaptive weight selection could affect test performance (which is the ultimate goal of most ERM applications), since the weights are assigned based on the training dataset.

### Questions
* What are the challenges that prevent deriving non-asymptotic convergence guarantees for UPGrad?
* Can the authors provide a runtime comparison for the standard SGD with ERM vs UPGrad applied for IWRM?
* Can the authors comment/provide evidence how reformulating ERM as IWRM affect the test performance of the final model?

### Soundness
3

### Presentation
3

### Contribution
3
