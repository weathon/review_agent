# Evaluating Data Influence in Meta Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
As one of the most fundamental models, meta learning aims to effectively address few-shot learning challenges. However, it still faces significant issues related to the training data, such as training inefficiencies due to numerous low-contribution tasks in large datasets and substantial noise from incorrect labels. Thus, training data attribution methods are needed for meta learning. However, the dual-layer structure of meta learning complicates the modeling of training data contributions because of the interdependent influence between meta parameters and task-specific parameters, making existing data influence evaluation tools inapplicable or inaccurate. To address these challenges, based on the influence function, we propose a general data attribution evaluation framework for meta learning within the bilevel optimization framework. Our approach introduces task influence functions (task-IF) and instance influence functions (instance-IF) to accurately assess the impact of specific tasks and individual data points in closed forms. This framework comprehensively models data contributions across both the inner and outer training processes, capturing the direct effects of data points on meta parameters as well as their indirect influence through task-specific parameters. We also provide several strategies to enhance computational efficiency and scalability. Experimental results demonstrate the framework's effectiveness in training data evaluation via several downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a framework for evaluating the influence of training data in the context of MAML, which is formulated as a bilevel optimization problem. The authors propose task influence functions (task-IF) and instance influence functions (instance-IF). These methods are designed to accurately assess the impact of entire tasks and individual data points on the meta-training process. The paper provides a theoretical foundation for these methods, including the use of total derivatives to capture the interdependent nature of meta and task-specific parameters. To address the computational complexity, the authors propose acceleration techniques such as EK-FAC and Neumann series approximation. The effectiveness of the proposed framework is demonstrated through experiments on several benchmark datasets, showing its utility in tasks like identifying and removing harmful data.

### Strengths
1. Novelty: The paper is the first to formally and introduce influence functions into the bilevel optimization framework for meta-learning, to thoroughly include task influence and data(training/validation) influence. 

2. Comprehensive and Well-Structured Framework: The proposed framework is comprehensive, addressing data influence at both the task and instance levels, and for both training and validation data. The distinction between these different levels of influence is well-motivated and clearly articulated.

### Weaknesses
1. The scope: The proposed IF evaluation method is only applicable for second-order MAML, which is only one specific algorithm among numerous  meta-learning algorithms. A narrowed title and scope, e.g. "Evaluating data influence in MAML by IF" would be more proper and accurate.

2. Experimental and practical concern: The burden calculating IF is concerned, though some approximation and acceleration techniques have been applied. It seems there are much more simple and intuitive baselines in the application scenario of IF. For example, for retraining, a simple baseline is taking few gradient-ascend steps on the data to remove (both task/instance (train/valid)) are applicable), but has not been compared. For identifying harmful data, a simple baseline is evaluating  the loss scalar, but also has not been compared.

3. Related work: it seems that [1] has included task-if of MAML, which weakens the novelty and contribution of this work, but not has been cited or discussed.

[1] Mitsuka Y, Golestan S, Sufiyan Z, et al. TLXML: Task-Level Explanation of Meta-Learning via Influence Functions. arXiv preprint arXiv:2501.14271, 2025.

### Questions
Please refer to Questions

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies how to quantify the contribution of individual tasks and samples in meta-learning.It extends the classical influence function to the bilevel setting of meta-learning, where both task-specific and meta-level parameters interact.

It introduces two estimators: Task-IF, which measures the influence of entire tasks, and Instance-IF, which aims to measure the effect of individual samples inside tasks.

Experiments on few-shot benchmarks show that their method can efficiently approximate the results of full retraining while identifying harmful or noisy data.

In essence, the paper attempts to make data attribution feasible for meta-learning by deriving tractable influence estimators.

### Strengths
1. Introduced influence functions into meta-learning and adapted them to the bilevel optimization structure. 
2. Clarified why directly applying standard influence functions in bilevel optimization is incorrect and derived the corrected Task-IF formulation that properly accounts for inner–outer dependencies.
3. Designed two types of influence functions, Task-IF and Instance-IF, enabling multi-granularity data evaluation.

### Weaknesses
1. The proposed method for training instance influence, which models the outer-loss change as a simple additive proxy P, implicitly relies on the sufficiency of local, linear approximations. This two-stage linear approximation (first to compute P, then to compute its influence) may fail to capture the complex, non-linear effects of removing a data point, especially in optimization landscapes with high curvature or when the removal of a point induces a significant shift in the task-specific parameters.

2. The influence functions proposed in the paper are derived reasonably, but multiple approximations may lead to uncontrollable cumulative errors. The influence function is based on a first-order Taylor approximation; to avoid third-order derivatives, the total hessian matrix is simplified. Additionally, the computation of the inverse Hessian-vector products relies on iterative approximations. The paper does not provide an analysis of these cumulative errors.

3. The experiments lack diversity in tasks, as the paper only evaluates few-shot image classification tasks. However, the loss landscapes of different types of tasks can vary significantly, which may affect the performance of Hessian-based influence functions, the core method of this paper.

### Questions
1. Can you analyze the range of cumulative errors, or compare Task-IF and Instance-IF predictions of \Delta\lambda to the true change obtained by removing a task/sample and retraining?

2. In Proposition 4.8, the calculation of the Instance-IF for training data relies on the term D_\lambda P(\lambda^*, \theta_k(\lambda^*); \tilde{z}). Given that P is a function of the inner-level Hessian inverse H_{k,in}^{-1}, which itself depends on \lambda, the full derivative D_\lambda P would involve third-order derivatives (the derivative of H_{k,in}^{-1} with respect to \lambda). Could you clarify whether your implementation includes these third-order derivative terms? Did you use the same approximation strategy as Task-IF to ignore these higher-order derivatives?

3. How sensitive are your influence estimates to the conditioning of H_{in}? Are the performances of Task-IF and Instance-IF stable across different types of tasks?

### Soundness
2

### Presentation
2

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
This paper proposes a general, closed-form framework to evaluate training-data influence in meta-learning, introducing two complementary measures, task-IF and instance-IF. They quantify how entire tasks and individual examples affect meta parameters directly and indirectly through task-specific parameters. It formalizes meta-learning as a BLO problem, derives total gradients and a total Hessian tailored to the bilevel coupling, and shows how to estimate parameter changes from up-weighting or removing data without full retraining. To make the method practical at scale, the authors add acceleration strategies for inverse Hessian vector products and approximate the total Hessian, then demonstrate utility on standard few-shot benchmarks for tasks like harmful-data identification and editing efficiency.

### Strengths
- The paper cleanly formulates meta-learning as bilevel optimization and introduces complementary task- and instance-level influence measures tailored to that structure.

- It contributes practical scalability via acceleration techniques, making the closed-form influence derivations computationally feasible.

- The experimental setup is comprehensive.

### Weaknesses
- The paper lacks a true ablation study isolating the contribution of its key components.

- Accuracy gains over strong baselines are modest and sometimes below retraining, with the main advantage being runtime.

- Reproducibility is constrained by the absence of a code repository or data link in the submission.

### Questions
- Please provide an ablation isolating the contribution of each ingredient so we can see where the gains come from and whether any component is redundant.

- How do you choose the truncation depth J in practice? Reporting accuracy and runtime vs J would help.

- Can you quantify the error or give guidance on when EK-FAC is acceptable vs when the Neumann route is mandatory?

### Soundness
3

### Presentation
3

### Contribution
2
