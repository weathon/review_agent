# Leveraging Variable Sparsity to Refine Pareto Stationarity in Multi-Objective Optimization

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 8

## Abstract
Gradient-based multi-objective optimization (MOO) is essential in modern machine learning, with applications in e.g., multi-task learning, federated learning,  algorithmic fairness and reinforcement learning. In this work, we first reveal some limitations of Pareto stationarity, a widely accepted first-order condition for Pareto optimality, in the presence of sparse function-variable structures. Next, to account for such sparsity, we propose a novel solution concept termed Refined Pareto Stationarity (RPS), which we prove is always sandwiched between Pareto optimality and Pareto stationarity. We give an efficient partitioning algorithm to automatically mine the function-variable dependency and substantially trim non-optimal Pareto stationary solutions. Then, we show that gradient-based descent algorithms in MOO can be enhanced with our refined partitioning. In particular, we propose Multiple Gradient Descent Algorithm with Refined Partition (RP-MGDA) as an example method that converges to RPS, while still enjoying a similar per-step complexity and convergence rate. Lastly, we validate our approach through experiments on both synthetic examples and realistic application scenarios where distinct function-variable dependency structures appear. Our results highlight the importance of exploiting function-variable structure in gradient-based MOO, and provide a seamless enhancement to existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Multi-objective optimization (MOO) have many applications in machine learning. Multiple Gradient Descent Algorithm (MGDA) is essential to solve MOO, converging to a Pareto stationary solution, which serves as a first-order necessary condition for Pareto optimality. This work demonstrates that the Pareto Stationarity (PS) has limitations when sparse function-variable dependencies exist, and to address it, they propose a concept named Refined Pareto Stationarity (RPS). With a suitable designed partitioning procedure, they propose an optimization algorithm RP-MGDA, which is effective in both the theory and experiments.

### Strengths
- The contributions of this paper are very clear. They provide a refined concept, and based on this, they propose a novel MOO algorithm.

- Under some convexity, they prove RPS reduces exactly to Pareto optimality, whereas the widely-used PS does not, suggesting that the new solution concept is more superior.

- A more powerful algorithm is proposed, and the advantages are supported in both the convergence and the experiments.

### Weaknesses
- The aim of the partition of variables should be clearly interpreted in the Introduction.

- I apologize that I am not very familiar with this topic and the relevant references. I will carefully refer to the comments of other reviewers.

### Questions
Is variable sparsity a common phenomenon in MOO?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work highlights the limitations of Pareto stationarity when dealing with sparse function-variable structures, offering compelling examples to illustrate these constraints. To overcome these challenges, they introduce a new solution concept called Refined Pareto Stationarity (RPS) and present an efficient partitioning algorithm to automatically uncover function-variable dependencies.

### Strengths
1. This work is presented with good writing style, where the summarized problems with detailed explanations make it easy for readers to understand the problem addressed in this article. 

2. The set defined by Pareto stationarity is broader than that defined by Pareto optimal, a detail that previous algorithms based on Pareto stationarity have overlooked. This paper faces this gap and introduces algorithms to overcome this limitation, representing a significant improvement.

3.Theorems 1 and 2 provide a detailed discussion on the relationships among Refined Pareto Stationarity, Pareto optimality, and Pareto stationary points. Additionally, Theorems 3 and Corollary 1 offer theoretical guarantees for the specific convergence rate of RP-MGDA.
The proof seems solid but I have not carefully checked the whole Appendix.

4.The performance of MGDA and RP-MGDA was compared across various scenarios, demonstrating the effectiveness and versatility of RP-MGDA.

### Weaknesses
1.Computational Cost.  Neither MGDA nor RP-MGDA seems well-suited to large-scale machine learning problems, as stochastic variants (e.g. MOCO, MODO) are often more computationally efficient in practice. Can the RP-MGDA algorithm be adapted into a corresponding stochastic variant, and if so, would it still handle sparse parameter issues effectively after randomization?

2.Theoretical challenges. Compared to the RPS concept proposed in this paper, Pareto stationary points appear to be a clearer optimization target. What changes does the Refined Pareto Stationarity bring to the theoretical proof? What challenges arise from these changes, and how are you addressed?

3.Illustrations on Pareto front. The paper could provide a more in-depth analysis of Pareto optimal, Pareto stationarity, and Refined Pareto Stationarity. For example, the experiments could visualize the fronts corresponding to each of these concepts. Additionally, plotting the convergence trajectories of MGDA and RP-MGDA would further emphasize the effectiveness of RP-MGDA.

### Questions
1.Lack of practical examples. Could you provide some real-world examples of sparse function-variable structures? For instance, cases that exist in multi-objective federated learning or reinforcement learning.

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
3

### Summary
This paper studies the multi-objective optimization problem and reveals the limitation of the widely used metric, Pareto stationarity. Accordingly, the authors propose the refined Pareto stationarity sandwiched between Pareto stationarity and Pareto optimality. Then, the authors verify the benefit of their RP-MGDA empirically and theoretically.

### Strengths
1. This paper has a solid standing point that Pareto stationarity could be short in complex and sparse settings.
2. Experiments explore multiple variable dependency structures.

### Weaknesses
1. Some important parts are confusing in the paper. 1) How can we get the function-variable dependency structure? This is important because the dependency structures vary in different settings. 2) Algorithm REFINED_PARTITION is not clear. A more detailed explanation of it should be added. In the current version, I cannot say I fully understand it.
2. Since I did not fully understand the partition, what does "finer" or "coarser" partition mean? 
3. According to the illustration of the function-variable dependency structure and the experimental setup in lines 467-468, I understand this function-variable dependency structure as a dependency between objectives and "specific layers" in the model. However, I do not feel this setup fits my sense. From my point of view, a dependency should be between objectives and some part of weights in each layer if the model is sparse (Correct me if I am wrong). A simple example would be this: suppose we have m objectives and the model is a n-layer neural network. Then the objective $i$ is dependent on weights $[w_{1, i},..., w_{n,i}]$ where the first index of weights represents the layer index and the second index of weight represents the objective index. Why do the authors consider the dependency in your way? Also, if consider my settings, how to do the partition?
4. Though this paper has considered multiple dependency structures in the experiments, tasks, and datasets are a bit easy. Previous related papers compare the performance of Cityscapes, NYU-v2, CelebA, etc. Do authors consider checking these experiments? In addition, this paper only compares MGDA and RP-MGDA, which is short in the number of methods. Lastly, can the partition be added to other MGDA-based methods?

### Questions
Please check the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
By leveraging refined variable partitioning, this work introduces a novel solution concept, Refined Pareto Stationarity (RPS), and a variant of the Multiple Gradient Descent Algorithm (MGDA), termed RP-MGDA, to address limitations of Pareto stationarity and MGDA in multi-objective optimization. RPS provides a sharper characterization than Pareto stationarity, and RP-MGDA is proven to converge to RPS. Comprehensive experiments demonstrate the superior performance of RP-MGDA over vanilla MGDA.

### Strengths
$\textbf{S1:}$ This work reveals limitations of Pareto stationarity and  Multiple Gradient Descent Algorithm (MGDA), commonly used in multi-objective optimization, particularly when the function-variable dependency structure is sparse. The authors provide illustrative examples showing that a variable partitioning scheme is crucial for addressing these limitations.

$\textbf{S2:}$ Building on this insight, the authors introduce a novel solution concept, Refined Pareto Stationarity (RPS), which corresponds to the finest (or refined) variable partition aligned with the function-variable dependency structure. RPS is shown to be a sharper characterization than Pareto stationarity, from both the necessary and sufficient conditions perspective for Pareto optimality.  

$\textbf{S3:}$ Utilizing this refined variable partitioning, the authors propose a variant of MGDA, termed RP-MGDA, which converges to RPS and is theoretically more efficient. Empirical results further demonstrate the superior performance of RP-MGDA compared to vanilla MGDA.

### Weaknesses
$\textbf{W1:}$ If the full gradient is used in (11) of Definition 4 (Generalized Pareto Stationarity), then any Pareto stationary point with respect to any variable partition would also be Pareto stationary with respect to the trivial partition. This suggests that the current version of Definition 4 is not consistent with the motivation behind RPS or Algorithm 1 (RP-MGDA). It seems that replacing the full gradient $\nabla \bf{f}^{P_j}$ with the partial gradient $\nabla_{\bf{w}_{P_j}} \bf{f}^{P_j}$ could address this issue. Please let me know if my understanding is incorrect, as I may adjust my ratings based on your response.   

$\textbf{W2:}$ Since RP-MGDA and the vanilla MGDA have a similar convergence rate, the authors discuss the computational complexity of RP-MGDA at the end of Section 6 and state that RP-MGDA is theoretically cheaper than the vanilla MGDA, aside from the one-time overhead in line 1. However, the supporting argument lacks detail. A more thorough complexity analysis, including explicit computational cost comparisons for solving the dual subproblem in both algorithms, would be beneficial in clarifying the computational savings.

### Questions
Apart from the questions raised in the Weaknesses section, I have a few additional questions:

$\textbf{Q1:}$ In Theorem 3, it appears that $\eta \leq \min_i \frac{1}{L_i}$ may be insufficient, as it seems that (32) cannot be derived directly from (31) using (33). Could you please verify (32) in the proof of Theorem 3 on Page 16? If my observation is correct, one alternative might be to set $\eta \leq \min_i \frac{1}{2L_i}$ in Theorem 3 and adjust the proof accordingly. 

$\textbf{Q2:}$ In the experiments of Section 7.2, is a stochastic variant of RP-MGDA used? Additionally, when ReLU is applied, what type of derivative is used in the implementation of RP-MGDA? Could this method be extended to a stochastic setting, and what significant challenges might arise in making such an extension?

### Soundness
3

### Presentation
4

### Contribution
3
