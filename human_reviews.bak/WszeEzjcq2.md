# What's Wrong With Non-Autoregressive Graph Neural Networks in Neural Combinatorial Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 3

## Abstract
Neural combinatorial optimization (NCO) leverages machine learning models to tackle complex combinatorial problems by learning heuristics or direct solution construction. Graph Neural Networks (GNNs) are particularly effective for NCO due to their ability to capture the relational structure inherent in many such problems. In this work, we examine the supervised non-autoregressive (NAR) solution construction framework, revealing a misalignment between training objective and solution quality. Specifically, through experiments on six GNN architectures across three problems—Traveling Salesperson Problem (TSP), Maximum Independent Set (MIS), and Minimum Vertex Cover (MVC)—we show that lower training loss does not correlate with lower optimality gap. To address this, we propose a supervised autoregressive (AR) framework that leverages the conditional dependencies between variables by training to complete partial solutions. Empirical results show that the proposed AR framework does not exhibit the same misalignment and consistently improves performance. We further compare the proposed AR framework against existing supervised GNN-based methods and achieve superior performance, especially in terms of generalizing to larger problem instances.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper identifies a critical limitation in current supervised non-autoregressive GNN approaches for neural combinatorial optimization, where a misalignment exists between the training loss and the optimality gap. The problem is illustrated by testing various GNN architectures across multiple combinatorial optimization tasks, revealing that a lower training loss does not necessarily result in better solutions. The authors attribute this issue to the use of a single probability map, which does not align well with search algorithms that greedily build solutions. To address this, they propose an autoregressive approach in which GNNs are trained to complete partial solutions, producing probability maps conditioned on the current solution state. Training is accomplished by sampling subsets of optimal solutions. Empirical results indicate that this approach successfully mitigates the misalignment and outperforms existing GNN-based methods, generalizing to larger problem instances.

### Strengths
1. The paper addresses a significant issue of misalignment between training loss and optimality gap, which is common in existing supervised GNN-based approaches.
2. The proposed autoregressive GNN approach, which completes partial solutions and can correct suboptimal decisions made earlier, is innovative, effective, and well-suited.
3. The empirical results are promising, with the first experiment validating that the method addresses the misalignment issue and the second demonstrating superior performance over other GNN-based methods.
4. The writing is clear and well-structured, with strong motivations and a clearly explained methodology.

### Weaknesses
1. In the experiments where the misalignment problem is observed, only a greedy search algorithm is used, where at each step, one variable is added to the solution. This brings the question of whether the problem is universal, i.e. if the misalignment still occur with other search algorithms. For example, if the search algorithm is not iterative in nature (a simple example is setting the threshold of 0.5 to determine if a solution should be included), whether the misalignment is still a problem. 
2. The greedy search algorithm adds one variable at a time. However, from my understanding, the GNN learns to complete a partial solution, not having one variable change at a time. This seems to misalign with the conditional probability desired by the search algorithm.  
3. What is the multimodal nature of combinatorial optimization that the author mentions that the proposed method better captures?

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the supervised Non-Autoregressive Graph Neural Network frameworks and finds a misalignment between the training objective (i.e., minimizing the loss function) and the quality of the constructed solutions. 
In summary : improvements in the quality of the probability maps do not necessarily lead to better solutions.
The reason authors provide is that NAR approaches assume independence between different variables, ignoring the dependencies present in combinatorial optimization problems.

The paper proposes using autoregressive (AR) models with supervised learning. They propose to consider the previously predicted variables when predicting the next one, capturing the conditional dependencies. The proposed framework propose to perform training to complete partial solution.

The authors empirical show that their proposed method outperforms Non Autoregressive methods on 3 problems(TSP, MVC and MIS) . Further it also generalizes better to larger size data.

Concerns:
I have some concerns related to amount of training data used and usage of only greedy decoding to perform benchmarking of existing works. Please see weakness.

### Strengths
1. The idea of using Supervised learning(SL) and auto-regressive approach of CO is relevant.
2. Making prediction based upon partial solution seems to be novel for NCO and SL.

### Weaknesses
1. For training, we used 10,000 instances of TSP503 for our model. For the baselines,
1,502,000 instances of TSP50 are used to train DIFUSCO and 10,000 instances of TSP50 are used to
train EFFICIENTTSP, as per their original manuscripts.

-> is 10000 a large number? previous studies have used millions of instances[A].
Why was it limited to only 10000? How does the performance change with increase in dataset size ( for different methods).
Does the conclusion remain same. I would expect to see training/validation curves for different problems and methods.



2. "For each problem, the training set consists of 5,000 random synthetically generated
problem instances, each with 100 nodes"
For explaining failure of NR supervised learning methods the authors use 5000 training samples. It seems to be a very small number.  Check [A]. I would recommend the authors to use a significantly large training dataset.

3. "we employ greedy search in order to evaluate the impact of the probability maps in isolation."

Why only greedy for all problems, what restricted the authors not to use ideas like beam-search/ sampling for all problems, if not for TSP ? Check [A]. Can we conclude with just greedy decoding?



[A] An Efficient Graph Convolutional Network Technique
for the Travelling Salesman Problem
https://arxiv.org/pdf/1906.01227

### Questions
Check weakness.  I am majorly concerned by the amount of training data used for the experiments.

### Soundness
1

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
5

### Summary
This paper investigates non-autoregressive (NAR) Graph Neural Networks (GNNs) for neural combinatorial optimization and identifies a key misalignment between training loss and solution quality. The authors attribute this misalignment to NAR models’ inability to capture inter-variable dependencies, and they propose an autoregressive (AR) alternative to improve the alignment between training objectives and solution quality. Empirical evaluations on standard problems such as Traveling Salesperson Problem (TSP), Maximum Independent Set (MIS), and Minimum Vertex Cover (MVC) demonstrate that the AR approach outperforms existing supervised GNN-based NAR methods, particularly in generalizing to larger instances.

### Strengths
This work stands out for its empirical examination of various GNN architectures and the experimental insights into the limitations of NAR models on the combinatorial optimization tasks TSP, MIS, and MVC. The proposed AR framework, which iteratively builds solutions based on partial solutions, shows promising improvements over traditional NAR approaches. Furthermore, the authors provide a comprehensive experimental setup and benchmarks, which enhances the clarity of the paper and the credibility of its results.

### Weaknesses
While the paper identifies valid shortcomings in NAR methods, it does not fully address these limitations within the NAR framework itself. Instead of proposing an enhancement that would make NAR models more effective, the authors shift to an AR approach, which is already well-established for combinatorial optimization tasks. Given that current state-of-the-art AR models, particularly in the TSP domain, achieve near-optimal solutions on problems up to TSP200, the choice to compare a new AR model to NAR models could be seen as a misalignment of goals. Although the empirical results favor the AR model over NAR approaches, this comparison does not substantially advance the field of NAR or show a path forward for improving NAR’s inherent limitations. Moreover, the results on the AR framework fall short of reaching or challenging the current state-of-the-art in AR models, which diminishes the work’s impact in terms of advancing solution quality or generalization ability for AR approaches.

### Questions
Can the authors clarify why an autoregressive method was chosen instead of attempting to refine or address the limitations of NAR methods directly? It would be helpful to understand whether any NAR-specific modifications were considered.

Given that state-of-the-art AR approaches already perform well for small and medium-sized TSP instances, how does the proposed AR method compare to these benchmarks in terms of optimality and scalability? The AR framework here still falls short of achieving performance competitive with top-performing AR models, which raises the question of its comparative effectiveness.

### Soundness
2

### Presentation
2

### Contribution
1
