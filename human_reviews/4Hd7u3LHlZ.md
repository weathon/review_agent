# Primal-Dual Graph Neural Networks for General NP-Hard Combinatorial Optimization

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 6, 5, 5

## Abstract
Neural algorithmic reasoning (NAR) seeks to train neural networks, particularly Graph Neural Networks (GNNs), to simulate and generalize traditional algorithms, enabling them to perform structured reasoning on complex data. Previous research has primarily focused on algorithms for polynomial-time-solvable problems. However, many of the most critical problems in practice are NP-hard, exposing a critical gap in NAR. In this work, we propose a general NAR framework to learn algorithms for NP-hard problems, built on the classical primal-dual framework for designing efficient approximation algorithms. We enhance this framework by integrating optimal solutions to these NP-hard problems, enabling the model to surpass the performance of the approximation algorithms it was initially trained on. To the best of our knowledge, this is the first NAR method explicitly designed to surpass the performance of the classical algorithm on which it is trained. We evaluate our framework on several NP-hard problems, demonstrating its ability to generalize to larger and out-of-distribution graph families. In addition, we demonstrate the practical utility of the framework in two key applications: as a warm start for commercial solvers to reduce search time, and as a tool to generate embeddings that enhance predictive performance on real-world datasets. Our results highlight the scalability and effectiveness of the NAR framework for tackling complex combinatorial optimization problems, advancing their utility beyond the scope of traditional polynomial-time-solvable problems

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper propose a new GNN architecture, termed PDGNN, for solving the integer programming $\min \vec{w}^T\cdot\vec{x}$ s.t. $A\vec{x}\geq \vec{1}$ and $x_i\in \{0,1\}$, which can model many combinatorial optimization problems, such as set cover and hitting set. This GNN is designed by unrolling a primal-dual approximation algorithm. Numerical experiments are conducted to evaluate the performance of PDGNN.

### Strengths
The observation that the primal-dual algorithm naturally align with message-passing GNN is interesting. The integer programming studied in this paper is a quite general problem. The writing is clear and easy to follow.

### Weaknesses
1. About Section 1.1: The phenomenon that GNNs designed by unrolling an algorithm surpass the performance of the algorithm itself is not surprising (since the algorithm is an instantiation of the GNN, so the GNN has enough expressive power to surpass). In fact, the phenomenon is general: see. e.g.  "PDHG-unrolled learning-to-optimize method for large-scale linear programming" in ICML 2024. As in
 real-world scenarios, PDGNN will commonly be used as a warm start, so the paper could highlight how much computational time can be saved by employing PDGNN  as a warm-start.

2. As shown in Table 3, by employing PDGNN  as a warm-start, the commercial solver achieve a $\leq 1.1\times$ speedup compared to the vanilla version. The improvement did not impress me, as I had been expecting a $\geq 2\times$ speedup.

3. The paper proves that PDGNN can exactly replicates the primal-dual algorithm, but lacks explicit theoretical guarantees on the efficiency and effectiveness of PDGNN. For example, I would like to see theorems like "PDGNN  of xx layers has enough expressive power to output an xx-approximate solution for xxx problems"，or discussions on the challenges in deriving such theoretical guarantees or intuitions about what kind of guarantees might be possible given the current framework.

### Questions
1. Refer to "Weaknesses".

2. In the experiments, how many layers does PDGNN has?

### Soundness
3

### Presentation
3

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
This paper design a GNN model based on the primal-dual algorithm to solve CO problems including Minimum Vertex Cover, Minimum Set Cover, and Minimum hitting set. The retrained model can be used to warm start a commercial solvers, and also applicable to other problems  closely connected with the studied problems, eg node classification.

### Strengths
Writing is good with many details. The method looks sound. Experiments are comprehensive.

### Weaknesses
- To me, it is not clear whether the work is important in literature, especially in dealing with the three problems.

- Important baselines are missing. For example, other neural solvers for the three problems are not provided. GIN is not designed with the purpose of solving the problem. So the comparison with merely GIN can be unfair. To demonstrate the effectiveness, it is necessary to include more solvers for the problem.

- Authors give results of model performance on OOD data in Table 2. However, since authors do not give results of other baselines, the OOD generalizability cannot be proved directly. By the way, no OOD results on the minimum set cover problem are provided.

### Questions
- The three problems are closely related. Can the method be applied on other problems that can work with the  primal-dual algorithm?

- In table 2, why use geometric mean for total time while arithmetic mean for optimal time?

- what is the relation of PDGNN and GAT, GCN and SAGE? How are they combined as shown in Table 4? What are the inputs and outputs of each module? Since the main text does not mention how are PDGNN works with other models, the results confused me.

- what does the claimed OOD generalizability comes from? There is no module designed for the purpose. So it confuses me why the method has good OOD generalizability.

Since I am not an expert in these problems, I would consider raising my score based on the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an approach that integrates the primal-dual method with GNNs to solve NP-hard combinatorial optimization problems more efficiently. 
It establishes theoretical guarantees that the proposed framework can replicate the classical primal-dual method. 
By refining the optimal solutions for these problems, the model surpasses the performance of conventional algorithms. 
Additionally, by utilizing predictions as warm starts, the search time for commercial solvers can be reduced.
Numerical results showcase the scalability and effectiveness of the proposed framework.

### Strengths
1. The paper proposes an approach that is the first NAR method to surpass the learned algorithm.
2. Author conducted several experiments with good numerical results favoring the proposed framework.
3. Theoretical proofs are given to validate the effectiveness of the proposed framework.

### Weaknesses
While the proposed framework is designed to address NP-hard combinatorial optimization problems, the proposed framework primarily operates on linear programming (LP) relaxation, which appears to be an extension to existing methods, such as those referenced in [1] and [2]. 
Besides, there are also many works that directly predicts solutions to CO problems. e.g. [3] and [4].
An extended discussion, especially with experiments, on how the proposed approach differs from them would strengthen the authors' claims.

[1]. Bingheng Li, Linxin Yang, Yupeng Chen, Senmiao Wang, Qian Chen, Haitao Mao, Yao Ma, Akang Wang, Tian Ding, Jiliang Tang, and Ruoyu Sun. Pdhg-unrolled learning-to-optimize method for large-scale linear programming, 2024.

[2]. Ziang Chen, Jialin Liu, Xinshang Wang, Jianfeng Lu, and Wotao Yin. On representing linear programs by graph neural networks, 2023.

[3].  Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O’Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, Ravichandra Addanki, Tharindi Hapuarachchi, Thomas Keck, James Keeling, Pushmeet Kohli, Ira Ktena, Yujia Li, Oriol Vinyals, and Yori Zwols. Solving mixed integer programs using neural networks, 2021.

[4]. Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron Courville, Yoshua Bengio, and Ling Pan. Let the flows tell: Solving graph combinatorial optimization problems with gflownets, 2023.

### Questions
Please refer to "Weakness".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to simulate the primal-dual framework for approximation algorithms by Goemans and Williamson using a graph neural network. The paper follows the standard neural algorithmic reasoning blueprint with an encoder-processor-decoder approach and it leverages a gnn that performs message passing on a bipartite graph representation of the problem. The model is used to solve NP-hard problems such as minimum vertex cover, minimum hitting set, and minimum set cover. The model also leverages optimal solutions to the problem as additional supervision and is tested on real and synthetic graphs.

### Strengths
- If I understand correctly, this primal dual approach will necessarily yield a certificate of solution quality since the dual objective bounds the primal, which is not easy to obtain with neural nets on those problems.
- Ablations and experiments on real world data are helpful.

### Weaknesses
My main concerns are the following:
- The experimental comparisons are inadequate. For example, MVC (and its complement problem of max independent set) have been tackled by several works in the literature (examples can be found in [1,2,3]). Warm-starting solvers has also been done (e.g., see 4). In my view,  discussion and comparisons with some work from the literature on the core experiments as well as the warm start ones would help clarify where the model stands overall. Furthermore, the experiments on synthetic instances are rather small-scale (especially table 1). [2,3] provide an experimental setting with hard synthetic instances of larger sizes for MVC. It would be good to see how the model performs against some of those baselines on those instances.
- On the topic of scale, it would be good to see what the performance looks like if the synthetic instances were scaled up an order of magnitude. Right now, it seems like it would take a lot of supervision to train a scalable version of this algorithm, especially if one wanted to leverage optimal solutions, which become costly for NP-hard problems as the instance size grows.
- A central goal of this paper is to have a machine-learning model that closely aligns with the primal-dual framework. The numbers seem to agree in table 1 when it comes to MVC and MHS but for MSC we see that not having access to the optimal solutions seems to lead to results worse than the primal-dual algorithm itself. This points to the broader issue that even though the model is designed to closely align with this framework, the approximation guarantee is not secured in practice.

Overall, this is an interesting approach to neural combinatorial optimization. However, I find the empirical results unconvincing at this stage and ultimately the algorithm does not seem to preserve the approximation guarantee in practice so I am unsure about the contribution as a whole. At this stage. I cannot recommend acceptance but I am open to reconsidering after the rebuttal.




1. Khalil, Elias, et al. "Learning combinatorial optimization algorithms over graphs." Advances in neural information processing systems 30 (2017).
2. Brusca, Lorenzo, et al. "Maximum independent set: self-training through dynamic programming." Advances in Neural Information Processing Systems 36 (2023)
3. Yau, Morris, et al. "Are Graph Neural Networks Optimal Approximation Algorithms?." arXiv preprint arXiv:2310.00526 (2023).
4. Benidis, Konstantinos, et al. "Solving recurrent MIPs with semi-supervised graph neural networks." arXiv preprint arXiv:2302.11992 (2023).

### Questions
Do the mean-max aggregation lines in table 1 use optimal solutions in training as well? I find it a little strange that max aggregation performs better than the algorithm without the optimal solutions. Shouldn't the problem-specific aggregations do better?

### Soundness
2

### Presentation
2

### Contribution
1
