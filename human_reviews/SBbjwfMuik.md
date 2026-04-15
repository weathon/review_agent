# From Global Assessment to Local Selection: Efficiently Solving Traveling Salesman Problems of All Sizes

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
The Traveling Salesman Problem (TSP) is a well-known combinatorial optimization problem with broad real-world applications. Recent advancements in neural network-based TSP solvers have shown promising results. Nonetheless, these models often struggle to efficiently solve both small- and large-scale TSPs using the same set of pre-trained model parameters, limiting their practical utility. To address this issue, we introduce a novel neural TSP solver named GELD, built upon our proposed broad global assessment and refined local selection framework. Specifically, GELD integrates a lightweight Global-view Encoder (GE) with a heavyweight Local-view Decoder (LD) to enrich embedding representation while accelerating the decision-making process. Moreover, GE incorporates a novel low-complexity attention mechanism, allowing GELD to achieve low inference latency and scalability to larger-scale TSPs. Additionally, we propose a two-stage training strategy that utilizes training instances of different sizes to bolster GELD's generalization ability. Extensive experiments conducted on both synthetic and real-world datasets demonstrate that GELD outperforms seven state-of-the-art models considering both solution quality and inference speed. Furthermore, GELD can be employed as a post-processing method to exchange affordable computing time for significantly improved solution quality, capable of solving TSPs with up to 744,710 nodes without relying on divide-and-conquer strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Neural TSP solvers struggle to generalize across different TSP sizes and often have high computational complexity. The goal is to develop a unified model that can solve both small-scale and large-scale TSPs quickly and improve solution quality. It combines a Global-view Encoder (GE) and a Local-view Decoder (LD). GE uses a novel Region-Average Linear Attention (RALA) mechanism with low time-space complexity to capture global information. LD autoregressively selects the next node within a local selection range. A two-stage training approach is proposed. The first stage uses supervised learning on small-scale instances, and the second stage applies self-improvement learning on larger instances with a curriculum learning strategy. GELD outperforms seven state-of-the-art models in terms of solution quality and inference speed on both synthetic and real-world datasets. It can handle all test instances, while some baselines fail due to GPU memory constraints. It can also solve extremely large TSPs (up to 744,710 nodes) when integrated with heuristic algorithms.

### Strengths
The combination of a lightweight Global-view Encoder (GE) with a heavyweight Local-view Decoder (LD) and the use of Region-Average Linear Attention (RALA) in GE allow for efficient handling of global and local information, addressing the limitations of previous models that focused either too much on global or local aspects.

### Weaknesses
Although the hyperparameters are set based on certain considerations, a more in-depth exploration could be beneficial. For example, the impact of different values of m_r  and m_c  (the numbers of rows and columns for partitioning in RALA) on the model's performance for different TSP sizes and distributions could be further investigated. 

The comparison with traditional heuristics is limited. While Random Insertion is used as a baseline in some experiments, other well-known heuristics (e.g., Lin-Kernighan, 2-opt) could be included in a more comprehensive comparison. This would provide a better understanding of how GELD compares to traditional methods in terms of solution quality and computational efficiency.

### Questions
1） In the two-stage training strategy, why was the maximum training size n_max set to 1000? What was the rationale behind this choice, and how did you explore other possible values?

2) The Region-Average Linear Attention (RALA) mechanism is a key innovation in the Global-view Encoder (GE). However, it is not clear how the choice of partitioning the nodes into regions (controlled by m_r and m_c) affects the balance between capturing global and local information. Can you provide more insights into this?

3) The Local-view Decoder (LD) restricts the candidate set to the k-nearest neighbors. How was the value of k determined, and what is the sensitivity of the model's performance to changes in k?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents ``GELD``, a neural TSP solver that efficiently handles TSP of various scales and distributions. It features a global encoder and local decoder with a novel attention mechanism, and a two-stage training strategy. The author claims that ``GELD`` outperforms state-of-the-art models in solution quality and speed, offering a promising approach for solving TSP.

### Strengths
* The author propose a novel Region-Average Linear Attention (RALA) mechanism within GE which operates with $O(n)$ time-space complexity.
* Using the ``PRC`` post-processing method, ``GELD`` first attempt to solve ultra-large-scale real-world TSP problems.

### Weaknesses
* **Weak experimental quantity**: Obviously, as an auto-regressive model, conducting experiments only on TSP is far from enough. I believe, as the author mentioned in the summary section, at least another type of CO problem, such as ``CVRP`` or ``JSSP``, needs to be done.
* **Low novelty**: The proposed ``PRC`` is conceptually similar to ``LCP`` in ``GLOP`` . Methodologically, especially in solving large-scale TSP problems, the two are almost identical, both relying on iterative reconstruction based on random greedy insertion.
* **Poor experimental setup and control**: Firstly, as a well-researched problem, TSP has many publicly available datasets. However, the author didn't test their approach on these datasets, which indirectly prevents comparisons with some of the current state-of-the-art TSP neural solvers, such as ``DIFUSCO`` and ``T2T``.  Secondly, when referencing ``BQ-NCO``, only their greedy approach was cited, not their beam-search, while ``GELD`` uses beam-search.

### Questions
* Please explain the three points mentioned in the ``Weaknesses`` section.
* Traditional solvers like ``LKH`` have already performed exceptionally well on TSP problems, and works like ``NeuroLKH`` further demonstrate that ``LKH`` can achieve high-quality solutions for large-scale TSP instances in very short times. So what are the advantages of Neural-TSP-Solvers like the one in this paper?
* Question regarding GELD's experimental data: I retested the uniform data test set locally using ``LKH``, but the results I obtained differ significantly from those in the repository. Below are my results. It should be noted that I used 20 threads for the solution, so I also divided the reported time in the paper by 20 accordingly.

|           |  Author Settings   | Author Results | Reviewer Settings | Reviewer Results |
|:---------:|:------------------:|:--------------:|:-----------------:|:----------------:|
|  TSP-100  | 20K iters, 10 runs |   7.8693, 8s   | 500 iters, 1 runs |    7.8694, 6s    |
|  TSP-500  | 20K iters, 10 runs | 16.5601, 666s  | 500 iters, 1 runs |   16.5532, 47s   |
| TSP-1000  | 20K iters, 10 runs | 23.2215, 2736s | 500 iters, 1 runs |   23.1560, 74s   |
| TSP-5000  | 20K iters, 1 runs  | 50.9830, 306s  | 500 iters, 1 runs |   50.9860, 72s   |
| TSP-10000 | 20K iters, 1 runs  | 73.1436, 5616s | 500 iters, 1 runs |  71.8154, 261s   |

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Authors propose a neural solver GELD which integrates a lightweight Global-view Encoder with a heavyweight Local-view Decoder to enrich embedding representation
while accelerating the decision-making process. Experimental results show that GELD is able to significantly improve the  solution quality of both small-scale and large-scale TSP instances with affordable computing time.

### Strengths
1. This paper is easy to follow.
2. The Region-Average Linear Attention Structure seems interesting and could be applied in other applications.

### Weaknesses
1. The motivation is not clear. Authors mention that Divide&Conquer may fail to provide valuable insights for solving other COPs such as JSSP, but the results of other problems are missing.
2. The contribution 1 is not rigorous. Heatmap-based methods such as Att-GCN and D&R methods Select and Optimize[1] also use the same set of pre-trained model parameters for all the  TSP scales.
3. Some important points are missing. GELD seems to be an iterative framework, and the local part of the solution needs to be reconstructed for each iteration. The detailed illustration of the working process is missing.
4. The comparison methods in experiments are not complete. Heatmap-based approaches(Att-GCN[2], DIFUSCO[3], etc.) and the recent work ICAM[4] should be included. Furthermore, it seems unfair to combine GELD with different search strategies such as BS and PRC, while only the greedy search results are listed for other baselines

### Questions
See weekness.

Reference:

1. Hanni Cheng, Haosi Zheng, Ya Cong, Weihao Jiang, and Shiliang Pu. Select and optimize: Learning to aolve large-scale tsp instances. AISTATS2023.

2. Zhang-Hua Fu, Kai-Bin Qiu, and Hongyuan Zha. Generalize a small pre-trained model to arbitrarily
large tsp instances. AAAI2021.

3. Zhiqing Sun and Yiming Yang. DIFUSCO: Graph-based diffusion solvers for combinatorial optimization. NeurIPS2023.

4. Changliang Zhou, Xi Lin, Zhenkun Wang, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. Instanceconditioned adaptation for large-scale generalization of neural combinatorial optimization. arXiv preprint
arXiv:2405.01906, 2024

### Soundness
3

### Presentation
2

### Contribution
2
