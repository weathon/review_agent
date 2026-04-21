# Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization

- Avg Score: 5.75
- Decision: Reject
- Scores: 5, 8, 5, 5

## Abstract
Deep reinforcement learning (DRL) has been widely used for dynamic algorithm configuration, especially for evolutionary algorithms, which benefit from adaptive update of parameters during the algorithmic execution. However, applying DRL to algorithm configuration for multi-objective combinatorial optimization (MOCO) problems remains relatively unexplored. This paper presents a novel graph neural network (GNN) based DRL to configure multi-objective evolutionary algorithms. We model the dynamic algorithm configuration as a Markov decision process, representing the convergence of solutions in the objective space by a graph, with their embeddings learned by a GNN to enhance the state representation. Experiments on diverse MOCO challenges indicate that our method outperforms traditional and DRL-based algorithm configuration methods in terms of efficacy and adaptability. It also exhibits advantageous generalizability across objective types and problem sizes, and prospective applicability to different evolutionary algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a DRL-based algorithm configuration framework (GS-MODAC) for addressing multi-objective COPs. The authors propose using GNN as a feature extraction mechanism for capturing the optimisation dynamics. They then formulate the optimisation process of COPs as MDP, where the state is the GNN-based pareto feature, the action is cossing the hyper-parameter values for the multi-objective algorithm, and the reward is a tailor one with capability of assigning more credits for convergence situations to reinforce the learning. The experiments are facilitated to validate the proposed method on several representative COPs, including the in-distribution generalization and out-of-distribution generalization. Under the given experimental settings, GS-MODAC achieves competitive results compared to several baselines, including SMAC and MADAC,

### Strengths
Overall, the writing is reader-friendly.

### Weaknesses
Several major weaknesses hinder this paper to get the acceptance boardline:

a) Limited review on related works: In the last two years, many related works that leverage RL/DRL for algorithm configuration in EAs or SI algorithms are proposed. However, this paper only reviews the related works before 2022. Even for papers before 2022, there are already realted works investigating how to configure multi-obejctive optimizers through learning system, for instance: 

[1] Ning W et al. (2018). https://link.springer.com/article/10.1007/s13748-018-0155-7.

[2] Huang et al. (2020).  https://www.sciencedirect.com/science/article/abs/pii/S1568494620306311.

[3] Tian et al.  (2023). https://ieeexplore.ieee.org/abstract/document/9712324.

there are more. 

b)  Limited novelty: The paradigm in this paper show certain similarity with these related works despite the tartget optmisation problems and the GNN-based feature extraction. However this novelty is also not 'novel' enough considering a recent NuerIPS paper (Chen et al (2023). https://arxiv.org/abs/2310.15195) , where graph attention is used for autonomous pareto feature extraction. If the authors claim this novelty, an in-depth analysis should be carried out to compare at least the above usage of graph information.

c)  Unclear elaboration of methodology: what exactly is the node embedding and the final state representation?

d)  Overpromise in adaptability: the authors claim that: 'prospective applicability to different evolutionary algorithms'. However, the action spaces in different multi-objective optimizers can vary. This means the proposed network structure is not generic to handle this. More importantly, the authors claim than: 'We trained GS-MODAC for each problem configuration with randomly generated
problem-instance sizes'. I am confused here since an advantage obervaed in recent learning-assisted EC works is that the policy could be trained on all problem configurations to achieve maximum generalization. Can a GS-MODAC model trained on FJSP 5j5m generalize well on other configurations? Can GS-MODAC be trained on all isntances of all configurations and then generalize well?  

e) Limited in-depth analysis: the experimental results revolve around the performance, this leads to the limited interpretability of this work. For instance, in RQ 4, can authors explain why GS-MODAC can transfer the knowledge learned on C/D objectives to A/B objectivess. Is this due to the similarity among the objectives? More importantly, what has GS-MODAC learned? I suggest two improvements on this point: i) visualization of the learned graph scores to analyse the learned pareto pattern. ii) analyse the relationship between different pareto patterns and the action distributions controlled by GS-MODAC.  

f) Minors: when we talk about the algorithms in EC domain, evolutionary algorithms (EAs) include GA, DE, ES etc., swarm intelligence algorithms (SI) include PSO, ACO etc. Such definition should be made clear. (NSGAII->EAs, MOPSO->SI)

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper describes a GNN based DRL approach to configurating multi-objective evolutionary algorithms for combinatorial optisation problems. Flexible job shop scheduling and capitalised Vehicle Routing problems are considered in the paper. The results show that the proposed method achieved better results than existing methods and has wider adaptability.

### Strengths
1. the paper considered graph representation in neural networks in DRL, which is good for complex combinatorial optimisation.

2. The motivations and contributions are very clear and relatively convincing. 

3. The results and analyses are good.

### Weaknesses
1. A major weakness is that this paper only considered static/certain problems for FJSS and CVRP. It did not consider dynamic or uncertain problems, which are more difficult.

2. The shops and CVRP problems used are relatively small. For example, I have seen many FJSS problems use over 5000 jobs.

### Questions
Genetic programming and other OR methods are widely used to solve JSS and VRP problems. Why did not you consider comparisons with those commonly used methods in the paper?

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
In this work, the authors propose a DRL (Deep Reinforcement Learning)-based dynamic algorithm configuration method, GS-MODAC, to solve multi-objective combinatorial optimization (MOCO) problems. When modeling the parameter configuration problem as a MDP (Markov Decision Process), GS-MODAC employs graph neural networks (GNNs) to learn state representations, which distinguishes GS-MODAC from related studies. Additionally, a shared reward function is developed and it is applicable to different problem instances. The authors conduct a set of experiments on the variants of two MOCO problems, showing the effectiveness of the proposed algorithm and its generalizability.

### Strengths
1. The explanation of background and motivation is clear.
2. The writing is well-organized.
3. The authors conduct experiments from different perspectives to evaluate the performance of their algorithm.

### Weaknesses
1. Graph-based state representation is one of the main contributions in this work (as listed in the end of Section 1). In Section 3.1, the paragraph of states, the authors provide some explanations about how to construct the graph-based state representation and emphasize that this state configuration is effective. Further explanation is expected to clarify why this representation is more effective than alternatives.

2. How do states and rewards interact with the DRL agent? In Section 3, there is little information about the DRL agent in GS-MODAC. The authors can add a brief overview of how the DRL agent uses the state and reward information to learn and make decisions. Alternatively, the authors can add a brief introduction to DRL in Section 2.

3. In Section 3.2, many operations are described without details and explanations. For example, in line 266, what is the reason for introducing a global mean pooling operation to average the node embeddings? Similarly, the reason for including an additional feature vector and how this feature vector is incorporated into the state representation should be explained. The authors should provide brief justifications for each component design, and explain how each component contributes to the overall performance of GS-MODAC.
   
4. I cannot observe the results from Table2, as concluded by the authors on page 8, RQ2:  
    4.1. The authors claim that “the model trained on smaller instances and deployed on larger instances experienced a slight decline in performance but still managed to find better solutions compared to all the benchmarks provided.” What do the authors mean by “all the benchmarks”? Are they referring to the comparison algorithms? If so, as reported in Table 1, GS-MODAC that trained on Bi-CVRP - 100 customers performs worse than MADAC that trained on Bi-CVRP - 500 customers.  
    4.2. In addition, the authors claim that “the models trained on larger instances and deployed on smaller instances could improve results further”, but models trained on Bi-CVRP - 200 customers and deployed on Bi-CVRP - 100 customers found similar instead of better solutions than models directly trained on these smaller-sized instances. Moreover, models trained on Bi-CVRP - 500 customers and deployed on Bi-CVRP - 100 customers found worse solutions than models directly trained Bi-CVRP - 100 customers.  
The authors should clarify their claims with more specific evidence from Table 2.

5. In Table 3, only the baseline NSGA-II is compared, which is not very persuasive. To demonstrate effectiveness, I suggest adding other AC/DAC algorithms and the GS-MODAC directly trained on these two complex problems as comparisons.

6. In Table 6, the experiments are conducted on CVRP problems of sizes 20, 50, and 100. Why not keep the settings consistent with the previous NSGA-II experiments (100, 200, and 500)? Is there any specific reason for this choice? Furthermore, I suggest including other AC/DAC comparison algorithms in this set of experiments.

7. In Table 7, the most related work MADAC should be compared.

8. The names of problem instances are not consistent. some CVRP problems are denoted as Bi-CVRP-100 customers in Table 1, but are denoted as Bi-CVRP-100 nodes or Bi-CVRP-100 n in Table2.

### Questions
1. GS-MODAC requires an additional run of EAs with a high budget (e.g. doubled) to obtain $HV_{ideal}$. Is this realistic in real-world applications? How does a double-budget NSGA-II perform when compared with the proposed GS-MODAC ?

2. In Figure 2, why wasn't SMAC3 included in the comparison? According to the results reported in Table 1, SMAC3 outperforms NSGA-II on Tri-FJSP 10j5m.

3. In Table 9, why is the $HV mean$ not used as a performance indicator, just as reported in the previous tables?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a DRL method based on GNN, named GS-MODAC. The method models the algorithm configuration problem as a Markov decision process and leverages GNN to learn embeddings of solutions in the objective space, thereby enhancing state representation. However, there are some issues that need to be improved.

### Strengths
1. The idea is clear.
2. The authors provide experiments.

### Weaknesses
1. The current status of research lacks in-depth analysis, and the problems and key challenges that this research aims to address have not been accurately summarized.
2. In each iteration, the state is represented by embedding information from the current population solutions using a GNN, effectively capturing the structure and diversity of the solutions. However, GCN is transductive GNN, it has limited transferability to new graph structures. In other words, the GNN requires retraining on the graph structure with each iteration of the algorithm, which leads to significant computational costs. The authors should take this issue into account.
3.  As illustrated in Figure 1, each DRL iteration is essentially based on the transition between the current population state and the next iteration's population state. Specifically, the DRL strategy generates new parameter configurations based on the current state (i.e., the performance of the current population), which guide the optimization algorithm's search in the next iteration. This approach requires updating the DRL parameters at each iteration, which similarly incurs substantial computational costs.
4.  Considering comments 2 and 3, the computational complexity of this method is high. However, this paper does not include an analysis of the complexity.
5. The state configuration embedding method based on GCN is a key aspect of this research. However, Figure 1 does not effectively highlight the central role of GCN.
6. The figures provided in this paper, including the framework and experimental result, are not clear. Please provide high-resolution versions.
7. The conclusion lacks an in-depth analysis of the limitations of the this research and potential directions for future improvements. It is suggested that the authors include relevant content on these aspects.

### Questions
see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
