# Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Neural combinatorial optimization (NCO) has made significant advances in applying deep learning techniques to efficiently and effectively solve single-objective flexible job shop scheduling problems (FJSPs). However, the more practical multi-objective FJSPs (MOFJSPs) remain underexplored, limiting the applicability of NCO in multi-criteria decision-making scenarios. In this paper, we propose a decomposition-based NCO method to solve MOFJSPs. We present the dual conditional attention network (DCAN), a neural network architecture that takes the objective preferences along with the problem instance, aiming to learn adaptable policies over the preferences. By decomposing an MOFJSP into a set of subproblems with different preferences, the learned DCAN policies generate a set of solutions that reflect the corresponding trade-offs. We customize the Proximal Policy Optimization algorithm based on decomposition to effectively train the policy network for multiple objectives and define the state and reward based on combinations of different objectives. Extensive results showcase that our approach outperforms traditional multi-objective optimization methods and generalizes well across diverse types of problem instances.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the flexible job shop scheduling problem with multiple objectives. The authors propose a Dual Conditional Attention Network that learns adaptable scheduling policies. They employ a decomposition-based neural combinatorial optimization approach, solving subproblems with different objective preferences, and use PPO for training. The learned policy can generate a set of Pareto-optimal solutions.

### Strengths
The paper tackles a practically important and realistic problem, as real-world scheduling problems typically involve multiple conflicting objectives.

The approach of learning adaptable policies conditioned on objective preferences is promising and could improve generalization.

Generating a Pareto set of solutions using a single trained model is an attractive feature for practical multi-objective scheduling.

### Weaknesses
The four considered objectives (makespan, total tardiness, average flow time, and total cost) are not well justified. The first three are closely correlated regular measures, while the inclusion of total cost lacks sufficient motivation or explanation regarding its relevance in real-world FJSPs.

The paper does not analyze the interrelationship among the four objectives, which limits understanding of the trade-offs the model learns.

While the paper presents an interesting approach, the overall contribution remains unclear. It would be helpful to explicitly highlight what new insight or technical novelty it brings beyond existing multi-objective scheduling and NCO frameworks.

The method uses the lower bound of completion time as the state and reward, but this can cause instability in learning depending on the tightness of the bound. This issue should be discussed in detail.

While several performance-enhancing techniques are proposed, the paper lacks an ablation study to quantify their individual contributions.

### Questions
1. How can practical constraints such as setup times or machine dedication be incorporated into your framework?
2. How does the model perform when including objectives that are not regular measures (e.g., earliness)?
3. Many recent studies employ REINFORCE for scheduling. What motivated your choice of PPO?
4. Using lower bounds as rewards can lead to inconsistent gradients depending on their tightness. How do you handle or mitigate this issue?
5. Please provide an ablation study to analyze the contribution of each proposed component.

### Soundness
2

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
3

### Summary
The paper presents two extensions to an existing neural network architecture for the FJSSP, namely the Dual Attention Network, enabling it to address multi-objective combinatorial optimization problems. The authors aim to approximate the standard Pareto front and identify the corresponding Pareto set.

The first proposed method is a straightforward extension in which the weighted preference vector of the objectives is concatenated with each operation and machine feature. The rest of the network remains unchanged from the single-objective formulation. The second method integrates the preference vector directly into the update process of the operation and machine embeddings during the attention mechanism.

The authors investigate four optimization objectives: makespan, tardiness, flow time, and cost. The two proposed extensions are evaluated against standard metaheuristic algorithms and a mathematical solver across several benchmark datasets. Additionally, the approach is extended to two related problem types, the classical JSSP and the FFSP (discussed in the appendix).

The results demonstrate that both neural network extensions achieve significantly faster computation times compared to metaheuristic algorithms and solver-based methods across all instance types. However, the metaheuristics outperform the proposed approaches on certain, smaller problem instances, while the advantage of the proposed methods becomes particularly evident for larger instances that are computationally challenging for traditional optimization techniques. It was also shown that the second proposed approach, in which the preference is integrated into the attention mechanism, performs better than the approach where the preference vector is trivially inserted.

The authors want to publish code after acceptance.

### Strengths
- Well-motivated problem formulation, clearly highlighting that neural network architectures have rarely been applied to multi-objective variants of the Flexible Job Shop Scheduling Problem (FJSSP).

- Introduction of two new methods that improve solution quality, particularly with respect to inference time: The first method, while not highly innovative, forms a solid baseline. The second method shows clear advancements beyond the baseline.

- Comprehensive evaluation setup: Both methods are tested using greedy and sampling-based inference strategies. This allows a thorough assessment under different inference regimes.

- Novel reward function design: Reward is defined based on the change in the theoretical minimum lower bound before and after an action for each metric. This differs from previous work, which typically uses dense or sparse rewards derived directly from the actual schedule rather than theoretical lower bounds.

- Well-structured and informative appendix: Contains mathematical assumptions, details on key performance metrics, description of the critic network architecture (PPO), and a step-by-step reward calculation.

- Transparent discussion of limitations: Authors acknowledge where their methods underperform on small instances. They provide a convincing explanation why their approach performs better on larger instances.

- Demonstrates understanding of the problem characteristics and adds credibility to the evaluation.

### Weaknesses
- Although the authors acknowledge that other deep learning solutions typically train separate models for individual preference vectors, no such method is included in the experiments. Including at least one representative baseline (e.g., policy-based or value-based DRL approaches from recent literature) would provide meaningful context and allow readers to better assess the relative performance and contribution of the proposed methods.

- The description of how the preference vector is incorporated into the second method (architecture with dual inputs) remains unclear. A visual schematic, for example a processing pipeline similar to those used in DAN literature, would help illustrate how the model processes and uses preference information.

- While the reward formulation (based on changes in theoretical lower bounds) is novel, the paper does not empirically justify why this reward is advantageous over more traditional dense or sparse scheduling rewards. A small ablation experiment (for example in the Appendix) comparing the proposed reward against a common baseline reward would strengthen the motivation and demonstrate its impact on learning behavior and performance.

### Questions
1. Since the study introduces four distinct optimization objectives, why was the method not evaluated on all four objectives simultaneously? The experiments only include comparisons for selected two-objective and three-objective cases. Was this limitation due to the constraints of the metaheuristic algorithms and the solver, or does it reflect a limitation of the proposed model itself?
2. In Table 3, it would be helpful to see results for additional sampling iterations. Given that the proposed method can generate substantially more samples within the same computational time as the metaheuristic approaches, it is possible that further sampling could enable it to outperform the metaheuristics. Could you clarify why this comparison was not included?
3. For the ablation studies, why was only NSGA-II considered and not the newer NSGA-III? The latter represents a more state-of-the-art approach for multi-objective optimization.
4. The focus of current research in multi-objective scheduling is shifting toward real-world applications, where objectives are defined for more practical use cases such as energy consumption and energy cost. Why were such abstract objectives chosen in this study? No citations were provided to justify the selection of the investigated objectives.
5. There already exist methods that apply deep learning to multi-objective scheduling. Why were these methods ignored? At least some ablation studies, for example on small instances, should have been included as a comparison against the proposed method.
6. The IGD+ metric was mentioned and used in the Appendix to compare different approaches. Why was it not included in all the tables? Since it represents an alternative to the hypervolume metric, wouldn’t it be reasonable to include both for completeness?
7. Both methods do not appear to be overly complex, and it seems straightforward to combine them into a single approach—for instance, DCAN with the feature space of WI-CAN. Why was this not investigated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a decomposition-based learning method to solve multi-objective flexible job shop scheduling (FJSP), where each subproblem associated with a preference vector. They define two neural architectures based on the DAN architecture from the previous literature for single-objective FSJP. WI-DAN concatenates the preference vector to the feature vector whereas DCAN leverages dual attention with conditional operation message attention block and conditional machine message attention block. The authors evaluate the performance of their proposed method on a variety of FSJP benchmarks, comparing with two commonly used multi-objective evolutionary algorithms, to show effectiveness.

### Strengths
1. The paper is easy to read and the proposed method makes sense for multi-objective optimization.

2. Empirical results seem promising for the given method.

### Weaknesses
1. There have been many works on learning for multi-objective combinatorial optimization for other COPs (e.g. routing). This paper does not give sufficient discussion of these related works (e.g. the related work section only mentions a few works for multi-objective FJSP, but not for other COPs at all). Similarly, there’s no discussion on (1) whether the learning method used in this paper has been applied to other COPs, and (2) whether techniques from other COPs can apply to FJSP considered in this paper. The authors should rewrite the related works and discussions to better connect to existing literature. 

2. In general, I’m concerned about the novelty of this paper. The decomposition based PPO algorithm seems to be a standard RL algorithm for multi-objective learning, and the neural architecture design seems to be a straightforward extension from the DAN architecture in the previous work. Given this, I’m worried that this paper does not meet the bar for a high quality ML conference like ICLR.

### Questions
See the weaknesses. And further, based on the applicability of multi-objective learning methods for other COPs, the author should consider comparing with those applicable learning methods. Currently, I feel like the number of baselines the author compared is too few. And further, I think it will strengthen the paper if the authors can try to apply their proposed method to more scheduling variants beyond FJSP.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to use DRL for the multi objective FJSSP. It compares two different neural architectures and also compares to standard MOCO algorithms such as NSGAII. For large instances the algorithm that samples using the trained neural networks has good results. The conditional attention network outperforms the preference vector input network.

### Strengths
Good experimental results for large instances
Comparison of two neural architectures
Design of the DCAN architecture
Better results than meta-heuristic approaches

### Weaknesses
Simple sampling strategy
NSGA-II has better result on public dataset instances for the 3-objective problem

### Questions
How does the results of best network evolve with samples?

### Soundness
3

### Presentation
3

### Contribution
3
