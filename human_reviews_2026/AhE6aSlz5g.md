# Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 4

## Abstract
Multi-task Vehicle Routing Problems (VRPs) aim to minimize routing costs while satisfying diverse constraints. Existing solvers typically adopt a unified reinforcement learning (RL) framework to learn generalizable patterns across tasks. However, they often overlook the constraint and node dynamics during the decision process, making the model fail to accurately react to the current context. To address this limitation, we propose Chain-of-Context Learning (CCL), a novel framework that progressively captures the evolving context to guide fine-grained node adaptation. Specifically, CCL constructs step-wise contextual information via a Relevance-Guided Context Reformulation (RGCR) module, which adaptively prioritizes salient constraints. This context then guides node updates through a Trajectory-Shared Node Re-embedding (TSNR) module, which aggregates shared node features from all trajectories' contexts and uses them to update inputs for the next step. By modeling evolving preferences of the RL agent, CCL captures step-by-step dependencies in sequential decision-making. We evaluate CCL on 48 diverse VRP variants, including 16 in-distribution and 32 out-of-distribution (with unseen constraints) tasks. Experimental results show that CCL performs favorably against the state-of-the-art baselines, achieving the best performance on all in-distribution tasks and the majority of out-of-distribution tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Chain-of-Context Learning (CCL), a new neural framework for multi-task Vehicle Routing Problems (VRPs) that aims to improve constraint awareness and dynamic adaptation during solution construction. CCL combines two novel modules: Relevance-Guided Context Reformulation (RGCR), which adaptively highlights salient constraints for the current state, and Trajectory-Shared Node Re-embedding (TSNR), which stepwise updates node embeddings based on historical context and multi-trajectory exploration. The model achieves better results on multi-task VRP benchmarks of 48 variants.

### Strengths
1. This paper is clearly written.
2. CCL achieves state-of-the-art results on all in-distribution tasks and most out-of-distribution tasks, with performance improvements that are robust to hyperparameter and random seed variations, as shown in standard deviation analyses.
3. The ablation study is thorough.

### Weaknesses
1. **My main concern with this paper is that there is no theoretical basis for your design.**


Some designs are not reasonable, violating some common recognition. You mentioned in the Introduction that "They fail to leverage information accumulated in previous decoding steps, thereby limiting the model’s ability to capture and preserve historical preferences, which is an essential factor for coherent sequential decision-making." However, usually we define the RL process on the Markov Process, but your statement violates the Markov property. If you intend to include all past information in the state space, this would undoubtedly significantly increase the learning complexity, but this phenomenon is not observed in the training curves. 

Specifically, I do not get the importance of providing the whole change trace of constraints as you mentioned. Your example "For example, if a customer’s time window is about to be violated, the model should prioritize handling that node immediately.", why can the customer feel the emergency with only existing scaler-based methods? They will probably make a wise decision based on the low number. 

We know that, according to the Markov property, we can simplify this problem by partially solving it. According to you, such simplification (losing information about historical constraints) is actually detrimental to decision-making, which clearly contradicts the facts.

In conclusion, I believe the motivation in this paper is somehow flawed, and the authors should rethink their interpretation of their empirical findings.

### Questions
Given that many of your settings are flawed yet yield effective results, I would appreciate it if you could provide a reasonable explanation. If you can articulate the strengths in a reasonable manner, I am willing to increase the rating.

### Soundness
2

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
4

### Summary
This paper addresses the challenge of solving multi-task VRPs, where existing neural solvers often fail to react to dynamic, step-wise changes in constraints and node states. The authors propose Chain-of-Context Learning (CCL), a novel framework that enhances the decoding process. CCL introduces two key modules: (1) a Relevance-Guided Context Reformulation (RGCR) module, which constructs a step-wise context by adaptively prioritizing salient constraint information, and (2) a Trajectory-Shared Node Re-embedding (TSNR) module, which efficiently updates node representations based on this dynamic context, capturing sequential dependencies without full re-computation. The authors evaluate CCL on a comprehensive benchmark of 48 VRP variants, demonstrating SOTA performance on all 16 in-distribution tasks and 32 out-of-distribution (OOD) tasks.

### Strengths
1. The core contribution, explicitly modeling step-wise constraint dynamics within the decoder, is a novel and intuitive approach.
2. The paper is well-written, and its structure is logical.
3. The paper's empirical evaluation is thorough and rigorous. The use of a large benchmark (48 variants)  provides strong evidence for the method's generalization capabilities. The ablation studies are comprehensive.

### Weaknesses
1. In the Introduction (lines 56–57), the paper claims that existing methods fail to consider node priority. Could the authors provide illustrative examples or visualizations to demonstrate this point and further clarify how the proposed method captures node priority more effectively?

2. Works such as Jieyi Bi’s PIP and others have also addressed constraint-aware learning, though often in single-task settings like TSPTW. It would be valuable to distinguish the proposed method from these studies and discuss whether such single-task methods could be adapted to multi-task problems.

3. The rationale for the superiority of RGCR in integrating constraints into the embedding space is unclear. Incorporating constraints to improve performance is intuitively reasonable, but the paper does not explain why the proposed approach achieves better integration. A more intuitive explanation or analysis would be helpful.

### Questions
In Table 3, the ablation studies are conducted based on ReLD (CCL$^\dagger$}). I would like to know whether the effectiveness of the proposed model depends on the ReLD module.

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
This paper proposes Chain-of-Context Learning (CCL), a novel reinforcement learning framework for solving multi-task Vehicle Routing Problems (VRPs). CCL introduces two key modules:  Relevance-Guided Context Reformulation (RGCR), which dynamically prioritizes constraints at each decoding step. Trajectory-Shared Node Re-embedding (TSNR), which efficiently refines node embeddings across multiple trajectories using shared context. The method is evaluated on 48 VRP variants (16 in-distribution, 32 out-of-distribution), demonstrating state-of-the-art performance on both seen and unseen tasks, particularly those involving complex constraints like time windows.

### Strengths
**Originality**: The idea of step-wise, context-aware node re-embedding is novel in the multi-task VRP setting. Unlike prior methods that use static embeddings, CCL captures evolving constraint priorities and node states, addressing a clear gap in the literature.

**Quality**: The paper is technically sound, with well-designed modules (RGCR and TSNR) and thorough experiments. Ablation studies and complexity analyses validate the design choices.

### Weaknesses
**Weaknesses**

1. **Methodological Complexity and Limited Generalizability**: The proposed CCL framework introduces significant architectural complexity through its two specialized modules (RGCR and TSNR). While effective for multi-task VRPs, the approach appears highly tailored to this specific problem domain. The paper would benefit from discussing how these components might generalize to other combinatorial optimization problems beyond VRPs, or what adaptations would be necessary for broader applicability.
2. **Narrow Technical Contribution Focus**: The primary innovation appears concentrated on neural architecture design rather than broader methodological contributions. The work introduces novel network components but doesn't substantially advance the underlying reinforcement learning paradigm, problem formulations, or theoretical foundations. A more balanced contribution spanning architectural, algorithmic, and theoretical dimensions would strengthen the paper's impact.
3. **Insufficient Formalization of RL Framework**: The reinforcement learning formulation lacks rigorous mathematical specification. The paper would be significantly strengthened by explicitly defining the core Markov Decision Process (MDP) components.
4. **Presentation Issues in Figures**:
    - **Figure 1**: The loss term contains unclear symbols or potential encoding issues.
    - **Figure 2**: There is a typo in the label "Bais" which should be corrected to "Bias".

### Questions
1. The update probabilities $P_{tr}$ and $P_{ts}$ are set empirically. Have the authors considered a learned or adaptive scheduling mechanism for these parameters?
2. How does CCL perform on highly constrained or infeasible instances? Is there a mechanism to handle or recover from constraint violations during decoding?
3. The paper focuses on Euclidean VRPs. Can CCL be extended to non-Euclidean or graph-structured routing problems? What modifications would be required?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the niche problem of multi-task vehicle routing optimisation, where a vehicle must deliver goods from a depot to customers arranged variably in 2D space as a graph of nodes. The authors introduce a method to aggregate the context (encoding information from multiple constraints and node features) while incorporating step-wise relevance of each constraint using its similarity to node features. They further add a module to incorporate sequential/temporal information from multiple trajectories and other nodes using multi-head attention. Contexts/features are encoded and actions are decoded using an encoder-decoder architecture. Results over a range of VRP tasks show slight improvements over state-of-the-art baselines in terms of performance, but with increased inference time.

### Strengths
* The methodology and diagram are clear
* Fairly many tasks (comparative and ablative) are tested on
* The results discussion and analysis are detailed

### Weaknesses
* The motivation for this architecture is not very clear to me. There have already been works that incorporate context from other nodes (Kool et al 2019) or use sequential information (Nazari et al 2018)
* The architecture consists of several components which are not novel in this space. At the same time, they are combined in a complicated way and I don’t understand the need for this complication (e.g. combining constraint embeddings with node features many times over in different ways in RGCR)
* The results show extremely incremental improvements in performance, to the point where it’s hard to tell if they came from the architecture or just the size of the networks etc.
* The degradations in inference time, by contrast, are significant
* The method only applies to a single agent (let me know if I’m wrong here as this wasn’t completely clear), unlike the SOTA baseline MTPOMO
* Minor: ‘bias’ is misspelled ‘bais’ in the architecture diagram and the bar chart

### Questions
* I don’t understand how the action selection is RL? It seems to be a form of offline return (sum of future rewards) maximisation but there is no value iteration or policy gradient

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Chain-of-Context Learning (CCL), a novel framework designed to enhance decision-making in multi-task Vehicle Routing Problems (VRPs) by dynamically adapting to evolving constraints. Traditional heuristic methods for VRPs are computationally intensive and inflexible, while existing neural approaches rely on static embeddings that fail to capture changing constraint priorities. CCL overcomes these challenges through two key modules: the Relevance-Guided Context Reformulation (RGCR) module, which dynamically prioritizes salient constraints, and the Trajectory-Shared Node Re-embedding (TSNR) module, which refines node representations by aggregating contextual information from multiple trajectories. Using a transformer-based encoder and reinforcement learning, CCL jointly models node and constraint dynamics to make adaptive routing decisions. Extensive experiments on 48 VRP variants show that CCL consistently outperforms state-of-the-art baselines, including CaDA, achieving superior routing efficiency and generalization to out-of-distribution tasks. Ablation studies confirm the complementary effectiveness of RGCR and TSNR, while complexity analysis shows that the slight increase in inference time is justified by significant performance gains. Furthermore, CCL demonstrates robust performance on real-world VRPTW instances, achieving the lowest average performance gap among all tested methods.

### Strengths
1. The problem addressed in this work is of high significance in the field of operations research. It deals with a complex and practically relevant challenge that aligns with current trends in optimization and decision-making under uncertainty. 
	
2. The authors have provided a thorough and well-balanced review of the existing literature related to the problem under study. The cited works effectively capture both the foundational research and recent advancements in the area, demonstrating a clear understanding of the current state of the art. The discussion successfully positions the proposed work within the broader research landscape, highlighting how it builds upon and differentiates itself from prior studies.
	
3. The manuscript is well organized and presented in a logical, easy-to-follow manner. The problem statement, methodology, and experimental results are clearly delineated, making it straightforward for the reader to follow the development of ideas. The figures, tables, and explanations are used effectively to support the narrative and to communicate key insights.
	
4. The authors have conducted a comprehensive experimental analysis that strengthens the credibility of their claims. The inclusion of evaluations on out-of-distribution scenarios demonstrates the robustness and generalizability of the proposed approach. Furthermore, the trade-off analysis and ablation studies provide valuable insights into the contribution of individual components and the practical implications of design choices.

### Weaknesses
1. The problem considered in this work is not a notoriously difficult problem to solve with existing learning and non-learning-based methods. Considering the complexities and the size of the problem, optimal solutions can be obtained by formulating the problem as Integer Linear Programming, and using commercial solvers given enough computing time. Since the problem is deterministic, computing time is not a limitation unless the problem size is significantly large. Therefore,  I would encourage the authors to provide more justification on why a learning-based method should be used for a small-scale deterministic combinatorial optimization problem.

2. The baselines used for comparison are not strong. Provided that the problem along with the six constraints do not invoke any dynamically generated tasks or uncertainties, there are many traditional methods including Integer Linear Programing (ILP) based methods to compare. I would also encourage the authors to use the LKH solver to compare the performance of their proposed method. These centralized traditional methods usually provide optimal solutions compared to learning-based method. For deterministic problems like this, where optimal solution can be pre-computed, the use of learning-based approaches is not justified. I would also encourage the authors to provide a short (1-2 sentence) description about the baselines used, including any parameter/configuration settings.

3. The manuscript lacks a formal problem formulation. In order to apply Reinforcement Learning, the problem has to be formulate as a Markov Decision Process, explaining the state space, action space, and the reward. Having the problem formulation gives the reader a better understanding of  how/why the proposed decision-making architecture works.

4. I would encourage the authors to use statistical tests such as T-test or ANOVA to compare their method against other baselines. The statistical tests can provide evidence if the difference in mean is significant.

### Questions
1. In equation 1, shouldn't the dimensions of H be (N+2) x D?
	
2. What is N&L, and C&L in figure 1?

3. What does "Multiple Trajectories" mean in figure 1?

4. What does "multi-trajectory context" mean?

5. What is the dimension of the concatenated vector in equation 5?

6. What does "trajectory" refers to in the context of the manuscript? Is it the same as a sub-route?
The linehaul and backhaul demands are a node-specific quantity. It is not clear why they are expressed using the notations for trajectory an decoding step (i and j).

### Soundness
2

### Presentation
3

### Contribution
2
