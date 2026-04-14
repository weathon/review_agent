# Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 8

## Abstract
Generative Flow Networks (GFlowNets), a new family of probabilistic samplers, have demonstrated remarkable capabilities to generate diverse sets of high-reward candidates, in contrast to standard return maximization approaches (e.g., reinforcement learning) which often converge to a single optimal solution. Recent works have focused on developing goal-conditioned GFlowNets, which aim to train a single GFlowNet capable of achieving different outcomes as the task specifies. However, training such models is challenging due to extremely sparse rewards, particularly in high-dimensional problems. Moreover, previous methods suffer from the limited coverage of explored trajectories during training, which presents more pronounced challenges when only offline data is available. In this work, we propose a novel method called \textbf{R}etrospective \textbf{B}ackward \textbf{S}ynthesis (\textbf{RBS}) to address these critical problems. Specifically, RBS synthesizes new backward trajectories in goal-conditioned GFlowNets to enrich training trajectories with enhanced quality and diversity, thereby introducing copious learnable signals for effectively tackling the sparse reward problem. Extensive empirical results show that our method improves sample efficiency by a large margin and outperforms strong baselines on various standard evaluation benchmarks. Our codes are available at \url{https://github.com/tinnerhrhe/Goal-Conditioned-GFN}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the challenge of training goal-conditioned Generative Flow Networks (GC-GFlowNets) in environments with sparse rewards and limited offline data. It introduces Retrospective Backward Synthesis (RBS), which synthesizes new backward trajectories to enrich training data, improving sample efficiency and diversity. Experiments demonstrate that RBS significantly improves performance and generalization in various benchmarks.

### Strengths
This paper proposes a novel method called Retrospective Backward Synthesis (RBS), which synthesizes new backward trajectories in goal-conditioned GFlowNets to improve the quality and diversity of training trajectories. This approach introduces rich learnable signals, effectively addressing the sparse reward problem.

### Weaknesses
1.The experimental tasks are relatively simple and insufficiently comprehensive.

2.The latest goal-conditioned reinforcement learning algorithms are not selected for comparison.

### Questions
1.	Age-Based Sampling is a very straightforward technique. How does it compare to previous methods like Prioritized Experience Replay (PER)? Did the authors attempt using PER as well?

2.	Regarding the experimental setup, since you’ve compared your method with reinforcement learning approaches, I assume that these experiments share the same tasks as those in reinforcement learning. If that’s the case, why weren’t newer RL methods selected as baselines? Additionally, for tasks like bit sequence generation, TF binding generation, and AMP generation, is DQN an appropriate baseline?

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
3

### Summary
This paper addresses key challenges in goal-conditioned Generative Flow Networks (GFlowNets), specifically the problems of sparse rewards and limited trajectory coverage. The authors introduce Retrospective Backward Synthesis (RBS), a method that generates additional backward trajectories to expand the training data. Their approach aims to improve both the quality and diversity of training trajectories, providing more learning signals in scenarios with sparse rewards. Empirical evaluations demonstrate improved sample efficiency and performance compared to baseline methods across multiple benchmarks.

### Strengths
* The paper is well-written and straightforward to understand.
* Retrospective Backward Synthesis (RBS) is introduced with clear motivation, and the paper also presents training techniques such as backward policy regularization. 
* Empirical results demonstrate that the proposed method outperforms baselines, showing improved performance and sample efficiency.

### Weaknesses
* The evaluation tasks do not include key benchmarks like RNA Generation from Pan et al. (2023a), which limits direct comparison.
* The differences between the proposed RBS method and OC-GAFN are not clearly articulated. A more comprehensive discussion is needed to clarify the specific advantages of the RBS method. 
* It remains unclear how goals are defined across the evaluated tasks, which could impact generalizability and reproducibility.

### Questions
Please address my concerns in the weakness

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
3

### Summary
The paper proposes Retrospective Backward Synthesis (RBS), a novel method to enhance the training of goal-conditioned Generative Flow Networks (GC-GFlowNets). GC-GFlowNets have shown potential in generating diverse sets of high-reward candidates but face challenges due to sparse reward structures and limited coverage of explored trajectories, especially when using offline data. To address these limitations, RBS synthesizes backward trajectories that originate from a desired goal, enriching the training data with high-quality, diverse samples. This approach helps transform unsuccessful action sequences into positive learning experiences, thereby improving sample efficiency and generalizability.

The authors introduce additional techniques, such as reward signal intensification and backward policy regularization, to stabilize training and prevent mode collapse. Empirical results across various benchmarks, including GridWorld and bit sequence generation, demonstrate that RBS outperforms state-of-the-art methods in terms of success rates, sample efficiency, and scalability. Notably, RBS achieves nearly 100% success in large-scale tasks where competing approaches fail, highlighting its robustness and potential for further advancements in GC-GFlowNets.

### Strengths
- Backward-Looking Strategy for Enhanced Training: The proposed Retrospective Backward Synthesis (RBS) method utilizes a backward-looking strategy to synthesize trajectories from the goal state, significantly enriching training data. This approach effectively improves sample efficiency by converting failed experiences into successful learning signals, addressing the sparse reward problem.
- Empirical Validation of Sample Efficiency: The paper presents strong empirical results across a range of benchmarks, demonstrating that RBS markedly improves sample efficiency. The method achieves nearly 100% success rates in complex tasks where state-of-the-art baselines fall short, underscoring its practical impact.
- Clear Writing and Presentation: The paper is well-written and presented, with clear explanations, structured methodology, and comprehensive experimental results. The clarity facilitates a strong understanding of both the theoretical and practical aspects of the proposed approach.

### Weaknesses
- Scalability and Continuous Environments: The paper’s experiments focus on relatively simple and discrete environments, raising concerns about how well the Retrospective Backward Synthesis (RBS) method would scale to more complex, continuous, real-world tasks. The absence of testing in high-dimensional or continuous state-action spaces limits insights into its broader applicability.
- Tuning Challenges: The method's reliance on hyperparameters, such as reward scaling and backward policy regularization, introduces tuning challenges. While these components are beneficial for stabilizing training, they require careful adjustment, potentially impacting the ease of replication and practical deployment in varied scenarios.
- Lack of Comparison with Model-Based RL: Despite the inherent use of backward trajectory synthesis, which resembles model-based planning, the paper does not compare RBS with established model-based RL approaches such as MBPO or Dreamer. This omission makes it difficult to assess how RBS performs relative to other methods that also utilize environment models for planning and sample efficiency.

### Questions
- In algorithm 1) line 6, how do we guarantee that the backward policy could reach $s_0$ from $y$ each time?
- Are tuning for reward intensification and backward policy regularization difficult? What the the effect of hyper-parameters on the performance?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduces a novel method, Retrospective Backward Synthesis (RBS), aimed at enhancing the training of goal-conditioned Generative Flow Networks (GFlowNets) by synthesizing new backward trajectories. RBS augments "virtual" backward trajectories in
goal-conditioned GFlowNets to enrich training trajectories with enhanced quality and diversity. RBS improves the sample efficiency and performance of GFlowNets across a range of tasks, including sequence generation and biological sequence design.

### Strengths
1. The paper identifies and targets a significant issue in the training of goal-conditioned GFlowNets, offering a practical and innovative solution. Augmenting backward trajectories for training is interesting.
2. Comprehensive empirical results are provided, demonstrating the effectiveness of RBS over existing methods on multiple benchmarks.

### Weaknesses
1. Limited Discussion on Potential Drawbacks: The paper does not sufficiently address the potential limitations of RBS. For instance, there is no discussion about the computational overhead of synthesizing backward trajectories, nor is there a mention of whether the method is robust to different types of reward structures or environment dynamics.
2. Relevance and Scope of Application: The improvements are made specifically within the context of GC-GFlowNets, which may limit the applicability of the method.
3. Comparison with Diffusion Policies: Given the similarities between the proposed RBS and diffusion policies, a direct comparison would be valuable to understand the unique contributions and differences of RBS.
4. Assumptions on Environment Dynamics: It is unclear whether the proposed RBS method assumes or requires any particular properties of the environment, such as determinism or stochasticity. If the backward dynamics are infeasible or the environment is highly stochastic, the performance of RBS may be affected, and this should be addressed.
5. Quality of Synthetic Trajectories: The paper should include a discussion on how to ensure the quality of the synthesized backward trajectories, especially in cases where such trajectories may not correspond to realistic or feasible paths in the actual environment.
6. Lack of Comparison with Goal-Conditioned RL: Without a comparison to goal-conditioned RL, it is difficult for readers to fully appreciate the relative strengths and weaknesses of GC-GFlowNets. Including such a comparison would provide a more complete picture of the method's positioning within the broader field of goal-directed learning.
7. The authors may further investigate existing literature on augmenting backward trajectories for sample-efficient RL or backward learning in goal-conditioned RL, which makes the paper more comprehensive.

### Questions
1. How does RBS compare with diffusion policies, and in what scenarios does RBS offer distinct advantages?
2. Does RBS assume deterministic or stochastic environments, and how does it handle situations where the backward dynamics are not straightforward?
3. How can the authors ensure that the synthesized backward trajectories are meaningful and do not lead to false positives in the learning process?
4. Could the authors include a comparison with goal-conditioned RL methods to highlight the specific benefits of using GC-GFlowNets?

### Soundness
3

### Presentation
3

### Contribution
3
