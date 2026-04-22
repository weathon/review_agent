# RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road.
While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive.
We propose RDAR, a strategy to learn per-agent relevance — how much each agent influences the behavior of the controlled vehicle — by identifying which agents can be excluded from the input to a pre-trained behavior model.
We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection.
We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state-of-the-art behavior model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RDAR , an RL-based framework for estimating which surrounding agents truly influence the ego vehicle’s behavior in autonomous driving. Instead of processing all scene entities with quadratic attention cost, RDAR formulates relevance estimation as an MDP, where actions are binary masks selecting agents passed to a frozen driving policy. A scoring policy is trained in closed-loop using off-policy actor–critic with V-trace, optimizing standard driving rewards (safety, progress). At inference, the top-k most relevant agents are retained, reducing input complexity. Evaluated on large-scale real-world data across several cities, RDAR maintains nearly identical driving performance to the full model while processing far fewer agents, achieving O(1) efficiency compared to O(N) attribution methods.

### Strengths
The paper proposes a conceptually novel framing of agent relevance estimation as an RL-based agent masking problem. RDAR offers substantial computational savings by enabling per-agent filtering in one forward pass.

The paper is well-structured. The method is designed to complement existing driving policies, which offers generalizable possiblilities. Visualizations demonstrate that RDAR’s learned relevance maps correspond well with human intuition

### Weaknesses
The main innovation lies in problem framing rather than algorithmic advancement. The paper states RDAR uses the same reward as the driving policy, but does not provide its explicit form or show ablations on reward sensitivity.

Experiments are confined to one in-house dataset. It remains unclear whether RDAR generalizes to other planners (e.g., learning-based or rule-based), datasets (e.g., nuPlan, WOMD), or domain shifts.

Although results are strong on average, the paper lacks qualitative or quantitative analysis of cases where RDAR incorrectly masks critical agents

The training requires large-scale distributed RL (IMPALA-like), but details on computational cost, convergence time, and sample efficiency are not reported.

The paper does not include comparisons or discussion of recent interaction reasoning frameworks such as JFP or braid-based, which explicitly model multi-agent dependencies.

### Questions
How sensitive is RDAR’s performance to the definition of the driving reward? 

If the frozen driving policy πᴰ is changed (e.g., a smaller or less robust model), can RDAR be reused or fine-tuned efficiently, or must it be retrained from scratch?

How does RDAR scale in dense scenes with hundreds of agents? Does the learned relevance distribution degrade or saturate under heavy traffic?

How does RDAR compare to lightweight attention sparsification techniques (e.g., Performer, linear attention) in accuracy and compute trade-offs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents RDAR (Reward-Driven Agent Relevance), a reinforcement learning framework for estimating the relevance of surrounding agents in autonomous driving scenarios. Experiments on large-scale urban datasets from cities such as Las Vegas, Seattle, and San Francisco demonstrate that RDAR can achieve similar levels of safety, comfort, and progress while processing only about ten agents per frame, leading to an order-of-magnitude reduction in computational complexity. Although the method is technically well-executed and produces interpretable results, its main contribution lies in applying existing reinforcement learning tools to a specific application domain rather than introducing a new theoretical framework. As a result, while the work offers practical impact for real-time autonomous driving, it may be regarded as too applied and incremental for ICLR’s main track standards.

### Strengths
The paper’s primary strength lies in the clarity and practicality of its idea. RDAR provides a simple yet effective method for reducing computational overhead in autonomous driving models by explicitly learning per-agent relevance through reward feedback. The formulation as an MDP is clearly presented, and the training procedure--combining the Gumbel-top-k trick for sampling and an actor–critic framework--is described in a reproducible and technically sound manner. The experiments are extensive, using multiple quantitative metrics such as collision rate, off-road violations, comfort, and progress, all of which support the effectiveness of the proposed method. The visualizations of relevance scores are also intuitive and align well with human reasoning about which agents matter most in driving decisions, adding interpretability to the model’s behavior. Furthermore, the method has clear industrial relevance, as it can be integrated into real-time driving stacks to prioritize computational resources dynamically and improve inference efficiency. Overall, the paper succeeds in demonstrating a conceptually simple but practically impactful contribution to the field of autonomous driving.

### Weaknesses
Despite its practical utility, the paper’s novelty and theoretical depth are limited. The central idea--using reinforcement learning to optimize a masking policy that filters input agents--is relatively straightforward and can be seen as an applied variant of existing work in reinforcement-learning-based feature selection. The paper does not introduce new algorithmic components, theoretical insights, or analyses that would substantiate it as a methodological breakthrough. Moreover, the work is tightly coupled to the autonomous driving domain, relying on proprietary datasets and simulators, and lacks evidence of generalizability to other contexts. The proposed system assumes access to a high-quality, frozen driving policy, meaning that RDAR’s success depends heavily on the underlying planner’s design rather than the novelty of the relevance model itself. Importantly, there is no theoretical justification or analysis that explains why reward-driven relevance estimation should outperform attention or attribution-based alternatives, nor any guarantee regarding optimality or convergence. Consequently, while the results are compelling, the paper does not provide sufficient conceptual depth or generality to meet the expectations of ICLR’s main track, which prioritizes fundamental advances in machine learning theory and algorithms. The work would likely be more appropriate for conferences such as CoRL, IROS, or ITSC, or for specialized NeurIPS or ICLR workshops focusing on real-world reinforcement learning or autonomous driving applications.

### Questions
Several questions arise regarding the stability, generality, and theoretical grounding of the proposed approach. It would be useful to know whether the learned relevance scores remain consistent over time and across scenes, or whether they fluctuate as agents enter and leave the field of view. Another question concerns the generalizability of RDAR: can the framework be applied to other multi-agent or multi-object systems, such as swarm robotics or video understanding, without substantial retraining? Additionally, the method currently requires setting a fixed number of selected agents, k, could this threshold instead be learned adaptively as part of the policy? Since the approach is reward-driven, it would also be valuable to assess its robustness to variations in the reward function or driving policy, as both could alter the learned relevance distribution. Finally, a deeper theoretical connection between the reward-driven relevance policy and established notions such as value-function-based importance weighting or causal influence would strengthen the contribution and situate the work more firmly within reinforcement learning theory.

### Soundness
2

### Presentation
2

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
This paper proposes RDAR, a reinforcement-learning method that learns per-agent relevance in autonomous driving by masking out non-critical agents from a frozen driving policy’s input. The relevance estimator is trained as an MDP with reward signals from driving performance. Experiments on large-scale driving data show comparable safety and comfort while processing far fewer agents.

### Strengths
1. RDAR improves computational efficiency: a single-pass O(1) scoring step replaces O(N) attribution, and with k≈10 it matches full-input safety/progress while delivering quadratic savings when downstream attention is pairwise. 

2. Relevance heatmaps match human intuition (e.g., yield/stop cases), making debugging easier and supporting targeted compute allocation.

### Weaknesses
1. Experiments rely on a non-public large-scale driving dataset, preventing others from reproducing or benchmarking RDAR on open platforms (e.g., Waymo, nuScenes).
2. The scoring policy’s quality is tied to the frozen driving model and reward design; biases or suboptimal rewards in the base system could propagate to RDAR’s relevance learning.
3. A constant number of selected agents may not suit varying scene complexity; adaptive or learnable selection would better reflect real-world driving conditions. 
4. Although inference is efficient, the closed-loop RL training likely requires heavy simulation and distributed resources, yet the paper provides no runtime or resource metrics.

### Questions
1. Can k be dynamically adapted per scenario?
2. How stable is agent selection over time?
3. Any plan to test on public datasets ?
4. Can the resource savings be quantitatively measured?

### Soundness
3

### Presentation
3

### Contribution
2
