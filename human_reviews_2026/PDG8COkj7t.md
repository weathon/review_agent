# Test-Time Graph Search for Goal-Conditioned Reinforcement Learning

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Offline goal-conditioned reinforcement learning (GCRL) trains policies that reach user-specified goals at test time, providing a simple, unsupervised, domain-agnostic way to extract diverse behaviors from unlabeled, reward-free datasets. Nonetheless, long-horizon decision making remains difficult for GCRL agents due to temporal credit assignment and error accumulation, and the offline setting amplifies these effects. To alleviate this issue, we introduce Test-Time Graph Search (TTGS), a lightweight planning wrapper for pretrained GCRL policies which only uses the pretraining dataset. TTGS accepts any state-space distance or cost signal, builds a weighted graph over dataset states, and performs fast search to assemble a sequence of subgoals that a frozen policy executes. When the base learner is value-based, the distance is derived directly from the learned goal-conditioned value function, so no handcrafted metric is needed. TTGS requires no changes to training, no additional supervision, no online interaction, and no privileged information, and it runs entirely at inference. On the OGBench benchmark, TTGS improves success rates of multiple base learners on challenging locomotion tasks, demonstrating the benefit of simple metric-guided test-time planning for offline GCRL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a test-time routine for goal-conditioned reinforcement learning. By relying on (often available) pre-training dataset and an estimator of temporal distances (e.g. the goal-conditioned critic), the algorithm constructs a graph whose vertices are states uniformly sampled from the dataset. Each edge is weighted according to the (potentially learned) distance between its two endpoints; long edges are penalized in order to compensate for estimation errors. Dijkstra's algorithm can then find the shortest path from the current state to the goal over the graph: the next vertex on the path can be commanded to the policy as a subgoal. The authors provide evidence for the decay in current algorithms' performance on distant goals (Figure 2) as a motivation of this work. The method is then evaluated in giant versions of diverse OGBench mazes. Finally, the authors ablate the criterion for subgoal selection, as well as relevant hyperparameters.

### Strengths
- This work is clearly presented and well-written.
- The motivation is clear; Figure 2 is instrumental to highlighting it.
- The experiments are well designed and informative with respect to the method's effectiveness.

### Weaknesses
- The main weakness of this work lies, in my opinion, in its novelty. Methodologically, the proposed technique matches Search on The Replay Buffer (Eysenbach 2019), among many other works citing it and predating it. To the best of my knowledge, the main novelty is in the soft penalization of edge weights, which however is exponential and therefore rather hard. It is not clear whether this should be a fundamentally better option than the standard hard penalization scheme. The claim on lines 52-53 ("these approaches require specialized training or additional data") does not apply to SORB to the best of my knowledge.
- The evaluation in this work is limited to mazes, in which semi-parametric techniques are known to work well, as reported in SORB. The real challenge of these approaches lies, in my opinion, in less structured environment (e.g. manipulation). While well-planned, the experimental section does not report significantly new findings.

### Questions
## Minor issues and questions
- Why do self-loops need to be penalized (line 245)? Is there a reason for Dijkstra to select self-loops?
- Why is GCIQL outperforming HIQL in Figure 2c?
- Line 312: the comparison with HIQL is not a clean ablation of learned vs non-learned high-level planner. HIQL uses a single n-steps ahead value estimate to guide subgoal selection, while the proposed method performs full planning on the graph, which is inherently more capable.
- Line 441: could hierarchical clustering help in reducing the number of pairwise distance to be computed? This might be an interesting extension in case compute is seen as a limitation.
- The performance of GAS on pointmaze in Table 3 stands out in the column. Is there a reason why the method performs poorly?

## Conclusion
While this submission is well written and well designed, I believe that it does not make a significant (methodological or empirical) contribution on top of existing graph-search-based planning methods such as SORB. In case this stems from a misunderstanding, I am happy to discuss this point at length in the rebuttal.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Test-Time Graph Search (TTGS), a lightweight test-time planning framework that enhances the long-horizon performance of pretrained goal-conditioned reinforcement learning (GCRL) agents. The key idea is to perform graph search using the learned goal-conditioned value function without retraining or fine-tuning the policy. TTGS constructs a weighted state graph from the offline dataset, where edge weights are derived from the frozen value function V(s,g). At inference time, it applies Dijkstra’s shortest-path search on this graph to generate a sequence of intermediate subgoals, which the frozen policy executes sequentially.

### Strengths
* The paper presents a training-free test-time planning framework for pretrained goal-conditioned reinforcement learning (GCRL) agents.
While structurally related to prior graph-based methods such as DHRL, NGTE, and GAS—which all learn value or distance functions and perform Dijkstra-based planning—TTGS distinguishes itself conceptually by reusing an already trained goal-conditioned value function from existing offline RL algorithms (e.g., HIQL, GCIQL, QRL) rather than learning new distance estimators or hierarchical policies.

* This reframing of the inference-time usage of a value function as a graph-based planner is a creative reuse of existing components, highlighting that strong long-horizon reasoning can emerge without retraining or additional supervision. In that sense, TTGS’s originality lies not in architectural novelty, but in its minimal yet effective re-interpretation of value-based RL

### Weaknesses
### 1. Limited algorithmic novelty beyond existing graph-based GCRL frameworks.

While TTGS is presented as a test-time planning framework, its core structure (value function–based distance estimation, graph construction, and Dijkstra search) is conceptually similar to prior works such as GAS (Baek et al., 2025), NGTE (Park et al., 2024), and DHRL (Lee et al., 2022). These methods learn a value or distance estimator and perform shortest-path search over state graphs to derive subgoal sequences.

The main distinction of TTGS is that it reuses a pretrained goal-conditioned value function rather than learning one jointly with the planner.
However, this difference is largely procedural (when the graph is built) rather than algorithmic. From a systems perspective, both TTGS and GAS follow the same computational pipeline: Value/Distance Learning + Graph Construction + Graph Search.

Therefore, the “training-free” claim appears somewhat overstated. Unless TTGS can demonstrate reuse of a foundation-scale pretrained model across new domains without any retraining or data-specific adaptation, the claimed efficiency advantage remains questionable.

### 2. Insufficient comparison and fairness issues with representation-based offline planners:

The paper highlights its superiority over value-based baselines such as HIQL, GCIQL, and QRL in the main results, but it does not include comparisons with recent temporal-distance representation or graph-based approaches. Although GAS (Baek et al., 2025) is mentioned in the Appendix, a precise and analytical comparison is lacking. The authors claim that constructing the graph by random sampling from the replay buffer yields similar performance to GAS’s proposed node-selection strategy, but this is not convincingly demonstrated.

For example, on antmaze-giant-stitch-v0 and antmaze-large-explore-v0, GAS achieves higher success rates than TTGS.
The paper does not analyze why such differences occur or what aspects of the graph construction contribute to these results.
Moreover, TTGS appears to use per-environment hyperparameter tuning (e.g., τ, T), while GAS results are cited using default hyperparameters from antmaze when reporting performance on humanoidmaze, which is not directly evaluated in GAS.
This asymmetry in evaluation settings undermines the fairness of the comparison.
A proper head-to-head evaluation using identical experimental protocols would make the empirical contribution more convincing.

### 3. Applicability to non-navigational or high-dimensional tasks remains unclear:

Most experiments involve navigation-based maze tasks where geometric distance provides a natural metric. It remains unclear whether TTGS can generalize to non-navigational domains,
such as manipulation tasks where goals are semantic, multimodal, or compositional (e.g., object placement or visual rearrangement). Including non-navigation benchmarks (e.g., numerical and visual manipulations tasks) would help establish broader generality and real-world relevance.

### Questions
Please provide your responses to the weaknesses mentioned above.

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
4

### Summary
The work studies goal-conditioned reinforcement learning (GCRL) on long-horizon tasks.
The work observes that the policies trained by common GCRL algorithms do not exploit all geometric structure learned by the goal-conditioned value function.
They propose a training-free test-time method that leverages only the offline pre-training data to improve policies during evaluation.
Their proposed method constructs a directed graph of subsampled states, leveraging the goal-conditioned value function to compute distances.
At the beginning of an evaluation episode, Dijkstra's algorithm is used to find a shortest path to the goal, and during evaluation the agent adaptively attempts to reach one of the subgoals along the estimated shortest path.

### Strengths
The paper proposes a novel method for GCRL and extensively benchmarks it against several common GCRL algorithms on hard tasks.
The experimental results in long-horizon planning are strong and convincing.
Further, the paper includes extensive ablation studies.

### Weaknesses
* Results would be stronger if compared against more recent "hierarchy-inspired" methods such as SAW [1] or test-time methods such as GC-TTT [2].
* The proposed method is not truly adaptive during evaluation, since the determined shortest path remains fixed. It may be beneficial to recompute a shortest path if the agent goes astray from the originally computed path at intermediate subgoals.
* Building an explicit graph (even if only over subsampled states) may be prohibitive in larger environments.
* Based on the results of the paper, a natural question is whether the proposed explicit hierarchical search could be distilled into a high-level policy (similar to HIQL). In other words, are there specific design choices in HIQL / the architecture that prevent effective training of the high-level policy?
* The optimal hyperparameters seem to depend relatively strongly on the environment (though the performance gains over the initial policy are robust).

[1]: Zhou et al., Flattening Hierarchies with Policy Bootstrapping. https://arxiv.org/pdf/2505.14975
[2]: Bagatella et al., Test-time Offline Reinforcement Learning on Goal-related Experience. https://arxiv.org/pdf/2507.18809

### Questions
* In figure 2, what is your understanding for why HIQL fails in this example?
* Did you ablate the clipping of $\hat{d}$?
* Can you comment on the relation to other test-time methods for GCRL such as GC-TTT [2, above]?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Offline GCRL methods struggle especially in long-horizon tasks because of temporal credit assignment problem. Recent approaches propose decomposing problem into reachable subproblems via hierarchical methods or graph-search methods, but they still require specialized training or additional data beyond the offline training data. This paper proposes a test-time adaptation approach for improving the performance of offline GCRL without requiring additional data during the test time. The idea is first building a graph of distances (calculated using the learned value function) using samples from the offline dataset, then finding the shortest path from start to the goal, and selecting reachable subgoals on the path. Experimental results shows graph-based test time fine-tuning improves the performance.

### Strengths
**Interesting Approach:** Using test-time fine tuning for offline GCRL via constructing a graph is interesting. 

**Improved Performance:** Experimental results shows performance improvements over baselines.

### Weaknesses
- **Computational Complexity**: The proposed approach requires constructing a graph based on large offline datasets, then calculating the shortest path, and selecting subgoals at test deployment. Because of the graph construction and shortest-path calculation, the proposed method seems computationally complex. Therefore, the computational overhead over baselines must be discussed clearly. Wall-clock times and FLOPs should be presented for both TTGS and baselines to show how much computational overhead is required.

- **Following the Shortest Path**: The idea is built on the assumption that the agent will initially follow the shortest path, since the shortest path is only calculated once. If the agent deviates from this path, it might become unusable. This must be discussed clearly, because due to the credit assignment problem in offline RL, the value function is generally noisy, which might lead the agent away from the shortest path. What happens if the agent ends up in a different corner, far from the shortest path? Have you ever considered re-calculating the shortest path at regular intervals during test time? 

- **Distance Prediction**: It is stated that the authors find the value function "very reliable" for calculating the distance near goal states, which is not clearly presented or discussed.

- **Experimental Setting**: The proposed approach is only evaluated on five long-horizon tasks against baselines. The applicability of the proposed approach to short- and medium-horizon tasks should be discussed, and if possible, those tasks should be included in experiments. In addition, some environments in Table 1 are omitted from Figure 3, which is concerning. Please report results for all environments in Table 1 in Figure 3 as well.

### Questions
- How computationally complex is the proposed approach?
- Why is the shortest path only calculated once at the beginning of the test? Have you ever considered re-calculating the shortest path from the current state to the goal at certain intervals during test time deployment?
- What happens if the agent deviates from the calculated shortest path because of the noisy value function it learned during training? Since the shortest path is calculated once, when the agent deviates from it, does this render this shortest path sub-optimal?
- Why are some environments in Table 1 not reported in Figure 3?
- Can you please also elaborate on the applicability of the proposed approach to short- and medium-horizon tasks?

### Soundness
2

### Presentation
2

### Contribution
2
