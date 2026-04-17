# Scaling Goal-conditioned Reinforcement Learning with Multistep Quasimetric Distances

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 6, 4

## Abstract
The problem of learning how to reach goals in an environment has been a long-
standing challenge in for AI researchers. Effective goal-conditioned reinforcement
learning (GCRL) methods promise to enable reaching distant goals without task-
specific rewards by stitching together past experiences of different complexity.
Mathematically, there is a duality between the notion of optimal goal-reaching
value functions (the likelihood of success at reaching a goal) and temporal dis-
tances (transit times states). Recent works have exploited this property by learning
quasimetric distance representations that stitch long-horizon behaviors using the in-
ductive bias of their architecture. These methods have shown promise in simulated
benchmarks, reducing value learning to a shortest-path problem. But quasimet-
ric, and more generally, goal-conditioned RL methods still struggle in complex
environments with stochasticity and high-dimensional (visual) observations. There
is a fundamental tension between the local dynamic programming (TD backups,
temporal distances) that enables optimal shortest-path reasoning in theory and the
statistical global MC updates (multistep returns, suboptimal in theory). We show
how these approaches can be integrated into a practical GCRL method that fits a
quasimetric distance using a multistep Monte-Carlo return. We show our method
outperforms existing GCRL methods on long-horizon simulated tasks with up to
4000 steps, even with visual observations. We also demonstrate that our method
can enable stitching in the real-world robotic manipulation domain (Bridge setup).
Our approach is the first end-to-end GCRL method that enables multistep stitching
in this real-world manipulation domain from an unlabeled offline dataset of visual
observations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on offline goal-reaching reinforcement learning, and pinpoints two families of methods: on-policy algorithms based on Monte-Carlo returns (and in several cases contrastive learning), and off-policy algorithms which enforce the known invariances of goal-conditioned value function through quasi-metric network architectures. This works extends TMD (Myers 2025), which previously united this two frameworks, by considering multi-step returns. Interestingly, this induces significant gains in performance across standard goal-conditioned tasks from the OGBench suite. The authors further perform an evaluation on the BRIDGE hardware setup, which confirms the strong performance of the algorithm. These results are accompanied by relevant ablations on the multi-step loss.

### Strengths
- Empirical results are very convincing, both in breadth (involving a large number of OGBench tasks as well as hardware experiments) and in outcomes (displaying large improvements over strong baselines).

### Weaknesses
- The proposed algorithm is strongly aligned with TMD, but in my opinion does not fully acknowledge this. To the best of my knowledge, each component in Section 4 (except for (11), which is a direct multi-step extension of the TD loss (12)) was already introduced in TMD. If I understand correctly, the final algorithm is simply TMD with a multi-step loss. If this is the case, it should be acknowledged in full.
- This works claims to combine off-policy and on-policy learning, but does not comment on whether the final value estimates are recovering the optimal value, or the value of the behavioral policy. Can the authors provide a formal discussion of what the proposed objectives recover, and why it is motivated?
- Related works are overall short (e.g. there is a single reference in the paragraph on GCRL), poorly formatted (several undefined references) and poorly written (the structure of the sentences in the last two paragraph is incorrect).

### Questions
### Minor issues and questions
- Line 12: remove "in"
- There are several broken or wrongly formatted references through the paper
- Line 122: this sentence is hard to follow: "X have demonstrated that Y instead of Z"
- Equation 7: the notation is unclear, as $s$ and $a$ appear under the expectation, as well as on the left side. formulating the expectation over $s'$ alone would be cleaner in my opinion.
- Equation 10: this choice seems to follow from TMD, which should be referred to in this case
- Figure 1: the caption refers to a 10x larger horizon, is this an overstatement? What is the length of optimal trajectories in giant and colossal mazes?
- What is the reasoning behind the selection of environments in Table 1? e.g. why is antmaze-large-explore evaluated instead of antmaze-large-navigate? The current selection appears somewhat arbitrary.
- Two of the ablation studies (matching the first two questions in 5.3) were relegated to the Appendix. This should be noted in the text.
- Table 3: how is the regularization parameter $\alpha$ tuned in each environment/algorithm combination?

### Conclusion
Despite the impressive empirical results, I currently lean towards rejection. To the best of my understanding, this method is an n-step extension of TMD, which is a minor contribution but is not problematic per se. My main concern is that this strong connection between the algorithms is not directly highlighted in the paper, which as a result seems to over claim its contribution. In am happy to further discuss whether my understanding is correct during the rebuttal phase. Moreover, unlike TMD, this works lacks an analysis of the objective and its solutions.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a new method for offline GCRL. The authors introduce Multistep Quasimetric Estimation (MQE), which applies a multi-step Monte-Carlo return to a quasimetric distance-based method. They evaluate on OGBench and a real robot, where MQE can outperform all baselines.

### Strengths
1. The theoretical part is clear and easy to understand. The method is simple, but it still brings a notable improvement in performance. This makes the idea accessible for other researchers and shows that even a straightforward change can lead to meaningful progress.

2. The experiments provide strong comparison results. The authors choose many baselines, including some recent works. This helps confirm that the gains are not due to weak baselines and shows that the method remains strong under fair comparisons.

3. The paper includes real-world testing. The method achieves a higher success rate in practice, and it is able to combine multiple motion segments into complete trajectories. This supports the claim that the approach can work outside simulation.

4. There is enough hyperparameter study. The authors include an analysis that explains how different settings affect performance. This makes the method easier to apply and helps readers understand which parts matter most.

5. The code had been released, showing its reproducibility

### Weaknesses
1. While MQE can indeed improve performance, the choice of hyperparameters and the selection of the best waypoint introduce heavy computational cost. This limits the method’s practical use, especially when scaling to more complex tasks or real-time applications.

2. There are many citation errors in the paper, for example, at line 96, 91, 399, and 431. In addition, there are indexing mistakes and incorrect claims. For instance, in Algorithm 1, the loss at line 5 should refer to Eq. 11. Another example is at line 435, where the text claims that TRA-g reaches a positive success rate in the quadruple PnP task, but the table shows 0/15. These issues show that the writing and validation of statements are not strict enough.

3. The value of $\alpha$ used during policy extraction is a hyperparameter that must be tuned separately for different environments. This shows that the method needs heavy hyperparameter tuning to reach the reported performance, which is not practical. I suspect that much of the performance gain shown in the paper comes from tuning these hyperparameters instead of the strength of the method itself.

### Questions
1. What happens if the hyperparameters are not tuned, or only tuned with a small number of trials? How much does the performance drop in that case?

2. Can the authors provide theoretical support for using a geometric distribution to select the waypoint? Why is this distribution a reasonable choice compared with other alternatives?

3. In line 386, the paper states that the task "has never been completed without the use of hierarchical policies or high-level planners". Then why is there no comparison against hierarchical policies? Also, can MQE be integrated into a hierarchical framework? If so, why not include such a comparison to better understand its advantages and limitations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Multistep Quasimetric Estimation (MQE), a novel goal-conditioned reinforcement learning (GCRL) method that unifies temporal-difference (TD) and Monte Carlo (MC) learning through a quasimetric distance representation. MQE leverages multistep returns under a quasimetric architecture to propagate value information efficiently across long horizons while maintaining theoretical consistency with optimal value functions. It further enforces action invariance and one-step consistency constraints to stabilize learning. Empirically, MQE achieves state-of-the-art performance on long-horizon offline GCRL benchmarks (up to 4000 steps) and demonstrates strong compositional generalization in real-world robotic manipulation tasks. Its key contribution is showing that multistep temporal consistency and quasimetric structures can be combined to enable scalable, end-to-end goal-reaching from raw, unlabeled offline data.

### Strengths
This paper makes an original contribution by unifying multistep value learning and quasimetric representations into a single, scalable framework for goal-conditioned reinforcement learning. The idea of integrating multistep temporal consistency with quasimetric architectures is both novel and technically elegant, addressing long-standing limitations in horizon generalization and stability. The paper is of high quality, with strong theoretical grounding, clear algorithmic exposition, and comprehensive experiments across both simulated and real-world robotic domains. Its clarity allows readers to follow complex ideas with precision, and its significance lies in demonstrating a practical path toward scalable, compositional goal-reaching in offline RL—bridging a crucial gap between theory and real-world applicability.

### Weaknesses
While the paper is strong overall, several aspects could be improved to strengthen its impact. The theoretical analysis, while elegant, could benefit from clearer discussion of its assumptions and limitations—particularly regarding stability and convergence in high-dimensional or stochastic environments. Finally, the presentation could be improved by providing more intuition and visualization of the learned quasimetric distances to help readers better grasp the geometric and representational properties driving the observed performance gains.

### Questions
1.Could the authors clarify the conditions under which the proposed multistep quasimetric estimation guarantees convergence or consistency? Specifically, how sensitive are these guarantees to function approximation errors or off-policy data distributions?

2.The method integrates multistep temporal consistency into the quasimetric framework, but it remains unclear how much of the empirical gain arises from longer-horizon updates versus the quasimetric structure itself. Could the authors provide quantitative or qualitative evidence disentangling these effects (e.g., via controlled ablations or visualizations)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of scaling RL to real-world, long-horizon robotics-tasks via offline GCRL. The main contribution of the paper is a novel algorithm, Multistep Quasimetric Estimation (MQE). The main technical innovation is how to learn multistep returns under quasimetric architectures while preserving the ability to learn the optimal value function. The authors show that MQE obtains state-of-the-art results in challenging long-horizon GCRL tasks in OGBench and a real-world robotics task.

### Strengths
- The paper addresses an fundamental challenge in RL - namely, how can we scale to real-world tasks? Personally, I think the approach of trying to induce compositional learning via offline GCRL, succ, and goal-stitching is very promising. The topic will definitely be interesting to the ICLR community.
    
- The proposed approach is empirically highly effective compared to baselines in state-based datasets and the real-world robotics task. Overall, the evaluation is thorough. There is a large-scale evaluation of many methods and tasks.

### Weaknesses
- **Writing:** the writing is the biggest weakness of this paper. Unfortunately, the authors focus on explaining technical details of the method to an audience of experts in offline RL / GCRL, while failing to communicate high level, key messages to a general audience (see below). I’m concerned that this will limit its interest/impact to the general ICLR audience.
    
    - The motivation and the problem being solved is not very coherent at the beginning of the paper. I found myself wondering for a while whether the authors were addressing GCRL, or offline RL. It took until I almost finished skim-reading the paper to confirm that the authors address offline GCRL. Looking back at the paper, I think this occurred because the abstract only addresses GCRL rather than offline GCRL, and the term “Offline GCRL” only appears in conjunction in 2 places in the manuscript.
        
    - What is the key insight of the paper? Why do multistep returns matter so much torwards improving performance on the long-horizon offline tasks?
        
    - Some minor typos in the citations:
        
        - Please make sure to use citep where appropriate.
            
        - See Lines 91, 96, 122, 399
            
    - Lack of clarity:
        
        - What is task prowess metric in Fig 3 and how does it differ from success rate?
            
- **Soundness of Method**: The proposed method relies on trick of sampling next states w/ higher prob according to a Bernoulli distribution, thus indicating the auxiliary loss in 4.2 is not sufficient to fix bias issues. Can the authors comment on this point? What is the relative contribution of this trick to the final performance of MQE vs the auxiliary loss?

### Questions
- The waypoints are randomly sampled according to the geometric distribution. Are there any theoretical benefits of geom distr. over other distributions, esp those considered in App. E?
    
- What is the benefit of multi-step return in this setting? From another angle, what is the problem with single step returns?
    
- Intuitively, why does Lp violate optimality while Lt doesn't? Can the authors provide an illustrative example or proof?
    
- Expts:
    
    - Why does MQE do particularly poorly on cube_triple_noisy?
        
    - Why does MQE underperform baselines on visual tasks, especially compared to HIQL?
        
    - Why not compare against HIRL in Bridge Data?
        
- Is this method applicable to online settings?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Multistep Quasimetric Estimation (MQE), an offline GCRL method that integrates multistep returns within a quasimetric network architecture.
The paper aims to reconcile the tension between the theoretical optimality of local TD updates and the superior horizon scaling of global Monte-Carlo methods.
MQE updates values by regressing distances using geometrically sampled "waypoints" and enforcing action invariance.
MQE is evaluated on complex stitching and noisy tasks from OGBench where it outperforms common baselines, including often outperforming the hierarchical HIQL.
Moreover, the paper presents impressive compositional generalization in real-world robotic manipulation.

### Strengths
The paper's main strength is the integration of multistep backups into the quasimetric learning framework.
This enables superior horizon generalization compared to prior methods, which the paper demonstrates extensively on challenging environments from OGBench requiring extremely long-horizon planning.
The success in the real-world BridgeData experiments is compelling.
To my knowledge, complex multi-stage compositionality (e.g., Quadruple Pick and Place) using a flat (non-hierarchical) policy architecture represents significant progress in scalable GCRL.

### Weaknesses
* The multistep backup (Eq. 9) is inherently biased towards the behavior policy​ when the step k′>1. While the authors mitigate this with 1-step consistency weighting and action invariance, the justification for how this combination robustly overcomes the strong bias of the multistep return is primarily empirical.
* In my view, the paper is missing a comparison to recent work [1] that addresses the identical challenge of achieving long-horizon GCRL performance with a flat policy.

The presentation has several issues:
* The related work section has several missing or ill-formatted citations.
* The paper has several incomplete sentences, e.g., line 91.
* Page 5 includes a lot of extra spacing.
* Overall, I find the methods section (Section 4) hard to follow. It would be beneficial to add structure to the section and provide additional guidance to the reader.

### Questions
* Is the quasimetric architecture essential for the success of the multistep backup, or could this backup strategy also improve standard value-based methods (e.g., IQL) without the explicit distance structure?

Note: The paper title of the PDF does not match the title on OpenReview.

### Soundness
2

### Presentation
1

### Contribution
3
