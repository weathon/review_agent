# Lifelong Learning with Behavior Consolidation for Vehicle Routing

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Recent neural solvers have demonstrated promising performance in learning to solve routing problems. However, existing studies are primarily based on one-off training on one or a set of predefined problem distributions and scales, i.e., tasks. 
When a new task arises, they typically rely on either zero-shot generalization, which may be poor due to the discrepancies between the new task and the training task(s), or fine-tuning the pretrained solver on the new task, which possibly leads to catastrophic forgetting of knowledge acquired from previous tasks. This paper explores a novel lifelong learning paradigm for neural VRP solvers, where multiple tasks with diverse distributions and scales arise sequentially over time. Solvers are required to effectively and efficiently learn to solve new tasks while maintaining their performance on previously learned tasks. Consequently, a novel framework called Lifelong Learning Router with Behavior Consolidation (LLR-BC) is proposed. LLR-BC consolidates prior knowledge effectively by aligning behaviors of the solver trained on a new task with the buffered ones in a decision-seeking way. To encourage more focus on crucial experiences, LLR-BC assigns greater consolidated weights to decisions with lower confidence. Extensive experiments on capacitated vehicle routing problems and traveling salesman problems demonstrate LLR-BC’s effectiveness in training high-performance neural solvers in a lifelong learning setting, addressing the catastrophic forgetting issue, maintaining their plasticity, and improving zero-shot generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a lifelong learning setting for the VRP domain and proposes the Lifelong Learning Router with Behavior Consolidation (LLR-BC) to address the issue of catastrophic forgetting. Specifically, LLR-BC incorporates Confidence-aware Experience Weighting (CaEW) to prioritize critical experiences and Decision-seeking Behavior Consolidation (DsBC) to emphasize replicating past decisions. Extensive experiments on TSP and CVRP benchmarks demonstrate the effectiveness of LLR-BC.

### Strengths
* The paper is overall well-written and easy to follow.
* The studied topic, i.e., generalization of neural VRP solvers, is important, and the proposed setting is sound.
* The experiments are generally extensive and solid, demonstrating that the catastrophic forgetting issue is significantly mitigated, as shown in Fig. 3.

### Weaknesses
* This paper applies lifelong learning to the VRP domain, incorporating a reweighting mechanism and reverse KL divergence. The overall technical novelty appears somewhat limited.
* In line with the no free lunch principle, while the proposed method mitigates catastrophic forgetting on previous tasks, it may incur trade-offs elsewhere. Could the authors clarify what is sacrificed or traded off in their approach or setting?
* The caption of Fig. 2 should explicitly explain all notations.
* The writing in two parts could be improved:
  * In Section 3.1, the authors should first introduce the principle of reservoir sampling.
  * In Section 3.3, the connection between KLD and RKLD should be highlighted as a change in direction from $KL(\pi_\theta | \mathcal{P}) \to KL(\mathcal{P} | \pi_\theta)$. The authors should also mathematically explain why this directional change fundamentally alters model behaviour during learning.
* The experiments primarily compare lifelong learning baselines, but lack a comprehensive evaluation against traditional and neural VRP solvers. Could the authors include direct comparisons with recent generalizable neural VRP solvers, e.g., by multi-task training them on the same task distribution?
* Minor on line 152: (i.e., )

----

I intend to assign a borderline rejection in the first round of evaluation and would like to see the authors’ rebuttal addressing my main concerns, particularly the experimental comparisons.

### Questions
* Is Fig. 1 a conceptual illustration or based on real experimental results?
* Could LLR-BC be applied to SL-based solvers as well?
* The experience buffer is state-based, why not consider an instance-based design?
* Is there any empirical evidence supporting the claims made in lines 236–240?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a lifelong learning framework called LLR-BC for learning to solve Vehicle Routing Problems (VRPs) across different distributions and scales. LLR-BC employs a customized experience replay strategy across sequentially arising tasks to prevent catastrophic forgetting. The approach introduces two key components: a confidence-aware experience weighting (CaEW) module and a decision-seeking behavior consolidation (DsBC) loss based on reverse KL divergence. Results on TSP and CVRP validate the effectiveness of the proposed methods against baselines.

### Strengths
* The paper studies an interesting and relatively underexplored area for neural VRP solvers: lifelong learning across diverse problem distributions and scales. 
* The presentation and illustrations are clear, making the paper easy to follow. 
* The work presents comprehensive experiments covering multiple task orders, diverse distributions and scales, and multiple backbone VRP neural solvers.
* The performance appears promising under the employed metrics.

### Weaknesses
* There is some conceptual overlap with existing works, such as Li et al. 2024 and Feng et al. 2025, which address similar lifelong learning setups. It would be useful to directly compare with these existing methods. Also, while the customization of the replay buffer for VRPs is appreciated, the idea of experience replay is a standard technique in lifelong learning literature.
* The paper proposes using the variance of node selection probabilities as a proxy for model confidence. However, this lacks clear empirical evidence or theoretical justification that probability variance truly represents confidence. My concern is that this may be too strong and fails to distinguish between two scenarios: (1) a genuinely low-confidence model with high uncertainty, and (2) a policy that has learned multi-modal decisions where multiple nodes are equally good selections at certain division points with high confidence. 
* The design and naming regarding “DsBC” seem confusing to me, as the loss functions in Equations (1) and (2) are relatively straightforward applications of reverse KL divergence. It is unclear what new conceptual insights or contributions DsBC provides here.
* While the experiments are extensive in scope, the absolute performance of the model on each individual task is never reported. In practice, practitioners care not only about whether a lifelong learning model can avoid catastrophic forgetting, but also about the actual solution quality (such as optimality gaps) achieved on each new task. 
* The paper lacks a comparison with other state-of-the-art methods that target generalization across distributions without requiring additional training or lifelong learning mechanisms.

### Questions
Other than the questions listed in the weakness, I also have the following questions:
* Which epoch of the model training is used to update the buffer experience, and what is the rationale for this choice? How sensitive is performance to this decision?
* How are the hyperparameters for buffer batch size determined?
* How well does variance actually serve as a proxy for confidence? Are there empirical studies or theoretical analyses that validate this design choice against alternative confidence measures?
* How diverse is the experience buffer after running the algorithm? Does the buffer maintain sufficient diversity across tasks, or does it become dominated by certain task types or distributions?

### Soundness
2

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
4

### Summary
This paper proposes LLR-BC solvers that address the catastrophic forgetting problem when adapting to sequential tasks with different scales and distributions. It further introduces two key modules: CaEW and DsBC. Moreover, Extensive experiments demonstrate the effectiveness of LLR-BC.

### Strengths
1.	The paper introduces two novel modules CaEW and DsBC， with the innovative use of Reverse KL Divergence enabling decision-focused behavior consolidation. It also removes the common “fixed task order” constraint, evaluating on randomized sequential tasks for greater realism and originality.
2.	The model's performance was evaluated across multiple task sequences and VRP scenarios of varying scales and distributions. Furthermore, the paper includes comprehensive ablation experiments and hyperparameter sensitivity analyses, demonstrating the independent effectiveness of the CaEW and DsBC modules as well as the model's robustness to parameter variations.
3.	The presentation is clear and easy to follow.

### Weaknesses
1. Limited Innovation: The paper shows limited novelty based on the existing lifelong VRP models. The comparative experiments include only Fine-tuning, EWC, and LiBOG. However, no comparisons are made with more recent lifelong learning approaches (e.g., Li et al., 2024; Feng et al., 2025) or with cross-problem/multi-task frameworks. Including these would better position the proposed method within the current research landscape.
2. Potential Bias in Experience Buffering: Experience buffering is conducted only during the final epoch of each task. A further analysis or ablation study on this design choice would enhance the completeness and reliability of the evaluation.
3. The manuscript contains many typos (e.g., “stuides” → “studies,” “LLB-RC” → “LLR-BC”)

### Questions
Clarification on the Logic of the CaEW Mechanism: In Section3.2 , CaEW aims to assign higher weights to “low-confidence” decisions, noting that “lower variance indicates lower confidence.” However, the formula actually assigns higher weights to experiences with low variance (i.e., high confidence). Could you please clarify?

Regarding the limitation of fixed sampling size |ε|: In conclusion that using a fixed |ε| is a limitation when task scales vary significantly. Could you elaborate on why this is problematic? Is it because tasks of different scales require different numbers of experiences for effective consolidation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a lifelong learning framework, LLR-BC, for enhancing the generalization performance of existing neural models for vehicle routing problems (VRPs). It trains multiple tasks (including VRPs with different distributions and sizes) sequentially, and employs experience replay strategy and introduces behavior consolidation to preserve knowledge of prior tasks while learning new ones. Experimental results show the effectiveness of the proposed method.

### Strengths
1. This paper employs lifelong learning to solve the generalization issue in VRPs across different distributions and problem sizes.
2. In experiments, this paper introduces interesting evaluation metrics.
3. Experimental results show the effectiveness of the proposed method against baseline methods.

### Weaknesses
1. The core idea of this paper, using lifelong learning to address the generalization issue of VRP while alleviating catastrophic forgetting is similar to [1][2]. Although the authors claim that [1][2] focuses on a fixed task order, while the proposed LLR-BC could handle flexible task orders, the core contribution seems to be similar.
2. This paper aims to enhance the generalization ability of neural models for VRPs, while the comparison of the LLR-BC with other neural models that are specialized for generalizability enhancement is not reported.
3. The experience replay strategy is a standard strategy in lifelong learning. And the core idea of behavior consolidation seems to be similar to "inter-task regularization" and "intra-task regularization" in [1], which both learn to preserve knowledge from past ones when training on new tasks. A more clear discrepancy between [1] and this paper should be explained.

[1] Jingwen Li, Zhiguang Cao, Yaoxin Wu, and Tang Liu. Enhancing the cross-size generalization for solving vehicle routing problems via continual learning, 2024. URL https://openreview.net/forum?id=WdvT2UgsTK.

[2] Shaodi Feng, Zhuoyi Lin, Jianan Zhou, Cong Zhang, Jingwen Li, Kuan-Wen Chen, Senthilnath Jayavelu, and Yew-Soon Ong. Lifelong learner: Discovering versatile neural solvers for vehicle routing problems, 2025. URL https://arxiv.org/abs/2508.11679.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
