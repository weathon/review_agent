# Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Scalable robot policy pre-training has been hindered by the high cost of collecting high-quality demonstrations for each platform. In this study, we address this issue by uniting offline reinforcement learning (offline RL) with cross-embodiment learning. Offline RL leverages both expert and abundant suboptimal data, and cross-embodiment learning aggregates heterogeneous robot trajectories across diverse morphologies to acquire universal control priors. We perform a systematic analysis of this offline RL and cross-embodiment paradigm, providing a principled understanding of its strengths and limitations. To evaluate this offline RL and cross-embodiment paradigm, we construct a suite of locomotion datasets spanning 16 distinct robot platforms. Our experiments confirm that this combined approach excels at pre-training with datasets rich in suboptimal trajectories, outperforming pure behavior cloning. However, as the proportion of suboptimal data and the number of robot types increase, we observe that conflicting gradients across morphologies begin to impede learning. To mitigate this, we introduce an embodiment-based grouping strategy in which robots are clustered by morphological similarity and the model is updated with a group gradient. This simple, static grouping substantially reduces inter-robot conflicts and outperforms existing conflict-resolution methods. Project page: https://haruki-abe.github.io/cross_embodiment_offline_rl_website

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed cross-embodiment benchmark tasks and dataset construction paradigms for evaluating offline RL algorithms. Through empirical studies, the paper identifies both positive and negative knowledge transfer between robots with different morphology when training on suboptimal datasets. The author then hypothesized that the negative knowledge transfer happened because of morphological dissimilarity between groups of robots. Quantitatively, this can be observed from the gradient conflicts of the robot-dependent actor losses. To mitigate this issue, the paper proposed a graph-based grouping strategy using the morphology of different robots, showing better cross-embodiment offline RL performances.

### Strengths
- Training offline RL algorithms on cross-embodiment datasets is an important step towards scalable robotic foundation models. The paper proposed a cross-embodiment dataset collection paradigm to evaluate offline RL algorithms.

- Through empirical studies, the paper relates the negative knowledge in cross-embodiment learning to the gradient conflicts of robot-dependent actor losses.

- The paper then proposed a graph-based grouping strategy using the morphology similarity of different robots to mitigate this issue.

### Weaknesses
- Although the paper discussed a set of optimal and suboptimal datasets, the set of rewards / tasks are somewhat limited. Specifically, the paper only considers simple locomotion rewards (following velocities) for different robots.

- One of the motivations of the paper is to compare the performance of offline RL algorithms trained on cross-embodiment datasets. However, the paper primarily focuses on the IQL algorithm and its variants, which limit the conclusions from experiments.

- Since the paper focuses on empirical studies, including the dataset and open-source code in the submission would strengthen the conclusions.

### Questions
- Sec. 3.1: What’s the purpose of the reward function? Or what’s the task of each MDP? Are different robotics solving the same task?
- Sec. 3.3: The explanation of the action encoder is not clear. Do different robots share the same action support? Although they share the same action space, some action dimensions might be redundant for some robots.
- line 174: What are “Forward” and “Backward” variants of the dataset?
- Sec. 4.1: The meaning of each column in Table 1 is not clear from the text in Sec 4.1. The introductions of each baseline are deferred into Sec 6, which makes the columns in Table 1 confusing.
- Sec. 4.2 and Sec. 4.3: If I understand correctly, the single robot variant does not have a pre-training stage. Is it a fair comparison between an algorithm with cross-embodiment pre-training and task-specific fine-tuning with an algorithm only involving task-specific training? Intuitively, the number of gradient updates are different.
- The experiment results for the texts between line 311 - line 318 are missing.
- Is there a quantitative relationship or a metric between the embodiment distance (Fig 3 (a)) and the gradient cosine similarity (Fig 3 (b))?

### Soundness
3

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
This study investigates how to apply offline reinforcement learning (Offline RL) on cross-embodiment robotic datasets containing a large amount of suboptimal data. This study finds that when robot embodiments are diverse, gradient conflicts across different types of robots , hinder learning and lead to negative transfer. To address this issue, this study proposes an embodiment grouping (EG) strategy based on morphological similarity, which clusters morphologically similar robots into groups and updates the policy sequentially per group, effectively mitigating gradient conflicts. Experiments demonstrate that this method achieves significant performance improvements on datasets dominated by suboptimal data, with an average gain as high as 39.8%.

### Strengths
1.The experimental results are strong, with experiments conducted on as many as 16 different types of robots. The proposed method shows a clear improvement over the baseline.

2.The paper employs a clear validation approach to demonstrate the impact of gradient conflicts caused by cross-embodiment data.

### Weaknesses
1.The work lacks real-robot experiments. All experiments are conducted in simulated environments, with no validation on physical robots.

2.Certain acronym definitions appear after their first use in the paper; for example, “EG” appears in Table 1 before being formally introduced, which affects readability.

3.The implementation only evaluates forward and backward motions across different robots, lacking validation on a broader range of tasks.

### Questions
1.Robot locomotion can be considered to have three degrees of freedom: forward/backward, left/right, and turning left/right. This paper only addresses tasks involving the forward/backward degree of freedom. Could the authors provide results trained on all three degrees of freedom?

2.In Algorithm 1, the grouping strategy is applied only to the actor and not to the critic. What is the rationale behind this design choice?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies using cross-embodiment with offline RL on locomotion tasks. It finds that offline RL is more valuable when offline dataset is more sub-optimal. Furthermore, it notices that cross-embodiment transfer of some robots have negative impacts, owing to large morpholy gap and conflict gradients. The paper then proposes valuable method to group different groups to prevent conflict gradients. Experiment show that the proposed method has strong improvement over baseline offline RL, comparing to other multi-task gradient techniques.

### Strengths
1. The setting of cross-embodiment offline RL is novel in the community.
2. The proposed method is easy to understand and is effective in practice.
3. The comparison involve gradient projection techniques used in continual learning / multi-task literature,
3. The evaluation of different embodiment is comprehensive.

### Weaknesses
1. Tasks only contain locomotion on walking. A comprehensive analysis / benchmark should involve more diverse tasks.
2. The baseline offline RL only contains IQL. More algorithms like CQL e.t.c should be evaluated. It is unclear if the claim can be extended.

### Questions
1. Can the results generalize to other imitaiton learning / offline RL algorithms.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Embodiment Grouping (EG), a morphology-aware approach for cross-embodiment offline RL. The authors observe that gradient conflicts arise when training shared policies across diverse morphologies. They measure embodiment similarity via the Fused Gromov–Wasserstein (FGW) distance, which jointly considers kinematic structure and behavior embeddings, and use this to cluster robots into morphology-aligned groups for group-wise policy training. The result: better policy transfer, stability, and data efficiency across heterogeneous robot datasets.

### Strengths
1.	The paper fills an important gap between robot foundation models and morphology-aware generalization. Prior work (e.g., URMA) has explored transfer across morphologies but typically ignores the destructive gradient interference that arises when training across incompatible embodiments. EG’s use of FGW distance to quantify morphology-level relationships and structure training groups is an elegant and novel solution.

2.	The experimental setup is extensive and thoughtfully designed. Using 16 diverse morphologies (quadrupeds, bipeds, manipulators) provides a broad basis for claims. The analysis linking gradient cosine similarity and policy degradation offers convincing causal evidence. Ablation studies on group count and morphological embedding confirm robustness of results.

3.	The paper explains why shared gradient updates can harm policies across dissimilar bodies, which is a phenomenon often observed but seldom quantified. Their visualizations of inter-group gradient alignments are insightful and elevate the reader’s understanding of multi-embodiment interference.

### Weaknesses
1.	While FGW distance effectively captures morphological similarity, there is no formal argument linking FGW similarity to gradient alignment or loss landscape smoothness. Without such a bridge, the theoretical contribution remains descriptive rather than predictive.

2.	Embodiment relationships evolve as policies adapt. Fixed grouping may lead to stale partitions that no longer reflect actual learning dynamics.

3.	The morphological embeddings used to compute FGW distance come from URMA, which itself encodes behavioral information. This could inflate grouping performance.

4.	The experiments focus exclusively on locomotion. It’s unclear if EG transfers to manipulation tasks, where control topology differs drastically.

### Questions
1.	How does EG perform as the number of embodiments scales? Does the FGW computation remain tractable?

2.	Could group-wise contrastive representation learning replace fixed clustering?

3.	Is there an optimal number of groups, or does performance plateau?

### Soundness
2

### Presentation
3

### Contribution
3
