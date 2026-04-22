# DreamExplorations: Leveraging Suboptimal Noisy Robot Trajectories in Offline RL

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Exploration is a desirable characteristic in online reinforcement learning, where the online agent can interact with the environment, explore the diverse states, and update the policy. However, since the datasets of offline reinforcement learning are static and the traditional offline RL algorithms always rely on the relatively good quality of demo agents, it is very hard to explore the diversity of state space. In this paper, we have found out that in offline goal-conditioned reinforcement learning (OGCRL), we can theoretically leverage suboptimal/high noisy datasets for state exploration and we have designed a pipeline to use them. In this case, the highly noisy datasets which are always discarded and regarded as useless datasets in previous researches are used as exploration experts to keep improving the performances of offline reinforcement learning as we scale the sizes of suboptimal datasets. Experimental results demonstrate that our method consistently outperforms baselines and significantly improves models trained solely on high-quality data, especially in environments with large state spaces. This work highlights the untapped potential of imperfect data in enhancing the robustness and generalization of offline RL. We will open-source our code after publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
**Summary.** This work studies how do we leverage sub-optimal trajectories in offline RL settings. The authors propose an algorithm for merging value information from both high- and low-quality datasets. They claim that suboptimal data can enhance state space coverage and improve value estimation. Their pipeline involves three key properties: 
- decoupled value learning for exploitation and exploration
- a learned novelty ratio network to weight the contributions of each value function
- mixture-based policy training.

---
**Review summary.** The paper tries to tackle a popular problem in offline RL, but its technical and empirical depth is insufficient for acceptance. The approach is conceptually intuitive yet incremental, the theory and practice linkage is weak, and the writing quality undermines the clarity of otherwise reasonable ideas. Strengthening the mathematical rigor, providing richer experimental analyses, and improving exposition could elevate the work for future submission. Therefore, the reviewer assigns an initial score of 2 and plan to revisit this rating after the authors address the concerns and questions raised in this review.

### Strengths
**Writing**
- Figures and tables are numerous and relevant to the claims.

---
**Methodology**
- The proposed decoupled value learning and novelty-weighted value fusion present a clean and modular extension to HIQL.
- The idea of using suboptimal data to improve state coverage and generalization is practical. 

---
**Theory**
- The bias-variance decomposition and the closed-form derivation of the optimal weight demonstrate an attempt to ground the intuition.

---
**Experiments**
- The didactic simulation is helpfuul to understand how combining suboptimal and expert trajectories might reduce estimation error.
-  Ablation studies test warm-up, data ratio, and novelty parameters, giving some view into design sensitivity.
- The scaling analysis is useful for highlighting diminishing returns from excessive suboptimal data.

### Weaknesses
**Writing**
- The reviewer thinks that writing quality is usbpar for ICLR submission.
    - Several sentences are tautological, e.g., *We have proposed extensive experiments to thoroughly evaluate the effectiveness and robustness of our proposed algorithms.* 
    - Several sentences are verbose and unconvincing, e.g., *This insight is inspired by the development of large-scale language models. The remarkable performance of GPT-4 and DeepSeek-R1 stems not from carefully filtered, perfect corpora, but from massive and heterogeneous datasets.*
    - Logical flow between theory (Sec. 4.5–4.6) and method (Sec. 5) is abrupt.
    - Related works section seems dated. Please discuss and survey recent heterogeneous/offline RL methods.
- Some figures lack clear legends, scales, or quantitative interpretation, reducing clarity.

---
**Methodology**
- The reviewer thinks that the proposed solution is incremental relative to HIQL. Essentially, introduces an extra scalar gating function over two separately trained value networks.
- There is no empirical or theoretical validation that the learned novelty network $R_\psi$ approximates the analytic optimal weighting derived earlier.
- The loss definition in Eq. (15) with fixed hyperparameters $A^+, A^-$ is heuristic and lacks adaptive justification.

---
**Theory**
- The derivations are shallow and partially inconsistent with the later implementation. Constants $c_1, c_2, d_1, d_2$ are assumed environment-invariant without evidence, and their empirical interpretability is unexamined.
- The reviewer thinks that the theory to practice transition (from Eq. 11 to Eq. 16) is abrupt. There is no quantitative connection between analytic and learned weights.

---
**Experiments**
- No analysis of alternative fusion methods (nonlinear, uniform, hard assignment) is provided, so the benefit of the proposed linear weighting remains unsubstantiated.
- All results lack standard deviations, error bars, or significance tests.
- The novelty network’s learned behavior is never visualized or quantitatively analyzed; we do not see evidence that $R_\psi(s)$ correlates with novelty or visitation frequency.

### Questions
- How is the novelty network $R_\psi(s)$ validated in practice? Can the authors show visualizations of its predictions over the state space and their correlation with visitation frequencies?
- Are constants $c_1,c_2,d_1,d_2$ from the analytic derivation estimated or tuned empirically, and how sensitive is the model to their assumptions?
- Would nonlinear or uniform mixture strategies yield similar improvements?
- How does the algorithm behave when scaling suboptimal data in large-scale or realistic domains, does performance degrade predictably?
- Can the approach handle other forms of suboptimality, such as sensor corruption or occlusion, rather than stochastic action noise?
- How are $A^+, A^-$ selected across tasks, and do they require domain-specific tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a method for leveraging suboptimal trajectories as a useful learning signal in offline reinforcement learning. While it is widely recognized that suboptimal data can contribute valuable information when integrated appropriately, the challenge lies in how to effectively combine it with high-quality behavior data.

To address this, the authors introduce an approach that trains separate value functions over two datasets: one representing high-quality behavior and another derived from exploratory, suboptimal trajectories. A novelty score estimation network is employed to estimate whether a given state originates from the high- or low-quality dataset. Based on this estimation, the method constructs a novelty ratio–weighted mixture of the two value functions. This weighted integration allows for good results on the DM control suite.

### Strengths
- the paper is very clear
- the paper is easy to follow
- the presentation and the toy example help understanding

### Weaknesses
It is unclear why the paper focuses exclusively on offline goal-conditioned RL, rather than situating the work within the broader offline RL literature. Expanding the discussion to include related methods in general offline RL would help clarify the broader relevance of the proposed approach.

The claim of achieving a new state-of-the-art algorithm is not sufficiently supported by the current set of experiments. To substantiate this claim, evaluation on established benchmarks—such as AntMaze, Kitchen, CALVIN, Procgen Maze, Visual AntMaze, and Roboverse—following HIQL evaluation protocols would be necessary.

In particular, Figure 6, subfigures (c–f) do not show a clear relationship between data mix and performance, suggesting that optimal data mixing may not be the key factor influencing results (for most tasks). This point warrants further clarification and possibly additional analysis.

Several of the claims may depend heavily on the choice and quality of the non-expert data used. A more rigorous discussion or ablation study examining the sensitivity of results to different non-expert data sources would strengthen the manuscript

### Questions
Please address the weaknesses 1 - 4 above.

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
The paper introduces DREAMEXPLORATIONS, a method for offline goal-conditioned reinforcement learning (OGCRL) that addresses policy overfitting by leveraging suboptimal and noisy trajectories, which are typically discarded by traditional methods, to achieve better state exploration and generalization. The core mechanism involves decoupled value learning, where separate value functions are trained for exploitation (using high-quality data) and exploration (using low-quality data). These separate functions are then combined using a novelty estimation network that predicts a blending ratio, dictating the optimal linear mixing of the two value signals to guide policy learning. This strategy successfully utilizes imperfect data as an asset, demonstrating consistent performance improvements over baselines, particularly in environments requiring extensive state coverage.

### Strengths
A key strength of the paper lies in its innovative reframing of suboptimal data as an asset. The work addresses a critical limitation in Offline Goal-Conditioned Reinforcement Learning (OGCRL) by demonstrating how highly noisy or suboptimal datasets, which prior research often disregards as “useless,” can instead be effectively leveraged. By harnessing this imperfect data, the approach promotes more diverse state exploration and contributes to improved robustness and generalization of the learned policy. This perspective not only challenges conventional assumptions in offline RL but also opens new avenues for utilizing previously overlooked data sources.

### Weaknesses
W1 

The assumption that access to data from the optimal policy is available appears quite strong and may limit the practical applicability of the proposed approach. It would be helpful if the authors could discuss how the method performs when only suboptimal or noisy data is available.

W2

The idea of leveraging lower-quality data closely resembles that in [1] CCLF: A Contrastive-Curiosity-Driven Learning Framework for Sample-Efficient Reinforcement Learning. It would strengthen the paper to include experimental comparisons with this work to clarify the relative advantages of the proposed approach.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
