# Offline Reinforcement Learning with Adaptive Feature Fusion

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Return-conditioned supervised learning (RCSL) algorithms have demonstrated strong generative capabilities in offline reinforcement learning (RL) by learning action distributions based on both the state and the return. However, many existing approaches treat RL as a conditional sequence modeling task, which can lead to an overreliance on suboptimal past experiences, impairing decision-making and reducing the effectiveness of trajectory synthesis. To address these limitations, we propose a novel approach, the Q-Augmented Dual-Feature Fusion Decision Transformer (QDFFDT), which adaptively combines both global sequence features and local immediate features through a learnable fusion mechanism. This model improves generalization across different tasks without the need for extensive hyperparameter tuning. Experimental results on the D4RL benchmark show that QDFFDT outperforms current methods, establishing new state-of-the-art performance and demonstrating the power of adaptive feature fusion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces QDFFDT and DFFDT, novel offline RL models that mitigates the limitations of purely sequence-based RCSL. DFFDT fuses global sequential and local immediate features through a learnable fusion weight, enabling effective balance between long-term context and Markovian decision signals. QDFFDT further integrates Q-learning to enhance value estimation and policy improvement. Experiments on the D4RL benchmark demonstrate that QDFFDT achieves SOTA across diverse domains.

### Strengths
The paper presents a clear illustration of the limitations of prior return-conditioned sequence models and introduces an adaptive transformer architecture that empirically achieves strong performance across diverse offline RL tasks.

### Weaknesses
The paper’s contribution primarily lies in modifying the architecture of the Decision Transformer for sequential decision modeling. The authors aim to learn a mapping $f(a_t \mid \hat{R}_{\leq t}, s_{\leq t})$ and introduce a dual-feature fusion mechanism to encourage attention to immediate Markovian information. However, this design raises a theoretical concern:

The input information remains unchanged, and a sufficiently expressive Transformer should, in principle, be capable of learning any desired feature weighting — including the adaptive combination $((1-\hat{\alpha}(s_t))h^{loc}_t + \hat{\alpha}(s_t)h^{glo}_t)$ — directly through its self-attention and learned parameters. The additional fusion module, therefore, seems architecturally redundant unless the authors can demonstrate a theoretical advantage.

### Questions
1. I suggest that the authors improve the Abstract and Introduction to define the problem and present their motivation more clearly. For example, in the Abstract, they state that existing approaches “lead to an overreliance on suboptimal past experience,” which is not immediately clear to the reader. In the Introduction, the authors should describe the problem more explicitly in the second paragraph and clarify the method and its rationale in the third paragraph. They may also consider moving Section 4 into the Introduction.

2. It would be helpful to include the Q-component in Figure 3. With this component, the architecture corresponds to QDFFDT; without it, it represents DFFDT.

### Soundness
2

### Presentation
2

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
This paper proposes QDFFDT (Q-Augmented Dual-Feature Fusion Decision Transformer), an offline reinforcement learning framework that adaptively combines global sequence modeling and local single-step (Markovian) modeling through a learnable fusion mechanism. The method extends return-conditioned supervised learning (RCSL) by learning a state-dependent fusion coefficient based on the expectile regression of return-to-go values, balancing the influence of contextual and immediate features. Additionally, a Q-learning module is integrated to improve policy optimization and mitigate the overreliance on suboptimal trajectory patterns common in sequence-based methods like Decision Transformer (DT) and QT. Experiments on the D4RL benchmark across Gym-MuJoCo, Maze2D, AntMaze, Kitchen, and Adroit show that QDFFDT achieves consistent state-of-the-art performance, improving generalization without extensive hyperparameter tuning.

### Strengths
- Well-motivated insight: The paper identifies a real weakness in RCSL methods—overfitting to suboptimal sequence fragments that hinder trajectory stitching—and directly targets it with an adaptive fusion approach.

- Elegant hybrid design: The fusion of sequential and immediate features through a learned state-conditioned weight is conceptually neat and grounded in the Markov property. It’s a balanced architectural innovation rather than an ad hoc ensemble.

- Integration of value learning: The Q-augmentation complements RCSL’s generative bias, yielding both theoretical plausibility and empirical strength.

- Comprehensive experiments: The authors evaluate on five major domains (Gym, Maze2D, AntMaze, Kitchen, Adroit) against more than 15 baselines, consistently outperforming DT, QT, and RCSL variants. Ablations (static vs. adaptive α̂, sequence-only vs. single-step) are insightful and well-documented.

- Clarity and reproducibility: The paper is clearly written, with structured explanations, pseudo-code, and full hyperparameter tables. The method is easy to reimplement.

### Weaknesses
- Limited theoretical contribution: Despite clear empirical validation, the method is largely architectural. The theoretical motivation for how expectile-based weighting improves trajectory stitching remains heuristic.

- Overcomplexity relative to gain: The approach introduces five networks and multiple losses (expectile, α-loss, Q-loss, MSE) for modest algorithmic advancement; simplicity compared to QT or ACT is reduced.

- Incremental relationship to prior work: The approach closely resembles QT (Hu et al., 2024) with an extra fusion branch and expectile weighting. The novelty over these strong baselines feels evolutionary rather than revolutionary.

### Questions
1. How sensitive is QDFFDT to the expectile level σ and the temperature parameter T in α-learning?

2. Does α̂(s) exhibit meaningful variation across states or converge to near-constant values? Can this be visualized?

3. What is the runtime overhead compared to QT or DT in wall-clock training?

4. Would replacing the Transformer with a lightweight recurrent or convolutional encoder yield similar benefits?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper combines long-term sequential modeling and local immediate modeling via an adaptive feature fusion mechanism for efficient offline return-conditioned supervised learning (RCSL). Previous pure sequence modeling methods tend to suffer from replicating suboptimal trajectories in the training dataset, and additional Q-value guidance cannot eliminate this drawback. By estimating an approximated max-return for each state, the proposed method DFFDT can adaptively rely more on the local one-step action prediction when suboptimal RTG signals are observed (after comparing them with the estimated V-values). With the aid of the Q value, the variant QDFFDT achieves superior performance compared to strong baselines on D4RL benchmarks and is claimed to be flexible in balancing the use of global and local features.

### Strengths
- The paper is well-written and easy to follow. Figure 3 efficiently captures the main idea of this paper. Notations are clean and consistent.
- The core component, feature fusion, is well-motivated through a necessary didactic example in Section 4 and demonstrates that the Q-value guidance alone is insufficient for counteracting the “momentum”  of replicating the suboptimal trajectories.
- QDFFDT performs well on D4RL benchmarks with an acceptable increase in training complexity (as shown in Appendix E).
- Appendix includes enough experimental details for reproduction purposes.

### Weaknesses
- **In terms of the didactic example,** why does QT perform worse when $\eta=10.0$ than when $\eta=1.0$? Increasing $\eta$ gradually eclipses the effect of the DT loss and enables QT to choose actions with the highest Q values (which is true for $\eta=0.01-1.0$). Intuitively, setting $\eta$ to a very large number should completely erase the effect of the BC term, and thus QT should be able to achieve a higher success rate.
- In my opinion, the core contribution of this paper is the content in Sections 4 and equations (3) and (5), which trains an Alpha network to adaptively trade off between global sequential features and local immediate features. **However, I have the following concerns about the effectiveness of feature fusion, despite QDFFDT’s strong performance shown in Table 2:**
    1. In terms of the training objective of the Alpha network, a naive solution to minimize it could predict $\alpha_\omega (s) = 0 $ for most of the state. And when $V_\psi (s) \leq \hat{R}$, there is no gradient signal to control the training behavior. I suspect that this adaptive mechanism would eventually bias towards learning from local features more than from global features. This suspicion is amplified, especially when the authors have to manually set $\hat{\alpha}=1$ for 3 of the 4 evaluated Atari games to ensure the model fully relies on the sequence modelling (since it benefits largely for these partially observed environments). Moreover, in Appendix D.3, it seems to be true that the learned $\alpha$ is small in general. Could the authors comment on this? What is the average learned $\alpha_\omega(s)$ for these datasets?
    2. In terms of the ablation study, Figure 4 gives me a feeling that (1) for environments with the Markovian property, local modeling alone is enough; and (2) for partially observed environments, global modeling alone is enough. The performance gain of feature fusion is not as prominent as in Table 2, if we take the variance into consideration. Given the question (a) above, it makes sense for (Q)DFFDT working relatively well for Gym and Antmaze, since it might predict low $\alpha$ most of the time. And what is the performance for Atari, if $\hat{\alpha}$ is not manually set?
    3. In summary, I feel that it is a bit far-fetched to conclude that “These results highlight the effectiveness of the proposed adaptive fusion mechanism in combining global and local information without the need for extensive hyperparameter tuning.” (Lines 473-475). The authors should provide more evidence to defend this core argument.
- **Some experimental details need more clarification:**
    1. In Table 2, why does QT perform poorly for antmaze-l, while the reimplementation of QT, QRC-Transformer, performs much better in Figure 4(b)?
    2. I am aware that most of the offline methods fail on Adroit-door, Adroit-hammer, and Adroit-relocate. However, QT has made some progress on those challenging ones. I wonder how QDFFDT performs in those environments.
    3. For papers claiming SOTA results, it is common for them to tune hyperparameters. However, I found that the authors reduce context length from 20 to 5 for tasks outside the Gym domain, which influences the analysis since this hyperparameter also controls the strength of global feature modeling. If the proposed adaptive feature fusion works well, it should be able to handle longer context for those environments with a Markovian property. How does the performance of QDFFDT change with different context lengths?

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
