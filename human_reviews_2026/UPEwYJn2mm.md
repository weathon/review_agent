# Wavelet Predictive Representations for Non-Stationary Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
The real world is inherently non-stationary, with ever-changing factors, such as weather conditions and traffic flows, making it challenging for agents to adapt to varying environmental dynamics. Non-Stationary Reinforcement Learning (NSRL) addresses this challenge by training agents to adapt rapidly to sequences of distinct Markov Decision Processes (MDPs). However, existing NSRL approaches often focus on tasks with regularly evolving patterns, leading to limited adaptability in highly dynamic settings. Inspired by the success of Wavelet analysis in time series modeling, specifically its ability to capture signal trends at multiple scales, we propose WISDOM to leverage wavelet-domain predictive task representations to enhance NSRL. WISDOM captures these multi-scale features in evolving MDP sequences by transforming task representation sequences into the wavelet domain, where wavelet coefficients represent both global trends and fine-grained variations of non-stationary changes. In addition to the auto-regressive modeling commonly employed in time series forecasting, we devise a wavelet temporal difference (TD) update operator to enhance tracking and prediction of MDP evolution. We theoretically prove the convergence of this operator and demonstrate policy improvement with wavelet task representations. Experiments on diverse benchmarks show that WISDOM significantly outperforms existing baselines in both sample efficiency and asymptotic performance, demonstrating its remarkable adaptability in complex environments characterized by non-stationary and stochastically evolving tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address an important problem: the non-stationary environment in deep reinforcement learning.
To this end, this paper proposes WISDOM to leverage wavelet-domain predictive task representations to enhance non-stationary RL.
Experiments show the effectiveness of the proposed method.

### Strengths
1. The paper is overall well-organized and easy to follow.
2. The non-stationary environment in deep reinforcement learning is an important and interesting problem.
3. The authors provide sufficient theoretical support, and the proposed method shows decent performance in numerical experiments.

### Weaknesses
1. Compared to existing methods, the proposed method seems to perform decently in the numerical experiments, but the improvement is not that significant, especially for the final performance in Fig.3. Without additional analysis on the quality of the learned representation, it is not clear if the performance benefits indeed come from the proposed wavelet-domain representation.

2. I think the major contribution of this work is the proposed new representation in NSRL. Thus, it would be good to discuss the metrics that can directly evaluate the "quality" or "informativeness" of the learned representations. Such metrics might include Centered Kernel Alignment and Mutual Information Neural Estimation.

3. The time cost of different methods should be given.

### Questions
1. Can you connect your work with the non-stationary policy methods?

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
3

### Summary
This paper proposes WISDOM, a method for Non-Stationary Reinforcement Learning (NSRL) that leverages wavelet-domain predictive task representations. The key idea is to perform a discrete wavelet transform (DWT) on latent task representations to capture both low-frequency trends and high-frequency variations in evolving MDPs. The authors introduce a wavelet temporal-difference update operator, theoretically prove its contraction property and convergence, and integrate it into a meta-RL framework for improved adaptability to dynamic environments. Experiments on Meta-World, MuJoCo, and Type-1 Diabetes benchmarks demonstrate improved sample efficiency and final performance compared to strong baselines such as PEARL, SeCBAD, CEMRL, and COREP.

### Strengths
1.The paper presents a creative use of wavelet decomposition for temporal abstraction in non-stationary task representation. Unlike conventional Fourier- or RNN-based modeling, the proposed approach explicitly encodes multi-scale temporal information.

2.The authors rigorously define the wavelet TD operator and prove its contraction mapping property (Theorem 1), establishing convergence guarantees rarely seen in representation-learning-based NSRL methods.

3.The appendices include detailed analyses on the effects of decomposition levels, TD loss coefficients, and architectural components, which substantiate the design choices.

### Weaknesses
1.While wavelet theory is well-motivated for time-series decomposition, the link between wavelet coefficients and latent task dynamics could be more clearly articulated.

2.The paper does not discuss the additional computational cost introduced by recursive wavelet decomposition and reconstruction. An empirical runtime comparison (e.g., training time per step or GPU hours) would make the work more practically relevant.

3.Although WISDOM outperforms baselines on the six reported Meta-World tasks, the evaluation scope remains narrow. Meta-World environments vary significantly in difficulty (e.g., reach vs. door-open vs. hammer) [1], and the paper only tests on relatively moderate-difficulty tasks. It remains unclear how WISDOM performs in harder manipulation environments or more complex, partially observable settings. Expanding the evaluation to a broader range of task difficulties would help establish the robustness and applicability of the proposed approach.

[1] Seo Y, Hafner D, Liu H, et al. Masked world models for visual control[C]//Conference on Robot Learning. PMLR, 2023: 1332-1344.

### Questions
1.How sensitive is WISDOM to the choice of the wavelet basis?

2.Does the wavelet TD update improve stability when the reward signal is extremely sparse, or mainly when the task transition distribution drifts?

3.Could the proposed representation be integrated with model-based RL frameworks (e.g. world model-based RL) to further leverage the predictive nature of wavelet transforms?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel reinforcement learning (RL) method for non-stationary domains. In particular, it is considered the setting in which the agent interacts with a sequence of MPDs, and the evolution of such MPDs is determined by a history-dependent stochastic process. To tackle this setting, the authors propose to track task  evolution via a Wavelet representation network. The proposed method, named WISDOM, is based on a Soft Actor-Critic (SAC) agent that also learns a wavelet representation network, which predicts a vector representation of the current task, and is optimized via a TD-learning loss combined with an auto-regressive loss function. The paper introduces two theorems that characterize policy performance with respect to wavelet-domain features. The experiments considered Meta-World, Type-1 Diabetes, and MuJoCo benchmarks, and compared the proposed approach with many competing methods in non-stationary RL.

### Strengths
- The idea of using a Wavelet representation to track non-stationary in RL is interesting and novel.
- The experiments are extensive, considered challenging tasks, and have evaluated many competing methods. Moreover,  the authors provided the source code for reproducing the experiments.

### Weaknesses
- The paper can benefit from a more rigorous/precise mathematical notation in some parts of the text (see Questions below).
- The description and discussion of Theorem 3 require significant improvements in terms of clarity. For instance, the variable $J_{WISDOM}$ is not defined, nor is it clearer what is meant by $its iterated history policy$.

### Questions
- Section 3.2 would benefit from more details on the problem formulation. In particular, I suggest discussing the intuition behind the formulation in Eq. 1. For instance, $\mathcal{T}$ is a set of tasks, and $p(\mathcal{T})$ is a distribution of tasks. Then, the authors introduce $p(\mathcal{T_t})$, whose definition is not very clear. In Eq. 1., it is not clear how an expected value over $\mathcal{T}_t)$ relates to the expected value over $\omega_h$. Providing an example of a real-world scenario where Eq. 1 makes sense would improve the clarity of this section.

- In line 257, $\Gamma$ is not defined. Why does this operator not have an expected value over $z_{t+1}$? Moreover, why does this operator not depend on the policy being followed?

- How does the policy being followed impact the Wavelet representations? How does the model differ from non-stationarity of the task, to non-stationarity caused by policy being updated and changing

- “Stabilize the training and is updated separately and softly.” Please define what is meant by “softly”. Did you mean with polyak averaging?

- In Theorems 2 and 3, the variable $J_\pi$ was not defined. Is it the performance in a single episode? An average over the entire space of tasks? Please be precise.

- There is also a class of non-stationary RL methods that track context changes in the MDP, which could have been at least discussed in the related work. See [1-3].

[1] Silva, Bruno C. da, Eduardo W. Basso, Ana L. C. Bazzan, and Paulo M. Engel. ‘Dealing with Non-Stationary Environments Using Context Detection’. Proceedings of the 23rd International Conference on Machine Learning, 2006.

[2] Alegre, Lucas N., Ana L. C. Bazzan, and Bruno C. da Silva. ‘Minimum-Delay Adaptation in Non-Stationary Reinforcement Learning via Online High-Confidence Change-Point Detection’. Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems, 2021.

[3] Feng, Fan, Biwei Huang, Kun Zhang, and Sara Magliacane. ‘Factored Adaptation for Non-Stationary Reinforcement Learning’. Advances in Neural Information Processing Systems 35, 2022.

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
This paper introduces WISDOM, a context-based non-stationary RL that models sequences of latent task representations in the wavelet domain. A learnable wavelet representation network, composed of two dilated causal convolution layers functioning as adaptive low- and high-pass filters, decomposes context embeddings into approximation and detail coefficients, selectively downsamples detail terms, and reconstructs an enhanced representation used by the policy and critic. 
Training jointly applies a contraction-guaranteed wavelet TD operator and an autoregressive likelihood objective. The theoretical analysis shows that wavelet-domain features can bound policy performance differences and that reconstructed representations improve contextual policy learning under certain assumptions.

### Strengths
- The paper evaluates non-stationary RL scenarios across Meta-World, MuJoCo, and Type-1 Diabetes environments. The inclusion of multiple benchmarks and difficulty levels provides a strong empirical basis for assessing adaptability and generalization. 

- WISDOM shows particularly superior performance on Meta-World, while achieving comparable or better results in MuJoCo and other environments, supporting the claimed contributions.

- The paper is easy to read and follow.

### Weaknesses
- Discussion is lacking on how the proposed wavelet-based approach for tracking fast and slow changes connects to prior methods. This omission makes the contribution somewhat unclear; it is not well explained why alternative approaches could not be applied or how the proposed one provides more than an implementation variant.

- Among the baselines, COREP also models non-stationarity by disentangling stable and varying causal factors through a dual-graph structure, conceptually related to WISDOM's multi-scale decomposition. A clearer discussion on why WISDOM's frequency-based modeling achieves stronger results than COREP would further strengthen the paper.

- To further enhance reproducibility, it would be helpful to include additional details on how hyperparameter tuning was performed for each baseline, particularly in environments not examined in their original studies. In the same context, offering clarification on why CEMRL exhibits lower performance on Meta-World compared to other methods would further enrich the empirical analysis and provide valuable insights for replication and interpretation.

### Questions
- There is an interesting sharp performance increase pattern around 4e5 time steps in Figure 3 (Meta-World experiments) for WISDOM (except Plate-Slide-Back). Likewise, in Figure 5 (MuJoCo experiments) around 0.2 million steps, all baselines show a sudden performance improvement. What factors cause these abrupt jumps in performance?

### Soundness
3

### Presentation
3

### Contribution
3
