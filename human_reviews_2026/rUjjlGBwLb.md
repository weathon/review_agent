# Yes, Q-learning Helps Offline In-Context RL

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Existing offline in-context reinforcement learning (ICRL) methods have predominantly relied on supervised training objectives, which are known to have limitations in offline RL settings. In this study, we explore the integration of RL objectives within an offline ICRL framework. Through experiments on more than 150 GridWorld and MuJoCo environment-derived datasets, we demonstrate that optimizing RL objectives directly improves performance by approximately 30% on average compared to widely adopted Algorithm Distillation (AD), across various dataset coverages, structures, expertise levels, and environmental complexities. Furthermore, in the challenging XLand-MiniGrid environment, RL objectives doubled the performance of AD. Our results also reveal that the addition of conservatism during value learning brings additional improvements in almost all settings tested. Our findings emphasize the importance of aligning ICRL learning objectives with the RL reward-maximization goal, and demonstrate that offline RL is a promising direction for advancing ICRL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper demonstrates that incorporating a reinforcement learning (RL) objective into offline in-context reinforcement learning (ICRL) leads to significant performance improvements. Furthermore, the authors show through extensive experiments that their approach consistently outperforms previous ICRL studies across diverse environments and dataset configurations.

### Strengths
* The paper is clearly written and easy to follow.
* It presents experimental results on a wide range of environments, including both discrete and continuous control tasks.
* The proposed method shows robustness across various dataset variations, suggesting strong applicability in practical settings.

### Weaknesses
* The paper’s main contribution lies in showing that existing RL algorithms can be effectively adapted within the ICRL framework. While the approach achieves strong empirical results, it mainly builds upon established RL algorithms rather than proposing a new method specifically tailored for ICRL.
* Although the paper evaluates multiple RL algorithms and concludes that RL-based objectives outperform prior approaches, it provides little analysis on why certain algorithms perform better or worse under specific conditions. A discussion on the characteristics and trade-offs of each algorithm would enhance the paper's insight.
* Figure 2 is difficult to interpret. According to the diagram, $a_t$ appears to be derived directly from $c_t$. Still, as I understand, in the discrete case, the action is selected via an argmax over the value function, whereas in the continuous case, it is produced by a policy head. The current figure seems to mix both formulations ambiguously.
* Figure 3 occupies a large space but conveys an unclear message. The upper part of the figure, in particular, shows nearly identical trends across all episodes, which makes it difficult to discern the specific insight or conclusion the figure intends to highlight.
* The sentence at line 165 is duplicated.

### Questions
* Is there a specific reason for using the NAUC metric? The reported results seem consistent with other metrics, so presenting multiple metrics in a table might be more informative.
* Why does CQL achieve the best test performance in Figure 3? Since this work focuses on ICRL, it would be helpful to explain why models perform similarly during training but diverge at test time.
* Similarly, can the authors provide an explanation for the following patterns: why AD performs best at DR9 with 20 and 40 in Figure 4, why IC-DQN performs uniquely well at DR19, and why AD outperforms IC-DQN at K2D13 in Figure 6? Clarifying these results would strengthen the empirical analysis.

### Soundness
3

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
4

### Summary
This paper explores **integrating RL objectives to offline ICRL** framework, namely Algorithm Distillation (AD) [1], and investigate whether it can achieve significantly better results in offline ICRL in various axes (coverage, expertise, learning history, continuous state/action).
Experimental results show that additionally optimizing RL objectives improves performance compared to standard AD.
Specifically, combining AD with RL objectives enhances data efficiency in both discrete and continuous domains, and robustness to various axes (e.g. coverage and expertise of the dataset, number of target repetition)

[1] Michael Laskin et al., “In-context Reinforcement Learning with Algorithm Distillation”, ICLR 2023

### Strengths
**S1. Clear organization and presentation**
* The paper is well structured, allowing the reader to easily follow the main argument and its supporting evidence.
* Experimental results are presented clearly, and the setup is well designed to validate the central claim that “RL objectives improve AD.”

**S2. Useful insights in offline ICRL**
* Through its experimental findings, the paper provides valuable insights into the role of offline RL objectives in ICRL.
* For example, it highlights the benefits of using offline rather than online RL objectives, and demonstrates improved data efficiency.

### Weaknesses
**W1. (Slightly) limited novelty**
* The methodological novelty of the work is somewhat limited, as prior studies have explored similarly applying RL objectives in online ICRL settings, namely AMAGO-2 [2], ReLIC [3] (note that AMAGO-2 also employs Transformers in a similar context).
* That said, I view this as a moderate rather than critical weakness; offline RL setting poses unique challenges, the paper’s analytical perspective and insights remain valuable even if the methodological novelty is limited.

**W2. Scalability claims**
* While the authors claims “this is the first work to apply RL objectives in an offline ICRL setting with a **scalable Transformer architecture**” for the use of a Transformer architecture, the experiments are limited to small-scale datasets. 
* Including experiments on larger-scale datasets (e.g., the full xland-minigrid dataset) or with larger Transformer models would provide stronger evidence for the claimed scalability.

[2] Jake Grigsby et al., “AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers”, NeurIPS 2024
[3] Ahmad Elawady et al., "ReLIC: A Recipe for 64k Steps of In-Context Reinforcement Learning for Embodied AI", 2025.

### Questions
**Q1. Large-scale experiments.**
* Have the observed improvements been evaluated on large-scale datasets (e.g., the full xland-minigrid dataset) or with higher-capacity Transformer architectures (i.e., models with more parameters)?
* Such experiments could provide deeper insights into the role and importance of the Transformer architecture.

**Q2. Pixel-based environments**
* Do the proposed methods and conclusions generalize to pixel-based environments (e.g., DMLab Watermaze from the AD paper)?
* While not strictly necessary, demonstrating this would substantially strengthen the generality and robustness of the claims.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the gap between supervised training objectives and RL objectives within offline In-Context RL. The authors compare Algorithm Distillation (AD) with several Q learning methods (DQN, CQL, IQL) in discrete environments, and TD3 in continuous ones. The experiments covers a wide range of settings, such as varying dataset coverage, expertise levels of the offline data, presence or absence of learning-history structure, and both discrete and continous environments. The study also introduces a new environment, XLand-MiniGrid, to further test generalization. The results show that RL objective is substantially better than AD on average, especially under more challenging scenarios, e.g., limited learning history per target, low quality datasets, or randomized learning structures. All the findings demonstrate that aligning the ICRL training objective with the true RL objective is both necessary and beneficial.

### Strengths
1. The problem is very clear: in Offline-ICLR, supervised training objective does not optimize the return directly, making it hard to learn in the suboptimal trajectories, or keeping robust in low quality datasets. This paper adopts a unified backbone and compares AD with various RL objectives in its experiments. The conclusion that RL consistently outperforms AD is straightforward and significant.
1. The experiments study a wide range of settings including training targets * number of histories, expert level of offline data, with/without learning history structure, discrete/continuous action space and a new environment. All these findings make the conclusion strong and widely fitted.
1. The supplementary material provides sufficient details about the experiments for both analysis and reproduction.

### Weaknesses
1. This paper is more like an experimental report than a scientific research paper. It systematically studies the RL objectives and supervised objective in a unified AD backbone, but it lacks algorithmic design on how to bridge the two objectives or theoretical insights on why RL-based objectives yield better generalization and robustness in offline in-context settings. 
1. This paper primarily provides empirical observations but lacks in depth analysis of some important findings. For example, the reason why both offline RL and AD outperform online TD3 in continuous environments is rather important but not discussed. Typically, online algorithm has greater potential to explore the environment, whereas offline RL is more prone to overestimate unseen states and actions. The results are somewhat counterintuitive. Therefore, a deeper investigation into why offline methods perform better in this setting would significantly strengthen the paper.
1. AD alone is not a sufficient baseline for this study. A more comprehensive evaluation should include stronger supervised objectives such as the Decision Transformer (DT), which conditions on return-to-go and represents a more goal-directed form of supervised sequence modeling.

### Questions
1. Is it possible to add DT and AMAGO as baselines in a few representative environments to compare against RL objective?
1. Could the authors provide other evaluation metrics, for example, final return, average success rate of last K episodes? This will help avoid single-metric bias.

### Soundness
2

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
This paper empirically investigates whether incorporating reinforcement learning (RL) objectives into offline in-context reinforcement learning (ICRL) improves performance compared to purely supervised objectives such as Algorithm Distillation (AD). The proposed approach replaces the next-action prediction head of AD with RL-based value and policy heads. Extensive experiments are conducted on discrete (Dark Room, Key-to-Door) and continuous (MuJoCo) environments, and results show that optimizing explicit RL objectives substantially improves both performance and robustness compared to AD.

### Strengths
S1. (Broad empirical evaluation)
This work provides a broad empirical comparison between AD and RL-based ICRL methods (IC-DQN, IC-CQL, IC-IQL, IC-TD3(+BC)) across a wide variety of datasets, including discrete and continuous action spaces. The analyses cover factors such as data coverage, expertise levels, and unordered histories.

S2. (Robustness and Generalization)
Empirical results consistently demonstrate that RL-based optimization yields higher NAUC scores, improved robustness to limited data coverage, and better generalization to unseen tasks. In particular, the integration of conservative or regularized RL methods provides stable improvements across dataset qualities and structures.

S3. (Extension to continuous domains)
Experiments on the MuJoCo environments show that offline RL–based ICRL maintains performance advantages even in continuous spaces, supporting the generality of the approach beyond discrete benchmarks.

### Weaknesses
W1. (Limited novelty)
Although the proposed RL-based ICRL framework is empirically valuable, its architectural novelty is limited. The main contribution seems to involve replacing the supervised objective of AD with standard RL objectives, while adopting several design choices already introduced in prior works (e.g., AMAGO), such as incorporating done flags and timestep embeddings in the context representation.

W2. (Simplified continuous policy evaluation)
Experiments in continuous domains rely solely on deterministic policies, leaving unclear how stochastic policies, which are widely-used in offline RL, would perform under the setting.

W3. (Lack of ablation on gradient detachment)
Gradients from the policy head are detached from the backbone to stabilize training, but no ablation is provided to justify this design choice or quantify its impact.

W4. (Incomplete comparison to Meta RL works)
The experimental evaluation focuses exclusively on comparisons with Algorithm Distillation (AD). Including results against other offline meta-RL baselines (e.g., FOCAL[1], BOReL[2], Meta-DT[3]) would help determine whether simply substituting an offline RL objective within a Transformer architecture is sufficient, or if additional meta-learning mechanisms are required for performance and generalization.


[1] Li, Lanqing, Rui Yang, and Dijun Luo. "FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization." International Conference on Learning Representations.

[2] Dorfman, Ron, Idan Shenfeld, and Aviv Tamar. "Offline Meta Reinforcement Learning--Identifiability Challenges and Effective Data Collection Strategies." Advances in Neural Information Processing Systems 34 (2021): 4607-4618.

[3] Wang, Zhi, et al. "Meta-DT: Offline meta-RL as conditional sequence modeling with world model disentanglement." Advances in Neural Information Processing Systems 37 (2024): 44845-44870.

### Questions
Could you provide clarifications on the weaknesses?

### Soundness
3

### Presentation
4

### Contribution
2
