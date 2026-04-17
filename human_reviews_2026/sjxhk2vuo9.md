# Long-Horizon Model-Based Offline Reinforcement Learning Without Conservatism

- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Popular offline reinforcement learning (RL) methods rely on *conservatism*, either by penalizing out-of-dataset actions or by restricting planning horizons. In this work, we question the universality of this principle and instead revisit a complementary one: a *Bayesian* perspective. Rather than enforcing conservatism, the Bayesian approach tackles epistemic uncertainty in offline data by modeling a posterior distribution over plausible world models and training a history-dependent agent to maximize expected rewards, enabling test-time generalization. We first illustrate, in a bandit setting, that Bayesianism excels on low-quality datasets where conservatism fails. We then scale the principle to realistic tasks, identifying key design choices, such as layer normalization in the world model and adaptive long-horizon planning, that mitigate compounding error and value overestimation. These yield our practical algorithm, Neubay, grounded in the *neu*tral *Bay*esian principle. On D4RL and NeoRL benchmarks, Neubay generally matches or surpasses leading conservative algorithms, achieving new state-of-the-art on 7 datasets. Notably, it succeeds with planning horizons of several hundred steps, challenging common belief. Finally, we characterize when Neubay is preferable to conservatism, laying the foundation for a new direction in offline and model-based RL.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper questions the dominance of the conservative principle in offline reinforcement learning (RL) and instead explores a Bayesian alternative. The authors propose NEUBAY, a model-based offline RL algorithm derived from a neutral Bayesian (NEUtral BAYesian) perspective that optimizes expected returns over a posterior distribution of world models, rather than enforcing pessimism or short horizons. Key innovations include (1) using layer normalization in world models to control compounding error, (2) adaptive long-horizon planning based on epistemic uncertainty thresholds, and (3) a stabilized recurrent training pipeline for history-dependent agents. NEUBAY is evaluated on D4RL and NeoRL benchmarks across 33 datasets and achieves new state-of-the-art results on 7 datasets, particularly excelling on low-quality or moderate-coverage data (see Table 1; Sec. 5.1).

### Strengths
- The core problem addressed in this paper is how to abandon the reliance on conservatism in offline RL algorithms, which I believe is a very meaningful issue. Conservatism constrains policy optimization within a limited range, inevitably leading to suboptimal policies with poor generalization ability. The proposed NEUBAY method breaks this restriction and demonstrates that even without conservatism, offline RL can still perform well, offering a viable alternative paradigm.
- The paper, through the bandit experiment, clearly demonstrates how the uncertainty penalty harms generalization performance, while Bayesianism can effectively adapt to various cases, thereby validating that the authors’ claim is both correct and reasonable.
- NEUBAY is compared against a wide range of baselines, including numerous recent model-free and model-based offline RL algorithms. Ultimately, NEUBAY achieves outstanding performance on most tasks. Such comprehensive comparisons and significant performance improvements clearly demonstrate the effectiveness of the proposed method.
- The authors designed several ablation studies targeting their design choices, showing the contribution of each component in the method. These results provide strong support for the soundness and rationale of the proposed design choices.

### Weaknesses
- NEUBAY uses a larger ensemble size in the dynamics model compared to previous methods, which seems to introduce additional computational overhead, including increased runtime and GPU memory usage.
- I think the NEUBAY method is overall relatively complex and introduces some new hyperparameters, which seems to make the algorithm more difficult to tune.

### Questions
- Could the authors provide a comparison of NEUBAY with previous methods in terms of time and GPU memory consumption?
- NEUBAY’s actor and critic utilize historical information. How would NEUBAY’s performance change if historical information were not used? If the actors and critics of other methods also used historical information, would NEUBAY’s performance advantage still be significant?”

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes NEUBAY, a Bayesian-perspective algorithm that addresses epistemic uncertainty in offline datasets by modeling a posterior distribution. The key idea builds on the controlling compounding error in epistemic POMDP modeling (increasing the ensemble size and incorporating the layer normalization in the model) and addressing the long-horizon planning. In the experiments, NEUBAY outperforms for several offline datasets, but not for most of the offline datasets.

### Strengths
1. The authors conducted several ablation studies, including variations in ensemble size and truncation threshold, to demonstrate the strengths of the proposed algorithm.

2. Providing comprehensive comparisons against a wide range of offline RL baselines effectively supports the validity of the proposed approach.

### Weaknesses
1. To the best of my knowledge, there exists a method that unrestricts the planning horizon in offline model-based RL. MBOP [1] and MOPP [2] encourage more aggressive trajectory rollouts. In particular, MOPP prunes out problematic trajectories to avoid potential out-of-distribution samples. This approach is highly similar to Algorithm 2 in NEUBAY, as both involve pruning (or truncating) trajectories based on a threshold and performing rollouts without a fixed planning horizon. Overall, the two methods share a significant conceptual overlap.

2. In Figure 5, it is difficult to interpret how the inclusion of layer normalization affects the model’s performance. For example, there are too many colored curves, making it unclear which line corresponds to which experimental setting.

3. In Section 3, the authors emphasize that the Bayesian framework enables the model to perform well even with low-quality data. However, it is unclear whether the method consistently demonstrates superior performance compared to other algorithms on the D4RL MuJoCo random datasets in Table 1.

4. The ensemble size was set to 100. This is similar to the perspective in model-free algorithms such as SAC-N, where a large ensemble contributes to improved performance. Therefore, it would be valuable to demonstrate whether the proposed method (NEUBAY) still performs well when the ensemble size is reduced to a comparable level (e.g., 5 or 7), as done in other algorithms. This would be similar in spirit to how EDAC was motivated by the idea of SAC-N.

5. It is also important to include a comparison of running time with respect to the ensemble size, as increasing the ensemble size can significantly affect computational efficiency.

 
[1] Argenson, Arthur, and Gabriel Dulac-Arnold. "Model-based offline planning." arXiv preprint arXiv:2008.05556 (2020).

[2] Zhan, Xianyuan, Xiangyu Zhu, and Haoran Xu. "Model-based offline planning with trajectory pruning." arXiv preprint arXiv:2105.07351 (2021).

### Questions
The issues discussed under Weakness capture and represent the key questions regarding this paper.

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
This paper proposes NEUBAY, a model-based offline RL algorithm that challenges the dominant principle of conservatism. Instead of using explicit uncertainty penalties, NEUBAY adopts a Bayesian perspective, modeling a posterior over world models and training a history-dependent agent. Its key mechanism is an adaptive planning horizon, where rollouts are truncated based on epistemic uncertainty, allowing for potentially very long planning horizons.

### Strengths
[S1] The key idea of replacing explicit conservative penalties with planning horizon truncation based on epistemic uncertainty is novel and well-motivated. This opens a new line for integrating uncertainty into model-based offline RL.

### Weaknesses
[W1] There is a significant disconnect between the method's justification (enabling long-horizon planning) and its empirical results. The algorithm performs poorly on tasks that genuinely require long-horizon planning and sparse reward handling (e.g., AntMaze large, 0% normalized score).

Conversely, NEUBAY's successes are on benchmarks (e.g., D4RL MuJoCo) that are largely reactive, fully observable and where strong performance is achievable with much shorter horizons (1~10 steps) or even model-free, fully observed MDP methods. While the paper shows NEUBAY uses very long horizons (e.g., 512-1000 steps) on these tasks, this success does not sufficiently demonstrate why such long-horizon planning is necessary, given that simpler methods also perform well.

[W2] The proposed algorithm has high architectural complexity, integrating deep ensembles ($N=100$), LRU-based recurrent encoders,  specific context-handling mechanisms and task-dependent hyperparameters (e.g., real data ratio, encoder learning rate, gradient clipping). The complexity limits the method's reproducibility and practical applications.

### Questions
[Q1] Following W1, are there any tasks that can demonstrate long-horizon planning (e.g., 512-1000 steps)? Ideally, such tasks should be highlighted in the paper.

[Q2] Following W2, can NEUBAY be simplified, emphasizing the adaptive horizon while minimizing its complexity?

### Soundness
3

### Presentation
4

### Contribution
2
