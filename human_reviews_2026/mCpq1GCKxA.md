# Simplicial Embeddings Improve Sample Efficiency in Actor–Critic Agents

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients.
When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the use of simplicial embeddings (which lead to sparse and discrete features) as a way to improve the sample efficiency, stability and performance of Reinforcement Learning agents. Furthermore, the paper adds to the body of work showing that non-stationary targets can cause a loss of plasticity, and demonstrates that SEM can mitigate this effect.

### Strengths
This paper presents a simple, practical idea as a quick and effective way to improve sample efficiency in Reinforcement Learning, which can be easily used in almost any algorithm. The analysis on why SEM improves performance is deep and uses a variety of metrics previously proposed in the literature. Furthermore, the evaluation is quite thorough, including different algorithms and environments.

### Weaknesses
My main problem with this paper is the lack of comparison to prior methods that solve very similar problems. This paper correctly cites a wide range of prior work that propose solutions to the problem of degradation of learned representations under non-stationarity ([1, 2, 3] and many more), causing plasticity loss and reduced performance. Despite this large variety of previous similar work, the paper also exclusively compares against baselines with no previous similar solutions. While many figures show improvement, none show whether this is actually an improvement over prior existing work. For this work to be accepted at a venue of this standard, I would expect that Figures 1, 9 and 10 would include a comparison against prior strategies.

Figures 9 and 10 have inconsistent axes. The left and middle plots present very similar data (million of frames on the x axis, and performance on the y axis), yet are presented under different scales. This inconsistency makes the results a little harder to read.

No source code was included in the supplementary material, and there is no mention of source code being released after review period. This hurts the reproducibility of this work.

Figure 10 uses the Atari-10 benchmark [4], but presents results as IQM performance. The purpose of the Atari-10 paper was to predict the **median** performance, using a specific regression procedure provided by the authors (which weights the importance of each game). Given that the paper reports IQM performance, it appears the procedure was not followed.

There is a typo on line 215.

[1] Lyle, Clare, et al. "Disentangling the Causes of Plasticity Loss in Neural Networks." Conference on Lifelong Learning Agents. PMLR, 2025.

[2] Ceron, Johan Samir Obando, Aaron Courville, and Pablo Samuel Castro. "In value-based deep reinforcement learning, a pruned network is a good network." International Conference on Machine Learning. PMLR, 2024.

[3] Willi, Timon, et al. "Mixture of Experts in a Mixture of RL settings." Reinforcement Learning Conference.

[4] Aitchison, Matthew, Penny Sweetser, and Marcus Hutter. "Atari-5: Distilling the arcade learning environment down to five games." International Conference on Machine Learning. PMLR, 2023.

### Questions
How does this method compare against prior plasticity loss mitigation strategies? Does this method provide better performance? Is it better at preventing representation collapse?

It is unclear to me why this paper focuses on actor-critic algorithms specifically. Line 323 provides literature suggesting that non-actor-critic approaches can also benefit from sparsity in representations. Given this, it seems odd to me that the paper is so heavily tied to actor-critic methods (including the title). Why is this the case?

What do the shaded areas on the different figures represent (Standard error, 95% Confidence intervals, etc)? Additionally, how many seeds does Figure 1 use?

I think this paper is valuable to the RL community; however, given the lack of comparison against prior methods, I cannot yet advocate for acceptance. If this issue is remedied, and the method demonstrates superior properties to prior methods, I am willing to raise my score.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies representation collapse as a root cause of poor sample efficiency and training instability in deep RL, arguing that non-stationary, bootstrapped targets make critics ill-conditioned and prone to drifting value estimates. It proposes Simplicial Embeddings (SEM)—a lightweight, group-wise softmax projection that constrains latent features to a product of simplices—to bound features, induce sparsity, and maintain diversity, thereby stabilizing learning. The proposed method is proven to be effective on different base RL algorithms across mainstream benchmarks.

### Strengths
**Clear problem framing.** Ties critic instability to non-stationarity and representation collapse with an accessible toy study.

**Simple yet effective, plug-and-play mechanism:** SEM is like an activation with better gradient properties; no extra losses or estimators. 

**Broad robustness:** Benefits persist across data-limited settings and simplified agent variants; compares against reasonable discrete/structured alternatives.

### Weaknesses
**Inappropriate toy examples.** The CIFAR-10 supervised learning experiment with shuffled labels is not a straightforward toy example for demonstrating the non-stationarities in value learning.

**L, V selection.** It is weird that the limited SEM capacity leads to better performance. Especially the setting that $L=1, V=64$ almost achieves the best performance in Fig. 7 if the y-axes are shared. Doesn't that mean the softmax operator is the best regularizer?

### Questions
1. What's the best setting of L and V? Did you change the values of L and V depending on the benchmark and task?

### Soundness
3

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
3

### Summary
The paper proposes the use of simplicial embeddings (SEM) to improve representation stability in reinforcement learning (RL), particularly under distributional shifts. By incorporating SEM into the penultimate layers of actor and critic networks, the authors report faster learning across several RL environments.

### Strengths
- The authors demonstrate that using simplicial embeddings generally leads to faster or comparable learning speed across multiple environments.
- The paper investigates which components of actor-critic architectures benefit most from SEM, analyzes the impact of hyperparameters, and compares SEM with other representation-learning baselines.

### Weaknesses
- The use of shuffled CIFAR-10 labels as a proxy for distributional shift is questionable. The claim that this setup "mimics the bootstrap dynamics of RL" is conceptually weak: label shuffling introduces discrete, externally imposed changes rather than the gradual shifts seen in RL. It is also unclear what bootstrapping dynamics are in this context.
- In the accompanying Figure, the authors show the loss of classifiers for stationary targets and for label shuffling with and without SEM. Before the first shuffling occurs, the distribution shift without SEM and the stationary variants should behave similarly, but they do not. This discrepancy suggests confounding factors may explain the poorer performance of the shifting variant without SEM.
- The claim that SEM improves critic representations is made before any RL experiments are introduced, which makes this conclusion premature and poorly supported by the evidence presented.
- The Appendix incorrectly describes the "Atari-10" benchmark as including 26 games. In reality, Atari-10 consists of 10 games, while 26-game subsets are part of benchmarks such as Atari-100k.
- Overall, the paper's presentation is the weakest aspect. Clarity and experimental justification need substantial improvement.

### Questions
- How are the groups chosen for normalization within the softmax operation? Is it just connected parts of the latent vectors?
- In Figure 4, the "Critic - Net e-rank" curve continues to rise at 100k steps. What happens beyond this range?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates the use of a particular rank encouraging representation for state embedding in actor-critic reinforcement learning methods. The particular representation used is the simplicial embedding, which has been introduced in prior work which constrains the embedding space to be the concatenation of equal dimension simplexes. This representation can be used as an embedding in the actor, critic, or both. The novel contribution of this work is to show that this previously proposed embedding can improve the performance of actor-critic RL algorithms in terms of a combination of sample efficiency, returns and variance reduction in the learning curves. The paper mostly validates its claims empirically, while providing some possible intuition behind the performance differences observed with the use of simplicity embeddings. The paper does conduct meaningful empirical analysis to do the same, and I feel most of my questions regarding the claims were sufficiently answered by the experiments. I list some remaining concerns in the weaknesses section below.

### Strengths
1. Most importantly, the proposed use of SEMs does seem to show promising performance improvements consistently across a variety of benchmarks and tasks at least for continuous control problems.
2. The SEM formulation is quite simple and should not add significant extra computational cost in actor-critic training, which I appreciate.

### Weaknesses
1. I am unsure about the baseline choices of other representation methods used in Fig. 7 (left). Given that the authors' hypothesis about SEM working well is that it encourages the encoding to have high effective rank, are the methods compared against in Fig. 7 (left) also methods that specifically focus on creating high effective rank? If so, please mention this, if not, then I think other baselines should be used.
2. One baseline I think should have been used for all experiments throughout the paper but has not been used is the case when $V=1$, i.e., instead of a group of $L$ $\Delta^{V-1}$ simplexes, use a single $\Delta^{LV-1}$. While technically this single simplex case is also a type of SEM, I this comparison is needed for the claim of SEMs promoting diversity (lines 182-184) through the group structure.
1. Lines 365-366 (referring to Fig. 7 right)) : `We find that increasing V generally improves performance, but only up to a certain point.` Respectfully, I disagree with this claim. In the two right most plots of Fig 7., we see $V=4$ having significantly worse performance than others *only* when $L=1$. I think this trend is more aptly explained by the low overall capacity here (as $LV=4$). When $L=64$, we see all choices of $V$ performing comparably, with $V=4$ in fact slightly better.

### Questions
Besides the 3 points raised in weaknesses section, please address the following:
1. In the experiments where you are averaging over different tasks (like Fig. 3), were the state dimensions the same?
2. What is the value of $L$ in Fig. 3?

Requested Changes:
1. Given that the dependence of perfomance on representation capacity, please explicitly state all $L$, $V$ values in the figures/captions where not already done. If the state dimensions vary, please mention those as well.
2. Minor typo: line 204: contex -> context.

### Soundness
3

### Presentation
4

### Contribution
3
