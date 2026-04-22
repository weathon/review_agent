# COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Many alignment methods, including reinforcement learning from human feedback (RLHF), rely on the Bradley-Terry reward assumption, which is not always sufficient to capture the full range and complexity of general human preferences. We explore RLHF under a general preference framework by modeling the alignment problem as a two-player zero-sum game in a game-theoretic framework, where the Nash equilibrium policy guarantees a 50\% win rate against any competing policy.   However, previous self-play algorithms for finding the Nash policy either diverge or only converge to a Nash policy in a modified game, even in a simple synthetic setting, thereby failing to maintain the 50\% win rate guarantee against all other policies. We propose a meta-algorithm, **Co**nvergent **M**eta **Al**ignment Algorithm (COMAL), for language model alignment with general preferences, inspired by convergent algorithms in game theory. We provide theoretical analysis that our meta-algorithm converges to an exact Nash policy in the last iterate and demonstrate its effectiveness on a range of synthetic and preference optimization datasets. COMAL is simple and can be integrated with many existing methods designed for preference optimization with minimal changes, and empirically it consistently maintains above 60.2\% and 56.8\% win rates, when applied to Llama-3-8B-Instruct and Qwen2.5-7B, against all compared algorithms under controlled evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a method of general preference optimization that is guaranteed converging to Nash Equilibrium of the preference optimization game, by simply changing to updating the reference policy. The authors have shown the previous methods suffering from convergence via synthetic examples, while their proposed method COMAL resolves this problem. The experiments are conducted on two base models, which have shown stable training curve and improved win rates.

### Strengths
The manuscript is well-written and every point is extremely clear and well-explained. The authors provided synthetic examples to clearly demonstrate the problems of previous methods. 

This manuscript is meanwhile exhaustive. Basically all existing methods in general preference optimization have been discussed and compared. The analysis and the experiments are thorough.

The most valuable plots to me are in Fig 2, demonstrating the stability of the proposed methods and the instability of the others. The authors manage to conduct longer iterations running and manage to show stability of the training curve.

### Weaknesses
- The main concern is that although the difference from Magnetic Preference Optimization (MPO) [Wang et al.] has been discussed in the paper, the contribution on top of MPO feels small. 

- Another issue is that results on standard benchmarks, which are more important, don't show significant improvement. I'm not sure if it is due to not using a preference model such as [3] directly. Another notable point is that INPO unlike SPPO didn't apply pairwise comparison to estimate the preference and conduct mirror descent. Anyways, I think the limited improvement weaken the manuscript.

The paper overall is very good. It's just the two concerns above bothering me. I'm happy to increase the rating if the concerns are addressed (at least to some extent).

Minor:
 - Missing Related Work [1,2]

[1] Pásztor, B., Buening, T. K., & Krause, A. Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game. In Eighteenth European Workshop on Reinforcement Learning.
[2] Tang, X., Yoon, S., Son, S., Yuan, H., Gu, Q., & Bogunovic, I. (2025). Game-Theoretic Regularized Self-Play Alignment of Large Language Models. arXiv preprint arXiv:2503.00030.
[3] Zhang, Y., Zhang, G., Wu, Y., Xu, K., & Gu, Q. (2024). General preference modeling with preference representations for aligning language models.

### Questions
I agree that the original Nash Equilibrium should be the first goal of solving a game. But in practice, especially in the complex environment of preference optimization, removing regularization (or updating reference model) is better or not is unclear. At least according to INPO, keeping the regularization leads to significant improvement on standard benchmarks.

### Soundness
3

### Presentation
4

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
This paper proposes COMAL, a meta-algorithm for aligning language models with general preferences. COMAL models the alignment problem as a two-player zero-sum game and converges to a robust Nash equilibrium policy. It integrates easily with existing methods and shows significant performance gains in experiments compared to prior methods.

### Strengths
1 COMAL demonstrates robust performance across different model sizes, including 8B, 7B, and 1.5B models, highlighting its scalability and practical applicability.
2 COMAL can be easily integrated into existing preference optimization algorithms with minimal changes.
3 COMAL is the first algorithm with a provable guarantee of last-iterate convergence to the unregularized Nash equilibrium.

### Weaknesses
1 The approach only trains on a single preference oracle. This narrow training setup raises concerns about COMAL’s ability to generalize. The Arena Hard results suggest this limitation, as COMAL underperforms Iter-IPO under GPT-4 evaluation, indicating potential overfitting to the specific training oracle. The paper would be strengthened by training on heterogeneous preference sources and demonstrating robust performance across current evaluation setting.
2 The paper strongly motivates the work by arguing that BT models cannot capture intransitive preferences, framing this as a key limitation to address. However, the experiments use a mixture of two BT models (Eq. 4) as the preference oracle without verifying whether this mixture actually exhibits the claimed intransitivity. Although synthetic experiments show that the method works on hand-crafted cyclic preferences, it remains unclear whether the preference data contains meaningful non-transitive structure. Providing evidence of intransitivity in the experimental setup or validating the method on datasets with verified cyclic preferences would better support the paper’s core claims.
3 The core contribution of the paper lies in applying the prox-method to LLM alignment via adaptive reference policy updates. While this is a significant application of an existing method, it would be helpful to clarify how this approach introduces fundamentally new ideas or significantly extends existing techniques.

### Questions
1 Could the authors clarify why the initial win rate in Figure 2 is not around 50% as expected? Does “iteration 0” correspond to the BASE model before training or to an updated model after the first optimization step?
2 Can the authors give detailed comparision between COMAL and the concurrent work by Wang et al[1], and clarify the advantages of COMAL in terms of alignment applications?
[1] Wang, Mingzhi, et al. "Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Model Alignment." The Thirteenth International Conference on Learning Representations.

### Soundness
2

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
This paper proposes, a meta-algorithm designed to LLMs with general human preferences that may be intransitive or cyclic, going beyond the Bradley–Terry (BT) assumption underlying RLHF and DPO. COMAL is design for solving NLHF and seeks the Nash equilibrium (NE) policy that guarantees a 50% win rate against any competing policy. Unlike existing self-play algorithms such as iterative IPO or SPPO, which either diverge or only converge to equilibria in a regularized game, COMAL guarantees monotone KL convergence and last-iterate convergence to the NE, even under approximate subproblem solutions. Empirically, COMAL achieves the best performance among baselines (DPO, IPO, INPO, etc.) on synthetic intransitive games and real LLM fine-tuning tasks with Llama-3-8B-Instruct and Qwen2.5-7B on the UltraFeedback dataset, maintaining >60% win rate against competing algorithms and stable training dynamics.

### Strengths
Strengths:

1. theoretical contribution: COMAL provides a last-iterate convergence guarantee to the unregularized Nash equilibrium in the alignment game, while prior methods only achieved average-iterate or regularized convergence.

2. The experiments is comprehensive, convincingly show that the algorithms outperform existing approaches.

### Weaknesses
1. The main technique used to achieve last-iterate convergence is the introduction of proximal steps, which is a standard approach in optimization theory. Therefore, the theoretical contribution of this work appears limited.

2. As the authors note, Wang et al. also obtain last-iterate convergence results for the original game. I would appreciate a more comprehensive comparison between the two approaches, rather than mentioning it only in a footnote. Since these works are concurrent, I will not lower my rating due to similarity; however, for scientific completeness, a detailed comparison would be valuable.

3. Experiments: NLHF is primarily proposed to address cyclic preferences. However, the current experimental setup still follows the standard binary comparison framework and evaluates performance accordingly, leaving a conceptual gap between the theoretical motivation and the empirical validation.

The experimental design in [1] offers an interesting and informative direction. In the UltraFeedback dataset, each response is evaluated across multiple dimensions, such as honesty, truthfulness, and helpfulness. These dimensions can naturally be viewed as distinct “voters” or preference sources, making it possible to construct datasets with cyclic preferences. For example:

Cycle 1: Honest ≻ Truthful ≻ Helpful ≻ Honest

Cycle 2: IF ≻ Truthful ≻ Helpful ≻ IF

Cycle 3: IF ≻ Honest ≻ Helpful ≻ IF

Cycle 4: IF ≻ Honest ≻ Truthful ≻ IF

Exploring such cyclic-preference settings would align more closely with the theoretical foundation of NLHF and provide stronger empirical support for its claims. I suggest the authors consider such setups in future work.

[1] Y. Zhang, G. Zhang, Y. Wu, K. Xu, Q. Gu. Beyond Bradley–Terry Models: A General Preference Model for Language Model Alignment.

---
Other Related Work:

The paper emphasizes the importance of convergence to the Nash equilibrium (NE) of the original game, rather than to the modified KL-regularized game. Paper [2] provides evidence supporting this distinction. Specifically, when human preferences are cyclic (e.g., A ≻ B ≻ C ≻ A), if the preference profile contains a Condorcet winner, the NE of the original game corresponds to that winner; if not, the NE must be a mixed-strategy equilibrium.

[2] Kaizhao Liu, Qi Long, Zhekun Shi, Weijie J. Su, Jiancong Xiao. Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper introduces COMAL, a meta-algorithm to align LLMs with general human preferences, which standard methods based on the Bradley-Terry (BT) model fail to capture (e.g., intransitive or cyclic preferences). It reframes alignment as a two-player zero-sum game and proposes an iterative method that, unlike existing algorithms that diverge or solve a modified problem, is theoretically guaranteed to converge in its last iterate to the exact Nash equilibrium policy. This ensures a robust 50% win rate against any competing policy. COMAL functions as an outer loop that dynamically updates a reference policy, allowing it to be integrated with many existing preference optimization methods like DPO or INPO. Experiments on synthetic games and large models like Llama-3-8B and Qwen2.5-7B demonstrate that COMAL converges correctly where others fail and consistently achieves superior win rates (e.g., over 60.2% and 56.8% against all competitors, respectively) while maintaining training stability.

### Strengths
- Originality: This paper proposes a meta-algorithmic framework that unifies many existing preference optimization methods (like DPO, PPO, and INPO) by re-interpreting them as practical implementations of a "Prox operator". 
- Theory: This paper proves last-iterate convergence (Theorem 3) when the inner regularized game is solved only approximately.
- Clarity: This paper proposes meta-algorithm (Algorithm 1) first and then connects it to the underlying theory and practical implementations, makes the complex framework easy to follow.

### Weaknesses
- The main LLM experiments use a "mixture of two BT reward models" as the preference oracle, which doesn't necessarily exhibit the complex intransitivity that defines the core problem. So, experiments should include a preference oracle that explicitly has such phenomenon.
- Theorems 1 and 3 rely on solving a regularized game approximately, but it is unclear that in practice if the using the solver to solve the problem in the parameter space can produce a result which has enough close KL divergence. So, I think the theory is inadequate. Could you establish relation between the parameter approximation and distribution approximation. 
- In experimental details, in LLaMA, $\eta^{-1}$ is changed from 0.002 to 0.1 after the first round, and $\eta^{-1}$ is fixed to be 0.002 in Qwen. It seems that the performance is sensitive to its choice. A more detailed discussion on the choice of $\eta$ is required, such as the ablation study. Or the reason why you adopt this schedule.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
