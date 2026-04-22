# Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Gaussian policies have dominated continuous control in deep reinforcement learning (RL), yet they suffer from a fundamental mismatch: their unbounded support requires ad-hoc squashing functions that distort the geometry of bounded action spaces.
While von Mises-Fisher (vMF) distributions offer a theoretically grounded alternative on the sphere, their reliance on Bessel functions and rejection sampling hinders practical adoption.
We propose \textbf{Geometric Action Control (GAC)}, a novel action generation paradigm that preserves the geometric benefits of spherical distributions while \textit{simplifying computation}.
GAC decomposes action generation into a direction vector and a learnable concentration parameter, enabling efficient interpolation between deterministic actions and uniform spherical noise.
This design reduces parameter count from \(2d\) to \(d+1\), and avoids the \(O(dk)\) complexity of vMF rejection sampling, achieving simple \(O(d)\) operations.
Empirically, GAC consistently matches or exceeds state-of-the-art methods across six MuJoCo benchmarks, achieving 37.6\% improvement over SAC on Ant-v4 and up to 112\% on complex DMControl tasks, demonstrating strong performance across diverse benchmarks.
Our ablation studies reveal that both \textbf{spherical normalization} and \textbf{adaptive concentration control} are essential to GAC's success.
These findings suggest that robust and efficient continuous control does not require complex distributions, but a principled respect for the geometry of action spaces. Code and pretrained models are available in supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel method for action sampling in continuous control environment. Authors point out that the common practice of sampling from a diagonal Gaussian have a fundamental mismatch: the unbounded support of Gaussian and bounded support of action space.

To this end, they propose to instead constrain the action space onto a unit sphere, where the network outputs a deterministic action $\mu$ and state-dependent exploration paramter $w(\kappa)$. As exploration is achieved simply by interpolating between predicted action ($\mu$) and random noise ($\epsilon$), the aforementioned mismatch is addressed, while still applicable to algorithms such as SAC.

Efficacy of the method is shown by experimenting on 6 Mujoco environments, where they outperform naive SAC, TD3 and PPO on 4 of the 6 environments.

### Strengths
- **Clear motivation**. Authors clearly state the issue of action space handling with Gaussian and the limitation of previous method (vMP), and propose a novel method for addressing the problem.
- **Simplified SAC, theoretically grounded**. Authors show that their method can be integrated with SAC, which removes the need for temperature tuning. While this immediately raises the question of theoretical soundness, authors also provide the proof that the algorithm maintains contraction even after the simplification.

### Weaknesses
- **Over-limitation of action space**. The proposed method limits all actions to have the same magnitude. This raises the concern: fixing the action's magnitude removes the to flexibly choose between 'active motions' (high action magnitude) and 'passive motions' (low action magnitude). Indeed, authors show that the choice of magnitude $r$ is crucial to the performance.
- **Limited results and evaluation benchmark**. Combined with above, the results of Mujoco do not seem very compelling. Compared to SAC, GAC outperforms in one environment (Ant-v4) but is also outperformed by SAC in Pusher-v4. To gain more persuasiveness, I suggest expanding the evaluation benchmark to e.g., DMC, Metaworld.
- **No comparison with vMF**. Since vMF has been mentioned as prior work throughout the paper, it seems natural for them to appear as one of the baselines (despite its practical difficulty).

### Questions
- Line 234: Can $\epsilon \sim \text{Uniform}(S^{d-1})$ be considered a normalized Gaussian noise? If not, could there be a better way to add noise that's centered on $\mu$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Geometric Action Control (GAC), a distribution-free policy paradigm that generates actions via a mixing operation with a unit sphere, replacing traditional Gaussian policies to avoid gradient saturation. GAC performances near SOTA models on continuous control Mujoco benchmarks.

### Strengths
This paper is overall logically clear, presenting an interesting perspective on the action exploration in continuous control tasks. With this perspective, this paper reformulates the exploration distribution to a unit sphere. Compared to a policy with von Mises-Fisher distributial output, it reduces the sample complexity.

### Weaknesses
This paper seems to overstate its contribution by comparing to a distribution that is not commonly used in RL, especially on the point about sample complexity. It is better to have a clearer comparison between the proposed method with deterministic models and Gaussian based models for the "Geometric" feature and sample complexity, respectively. The "Geometric" feature in this work is similar to deterministic models, which is not clearly stated in this paper.

### Questions
Could the authors compare the overall computational complexity of GAC with deterministic and Gaussian based models, instead of only emphases the sample complexity?

This model assumes the range of the action space is mapped to a unit. What if the range of the action space is unknown? What if the best actions are outside the unit sphere for exploration

### Soundness
3

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
3

### Summary
The paper focuses on the issue of distorted sampling space caused by Gaussian policies in a bounded action space. To solve this, a common technique is to use "squashing" functions, such as tanh, to map the Gaussian samples into a bounded range, causing actions to cluster near the boundaries. While alternatives like von Mises-Fisher (vMF) distributions are theoretically grounded on the unit sphere, they are computationally expensive. The paper proposes to retain the geometric benefits of spherical distributions by interpolating between a deterministic action direction and a uniformly sampled unit sphere vector. The action follows the direction and gets multiplied by a magnitude scalar. The proposed algorithm shows comparative empirical results to SAC, PPO, and TD3 on six MuJoCo tasks.

### Strengths
The paper provides a strong motivation and a clear presentation of the algorithm.

It also delivers a comprehensive experimental analysis, featuring multiple baseline comparisons, thorough ablation studies, and an in-depth examination of convergence behavior and the sampling landscape.

### Weaknesses
The algorithm introduces an extra hyperparameter, action magnitude, but the robustness analysis across tasks is missing. Though the algorithm is well-motivated, the performance improvement is limited at the cost of an extra hyperparameter.

### Questions
1. As shown in Figure A.2, 46.4% of all pre-squashed action samples fall into regions of "Gradient Saturation". However, why are the performances of baselines not deteriorated by this issue?

2. What distribution is the current action following? Figure A.1, GAC samples from a subsurface instead of a unit ball. Will it cause an exploration issue?

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
5

### Summary
This paper proposes a novel action generation paradigm that preserves the geometric benefits of spherical distributions while simplifying computation. GAC decomposes action generation into a direction vector and a learnable concentration parameter. It focuses on an important topic, policy distribution. The experimental results show its efficiency across MuJoCo benchmarks.

### Strengths
The motivation of this work is novel and interesting, which could be an important work in the RL community.
GAC represents policies through two components: a direction network that outputs unit vectors indicating preferred action orientations, and a concentration network that controls exploration by interpolating between deterministic directions and uniform spher-
ical noise.
It takes a novel perspective on whether the distribution paradigm itself is necessary, which also promotes efficient exploration.
The paper offers thorough theoretical analysis, with clearly stated assumptions, theorems, and proofs.

### Weaknesses
More benchmarks may be needed to test and provide solid experimental results.

Regarding the policy distribution, I recommend that the authors add the discretization policy distribution topic works.

Discretizing continuous action space for on-policy optimization, AAAI, 2020 
Discretizing Continuous Action Space With Unimodal Probability Distributions for On-Policy Reinforcement Learning, IEEE TNNLS, 2024.

### Questions
In Figure 2, two tasks do not have an advantage. Can the authors provide empirical results on more environments?
How to guarantee that the unnecessary dimension does not influence the optimal policy or the optimal policy theoretical analysis?

### Soundness
4

### Presentation
3

### Contribution
4
