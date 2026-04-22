# Adapting Rewards to the Agent Using Rational Activation Functions

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Fixed environment rewards can lead to miscalibrated gradients, instability, and inefficient learning when signals are poorly scaled relative to the agent's updates. We introduce \textbf{Rational Reward Shaping (RRS)}, a reward transformation that converts raw rewards into normalized signals aligned with the agent's experience. RRS combines experience-normalized scaling with a monotone rational activation to reshape sensitivity and curvature while preserving reward order. It adapts automatically to changing reward regimes and integrates seamlessly into standard actor–critic updates--simply replacing the immediate reward in the target--requiring minimal code changes and no task-specific reward engineering. Across DDPG, TD3, and SAC on six MuJoCo benchmarks, RRS consistently improves average returns in both noiseless and perturbed-reward settings, with larger gains under noise, while incurring only 6\% average wall-clock overhead. RRS provides a general, plug-and-play method to produce better-calibrated reward signals, strengthening learning without modifying environment design. Source code is available at: \url{https://github.com/anonymouszxcv16/RRS}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- This paper addresses the mismatch between fixed environmental rewards and agent capabilities in deep reinforcement learning (DRL) by proposing a reward shaping scheme that combines empirical normalization (to match reward scales) and a monotonically decreasing rational activation function (to match learning sensitivity), which can be integrated into algorithms like DDPG, TD3, and SAC without modifying their core frameworks.

- The proposed method is validated on 6 continuous control tasks in MuJoCo (covering both noiseless and noisy scenarios), showing improved average returns and training stability under most configurations—with more significant gains in noisy environments—thus demonstrating its effectiveness and anti-interference ability.

### Strengths
- Addressing the issue of "mismatch between fixed environmental rewards and agent capabilities" in deep reinforcement learning (DRL), the proposed method directly tackles the problems of improper gradient calibration and unstable training caused by reward signals exceeding the agent’s internalization capacity. This is achieved through a design integrating "empirical normalization for matching reward scales + adaptive rational activation for matching learning sensitivity". Notably, in noisy environments with the reward-sensitive DDPG algorithm, it achieves a significant performance improvement, validating the effectiveness of the scheme.

- The method is concise and easy to implement. The monotonic rational activation function dynamically adjusts reward curvature while preserving the reward order. Without modifying the core framework of existing algorithms, it can be integrated into DDPG, TD3, and SAC algorithms merely by replacing the original reward with the shaped reward in the target Q-value calculation—eliminating the need for task-specific reward engineering.

- The experimental validation is comprehensive and robust, covering three dimensions: "algorithms (DDPG, TD3, SAC) – environments (6 continuous control tasks in MuJoCo, ranging from the simple Ant to the complex Humanoid) – signals (noiseless and multiplicative interference noise scenarios)". Under most configurations, it enhances both average return and training stability, with more prominent gains in noisy environments. This fully demonstrates the method’s generality and anti-interference capability.

### Weaknesses
- The adaptive tuning of α suffers from a domain limitation. The paper assumes that α covers the interval [0.5, 1] via the scaled_sigmoid function; however, the actual input x is always greater than 0 (due to ξ>0), leading to the output of sigmoid(x) being consistently greater than 0.5. Consequently, the actual value range of scaled_sigmoid(x) is only (0.75, 1), violating the initial assumption that "large α is required for sparse rewards and small α for dense rewards". This weakens the method’s adaptability to scenarios with low-variance dense rewards, and the paper fails to explain this contradiction.

- Key ablation experiments are absent. The individual effects of "empirical normalization" and "rational activation function" are not verified independently—neither the performance of using only rational activation (with empirical normalization removed) nor that of using only empirical normalization (with rational activation replaced by conventional activation functions) is tested. As a result, it is impossible to determine whether the performance improvement of RRS stems from reward scale optimization, reward curvature reshaping, or their synergy, which undermines the persuasiveness of the method.

- In Section 3.2, this paper employs a monotonically decreasing rational activation function, which poses unresolved contradictions with the fundamental principles of reinforcement learning (RL). By its very nature, RL optimizes for the maximization of cumulative rewards; however, a monotonically decreasing transformation inverts this objective: maximizing the adjusted reward is effectively equivalent to minimizing the original reward.

- The adaptive tuning of α depends only on reward statistics (inverse standard deviation) and is not linked to the agent’s actual capability indicators (e.g., state dimension). This makes it impossible to quantitatively explain how α matches agents with different capabilities, resulting in a lack of theoretical depth.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
3

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
The paper proposes Rational Reward Shaping (RRS), which normalizes rewards and applies a “rational activation function” with an adaptive parameter α to align rewards with agent capacity.
Experiments on DDPG, TD3, and SAC in MuJoCo show moderate improvements, especially under noisy rewards.

The paper reads as a heuristic reward transformation with unclear motivation.
“Rational activation” and “capacity-aware shaping” are not conceptually substantiated.
Despite clean writing and broad experiments, the work lacks theoretical grounding, internal consistency, and precise presentation.

### Strengths
- Clear structure and broad experimental coverage.
- Implementation is simple and integrates easily.
- Results show small but consistent gains across several environments.

### Weaknesses
- While Eq. (2) cites prior work on rational activations, the paper does not explain how that reference conceptually relates to the proposed transformation; there is no discussion in Related Work or Preliminaries clarifying its theoretical basis.
- The introduction section introduces capacity limitation (L40–L56) with no clear connection to reward shaping. Sentences like “the effectiveness of both exploration-exploitation balance and reward shaping is fundamentally shaped by the agent’s capacity” (L55-L56) are confusing.
- “Capacity-aware” remains undefined and unmeasured; the method effectively acts as a heuristic rescaling.
- The adaptive α-update rule lacks theoretical justification or formal analysis of its stability or convergence.
- The meaning of “improvement” is unclear. It is not specified what baseline the values are computed against, and the meaning of averaging them across environments is questionable. In Table 2, improvements appear relative to the left baseline algorithm, whereas in Table 3 they seem measured against the noiseless setting, causing inconsistency and confusion.
- Claims such as “higher noise robustness tend to benefits from higher values of α” (L398) are unsupported.
- Several phrases (e.g., “unlimited” L50, “differences of potentials” L85) require clearer explanations, and multiple citation parentheses are formatted incorrectly throughout.

### Questions
Covered within the Weaknesses section above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Rational Reward Shaping (RRS): replace the environment reward in actor–critic targets with a capacity-aware, monotone transformation that combines (i) experience-based normalization from the replay buffer and (ii) a rational activation with an auto-tuned curvature parameter \alpha driven by recent reward variability. The method is drop-in for DDPG/TD3/SAC and aims to stabilize gradients and improve sample-efficiency without hand-crafted task shaping. On six MuJoCo tasks, RRS variants often improve returns in both noiseless and perturbed-reward settings (with larger gains under noise), while adding little implementation overhead. The paper also analyses alpha’s evolution, reward variability, and runtime.

### Strengths
- Simple, drop-in idea with low engineering overhead.
- Broad coverage (DDPG/TD3/SAC) across several tasks.
- Analyses of curvature parameter dynamics and reward variability add useful diagnostics.
- Shows notable improvements in some settings, particularly with reward noise.

### Weaknesses
- Transform outputs strictly positive rewards, potentially changing optimal policies (not policy-invariant shaping).
- Limited ablations
- Statistical evidence is underpowered for strong claims.
While mean±std are reported, robustness claims would be stronger with more seeds, 95% confidence intervals, and paired significance tests per environment, following best practices (e.g., https://jmlr.org/papers/volume25/23-0183/23-0183.pdf). Seed-level violin plots would clarify variance and overlap.

### Questions
- Can you characterise when your monotone, non-affine (and always-positive) transform preserves optimal policies? If not, please reframe as a surrogate objective and discuss bias.
- Why claim boundedness for your normalisation? Would z-score or min–max over replay be more appropriate, and how would results change?
- Did you retune or auto-tune SAC’s temperature under reward rescaling?

- Typo to fix line 161 "a a"

### Soundness
2

### Presentation
2

### Contribution
2
