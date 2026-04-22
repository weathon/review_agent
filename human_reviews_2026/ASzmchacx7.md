# Reward Learning through Ranking Mean Squared Error

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Reward design remains a significant bottleneck in applying reinforcement learning (RL) to real-world problems. 
A popular alternative is reward learning, where reward functions are inferred from human feedback rather than manually specified.
Recent work has proposed learning reward functions from human feedback in the form of ratings, rather than traditional binary preferences, enabling richer and potentially less cognitively demanding supervision. 
Building on this paradigm, we introduce a new rating-based RL method,  Ranked Return Regression for RL (R4).
At its core, R4 employs a novel ranking mean squared error (rMSE) loss, which treats teacher-provided ratings as ordinal targets.
Our approach learns from a dataset of trajectory–rating pairs, where each trajectory is labeled with a discrete rating (e.g., "bad,''  "neutral,'' "good'').
At each training step, we sample a set of trajectories, predict their returns, and rank them using a differentiable sorting operator (soft ranks). We then optimize a mean squared error loss between the resulting soft ranks and the teacher's ratings.
Unlike prior rating-based approaches, R4 offers formal guarantees: its solution set is provably minimal and complete under mild assumptions. Empirically, using simulated human feedback, we demonstrate that R4 consistently matches or outperforms existing rating and preference-based RL methods on robotic locomotion benchmarks from OpenAI Gym and the DeepMind Control Suite, while requiring significantly less feedback.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Ranked Return Regression (R4), which trains a reward model from multi-class trajectory ratings using a ranking mean squared error (rMSE) loss built on soft ranks, and claims theoretical guarantees plus empirical gains on locomotion and DMC benchmarks. While this paper solves reward modeling by using the MSE loss of ordinal feedback, it is not a practical method because it requires human rating (classification) of each trajectory.

### Strengths
1. The idea of using ranking information from ratings is clear and simple.

2. The method is easy to implement within standard reward learning pipelines.

3. The experiments show some improvement over the chosen baselines on locomotion tasks under simulated settings.

### Weaknesses
1. The ordinal feedback is not a new idea for preference-based RL. The paper does not convincingly show that rMSE gives qualitatively different behavior than straightforward regression-to-targets, ordinal regression, or a calibrated cross-entropy approach.

2. There is a lack of related work [1,2]. [1] is an important one that theoretically analyzes why ordinal feedback can work better than pair-wise ones. Meanwhile, some work has explored using tied preference to improve PbRL [2], which should be included in related work and compared with R4.

[1] Liu, S., Pan, Y., Chen, G., & Li, X. Reward Modeling with Ordinal Feedback: Wisdom of the Crowd. 2025

[2] Liu, J., Ge, D., and Zhu, R. Reward learning from preference with ties. 2024

3. The most serious problem is that ordinal feedback is impractical in practice. This paper only improves the loss function without fundamentally solving the problem. Trajectory rating for real humans is a difficult and error-prone task, and solving this problem is the key to ordinal feedback.

4. The experiments use a scripted teacher to provide rates for trajectories, which can demonstrate R4's improvement over RBRL. However, it can not fully convince that R4 is better than comparison-based methods. Although they use the same number of feedback (number of rating trajectories = number of comparisons), this is not a fair comparison, as ratings are much more difficult to compare for a real human. Therefore, it is extremely important to prove the robustness of this method. The paper must show sensitivity to label noise, inter-rater variance, and biased raters (human-in-the-loop experiments); otherwise, claims about “less human effort” or “better sample efficiency” are ungrounded.

5. The experiment results are all curves but lack the IQM result report [3], which has the ability to accurately measure task performance across different tasks and seeds.

[3] M., Castro, P. S., Courville, A. C., & Bellemare, M. Deep reinforcement learning at the edge of the statistical precipice. 2021

6. The paper does not provide pseudocode, which makes it hard to understand.

### Questions
1. How sensitive is the method to the soft ranking temperature and other hyperparameters?

2. How does the method behave with strong class imbalance or missing classes?

3. How robust is the method to noisy and inconsistent ratings across different raters?

4. What is the relationship between learned reward sums and true returns? Is it more accurate than comparison-based methods?

### Soundness
2

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
4

### Summary
The paper proposes a rating‑based reward‑learning method that treats human ratings as ordinal supervision. Given trajectory–rating pairs, the proposed method R4 predicts returns with a reward model, applies a differentiable ranking operator to obtain soft ranks, and minimizes a ranking mean squared error (rMSE) between these soft ranks and the (ordered) rating labels. The authors argue this avoids hand‑crafted decision boundaries used in Rating‑based RL (RbRL) and preserves within‑class variance. They provide theory claiming the rMSE objective is “minimal and complete” under assumptions (deterministic realizability, correct binning, and an exact differentiable ranking), and evaluate R4 with simulated raters in Gym and DMC locomotion tasks.

### Strengths
- Simple, practical objective. Treating ratings as ordered targets and aligning ranks, instead of absolute scores, gives an intuitive loss that removes the design burden of rating bin boundaries required by RbRL.
- Leverages modern differentiable ranking. Building on fast, differentiable ranking/sorting (permutahedron projections) is sensible and computationally efficient compared to older proxies.
- Broad empirical envs on standard control suites. The paper reproduces common baselines (PEBBLE, SURF, QPA) and reports consistent improvements in several tasks, with ablations on feedback scheduling and sampling.

### Weaknesses
- The “minimality and completeness” results crucially assume (i) deterministic reward realizability, (ii) perfect ordinal binning by the teacher, and (iii) exact differentiable ranking that outputs true ranks. In practice, (iii) is not guaranteed—soft‑rank operators are precise for soft projections but do not generally equal hard ranks except in limiting or special cases. The paper cites [1] for “exact computation,” but that exactness refers to the soft operator and its gradients, not equality to hard ranks; the assumption that soft ranks equal true ranks is unrealistic for finite regularization and noisy scores.
- All results derive ratings from the ground‑truth reward, which risks optimistic conclusions and sidesteps the very issues that make reward learning hard (ambiguity, inconsistency, bias). Prior work has highlighted reward misspecification and gaming; without human‑rated studies or at least real‑world noisy labels, it is hard to assess robustness.
- Missing or under‑engaged related works: Rating/ordinal RM beyond RbRL: recent ordinal feedback reward modeling for LLMs extends beyond binary preferences and analyzes conditions for unbiasedness with graded labels—highly relevant to “ratings as ordinals.” [2]. VPIL models vague pairwise during the learning [3]. LiPO/PLPO optimize with ranked lists rather than pairs; while policy‑side, they show the broader move from pairwise to listwise/ordinal signals [4].

## References
[1] Fast differentiable sorting and ranking.

[2] Reward Modeling with Ordinal Feedback: Wisdom of the Crowd.

[3] Imitation Learning from Vague Feedback.

[4] LiPO: Listwise Preference Optimization through Learning-to-Rank.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
4

### Summary
This paper proposes Ranked Return Regression for RL (R4), a rating-based reward learning method that learns reward functions from ordinal human ratings instead of pairwise preferences or manually designed rewards. R4 introduces a ranking mean squared error (rMSE) loss that aligns predicted returns with teacher-provided ratings through differentiable ranking. The authors provide theoretical guarantees of completeness and minimality, and demonstrate through experiments on OpenAI Gym and DeepMind Control Suite that R4 outperforms existing rating- and preference-based methods.

### Strengths
1. The proposed ranking mean squared error (rMSE) loss is conceptually simple yet technically elegant, effectively leveraging ordinal ratings through differentiable ranking.

2. The theoretical analysis provides formal guarantees of completeness and minimality, strengthening the method’s conceptual foundation.

3. The experimental results encompass both offline and online feedback settings across multiple continuous-control benchmarks.

4. The ablation studies are carefully designed, demonstrating the stability of the core method and the incremental benefits of auxiliary design choices such as dynamic feedback scheduling and stratified sampling.

### Weaknesses
1. The paper provides limited conceptual and empirical discussion on how rating-based feedback compares to preference-based feedback. While the authors claim that ratings are more informative and cognitively efficient, this assumption is not rigorously analyzed.

2. The theoretical results rely on several strong assumptions, such as deterministic reward realizability, perfectly consistent rating bins, and nearly exact differentiable ranking, which may not hold in realistic human feedback scenarios. 
I acknowledge that Assumption 5 attempts to relax the requirement of exact ranking by introducing bounded ranking errors, but it only addresses the approximation error of the differentiable ranking operator rather than the inherent noise or inconsistency in human feedback.

3. All experiments are conducted with simulated feedback derived from environment rewards. As a result, the evaluation does not account for real-world human variability, label noise, or subjective inconsistencies that often arise in practice. Moreover, the experimental environments (DMC and MuJoCo) and the rating-based setting are not well aligned. It would be extremely difficult for human annotators to provide consistent absolute ratings or rankings for these continuous-control trajectories, which differ subtly in motion quality rather than in clear success or failure outcomes. This mismatch raises questions about how well the current results translate to realistic human feedback scenarios.

4. The proposed framework's generalization to true human feedback or more complex domains remains unverified, limiting the empirical validity of its broader claims.

### Questions
1. How would the proposed rMSE formulation perform when ratings come from real human annotators, whose feedback may be noisy, inconsistent, or influenced by context?
Given that all experiments use simulated feedback, how might the results change in a genuine human-in-the-loop setting?

2. Does the method remain theoretically sound or empirically stable if the assumption of perfectly ordered or deterministic ratings is relaxed?

3. How sensitive is R4 to the number of rating levels or to non-uniform distributions of ratings (e.g., heavy bias toward mid-level ratings)?

4. Conceptually, the current formulation seems to discretize what could otherwise be treated as continuous feedback. To what extent is R4 genuinely modeling human ordinal judgment, as opposed to learning from a discretized version of continuous return values?

Overall, I find the proposed approach well-motivated and technically sound within the ranking-based setting. The method provides a reasonable and effective formulation for learning from ordinal ratings, and the rMSE loss represents a meaningful contribution in this direction. However, the paper would benefit from a deeper conceptual discussion of how rating-based feedback fundamentally differs from preference-based supervision, both in terms of information structure and practical trade-offs. Moreover, the theoretical and experimental analyses rely on strong and idealized assumptions. A broader reflection on these assumptions, or additional experiments that relax them, especially regarding their implications for real human feedback, would substantially strengthen the paper's overall impact and credibility. While I appreciate the noise-related experiments included in the appendix, these simulations do not fully capture the complexity and inconsistency of real human feedback, and thus leave open questions about the method’s robustness in practical settings.

### Soundness
3

### Presentation
3

### Contribution
2
