# Relative Value Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
In reinforcement learning (RL), critics traditionally learn absolute state values, estimating how good a particular situation is in isolation. Adding any constant to $V(s)$ leaves action preferences unchanged. Thus only value differences are relevant for decision making.
Motivated by this fact, we ask the question whether these differences can be learned directly. For this, we propose \emph{Relative Value Learning} (RV), a framework that considers antisymmetric value differences $\Delta(s_i, s_j) = V(s_i) - V(s_j)$. We define a new pairwise Bellman operator and prove it is a $\gamma$-contraction with a unique fixed point equal to the true value differences, derive well-posed $1$-step/$n$-step/$\lambda$-return targets and reconstruct generalized advantage estimation from pairwise differences to obtain an unbiased policy-gradient estimator (R-GAE). Besides rigorous theoretical contributions, we integrate RV with PPO and achieve competitive performance on the Atari benchmark (49 games, ALE) compared to standard PPO, indicating that relative value estimation is an effective alternative to absolute critics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes relative value learning, which considers value differences between states instead of the usual absolute values of states. The authors show that value differences satisfy a kind of Bellman operator and inherit the contraction property, which can be utilized to learn value differences in a way similar to classical values (e.g., TD ($\lambda$)). Similar to GAE, the value differences can also be used to construct advantage estimators using TD errors, and the authors prove that such "advantages" are unbiased for policy gradients. Empirical results for the ALE were given, and results show competitive performance compared to GAE and DAE.

### Strengths
The paper is well-motivated (i.e., the absolute scale is irrelevant in terms of policy improvement), clearly written, and easy to follow.
The theoretical results showing that the value differences satisfy a Bellman equation and can be similarly estimated like classical value functions are quite intriguing, and might open up new ways to design RL algorithms.

### Weaknesses
While convergence results were given, it is not immediately clear whether learning the relative values actually provides any benefit over traditional absolute value estimators, and empirical results seem to suggest that the two are perhaps not too different in terms of policy optimization.

Regarding the experiments:
1. I would highly suggest reporting aggregated metrics (e.g., median, IQM, or similar) and learning curves alongside per-environment scores. Currently, it's very difficult to assess the effectiveness of the proposed approach simply from Table 1 and the description on line 431.
2. It is now well-known that deep RL algorithms often come with high variance, and reporting statistical significance is necessary for comparing the performance of different algorithms [1].
3. The comparison to DAE is not entirely fair, as the reported results [2] use a much larger number of actors (1024 compared to 8), corresponding to significantly fewer PPO iterations.

There are also some interesting details that I think are worth mentioning in the main text, such as how the state pairing is done in practice (line 806). 

[1] Agarwal, Rishabh, et al. "Deep reinforcement learning at the edge of the statistical precipice."  NeurIPS 2021.

[2] Pan, Hsiao-Ru, et al. "Direct advantage estimation." NeurIPS 2022.

### Questions
1. It looks like relative values can eliminate the shift invariance of the reward functions, and I wonder if there are remaining redundant degrees of freedom that can be fixed. Can the authors elaborate on this?
2. While Section 4 suggests that $B_t$ can increase the variance of the policy gradient estimator and points the reader to Appendix C, this problem is not really addressed in Appendix C either, aside from an empirical demonstration. Can the authors provide a more in-depth analysis on this?
3. Regarding the critic training, can the authors share more details on how the pair sampling affects performance (line 806-809)? Also, it is not clear to me how $\mu$ (Eq 28) affects convergence theoretically. I imagine the $(i, j)$ have to follow a certain distribution that is related to the sampling policy, but this is not really discussed.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Relative Value Learning (RV), which focuses on learning value differences between states, ∆(si, sj) = V(si) − V(sj), rather than absolute state values. It introduces a pairwise Bellman operator, proves its convergence, and derives targets for 1-step, n-step, and λ-return updates, as well as a policy-gradient estimator based on these differences (R-GAE). The method is integrated with PPO and tested on Atari games, showing performance comparable to standard PPO.

### Strengths
- The paper identifies a meaningful gap in reinforcement learning: the focus on absolute state values when only differences matter for decision-making.
- It provides well-defined mathematical formulations, including the pairwise Bellman operator, and proves properties like γ-contraction.
- It is implemented with PPO, showing the approach is compatible with widely used RL algorithms. The experiments cover a broad benchmark (49 Atari games), providing evidence that the approach is competitive with standard methods.
- The paper is overall well-written, with clear explanations.

### Weaknesses
- The paper does not beat state-of-the-art results in the experiments
- The paper is well written with clear mathematical reasoning but I found some parts a little less straightforward in the explanations (e.g. relative value initialization, see also the question below)

### Questions
- Given the case where both successors are absorbing with zero value, could that provide a way to anchor the values?
- Would it be possible to visualize the value differences between states for a few states or trajectories?
- Why does the paper focuses on 49 while the usual full set of ATARI games includes 57 games?
- In the conclusion, you mention "We hope RV serves as a building block for algorithms that reason natively in the relative domain where decisions are actually made.". What would be the potential advantages given that performance-wise, it does not seem to improve learning? Any setting where this approach might be superior to usual Q-learning based for instance?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Relative Value Learning (RV), a reinforcement learning framework that focuses on learning pairwise value differences $\Delta(s_i, s_j) = V(s_i) - V(s_j)$ instead of absolute value functions. The authors derive a “pairwise Bellman operator,” show that it is a γ-contraction, and propose a relative version of Generalized Advantage Estimation (R-GAE). They integrate RV with PPO and evaluate the approach on Atari games, reporting comparable or slightly improved performance in some cases.

### Strengths
- The paper is clearly written and well organized
- The empirical results are conducted on a reasonably broad benchmark

### Weaknesses
- Limited theoretical novelty. The main theoretical results are direct consequences of standard value-based RL theory. The proposed “pairwise Bellman operator” is simply the subtraction of two standard Bellman equations, and the contraction proof follows immediately from classical results. Similarly, R-GAE and the unbiasedness result are reformulations of known properties of GAE. While the relative formulation is conceptually neat, it doesn’t introduce new theoretical insights.
- Marginal empirical contribution. The empirical results show that PPO+RV performs on par with PPO and DAE, but without consistent or significant improvement. This makes it difficult to justify publication at a top-tier venue like ICLR on empirical grounds alone.
- Insufficient related work discussion. The paper cites relatively few references and overlooks related research on invariance, reward shaping, normalization, and average-reward formulations, which would help clarify where this work fits in the broader context.
- Overall, the paper feels more like a conceptual reinterpretation of existing ideas than a substantial theoretical or algorithmic advance.

### Questions
1. Since the pairwise Bellman operator is mathematically equivalent to subtracting two standard Bellman updates, what concrete advantage does this formulation bring in practice beyond removing the value offset?
2. How sensitive is the performance of PPO+RV to the way state pairs are sampled?
3. Have you tested RV in continuous-control or off-policy settings, where relative value estimation might interact differently with bootstrapping?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes learning relative value differences compared to directly predicting state value. They integrate the proposed idea of relative value learning with PPO and evaluate its performance on the Atari benchmark.

### Strengths
- The idea of estimating relative values is well motivated and might additionally have benefits under function approximation settings (as noted in \[1\])  
- The paper is well written and the details of the method and experimental setup are clearly outlined.

### Weaknesses
- The idea of using cost-to-go differences or relative value was explored earlier in \[1\], so perhaps the idea of estimating value difference is not necessarily novel. However, its application in DeepRL might be novel.  
- The proposed approach does not necessarily improve over the PPO baseline in the evaluated benchmark, it would be good to also indicate the observed variance in scores to suitably contrast against the baselines.

### Questions
- Based on the above, could the authors differentiate their contributions from the mentioned prior work \[1\] and also report variance in performance on the benchmark?

**References**  
\[1\] Bertsekas, “Differential Training of Rollout Policies”, Proc. of the 35th Allerton Conference on Communication, Control, and Computing, Allerton Park, Ill., October 1997

### Soundness
2

### Presentation
2

### Contribution
2
