# Relative Entropy Pathwise Policy Optimization

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Score-function based methods for policy learning, such as REINFORCE and PPO, have delivered strong results in game-playing and robotics, yet their high variance often undermines training stability. Improving a policy through state-action value functions, for example by differentiating Q with regard to the policy, alleviates the variance issues. However, this requires an accurate action-conditioned value
function, which is notoriously hard to learn without relying on replay buffers for reusing past off-policy data. We present Relative Entropy Pathwise Policy Optimization, an algorithm that trains Q-value models purely from on-policy trajectories, unlocking the use of Q function derivatives to compute policy updates in the context of on-policy learning. We show how to combine stochastic policies
for exploration with constrained updates for stable training, and evaluate important architectural components that stabilize value function learning. This results in an efficient on-policy algorithm that combines the stability of Q-based policy gradients with the simplicity and minimal memory footprint of standard on-policy learning. Compared to state-of-the-art on two standard GPU-parallelized benchmarks, REPPO provides strong empirical performance at superior sample efficiency, wall-clock time, memory footprint, and hyperparameter robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an on-policy algorithm that utilizes a pathwise gradient estimator with a learned Q-function, along with specific design choices to stabilize the proposed algorithm. The proposed algorithm only involves computing policy updates on on-policy actions, which avoids importance sampling and instability in the traditional PPO method.

### Strengths
- The design principle of the algorithm is, to my knowledge, novel, sound, and interesting
- The algorithm is more on-policy than traditional PPO, and therefore, more stable. The authors showcase the performance benefits in extensive benchmark tests
- Compared with off-policy methods, the proposed method does not require large memory consumption for a replay buffer
- The presentation is clear and flows smoothly

### Weaknesses
- I think a more thorough justification (theoretical and/or empirical) about what the main factors are that bring the claimed performance benefit can significantly strengthen the authors' claims; cf. my question about Figure 3 below. Also, if learning a Q function instead of V function alleviates the high-variance problem in PPO, what are the potential costs? I think a more careful justification will have longer-term benefits for the community and further development of this type of method
- Algorithm 1 (line 1364 of the document): V_{t+1} is not equal to Q_{\phi}(x_{t+1}, a_{t+1}) (missing an expectation over action)
- Policy optimization objective: the actual objective function used is not given in the text. Also, the notation used KL_{tar} is a bit arbitrary 
- If some of the improtant baselines are indeed using different number of total samples (see my question below), then, I will suggest a proper comparison to those baselines with same access to the environment samples as REPPO  
- Minor:
    - Line 246 typo “constraint it” -> “is”
    - Equation 12 is the gating reverse KL, whereas other places are using the forward KL divergence. Which one is true?
    - Current way of writing equation 12 is a bit off; consider pulling out the curly brace definition separately

### Questions
- In Figure 2: why does PPG with learned surrogate optimize better than the ground truth, even if the contour lines of the objective look quite simple and one would expect following the gradient should be the best direction? Also, what do you use as Q(s, a) for the score-based estimator in this case (should the score-based estimator also have different variants using different surrogates for Q(s, a))?

- Figure 3: The fact that the performance difference between REPPO (pathwise) and REPPO (score-based, Q) is tiny, and they both outperform PPO and REPPO (score-based, GAE) by a large margin, makes me wonder where the main performance gain of REPPO comes from. I am thinking that the performance benefit might mainly come from the fact that REPPO (pathwise) and REPPO (score-based, Q) query on-policy actions computed from the current policy (as opposed to the other two that use fixed, potentially off-policy actions sampled from data batches), which avoids the need for importance sampling scores. If that’s the case, then using a score-based vs. a pathwise estimator only makes a rather small difference. Please correct me if I am missing anything here. 

- Figure 4: Why are many of the baseline results using a different number of environment steps? For example, is SAC using 5M steps, i.e., 1/10 of those used by REPPO? If that’s the case, I think it is not a fair comparison, and the result for SAC is not a meaningful reference. Therefore, the claim about statistically significant improvement over SAC is overclaiming as well.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the REPPO algorithm that optimizes the policy by directly differentiating the Q function in a completely on-policy setting.

### Strengths
+ The problem studied in this paper is motivated well.

+ Presentation is clear and easy to follow.

+ Empirical results are comprehensive and the proposed algorithm performs well across several benchmark environments. It is particularly interesting to see that directly optimizing the $Q$ function is feasible for on-policy methods, which is often believed to rely on off-policy data.

### Weaknesses
I have some questions regarding the design of the REPPO algorithm.

+ Upon checking the code, I found that the critic function does not directly take the state-action pair as the input, but a latent state $\phi(s, a)$ which is the output of a feature module. That is, when differentiating the policy objective (Equation 12), using the reparameterization trick it yields 
$$ \nabla_\theta Q(x_i, a) = \nabla_\theta Q_\eta ( \phi(s_i, a) ) = \frac{\partial Q}{\partial \eta} \frac{\partial \phi}{\partial a} \frac{\partial a}{\partial \theta}$$
where I denote the critic parameters by $\eta$ instead of $\phi$ to avoid abusing notation. I wonder if the feature module $\phi$ is critical to the algorithm as the value landscape is much simpler in the latent space.

+ Can you provide more insights on why the HL-Gauss loss is better than MSE in critic training?

+ The rollout length in the code is 128, which is longer than horizons commonly used in on-policy methods. I wonder how different lengths affect the performance, especially for the tasks with short horizon such as Maniskill manipulation problems. 

+ Training the Q function can be numerically unstable, which has motivated many tricks in off-policy methods such as double Q learning, employing target networks, etc.. In Section 3.3, the paper introduces three methods to enhance the stability. However, as shown in the ablation study (Figure 10), removing any of them does not lead to da significant performance drop. Can you give me some intuition on why the critic training in REPPO is more robust than its off-policy counterparts beyond these three points?

+ Typically, directly differentiating the Q function requires it to have a nice first-order approximation to the local value landscape in the state-action space so that its gradient with respect to the action yields a correct updating direction. To achieve this, off-policy methods use very short horizons for the TD estimation, such as one step in TD3 and SAC and five steps in TDMPC, under the consideration that longer trajectories would introduce more noise into the estimation and affect the accuracy of the gradient estimation. As mentioned earlier, REPPO uses a very long TD horizon, and I wonder why its Q function can still effectively capture the gradient direction in this case.

I would be happy to increase my score if the authors can address my concerns.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors are motivated by the goal of formulating a modern on-policy reinforcement learning algorithm that dispenses with the large replay buffers characteristic of many state-of-the-art off-policy methods, thereby reducing the associated memory footprint and data management overhead. To this end, they propose Relative Entropy Pathwise Policy Optimization (REPPO), an on-policy actor-critic algorithm that builds upon the maximum entropy framework and utilizes the pathwise policy gradient, similar to Soft Actor-Critic (SAC).

The core of their contribution lies in adapting these off-policy components to the more challenging (wrt value-function learning) on-policy setting. In particular, they replace the standard single-step TD targets with multi-step TD(λ) value targets. This is a critical choice, as it aims to reduce the bias in value estimates that is particularly severe in the on-policy regime due to the high temporal correlation of the training data. Furthermore, to ensure stable policy updates, the authors employ a KL-regularized trust region, dynamically tuning the Lagrange multipliers using a simultaneous gradient-based approach inspired by SAC. Finally, the algorithm incorporates several recent advances in value function learning to further enhance stability, including the use of a categorical Q-function representation, layer normalization, and auxiliary tasks. The authors provide a thorough empirical evaluation comparing REPPO against several state-of-the-art algorithms, reporting significant performance improvements in terms of final performance and stability.

### Strengths
*   The paper presents a compelling and timely approach to on-policy learning, which is of significant interest to the community, particularly in the context of modern, massively parallelized simulation environments where data generation is no longer the primary bottleneck.
*   The proposed method is a well-motivated synthesis of several powerful, existing algorithmic components. The originality lies in their successful adaptation and integration for the on-policy setting.
*   The claims are supported by an extensive empirical evaluation, including thorough ablation studies, which demonstrates strong performance against relevant baselines.

### Weaknesses
The primary weaknesses of the paper lie in its exposition, where the mathematical notation could be more precise and consistent to improve clarity.
*   The notation in Equation (1) is slightly ambiguous. The expectation is written as being over the state distribution $s \sim \mu_\pi$, but the term inside also depends on the action $a$. It should be made explicit that the action is sampled from the policy, i.e., $a \sim \pi_\theta(\cdot|s)$.
*   The presentation of the pathwise policy gradient in Equation (2) could be clearer:
    *   The notation $a = \pi_\theta(x)$ is used without introduction. This notation typically implies a deterministic policy, whereas the algorithm employs a reparameterized stochastic policy. This should be clarified to avoid confusion with the Deterministic Policy Gradient theorem.
    *   Furthermore, the expectation over the noise variable $\epsilon$, which is essential to the reparameterization trick, is not made explicit in the expression.
*   There is a notational inconsistency between different parts of the paper. For instance, Equations (8-11) use $\pi_\theta(x)$ to denote the policy distribution, while Equation (12) uses the more explicit conditional notation $\pi_\theta(a|x)$. Consistent notation should be used throughout.

Typos/Nitpicks:
*   In the caption for Figure 4, it is stated that "we report the final performance at 100 million steps." However, the corresponding plot in Figure 4b includes a curve for "PPO (200M)". Please clarify the exact training horizon being reported.
*   Line 368: "perforamnce" should be "performance".

### Questions
*   A minor but interesting point from Figure 2 is that the pathwise gradient with the ground truth objective appears to converge more slowly than with the strong surrogate. Could the authors elaborate on the potential reasons for this phenomenon? 
*   Regarding the policy update, the authors supplement the automatic multiplier tuning with a heuristic clipping of the actor loss (Equation 12). It would be valuable for the authors to comment on the empirical importance of this clipping mechanism. How does it contribute to the algorithm's overall stability and performance?
*  I'd suggest improving on the points mentioned under "Weaknesses".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposed utilising the pathwise gradient estimator with an accurately learned surrogate value function to construct more efficient on-policy algorithms. The approach is empirically evaluated across multiple datasets to demonstrate its effectiveness.

### Strengths
- The paper is well-written and easy to follow.
- Figure 2 shows Illustrative examples of different estimators (policy gradient estimator vs. Different pathwise gradient estimators).
- Experiments contain various benchmarks (DMC suite and ManiSkill environment) and baselines.
- The efficiency of the algorithm is shown in different settings: number of samples to reach a certain performance level, memory, and wall-clock time.

### Weaknesses
See the questions section.

### Questions
Minors: the meaning of $x’_i$ in equation (6)?

### Soundness
3

### Presentation
4

### Contribution
3
