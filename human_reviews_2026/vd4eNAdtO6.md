# Q-Learning with Adjoint Matching

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
We propose Q-learning with Adjoint Matching (QAM), a novel TD-based reinforcement learning (RL) algorithm that tackles a long-standing challenge in continuous-action RL: efficient optimization of an expressive diffusion or flow-matching policy with respect to a parameterized Q-function. Effective optimization requires exploiting the first-order information of the critic, but it is challenging to do so for flow or diffusion policies because direct gradient-based optimization via backpropagation through their multi-step denoising process is numerically unstable. Existing methods work around this either by only using the value and discarding the gradient information, or by relying on approximations that sacrifice policy expressivity or bias the learned policy. QAM sidesteps both of these challenges by leveraging adjoint matching, a recently proposed technique in generative modeling, which transforms the critic's action gradient to form a step-wise objective function that is free from unstable backpropagation, while providing an unbiased, expressive policy at the optimum. Combined with temporal-difference backup for critic learning, QAM consistently outperforms prior approaches on hard, sparse reward tasks in both offline and offline-to-online RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Q-learning with Adjoint Matching (QAM), a novel temporal-difference (TD) based reinforcement learning algorithm that addresses the challenge of optimizing expressive flow/diffusion policies with respect to a parameterized critic function. The key innovation is leveraging adjoint matching, a technique from generative modeling, to utilize the critic's action gradient directly, thereby avoiding the numerical instability that arises from backpropagation through multi-step denoising processes.

### Strengths
1. Novel and well-motivated approach: The paper addresses a fundamental tension in continuous-action RL between policy expressivity and optimization tractability.
2. Strong theoretical foundation: The method builds on solid mathematical principles from stochastic optimal control and adjoint methods.
3. Comprehensive experimental evaluation: The paper includes extensive comparisons against eight representative baselines across multiple challenging domains. The categorization of baselines (DDPG-based, value-only, gradient-with-approximations, post-processing, Gaussian) is thoughtful and covers the landscape of existing approaches well.
4. Practical considerations: The authors provide detailed implementation guidance, including discrete approximations, gradient clipping, and ensemble critic training.
5. Clear presentation: The paper is generally well-written with good motivation, clear problem formulation, and helpful visual aids.

### Weaknesses
1. Insufficient stability analysis in experiments: Although the success rate is reported, the paper lacks metrics that directly assess stability, such as standard deviation across seeds or training loss curves. These would provide concrete evidence of the claimed stability advantages.
2. Limited domain diversity: All evaluated tasks are from OGBench with similar characteristics (long-horizon, sparse rewards). The generalizability to dense reward tasks (e.g., MuJoCo locomotion in D4RL), tasks with different action space dimensions, shorter horizon tasks, and continuous reward settings remains unclear. This significantly limits the assessment of the method's broad applicability.
3. Impact on Q-value estimation: The paper does not analyze how QAM affects the critic learning process. Since the policy used for target value backup changes during training, this could introduce additional variance or bias in TD targets. An empirical analysis that shows Q-value estimation error over training, compares learned Q-values with Monte Carlo returns, and examines the impact on critic ensemble disagreement would provide important insights.
4. Incomplete sensitivity analysis: The impact of ensemble size and pessimistic coefficient is not studied, despite being set to specific values. No ablation on the choice of "memoryless" SDE vs. alternative noise schedules. The interaction between temperature and clipping is not explored.
5. CGQL is presented as a "novel baseline" without prior work validation. Several baselines (DSRL, FEdit) have been modified from their original implementations, which may not represent their optimal performance. The choice to disable best-of-N sampling for IFQL may unfairly handicap it. 
6. Missing ablations: What happens if using the "basic" adjoint matching (Eq. 12) instead of the "lean" version (Eq. 14)? How does QAM perform with different critic architectures or learning rates?
7. Limited theoretical analysis of stability: While the paper claims that adjoint matching avoids instability by not backpropagating through the residual flow model, the theoretical justification is largely intuitive. The discussion on pages 5-6 about ill-conditioned gradients is informal. A rigorous stability analysis (e.g., condition number bounds, gradient variance analysis, or convergence rate guarantees) would significantly strengthen the claims.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a TD-based reinforcement learning algorithm to efficiently optimize an expressive diffusion or flow-matching policy with respect to a parameterized value function. While the effectiveness is validated, there is a lack of theoretical proof for the proposed approach, and the experiments are not comprehensive.

### Strengths
1. Novel algorithm
2. Effectiveness validated

### Weaknesses
1. Lack of theoretical proof
2. Incomprehensive experimental setting

### Questions
1. The paper lacks a theoretical proof to demonstrate the stability of the proposed method and to explain why it outperforms existing methods.  

2. It is recommended to further demonstrate the applicability of the proposed method on tasks beyond long-horizon tasks, such as the MuJoCo tasks in D4RL.  

3. A comparison with more state-of-the-art offline-to-online DRL methods should be included in the experiments.  

4. How does the performance of the proposed method vary with changes in the number of critics, and does the number of critics affect Q-value evaluation?  

5. The proposed method includes many components. It is recommended to analyze the time and space complexity of the proposed method, and to compare its throughput and memory usage with baseline methods in the experiments to assess practical hardware efficiency.  

6. How does the performance of the proposed method change as the offline dataset size decreases and its quality declines?

### Soundness
2

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
The paper introduces Q-Learning with Adjoint Matching (QAM), an actor-critic algorithm with flow-matching policies that uses adjoint matching (a recent technique from generative flow modeling). The paper tackles the problem of optimization instability in backpropagating the critic through multi-step denoising processes in flow or diffusion models. Previous methods either discarded the action gradient and solely relied on action values or only used one-step approximations. QAM uses adjoint matching (AM) to incorporate the critic’s action gradient in training a flow policy subject to a prior constraint. QAM assumes access to a fixed behavior (prior) flow policy and then learns a residual flow policy, which is optimized towards the AM objective. Leveraging the convergence guarantees of AM, the paper claims this converges to the optimal prior regularized policy. The experiments demonstrate QAM has lower hyperparameter sensitivity and better performance compared to many baselines in offline and offline-to-online settings on various OGBench domains.

### Strengths
The paper is well-motivated: the problem being addressed is clear, and there are clear arguments for using AM.

The experiments compare to many different algorithms.

### Weaknesses
The theoretical guarantees are not detailed, and the word convergence only appears in the intro and conclusion. The derivation relies strongly on SOC objectives from Adjoint Matching, and a clear motivation of the formulation in RL setting, with clearly stated assumptions can make the paper self-contained.

The experiments are quite comprehensive in scope, with many algorithms, which is laudable. However, there are currently three key issues to address. 

The first is around how confidence intervals are computed. To quote:
“We report the means and the 95% bootstrapped confidence intervals are computed over 4 seeds”
This is an issue. You cannot get bootstrap confidence intervals from 4 seeds. The number of seeds should be increased, or CIs should only be reported using aggregate performance across environments. I understand experiments can be expensive, however, it is key to only make claims supported by the evidence. It is possible to only show qualitative behavior, such as individual runs in an environment (with 4 lines), and then make stronger claims aggregating across environments. Or, of course, the number of runs (seeds) can be increased.

The second key concern is how hyperparameters were chosen. Appendix D lists the hyperparameters and states that they needed to be tuned for each domain. How was this done? When tuning per environment, it is key to control for the number each method gets to tune, and in general can be difficult to keep fair. Tuning hypers can be very misleading in terms of the actual utility of an algorithm, because different settings of hypers can essentially give you a different algorithm. Ideally, it is better to tune across environments or at least explain why the current tuning is valid and not misleading.

The third key concern is a missing baseline (in Table 1): a strong offline RL method that is not focused on diffusion policies. RLPD is used later, which just uses a Gaussian, but it is not an offline method. In general, for both the offline setting and the offline-online setting, it could be more informative to consider IQL. Note also that one baseline IFQL uses IQL, so it is sensible to include for this reason too. It is a strong good baseline for the offline experiments. And, for offline-online, it is also likely a stronger baseline that RLPD, since RLPD does not make as much use of the offline data as IQL. I think it is key to include IQL, or another method that has been shown to perform well in offline and offline-online, such as In-sample Actor Critic (see https://arxiv.org/abs/2302.14372). This method has an update that is a lot like SAC with a small modification to handle out-of-distribution actions.


(Putting Minor Comments here, because there is not separate box for them. These Minor Comments are not big weaknesses.)
1. “more convenient to directly use the same offline RL objective for both offline pre-training and online fine-tuning” Why is it more convenient? There are stronger reasons for doing so and should be listed here.

2. The paragraph at the beginning of Section 2 is too long and should be split up into multiple ideas. Long paragraphs throughout should be broken into multiple, with clear topic sentences, for readability and clarity.

3. It is a bit confusing to use v and r in the formulas for adjoint matching, even if that is their original terminology, due to the clash with value functions in RL and the fact that you will use q rather than r. You could simply use q right away, and consider a different variable for v

4. The sensitivity analysis is highly appreciated, but there could be just a little bit more there. In the sensitivity analysis in Figure 4, there is not a lot of difference between the values of tau nor without gradient clipping. This suggests more values of tau should be tested to truly see when there is failure. This is not critical, but would make this result more useful.

### Questions
Questions
1. How do you plan to address the issue with CIs based on 4 seeds?
2. Can you explain how hyperparameters were tuned? (see above)
3. Is it possible to run an experiment compared to a stronger offline method, like IQL or Insample AC, with a Gaussian policy? Or can you explain why you do not think this is needed?

The current decision is based on these omissions, but I am willing to adjust my score based on the responses from the authors. The current scores reflect uncertainty on the outcomes, and Soundness and Contribution would particularly be increased when addressing these omissions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the instability that arises when backpropagating policy gradients through action flows, which often leads to practical difficulties when optimizing a policy with a learned Q-network. To tackle this issue, the authors employ a recent technique called adjoint matching. Rather than training a single flow-matching policy solely on an offline dataset, they utilize the behavior policy to generate adjoint states that provide direct step-to-step supervision signals for the residual policy. This results in a stable adjoint-matching objective that enables local supervision over the residual flow while still recovering an optimal prior-regularized policy. The approach is compatible with settings where a prior policy is provided, and the authors demonstrate its effectiveness through experiments in both offline and offline-to-online scenarios.

### Strengths
The paper addresses a critical challenge in flow-matching-based policy learning. While the current setting is limited to cases where a behavior policy is available, the authors introduce a well-grounded formulation that establishes a clear connection to adjoint-matching techniques in generative modeling. The resulting policy maintains strong expressivity, enabling more diverse action distributions to be represented. I believe this capability can offer benefits to reinforcement learning that extend beyond the specific experimental settings evaluated in the paper.

### Weaknesses
The method relies heavily on accurate action gradients, though this dependence is common across many RL algorithms. The authors address this challenge in practice, for example by training a critic ensemble for more reliable gradient estimates. That said, there may be further opportunities to explore more advanced techniques to improve gradient accuracy.

The evaluation is limited to OGbench domains. Including results on benchmarks with more diverse characteristics would further strengthen the paper’s contributions. For example, demonstrating performance across policies of varying quality could offer additional insights into the method’s robustness and general applicability, dense rewards, offline datasets of different qualities, etc. If an offline dataset is collected using a policy with low behavioral diversity, it is unclear how this would impact the effectiveness of the proposed approach. A discussion or empirical analysis on this aspect would be valuable.

### Questions
I am also interested in the computational cost of the method—approximately how many GPU hours are required to train a policy? Also, is there any observed numerical instabilities for high dimensionality?

### Soundness
3

### Presentation
2

### Contribution
3
