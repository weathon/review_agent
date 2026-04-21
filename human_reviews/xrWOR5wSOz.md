# Replacing Implicit Regression with Classification in Policy Gradient Reinforcement Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 5, 6, 3

## Abstract
Stochastic policy gradient methods are a fundamental class of reinforcement learning algorithms. When using these algorithms for continuous control it is common to parameterize the policy using a Gaussian distribution. In this paper, we show that the policy gradient with Gaussian policies can be viewed as the gradient of a weighted least-squares objective function. That is, policy gradient algorithms are implicitly implementing a form of regression. A number of recent works have shown that reformulating regression problems as classification problems can improve learning. Inspired by these works, we investigate whether replacing this implicit regression with classification can improve the data efficiency and stability of policy learning. Toward this end, we introduce a novel policy gradient surrogate objective for softmax policies over a discretized action space. This surrogate objective uses a form of cross-entropy loss as a replacement for the implicit least-squares loss found in the surrogate loss for Gaussian policies. We extend prior theoretical analysis of this loss to our policy gradient surrogate objective and then provide experiments showing that this novel loss improves the data efficiency of stochastic policy gradient learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an innovative approach to enhancing the efficiency of policy gradient reinforcement learning by reformulating the implicit regression typically used with Gaussian policies commonly used in continous control as a classification problem. The authors introduce a novel surrogate loss, leveraging cross-entropy loss and softmax policies over discretized actions, and provide both theoretical analysis and empirical evidence supporting this new approach. Overall, the paper addresses a relevant challenge in reinforcement learning, with convincing results and a clear methodology.

### Strengths
*Theory*: The paper's proposal to replace regression with classification in policy gradient algorithms is compelling, backed by theoretical bounds showing reduced gradient norms for the cross-entropy-based loss. This theoretical contribution offers insights into the policy optimization algorithms in continous control.

*Experiments*: The paper presents comprehensive experimental results across several continuous control environments, effectively showcasing the advantages of the proposed classification-based surrogate loss. The consistency of the performance gains in data efficiency, stability, and convergence across diverse tasks solidifies the approach's applicability.

### Weaknesses
1. The notation $ l $, used initially to represent the Lipschitz constant of the policy network’s output, is later redefined as an index number (line 194). This dual usage is confusing and could benefit from more consistent notation.

2. Certain variables, like $c$, could be recalled where they are used for better readability. For instance, $c$ (defined in line 196) reappears in line 233, but its earlier definition is difficult to locate due to its inline placement in the formula.

### Questions
The author assumes that the policy network’s output is $l$-Lipschitz, which seems to be a strong assumption. Could you give some insights why the assumption holds in the experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates whether replacing implicit regression in policy gradient can improve the training efficiency of policy learning. It introduces a surrogate loss used to reformulate the implicit regression of continuous actions as a classification of discrete actions.  They empirically investigate the use of cross-entropy in the introduced loss as an alternative to the Gaussian policies

### Strengths
The paper is well-written and effectively discusses its connections with other methods in the literature, explaining the motivations behind its approach. It provides a thorough theoretical analysis of gradient norms, showing why the classification-based surrogate loss might offer advantages. Additionally, experiments on environments (such as continuous Mountain Car, HalfCheetah, and Ant) demonstrate that the classification-based approach generally improves data efficiency. The paper also includes thorough sensitivity and exploration ablation studies, along with well-documented hyperparameter tuning.

### Weaknesses
Discretizing action spaces may lead to performance limitations in domains that require precise control or have high dimensionality, which the authors acknowledge.
The contributions appear incremental, as the method builds heavily on prior work, particularly Imani and White (2018). The term "novel" used in the text may be an overstatement, as the core methodology closely follows established ideas.
The analysis focuses primarily on deriving bounds for gradient norms, suggesting that a smaller bound should lead to greater stability. However, this relationship is indirect, and the analysis does not rigorously connect the smaller gradient norms to concrete improvements in convergence rate. Additionally, key assumptions underlying the theoretical results are not clearly and explicitly stated, making it difficult for readers to assess the validity of the analysis. Furthermore, the propositions are not self-contained or fully descriptive

Minor weakness: The paper lacks clarity in section 5.2.1. It is unclear whether the 20 trials use different seeds, which is important for interpreting the robustness of the results.

### Questions
Did you run 20 trials with different seeds, or were the seeds the same for each trial?

I expected the method to perform poorly in higher-dimensional settings, yet the experiments show the opposite. Is there a theoretical justification for this result?

I will increase my score if you address my concerns in the weaknesses part.

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
The study introduces an innovative approach to the policy gradient (PG) objective utilized in Gaussian-based policy models within the field of reinforcement learning (RL). It redefines the conventional PG objective by framing it as a weighted squared loss function. In this formulation, the squared loss measures the discrepancy between the chosen action and the action prescribed by the policy, while the weighting factor is derived from the advantage measure, augmented by an additive constant. This conceptual reconfiguration enables the authors to devise a new surrogate function that transitions the methodology from a regression-based to a classification-based model. The paper substantiates the practical utility and enhanced performance of these proposed methods through a series of empirical validations.

### Strengths
1. The paper is intellectually interesting, as it proposes some novel angle to think about continuous control based on Gaussian distribution;
2. The presentation is clear;
3. The empirical results look promising.

### Weaknesses
I do not have many comments for this paper; so I will keep it short. My primary concerns are below. 

1. The reason for why softmax could be beneficial is not clear. Softmax, similar to sigmoid function, could have saturated gradient issue. This has been pointed out by existing work. The author cited one by Shivam et al. in the conclusion, but note that this issue is not restricted to nonstationary learning setting. 

2. The paper present some theoretical argument regarding bounded gradient norm, I doubt if it really supports the claim of improved sample efficiency. Note that a strong gradient signal could intuitively improve learning efficiency. 

Note that the cited paper by Ehsan et al about histogram loss, they claim improved generalization, rather than sample efficiency. 

3. Given the popularity of the deterministic policy, it might be interesting to see how the proposed method compares against deterministic control. Though this is not necessary to support the main claims of this paper.

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The recent work by Farebrother et al  (2024) showed that instead of training value functions by regression, discretizing the space and using classification instead can yield gains in terms of scalability and performance. Following along these lines, the current work aims to perform something similar for policy-based reinforcement learning. In particular, if we consider a Gaussian policy, the policy gradient w.r.t. $\mu$ is obtained by differentiating a surrogate loss that roughly looks like $A(x-\mu)^2$, which seems like a weighted regression loss (if we consider having target samples at all $x$). Along these lines, the authors change the objective to a weighted classification loss by discretizing the space, and swapping out the $(x-\mu)^2$ bit with a cross-entropy loss.
They performed experiments on applying this idea together with the A2C algorithm on continuous control tasks such as acrobot, mountain car, halfcheetah and ant.
The work also included some theoretical derivations showing that the gradient magnitude is smaller compared to a Gaussian policy.
There were also some other technical additions, e.g., the discretization is performed coordinate wise separately for each action dimension, the logits in each bit affect the probability in nearby bins as well etc. The performance on HalfCheetah went up to around ~2000 (on the Gymnasium version).

### Strengths
- The recent results about critic learning with classification are intriguing, so it is interesting to think about whether they are also applicable to policy-based learning.
- Related work appears adequately discussed.

### Weaknesses
- The performance of the proposed method on HalfCheetah is only around ~2000, whereas typical good performance on the Gym version is over 10000. Performance on other tasks like Acrobot and Ant, also does not seem close to competitive with good performance on these tasks. Moreover, whilst in the current paper, the A2C benchmark achieves around ~1000 on HalfCheetah, there exist other implementations (e.g., https://tianshou.org/en/v0.4.8/tutorials/benchmark.html) where the plain A2C also achieves ~2000, similar to the proposed method in the current paper. Furthermore, e.g., in Ant the performance of the proposed algorithm in the current paper is ~1500, whereas the tianshou benchmark A2C gets over 3000. From this point of view, the result is not convincing both in terms of whether it really improves over A2C, and also in terms of whether it would improve performance on algorithms that achieve more competitive performance, e.g., PPO.
- I was not convinced by the theoretical results. These results showed that the gradient magnitudes become smaller; however, gradient magnitudes themselves may not tell us about the optimization difficulty. Even just rescaling the objective function will change the gradient magnitudes, but will not change the optimization problem. Perhaps some other metric like the condition number, etc. would have been a more convincing theoretical result to me.
- I didn’t fully see why the method is described as a weighted classification instead of just saying that you are using a discrete policy representation and computing the policy gradient with your representation. The cross entropy is basically the log of the picked actions, but this is the same as in the policy gradient, so couldn’t the method just be seen as switching the Gaussian probability distribution to a discrete one, and applying the standard policy gradient methods?
- The experimental work is not thorough, and is mainly looking at return curves. Providing other kind of experimental evidence in addition to reward curves would be more convincing.

### Questions
Why is the performance not competitive?

Why do you consider the method as classification, instead of just thinking of it as changing the policy representation to a discrete one, and applying policy gradients with this new representation?

Small comment:
Equation 4 requires a stop gradient on $A$. Otherwise it is not an unbiased estimator of equation 1.

### Soundness
2

### Presentation
2

### Contribution
1
