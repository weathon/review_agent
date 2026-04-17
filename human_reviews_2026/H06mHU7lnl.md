# When does Predictive Inverse Dynamics Outperform Behavior Cloning? Exploring the Role of Action and State Uncertainty

- Decision: Reject
- Scores: 4, 4, 2, 6, 4

## Abstract
Offline imitation learning aims to train agents from demonstrations without interacting with the environment, but standard approaches like behavior cloning (BC) often fail when expert demonstrations are limited. Recent work has introduced a class of architectures we call predictive inverse dynamics models (PIDM), which combine a future state predictor with an inverse dynamics model (IDM) to infer actions to reach the predicted future states. While PIDM often outperforms BC, the reasons behind its benefits remain unclear.
In this paper, we analyze PIDM in the offline imitation learning setting and provide a theoretical explanation: conditioning the IDM on the predicted future state reduces variance, whereas predicting the future state introduces bias. We establish conditions on the state predictor bias for PIDM to achieve lower prediction error and higher sample efficiency than BC, with the gap widening when additional data sources are available. The efficiency gain is characterized by the variance of actions conditioned on future states, highlighting PIDM’s ability to reduce uncertainty in states where future context is informative. We validate these insights empirically under more general conditions in 2D navigation tasks using human demonstrations, where BC requires up to five times (three times on average) more samples than PIDM to reach comparable performance. Finally, we extend our evaluation to a complex 3D environment in a modern video game with high-dimensional visual inputs and stochastic transitions, showing BC requires over 66% more samples than PIDM in a realistic setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a thorough theoretical and empirical analysis of Predictive Inverse Dynamics Models (PIDMs) in the context of offline imitation learning, specifically contrasting them with the standard approach of Behavior Cloning (BC). The central research question is to explain ​​why and under what conditions PIDM achieves superior sample efficiency compared to BC​​, particularly in low-data regimes where expert demonstrations are scarce. The authors' primary contribution is a ​​novel theoretical framework​​ that quantifies the advantage of PIDM. They demonstrate that, under the assumption of a perfect state predictor, the expected prediction error for actions in PIDM is provably lower than or equal to that of BC. The theory further establishes a direct connection between this error reduction and sample efficiency gains, showing that PIDM can achieve a given performance level with fewer demonstrations.

### Strengths
1. A primary strength of this paper is its significant contribution to moving beyond empirical observation to a ​​principled theoretical understanding​​ of Predictive Inverse Dynamics Models (PIDMs). The authors provide a clear and compelling theoretical framework that explains why PIDM outperforms Behavior Cloning (BC) in sample efficiency.
2. Another strength of this paper lies in its ​​exceptionally thorough and multi-faceted experimental validation​​, which provides compelling, direct evidence for the theoretical claims. The empirical strategy is meticulously designed to test the theoretical predictions under increasingly general conditions, creating a powerful and convincing evidence chain.

### Weaknesses
While the paper makes a valuable contribution by providing a theoretical foundation for PIDMs, a significant weakness lies in the ​​limited novelty of its methodological contribution​​. The core architecture under study—decomposing policy learning into a future state predictor and an inverse dynamics model—is not a new invention but a re-investigation of an existing and increasingly popular paradigm. The paper itself positions the work as an explanation for "recent work", acknowledging that it is analyzing a pre-existing class of methods. The paper's primary contribution is thus ​​retrospective analytical justification​​ rather than ​​prospective algorithmic innovation​​.

The goal-conditioned or future-conditioned imitation learning paradigm, which PIDM embodies, has been extensively explored in various forms, such as in goal-conditioned behavior cloning, hindsight experience replay, and other methods that leverage future state information to guide policy learning. The paper's approach of using a state predictor (which is analogous to an action-free forward model) and an IDM policy is a valid but well-established instantiation of this idea. The authors do not propose a novel architecture, a more efficient training scheme, or a significant modification to the existing PIDM framework. Instead, they offer a rigorous theoretical explanation for why an existing approach works well, which, while intellectually valuable, does not advance the methodology itself.

### Questions
Is it possible to validate PIDM's performance on standard continuous control benchmarks, such as MuJoCo locomotion tasks?

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
This work aims to study the reason why predictive inverse dynamics models (PIDMs) outperform behavior cloning (BC), even in low-data regimes. The core contribution lies in the theoretical analysis, which shows that conditioning on future states reduces action uncertainty, leading to improved sample efficiency.

### Strengths
1. The paper discusses a novel perspective on the PIDM framework, offering insights into *why* PIDM can outperform BC, particularly under low-data regimes. Unlike prior work that primarily focuses on scaling up the training of state predictors, this work shows the fundamental role of *uncertainty reduction* in explaining PIDM’s improved sample efficiency.
2. By simplifying high-dimensional inputs to a 2D navigation setup, the analysis effectively isolates the key factors contributing to PIDM’s advantage. This design disentangles the benefits of learned representation quality from those of the decomposed learning paradigm, showing that PIDM’s gains arise from reduced action uncertainty when conditioning on future states.

### Weaknesses
Overall, the title suggests a broad discussion of *“when”* PIDM outperforms BC, but the analysis is conducted under a much narrower and idealized setting.

1. The theoretical finding — that conditioning on future states can reduce the uncertainty of predicting actions — is intuitive: if the policy knows the goal, predicting the corresponding action naturally becomes easier. This, however, transfers the difficulty from predicting actions to predicting future states. The paper does not analyze the relative complexity of these two estimation problems or provide conditions under which the proposed advantage provably holds.
2. The theoretical framework assumes near-perfect state prediction, an assumption that rarely holds in real-world, high-dim environments — especially under the low-data regime emphasized by the authors. In practice, imperfect state predictors cause the IDM to be conditioned on OOD future states and leading to compounding error. The paper would benefit from a discussion of how state-prediction errors affect the claimed theoretical guarantees.
3. The experiments are relatively simple, even in the 3D game environments, and do not consider settings with high action complexity. Prior work has shown that tasks involving precise control (e.g., dexterous manipulation in robotics) often expose the limits of models that rely on accurate future-state prediction. A more systematic empirical study — examining how state and action complexity influence PIDM’s advantage over BC — would be necessary to substantiate the paper’s general claims.

I encourage the authors to either reframe their contribution around the idealized theoretical setting or extend the analysis to address the above concerns. I will carefully reconsider my recommendation after the rebuttal.

### Questions
1. How does varying the number of future steps $k$ used during inference affect the results?

2. In the 2D environment, how large is the state-prediction error (or variance) in regions with high $\Delta(s)$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a theoretical analysis explaining when and how **predictive inverse dynamics models (PIDM)** can outperform **behavior cloning (BC)** in offline imitation learning. They show that the conditioning on future states reduces the prediction variance of the policy, which benefits the sample efficiency of PIDM compared to BC.

### Strengths
**Motivation:** The paper tackles a relevant and underexplored question — why predictive inverse dynamics (PIDM) can outperform behavior cloning (BC) in offline imitation learning.

**Clarity:** The analysis is mathematically sound and well-structured, deriving intuitive results.

**Coherent:** The experimental results align well with the theoretical predictions.

### Weaknesses
- The theoretical analysis assumes point estimators, whereas recent imitation learning approaches (e.g., *Diffusion Policy*, *Flow Matching*) model full action distributions. Including a diffusion-policy baseline or an experiment using a distributional variant of PIDM would strengthen the paper’s practical relevance and clarify whether the predicted sample-efficiency advantage persists under more expressive policy classes.

- Equation (9) indicates that the efficiency gain depends on the stochasticity of the environment and the stochasticity of the expert policy. The current experiments introduce environmental noise but do not disentangle these two factors. Including controlled experiments that vary each source of randomness independently would provide valuable insight into how much of the observed efficiency gain is due to environmental stochasticity versus demonstration variability.

- The experimental environments are limited to navigation tasks, while imitation learning is widely applied to domains such as manipulation, locomotion, and autonomous driving. I would strongly encourage the authors to include a more diverse set of environments and tasks to demonstrate the generality of the assumptions.

- On page 8, the same symbol $k$ is used for both the prediction horizon ("$k$-step") and the number of clusters ("$k$-means"). While these meanings are unrelated, the reuse of notation can be confusing to readers. I would recommend adopting distinct notations.

### Questions
One open question is how practitioners can determine when to prefer PIDM over standard behavior cloning in practice. The current analysis shows that PIDM outperforms BC when future-state conditioning reduces action uncertainty, but it remains unclear how a user can recognize such conditions in a new domain. Could the authors provide more practical guidance or diagnostic criteria (e.g., measurable indicators of state–action ambiguity or future predictability) that would help a user decide when PIDM is expected to yield a tangible benefit?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Predictive inverse dynamics models (PIDM) sometimes perform significantly better than Behavioral Cloning (BC), but the underlying reason is underexplored. This paper provides theoretical proof and insight for the sample efficiency of PIDM versus BC, including:
1. The prediction error of an optimal estimator for PIDM is always less than or equal to that of BC. The gap is characterized by the expected conditional variance of actions given future states, averaged over the current state distribution.
2. Under asymptotic efficiency assumptions, this uncertainty reduction can translate into sample efficiency gains.

The paper conducts visualization experiments on simple 2D tasks to support the points listed above and further validates these findings in a complex 3D environment, demonstrating significant sample efficiency gains over BC in both settings.

### Strengths
- The paper's main strength is its original theoretical analysis formally explaining why PIDM can be more sample-efficient than BC. It proves that PIDM's optimal prediction error is always less than or equal to BC's, providing a significant and principled reason to use PIDM in situations with high action uncertainty. This provides a strong theoretical backbone for a previously unexplained empirical observation.
- The experiment and validation are of high quality and clearly support the theory. The 2D navigation experiments are particularly clear, as the author visualizes the theoretical error gap and shows the PIDM policy learns to attend to future states only in these high-uncertainty regions. The validation in a complex 3D world confirms these findings are applicable to real-world tasks.

### Weaknesses
- The connection between Theorem 2 and efficiency gains in Section 4 is not entirely clear and easy to follow. Theorem 2 provides a valuable theoretical link based on an "asymptotic efficiency" assumption. However, the paper's main empirical results are in the "low-data regime" (e.g., 1-30 trajectories). The paper would be more convincing if it discussed whether these asymptotic assumptions are expected to hold in such low-data settings, or how this potential mismatch affects the practical gains observed.
- The scope of the comparison is narrow. The paper compares PIDM against a simple, point-estimator BC.

### Questions
- Around line 48, the Introduction states a focus on the "low-data regime" where "no additional data can be assumed", but then links this to the "current AI landscape, where large foundation models are trained on massive datasets". This connection is confusing. Could the authors please clarify how their setting, which explicitly assumes no additional data, is "increasingly relevant" to a landscape defined by massive pre-training? Is the intended link about the fine-tuning or alignment of these large models to new tasks, for which supervision is limited?
- In Theorem 2, the conclusion that the sample efficiency ratio $\eta \triangleq \frac{n}{m} \ge 0$ seems trivial, as $n$ and $m$ are sample counts and must be positive. The more practical and significant result would be $\eta \ge 1$ (i.e., PIDM is at least as sample-efficient as BC, $n \ge m$). Does the theory guarantee $\eta \ge 1$? Based on Equation (9), this seems to depend on the ratio $\frac{C_\mu}{C_\xi}$. Can the authors clarify the conditions under which $\eta \ge 1$ is guaranteed by the theory?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why Predictive Inverse Dynamics Models (PIDM) outperform standard Behavior Cloning (BC) in offline imitation learning, particularly in low-data regimes. The authors theoretically show that decomposing policy learning into a state predictor and an inverse dynamics model reduces the expected prediction error relative to BC. The authors then provide empirical study to augment their theoretical contributions.

### Strengths
- Provides the formal theoretical result on why PIDM improves over BC.
- Proofs are simple and easy to follow.
- Empirical Study is provided on both 2D and 3D domains.

### Weaknesses
- The analysis does not quantify degradation when predictors are noisy or biased. At least Empirical tests under imperfect predictors would strengthen the overall idea. (Assumption of perfect or near-perfect state predictor)

- Missing comparison with recent BC variants that can also implicitly reduce uncertainty (e.g., implicit BC, diffusion BC) (Empirical limitation)

- Formal proof of theoretical results follow standard mathematical machinery with no novel analysis. (Novelty issue)

- Code is missing

### Questions
- In theorem 2, why does having $\eta \geq 0$ suffice? I thought $\eta \leq 1$ is what would make PIDM more sample efficient. Please correct me if there are any gaps in my understanding.

- In 2D experiment, you fix $k = 1$ whereas in 3D you train with multiple $k$ but evaluate at $k=1$. How sensitive are gains to $k$? Could you please sweep $k$ and report how the empirical $\Delta$ (and final performance) changes, balancing predictability of $s_{t+k}$ versus its disambiguation power?

- Your theory assumes access to a state predictor, while the experiments use non-learned predictors (instance-based in 2D; fixed-trajectory in 3D). Could you report results with a learned predictor (e.g., MLP/Transformer trained on $(s_t, s_{t+k})$​ to quantify how imperfect ‘future state distribution’ learning affects PIDM’s sample-efficiency advantage?

- Why is the code missing? Given that there are a number of experiments in the paper, how can I run and validate the claims? The anonymous code link or the zip file should have been provided in the supplementary file.

- Please fix minor formatting issues. For example, in line 107, after PIDM, ----- should not be there.

### Soundness
3

### Presentation
3

### Contribution
2
