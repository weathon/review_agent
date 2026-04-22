# Unleashing Flow Policies with Distributional Critics

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Flow-based policies have recently emerged as a powerful tool in offline and offline-to-online reinforcement learning, capable of modeling the complex, multimodal behaviors found in pre-collected datasets. However, the full potential of these expressive actors is often bottlenecked by their critics, which typically learn a single, scalar estimate of the expected return. To address this limitation, we introduce the Distributional Flow Critic (DFC), a novel critic architecture that learns the complete state-action return distribution. Instead of regressing to a single value, DFC employs flow matching to model the Q-value distribution as a continuous, flexible transformation from a simple base distribution to the complex target distribution of returns. By doing so, DFC provides the expressive flow-based policy with a rich, distributional Bellman target, which fully captures value uncertainty and offers a more stable and informative learning signal. Extensive experiments across D4RL and OGBench benchmarks demonstrate that our approach achieves strong performance, especially on tasks requiring multimodal action distributions, and excels in both offline and offline-to-online fine-tuning compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper seeks to enhance the expressiveness of value functions by learning value distributions instead of relying on expected scalar values. To achieve this, the paper employs diffusion models/flow matching to optimize self-bootstrapped value distributions, as these methods are well-suited for modeling complex distributions without parametric restrictions. However, once the value distributions are obtained, extracting the policy by maximizing these distributions becomes challenging due to the instability and inefficiency of optimizing through the full flow/diffusion chain. To address this, the authors apply distillation techniques along with quantile regression to transform the multi-step flow/diffusion ODEs/SDEs into a one-step noise-to-quantile distribution mapping. This simplification allows for efficient maximization of the distilled expected values to derive the policy. Experimental results on D4RL and OGBench demonstrate the strong potential of distributional RL and highlight the benefits of using diffusion/flow models to capture the risk and uncertainty inherent in value estimations.

### Strengths
1. Using diffusion/flow to model the value distributions is novel. Recently, there are also some methods that do the similar things, but they are concurrent works. 
2. The proposed methods seem easy to optimize, since all components are optimized in a decoupled manner, therefore has few limitations on training instability.
3. The empirical performance on D4RL and OGBench are strong, demonstrating the promises of combining distirbutional RL and flow/diffusions.
4. The presentation is good, easy to follow.

### Weaknesses
1. The authors have already obtained a distilled value distribution but still use its mean value to derive the policy, as shown in Section 3.3. This approach undermines the benefits of value distribution modeling, as the mean value could be directly regressed without incurring the additional computational costs.  It's necessary to provide a solid discussion as well as more empirical supports to explain why directly regressing to the mean value is suboptimal.

2. Since the authors have already obtained the entire value distribution, will it bring additional benefits by not using the mean value, but considering other properties of distributions?

3. I don't know if the self-bootstrapped value distribution learning in Eq (4-5) is stable and whether it requires additional tricks to stablize the training. Providing more details to optimize Eq (4-5) or showing some empirical experiments to visualize the learned value distirbution would be good and interesting to address this concern.

4. Figure 1 can be further beautified.

Currently, I have no additional major concerns. The manuscript is relatively strong, presenting an approach that is easy to optimize, the idea is relatively novel, the empirical results are strong. However, more empirical evidence or discussion is needed to address the points raised above. I will reconsider the score after the rebuttal.

### Questions
Please see weaknesses for details.

### Soundness
3

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
2

### Summary
This paper addresses the limitation where expressive flow-based policies are often hindered by simple critics that only estimate a single expected return value. To overcome this, this paper introduces the Distributional Flow Critic, which employs flow matching within a two-stage distillation architecture to learn the complete distribution of state-action returns, providing a richer learning signal. Experiments show that integrating DFC achieves strong performance, especially on tasks needing multimodal actions, in both offline and offline-to-online settings.

### Strengths
1. The paper's motivation is clearly articulated. It effectively highlights the mismatch between expressive actors and simplistic scalar critics in existing methods, particularly relevant given that flow policies can capture multimodal behavioral distributions.
2. The paper presents convincing results across multiple benchmarks, demonstrating that DFC achieves significant performance improvements over state-of-the-art baselines. The ablation studies are well-designed and validate the necessity of combining Flow Matching and Quantile Regression.
3. The paper is well-presented, making the methodology and results relatively easy to read and understand.

### Weaknesses
1. The novelty of this paper is somewhat limited. The main contribution appears to be the integration of distributional RL concepts into the existing Flow Q-Learning framework. Specifically, it involves modeling the return as a distribution to improve the critic's expressiveness and includes architectural modifications to enable the flow policy to leverage these distributional returns.
2. This paper does not provide a theoretical justification or convergence analysis of the proposed two-stage critic. There is no discussion of whether the distillation preserves Bellman consistency or introduces bias in the return distribution.
3. This paper lacks a discussion of computational overhead. While it claims to avoid the BPTT problem, the proposed framework is quite complex, notably requiring the training of two actors and two critics. This raises concerns for the reviewer regarding the potential introduction of significant additional computational load. An analysis of the computational cost would substantially strengthen the paper's claim regarding efficiency.

### Questions
1. Can the authors clarify what novel challenges, distinct from standard Distributional RL, arise when introducing distributional returns to flow policies, and how does the proposed architecture address them?
2.The reviewer notes that during the actor update, the expectation of the distributional return provided by the critic is used in the objective. Although this is a common practice in distributional RL, does utilizing only the mean potentially undermine the rich expressiveness of the distributional return itself within the context of a flow policy? For instance, might properties like multimodality in the return distribution be lost or significantly attenuated when reduced to a single expected value for the policy gradient calculation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Distributional Flow Critic (DFC), a critic architecture designed to learn the full distribution of state-action returns rather than a single scalar estimate. DFC extends Flow Q-Learning (FQL) by employing (1) a flow-matching critic that models the return distribution and (2) a quantile regression–based distillation step to stabilize learning. The distilled critic is then used by a flow-policy to maximize expected Q-values. The motivation is that a distributional critic can better match the expressivity of flow-based actors and provide richer learning signals.

### Strengths
1. Strong empirical results: DFC achieves impressive performance on challenging benchmarks such as antmaze-giant-singletask and puzzle-4x4-singletask, outperforming strong diffusion and flow-based baselines.
2. Stable architecture: The proposed two-stage critic design successfully addresses the instability of directly training a flow-based critic by leveraging distributional distillation.
3. Comprehensive evaluation: Experiments cover both offline and offline-to-online reinforcement learning settings, demonstrating robustness and sample efficiency.

### Weaknesses
1. Limited conceptual novelty: The distillation of a critic distribution from a flow-based model is conceptually similar to existing policy distillation techniques (e.g., FQL). The contribution mainly lies in adapting known ideas to the critic side.
2. Dependency on FQL performance: DFC is tightly coupled with FQL; it improves upon FQL where FQL performs well but fails in the same settings where FQL struggles. This raises questions about whether the critic is truly the bottleneck.
3. Lack of ablation clarity: While the ablation in Table 5 compares “Flow-only” and “Distributional-only” critics, it would be stronger if supported by explicit variance metrics or training curves demonstrating instability versus convergence.
4. Insufficient empirical evidence for bottleneck claim: The claim that “the critic is the bottleneck” is asserted but not convincingly demonstrated through diagnostics such as critic loss analysis or policy entropy trends.

### Questions
1. Figure 2 visualization: The figure effectively summarizes improvements but is visually overwhelming. The authors could highlight only a subset of key representative tasks to enhance readability.
2. Derivation of sample $z_j^{'}$ is slightly confusing since in Equation 4 it is written as function evaluation while in the text it is mentioned to be computed by solving the ODE
3. Is this exclusive to flow policy? The central claim—“a distributional critic unlocks superior performance for flow-based policies”—is convincing but overly restrictive. The authors should clarify whether DFC can also serve as a general-purpose critic for non-flow actors (e.g., ReBRAC, IQL). Testing on at least one non-flow baseline would strengthen generality.
4. Many prior works stabilize critic learning through ensembles. It would be helpful to specify whether DFC uses any ensemble mechanism, or whether the combination of flow matching and quantile regression alone ensures stability. Empirical comparison with ensemble-based baselines would clarify this.
5. The meaning of the arrows (“→”) is unclear and should be explained—do they indicate training progression or pre/post fine-tuning results? Additionally, the claim that “FC suffers from training instability” should be supported with quantitative evidence such as standard deviation across seeds or critic loss variance
6. The paper attributes FQL’s failure cases to the critic bottleneck. However, some tasks (e.g., large antmaze variants) remain unsolved even with distributional critics. A more nuanced discussion—e.g., whether performance is limited by actor expressivity, data coverage, or Bellman extrapolation—would be appreciated.
7. The claim in the paper is that the distributional Bellman operator will help the performance. Could you also combine other existing distributional methods with the FQL for a fair comparison to prove that DFC is novel?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes Distributional Flow Critic (DFC), a novel critic architecture that models the full return distribution using flow matching for offline and offline-to-online reinforcement learning. DFC addresses the information bottleneck in flow-based policies caused by scalar critics. The method distills a multi-step target flow critic into a single-step critic via quantile regression, avoiding backpropagation through ODE solvers. Core contributions include: (1) A two-stage critic design mitigating training instability (§3.2, Fig. 1), (2) Integration with Flow Q-Learning actors (§3.3), and (3) Empirical gains over baselines on 73 D4RL/OGBench tasks, with 10% average improvement in offline RL and superior fine-tuning performance.

### Strengths
- Novel Distributional Critic Architecture. The two-stage flow critic elegantly circumvents backpropagation through ODEs while capturing full return distributions. 
- Rigorous Multi-Benchmark Validation. Extensive tests on 73 tasks show consistent gains over FQL on OGBench manipulation.
- Effective Offline-to-Online Transfer. Fine-tuning with online data improves FQL effectively on long-horizon tasks.

### Weaknesses
- lack of an ablation study to illustrate the effectivenesss of different components. For example, how do different expectile hyperparameter $\hat\tau$ and sample numbers $M$ influence the performance?
- Modeling the critic $Q(s,a)$ as a distribution is quite eccentric. That is, the definition of $Q(s,a)$ is the expectation of the returned value, a number instead of a distribution.  This manner can also lead to unstable training due to the much more severe error propagation of the TD estimator, as shown in  Eq.4. If the author wants to claim the necessity of such a critic-modeling method, an intuitive toy example/experiments should be performed.
- The time-consuming comparison between the proposed DFC and other flow policy-based methods should be presented.

### Questions
- Can you theoretically show how the critic trained in DFC will converge?

### Soundness
2

### Presentation
2

### Contribution
2
