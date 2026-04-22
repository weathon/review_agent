# ASAP: Adaptive Sliding Agnostic Poisoning Attack on Federated Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
The primary risk in the federated learning (FL) framework arises from the potential for manipulating local training data and updates, known as a poisoning attack. Among various attack strategies, agnostic attacks have emerged as a significant category that attempts to operate without explicit knowledge of the server's aggregation rules (AGRs). However, existing AGR-agnostic attacks still suffer from a critical dependency: they rely heavily on staying inside the natural per-coordinate variance of honest client updates. These attacks typically operate by analyzing benign clients' gradient patterns, statistical properties, and behavioral characteristics to strategically position their malicious updates. Therefore, to overcome these fundamental limitations of current AGR-agnostic attacks, this work presents the Adaptive Sliding Agnostic Poisoning Attack (ASAP) on FL, which can adaptively, robustly and precisely manipulate the degree of poisoning without the knowledge of AGRs algorithm of the server. Instead of relying on benign client patterns, ASAP incorporates Adaptive Sliding Model Control (ASMC) theory --- a sophisticated robust nonlinear control framework that enables adaptive attack. We implement our attack through comprehensive experiments on state-of-the-art (SOTA) Byzantine-robust federated learning methods using real-world datasets. These evaluations reveal that ASAP significantly outperforms all existing agnostic attacks while maintaining complete independence from benign client information, representing a fundamental advancement in FL attack strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ASAP (Adaptive Sliding Agnostic Poisoning), a new poisoning attack on federated learning (FL). It argues that previous "agnostic" attacks are flawed, as they still rely on estimating benign client statistics. ASAP overcomes this by applying Adaptive Sliding Mode Control (ASMC). It models the FL process as a dynamical system, treating the unknown server aggregation rule (AGR) and benign client updates as a single "disturbance." ASAP then uses a Fourier series approximation to estimate this disturbance to generate a malicious "control law" update.

This novel approach allows the attacker to precisely steer the global model to a predefined target accuracy. The attack is truly agnostic, requiring no knowledge of the server's AGR or benign data. It is also tunable, allowing the adversary to control the attack's speed and target.

Experiments show ASAP outperforms SOTA agnostic attacks like LIE and Min-Max. It successfully forces the model to converge to specific, desired accuracies, even against robust defenses. The attack is also far more efficient, achieving its goal in a fraction of the communication rounds. The authors conclude that ASAP represents a new, more dangerous class of controllable attacks.

### Strengths
- The attack is completely independent of the knowledge of the AGR and does not need to estimate benign statistics unlike previous methods, which is very impressive.
- ASAP can precisely manipulate the global model to converge to a predefined target accuracy.
- Using ASMC from control theory to the FL poisoning problem is a novel contribution.
- The effectiveness is validated against a wide range of datasets and defenses and compared with a range of baseline attacks.

### Weaknesses
However, there are a few flaws in the paper that need to be addressed:

- The target accuracy values chosen are too high and do not represent a strong attack. A reader would be more interested in the results when the target accuracy is low, like the ones that the baseline attacks (like LIE and min-sum) achieve.
- Figure 1 says that the attack objective is chosen as the closest point to the global optima. It is unclear how an attack can be effective if it is so close to the optima. Does the figure represent a real training scenario, or is it meant for illustration purposes only?
- No analysis of per-round detection statistics was made in the paper - how many times were the malicious gradients actually flagged by the AGR - it would be interesting to see the TP/TN/FP/FN values to see how better ASAP is from the other attacks
- The percentage values in Table 1 seem to be incorrect. Table 3 has many incomplete cells, omits the accuracy data but shows high unexplained percentage values which seem to be flawed.
- In Fig 4, most attacks seem to work better than ASAP. It needs to be explained more clearly in the paper what the figure actually conveys. Alternatively, the authors could use a low target accuracy for a better comparison with the other attacks.
- There are many inconsistencies in the math (for eg, Eqn 17 and 26). Further, e_t is defined as the model error at places, and as the estimator error at other places, creating a logical gap in the paper's central proof.
- The paper does not explain how the byzantine-robust defenses fail against ASAP. 
- There are numerous typos in the paper starting from the first line of the introduction, to an incorrect citation of data poisoning (Line 41). There are more typos and inconsistencies further down the paper.

### Questions
- Why do the byzantine robust AGRs fail against ASAP? What is the False negative rate of malicious grad detection.
- Why is the target accuracy set so high?
- What is the non-iid bias in the primary results in Table 1?
- What is the evidence that backs Figure 1 that the attack objective is closest to the optimum
- Since the training is dynamic, how long does it take for ASAP to learn the orthonormal basis functions for Fourier approximation?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an adaptive poisoning attack in federated learning inspired by sliding-mode control. The attacker treats the global aggregation and benign dynamics as unknown disturbances and designs a controller (with a Fourier approximation term) to steer the global model toward a target trajectory or accuracy. The authors claim the method is aggregation-rule-agnostic and capable of precise convergence control. Experiments on standard datasets and multiple robust aggregators are reported.

### Strengths
1. Presents a novel angle by introducing control theory concepts into the FL security context.
2. Attempts to achieve controllable poisoning outcomes, which are more flexible than conventional accuracy-degradation attacks.
3. Evaluates across different robust aggregation rules and datasets, demonstrating generality in experimental setup.
4. Includes preliminary theoretical reasoning to motivate the method's convergence behavior.

### Weaknesses
1. The theoretical justification currently relies on continuous-time scalar analysis, and the transition to discrete, high-dimensional FL training is not fully formalized.
2. Certain steps (e.g., influence estimation) are described conceptually but lack clarity regarding feasibility under non-differentiable robust aggregators.
3. The method architecture combines multiple components (control law + Fourier approximation) and could benefit from clearer motivation for each module from an FL-threat-model perspective.
4. Experimental evaluation focuses on outcome metrics, while analysis of stealthiness or detectability is limited.
5. Several implementation details are abstracted, which may affect reproducibility and clarity for practitioners.

### Questions
1. Could the authors clarify the assumptions required for extending the theoretical analysis to discrete FL rounds?
2. How is influence or sensitivity information approximated in settings with non-differentiable aggregation rules?
3. Have the authors considered evaluating against dynamic anomaly-based defenses beyond static robust aggregators?
4. How sensitive is performance to the choice of Fourier components, and is there a principled selection approach?
5. How does the method behave when malicious clients participate sporadically or in low proportion?

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
4

### Summary
The paper proposes ASAP (Adaptive Sliding Agnostic Poisoning Attack), a novel poisoning attack framework for federated learning. ASAP leverages Adaptive Sliding Mode Control (ASMC) combined with Fourier series approximation to estimate unknown aggregation behaviors and precisely manipulate the poisoning process without prior knowledge of aggregation rules (AGRs). The authors provide theoretical analysis and experiments on CIFAR-10, MNIST, and Tiny ImageNet, showing that ASAP outperforms existing AGR-agnostic attacks such as LIE, Min-Max, and Min-Sum.

### Strengths
+ The idea of introducing adaptive control theory (ASMC) into federated attack design is original and technically interesting.
+ The method does not rely on benign client statistics or AGR structure, which potentially increases applicability in black-box settings.
+ The authors provide a control-theoretic convergence guarantee (Theorem 3.1) and demonstrate parameterized control over attack convergence and target precision.

### Weaknesses
- The convergence proof assumes differentiable and continuous AGRs, which do not hold for non-smooth robust aggregation rules such as Median or Krum. The mapping between the adaptive control input and the high-dimensional gradient updates remains unclear, which limits the theoretical rigor of the analysis.

- The paper states that the client is selected by the server and receives the current global model $g_t$, but never clarifies whether malicious client selection is random in each round or fixed. Given the absence of any mention of random sampling and the use of fixed client sampling rates (e.g., 0.5, 0.7, 1), it appears that the same size of malicious clients participates in every round. This assumption is unrealistic in real-world federated learning, where malicious participants may not always be active due to device availability or stochastic scheduling. A fixed number of malicious clients can artificially inflate the attack’s success rate and weaken the claimed robustness. The authors should clarify this setting and, ideally, introduce randomized or varying malicious participation to better validate ASAP’s generalizability under realistic FL dynamics.

- While ASAP achieves lower target accuracy, it lacks analysis of stealthiness or detectability under anomaly detection defenses. There is no evaluation of backdoor persistence or the impact on benign model convergence quality, which weakens the empirical completeness of the study.

- Although the ASMC formulation is new, the resulting behavior (adaptive control of convergence and target accuracy) is conceptually similar to FMPA (Zhang et al., 2023). The improvement lies mainly in methodology rather than in demonstrating a fundamentally new attack capability.

### Questions
1.How does Theorem 3.1 hold for non-differentiable aggregation rules such as Median or Krum?

2.Is malicious client participation random in each round, or are the same malicious clients fixed throughout training? If fixed, how might this design choice bias the results?

3.What is the computational overhead of Fourier coefficient estimation in high-dimensional models?

4.How does ASAP perform under secure aggregation, where attackers cannot directly modify global updates?

5.Could adaptive defenses (e.g., dynamic clipping) mitigate the attack, and how would ASAP respond?

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
3

### Summary
This paper addresses the problem of aggregation-rule agnostic poisoning in federated learning, where existing attacks (e.g., LIE, Min-Max, Min-Sum, FMPA) mentioned in this paper either rely on estimating benign client statistics or require partial knowledge of the server’s aggregation rule, limiting their applicability in realistic settings. To overcome this, the authors propose ASAP, an adaptive sliding-mode control (ASMC)–based attack framework that treats the entire FL process as a nonlinear dynamical system and models unknown aggregation effects as bounded disturbances. This design allows ASAP to precisely steer the global model toward a target accuracy without knowing the aggregation rule or benign updates. Extensive experiments on CIFAR-10, MNIST, and Tiny ImageNet under nine Byzantine-robust defenses show that ASAP achieves 2–5% deviation from target accuracy, while prior AGR-agnostic baselines deviate by 10–40%, and it requires only ~1/40 the communication rounds of Min-Max and LIE. The results demonstrate that ASAP consistently outperforms state-of-the-art agnostic attacks while maintaining full independence from benign information and offering controllable attack speed and precision.

### Strengths
++ It introduces a control-theoretic formulation (Adaptive Sliding Mode Control, ASMC) for federated poisoning: a rarely explored direction in FL security research.

++ It provides controllable poisoning: the first to enable precise adjustment of convergence speed and target accuracy under unknown aggregation rules, enabling more stealthy attacks.

++ It derives the attack dynamics from nonlinear system theory, using Lyapunov stability analysis to guarantee finite-time convergence toward an adversarial target. The control law and adaptive estimator are logically consistent and theoretically justified (e.g., sliding manifold design, adaptive law for bounded uncertainty).

++ Ablation studies demonstrate robustness against varying non-IID degrees, client numbers, and sampling rates, consistently outperforming other AGR-agnostic baselines

++ Writings:





Good-structured flow: motivation → theory → algorithm → experiments → analysis



Clear motivation and consistent terminology; key differences from baselines are well-explained

### Weaknesses
-- The work doesn’t directly compare to the latest 2024–2025 poisoning attacks (e.g., [1],  [2]), leaving a slight novelty gap in the context of recent research trends.

-- The continuous-time control formulation assumes smooth model dynamics, which may not hold in discrete, stochastic FL updates.

-- (minor) Though broad, experiments are all simulation-based; no real-world FL deployment or asynchronous communication setting is tested.



[1]. PoisonedFL — Model Poisoning Attacks via Multi-Round Consistency (Xie, Fang, Gong; CVPR 2025)

[2]. Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach (Li et al., TIFS 2024)

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
