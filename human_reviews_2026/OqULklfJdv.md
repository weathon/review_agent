# Behavior-Aware Off-Policy Selection in High-Stake Human-Centric Environments

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In many human-centric environments, such as education and healthcare, the unobservability of human underlying states has been recognized as a key obstacle for understanding individual needs, thus hindering out ability to provide personalized decision-making policies. Several reinforcement learning (RL)-related approaches have been used to facilitate sequential decision-making in these settings, including off-policy selection (OPS), which aids in safely evaluating and selecting optimal policies offline. However, existing OPS algorithms are unsuitable when both the state is unobserved and the setting requires a personalized policy. To address this challenge, we propose a behavior-aware adaptive policy selection framework (HBO) that first captures potentially unique characteristics of the state from human behaviors, and then estimates when and how to intervene with less uncertainty in a timely manner, with bounded error. HBO is evaluated over two real-world human-centric applications, intelligent tutoring and sepsis treatments, where it significantly enhanced participants' long-term course outcomes and survival rates. Broadly, our work enables improved policy personalization in high-stakes domains where extensive evaluation is not possible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a method for off-policy selection (OPS) in high-stakes, human-centric environments such as healthcare or education. Recognizing that individual behavior trajectories may not fully reveal decision-critical latent states, the authors introduce behavior-aware OPS: a framework that monitors ongoing behavior and switches to an optimized policy when necessary. The approach clusters behavior trajectories using semi-parametric sequence modeling (TICC) to define latent behavior states. It then defines critical behavioral patterns (CBPs) to trigger early intervention. Theoretical analysis establishes that selecting policies based on CBPs does not degrade expected returns compared to the baseline policy under standard OPE assumptions. Experiments on simulated tutoring and ICU sepsis treatment settings demonstrate that this approach improves outcomes over static policy switching baselines.

### Strengths
a. The paper introduces a clear and practical mechanism, CBPs, to detect early-warning signals of suboptimal performance, allowing targeted one-time policy switching tailored to individual behavior without full latent state identification.

b. Proposition 3.7 ensures that if CBPs are estimated conservatively, the selected policy will not underperform the baseline under assumptions.

c. On two high-stakes benchmarks (an intelligent tutoring system and ICU sepsis simulations), the behavior-aware OPS method achieves better student mastery rates and patient survival outcomes than conventional static or globally switched policies.

### Weaknesses
a. Despite some implications that individuals have different behavior-generating mechanisms, the method does not explicitly model population shift, confounding, or latent policies. The success seems to depend on good coverage in offline data.

b. The estimation of CBPs depends on pattern mining from clustered behavior trajectories. While effective, the method lacks theoretical characterization of the statistical guarantees or sensitivity of the resulting patterns, and the procedure for selecting pattern thresholds (ε, support) is heuristic.

### Questions
a. How sensitive is the OPS performance to the choice of TICC parameters and number of clusters? Could the policy switch decision degrade significantly if clustering misrepresents latent behavior states?

b. The CBP-based trigger is conservative and requires multiple conditions to match. Could the method benefit from learned scoring or ranking over patterns to allow softer, more adaptive switching criteria?

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
The paper presents HBO, a two-phase framework for personalizing decision policies in partially observable, high-stakes domains such as education and healthcare. HBO first analyzes historical trajectories offline to identify short critical behavioral patterns that signal poor outcomes. Online, it starts with a safe default policy and, when a critical pattern is detected in a new user’s data stream, selectively switches to an alternative pre-vetted policy using confidence-controlled importance sampling. The authors provide theoretical guarantees on the policy’s value and demonstrate improved learning gains and patient outcomes compared to standard non-adaptive and fully adaptive baselines.

### Strengths
I appreciate the authors' focus on safe, personalized policy selection in human-centric reinforcement learning.  The framework sensibly separates offline pattern mining from online, confidence-controlled policy switching. The inclusion of finite-sample error bounds for policy value estimation adds some theoretical reassurance about its reliability. Empirically, HBO shows promising results in both education and healthcare settings, outperforming several baselines and improving outcomes for difficult cases.

### Weaknesses
While the paper appears technically solid and well-motivated, I do not have a strong background in this area, so I cannot confidently assess whether the proposed approach represents a true state-of-the-art advancement.

To me, the largest issue is that the interpretability of the CBP remains limited, as they are derived from latent representations rather than explicit behavioral features, making it difficult for practitioners to understand or validate the reasoning behind policy switches. 

Moreover, Assumption 1, which requires a pre-approved safe initial policy for every participant, may not be realistic in less controlled or data-scarce environments. It would also be helpful if the authors could elaborate on the practicality of Assumption 1—perhaps by discussing scenarios where expert-approved policies are unavailable, and how the framework might adapt or relax this assumption.

### Questions
Please see the weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an adaptive off-policy selection framework for human-centric settings. It mines Critical Behavioral Patterns (CBPs) from historical data to determine when to switch a policy. Then, importance-sampling–based off-policy evaluation is used to estimate the value of candidate policies and select the most appropriate one. This idea is interesting and good, but the proposed approach suffers from notable limitations.

### Strengths
A solid and promising idea; most limitations are explicitly acknowledged by the authors in the paper.

### Weaknesses
1. A major limitation is the single-switch constraint: the method may adapt at most once. Rather than enforcing a hard single-switch constraint, adopt a budgeted scheme—allow up to B switches with explicit switching costs and safety thresholds—which seems more practical than a one-size-fits-all single switch.
2. The discrete-state mapping imposes strong structural assumptions on the observation/state space and is highly sensitive to hyperparameters such as the number of clusters K, window length, and the smoothing/regularization strength. Moreover, discretized representations introduce information loss, which can be especially problematic in safety-critical domains such as healthcare.
3. CBPs are identified by high-frequency subsequences, but frequency does not imply outcome relevance. Some symptoms occur often as part of normal treatment responses and therefore do not indicate that the policy is incorrect.

### Questions
1. In experiment (refer to L326), the offline data are collected by expert-designed policy, yet CBPs are mined from “poorly performing” trajectories. Does “poor” mean the lower-performing episodes within those expert trajectories. If so, this is questionable: it amounts to picking “the worst of the expert traces,” which may not reflect true failure modes.
2. Is it a more practical way to training a unified policy via Generalized Policy Improvement (GPI)[1]? A GPI-based generalist policy can subsume the base policies, eliminating the brittle step of detecting “critical moments” and avoiding online switching.

Minor mistakes
1. L404: normalized **leaning** gains  
2. Definition 3.3 explicitly states that if $\Delta(h) > 0$ (a positive look-ahead advantage), the action should be deferred (on hold) to the next step; however, Algorithm 1, line 11, implements “if $\Delta(h) > 0$ then Switch,” which contradicts the definition.

[1]Barreto, André, et al. "Successor features for transfer in reinforcement learning." *Advances in neural information processing systems* 30 (2017).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work tackles the problem of the off-policy selection in human-centric environments. In particular, the current off-policy selection does not work in scenarios when the state is unobserved and a personalized policy is required. Motivated by this point, the authors develop a new algorithm called HBO to capture the unique characteristics of the state and then provide intervention. The empirical experiments have been conducted to showcase the ability of the developed algorithm.

### Strengths
1. This paper is motivated by important human-centric decision-making problems. 
2. The manuscript is easy to understand and follow. 
3. The empirical evaluation has been provided to demonstrate the power of the algorithm.

### Weaknesses
1. The pronounced discussion between the off-policy learning and off-policy selection needs to be provided in the paper. 
2. CBP is an interesting approach to identifying critical behavioral changes from the historical trajectories. What is the unique advantagne of using CBP in HBO? 
3. The performance of the HBO relies on CBP. Could the authors provide some error analysis and explain how the quality of CBP will impact the learned policy? 
4. Assumption 3.4 is a strong assumption in offline settings. 
5. As the main applications are in human-centric environments, the data size is usually small. From this perspective, could the author provide some finite-sample theoretical analysis for evaluating the performance of the learned policy? 
6. Follow up with the last question. In experiments, the authors need to test the algorithm with extremely small data size and evaluate the performance.

### Questions
Please find the above weakness.

### Soundness
2

### Presentation
3

### Contribution
2
