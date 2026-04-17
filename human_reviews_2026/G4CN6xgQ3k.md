# Failure Modes of Maximum Entropy RLHF

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
In this paper, we show that Simple Preference Optimization (SimPO) can be derived as Maximum Entropy Reinforcement Learning with length-normalized temperature, providing a theoretical foundation for this reference-free method. Motivated by SimPO's strong performance in offline preference optimization, we investigate whether Maximum Entropy RL can achieve similar results in online RLHF settings. Our experiments find that Maximum Entropy RL consistently exhibits overoptimization and unstable KL dynamics, even at very low learning rates. Unlike KL-constrained methods that maintain stable training, entropy regularization fails to prevent reward hacking and appears to correlate with overoptimization. Lastly, we discuss possible explanations for why SimPO succeeds in offline settings while Maximum Entropy RL struggles in online scenarios. Our findings suggest that reference-free approaches may face distinct challenges when applied to online or offline preference learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates an asymmetry in preference optimization by first providing a novel theoretical foundation for the offline algorithm SimPO, deriving it from a Maximum Entropy (MaxEnt) RL framework. Motivated by this, the authors empirically test a standard online MaxEnt RLHF objective and find that it consistently fails. Unlike stable KL-constrained methods, the online MaxEnt approach suffers from severe overoptimization and reward hacking, with entropy regularization appearing to correlate with—rather than prevent—these instabilities.

### Strengths
1. The paper provides a significant theoretical contribution by deriving SimPO from a Maximum Entropy RL framework. This establishes a principled foundation for SimPO, which, unlike DPO, previously lacked a clear theoretical explanation.
2. This paper investigates that MaxEnt RLHF consistently fails due to overoptimization and entropy regularization correlates with reward hacking.

### Weaknesses
1. The paper's central claim of equivalence between SimPO and MaxEnt RL appears non-rigorous. The derivation (Eq. 14) relies on the cancellation of the partition function term $\alpha \log Z(x)$. However, this only holds for a constant α. The paper's linkage to SimPO uses a length-normalized temperature $\alpha = \beta / |y|$, which depends on the response y. Since $|y_w|$ and |$y_l|$ are generally not equal, their corresponding α terms differ, meaning the $\alpha \log Z(x)$  terms would not cancel. This is a critical point, as the original SimPO paper showed that length normalization is key to its performance.
2. The empirical claims about online failure, while strong, are based on experiments with Pythia-1B/2.8B models and the TL;DR summarization task. It is unclear if these same failure modes would persist on larger, more capable models (e.g., 7B+) or on more complex alignment tasks.

### Questions
1. Regarding the theoretical derivation (Eq. 14): Could you clarify how the $\alpha \log Z(x)$ terms cancel given that the link to SimPO requires a length-normalized temperature $\alpha = \beta / |y|$? Since $|y_w|$ and $|y_l|$ are not generally equal, their corresponding α values would differ. Does the theoretical link to MaxEnt RL still hold if this is corrected?
2. In line 320, the authors mention that "entropy actually exacerbates reward hacking rather than mitigating it." This claim seems counter-intuitive, since reward hacking is often caused by a collapsed distribution. Could the authors further explain this point?

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
3

### Summary
The paper explores the connection between the SimPO method and maximum entropy reinforcement learning, and examines the effectiveness of maximum entropy RL in online RLHF settings.

### Strengths
1) This paper establishes the relationship between SimPO and maximum entropy RL, providing a theoretical analytical perspective for SimPO.

2) Through experiments, it demonstrates that maximum entropy RL is difficult to apply to online RLHF and provides relevant analysis.

### Weaknesses
1) Although the paper offers in-depth analysis of the nature of SimPO, its contribution and innovation to the field of preference optimization are limited. For example, what are the implications of this paper's analysis for the design of preference learning algorithms or for addressing its existing challenges?

2) The poor performance of maximum entropy RL in online scenarios lacks thorough theoretical analysis and explanation of the underlying reasons. For example, what are the key factors (from a theoretical perspective) that lead to the failure of algorithms in online scenarios?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

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
This paper establishes a new theoretical foundation for Simple Preference Optimization (SimPO), a reference-free LLM alignment method. The authors demonstrate that SimPO's objective can be mathematically derived from the Maximum Entropy Reinforcement Learning (MaxEnt RL) framework, specifically as a solution with a length-normalized temperature . Motivated by SimPO's strong performance in offline settings, the paper investigates whether MaxEnt RL can be applied directly to online RLHF. Their experiments reveal that, unlike stable KL-constrained methods, online MaxEnt RL consistently leads to overoptimization and reward hacking, even at very low learning rates. The authors conclude that entropy regularization, while effective offline as SimPO, fails to provide sufficient safeguards in online training and may even correlate with instability.

### Strengths
My observation is that MELON's core defensive strategy is cleverly designed to detect when an agent's tool calls are hijacked, by comparing them against a masked, task-neutral re-execution . However, the paper's own analysis of its failure modes points out a major limitation: the defense is blind to "Response-Based Attacks," which accounted for over 72% of the cases where attacks evaded detection. This is because these attacks achieve their goal through the agent's text response rather than a tool call, a behavior MELON does not monitor. This suggests that while MELON is a strong solution for preventing unauthorized actions, it doesn't address the separate and more common problem of attacks that manipulate the agent's language output.

### Weaknesses
1. Novelty: Although this work shows the disadvatange of Simpo or simply maximizing or minimizing the entropy, this reward hacking issue is not a new thing to the community. Only pointing out this limitation makes the contribution of this work seems weak. The authors are suggested to give more insights about how to deal with the relationship between entropy and rewards.

2. The experiments are only on small models. It would be more convincing if they experiment on more diverse base models to make the results convincing.

3. Although another contribution stated by the authors is establishing the theoretical connection between SimPO and Maximum Entropy RL. They do not provide any formal results in the main paper. It is not clear if they prove any converegence results or under which condition the equivalence hold.

### Questions
1. In Figure 2, does the RLHF reward contain entropy?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a theoretical foundation for Simple Preference Optimization (SimPO) and establishes its connection to maximum entropy reinforcement learning (RL). The authors then conduct experiments showing that maximum entropy RL exhibits instability and over-optimization during online training, and that increasing the entropy coefficient can further amplify these instabilities.

### Strengths
a. It is interesting to observe the training instability and over-optimization phenomenon in online maximum-entropy RLHF.

b. The overall presentation is clear and easy to follow.

### Weaknesses
a. The connection between SimPO and maximum-entropy RLHF is not entirely solid. In maximum-entropy RLHF, the hyperparameter $\alpha$, which balances the reward term and the entropy exploration term, should be a fixed value and should not depend on $y$. However, in the connection established in Section 3.2, SimPO introduces a length normalization in this hyperparameter that depends on the response $y$, which varies across samples. Since length normalization is a key component of SimPO, analyzing its effect is crucial for understanding the algorithm’s behavior. Unfortunately, the authors do not provide a concrete analysis of this aspect, which limits the paper’s theoretical contribution. Furthermore, the paper does not explain how the margin hyperparameter $\gamma$ is related to maximum-entropy RL, making the claimed theoretical foundation of SimPO less convincing.

b. All experiments are conducted on relatively small models (e.g., Pythia 2.8B), which differ from widely used modern base models such as Qwen and LLaMA. This raises concerns about whether the observed phenomena generalize to larger and more realistic model scales. The authors should include experiments on different base models to strengthen the validity of their empirical findings.

### Questions
See Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
