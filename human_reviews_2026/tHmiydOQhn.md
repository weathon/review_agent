# Spectral Bellman Method: Unifying Representation and Exploration in RL

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Representation learning is critical to the empirical and theoretical success of reinforcement learning. However, many existing methods are induced from model-learning aspects, misaligning them with the RL task in hand. This work introduces the Spectral Bellman Method, a novel framework derived from the Inherent Bellman Error (IBE) condition. It aligns representation learning with the fundamental structure of Bellman updates across a space of possible value functions, making it directly suited for value-based RL. Our key insight is a fundamental spectral relationship: under the zero-IBE condition, the transformation of a distribution of value functions by the Bellman operator is intrinsically linked to the feature covariance structure. This connection yields a new, theoretically-grounded objective for learning state-action features that capture this Bellman-aligned covariance, requiring only a simple modification to existing algorithms. We demonstrate that our learned representations enable structured exploration by aligning feature covariance with Bellman dynamics, improving performance in hard-exploration and long-horizon tasks. Our framework naturally extends to multi-step Bellman operators, offering a principled path toward learning more powerful and structurally sound representations for value-based RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates representation learning in the low-inherent Bellman error (low-IBE) setting. 

It introduces a new method and loss function, termed the Spectral Bellman Method (SBM). 

The core insight lies in a fundamental spectral relationship. Under the zero-IBE condition, the Bellman operator’s transformation of the value function distribution is intrinsically tied to the feature covariance structure. 

The framework can also be extended to the multi-step Bellman operator.

### Strengths
The paper is well-written, clearly structured, and easy to follow, making the technical content highly accessible. 

The proposed SBM loss function is novel and represents a meaningful contribution to the literature. 

Its motivation is well-justified, particularly since directly optimizing the MSE of IBE can be challenging and prone to inefficiencies.

The theoretical properties of the SBM loss and its connections to well-established techniques, such as power iteration, alternating optimization methods, and Thompson sampling, are especially compelling. These connections not only provide deeper insight into the behavior of the method but also situate it within a broader algorithmic context, highlighting potential avenues for further exploration.

The objective function itself appears more symmetric with respect to the parameters  theta and phi, which may contribute to improved stability and interpretability during optimization. Overall, the combination of novelty, theoretical grounding, and intuitive motivation makes the proposed approach both interesting and promising.

### Weaknesses
The paper appears to lack a formal theoretical guarantee for the minimizer of the SBM objective. For instance, it would be helpful to know whether there is any guarantee that the minimizer is close to the optimal value function, or more generally, how the SBM solution relates to the true underlying objective. Providing such a guarantee, or at least an informal discussion of its plausibility, would strengthen the theoretical contribution of the work.

One of the stated motivations of the paper is that the SBM objective is easier and more tractable to optimize than the traditional MSE objective. However, the paper does not provide clear evidence or empirical support for this claim. It would be valuable if the authors could elaborate on this point, for example by explaining why SBM optimization is expected to be more stable, faster, or less sensitive to hyperparameters, and by providing either empirical comparisons or theoretical intuition that substantiate the claim. This would help readers better understand the practical advantages of adopting the SBM formulation over conventional approaches.

### Questions
Is there any theoretical guarantee that the minimizer of the SBM objective is close to the optimal value function? Since the primary goal of reinforcement learning is typically to learn a near-optimal or optimal Q-function, it is crucial to understand whether the proposed loss function is well-aligned with this objective. Clarifying this connection would strengthen the theoretical foundation of the work and help readers assess the practical relevance of SBM.

On the practical side, could the authors comment on whether the SBM-based Q-learning method, particularly in its connection to Thompson Sampling, is easier or more efficient to implement in practice? Insights on implementation challenges, stability, or computational considerations would be valuable for understanding the real-world applicability of the approach.

It would also be helpful to discuss the computational complexity of the proposed algorithm, including how it scales with state-action dimensions, representation size, and sample complexity. A clear analysis would allow readers to compare the approach against existing methods and better understand its feasibility for larger or more complex problems.

Since the paper emphasizes representation learning, it would be useful to clarify whether the learned features provide meaningful benefits, both theoretically and empirically. Do the features improve policy learning, generalization, or sample efficiency? 

Do the empirical experiments explicitly target the quality of these representations, or are the observed benefits mainly a byproduct of improved  Q-learning performance? Addressing these questions would help highlight the practical and theoretical significance of the representation learning component.

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
The work attempts to combine representation learning and exploration for better performance in RL tasks. In particular, they focus on the Atari benchmark. The method attempts to learn zero IBE features and they combines it with Thompson sampling for better exploration. They combine their approach with two algorithm - DQN and R2D2 and show performance improvement, particularly on games which demand hard exploration.

### Strengths
* The work is well motivated, it makes sense to learn representations which directly support exploration/learning better strategies.
* The theory is well motivated and sound. Specifically, spectral decomposition under zero IBE conditions and the way to use power iteration for SBM loss optimisation are useful results.
* The performance gains are good when SBM + TS is used with baseline algorithms, this is in-particular substantial in the hard exploration games.
* The extension to multi-step operators (Retrace for R2D2) is theoretically sound and practical.

### Weaknesses
* Baseline: The paper is missing two critical baselines - DQN + TS or any DQN based method which uses TS (for instance BDQN (Azizzadenesheli et al. 2018)) but does not use spectral features. Similar is the case with R2D2, the work is missing a critical baseline R2D2 + TS without spectral features (which I understand is might be non-trivial to implement but without this, it is very hard to truly guage SBM's effectiveness). This is extremely important to have to be able to gauge the efficacy of SBM features. The paper cites Azizzadenesheli et al. (2018) in their related work section but provides no justification for excluding it as a baseline. The current experimental design confounds two variables (representation learning + exploration strategy), making it difficult to attribute performance gains to either component independently. For instance, from table 1, it seems the major chunk of improvement is coming after adding TS to SBM (its good enough for DQN but with R2D2, R2D2 + eps → R2D2 + SBM + eps the improvement is pretty small (basically none in terms of median scores), whereas R2D2 + SBM + eps → R2D2 + SBM + TS shows +0.23 improvement overall and 0.16 on explore subset). This pattern suggests TS may contribute more to the gains than the spectral representation itself, which contradicts the paper's central claim that SBM representations "naturally facilitate structured exploration."
* For exploitation-heavy, precision-control games (breakout, frostbite), the performance sees a decline from the baseline algorithm (DQN specifically) which I am guessing is due to the fact that Thompson sampling may add harmful exploration noise. Authors should discuss a bit about this and suggest if there are ways to control this (e.g., annealing the Thompson Sampling variance parameter or adaptive mechanisms to reduce exploration noise when sufficient policy convergence is achieved). Some ablations and experiments showing when SBM helps vs. hurts performance would strengthen the paper. However, this is a minor thing.
* The Thompson Sampling implementation (Equation 5) directly follows Zanette et al. (2020b) without modification. The only claimed contribution is using SBM-learned features in the covariance, but without DQN+TS baselines, it's unclear if standard representations would work equally well.
* The paper provides no analysis of computational overhead with baselines.

### Questions
1. Can you provide results for DQN + TS and R2D2 + TS without spectral features? If not, any justification as to why there were excluded.
2. In Table 1, R2D2 → R2D2+SBM shows almost no improvement (in terms of median score for the explore subset), while R2D2+SBM → R2D2+SBM+TS shows +0.16 improvement. Doesn't this suggest TS contributes more than SBM? How do you reconcile this with your claim that SBM representations "naturally facilitate structured exploration"?
3. What is the computational overhead of alternating between Q-learning and SBM representation learning? 
4. On games like Breakout and Frostbite, SBM+TS performs worse than baseline DQN. Can you discuss this? Is this due to harmful exploration noise in exploitation-heavy games? Can you discuss/analyze this? However, as stated, this is a minor requirement.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the Spectral Bellman Method (SBM), a novel framework designed to unify representation learning and efficient exploration in reinforcement learning.The central problem addressed is that these two critical components are often treated as separate modules, failing to leverage their potential synergy. The motivation stems from the insight that while representations with low Inherent Bellman Error (IBE) are highly desirable for value-based methods, directly optimizing for them is computationally intractable. The authors identify a key spectral relationship: under the ideal zero-IBE condition, the structure of the Bellman operator is fundamentally linked to the covariance structure of the features themselves. To solve this, SBM proposes a new learning objective inspired by the power iteration method. Instead of directly minimizing Bellman error, this objective encourages the feature covariance to align with the Bellman dynamics across a distribution of value functions. This serves as a tractable proxy for learning low-IBE representations. The method is integrated into an alternating training loop where the learned feature covariance is naturally used to drive exploration via Thompson Sampling, which in turn collects data to improve the policy and refine the representation learning target.

### Strengths
Strength:
1.	The paper introduces a novel objective for representation learning by reframing the intractable problem of minimizing Inherent Bellman Error (IBE) into a more tractable proxy based on the spectral properties of the Bellman operator.

2.	The framework provides a tight, natural coupling between representation and exploration. The feature covariance matrix learned by SBM is directly used to guide Thompson Sampling, creating a coherent feedback loop where better representations inform more structured exploration.

### Weaknesses
Major：
1.	The theoretical motivation for the SBM objective, particularly the spectral decomposition outlined in Theorem 1, is quite elegant. This derivation hinges on the ideal assumption of zero Inherent Bellman Error (IBE), where the function space is perfectly closed under the Bellman operator. I am curious about how the proposed framework is expected to behave when this assumption is inevitably relaxed in practice, as is the case when using complex neural network approximators. Could the authors provide some additional intuition or analysis on the robustness of the SBM objective when the IBE is small but non-zero? For example, how does the connection between the feature covariance and the Bellman operator change in this scenario? Does the SBM loss still serve as a more effective proxy for minimizing IBE than the standard MSE objective? Adding a brief discussion on this point would be very helpful in bridging the compelling theory with the practical algorithm.

2.	The paper's alternating optimization between policy and representation is interesting. However, I have a conceptual question about the representation learning target. Since the SBM objective is tied to the Bellman operator defined by the current policy, it seems the representation might become overly specialized to a suboptimal policy early in training. Could this create a "representational trap" that makes it difficult for the agent to later discover and represent a truly optimal policy? I would appreciate the authors' perspective on this potential issue.

### Questions
1. Could the authors elaborate on the behavior of the SBM objective when the Inherent Bellman Error (IBE) is small but non-zero? Specifically, how does the relationship between the feature covariance and the Bellman operator evolve in this case, and does the SBM loss remain a more faithful proxy for minimizing IBE than the standard MSE objective? 

2. Given that the SBM objective depends on the Bellman operator of the current policy, how do the authors prevent the learned representation from becoming overly specialized to a suboptimal early policy? Could such coupling lead to a “representational trap,” and if so, what mechanisms or design choices mitigate this risk?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new approach, the Spectral Bellman Method, for learning representations that aid exploration. The approach is built upon the Inherent Bellman Error (IBE) condition, leveraging the key insight that connects feature covariance and zero-IBE to derive an algorithm that alternates between learning features and Q-values. Furthermore, the method also leverages the underlying structural properties to enhance exploration via the Thompson sampling technique. Experiments conducted on the full Atari game suite show improvements over other baselines, observing significant gains on deep exploration games and corroborating the effectiveness of the approach.

### Strengths
- The paper presents a well-executed idea. The proposed method has solid mathematical backing, and the derived algorithm stems directly from the theoretical insights. It is encouraging to see these insights translate into meaningful empirical gains.
- The key insight regarding the spectral properties and the zero-IBE condition is particularly elegant, as it helps reduce a complex optimization problem to a more tractable one. It is a further strength that the Thompson sampling exploration fits well with the approach, tackling both feature learning and exploration simultaneously.
- The resulting algorithm represents only a minor modification to existing algorithms, such as DQN or R2D2, yet it achieves substantial empirical gains on many Atari games, including those where exploration is challenging.

### Weaknesses
- The mathematical derivations in Section 3 could be significantly improved for clarity. As written, the section is overly dense. The authors should consider simplifying complex notations (e.g., $\tilde{\theta}(\theta)$) and adding intuitive explanations to make the theoretical statements more comprehensible
- The paper would also benefit from a small-scale, illustrative experiment. Such an experiment would be valuable for building intuition and helping isolate the source of the method's benefits (representations, exploration, or both). Additionally, a discussion or analysis of the method's sensitivity to key parameters (such as the choice of state-action distribution pairs and Q-function parameters) is a notable omission.
- The experimental comparison to other representation learning methods is narrow, as it only includes one such baseline (Online-PVNs). A more thorough evaluation against other common baselines&mdash;particularly those using an auxiliary loss for representation learning&mdash;is needed to fully contextualize the proposed method's performance. This is especially true since several such methods are already discussed in the related work section.

### Questions
- The paper's claim that "effective exploration aims to reduce uncertainty" (lines 110-112) is presented as a general-purpose objective for exploration. The authors should justify this framing, as it seems to imply that exploration is also a problem in policy evaluation, not just control.
- The paper's assertion (line 149) that not resulting in task-agnostic representations is a feature is counterintuitive. Given that task-agnostic representations are generally desirable for fast adaptation to changing goals, the authors should provide a clear justification for why their task-primary approach is a benefit and not a limitation.
- The introduction of Equation 2 lacks sufficient motivation and derivation. It appears abruptly in the text, and the authors should provide more insight into how this equation was derived and why it is the correct formulation in this context.
- Building on the weakness of the missing ablation studies, the paper should provide the intuition or justification for the specific choices of parameters used in the experiments (e.g., for the state-action distribution and Q-function). An explanation of why these particular choices were made is necessary.
- The algorithm, as presented in the pseudocode, appears to operate on a batch of data for each update. This raises questions about the method's feasibility and efficiency in a fully online or streaming setting. The authors should clarify whether their approach can be adapted for such scenarios.

### Soundness
4

### Presentation
3

### Contribution
3
