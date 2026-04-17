# How Does Layer Normalization Improve Deep $\boldsymbol{Q}$-learning?

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Layer normalization (LN) is among the most effective normalization schemes for deep $Q$-learning. However, its benefits remain not fully understood. We study LN through the lens of _gradient interference_. A gradient interference metric used in prior works is the inner product between semi-gradients of the temporal difference error on two random samples. We argue that, from the perspective of minimizing the loss, a more principled metric is to calculate the inner product between a semi-gradient and a full-gradient. We test this argument with offline deep $Q$-learning, without a target network, on four classic control tasks. However, counterintuitively, we find empirically that first-order gradient interference metrics _positively_ correlate with the training loss. We empirically show that adding a second-order gradient interference term gives more intuitive results. Theoretically, we provide supporting arguments from the linear regression setting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper investigates why Layer Normalization (LN) improves stability and performance in deep Q-learning. They argue that prior "gradient interference" metrics are incomplete. It proposes a mixed-gradient perspective and, via a second-order Taylor analysis of TD-loss decrement, introduces GI2, a metric combining first and second-order terms. Empirically, across four classic offline control tasks trained without a target network, first-order inference metrics counterintuitively increase with training loss, while GI2 decreases with loss.

### Strengths
- The motivation is clear: LN is effective while other normalization often do not work in deep Q-learning.
- GI2 adds a prinicpiled second-order term to loss-decrement reasoning, aligning theory with observed loss trends better than first-order methods.

### Weaknesses
- GI2 omits the Hessian term involving $\nabla^2 h$. While the paper argues that this term is complex but leaves its contribution untested.
- MountainCar consistenly breaks the main trends, but brief analysis is given.

### Questions
- Can you explain more on the deviation of the abnormal performance of the MountainCar example from the correlation patterns for GI2 and returns?
- Given GI2's negative correlation with loss, can practitioners monitor a proxy online to decide when to apply LN or adjust learning rates?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper provides an in-depth investigation into how Layer Normalization enhances performance and stability in Deep Q-Learning, primarily from the perspective of Gradient Interference. The authors present an interesting insight: the first-order gradient interference metric shows a positive correlation with training loss, whereas a second-order corrected metric offers a more intuitive explanation for LN's benefits. This claim is supported by extensive experiments.

### Strengths
1. The analysis of gradient interference, particularly the introduction of the second-order corrected term (GI₂) to explain LN's role in DQN, demonstrates significant insight to the existing literature. Investigating the impact of LN specifically on Deep Q-Learning is a valuable research direction.

### Weaknesses
The logical flow of this article is quite confusing to me, which may be due to my limited familiarity with the field of Q-learning. I hope the author can provide detailed classifications on the following points.

1. The narrative logic is somewhat confusing. The stated goal is to understand how LN improves Deep Q-Learning performance. However, Section 3.1 primarily discusses the effects of different optimizers on various metrics, and Section 3.3 only briefly mentions LN's performance improvements on several benchmarks without delving into the underlying reasons. The core theoretical attempt to answer the "how" question is concentrated in Section 3.4. However, the theoretical setting in 3.4 is highly simplified. Using data normalization as a direct analogy for analyzing Layer Normalization might be too trivial and lacks rigor; it feels more like a toy example than a formal theorem.

2. The paper is in a good direction but requires some revisions for presentation. 
- (a) In Table 1, the number of decimal places for the "random" column is inconsistent.
- (b) Phrases like "second-order–corrected metric in line 61" are awkwardly phrased. The article uses a large amount of "–", which can easily confuse readers with "-".
- (c) The formatting appears unpolished. For instance, there is excessive whitespace around Equation (11) on Page 7.

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates gradient interference in deep Q-learning, where gradients of the Q function for one state-action pair also affect other state action pairs.
To this end, the paper introduces several ways to measure gradient inference.
They look at how these correlate with return and loss, and attempt to use this to show that layer normalization helps Q-learning.

### Strengths
* The paper introduces several interesting metrics for gradient interference in Q-learning.
* There are some surprising results, especially about the relevance of the second order term in gradient interference.

### Weaknesses
* The relation to layer normalization is weak. Most of the paper is about gradient inference, and is independent of the architecture. The experiments are done with and without LN, but this is only a comparison, and doesn't give any insights into how LN is actually helping.
* There are several experiments where the learning rate is too high (figure 2 and figure 3), which causes the training loss to increase to huge values (>1e10). This then also makes some of the other plots useless, for example GI₂ has values around -1e50 for these failed runs.
  It also makes me question the conclusions that "SGI and MGI also positively correlate with training loss in our experiments" and "GI2 negatively correlates with training loss". The axes are so stretched that I can't clearly see any such correlations. And these absurd values would dominate any calculated correlation coefficient.
* If you make a claim of correlation, then this should be supported with some correlation coefficient.
* Theorem 2: "Under fixed $R(θ_t)$, $A^2_t/B_t$ is maximized when ∥x∥ is a constant in the data distribution."
  In this theoretical dataset, you can't just normalize $x$ without also affecting $y=x^Tθ$. So this is not a realistic representation of layer normalization. If $y$ was fixed, then scaling $x$ down for example would require $θ$ to scale up, which would then scale up $R(θ)$.

### Questions
* "However, counterintuitively, we find empirically that first-order gradient interference metrics positively correlate with the training loss."
  Why is this unexpected? I would think that less interference = better = lower loss. Or is that not what you meant by a positive correlation?
* The presentation of deep Q-learning using a "semi-gradient" is slightly nonstandard.
  What this paper calls a semi-gradient is just the gradient of the squared TD-error, no?
  Is this called a semi-gradient because the gradient is not propagated through the TD-target?
 * With a continuous state space, the self-interference term ($(s,a)=(s_t,a_t)$) in QGI, SGI, and MGI has measure 0, so it does not contribute relevantly to the expectation. Is that understanding correct? And if so, why is a different choice made for QGI and SGI/MGI?
 * "a standard single-layer neural network"
   Clarify this as a "single hidden-layer neural network", since some people will say that this network has two layers.
 * "absolute cosine similarity notation cos+"
   Is this definition necessary? You could define normal cosine similarity and write $|\cos(x,y)|$.
 * in the interpretation of theorem 2: "$\max_η\{2ηA_t − η²B_t\} = A_t^2/B_t^2$"
   Should be $A^2/B$.
 * The definition of $l_\theta$ is important, but it is split up by a page break and a figure, making it harder to find.
 * There is also a page break in the middle of (*) in section 3.2. The first line of that equation could be incorrectly read as the whole equation.

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper investigates the role of Layer Normalization (LN) in improving optimization dynamics of deep neural networks. The authors attempt to provide both theoretical and empirical insights by analyzing gradient scaling, variance stabilization, and optimization landscape smoothness under LN. They present several simplified derivations suggesting that LN reduces gradient variance and improves isotropy, followed by small-scale experiments on MLPs and Transformers to illustrate these effects. The paper aims to unify existing intuitions about LN’s benefits into a coherent explanation framework.

### Strengths
The paper tackles a fundamental and relevant question in deep learning — understanding why Layer Normalization helps training stability and convergence.

The overall motivation is clear, and the paper provides a structured narrative linking theory, gradient analysis, and empirical visualization.

The presentation is relatively readable, and figures (e.g., gradient norm distributions and optimization trajectories) help illustrate the main intuition.

### Weaknesses
Symbols like $\mu_i$, $\sigma_i$, and $\gamma$, $\beta$ switch between layer and neuron-level contexts without explicit indexing. This makes it difficult to follow what the normalization is actually applied to.

LN “preserves the direction of gradients while adjusting their scale,” but no formal proof or Lipschitz analysis is given. This is hand-wavy and lacks formal support.

The comparison with BatchNorm and RMSNorm is descriptive, not analytical. The same conclusions already appear in several existing studies.

Experiments are only on small-scale models (2-layer MLPs and tiny Transformers). This severely limits the claim of generality.

Reported improvements are within 0.2–0.4% on average accuracy — no significance testing or error bars are shown.

### Questions
See WeakNess, but I don't really know enough about this area of ​​research, so my evaluation may not be accurate. Perhaps the author's innovation lies more in the theoretical level.

### Soundness
3

### Presentation
2

### Contribution
2
