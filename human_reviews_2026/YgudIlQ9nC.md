# Never Saddle for Reparameterized Steepest Descent as Mirror Flow

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 6, 0, 4

## Abstract
How does the choice of optimization algorithm shape a model’s ability to learn features? To address this question for steepest descent methods --including sign descent, which is closely related to Adam --we introduce steepest mirror flows as a unifying theoretical framework. This framework reveals how optimization geometry governs learning dynamics, implicit bias, and sparsity and it provides two explanations for why Adam and AdamW often outperform SGD in fine-tuning. Focusing on diagonal linear networks and deep diagonal linear reparameterizations (a simplified proxy for attention), we show that steeper descent facilitates both saddle-point escape and feature learning. In contrast, gradient descent requires unrealistically large learning rates to escape saddles, an uncommon regime in fine-tuning. Empirically, we confirm that saddle-point escape is a central challenge in fine-tuning. Furthermore, we demonstrate that decoupled weight decay, as in AdamW, stabilizes feature learning by enforcing novel balance equations. Together, these results highlight two mechanisms how steepest descent can aid modern optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studied and compared the dynamical behaviors of different types of steepest descent algorithms. The authors focused on deep linear diagonal models and showed that a steepest flow in the weight matrices with respect to $L_p$ norm induces a "steepest mirror flow" in the end-to-end matrix. They analyzed how the choice of $p$ influences the dynamics, convergence rate and the effects of weight decay. Experiments on both linear and practical models are provided for the comparison between different algorithms.

### Strengths
The paper extends the study of reparametrization in gradient flow to steepest flow. This provides a useful platform for studying how reparametrization and optimization geometry interact to shape the training dynamics. I find this extension meaningful and valuable. Several established results are quite interesting: 

* Lemma 4.4: The balance equation for steepest flow in training deep diagonal linear network is new and interesting. 

* Theorem 4.6: The equivalence between steepest flow with reparametrization and "steepest mirror flow" is quite interesting. To my knowledge, previous work only showed the equivalence between gradient flow with reparametrization and mirror flow. 

* Corollary 4.8: It sheds light on how different choices of optimization geometry and reparametrization (network depth) could affect the convergence rate.

### Weaknesses
The main weakness is that the main claims are not well supported by the presented theory. The title begins with "Never saddle" and the authors emphasized both in the abstract and in Contributions that "we prove that steeper descent (lower $q$) simultaneously escapes saddles faster and supports feature learning", and that "decoupled weight decay ... yielding more stable feature learning". While these statements are strong and intriguing, I do not find them adequately supported by the theoretical results. 

**Regarding saddle escaping**

After Lemma 4.4 and in Figure 3, the authors claimed that "the (curved) path away from zero is shorter for smaller $q$, indicating faster saddle escape". This reasoning is not logically sound: First, the notion of a "shorter path" is ambiguous (e.g., between which endpoints?). Second, the geometry of the invariant manifold alone does not determine how fast the algorithm moves along the manifold. 

Besides the above, the only theoretical evidence for the claim that "lower $q$ escapes saddles faster" is based on two results: (i) Corollary 4.8: under initialization "$w_1 = 0$ and $w_i = \lambda > 0$", a smaller $q$ yields a larger coercivity constant $\mu$; and (ii) Theorem 4.2: when $R$ is a separable Bregman function, a larger $\mu$ yields a faster linear convergence rate. However, I do not find these two results sufficient for the claim, for the following reasons: 

* It is not established that the $R$ used in Corollary 4.8 (or Theorem 4.6) is a separable Bregman function. Thus it is unclear whether Theorem 4.2 applies; 

* The considered initialization "$w_1 = 0$ and $w_i = \lambda > 0$" lies in a measure-zero set; 

* Only a subset of saddles (those near the considered initializations) are analyzed, whereas, as noted by the authors, other saddles exist elsewhere in the parameter space. 

Therefore, even if Theorem 4.2 is applicable, the results only indicate that: for certain specific saddles and for a measure-zero set of initializations in their neighborhoods, a smaller $q$ leads to a faster escape. I thus think it is overstated to use the title "never saddle", or to claim that smaller $q$ results in faster saddle escaping, which sounds like a general statement, just based on these results. 



**Regarding feature learning** 

The authors claimed that "smaller $q$ supports feature learning". However, in the theory section (Section 4), I could not find a clear discussion on how the choice of $q$ affects feature learning. The following two aspects might be relevant, but it is unclear how they support the claim: 

* In Corollary 4.11 and 4.12, the authors discussed whether the function $R_{L_p, L}$ is a Legendre function under different $q$, network depth $L$, and metric exponent $m$. Then in the left panel of Figure 2, the authors indicated that a metric exponent slightly smaller than $1$ correspond to a feature learning regime (the green band in the figure). However, it is not explained why this is true and how does metric exponent relate to feature learning. For example, why does $m=0.95$ lead to feature learning, while $m=0.5$ or $m=1.5$ does not?

* In Theorem 4.14, the authors discussed the on manifold regularization induced by weight decay under different $q$. However, as shown in Table 1, when $q=2, L=2$, the decoupled weight decay induces an $L_1$ bias; whereas when $q=1, L=2$, it induces an $L_{3/2}$ bias, which is less sparse than $L_1$. As sparsity biases have been linked to feature learning in some settings, this observation actually suggests that $q=2$ may facilitate feature learning instead of a smaller $q$.  

The authors also claimed that "decoupled weight decay, as in AdamW, stabilizes feature learning by enforcing novel balance equation". The balance equation in Lemma 4.4 indicates that the weight decay encourages the weights to become more balanced during training. But it is not clear to me how this balancing translates into a more "stable" feature learning. 

I suggest that the authors clarify in the paper what they mean by "feature learning" and "stable feature learning", and then compare the algorithms with different values of $q$ as well as with and without weight decay.


**Initial recommendation**

Overall, while the study of how optimization geometry and reparametrization affect the dynamics and the proposed framework are very interesting, I find the main claims of this paper, in particular those on saddle escaping and feature learning, insufficiently supported. Therefore, my initial recommendation is rejection.

### Questions
I find some parts of the presentation unclear. Please see the questions below. 

**Notations** 

* In Example 3.2, is the expression "a deep diagonal linear network $g(m,w)=m \odot w$" actually referring to a shallow network, as there are only two weight matrices $\operatorname{Diag}(w), \operatorname{Diag}(m)$? 

* In Definition 3.3 and 4.5, does $\mathbb{I}_n$ denote the all-one vector? This symbol is commonly used for the identity matrix, so clarification would be helpful. 

* In Corollary 4.8, the notation $w_i=\lambda>0$ is ambiguous. Does it mean the vector $w_i$ has all entries equal to $\lambda$? 

**Regarding Theorem 4.6**

* $\lambda−L_p-$balancedness in Definition 4.5 is defined for shallow network with two weight matrices. Then what does "$\lambda−L_p-$balanced with respect to the first parameter $w_1$" mean in Theorem 4.6, which is stated for deep networks? 

* In Lemma 4.4 and Theorem 4.6, in what sense does "almost everywhere" mean (e.g.,"steepest descent satisfies a separable $L_p$-mirror flow almost everywhere")? Does it mean the result may fail only for a measure-zero set of initializations?

* Corollary 4.12 discussed when the function $R_{L_p,\ L}$ is a Legendre function. However, in Theorem 4.6, there seems no restriction on $p$ or $L$. Does Theorem 4.6 indicate that such an $R_{L_p,\ L}$ always exists, even in cases where it is not a Legendre function?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces steepest mirror flows as a unifying geometric framework to study how optimization algorithms influence reparametrization. By analyzing diagonal linear networks and deep diagonal reparameterizations, the authors show that steeper descent methods, such as sign-based variants, escape saddle points more efficiently than gradient descent. Empirical results from linear regression, classification, and fine-tuning experiments confirm the theoretical predictions of faster saddle escape, stable learning, and distinct sparsity dynamics between coupled and decoupled weight decay.

### Strengths
This is an interesting study with insightful findings. Combining mirror descent with reparameterization is a great idea.

### Weaknesses
The statement of the first contribution is misleading. This is not the first work studying GF and reparametrization. It is probably meant with respect to the family. 

For this type of work, assumptions are always problematic. They are very simplistic (diagonal networks) but it is very hard to potentially show more general results. 

In experiments, rather than studying the networks considered in the analysis, they should see if the results hold when the assumptions are not fulfilled (for example, other networks).

### Questions
What is meant by: iterates x_t converge? I assume it is meant lim_t->\infinity x_t

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes “steepest mirror flows” to explain why Adam/AdamW beat SGD in fine‑tuning via faster saddle escape and different implicit regularization.
Steeper (sign‑like) geometry helps saddle escape and that decoupled weight decay (AdamW) stabilizes feature learning, with theory for deep diagonal reparameterizations and small fine‑tuning case studies. Most formal results hold only in separable/diagonal settings; the transformer link is indirect; and empirical support for fine‑tuning claims is narrow.

### Strengths
The message that adam escapes saddles better than SGD is believable and may be cool. However, it is not well established. The rest of the claims are not substantiated.

### Weaknesses
#### How is this about fine-tuning transformers?
There is not even a linear diagonal attention there, no one ever claimed that a diagonal network is a good model for a transformer, because it is not. How do you argue this? 

#### Mirror flow study is incremental and does not adequately support the thesis.
While the diagonal‑network analysis is neat, it is very similar to existent ones and does not bring any real novelty to the community. 
*I believe, it is extremely incremental.*
It is way too limited to show that your sign-mirror-descent escapes saddles to claim that *adam* escapes saddles *in transformers*.
There is a too big of a jump from the mathematical argument to the goal. Moreover, I think that such a result is a perturbation of existent ones.

#### Order-2 saddles
So what? We know they are present in neural networks, actually, arguably also of higher order. A long line of works that you do not cite addresses this issue much more in general. That same line of work empirically and sometimes theoretically notice that they are not a problem in practice. You should discuss this line of research. On top of it saddles are not an issue in linear networks generally cause the standard initialization is with high probability outside of the area with saddles.

#### Title mismatch

Even the title is an oversell. It missmatch with the paper, this is not a paper about generally reparameterizing steepest descent as mirror flow. This is a paper about adam escaping saddles which lack novelty and does not support their claims. 

#### Experiments
For cifar10 actually SGD generalizes better than Adam with CNNs, and this is also a classical result. I really do not understand the whole point of the paper and how these are supporting experiments.

#### Conclusions
Even though experiments are present, I thus believe the central claims are not adequately supported and the research methodology is not sound.

### Questions
-

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies continuous-time steepest descent methods, specifically algorithms taking the form:
$$\begin{align} d x_t = - \mathrm{sign} \left( \nabla_x f(x_t) \right) \odot \left| \nabla_x f(x_t) \right|^{q-1}, \end{align}$$
where $q \in [1,2]$.

Different values of $q$ result in different trajectories. For example:
 For $q=2$, the algorithm becomes gradient flow, which is the continuous-time approximation of gradient descent. For $q=1$, it becomes SignGF (Sign Gradient Flow), which serves as a good proxy for studying optimizers like Adam. The architecture on which the paper focuses is deep diagonal reparameterization, defined as $x = g(w) = \prod_{i=1}^{L} w_i$, where $x$ is represented as the product of $L$ scalars.

The authors demonstrate that under $\lambda$-balanced initializations, these steepest flows can be re-parameterized as steepest mirror flows. Using this reparameterization, the paper analyzes how quickly the flow escapes saddle points and investigates the effect of both coupled and decoupled weight decay.

### Strengths
The paper provides a seperation result for signGD with coupled and decoupled weight decay and show that they have different regularization properties which is interesting.

### Weaknesses
a) The paper focuses on deep diagonal reparameterizations which is a product of one-dimensional variables with a particular initialization shape. The setting is restrictive to generalize the results of the paper.

### Questions
1) How is the manifold regularizer derived ? Is it due to the fact that steepest descent with de coupled weight decay can be written as 

$$ d( \nabla_x R(x_t) ) = - \mathrm {sign} \left( \nabla_x f(x_t) \right) \odot \left[   \nabla_x f(x_t) \right]^{q-1} dt  - \nabla M_{reg}(x) dt  $$

it would be nice to detail this as the manifold regularizer appears a bit abrupt. 

2) The saddle points defined in Theorem 4.3 are not saddle for with coupled or decoupled weight decay so the feature learning results only work without the weight decay. This point needs to be clearly mentioned in the manuscript. 

3) From the abstract, how is the deep diagonal reparameterizations a proxy for attention ?

### Soundness
3

### Presentation
3

### Contribution
2
