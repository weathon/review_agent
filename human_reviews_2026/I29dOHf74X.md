# Regret Analysis of RMSProp and AdamNC for Training Deep Interpolating Neural Networks

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
We provide a theoretical analysis for RMSProp and AdamNC (Adam without corrective terms) for training deep fully-connected neural networks with smooth activations, in the online learning setting. We focus on the binary classification tasks with logistic loss or exponential loss. We assume that the model can interpolate data, i.e., it can obtain an arbitrarily small loss $\varepsilon$, while the distance to the initialization is bounded by a decreasing function $g(\varepsilon)$. We show that the regret is upper bounded by $\mathcal{O}(\text{poly} [g(1/T )])$ provided that the width is at least $\mathcal{O}(\text{poly} [g(1/T )])$, where $T$ is the total iteration number. We further show that under NTK-separability, the regret is less than $\mathcal{O}(\text{poly}(\log T))$ when the width is larger than $\mathcal{O}(\text{poly}(\log T))$. We also provide a comparable regret bound for the scalar version of RMSProp and AdamNC, without requiring prior knowledge of problem parameters for learning rates. Our analysis can also be applied to smooth losses, leading to similar regret bounds.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper gives a regret analysis of RMSProp/AdamNC when training deep smooth neural networks in an online manner specifically in the NTK interpolating setting. The main results are sufficient conditions on the width of the network and iterations numbers that ensure regret analysis and the derivation of the regret bounds which parallel online SGD bounds. The approach is based on exploiting the weak-convexity of the objective trained by losses such as logistic loss in the kernel regime.

### Strengths
The majority of the paper and the stated results are well-written. The studied problem is new, whereas prior works mainly tackled the convex case. The results show that even small networks of poly-logarithmic width can have favorable regret bounds. The analysis nicely extends the current GD analyses to ADAM and covers the constant step-size for this algorithm.

### Weaknesses
-The take-away message of the paper is not well-stated. In particular, a brief discussion on the following questions seems lacking from the current version: what are the distinctions between the bounds resulting from this over bounds resulting from using S/GD? Can ADAM improve upon GD in width requirements or final regret bounds? Are the current bounds tight? How does the analysis stand compared to the analysis of convex objectives? 

-The analysis seems to rely on the known methods (especially (Taheri and Thrampoulidis, 2024)). It can be clarifying if you can include a discussion on the distinct steps from the current SGD analyses. 


-Do experiments verify the bounds on the regret bounds or the width conditions?

### Questions
please see above section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides convergence guarantees of training a deep neural network with RMSProp and AdamNC under interpolation assumption. Their assumptions and proof resemble [1] in fashion: the two important points are using an approximate convexity of deep neural network, and using the interpolation point as a "reference point" that has low training loss. With the interpolation assumption the regret becomes a function of $g$ that dictates the interpolation property, which is potentially better than standard results for well-behaving $g$. 

[1] Taheri, Hossein, and Christos Thrampoulidis. "Generalization and stability of interpolating neural networks with minimal width." Journal of Machine Learning Research 25.156 (2024): 1-41.

### Strengths
It is a solid contribution to extend certain results to different optimizers. Convergence guarantee better than O(\sqrt(T)) can be attained by certain assumptions look interesting.

Also, the paper is very well written and easy to understand the mathematical formalism. The lemmas are well stated, with exact assumptions, with theorems that look valid.

### Weaknesses
It would be better if more motivation was given for studying neural networks in the interpolation setting. Especially Assumption 4 seems like a very strong assumption to me, and I was a bit confused because in line 57-58 it states that the setting has been studied in different papers, whereas when I read the papers they do have min-margin assumptions but not exactly the one discussed in Assumption 4, except for [1]. So I have two questions:

- Is this theoretical assumption widely used in the exact form proposed in the paper? e.g. are there different papers that exactly show this form of assumption? If yes, it would be good to mention how they are used in different papers. If not but they are associated somehow, it would also be good to clarify it. It could be the case that I am missing something apparent.

- Is this theoretical assumption valid? e.g. is it verifyable by experiments? Are there any experiments that support this assumption?

Clarifying the questions would make the paper stronger.

### Questions
See weaknesses

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
2

### Summary
This paper studies the regret analysis of AdamNC (without debiasing correction) and RMSPropNC (as a special case of AdamNC without first momentum) with a structured loss function which is constructed as $F_t(w)=f(y_t\Phi(w,x_t))$ where $\Phi$ denotes a fully connected MLP neural network for binary classification tasks with data $\\{(x_t,y_t)\\}$. The main assumptions are that $f$ is self-bounded or smooth and that the model $\Phi$ has interpolation ability such that there exists a decreasing function $g(\epsilon)$ such that any $\epsilon$ corresponds to a nearly optimal parameter $w^{(\epsilon)}$ such that $\sum F_t(w^{(\epsilon)}) / T \le \epsilon$ and $\\|w^{(\epsilon)}-w_1\\|\le g(\epsilon)$. Under these assumptions, this paper provides a convergence guarantee of AdamNC and shows that it achieves $O(g^3(\epsilon/T))$ for self-bounded or smooth $f$ when the model width is larger than certain threshold. In particular, if the model is NTK-separable and $f$ is the exponential loss or logistic loss, then $g(\epsilon)$ has an explicit form $g(\epsilon) \sim \log(1/\epsilon)$, and the previous regret bound becomes $\mathrm{polylog}(T)$. The analysis is also extended to AdamNC-Norm, where the preconditioner aggregates the norm of gradient instead of per coordinate.

### Strengths
The main strength of this paper lies in its novelty significance. In particular, this paper provides a novel theoretical analysis of regret bound of training neural networks with the AdamNC optimizer, which is rarely studied in any prior work. This helps to better understand the empirical effective of the popular Adam optimizer from a different perspective. Moreover, the technical results on the theoretical analysis is very concrete. It provides systematic analysis under different assumption, e.g., the loss being self-bounded or smooth, and different model structures, e.g., with and without the dimension normalization per-layer.

### Weaknesses
One limitation is that the convergence results in this paper requires a minimum model width to be true, and that threshold is usually asymptotically larger than the convergence rate (e.g., the width needs to be $O(g^4(\epsilon/T))$ to achieve $g^3(\epsilon/T))$ in Thm 1 and 2). This setup does not reflect the practical setting of training neural networks, where the total iteration usually has larger orders compared to the model width.

### Questions
- In general (without NTK-separability), is there an explicit form for $g(\epsilon)$? Could the author provides some example to help understand the shape of this function and how it's related to practical training in real life?
- Assumption 4: does it implicitly assumes $\inf f = 0$ so that $F(w)\le \epsilon$ is always achievable for any small $\epsilon$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the convergence of adaptive methods (RMSProp and AdamNC) in deep fully connected networks for online binary classification with smooth activation functions. The authors prove an $O(polylog(T))$ regret bound for sufficiently wide networks in the NTK regime. This is comparable to rate for strongly convex online optimization. They also analyze scalar variants (RMSProp-Norm and AdamNC-Norm) that achieve similar bounds without requiring prior knowledge of problem constants. The proof builds on the idea of Bregman Proximal Gradient and applies it within the NTK framework.

### Strengths
-	Analyzing the convergence of adaptive methods is an important research question.
-	The paper is overall clearly written and includes a helpful proof sketch to illustrate the main idea.
-	The paper establishes the convergence of adaptive methods for deep networks in the NTK regime, which appears to be new.

### Weaknesses
-	The paper focuses on the NTK regime, where neural networks are known to behave similarly to linear or kernel methods. However, this setting does not always reflect the behavior of practical networks.
-	The main technique appears similar to those in Duchi et al. (2011) and Alacaoglu et al. (2020) for handling adaptive methods, and is applied here to the specific NTK setting. It would be helpful to clarify whether any new challenges arise in this context or if additional techniques were required to address them.

### Questions
- For Theorems 1 and 2, I wonder why the stepsize $\eta$ must have the exact order specified in the statement, rather than simply being any sufficiently small value. What is the intuition behind this requirement?
- Can Assumption 4, which assumes the existence of such a function $g(\epsilon)$, hold in more interesting regimes beyond the NTK setting (Assumption 5)?
- Do the results provide any insight into the potential advantages of using adaptive methods over vanilla gradient descent?

### Soundness
3

### Presentation
3

### Contribution
2
