# Noisy Interpolation Learning with Shallow Univariate ReLU Networks

- Avg Score: 8.00
- Decision: Accept (spotlight)
- Scores: 8, 8, 8

## Abstract
Understanding how overparameterized neural networks generalize despite perfect interpolation of noisy training data is a fundamental question. Mallinar et. al. (2022) noted that neural networks seem to often exhibit ``tempered overfitting'', wherein the population risk does not converge to the Bayes optimal error, but neither does it approach infinity, yielding non-trivial generalization. However, this has not been studied rigorously.  We provide the first rigorous analysis of the overfiting behaviour of regression with minimum norm ($\ell_2$ of weights), focusing on univariate two-layer ReLU networks.  We show overfitting is tempered (with high probability) when measured with respect to the $L_1$ loss, but also show that the situation is more complex than suggested by Mallinar et. al., and overfitting is catastrophic with respect to the $L_2$ loss, or when taking an expectation over the training set.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the min-$\ell_2$ norm interpolation learning of two-layer ReLU neural networks under the univariate data. The results start from linear splines, demonstrating that linear-spline interpolator exhibits tempered behavior. Then the min-norm interpolator is studied based on the relationship with the linear-spline. Then the results show that, tempered overfitting occurs in the Lp space with $1 \leq p <2$ but catastrophic overfitting occurs for $p > 2$.

### Strengths
The generalization performance of linear splines and min-norm solution is studied in terms of tempered overfitting and catastrophic overfitting

### Weaknesses
There is no distinct drawback in my view. The proof is based on the nice statistical property of $\ell_i$, which might be the key technical difficulty when extending to the $d$-dimensional data. For example, the two-dimensional data, we sort the data, split the space, define the risk in the two-dimensional interval. But the estimation based on $\ell_i$ is unclear to me.

Besides, comparison with (Kornowski et al. 2023) requires more discussion, especially in terms of the technical tools.

Apart from this, some references on the following topics in the related work are missing:
-	Neural networks, the representer theorem, splines, Banach spaces
-	Benign overfitting papers

### Questions
- Could you please intuitively explain why catastrophic overfitting occurs for $p>2$ under the min-norm solution while linear spline does not?

- Can Linear splines obtain better performance than the min-norm solution?
- What is the result under the min-$l1$ norm solution? Intuitively, there is no significant difference for univariate data but I expect that some results from previous min-l1 norm literature can be discussed under the univariate setting.
- What’s the meaning of “mildly dependent random variables” of $\ell_i$?
- How does $(n+1)/X -> 1$ hold with almost surely?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper delves into the nuanced behaviors of tempered overfitting and catastrophic overfitting within regression scenarios employing a min-norm two-layer ReLU network with skip connections. The key contribution lies in the establishment of significant results, notably demonstrating the occurrence of catastrophic overfitting when the $L_p$ loss is applied with $p\geq2$. Furthermore, the paper uncovers the phenomenon of tempered overfitting, which surfaces when utilizing the $L_p$ loss with $1\leq p<2$.

In a noteworthy extension of its findings, the paper also establishes that when working with samples distributed on a grid, tempered overfitting manifests for the $L_p$ loss with $p\geq1$. These results shed valuable light on the interplay between loss functions, network architecture, and data distribution in the context of regression with min-norm ReLU networks.

### Strengths
This paper is exceptionally well-crafted, boasting a highly organized structure that enhances its clarity and readability. The main paper is thoughtfully structured, and the presentation of the proof concept is remarkably accessible, thanks in part to the informative graphs provided.

The theoretical framework is excellent, as this paper conducts a comprehensive examination of the overfitting tendencies observed in min-norm ReLU networks within the context of regression. In doing so, it effectively bridges a critical gap, especially when contrasted with the closely related work by Kornowski et al. (2023), which primarily addressed overfitting within a classification setting.

### Weaknesses
It appears that this paper can be regarded as a subsequent work to Boursier & Flammarion (2023). The connection is evident as the pivotal lemma (Lemma 2.1) employed in this paper is directly drawn from Boursier & Flammarion (2023). Furthermore, the neural network model studied in this paper aligns with the one extensively examined in Boursier & Flammarion (2023). Consequently, the technical innovation in this paper seems somewhat limited in this regard.

It's important to note that this paper adopts a one-dimensional perspective, assuming $x$ to be a single dimension, confined within the range $x\sim[0,1]$. In contrast, Kornowski et al. (2023) considered high-dimensional $x$. Another differentiating factor is that this paper primarily focuses on characterizing the asymptotic behavior of population error and reconstruction error, while Kornowski et al. (2023) delved into an analysis that extends beyond the asymptotic realm.

In Theorem 4, which addresses catastrophic overfitting for $L_p$ with $p\geq 2$, it is essential to note that only a specific case with $f^*=0$ is considered. To enhance the depth and relevance of the analysis, it would be particularly intriguing to explore this phenomenon with a more general $f^*$.

### Questions
1. I have observed that in both your Theorem 1 and Theorem 2, you made the assumption that $f^*$ is a Lipschitz function. Assuming the Lipschitz constant is denoted as $c$, I am interested in understanding how the constants $C_p$ in Theorem 1 and the constant $C$ in Theorem 2 are related to this Lipschitz constant $c$.

2. What would be the impact on the analysis if we were to assume that $f^*$ is Holder continuous instead of Lipschitz continuous in your main theorem? Could this alternative assumption be beneficial, given that you are dealing with $L_p$ loss in this context?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tries to understand the generalization performance of overparametrized neural networks when interpolating noisy training data. Specifically, the authors consider the univariate 2-layer ReLU networks in regression setting and focus on the min $\ell_2$ norm interpolator. This paper shows that the generalization performance is subtle and depends on the factors such as the choice of loss at evaluation time and whether one is considering high probability case or taking expectation over the training samples. The overfitting is tempered (test loss neither goes to Bayes optimal nor to infinity) when loss is $L_p$ for $1<p<2$ and considering the high probability outcomes. The overfitting is catastrophic (test loss goes to infinity) when loss is $L_2$ or considering expectations.

### Strengths
-	Understanding the generalization performance of interpolating solutions, especially non-linear interpolators such as ReLU networks, is an important and interesting question in deep learning theory.
-	The paper gives a detailed characterization of the min $\ell_2$ norm interpolating ReLU networks, compares it with the linear splines, and shows the subtle generalization performance depending on the loss function. I believe this is a good result and it seems to be novel in the literature of implicit bias and benign overfitting.
-	The paper is overall well-written and easy-to-follow.

### Weaknesses
-	The paper focuses on the univariate ReLU networks which is relatively simple. (though it is understandable from technical point of view)
-	The results are in the asymptotic regime that sample size $n$ goes to infinity. Thus, it does not give explicit rate of convergence.

### Questions
-	The results in this paper seem to consider the asymptotic regime that sample size $n$ goes to infinity. I was wondering if there is a rate for this asymptotic convergence.
-	I was wondering if the min-norm interpolation considered in the paper can be naturally reached based on the implicit bias of some simple algorithms. For example, the authors mentioned that Shevchenko et al. showed similar spikes as the current paper. Does their algorithm exactly lead to the same implicit bias here?
-	I wonder if results like Theorem 1 and Theorem 5 are enough to say they are tempered overfitting, as they are only upper bound on the test loss. I feel there needs to also have a lower bound on the test loss?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
