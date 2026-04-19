# Learning Hierarchical Polynomials of Multiple Nonlinear Features

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6, 6

## Abstract
In deep learning theory, a critical question is to understand how neural networks learn hierarchical features. In this work, we study the learning of hierarchical polynomials of multiple nonlinear features using three-layer neural networks. We examine a broad class of functions of the form $f^{\star}=g^{\star}\circ \mathbf{p}$, where $\mathbf{p}:\mathbb{R}^{d} \rightarrow \mathbb{R}^{r}$ represents multiple quadratic features with $r \ll d$ and $g^{\star}:\mathbb{R}^{r}\rightarrow \mathbb{R}$ is a polynomial of degree $p$. This can be viewed as a nonlinear generalization of the multi-index model, and also an expansion upon previous work on nonlinear feature learning that focused only on a single feature (i.e. $r = 1$). 
    Our primary contribution shows that a three-layer neural network trained via layerwise gradient descent suffices for  
 - complete recovery of the space spanned by the nonlinear features 
- efficient learning of the target function $f^{\star}=g^{\star}\circ \mathbf{p}$ or transfer learning of $f=g\circ  \mathbf{p}$ with a different link function
   
within $\widetilde{\mathcal{O}}(d^4)$ samples and polynomial time.
For such hierarchical targets, our result substantially improves the sample complexity ${\Theta}(d^{2p})$ of the kernel methods, demonstrating the power of efficient feature learning. It is important to highlight that our results leverage novel techniques and thus manage to go beyond all prior settings such as single-index and multi-index models as well as models depending just on one nonlinear feature, contributing to a more comprehensive understanding of feature learning in deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work studies the problem of learning functions $f^* : \mathbb{R}^d \to \mathbb{R}$ of the form $f^* = g^* \circ p$ where $p : \mathbb{R}^d \to \mathbb{R}^r$ returns $r$ quadratic features of the input, and $g^* : \mathbb{R}^r \to \mathbb{R}$ is a polynomial of degree $p$. By analyzing a gradient-based algorithm for training a three-layer neural network with specific choices of activation, the authors prove that a sample complexity of $\tilde{O}(d^4)$ is sufficient for learning this function, as opposed to the $\Omega(d^{2p})$ sample complexity lower bound for learning with rotationally invariant kernels.

### Strengths
Presenting a guarantee for learning a polynomial of multiple quadratic features using three-layer neural networks is a new direction in the literature which extends beyond the guarantees for learning multi-index models with two-layer neural networks and learning polynomials of a single non-linear feature which existed in prior works. More broadly, this work contributes to finding settings in which networks deeper than two layers are natural for learning certain function classes, which is an area of particular interest for the community.

### Weaknesses
* The feature learning regime considered in this paper is limited in the sense that the first layer weights are fixed at random and not trained. Training these weights could lead to learning more complex hierarchies, such as hierarchies of multi-index models, although admittedly challenging for a theoretical analysis.

* The choice of activation for the second layer is rather specific, and choosing a different activation for each layer seems a bit odd when compared with standard three-layer neural networks. However, I believe there might be a way to choose the same activation as my impression is that $\sigma_1$ only has a constant part and no $Q_1$ part, thus one only needs to introduce a fixed bias to the output of the network. If this interpretation is correct this might be worth mentioning in the main text.

* While the statistical model is naturally suitable for three-layer networks, I could not find any lower bounds in this submission which explain that learning this function with a two-layer neural network is provably less efficient (in terms of samples or computation). I am wondering if the authors could use the results of prior works to show a statistical or computational lower bound holding in this setting.

### Questions
* What is the main technical challenge in directly generalizing the result of Nichani et al., 2023 to the setting of having multiple non-linear features? Specifically, if $f^* = g \circ p$ where $p = (p_1,...,p_r)$ and each $p_j$ is a degree $k$ polynomial of $x$, and $g$ is a degree $q$ polynomial, is it possible to argue that in the regime of constant $r$, the sample complexity of learning $f^*$ with three-layer neural networks is $d^k$, as opposed to $d^{kq}$ of kernel methods?

* On the discussion of learning linear features, a relevant work may be [1] that considers learning general multi-index models with two-layer neural networks via a noisy gradient descent algorithm. For single-index models, while the authors have included works in the CSQ framework, there are more recent advances on gradient-based learning with two-layer neural networks that go beyond the CSQ framework [2, 3], or consider structured rather than isotropic input covariance [4, 5].

---
[1] A. Mousavi-Hosseini et al. "Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics". arXiv 2024.

[2] Y. Dandi et al. "The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents". ICML 2024.

[3] J. D. Lee et al. "Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit". arXiv 2024.

[4] A. Mousavi-Hosseini et al. "Gradient-Based Feature Learning under Structured Data". NeurIPS 2023.

[5] J. Ba et al. "Learning in the presence of low-dimensional structure: a spiked random matrix perspective". NeurIPS 2023

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors prove that three-layer networks can learn nonlinear, multi-index targets more efficiently than kernel methods. Quite a bit of literature already exists on 2- and 3-layer networks learning single-index targets of linear and non-linear features, and also multi-index targets of linear features. To the best of my knowledge this is the first set of results for learning multiple, non-linear features. 

The analysis is valid of an algorithm that does a single step on the first layer weights, W, and then completely trains the rest of the network. The authors use this two-stage training methodology to also prove a transfer learning result, where the link function is changed. 

Finally the authors provide some empirical evidence that supports their theory. In particular, their approach appears to improve sample complexity over a random feature model. They also support their findings regarding efficient transfer learning and the reconstruction of quadratic features.

### Strengths
This is a strong submission in most respects: relevance, rigor, and clarity. I have minor comments and questions below.

### Weaknesses
1. Some of the assumptions might be too restrictive for the results to be relevant for practice (eg parts of Assumption 1, and Assumption 4). 
2. Algorithm 1 does but a single update on W. Then parameter a is trained on multiple steps. While this style of analysis is common in the area, there already exist articles that include analysis of many-step gradient descent for all parameters.

### Questions
1. The “balanced features” part of Assumption 1, where the operator norm of Ai satisfies a particular bound. This bound essentially tells us that each feature cannot depend on a sparse subset of inputs. Can you discuss the implications of that assumption and pertinence to real distributions? 
2. Similarly, can you discuss the implications of Assumption 4, especially the Gegenbauer expansion? Activation functions used in practice are not even, and many of them are actually odd.
3. Do you expect that the d^4 dependence on dimension is necessary for quadratic features?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this work, the authors studies theoretically the setting of using three-layer neural networks to learn a type of "multi-index quadratic models" under a uniform data distribution on the sphere. With suitable activation functions and layer-wise training, the model is proved to be able to learn both the output function and the quadratic features with a sample complexity that scales as $d^4$.

### Strengths
The type of model studied in this paper generalizes the one considered by Nichani et al. (2023) by allowing multiple quadratic features instead of just one in the first layer. To prove the learning guarantee, the authors introduced a nice technical result on the distribution of multiple nonlinear functions of Gaussian inputs, which is interesting mathematically and novel to my knowledge. The theoretical results appear to be sound though they have not been thoroughly checked.

### Weaknesses
**Beyond the mathematical and technical**: It could be argued that the series of assumptions on the layer-wise learning algorithm, the activation functions, the data distribution, etc. are quite specific and restrictive, and the contribution of this work (relative to e.g. Nichani et al., 2023 and Wang et al., 2023, which share much of a common analytical framework) resides more in the improved mathematical results than advancements in new learning theory insights.

**Some confusion regarding the proof of Theorem 1**: The authors' description of the second stage in the proof of Theorem 1 gives the impression that it breaks down into two parts: 1) showing the existence of an $a^*$ achieving small empirical loss, and 2) using Rademacher complexity analysis to bound the generalization error of $a^*$. But they alone wouldn't be sufficient since we know only the existence of the $a^*$ but not its learnability by the proposed algorithm. And indeed, the actual proof steps taken by the authors is different (and correctly so), where they use the strong convexity of the loss function to show that an alternative solution $\hat{a}$ can be learned by gradient descent and achieves low empirical loss as well as generalization error. Even though $a^*$ still makes its appearance in the proof argument in Section D.1.2, I suspect that it is not truly necessary and only complicates the proof. If this assessment is correct, it would imply that Proposition 2 is redundant for the proof of Theorem 1, which should be commented on in order to avoid confusion.

Minor typos:
1) In Section 1.1 on Page 2, "outer layer can achieves" --> "outer layer can achieve".
2) In Section 4.1 on Page 7, "trained presentations representations" --> "trained representations".
3) The links referencing Proposition 6 are misplaced.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies hierarchical feature learning in fully-connected neural networks when trained with gradient descent. In particular, the authors consider a nonlinear (quadratic) multi-index model with $f^*(x) = g^*(x^\top A_1 x, \ldots, x^\top A_r x)$ for $x \in \mathcal{S}^{d-1}$ and $r\ll d$. Under approximate orthogonality, well-balancedness conditions of the quadratic features, and additional well-conditioned assumptions on the link function, the authors show that a layer-wise gradient descent method on a three-layer neural network (first layer is never trained, and activations are carefully chosen) can learn $f^*$ with roughly $\tilde{O}(d^4 + d^2r^{p+1})$ (ignoring $\kappa$) to irreducible error $O(r^{p+2}/d^{1/6})$, and approximately recover the $r$-dimensional subspace to allow for transfer learning to a different link function.

### Strengths
- The paper studies an interesting model of feature learning, generalizing prior work that consider either linear multi-index settings or non-linear single-index models.
- The authors show that gradient descent recovers the subspace which allows them to give transfer learning results for any link function.
- To prove the results, the authors prove a new universality theorem for multiple-dimensions, generalizing previous known results. Though the proof seems to follow from existing tools in high dimensional probability, it seems novel.

### Weaknesses
- The paper makes several assumptions and tweaks that seem to be done to make the proofs easier: layer-wise training, removing lower order terms in activation functions, taking almost orthonormal equally important features, etc. Most of these are standard is deep learning theory, but the activation assumption seems more artificial, and tailored to make the analysis easier. Some assumptions (Assumptions 2 and 3) can potentially be relaxed if the goal is only getting low loss and not subspace recovery.
- Compared to prior work (Nichani et al. 2022) that was for general hierarchal functions, this paper is narrow in its scope and relies heavily on tools already available in this prior work, while requiring additional assumptions. I do understand that the prior work did not look at multiple quadratic features, but limiting to quadratic seems like a limitation of the techniques.
- The experiments are limited to following the theoretical assumptions precisely. This is not very insightful. If the experiments are to remain synthetic, then it would be worthwhile seeing how the scaling with different parameters in the bounds shows up in experiments. Also, how much can you relax the assumptions, and still get the same empirical results. It would also be worthwhile to come up with a real-world dataset where this quadratic model makes sense, and gains of three vs two layers can be shown.

### Questions
- Why did the authors consider only uniform over the sphere compared to Gaussian, which is more commonly studied in this literature? Do the results extend to more general distributions?
- There is no lower bound on the optimality of the dependence on $d$ and $\kappa$ in the bound. It should be possible to get $d^2$ potentially. What dependence on $\kappa$ is optimal?
- Figure 1 is not very easy to understand. Might be helpful to add more explanation to the caption and label the different blocks/terms. It took me a while to parse it and I did not find it very helpful to understand the proof. 
- The irreducible error term of $O(r^{p+2}/d^{1/6})$ is not discussed. This is much larger than the $\frac{\kappa^{1/3}}{d^{1/6}}$ term in Niching et al. (2022) and requires $r$ to be much smaller than $d$. Is this necessary?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors study layerwise training on a three layer neural network trained on a more general target function with a more involved "hierarchical feature" structure. They show sample complexity improvements over the classic kernel methods.

### Strengths
- Feature learning is one of the most fundamental problems in deep learning theory.

- The proof seem to be correct. Nice proof sketches are provided in the main text.

### Weaknesses
- I don't find the results of this paper particularly interesting. I would have easily guessed that these results should be obtainable for the case where r \neq 1, after looking at Nichani et al., 2023; Wang et al., 2023. See also my other comments below:

- The results of the paper are not tight; i.e., the sample complexity of O(d^4). Essentially, one would expect a sample complexity of $O(d^2)$, because there are $O(d^2)$ parameters to be learned in this setting. In particular, a more interesting study will consider $n \asymp d^2$ and provide a precise characterization of the limiting test error. In the single index case, this is similar to the benefit show by Ba et al. 2022 (and others) over the initial results of Damian et al 2022.

- In assumption 2, it is assumed that the Hessian has eigenvalues that are bounded below by a certain quantity. Can this be translated into a condition on the target function in terms of some quantity like information-exponent or leap-index? Right now, the assumption seems very artificial.

- What do the authors mean by " Moreover, when the entries of Ai are sampled i.i.d., the assumption is satisfied with high probability by Wigner’s semicircle law" in line 157?

- In the layer-wise training, the inner most layer is not trained and is kept at initialization; i.e., random features.

- Assumption 4 on the activation function is completely unrealistic and is only an artifact of the proof.

- What is the role of the update on the bias. Some prior work in the area consider neural networks without bias (e.g., the papers I list below). Can the authors comment on this please?

- The discussion of prior work on "Learning Linear Features" is very limited. There are many more results on this topic, studying feature in this case with detail. The authors should consider citing and discussing these papers (alongside any other paper on the topic that I might have missed here).

 [1] Dandi et al., The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents

 [2] Moniri et al., A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks

 [3] Cui et al., Asymptotics of feature learning in two-layer networks after one gradient-step

 [4] Dandi et al., A Random Matrix Theory Perspective on the Spectrum of Learned Features and Asymptotic Generalization Capabilities

 [5] Wang et al, Nonlinear spiked covariance matrices and signal propagation in deep neural networks

### Questions
please see the weakness section.

### Soundness
4

### Presentation
4

### Contribution
2
