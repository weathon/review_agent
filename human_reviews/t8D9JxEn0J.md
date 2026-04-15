# Malcom-PSGD: Inexact Proximal Stochastic Gradient Descent for Communication Efficient Decentralized Machine Learning

- Decision: Reject
- Scores: 5, 6, 5, 3

## Abstract
Recent research indicates that frequent model communication stands as a major bottleneck to the efficiency of decentralized machine learning (ML), particularly for large-scale and over-parameterized neural networks (NNs). In this paper, we introduce \textsc{Malcom-PSGD}, a new decentralized ML algorithm that strategically integrates gradient compression techniques with model sparsification. \textsc{Malcom-PSGD} leverages proximal stochastic gradient descent to handle the non-smoothness resulting from the $\ell_1$ regularization in model sparsification. Furthermore, we adapt vector source coding and dithering-based quantization for compressed gradient communication of sparsified models. Our analysis shows that decentralized proximal stochastic gradient descent with compressed communication has a convergence rate of $\mathcal{O}\left(\ln(t)/\sqrt{t}\right)$ assuming a diminishing learning rate and where $t$ denotes the number of iterations. Numerical results verify our theoretical findings and demonstrate that our method reduces communication costs by approximately $75$\% when compared to the state-of-the-art method.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper focused on the communication bottleneck in decentralized ML problems. The authors provided a method named MALCOM-PSGD to reduce communication cost, which integrates the model sparsification and gradient compression. An $O(1/\sqrt{t})$ rate was established to guarantee the convergence of the proposed algorithm. Also the authors used numerical experiments to validate the performance.

### Strengths
The paper addresses the challenge of overcoming communication bottlenecks in the context of decentralized machine learning, with a particular focus on training large-scale, over-parameterized neural network models. The authors provided theoretical analysis and numerical experiments to validate the performance of the proposed algorithm.

### Weaknesses
The paper studied the traditional communication bottleneck issue in decentralized ML problems. The proposed algorithm appears to amalgamate two pre-existing techniques, exhibiting rather modest advancements. Moreover, the theoretical analysis appears to follow a standard course, and there is room for enhancing the experimental aspects.

### Questions
1. Can authors provide more explanation for the reason of combining model sparisificaion and gradient compression? Though both of them can improve communication efficiency, can the combination achieve `1+1>2` improvement? Also, is there any additional challenge from either implementation or theoretical analysis caused by the combination? 

2. If we look at the convergence rate in Theorem 2, it is w.r.t. to $\mathcal{F}$ instead of $F$. To give a fair comparison, can the result be extended to  $F$?

3. After adding the model sparisificaion,  how will the model generalization be changed (gain or loss) in either theoretical way or numerical experiments? This part remains very unclear.

4. There are many other communication efficient methods, such as local updates/signSGD, why don't compare with them at least in the numerical experiments? I feel the numerical experiments can include many other communication efficient decentralized optimization methods.

5. From the experiments, the improvement w.r.t. test accuracy seems not very signification. Any analysis on it?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes MALCOM-PSGD, a communication efficient decentralized optimization algorithm for smooth and non-convex objectives that combines several existing techniques for communication cost reduction.  The first one is to promote the sparsity of node model parameters using $\ell_1$ regularization and MALCOM-PSGD optimizes the resulting non-smooth objective using proximal SGD.  Furthermore, MALCOM-PSGD applies residual quantization and source coding techniques to reduce the communication cost between decentralized nodes at each communication round. This work gives detailed analysis of the communication cost and convergence rate of MALCOM-PSGD in the synchronous setting and, specifically, shows with properly chosen learning rate, MALCOM-PSGD is able to achieve a convergence rate of $O(\ln t / \sqrt{t})$ with $t$ iterations. Finally, experiments on optimizing DNNs in a federated learning setting demonstrate the fast convergence rate and low communication cost of MALCOM-PSGD compared to the SOTA decentralized baseline, in both synchronous and asynchronous settings.

### Strengths
- This work considers the important and interesting problem of communication efficient decentralized optimization. While communication efficient distributed and centralized optimization is well understood, this work explores the relatively less well understood decentralized settings.

- The paper presents MALCOM-PSGD in a well-organized and easy-to-follow way.

### Weaknesses
- In Section 1 the 3rd point under “our contributions”, it is stated “Our use of compressed model updates and differential encoding allows us to reasonably assume we are creating a structure within our updates that this encoding scheme is most advantageous under.” This sentence seems a bit confusing and it is unclear why this encoding scheme the most advantageous. 

- The experiments section in the draft only considers the federated learning (FL) setting, which is essentially a distributed and centralized optimization, with a single choice of the mixing matrix W. Since the focus of the paper is in decentralized settings, it’d be better to present more results of MALCOM-PSGD under different mixing matrices W.

- Minor issue: in Eq.10, it should be $\sum_{i=1}^{n} \mathbb{E}[...] \leq \sum_{i=1}^{n} 2\mathbb{E}[...] + 2 ... $

- Minor issue: in Eq.16, it should be $\eta_k$ instead of $\eta_t$.

### Questions
- How does the mixing matrix W affect the convergence rate of MALCOM-SGD?

- Is $\omega$ in Theorem 2 the same as the one defined in Theorem 1?

- In Assumption 4, is $|| \mathbf{X}^{(t)}||$ the operator norm of $\mathbf{X}^{(t)}$ ? (unclear notation)

- Just as this work mentions in the introduction, sparsification and quantization are two major techniques for reducing communication cost in distributed optimization algorithms. If instead of quantizing the model residual (aka., line 3 in Algorithm 1) in MALCOM-PSGD, one sparsifies the model residual by applying, e.g., Rand-$k$ sparsification, can the current analysis of MALCOM-PSGD be extended to this new method?

- Is Eq.8 in Section 4 "bit rate analysis" used to compute the communication cost in the experiments?

- In Section 4 "bit rate analysis", it is stated "As the training converges, we expect a diminishing residual to be quantized in Step 3 of Alg. 1, resulting in a sparse quantized vector. " Does this imply as the training proceeds, the communication cost of the nodes at each round decreases? However, from the experiment results, e.g., plot (c) in Figure 1/2, the communication cost of MALCOM-PSGD remains the same across the communication rounds. Any comments on this?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to improve the communication efficiency of decentralized nonconvex optimization. In addition to compression that is heavily used recently, the authors also suggest to add an $\ell_1$ regularization to encourage model sparsity to help communication efficiency. The authors show that the convergence rate for consensus and objective convergence is matching the state-of-the-art. Moreover, the authors show in experiments that new method provides approximately 75% improvement in terms of communication costs.

### Strengths
The motivation of the paper, improving the communication efficiency for decentralized optimization and ML is definitely important. Moreover, the paper combines techniques from different areas including not only stochastic and decentralized optimization but also coding theory such as Elias coding and Golomb coding which is a positive. The paper does a decent job of comparing with the existing works (even though there are some unclear points about this that I touch on later). The experimental validation is helpful to show that the provided algorithm is a good candidate for achieving the main goal of the paper. The paper is also well-written and mostly polished.

### Weaknesses
Since the paper brings together many ideas, like stated in the beginning of Sect 3: "SGD, residual compression, communication and encoding, consensus aggregation and proximal optimization", it's at times difficult to distinguish in which aspects the work is different from the existing methods since sometimes both assumptions and problems are different than previous works. It is also not clear how significant the contribution is in addition to the existing techniques (which is not necessarily a reason for rejection since combination of existing tools can be interesting if the result is interesting enough, but this should be clarified much more).

For example, there is a decent comparison to Zeng & Yin (2018b) where the difference is handling the stochastic case. However, the comparison with more recent works, for example in Koloskova et al. 2021 is less clear. Is the main difference handling the $\ell_1$ regularizer? The authors mention a couple of times, for example right after Theorem 1 that "there are two key differences with Koloskova et al. 2021 for consensus bound: (1) decreasing step size instead of constant one (2) tighter upper bound in terms of C/\omega^3", with no further explanation. It is not clear why a decreasing step size is better or what the precise "tighter upper bound" is compared to previous work: what is the exact improvement in the bound? After I go look at Koloskova et al. 2021, I see that the "constant step size" in the existing paper depends on the final iterate $T$ whereas the decreasing step size in this paper depends on $t$, which I assume is the difference. Apart from not having to set $T$ in advance, what is the advantage of this? For the second part, tighter upper bound, it was less clear even after I had a quick look at Koloskova et al. (2021) (quick look since I ideally would prefer not to review other papers to be able to review one paper especially in such tight timelines as usual for ML conferences), the comparison is unclear because the bounds have different dependences. What corresponds to $C$ and $\omega$ in Koloskova et al. 2021 bound so that I can see what the improvement is?

Even more confusing is the difference on the assumptions. Rate of Koloskova et al. 2021 is on the gradient norm, which is standard for nonconvex optimization. I see that since the problem in this paper is regularized, gradient norm itself is not enough, but one can instead look at the prox-mapping norm. However, also surprisingly the rate in this paper in Theorem 2 is on the objective value, which is not standard for a nonconvex optimization problem. This points out to the difference on the assumptions, this current paper assumes coercivity, whereas Koloskova et al. 2021 does not (from my quick look again, please correct me if I am wrong). Unfortunately, there is no comment about this in the paper after Theorem 2, how come we can now get a guarantee on the objective value (a very strong guarantee for nonconvex optimization requiring very strong additional assumptions) whereas Koloskova et al. 2021 only gets gradient norm. What would we get by only assuming the standard assumptions such as Koloskova et al. 2021? Can we get a similar rate for the prox mapping?

On the same topic, I also had to have a quick look at Zeng & Yin (2018b) trying to understand the difference on assumptions. The eqs. (88), (89) that this paper uses in the middle of page 17, is from Lemma 21 in Zeng & Yin (2018b) that additionally assumes that the objective function is convex. This would explain the resulting rate in the objective value, whereas the current submission does not mention this. Can the authors explain if the estimations they use from Zeng & Yin (2018b) use convexity of the objective or not? As a result, does the current submissions use convexity (or any other additional assumptions) in some way or not?

Moreover, the paper mentioned in page 2 "our contributions" paragraph: "Our findings suggest that the gradient quantization process introduces notable aggregation errors ..... Making the conventional analytical tools on exact PGD inapplicable". Can you please clarify more? In my reading of the paper and the proofs, the analysis looks like a combination of Zeng & Yin (2018b) with Koloskova et al. (2021) and some tools from Alistarh et al. 2017 with Woldemariam et al. 2023 for encoding/decoding. If the authors claim that there are difficulties in combining these techniques, they should clearly state that. Even if there are not difficulties in combining techniques, this can also be fine if the result is strong enough. But this should be clarified by the authors.

The paper argues there is 75% communication improvement in practice, what about theory? Do the bounds predict any improvement? Moreover, what is the main sources of improvement in practice compared to choco-SGD? Is it using $\ell_1$ regularization leading to sparse solutions? If so, how to quantify this? Page 5 in paragraph "Source coding" mentions "intuitively" regularized problem produces sparse solutions, but can the authors provide a precise theoretical evidence for this? Moreover, the authors mention that the consensus aggregation  is different from Koloskova et al. 2021, since they use a scheme from Dimakis et al. 2010, is this also helping to improve the communication efficiency? These really need to be clarified.

Numerical results are a bit confusing. The paper solves the regularized problem whereas choco-SGD solves the unconstrained problem. How do the authors compare these two different methods solving different problems? Moreover, the authors say they use constant step size which is a bit disconnected from theory.

### Questions
- eq. (55) please provide a pointer to the definition of $\Phi^{(t+1)}$ as the sub gradient, for example after eq. (30). Also, after eq. (30) it calls $\Phi^{(t+1)}$ Subdifferential whereas it should be subgradient.

- Can you describe clearly Elias coding and Golomb coding used in Algorithm 2 (unfortunately many readers might be not familiar with coding theory) even if they are not essential for the purpose of the paper? Where do they come in to play in the analysis? Is it only the eq. (8) that is derived in the paper of Woldemariam et al. 2023? Also, for eq. (8) in Sect. 4 please provide a precise pointer in Woldemariam et al. 2023 where this result is proven so that a reader would know where to look. Also, please show clearly how Algorithm 2 fits within the main algorithm. In the "encoding" step of Algorithm 1, you may mention you call Algorithm 2 explicitly.

- Assumptions 3i is written in a bit confusing way, please consider writing it like $x_i \mapsto F_i(x_i)$ has $L_i$  as the Lipschitz constant for gradient so then it will be clear the sum is Lipschitz gradient  with the max of $L_i$. Please also provide more commentary about the coercivity in Assumption 3ii since it is not standard for nonconvex optimization and also different from existing works for example Koloskova et al. 2021.

- eq. (4) please explain better the difference of this scheme with Koloskova et al. 2021. Especially since the notations in the two papers are different, it is not easy for the reader to compare.

- footnote in page 4: What about theory? Does the theoretical results go through with asynchronous and time-varying network? If so, more justifications are needed.

- Second page says that Nesterov proposed proximal gradient descent whereas Nesterov in this paper points to earlier work (including a paper from 1981) for this "unaccelerated" PGD. Can you please correct the reference for proximal gradient descent?

- page.3 states "all prior analysis on convergence of quantized models rely on smoothness" as if this paper does not. But this paper also does, since all the proximal gradient methods do. They still use smoothness in a very central way and handle structural nonsmoothness with proximal operator. Better to be not misleading on this.

- It is not clear to me what is "inexact proximal gradient" referring to here. For example in the paper of Schmidt et al. 2011, inexactness is both on the gradient computation and proximal operator. Here the proximal operator seems to be exact, am I missing something? Is the inexactness due to compression and other techniques used for improving communication efficiency?

- Theorem 1: please point out to the definition of $\tau$ from Assumption 2 in Theorem 1 for improving readability.

- Right before eq. (58) the authors refer to some calculations in Koloskova et al. 2019 (eqs. (20)-(24)). Can you explicitly write these steps in the paper so that the reader will not have to go to another paper, get familiarized with their notation to come back, recall the notation of the current paper and then understand the steps?

- What can we get with the same set of assumptions as Koloskova et al. 2021 by not introducing more assumptions? This would probably be a rate on the prox-mapping.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers the non-convex decentralized learning problems with finite-sum, smooth, and non-convex loss functions with L1 regularization. The authors propose MALCOM-PSGD algorithm that strategically integrates gradient compression techniques with model sparsification. The proposed algorithm is guaranteed to converge at a sublinear rate with a diminishing stepsize. Numerical results are provided to show the advantages of the algorithm on saving of communication.

### Strengths
1. The proposed algorithm employs the residual compression via quantization and source coding methods to encode model sparsity, which can efficiently reduce the communication cost.
2. They provide a comprehensive convergence analysis of the MALCOM-PSGD algorithm, and its performance is substantiated by suitable experimental evidence.

### Weaknesses
1. The idea for model sparsification with L1 regularized training loss function and presented non-smooth problem are not surprising in distributed learning, which has been widely studied in the literature. 
2. The communication complexity is not provided; the authors is suggested to compare the communication complexity with related works.
3. The theoretical results fail to demonstrate the existence of a linear-speedup case in decentralized training.
4. Assumption 4 is directly imposed on the sequence generated by the algorithm, which is not well justified and appears to be stringent.
5. The proof techniques used in the paper are standard in decentralized learning and source code methods is not original as well. 
6. The algorithm design should be further clarified. Additionally, there is an abuse of notations, e.g. the constant $L$.

### Questions
refer to Weaknesses part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
