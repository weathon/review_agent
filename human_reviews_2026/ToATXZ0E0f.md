# Why SGD is not Brownian Motion: A New Perspective on Stochastic Dynamics

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
The conventional wisdom in deep learning theory often models Stochastic Gradient Descent (SGD) as a Brownian particle, described by a Langevin equation. In this work, we challenge this paradigm and propose a more fundamental perspective: SGD is best understood as deterministic dynamics within a fluctuating loss landscape. From first principles, we derive a master equation for the parameter evolution and its corresponding Fokker-Planck equation, which exhibits differences from the standard form used for Brownian motion. We analyze the resulting dynamics near minima, where the loss is approximately quadratic, and identify distinct behavioral regimes. The most intriguing behavior emerges in the presence of valleys. We demonstrate that in this regime, the dynamics do not converge to a stationary distribution. Instead, individual SGD trajectories diffuse along the floor of these valleys with an effective diffusion coefficient proportional to the learning rate. We empirically validate these theoretical claims through experiments on deep learning tasks in computer vision and natural language processing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Different from previous studies, this paper argues that **SGD is best understood as deterministic dynamics within a fluctuating loss landscape** and models the parameter evolution with a Fokker-Planck equation. Under a quadratic assumption near local minima, the authors found that the parameters saturate along the positively curvatured directions, while diffusing along the near-flat directions with a rate propotional to the step size. This theorectial finding is further empirically validated on different tasks including CV and NLP.

### Strengths
This paper is well-written and easy to follow. The perspective of viewing SGD as a deterministic motion across a fluctuating loss landscape is interesting. Most importantly, the authors identified two distinct regimes near the local minima that the parameter evolution varies along different directions determined by the Hessian eigenvalues.

### Weaknesses
While this paper provides some useful insights on why SGD is not Brownian motion, there remains several issues on the novelty and the correctness of theoretical contribution:

 - First, the finding that flater directions correspond to smaller fluctuations has been extensively studied before (Feng & Tu, 2021a), for example, see [1][2][3]. Particuarly, [3] showed that the variance of stochastic gradient  is proportional to the magnitude of the corresponding eigenvalues of the Hessian. So the statement **this important observation challenged prevailing intuitions...** is not extactly true and the authors should change the wording approriately.

 [1] Chaudhari et al., Entropy-sgd: Biasing gradient descent into wide valleys, 2016.

 [2] Sagun et al., Empirical analysis of the hessian of over-parametrized neural networks, 2017.

 [3] Daneshmand et al., Escaping saddles with stochastic gradients, 2018.
 
 - Second, I was wondering the intrinsic difference between the two viewpoints: a random motion along static potential and a determinstic motion along random potential. The empirical loss is averged over the training set and each step we minimize it with a stochastic gradient, which of course the randomness is incurred by the mini-batching. As far as I can tell, the former viewpoint is more intuitive and reasonable. Particularly, [4] has already showed that SGD is running on a convoluted (smoothed) version of the loss function. 

 [4] Kleinberg et al., An Alternative View: When Does SGD Escape Local Minima, 2018.

- Third, the assumptions are not verfied. Particularly, line 295 indicates the pre-dominant diagoality of $<H^2_{ij}>_{\delta L}$. But Figure 2(right) only demonstrates this for large variance index. Whether this still holds for small variance index?  I believe this is critically important for the derivation of the main result where $\lambda_i$ is close to zero. Moreover, I do not know how the constant $\epsilon$ appears in Equation (4). I notice that the authors introduce it at line 777, but how large it is when compared to $\lambda_i \approx 0$. My biggest concern is what will happen when $\lambda_i$ is much smaller than zero. We cannot preclude this occation, paticularly  when the training set is seriously unbalanced. Also, the used neural network is quite small, which may not reveal the true trend in the over-parameterized regime.

- Finally, it seems that the section of Conclusion is missing and some of the hyper-links are not properly formatted, e.g. line 197 "(3) provides..." and line 189 "Risken (1989)". Of course, some related works on modelling SGD with Fokker-Planck equation are missing as well, see [5][6].

[5] Tan et al., Understanding short-range memory effects in deep neural networks, 2023.

[6] Lucchi et al., On the Theoretical Properties of Noise Correlation in Stochastic Optimization, 2022.

### Questions
please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work investigates stochastic gradient descent (SGD) with resampling, where at every step the gradient is evaluated on a minibatch independently sampled from the training data with replacement. Its main contribution is to derive an approximation for the discrete Fokker-Planck equation for the evolution of the probability distribution of the weights in the small learning rate regime (Equation 2), capturing the leading order statistics of the SGD noise. This amounts to an interpretation of the stochastic SGD iterations as a deterministic evolution on a random potential. 

The authors then investigate the behavior of the Fokker-Planck equation around a critical point of the expected loss, deriving a closed-form expression for the covariance matrix of the weights trajectory when expressed in the basis of the averaged Hessian (Equation 4), under a few assumptions on the Hessian (L290-L298).

The main take-aways from this characterization are that the variance along directions of positive curvature grows initially and then saturate and that nearly flat directions lead to a diffusive diffusive behavior.  

Numerical experiments on Computer Vision and Natural Language processing are presented to illustrate the theoretical findings.

### Strengths
The main strength of the paper is that the setting studied is fairly general, with minimal assumptions made directly on the statistics of the loss rather than on the task. The insights drawn are interesting, and are corroborated by the numerical experiments.

### Weaknesses
The two main weaknesses of the manuscript are:

1. **Relation to previous work**. The first sentence in the abstract:
> "*The conventional wisdom in deep learning theory often models Stochastic Gradient Descent (SGD) as a Brownian particle, described by a Langevin equation. In this work, we challenge this paradigm [...]*" 

is highly exaggerated and misleading, as it suggests that the study of the structure of SGD noise is largely unexplored. In reality, there exists an extensive body of literature in machine learning, known under the umbrella of "*SGD vs. SDEs*", that addresses precisely this question, yet the manuscript does not adequately relate its contributions to these prior works --- see for instance this [blog post](https://francisbach.com/rethinking-sgd-noise/) and the list of references therein. In particular, the result that to leading order in the learning rate only the covariance of minibatch fluctuations contributes to the dynamics overlaps with results from the line of work by (Li et al. 2017). I strongly suggest the authors revise the relation to previous work in a revised version, lowering the tone on the predominance of "SGD = Langevin" in the ML literature (e.g. L11-13, L42-44, L56-58) and better emphasizing their contribution to the extensive list of work that studied this question. 

2. **The lack of polish and clarity**. The manuscript appears unpolished and gives the impression that it was submitted hastily, without adequate revision. The text would benefit from a thorough proofreading to correct the numerous typographical errors (a few examples are listed below, though the list is by no means exhaustive) and to improve the overall quality of the English (note that using LLMs for text correction is not prohibited by the [ICLR policy](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/) :) ) . Since the results strongly depend on the sampling scheme used, I suggest the authors define more explicitly SGD with replacement in the beginning of Section 3 for added clarity.

**References**:

- (Pillaud-Vivien 2022) Rethinking SGD's noise. URL: https://francisbach.com/rethinking-sgd-noise/
- (Li et al. 2017) Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms

**Typos**

- L196-197: "*This independence ensures that (3) provides an exact probabilistic description of SGD under this setting.*". What is (3) referring to here? Equation (3) only appears later in the text, and concerns the expansion of the loss around a critical point. Seems unrelated to this discussion.
- L281: "*Where $O_{ki}, O_{kl}$ is orthogonal matrix"
- L195: "*Within a such framework*"
- L280: "*Formally, eigenbasis of the*"
- L287-288 "*is almost diagonal matrix and small non-diagonal elements corresponds*"

### Questions
- Can the authors provide some intuition on what the assumptions 1-3 in L290-298 imply on the underlying problem? Do they have any concrete example where they can show these assumptions are satisfied? For example, what do they imply for a simple least-squares regression problem?

- In Figures 3 and 5, why the orders of magnitude of the y-axis in the theoretical (left) and numerical (right) plots differ? Minor note: the y-axix label in these plots seem to be cut.

### Soundness
3

### Presentation
1

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
The paper aims at improving the theoretical understanding of Stochastic Gradient Descent (SGD). It challenges the view that Stochastic Gradient Descent behaves like Brownian motion and is well-modeled by Langevin dynamics. Instead, it proposes that SGD is better understood as deterministic dynamics in a fluctuating loss landscape, with stochasticity arising from mini-batch sampling rather than external noise, leading to potentially non-gaussian gradient noise.

In particular, paper mains contributions are:
1) Derivation of a master equation and a discrete Fokker–Planck equation for SGD, highlighting differences from the standard Langevin approach.
2) Analysis Near Critical Points: Detailed study of SGD dynamics near minima, using the Hessian eigenbasis to distinguish between "rigid" (high curvature) and "diffusive" (flat) directions.
3) Experiments on vision (MNIST MLP) and language (NanoGPT on Shakespeare) tasks confirm theoretical predictions, especially the separation into diffusive and rigid modes.

### Strengths
- Paper is well motivated The limitations of the Langevin approach are clearly articulated, especially regarding noise scaling.

- The paper provides a novel and rigorous theoretical framework for understanding SGD, moving beyond the Langevin/Brownian analogy.

- It rigorously derives the Fokker–Planck equation for SGD, clarifying the role of mini-batch noise.

- The analysis in the Hessian eigenbasis is insightful, allowing for a clear separation of parameter dynamics. The connection between gradient and Hessian fluctuations is well-motivated and supported by derivations.

### Weaknesses
- The empirical validation is restricted to small models and datasets (e.g., 2 layers MLP on MNIST, NanoGPT on Tiny Shakespeare). It is unclear how well the theory scales to large, modern architectures and datasets.

- The analysis relies on several assumptions (e.g., independence of Hessian fluctuations in particular). It would be nice to expand on the trade-off and accuracy of those assumptions..

- While the theoretical insights are strong, the paper does not discuss in detail how these findings could influence practical optimization strategies.

### Questions
- How do the theoretical predictions hold for large-scale models (e.g., deep CNNs, Transformers) and real-world datasets? Are there empirical results or intuitions for such settings?

- What happens if the independence or diagonality assumptions about the Hessian are violated? Can the theory supports such as case?

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
4

### Summary
This is a paper tackling an interesting problem: insightful characterisation of SGD in terms of analysable continuous time analogues. It first describes the problem in terms of a fluctuating gradient field, considers the Fokker-plank equation, derives some perturbation analysis for the diffusion in the vicinity of a critical point, and does a number of related experiments.

### Strengths
This paper considers an important issue, and provides a direction of analysis, via a perturbation analysis around a critical point to obtain the dynamics at that critical point, and define the covariance of the diffusion dynamics around any critical point. It derives a potential useful analysis and does experiments on key models corresponding to that analysis. I commend the researchers for their work and encourage them to push it forward, but for some of the reasons below, I do not think this paper is ready for airtime just yet. 

The paper would benefit from:

A clear take-home. Why is there a benefit in doing all this analysis? Why does that help use.

A precise distinction between this analysis and one based on Lengevin diffusions, if that exists, starting with the FP and demonstrating explicitly that the analysis of SGD cannot be viewed as _any_ general form of diffusion, and that this distinction has actual impact on understanding.

Alternatively, remove the claim of a distinction, and work hard at relating the work to other diffusion analyses of SGD. Try removing invalid assumptions or make explicit that they are invalid, but insight can still be made by making them.  

Clean the paper writing up, define the problem being addressed, the insight that the authors bring to the paper and the point of the paper: why does the paper change the view of the world?

Tighten up the experiments to address the precise conclusions and do controlled ablations that make those conclusions convincing.

### Weaknesses
There are a few undefined terms on unexplained equations in parts. Explain the meaning of different indices etc.

"Nevertheless, this construction does not faithfully reproduce the actual SGD update, in which the stochastic contribution
naturally enters at order η, but not √η."

I think this is not correct: There are two parameters in SGD: learning rate and batch size, whose setting relate to one another, and between them they determine the relationship between the noise and drift terms in the corresponding Euler-Maruyama discretised diffusion. The stochasticity in SGD can then be controlled to be of whatever order one desires (see for example https://arxiv.org/abs/1711.04623 - already referenced). Furthermore the Lengevin formalism as discussed says nothing about the covariance structure of \xi, which is critical to the formalism. 

The statement that SGD is "deterministic dynamics in a fluctuating potential." is not insight but a statement of the obvious: it is what the SGD equation just says. Each new minibatch defines a new potential, related to one another as they are determined samples from the underlying full-batch. The insight for considering the Lengevin analogue comes from the recognition that the derivative of these potentials are unbiased (and independent if constructed properly) samples from some distribution, considered to be the noise distribution.

This paper is therefore suggesting that this direction is a red-herring and we should return to the basic starting viewpoint that "SGD is deterministic dynamics in a fluctuating potential" and arguably take a different tack. The question is then whether this claim is justified, or indeed whether this different tack is a different tack at all, or just the same thing written in a different way. I find this paper is just going back to the base and rederiving the original Lengevin formalism, but with the relevant covariance structure handled. However most papers in this field consider the problem of this covariance structure and there has been a long history of papers working out how best to characterise it, estimate it, or model it. 

(as an aside, please number all equations. How am I as a reviewer to communicate about equations if there are no numbers to refer to?)

In fact by considering the Fokker-Planck equation, the structure is inherently viewing this in terms of diffusion, it can end up being exactly the same as the Lengevin formalism, with the Hessian of the noise relating to the covariance structure of the noise terms. Most Lengevin analyses of learning dynamics understand this in terms of the Fokker-Planck equation. See https://proceedings.mlr.press/v97/zhu19e/zhu19e.pdf (already referenced) for example, and https://papers.nips.cc/paper_files/paper/2015/file/6f4922f45568161a8cdf4ad2299f6d23-Paper.pdf. So I am feeling that so far this is standard fare dressed up as something new, and I am not convinced this is outside of a Lengevin understanding of the problem.

Then on to the analysis of the near-critical point. This is specific to this paper. 

In the derivation the "simplifying assumptions, which are rather natural for NN under consideration. Namely we assume
that fluctuations of all components of ˜Gi and ˜Hij are independent." is not at all fair. By all means make that assumption, but considering them natural for neural networks is not accurate; given the impact of each parameter on the output is the product of all parameters along the unblocked paths from input to output, these are heavily non-independent. 

It is not really helpful to state that ˜Πnij is predominantly diagonal" when you have assumed (without validity) that H_{ij} is diagonal, given it dominates. Technically it is correct given the assumptions, but gives the wrong impression to the reader.

How does this result compare with previous related analsyses? I would want this to be set in the scene of the extensive existing work in this field. In particular, one paper of Zhanxing Zhu's group is referenced, but his other work considers many of these issues. Broader discussion would be very valuable.

I am really unsure about the take-home here. There is no conclusion, and the observations that there are rigid and diffusive directions is longstanding and a result of the overall loss surface and not of this formalism. The same results come out of previous Lengevin analysis. I fear I am a bit lost as to what conclusion I should draw from this paper.

I think writing this as "we move away from Lengevin to fluctuating gradient fields and then heading back to look at the Fokker-Planck equation is a bit of a circle, given the general formalism of Lengevin methods. Given most of the analysis is critical point analysis, there is actually no difference between a Lengevin and non-Lengevin diffusion analysis at such a point. If you really want to push this point I would start with a general diffusion formalism, the Fokker Planck and show exactly that there is no potential for which this is a Lengevin diffusion of any sort, and then demonstrate that this observation buys you something very significant. I don't see it at the moment, I am afraid.

### Questions
What conclusions do you want me to draw from this? What is the fundamental understanding I now have as a result of your paper that will change my view of the world? 

What am I missing about your argument?

### Soundness
2

### Presentation
2

### Contribution
2
