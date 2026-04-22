# Learning Where to Learn: Training Data Distribution Optimization for Scientific Machine Learning

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 2, 6, 6, 4

## Abstract
In scientific machine learning, models are routinely deployed with parameter values or boundary conditions far from those used in training. This paper studies the learning-where-to-learn problem of designing a training data distribution that minimizes average prediction error across a family of deployment regimes. A theoretical analysis shows how the training distribution shapes deployment accuracy. This motivates two adaptive algorithms based on bilevel or alternating optimization in the space of probability measures. Discretized implementations using parametric distribution classes or nonparametric particle-based gradient flows deliver optimized training distributions that outperform nonadaptive designs. Once trained, the resulting models exhibit improved sample complexity and robustness to distribution shift. This framework unlocks the potential of principled data acquisition for learning functions and solution operators of partial differential equations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission discusses training algorithms for operator learning: for some operator $G^\star$, find $\hat G$ such that
$$
\hat G = \arg\min_{G} \mathbb{E}_{x \sim \mu} \left[\|G(x)- G^\star(x)\|^2\right]
$$
The quality of predictions with $\hat G$ on test points depends on the distributions of training points $x \sim \mu$. 
The submission now discusses two algorithms for optimising those locations via optimising the average-case accuracy (in the sense that the distribution $\mu$ itself is a random variable), which leads to a bilevel optimisation problem: 
- One algorithm solves the bilevel problem under the assumption that the hypothesis functions are elements in a reproducing kernel Hilbert space; 
- The other algorithm is based on an upper bound of the average case accuracy that decomposes into the sum of in-distribution error and average deviation from the hypothesis measures, which can be optimised alternatingly (like in expectation maximisation).

The resulting algorithm is a theoretically founded way to optimise the training data distribution in operator learning, and experimental results show how this approach outperforms not optimising training data distributions.



**Summary of my recommendation:** Overall, I find the paper has distinct strengths and weaknesses, which makes it difficult to give a definitive recommendation.
On the one hand, I find the infinite-dimensional analysis compelling, but on the other hand, there are gaps in the experimental evaluation. 
And while I appreciate the technical thoroughness, the paper is significantly longer than the usual ICLR paper (47 pages), which hinders readability. Altogether, I recommend rejection for now, but I am giving a borderline score because I do appreciate the analysis. I am looking forward to reading what the other reviewers think.

### Strengths
**Infinite-dimensional analysis:** the algorithm applies to the infinite-dimensional setting (except for some implementation details, see Lines 343f, but the implementation needing discretisation is typical and not a weakness). 
When first reading the manuscript, I was surprised that optimising training distributions with average-case losses was new. I expected it to be known in statistical learning theory, but it turns out that it is only standard knowledge in finite-dimensional settings, and the paper discusses operator learning (which is infinite-dimensional).
As function-space perspectives have grown in popularity in both scientific machine learning and generative modelling recently, the generality of the method fits well into this growing body of literature.


**Correctness and clarity:** I have not found any issues with clarity (or correctness), which, since the derivation of the algorithm is relatively technical, should be considered a major strength of this submission. The solution of the bilevel optimisation problem in Section 4.1 is sound, and the approximation in Section 4.2 is theoretically well motivated.

**Related work:** Finally, I like how the paper discusses related work thoroughly (eg how the derivative in Proposition 4.9 relates to Wasserstein gradients). This is especially convincing since the fields of adaptive sampling and active learning are broad, and many different communities have to be acknowledged.

### Weaknesses
**Experimental evaluation:**
The biggest weakness of the submission, in my opinion, is the experimental evaluation:
- The bilevel optimisation problem from Section 4.1 is evaluated only on three finite-dimensional, textbook-style problems, which do not relate to operator learning. Since I read the main contribution of this submission as two algorithms that also work in an infinite-dimensional setup, both of them should be evaluated as such. I recommend benchmarking the bilevel gradient descent algorithm in an operator-learning framework. It would be okay to choose a simple problem, but the benchmark needs to include a function-space scenario.
- The related work section lists numerous algorithms for training data distribution in scientific ML, and I am surprised that none of these algorithms are used as baselines in the experiments. To embed this work better into the scientific ML literature, I recommend including at least one or two such baselines.
- The experiments use very simple PDEs (Poisson problem and transport equation, both in 2d). To make this work comparable to contemporary algorithms, I would recommend including other PDEs that are (at least slightly) more complicated, e.g. Burger's equation or the shallow-water problem; see the PDEBench paper for inspiration: https://arxiv.org/abs/2210.07182.

As is, these experimental weaknesses are the main reason why I lean towards rejection.


**Average-case vs alternatives:**
The paper focuses on average-case expected accuracies, and mentions that other measures would be possible, too (eg worst case; Line 174). Since this choice of average vs worst-case matters for performance, I would recommend justifying this choice somewhere close to Equation 3.3. 




**Length:** The submission is 47 pages long. While I understand that it is common practice to defer proofs and additional experiments to appendices, it is problematic if the main paper relies too strongly on these additional results. And I think with the present submission, this is the case:
- The related work section is essentially outsourced to the appendix. The main paper discusses approaches from numerical analysis and scientific machine learning, but active and meta learning, domain generalisation, and experimental design are entirely outsourced to Appendix A. All of these are important connections, and in my opinion, necessary to appreciate the main results of the submissions. 
- Many preliminaries are in Appendix B, not in Section 2.
- Crucial examples are only listed in the appendix (eg the one for Corollary 3.2)
- Appendix D contains so much more detail than Section 3 that it is almost required reading.


In total, about 80% of the results are in appendices (in terms of volume), and I find this a bit excessive. It's not something that can be changed easily,  but I wonder whether a longer-form paper would have been a more appropriate medium for presenting the results.

### Questions
The following points do not really affect my score, but I find them worth mentioning regardless:

- There is a similar paper by Subedi and Tewari (https://arxiv.org/abs/2504.03503). Since the preprint by Subedi and Tewari was uploaded to arXiv earlier this year, I think it might have been developed in parallel with the submission, so its existence shouldn't affect the acceptance/rejection of the present work. Still, a footnote or so in the submitted manuscript might help the reader connect the articles.
- Line 411f: In the explanation of the numerical examples, a diagram mapping out the Neumann-to-Dirichlet and the parameter-to-solution setups (as in, what is the input and what is the output) would make the descriptions easier to parse. If space permits, I would find that helpful. (If necessary, drop Algorithm 1 from the main paper, because I would expect that gradient descent can be assumed as known at a venue like ICLR.)

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies OOD generalization motivated by engineering application. This is done by trying to find an optimal training distribution $\nu$ that minimizes the error across a family of deployment distributions. The paper describes the framework in a rather dense manner. What I took away from it was that you first need an upper bound on the average OOD error, then link it to the 2-Wasserstein distance between train and deployment distributions. This bound will give one important component that will be used. After this, a bilevel algorithm is described. Extensive analysis/results are include but then demonstrated for kernel methods. A separate simpler algorithm minimizes the upper bound and used for the experimental results.

### Strengths
1. The paper at a high level is tackling a real problem. Distribution shift is a genuine challenge and the motivating example shown does show that performance can be improved. So the formulation of optimizing training distribution to minimize average error across deployment distributions is practically relevant and makes sense.

2. For a theoretically focused paper, the paper does test the approach across a few different settings. This includes EIT inverse etc and parametric/nonparametric. I feel that this is more than adequate. Also the alternating minimization algorithm is fine. I don't know if something similar exists in the OOD literature (not very familiar with that body of work) but it gives monotone decrease guarantees and can be used. A gradient flow option is also provided.

### Weaknesses
There are a number of issues I have with the paper and the presentation. 

1. I feel that the technical novelty is quite misrepresented. The paper uses a full basket of classical results (KR duality, adjoint state, envelope theorem) and does minimal effort to clarify which results/propositions should be viewed as contributions unique to this paper. Wouldn't (3.1) follow from existing results for Wasserstein stability or KR duality? The paper never clearly states: "This known results we will use" versus "Here is our novel contribution and this is why its interesting" This makes it exhausting to read the paper because you're wondering what's new. I will acknowledge that these techniques are being applied to operator learning setting.

 2. The theory needs Lipschitz continuous maps with known Lipschitz constants. In practice, one uses neural networks and estimate Lipschitz constants via sampling. Won't these constants be enormous for deep networks making the bounds a bit meaningless. Alg 2 minimizes an upper bound but is this bound tight or how useful it is. Does minimizing the bound correlate with actual OOD error? The entire theoretical framing rests on Lipschitz assumptions that are probably unrealistic. I think this is fine to some extent but the paper really should identify one or more clear insights from the theory and acknowledge what is going on. 

3. The optimized distributions sometimes help, but often only marginally. To my reading, the optimized Gaussian often performs similarly to simple baselines (barycenter, mixture)? Sometimes, a single test distribution with sufficient samples and training on the distribution itself is good. So this is confusing because there is no clear guidance on when optimization is worth the cost versus using simple heuristics.

4. It is also not clear whether the extensive and carefully developed RKHS theory apply to any of the paper's main experiments. The alternating algorithm works more generally, but then why develop the inapplicable RKHS machinery at such length? This disconnect between theory and experiments is never explained.

5. Finally, the presentation is excessively dense and also not well structured. The paper throws everything at us. There is no incremental buildup. A simple 2D function approximation example showing the core idea would be far more valuable than pages and pages of abstract machinery with no central theorem highlighted and other results feeding into it. The results appear as Proposition 3.1 etc with no narrative  explaining which are setup, which are contributions, and how they will connect. 

6.  The bilevel gradient descent needs computing adjoint states etc, then running iterative optimization. How many function evaluations of the ground truth are needed? What is total runtime analysis? How should we understand comparison of compute cost versus accuracy trade-offs? For expensive PDE solvers, is this optimization worth it or should one focus on more training samples from a simple distribution?

### Questions
please see weaknesses above. Answer as many as you can.

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
The paper studies the choice of training data distribution; to determines where a model learns from which is usually fixed and heuristic. The authors propose to optimize the training distribution itself, so that the resulting model generalizes better across anticipated deployment regimes that may differ from the training distribution. The paper proposes Lipschitz-based generalization bounds linking OOD error to training error and the Wasserstein distance between training and test distributions. Two adaptive practical algorithms are proposed: a bilevel procedure that directly minimizes OOD error and a computationally efficient alternating scheme that minimizes an upper bound on the OOD error. Empirical results on involving function approximation and forward and backward operator learning show that the proposed algorithms reduce OOD error, and often outperform nonadaptive baselines.

### Strengths
1. The paper has a good introduction to preliminaries.
2. The related work is discussed covering a broader perspective.
3. I like how the paper unifies adaptive sampling in PINNs, domain generalization, and optimal experimental design within a coherent mathematical framework.
4. The bounds relating OOD error, Lipschitz continuity, and Wasserstein distance are sound and help in interpreting the generalization guarantees.
5. The proposed algorithms are demonstrated both using parametric distributions as well as using non-parametric particle based gradient flows.

### Weaknesses
1. My main concern is regarding, the experimental evaluation which includes only a small number of baseline settings and lacks systematic ablations across models and problem types. As a result, it is difficult to assess how broadly the proposed training distribution optimization outperforms standard or heuristic designs.
   a) How do author(s) decide the choice of baseline problems?
   b) Can author(s) include comparison with some adaptive sampling method(s)? For example: the several methods that iteratively refine sampling locations (e.g., residual-based PINN adaptation, active learning). Some of these are mentioned in the related work(s).
   c) Can other example(s) be included to help assess the robustness and general applicability of the proposed framework?
   
2. The computational complexity of the proposed algorithm is not discussed extensively in the numerical experiments. Usually, the incorporation of bilevel optimization increases computation complexity. Can author(s) comment on the increase in wall-clock time of the proposed framework over heuristic (fixed) and adaptive baseline choice(s)?

3. The gradient estimation used for the bilevel optimization associated with both algorithm 1 and 2 is not very clear. Can the author(s) make this clear in the main paper? It will be good to have a discussion on what are the different choices that are available and how this can impact the convergence properties and performance in the numerical experiments.

4. How sensitive is the proposed optimization framework to inaccuracies or misspecification in the deployment prior Q?

### Questions
Please see weakness section and associated question.

### Soundness
3

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
2

### Summary
This paper presents a new and principled framework for optimizing training data distributions to enhance the out-of-distribution (OOD) generalization of models in scientific ML. The authors offer a theoretical analysis that links OOD error to the Wasserstein distance between training and test distributions. Building on this theory, they develop two algorithms: a bilevel optimization scheme and an alternating minimization scheme aimed at identifying optimal training distributions. The effectiveness of the proposed methods is demonstrated through extensive experiments.

### Strengths
1. The paper addresses the critical challenge of distribution shift in operator learning.

2. The techniques are motivated through fundamental theoretical analysis

3. The authors conduct a thorough empirical study

### Weaknesses
## Appropriate Baselines
I do not see a comparison against any baselines[1-5]. I am not sure how to evaluate the proposed approach. 

## Scaling the Approach
The experiments use architectures that, while suitable for demonstrating the theory, are not the current state-of-the-art (Fourier Neural Operator or transformer-based neural operators). 




1. MetaPhysiCa: OOD Robustness in Physics-informed Machine Learning
2. Open‑Sampling: Exploring Out‑of‑Distribution data for Re‑balancing Long‑tailed datasets
3. Active learning for neural PDE solvers 
4. Variational Bayesian Optimal Experimental Design
5. Learning to Sample for Discovering Governing Equations

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose to design an algorithm that identifies a training data distribution that generalizes well to test distributions, possibly in out-of-distribution settings. The authors frame this as a "learning-where-to-learn" problem, derive theoretical bounds for OOD error, and propose two practical algorithms that solve a bilevel optimization problem (for the kernel setup) and an alternating minimization problem (for the neural net setup).

Experiments are conducted over SciML applications (e.g. PDE operator learning problems such as EIT, Darcy Flow, and radiative transport), in which the authors show that their proposed approach can improve OOD generalization.

### Strengths
- The presentation of mathematical formalisms and algorithms are clear. 
- The paper aims to address a significant problem in SciML, namely improving performance of ML models under the the presence of train-test distribution mismatch. The authors' focus on operator learning is especially timely.
- I'm not particularly well-read on the neural operator literature, but I find the authors' approach to solving the bilevel optimization formulation is appropriate for the context, and is different from prior efforts in improving generalization performance in unseen boundary conditions (e.g. [1])
- On a related note, I wasn't able to check the proofs in detail due to limited reviewing time, but I'm optimistic that they appear to be correct.
- The authors have gone great lengths in detailing their design choices and motivations in the supplementary material.


[1] https://arxiv.org/abs/2504.19496

### Weaknesses
- The organization / presentation of key contributions of the paper could be improved. In my view, the core contribution of the paper is a tractable learning-where to learn algorithm, but the authors simultaneously introduce a variant of the popular NIO architecture (AMINO) in the appendix and show it demonstrate better OOD performance. Then, the key results are conducted over a combination of the key algorithms and AMINO, which can be confusing and in my opinion, diminish the contribution of the algorithm.
- The improved performance of the algorithm comes at the cost of more function evaluations (e.g. Figure 2, right), which can be expensive to obtain in certain scientific tasks.
- A minor point: while the theoretical analysis of the exact bi-level optimization problem (4.1) is nice, it is only implemented for simpler models such as kernel ridge regression. As the authors stated in D.6, their approach would suffer from vanishing gradient if the inner-loop model overfits (i.e. in the interpolation regime), which is often the case with neural nets. This limitation makes Algorithm 1 feels more like a warm up to Algorithm 2 rather than a standalone contribution.

### Questions
- My core concern is the unclear attribution of performance improvements to the proposed algorithms and AMINO. Would it be possible for the authors to ablate OOD performances on:
    - AMINO only with standard training
    - Algorithm 2 with NIO

This would greatly help me understand your contributions. I'm happy to read other reviews and go over additional experiments to adjust my score.

### Soundness
3

### Presentation
2

### Contribution
2
