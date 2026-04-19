# Diffusion & Adversarial Schrödinger Bridges via Iterative Proportional Markovian Fitting

- Decision: Reject
- Scores: 3, 6, 6, 6

## Abstract
The Iterative Markovian Fitting (IMF) procedure based on iterative reciprocal and Markovian projections has recently been proposed as a powerful method for solving the Schrödinger Bridge problem. However, it has been observed that for the practical implementation of this procedure, it is crucial to alternate between fitting a forward and backward time diffusion at each iteration. Such implementation is thought to be a practical heuristic, which is required to stabilize training and obtain good results in applications such as unpaired domain translation. In our work, we show that this heuristic closely connects with the pioneer approaches for the Schrödinger Bridge based on the Iterative Proportional Fitting (IPF) procedure. Namely, we find that the practical implementation of IMF is, in fact, a combination of IMF and IPF procedures, and we call this combination the Iterative Proportional Markovian Fitting (IPMF) procedure. We show both theoretically and practically that this combined IPMF procedure can converge under more general settings, thus, showing that the IPMF procedure opens a door towards developing a unified framework for solving Schrödinger Bridge problems.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this paper, the authors propose a reinterpretation of the Iterative Markovian Fitting (IMF). Iterative Markovian Fitting was proposed in [1, 2]. This scheme performs Markovian and reciprocal projections in order to find the Entropic Optimal Transport (EOT). A much older scheme to find EOT is Iterative Proportional Fitting (IPF) [3, 4]. The authors propose to reinterpret a popular implementation of IMF with forward/backward projections to link this specific IMF implementation with the IPF algorithm. The procedure is called IPMF (Iterative Proportional Markovian Fitting). They show that the proposed procedure converges for one-dimensional Gaussian targets. They then illustrate experimentally the stability of the forward/backward implementation of IMF, i.e. IPMF, for several starting couplings including some that do not satisfy the requirements of IMF and IPF. The experiments are conducted on high dimensional Gaussian, on Colored MNIST and on CelebA. 

[1] Shi et al. – Diffusion Schrödinger Bridge Matching

[2] Peluchetti – Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling

[3] Kullback – On information and sufficiency, Annals of Mathematics and Statistics

[4] Sinkhorn – A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices

### Strengths
* I think the paper is nicely written. There is proper acknowledgement of prior work and the authors have carefully introduced complex notions such as Iterative Proportional Fitting, Iterative Markovian Fitting, Schrodinger Bridges and Entropic Optimal Transport. 

* The authors are also discussing an interesting (and often overlooked) point in the implementation of EOT based samplers which is the initial coupling. While I think the current paper still suffers from major issues, there is value in this study as it puts the emphasis on 1) how stable the IMF is with respect to the original coupling 2) the interaction between IMF and IPF

### Weaknesses
* I think the paper has limited novelty. Indeed, the link between the IMF procedure and the IPF was already identified in  [1]. In particular, in [1] the following coupling “We now relate Algorithm 1 to the classical IPF and practical algorithms such as DSB (De Bortoli et al., 2021). Instead of initializing DSBM with $\Pi_{0,T}^0$ given by a coupling between $\pi_0, \pi_1$ , if we initialize it by $\Pi_{0,T}^0 = \mathbb{Q}_{0,T}$ [...] then DSBM also recovers the IPF iterates used in DSB.” Hence the connection between IMF and IPF through the initial coupling was already acknowledged in [1], see Proposition 10 for an explicit statement. In that regard, the IPMF rewriting is not needed to show that the IMF procedure reverts to the IPF one when the coupling is initialized as the IPF one which is one of the main points of the paper. 

* Overall I think the paper has limited impact. If it should be read as a theory paper then the result is quite weak (one dimensional Gaussian random variables) and there is not so much insight that is gained from the proof of the theorem, i.e. it seems extremely unclear that the proof generalizes to other settings and what would be a path to reach such general convergence results. If the paper is to be seen as a methodology paper then I think that there should be a clear motivation to choose couplings that are different from the ones of the IMF (i.e. couplings with correct marginals) or the ones from the IPF (couplings with first right marginal and output of the Brownian motion or related stochastic process). So far, I do not see in the paper a clear use case of such couplings. Hence I find the paper poorly motivated.

* Finally, I find the experiments to be quite poor. The only settings that are run are high dimensional Gaussians, 2d examples and unpaired image translation with colored MNIST and Celeba. I found the quantitative study of the last two experiments to be quite poor. For instance, FID scores are presented in Figure 5 and MSE are presented in Appendix C.2 and in that case it seems that changing the coupling yields worse results regarding the quality of the obtained coupling. I think, the authors should investigate other couplings (I know that the minibatch OT one was investigated in the case of Celeba) in order to show that there is something to be gained from the IPMF perspective.

### Questions
* Typo in line 59 “Mathcing” should be “Matching”

* Typo in line 374 – “staring” should be “starting”

* In the introduction the authors mention that the reverse Kullback-Leibler between the iterates of the IMF and the target EOT coupling decreases at each step of the IMF. However, I could not find such a result in [1, 2]. To the best of my knowledge, the theoretical study of the IMF is quite limited. 

* In Equation (9) and Equation (10) the forward and backward are done successively. Do you think it is possible to learn the backward and forward processes jointly? Is there any work investigating that direction? In that case, it would be interesting to understand the IPMF in that setting.

* It would be good to have some kind of diagram to illustrate the differences between IMF, IPF and IPMF as well as some explicit descriptions on when they are equal. Basically, the choice of the initial coupling matters a lot. If the initial coupling in the IPMF has the right marginals (and is in the reciprocal class) then we recover the IMF. If the initial coupling in the IPMF is Markov in the reciprocal class and has the right initial marginal then we recover the IPF. 

* What other starting coupling could be explored? This is related to my comment regarding the lack of applications in the “Weaknesses” section. 

* I want to point out that in Section 3.1. The rewriting of the IPMF is only meaningful if we start from a coupling that does not preserve the marginals, otherwise since the Markovian projection preserves the marginal there is no point applying the projection on the last marginal (or the first marginal). I think it is necessary to specify this. 

* It would be interesting to investigate what is the effect of approximating the projection in the convergence of the IMF, the IPF and the IPMF, similarly as in [3].

[1] Shi et al. – Diffusion Schrödinger Bridge Matching

[2] Peluchetti – Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling

[3] Chen et al. – Provably Convergent Schr ̈odinger Bridge with Applications to Probabilistic
Time Series Imputation

### Soundness
2

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
3

### Summary
This paper shows that the popular bidirectional IMF algorithm uses IPF implicitly and they provided a theoretical result proving that for 1d Gaussians, this algorithm IMPF can converges.

### Strengths
This paper is well written and shows a novel interesting results on connecting the IMF and IPF with theoretical results for 1d gaussian cases and empirical results for higher dimensional datasets.

### Weaknesses
Although this paper shows an interesting theoretical results, it's only for 1d Gaussian cases. I feel like the contribution is relatively weak. If authors can say something more, i.e, for multi-variate Gaussian, this paper will become stronger.

### Questions
1. What's the technical difficulties for generalizing the proof to multi-variate Gaussians or other situations?
2. Can you provide more intuition on why IMPF can usually work for any specific starting process?

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
3

### Summary
The paper targets the solution of the Schrodinger Bridge problem. It first revisits two standard methods to solve the problem: Iterative Markovian Fitting (IMF) and Iterative Proportional Fitting (IPF). Then, the authors identify a connection between both methods and the associated heuristics of their implementation. This leads them to propose the Iterative Proportional Markovian Fitting (IPMF). The authors present a theoretical formulation and analysis of IPMF, to then provide experimental validation over synthetic and real-world image datasets.

### Strengths
The paper is clear, and in the first pages it adopts a tutorial-like style. The presentation of the relevant background (IPF & IMF) is properly achieved and the proposal is delivered in context. The authors also prove a particular case of their proposal (Thm 3.2).

Experimentally, the paper includes empirical validation on synthetic (Gaussians of different dimensions) and image datasets (Celeba, MNIST). The experiments are complemented by both quantitative and qualitative performance evaluations.

### Weaknesses
There are some inaccuracies in the text (Appenix, markovian, Matkovian, etc). 

Perhaps due to the lack of familiarity of this reviewer with the SOTA on SB: the main contribution of the paper is the identification that IPF & IMF can be understood as the same procedure. However, what are the practical implications of this? Is this a a purely conceptual statement, or does this fact help practical applications of SB? Perhaps the authors can be more specific about the practical advantages of their proposal

### Questions
please see above

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
In the context of Schrödinger Bridge (SB) problems, the paper analyses theoretically some heuristic algorithms typically used in practice in the literature. Specifically, the authors give a new interpretation to these heuristics, as a combination of well-known procedures used in the context of theoretical analysis of SB problems. The authors name the resulting procedure "Iterative Proportional Markov Fitting" (IPMF).

The main result of the paper is to show the converge of IPMF when both the distributions $p_0$ and $p_1$ are unidimensional gaussians.
Furthermore, the authors study heuristically the convergence of IPMF in more interesting settings, by performing numerical experiments on some typical datasets and benchmarks.

### Strengths
- The paper is very well written.
- The problem in consideration and the background are introduced very clearly, so that the idea of the paper is easy to comprehend. 
- The proof for 1D gaussian, although limited, could lead to generalisations that improve on the results that are available in the literature at the moment.

### Weaknesses
- Section 4, presenting the experiments, seems to be a bit disconnected from the previous sections. While at the beginning the authors say the aim of the section is to show the convergence of the different procedures in more generic settings, the focus shifts on the comparison between the performance of different procedures. Keeping the focus on the convergence of the procedures to the target could enhance the readability of the section.

[Minor] 
Typos:
- Line 172: Matkovian
- Line 374: staring
- Line 412: Additionaly
- Line 417: Scrödinger
- Line 528: Appenix

A careful reading of the paper is suggested to remove additional typos.

### Questions
- In section 4.3, the results presented in Table 1 are difficult to interpret. What's the role of the "volatility" parameter $\epsilon$ and why it influences so much the performances of the different procedures?

### Soundness
3

### Presentation
3

### Contribution
2
