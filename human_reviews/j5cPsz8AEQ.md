# Physics-informed neural networks with unknown measurement noise

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5, 3

## Abstract
Physics-informed neural networks (PINNs) constitute a flexible approach to both finding solutions and identifying parameters of partial differential equations. Most works on the topic assume noiseless data, or data contaminated by weak Gaussian noise. We show that the standard PINN framework breaks down in case of non-Gaussian noise. We give a way of resolving this fundamental issue and we propose to jointly train an energy-based model (EBM) to learn the correct noise distribution. We illustrate the improved performance of our approach using multiple examples

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors introduce a new training procedure for PINNs which are adapted to unknown measurement noise, i.e., a training procedure which works for any noise model. This is done via EBMs, which are trained jointly with the PINN. Here the EBMs estimate a 1d noise model based on the estimation of the PINN (conditional to the point $t_i$). Since they only estimate a 1d distribution, the (usually intractable) normalization constant can be estimated via numerical integration. The approach is tested on several (partial) differential equations and benchmarked against the standard PINN.

### Strengths
The paper is easy to follow (except a few minor points). The idea is interesting and well-executed. The approach outperforms the standard PINN and offset PINN baseline. The experiments are well described, so that I think reproduction should be easy.

### Weaknesses
1) While the idea is heuristically clear, it would be interesting whether one can obtain theoretical guarantees. I have got the hunch that it should be possible to cast the framework into one of expectation maximization (EM) algorithms (maybe one slightly needs to change the loss and train alternating instead of jointly). Did the authors give this some thought? This would greatly strengthen the paper in my opinion. For this see e.g. [1]

2) The discussion in 4.1 and 4.2 is a bit confusing. While I think I got the gist of it, please make clear what variables the functions $\mu_{\varepsilon}$ and $\theta_0$ depend on. 

3) The metric logL is not clearly defined. How is that calculated in the case of a standard PINN, just Gaussian likelihood?

4) The non-Gaussian noise is a GMM. I would like to see physically more realistic noise models. One thing that could be interesting is whether this approach is able to learn mixed Gaussian noise, i.e., $y = f(t) + \eta_1 + f(t)\ \eta_2$ for normal $\eta_1,\eta_2$ with some variances. While this is still Gaussian, this is a noise model used in practice. 

5) Please make the relation to model errors [2] and [3] more clear. Although the model error framework tries to solve a different problem (Bayesian inversion) the ideas are somewhat similar.

6) A very similar is to train a surrogate on the data only (no PINN loss), then estimate the noise via an appropriate model, such as an EBM and then to train the surrogate on a combined loss. Please comment on this. 

[1] DeepGEM: Generalized Expectation-Maximization for Blind Inversion, Gao et al

[2] Iterative Updating of Model Error for Bayesian Inversion, Calvetti et al

[3] Noise-aware physics-informed machine learning
for robust PDE discovery, Thanasutives et al

### Questions
See weaknesses. I overall like the idea and think it has a lot of merit. A consideration of more realistic noise models and some theoretical guarantees would strenghten the article imo.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This article proposes a method for training physics informed neural networks (PINNs) when the distribution of measurement noise is unknown. The key idea is to learn noise distribution using an energy-based model on top of training of PINNs. A few numerical experiments show the usefulness of the proposed method.

### Strengths
The usefulness of the method is shown by numerical experiments for a few example problems.

### Weaknesses
There is little theoretical backing. Extension to high-dimensional and/or non-iid noises would require much heavier computation. Experiments are limited only to synthetic problems.

### Questions
Are there any practical problems that could be resolved by the proposed method?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes the integration of an energy-based model (EBM) to learn the distribution of the noise that is added to samples in a dataset to be modeled by a physics-informed neural network (PINN). A joint loss function is used to train the EBM and the PINN, whereas the EBM can be trained at the same time or with a delayed start with respect to the PINN. Numerical experiments use synthetic data governed by several well-known PDEs from physics, polluted with a variety of noise distributions, to test the performance of the proposed approach.

### Strengths
The approach is principled, the description is clear, the results are convincing.

### Weaknesses
The proposed approach integrates two well-known models from the literature; the approach is straightforward and the results are not surprising. EBMs have been used before in classification, generative modeling, and regression problems; the authors state that the novelty is in the leveraging of physical knowledge within PINNs. In addition, all the results are focused on synthetic data. Thus the impact of the proposed approach appears limited to the current combination of tools for the usual applications of PINNs.

Minor comments:

In Algorithm 1, within the training loop, i should be updated.

### Questions
To better evaluate the impact of the proposed approach, it would be good to discuss the following questions:

(1) How is the formulation of the proposed approach different from the integration of EBM to a regular neural network?

(2) Is there real-world data that would usually be modeled by a PINN where non-Gaussian additive noise is present and for which the proposed approach can be shown to provide better solutions than the baseline PINN?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
the paper propose a method to handle measurement noise that has non-zero bias (Eq.7) and algorithm 1. the paper is indeed very hard to read. I would suggest the authors rewrite the paper to allow readers to understand and therefore use this paper for the progress of science. then resubmit the paper in the next conference. I will explain why the paper is hard to read in the next section.

### Strengths
A learning method to handle more sophisticated measurement noise.

### Weaknesses
I try to help the authors by explaining why the paper is hard to read to me. I hope these feedback can help improve the writing for a future paper.

1. math symbols are not defined when they are first used. examples:

1a. page3, line 3, D_d = {d_d, y_d}. these symbols are not explained and define. y_d was explained only towards end of page 3.

1b. page3, line 3, what is "d"? is this the index of the data point? furthermore D_d is just a set with two elements. how to learn from a set of two elements?

1c. what is the math object of y_d? is it \mathbb{R}^m or \mathbb{R}? t_d \in \mathbb{R}? what is \lambda and what dimension is it?

1d. Eq2. t_c, how to get the colocation points?

1e. algorithm 1, "if i<i_ebm then", what is i?

2. page3 second paragraph. I read this paragraph many times, I still cannot understand it. this paragraph needs to be expanded and writing needs to be clear.

overall the math formulation needs to be improved a lot.

assessment on the results and experiment section becomes invalid if the methods section of the paper is not clear and people cannot reproduce this work.

### Questions
see above 'weakness' section.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
