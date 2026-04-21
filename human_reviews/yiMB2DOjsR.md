# Chain of Log-Concave Markov Chains

- Avg Score: 7.00
- Decision: Accept (poster)
- Scores: 8, 6, 8, 6

## Abstract
We introduce a theoretical framework for sampling from unnormalized densities based on a smoothing scheme that uses an isotropic Gaussian kernel with a single fixed noise scale. We prove one can decompose sampling from a density (minimal assumptions made on the density) into a sequence of sampling from log-concave conditional densities via accumulation of noisy measurements with equal noise levels. Our construction is unique in that it keeps track of a history of samples, making it non-Markovian as a whole, but it is lightweight algorithmically as the history only shows up in the form of a running empirical mean of samples. Our sampling algorithm generalizes walk-jump sampling (Saremi & Hyvärinen, 2019). The "walk" phase becomes a (non-Markovian) chain of (log-concave) Markov chains. The "jump" from the accumulated measurements is obtained by empirical Bayes. We study our sampling algorithm quantitatively using the 2-Wasserstein metric and compare it with various Langevin MCMC algorithms. We also report a remarkable capacity of our algorithm to "tunnel" between modes of a distribution.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a general framework for sampling  unnormalized probability densities. The main idea is based on smoothing the distribution
by adding Gaussian noise to the original sample. The process then repeats $m$ times,
each time independently adding gaussian noise to the original sample.
The final step involves computing the Bayes optimal predictor of the original sample based on the i.i.d. samples from the smoothed density.

The authors first demonstrate via an example that sampling all the smoothed observations jointly at once might not be optimal, since the 
sampling problem might become ill-conditioned as $m$ increases, which prohibits the use of standard MCMC methods. Subsequently, they demonstrate
that if instead sequential sampling is used, where each sample is taken conditional on the previous samples, then the condition number
of the sampling problem can only improve as $m$ increases, enabling the use of MCMC methods. They prove that this process succeeds to sample
in a general setting where the initial distribution is supported on a bounded subset of $\mathbb{R}^d$. Additionally, they evaluate their
approach experimentally, by sampling from elliptical gaussians (which are ill-conditioned) and mixtures of gaussians. 
The performance is then compared to that of standard Langevin MCMC algorithms without smoothing, showing significant benefits both
in terms of distributional convergence, as well as finding less dominant modes of the distribution.

### Strengths
This is a very original work, introducing an elegant and intuitive general framework for sampling from unnormalized probability densities. The idea of smoothing 
and walk-jump sampling has already appeared in prior work (Saremi & Hyvarinen, 2019), but this is the first time it has been applied to
the general problem of sampling from non-log concave densities. I like the intuition given by the authors about how smoothing helps by "filling in"
probability mass between modes of the original distribution, thus reducing the general problem to the log-concave setting.
One strength of this new approach is that the sequential sampling can simply be implemented by keeping a running average of all the previous
samples up until this point, so the memory footprint is small. 
Moreover, this paper is written with great clarity, as all theoretical results are explained in detail and they convincingly demonstrate the power of the new approach.
The experiments also complement the theory very nicely, by showing the benefit of running Langevin with smoothing. The tunneling phenomena that
are observed from the simulations are also very interesting. Lastly, most of the proofs follow from standard techniques, but they are at an
adequate level for ICLR.

### Weaknesses
One thing that is missing from this paper is the final step of computing the bayes estimator $\mathbb{E}[X|y_{1:m}]$ to obtain samples from the true 
distribution.
For that, it is needed to estimate the score function $\nabla \log p(y_{1:m})$ of the smoothed distribution. The authors acknowledge that this is 
a separate issue that needs to be addressed. I understand this might be beyond the scope of the current work, which is more focused on 
showing the benefits of smoothing. However, ultimately estimating the score function is a crucial step for this to be an end-to-end approach
for sampling and it is not clear what is the difficulty of this problem compared to the original sampling problem (is it strictly easier in some sense?).
In my opinion, this would make clear what the benefit of this approach over standard MCMC algorithms is.

### Questions
-Related to a previous point about estimating the score function, how exactly do the authors arrive at the expression (4.8) for estimating the
score function? In appendix B, a number of properties is proven about the score function, among which that it is related to $\mathbb{E}[X|y_{1:m}]$.
However, I don't see how this is captured in the empirical definition of (4.8). What is the meaning of taking a weighted average of
the $\epsilon_i$s?

-As the authors discuss in Remark 2, the convergence of the condition number to $1$ with rate $1/m$ holds as long as the original distribution involves 
some (even small) additive gaussian noise. Why is it enough that even a tiny bit of Gaussian noise be added to the distribution for this decay to happen?
Do the authors have some intuition about that statement?

-Related to the previous question, if we care about the expected hessian of the conditional sampling, I'm wondering whether it's possible to have a version of Theorem 1 showing that the expected condition number of the conditional sampling problem
converges to $1$ with rate $1/m$ without assuming that the original distribution contains some amount of Gaussian noise. The reason is
the following: the proof for all the claims about the "increase" in log-concavity basically hinge on equation (D.2), which asserts that
$\nabla^2 \log p(y_{1:m}) = -\sigma^{-2} I + \sigma^{-4} Cov(X|y_{1:m})$. The argument then proceeds by the observation that conditioning on more and
more observations can only reduce the variance of $X$ in expectation, so the "positive" part on the right hand side can only become smaller as
$m$ increases. The proof of Theorem 1 expresses $Cov(X|y_{1:m})$ as $O(1/ m) + O(1/m^2) Cov(Z|y_{1:m})$ 
and bounds the latter covariance by a constant, since $Z$ is supported on a bounded set (as the footnote in page 18 suggests). 
However, it seems conceivable to me that if we do not impose the assumption s $X = Z + N_0$, we could instead
show directly that $\mathbb{E}[Cov(X|y_{1:m})]$ decreases as $O(1/m)$ if $X$ is bounded. The reason I believe this is true is that, by Proposition 2, we know that the posterior
of $X$ conditioned on $y_{1:m}$ converges in distribution to $X$ at a rate of $1/m$. Thus, the conditional variance of $X$ given $y_{1:m}$ 
should decrease as $m$ becomes larger. Another intuitive way to see this is that we could for example simply take the average 
$\overline{y_m}$, which is $O(\sigma/m)$ close to $X$. Thus, we gain more and more information about $X$ as $m$ increases, which suggests
that $\mathbb{E}[Cov(X|y_{1:m})] = O(1/m)$ always. Have the authors considered this more general statement?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces an MCMC-type method that sequentially samples an unnormalized density with Gaussian smoothing. The critical theoretical contribution shows that with a proper choice of noise level and the sequential strategy, the conditional densities with compact support (or similar ones) become more and more log-concave "healthily." Through the process, the algorithm mainly keeps the means, given the properties of the Gaussian kernels. Internally, the algorithm employs an MCMC strategy (like Langevin) and methods for score function estimation.

### Strengths
The submission is a well-written research paper. The authors explained the concepts clearly and presented the critical theoretical contribution elegantly. The authors have done an excellent job of demonstrating the effectiveness of the proposed method and its potential applications. 

The problem itself is also fundamental and worth studying. The authors start by finding that "all (measurements) at once" (AAO) is suboptimal and then present a "one (measurements) at once" (OAT) algorithm that has solid theoretical guarantees. In particular, Gaussian smoothing helps transform a (nearly) compact density towards log-concavity so that it's feasible to sample from it, and measurements accumulation (in a non-Markovian manner) makes the density more and more log-concave and well-conditioned. Combining both effects makes the proposed OAT strategy superior to AAO ones.

Moreover, the paper demonstrates the value of combining theory and experiments in a research project. The theoretical analysis provides a solid foundation, while the experiments validate the proposed method and help readers gain more insights and understand its accuracy. Combining theory and experiments leads to a more complete understanding of the OAT strategies and opens up new avenues for further research.

### Weaknesses
One weakness is in the result presentation: The paper does not provide a direct performance guarantee on Algorithm 1. I understand the inner-loop method is the object of study here. Still, it should be feasible to make some assumptions around its properties and derive a theorem that captures the algorithm's performance as $m$ and $n_t$ increase. The error metric could ideally be a Wasserstain-type distance for consistency with the experiment section. 

The experiment evaluation partially supports and reflects the theoretical claims, but the settings could be more complex for a thorough evaluation. For example, it would be interesting to understand what magnitude the # sampling iterations must be for some rather complex distributions, e.g., $k$-mixture of Gaussians where $k$ increases. The densities used in Section 5 are well-designed to test specific hypotheses/claims. Yet, experiments should probably cover more than that, such as demonstrating the properties of the main algorithm(s), which can be critical for method adoption.

Minor points: 
- May I know the purpose of "4.1 Example" as it seems to be just deriving various quantities for a concrete/standard Gaussian mixture? I don't see how it helps the readers better understand Theorem 1.
- In Section 5.2, is there an intuitive explanation for why "the optimal $\sigma$ here is in fact larger than the noise level needed to make $p(y_1)$ log-concave"?
- Would it make sense to add more description for Figure 4, panels (c) and (d)? Currently, it is not entirely clear what they are offering.

### Questions
Please take a look at the "Weaknesses," where the comments cover both theory presentation and experiments, as well as some additional questions (under "minor points"). 

One additional question: 
- Is the actual code anonymously shared somewhere? It would be good to have it for result reproducibility and smooth adoption of the method.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a framework of sampling from unnormalized densities by transforming this problem to more amenable log-concave sampling.

### Strengths
**Originality**: The paper is an extension of the paper by Saremi & Srivastava. The proposed framework is very different from what I could find in the literature.

**Quality**: Once familiar with the literature (see weaknesses), I found the framework really elegant and simple. I learnt a lot from reading this paper.

**Clarity**: The paper is well-written (again see the caveat in the weaknesses section) for someone already familiar with this literature. The calculations are thorough without being burdensome. The explanations are crisp without being terse.

**Significance**: The final algorithm proposed is simple and should be easily adopted by the community. More importantly, I found the paper useful in learning how to think about sampling, and I believe it will be useful for the rest of the community also.

### Weaknesses
The paper could have done a better job at exposition of not so well-known notions like walk-jump sampling. The current exposition is good enough for someone intimately familiar with the literature, but for others, like me, it takes some reading of cited papers to not see ideas as being "pulled out of a hat".

### Questions
1. How to handle the case when the desired probability measure does not have a density?
2. It would have been insightful to discuss adversarial cases where the proposed sampling technique fails.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a framework for sampling from an unnormalized density $p\propto{\rm e}^{-f}$ on $\mathbb{R}^d$ by using a Gaussian smoothing scheme. It constructs $X\sim p$, $Y_i|X\sim_{\rm i.i.d.}\mathcal{N}(X,\sigma^2I)$ ($i\in[m]$) for some suitable $\sigma$ and $m$. By sequentially (one at a time) sampling the joint distribution of $Y_{1:m}$, the conditional distributions will become "more log-concave", and finally it outputs $\mathbb{E}[X|Y_{1:m}]$ as an approximate sample from $p$, which can be expressed in a function of $\bar{Y}_{1:m}$. The paper demonstrates the advantages of the sequential sampling scheme through some theoretical and empirical examples.

### Strengths
Overall, the paper introduces a novel theoretical framework that differs from existing MCMC methods or sequential inference models. The paper is well-written, with clear notations, convincing examples, and mathematically rigorous proofs. The structure is well-organized: the authors first introduce an example of an anisotropic Gaussian target distribution, which leads to the important observation that OAT is easier than AAO in terms of the condition number. Then, they present the theory for general distributions (once log-concave, always log-concave), which helps the reader understand the motivation.

### Weaknesses
Score estimation is essential for the algorithm's efficiency. However, most of the experiments in the paper use the analytic score function, which is not realistic in practice. Moreover, the score has to be estimated at each step of ${\rm MCMC}_\sigma$, which will significantly increase the algorithm's complexity.

The paper only compares the performance of accurate and approximate scores in appendix H, and the result is not very satisfactory. The estimated score performs well when $d=2$, but it deteriorates when $d=8$. These results suggest that the score estimate's precision could be much worse in high dimensions, which I believe is the paper's main weakness.

In 4.2.1, the proposed method for score estimation is based on importance sampling (which transforms the expectation w.r.t. the probability density $\nu\propto\exp(-f-\frac{1}{2\sigma^2}||\cdot-y||^2)$ to $\mu=\mathcal{N}(y,\sigma^2I)$). However, (Chatterjee and Diaconis, 2018) has shown that the sample size needed for accurate estimation by importance sampling is usually $\exp({\rm KL}(\nu||\mu))$ for general target functions. For this reason, (Huang et al, 2023, section 3) use Langevin Monte Carlo to sample from $\nu$ and combine it with importance sampling in a similar task of score estimation. The experiments in your paper indicate that the score estimator also suffers from high variance. The authors should experiment with different score estimators, instead of saying that "studying the covariance of the plug-in estimator and devising better estimators is beyond the scope of this paper".

References:

(Chatterjee and Diaconis, 2018) Sourav Chatterjee, Persi Diaconis. The sample size required in importance sampling. Ann. Appl. Probab. 28(2): 1099-1135 (April 2018). DOI: 10.1214/17-AAP1326.

(Huang et al., 2023) Xunpeng Huang, Hanze Dong, Yifan Hao, Yian Ma, Tong Zhang. Reverse Diffusion Monte Carlo. ArXiv preprint arXiv:2307.02037.

### Questions
1. The algorithm outputs a *biased* sample of $p_X$, but the bias $W_2^2(p_X,\hat{p}_{m^{-1/2}\sigma})$ can be made arbitrarily small by choosing $\sigma$ and $m$ such that $\frac{\sigma^2d}{m}$ is small enough. How can we choose these parameters in a principled way? In the Gaussian example (proposition 4), we can reduce $\kappa_1$ by increasing $\sigma$, and in theorem 1, we need a large $\sigma$ to make $p(y_1)$ strongly log-concave. However, this also increases the bias, so we need a larger $m$ to compensate that. How can we balance this trade-off between complexity and accuracy?

2. In the experiment, how do you choose the projection direction $\theta$ for estimating the Wasserstein-2 distance? A more fair comparison would be to use $\theta$ following the uniform distribution on the Euclidean unit ball (i.e., the *sliced* Wasserstein-2 distance). Efficient approximation algorithms are available, for example, in (Nadjahi et al., 2021).

Minor corrections:

1. In A.1, $-2\sigma^2\log p(y_{1:m}|x)=\sum_{t=1}^{m}||y_t-x||^2+{\rm cst}$. 
 
2. In remark 3, for (4.7) to hold, ${\rm cov}(Z|y_{1:m})\preceq{\rm cov}(Z|y_{1:m-1})$ almost surely; ... is very large, so that ${\rm cov}(Z|y_{1:m-1})\approx0$.

3. The exponential family in the third footnote should be $p(x|\nu)=\exp(-f(x)+\frac{1}{\sigma^2}(x-x_0)^\top(y-x_0)-\frac{\nu}{2\sigma^2}(||x-x_0||^2+||y-x_0||^2)-a(\nu))$. 

4. In the third paragraph of introduction and the second paragraph of section 2, $Y_t=X+N_t$, $t\in[m]$. $N_t$ are i.i.d. $\mathcal{N}(0,\sigma^2I)$ random variables that are also independent of $X$. In theorem 1, $X=Z+N_0$, $Z$ and $N_0$ are independent.

References:

(Nadjahi et al., 2021) Kimia Nadjahi, Alain Durmus, Pierre Jacob, Roland Badeau, Umut Şimşekli. Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections. In Advances in Neural Information Processing Systems, 2021.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
