# Capturing Uncertainty in Regression via Conditional Diffusion Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Quantifying uncertainty is a fundamental problem in both statistics and machine learning. Existing uncertainty quantification (UQ) approaches often suffer from significant limitations, including high computational cost, restrictive parametric assumptions, and overly conservative prediction intervals. In this paper, we propose a UQ framework based on diffusion models. In regression tasks, we learn the full conditional distribution of the response variable given the input features. Our method enables flexible, nonparametric modeling of complex conditional data distributions. We construct prediction intervals from the learned conditional distribution and establish theoretical guarantees on their coverage probabilities. Empirically, we conduct experiments on both synthetic and real-world regression tasks to evaluate the effectiveness of our approach. The results demonstrate that our method achieves competitive or superior performance in predictive uncertainty estimation compared to a range of established baselines, offering a powerful, efficient and theoretically grounded alternative for uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new framework for uncertainty quantification using conditional diffusion models for nonparametric regression with data-dependent noise. The authors present a rigorous theoretical analysis. Note: I am not particular familiar with the theory of diffiusion models, so my review can be biased at times.

### Strengths
- A rigorous mathematical analysis of the proposed framework is provided
- emprirical comparison of the proposed framework to standard baselinem methods is provided

### Weaknesses
- The paper frames the contribution as a new method in the abstract but then states in line 108 that the "work prioritizes theortical analysis". This is an inconsistent framing of your contribution
- The comparison to prior work on diffusion based uncertainty quantification is missing
- There is no comparison or discussion on the computational comparison to baseline methods
- The insights from the theortical analysis is not utilised in the experimental results. What do we gain from the theory and how can we see this in the experiments?

### Questions
- The font size in Figure 1 seems to be smaller. Please correct for that. 
- line 148. References end with "[...], Unlike". Some grammar is off here.
- if you have no section 5.2, then section 5.1 does not head a separate section.
- Experiments: Can you explain the baseline methods? They are not stated so within the main text.
- You mention that Han et al. (2022) provides a method for UQ using diffusion models. 
  - How is that different from your framework?
  - can you provide experimental results for diffusion based UQ to see the benefits of your method?
- Can you provide a computational analysis (theoretical and experimental) into the overhead of your method compared to baseline methods?
- Can you "verify" your theortical analysis in experiments? Or at least connect the theory to the experimental results?
- In the limitations there is distribution shift mentioned. Can you provide analysis or experiemntal results on this topic?

### Soundness
3

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
3

### Summary
The paper presents prediction intervals built from conditional diffusion models that learn the entire conditional distribution of the response given features. After training a score network, the method draws many synthetic responses at a new input and takes lower and upper empirical quantiles as an interval. The theory defines a risk for score learning, assumes smooth conditional densities with light-tailed responses and sub-Gaussian features, and proves a high-probability bound on the coverage error. That bound separates the error due to using a finite number of generated samples from the error caused by imperfect diffusion modeling at the test input, summarized by a shift coefficient that grows when the learned score generalizes poorly out of distribution.

### Strengths
The authors tackle a very important, yet quite "unsolved" problem, so the research direction itself is very promising. Just wanted to mention that. Furthermore, the method is spelled out concretely in a short algorithm that is easy to implement in practice (more or less sampling from the trained conditional diffusion and taking empirical quantiles). 

I also find the approach to uncertainty quantification via diffusion models quite appealing (while I am not sure about the actual benefits, it seems to me something sensible to try).

### Weaknesses
Note that I am by no means an expert when it comes to diffusion models, but I am quite confident about (general) uncertainty quantification in machine learning. I am open to discuss anything that the authors do not agree with me.

On the theoretical side let me remark the following: 

>The main theoretical guarantee appears stronger than what the proof delivers. Theorem 4.5 states that, with probability $1-\delta$, for any $x_{\text{new}}$, the coverage error of the proposed interval is bounded. However, the proof first controls the total variation distance between the true and generated conditionals in expectation over the training data (via the score-risk), and only then adds a high-probability DKW term for the Monte-Carlo error from using $n_1$ generated draws to estimate quantiles. The coverage expression itself evaluates the true CDF at generated quantile endpoints, and decomposes the error as $F-\hat F$ (modeling) plus $\hat F-F_{n_1}$ (sampling). 

To avoid overstating the result, the theorem should mirror this two-stage structure, make clear which randomness the probability $1-\delta$ is over, and clarify whether the bound is pointwise (fixed $x_{\text{new}}$) or uniform in $x$. 

>There are also log-factor inconsistencies across the derivation. Earlier steps use $(\log n)^{\max(17,\,1+\beta/2)}$ for $E[R(\hat s)]$, while the final bound reports $(\log n)^{\max(9,\,32+\beta/4)}$ (elsewhere $\max(9,\,1+\beta/4))$. This likely comes from taking a square-root after Jensen (which should halve the log exponent). 

Aligning these exponents would make the rate easier to verify.

>The guarantee is in-expectation over the training data and scaled by a distribution-shift coefficient $T(x_{\text{new}})$ that can be large out of distribution. 

This is a useful diagnostic but it is not a conformal-style frequentist coverage guarantee for the true conditional at test time. An explicit comparison to conformal prediction (what is and isn’t guaranteed here) would help.


Minor things: 

Typo just after proof of Lemma 4.8, "bounded" instead of "bouned", before Proof of Thm. 4.5 "invokes" instead of "invoke".

### Questions
The authors might know, that the aleatoric-epistemic dichotomy is quite an active (sub)-field of uncertainty quantification in machine learning. From reading the paper, it is not directly clear to me, how the authors represent here these uncertainties with their method? 
I would be very interested in the relationship of the diffusion based approach to aleatoric, and epistemic uncertainty. Seems quite fruitful to me. I have the feeling the authors did not consider (even mention) this at all. 

More on the topic of the paper, how do the authors recommend estimating or upper-bounding $T(x_{\text{new}})$ in practice? 

The DKW step requires the $n_1$ generated samples at $x_{\text{new}}$ to be i.i.d. from the generated conditional. Do you obtain these via independent restarts of the reverse SDE with fresh noise?

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
This paper is focusing on the theoretical analysis of diffusion models for predicting regression distributions with confidence intervals and UQ. The main value of the paper is the theoretical proofs. The novelty of the algorithm is limited, and the experimentation could be strengthened at a few places. Also, an experimental/practical connection to the bounds in Theorem 4.5 would help understanding the importance of the theorem.

The limitation of the proposed method lies in the use of a confidence interval. I wonder how the proposed method handles for example bimodal distributions.

I could not find the anonymized GitHub repository included with the supplementary materials. 

I admint that due to time constraints I had no chance to attempt to verify the proofs.

### Strengths
+ New UQ algorithm that works well for regression problems
+ Deep theoretical analysis
+ Reproducible (given a question below)

### Weaknesses
- A connection of the bounds in Theorem 4.5 with the practical performance could highlight the importance of the theoretical analysis.
- The reliance on confidence intervals limits the applicability of the proposed method, e.g. in the case of bimodal distributions, problems with aleatoric uncertainty of the type of multiple correct (range of) answers
- Baseline methods could have been extended

### Questions
I wonder why Split-CP and CQR, clearly weak baseline methods, were selected. Why not for example CARD from [Han et al 2022], which outperforms Deep Ensemble and Dropout in their paper.

PICP evaluates an interval, hence only works for unimodal. I also wonder how the size of the interval is taken into account. For example, how about using the energy score [Gneiting & Raftery, Strictly proper scoring rules, 2007] or [Jordan et al., Evaluating probabilistic forecasts with scoring rules. 2019]?

I could not find the anonymized GitHub repository included with the supplementary materials, please provide pointers.

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
This paper studies conditional diffusion models (CDMs) as an uncertainty quantification (UQ) framework for regression tasks. The authors consider nonparametric regression with data-dependent noise of the form $Y = f(X) + g(X)\varepsilon$, where $\varepsilon \sim \mathcal{N}(0,1)$. Rather than directly estimating the regression function, the method learns the full conditional distribution $P(Y|X)$ through a conditional diffusion model. The approach constructs prediction intervals by sampling from the learned conditional distribution and extracting appropriate quantiles. The paper mathematically proves convergence rates. Additionally, the paper empirically evaluates the performance of CDM on the UCI datasets. 

I am not familiar enough with the literature to judge the novelty of their algorithm.

In the abstract, the paper claims: “In this paper, we propose a novel UQ framework based on diffusion models.”

But later, the authors state:

“We are aware that Han et al. (2022) are one of the earliest efforts to apply diffusion methods for UQ in classification and regression [...]”. The paper should be clearer about which aspects of the CDM method/algorithm are novel. How is it different from Han et al. (2022)? Or is it only the theoretical and empirical analysis novel?

For related theory, you might want to cite https://arxiv.org/abs/2502.03435 (I think it is for classical DMs rather than for CDMs and under some constraints, but it also deals with training dynamics, while your result assumes that we can perfectly solve optimization problems. I think the results don’t have too much overlap, but it adds some interesting insights that even very large models don’t overfit if they are trained with a large step size.)

I think the authors should probably mention the concurrent work: https://openreview.net/pdf?id=IWQNhR6poq (and also check some of the works cited therein)

### Strengths
1. **Theoretical Contribution**: Theorem 4.5, provides a novel theoretical result on the coverage probability converging to the correct level with explicit bounds. This theoretical contribution is valuable even though the practical applicability remains unclear. I am absolutely not an expert on the current state of theory on CDMs, so I cannot guarantee that this result is actually novel.

2. **Heteroscedastic noise modeling**: To my understanding, thanks to learning the full conditional distribution $P(Y|X)$ via diffusion models, the framework handles data-dependent noise through the $g(X)\varepsilon$ formulation, which is more general than some other UQ methods that assume homoscedastic noise. In principle, CDM could also learn non-Gaussian noise.

3. **Nonparametric approach**: The diffusion model framework provides a flexible, nonparametric way to model complex conditional distributions without restrictive parametric assumptions.

4. **Results:** The RMSE and NLL of CDM can quite often outperform DE, which is a strong contribution. However, it is not fully clear how the hyperparameter optimization (HPO) was done to obtain these results.

### Weaknesses
1. **Missing code despite mentioning it in the reproducibility statement**: The paper appears to have indicated code/supplementary materials were provided, yet it is not accessible to reviewers.

2. **Methodological evaluation can be improved**:

   - RMSE focus is unnecessarily strong when the focus of the paper is UQ. 
   - RMSE differs between the methods quite a lot, indicating large changes in the point predictor across methods. Therefore, it is unclear if the UQ is better or if it is rather the point predictor that is better. It would be interesting to obtain a more fine-grained understanding of this. (You could compare the performance of the models if they are all given the same point predictor and only use the uncertainties from the different methods.) There should also be some explanation for why the point-predictors vary so much. E.g., without reading the appendix, it is not clear that CQR uses the mid-value as a point predictor.
   - While you do achieve strong NLL results, further metrics would be helpful. For example, interval width analysis, calibration plots, or quantile loss, some kind of normalized calibrated interval width, etc. More UQ-specific metrics would be valuable. Even though NLL is a proper scoring rule, the Gaussian assumption is a strong assumption in practice, as you denote in Appendix G.3. Especially for the CDM, it should be able to learn non-Gaussian distributions, but you don’t include any metric that explicitly measures how well it learns non-Gaussian distributions. For the NLL, you simply fit a Gaussian through the predictive samples of the CDM? Computing the quantile loss of the estimated quantiles would evaluate how well the methods can learn non-Gaussian distributions. The Gaussian case is, of course, nicely aligned with your theory, but intuitively, non-Gaussian distributions should be a strength of the CDM.
   -  The PICP metric alone, without the interval width, is hard to interpret. Is 100% coverage really desirable? I would say only if the intervals are not too wide. The low NLL kind of indicates narrow intervals already, but it would be nice to see it explicitly. The NLL does not directly say much about the quality of the (non-Gaussian) quantiles, which could deviate quite a lot from the Gaussian quantiles (at least in theory). 

3. **Computationally heavy due to 1000 steps**: 1000 diffusion steps for 1D regression outputs seems like a lot, and while reported wall-clock is comparable to DE and much faster than MC Dropout in Table 5, accelerating sampling (e.g., DDIM/DPM-Solver) would likely reduce cost further.
At inference time, your implementation of DE is very different than the original one. The original one is much faster and more precise. The original DE paper only needs 1 forward pass for each of the ensemble members and then computes the std of the 5 point predictors and aggregates the predicted variances of each of the 5 ensemble members via a closed formula without any form of sampling.
Maybe you also want to discuss possibilities for parallelization? I think for DE training can be easily parallelized, making it much faster in practice. For CDM, Inference could be sped up by parallelization, which would speed it up a lot but still be slower than DE.

4. **Limitations of the theoretical guarantees**: The coverage bound seems to depend on $\mathcal{T}(x_{\text{new}})$ term, for which i have little intuition. How can this term be bounded? Can you provide more intuition on this term? I think this term can get very large very easily. For some model classes, this term might be infinite? What assumptions do you need to make on the hypothesis class $\mathcal{F}$ for this term to be finite? Copactness would probably work.

5. **Missing critical comparisons**: No comparison with recent diffusion UQ (e.g.,[1],[2]). There are just so many diffusion UQ papers that you can compare against, so the novelty is not clear to me.

6. **Unsuitable submission track**: As a minor remark, this should have been in "Probabilistic methods (e.g., variational inference, causal inference, Gaussian processes)", not "generative models"

7. Epistemic Uncertainty is not modeled. I assume that for smaller training datasets or for far OOD data, this model would fail severely to capture the uncertainty. It would be an interesting extension to combine this model with epistemic uncertainty.

8. In the introduction, the literature review does not distinguish between methods that estimate aleatoric uncertainty, epistemic uncertainty, or both, which would be relevant for some applications

9. Line 2153: “before in Section G.2.2”. Delete the “before” if Section G.2.2 is after G.2. In general, in the first paragraph of G.2, it is not clear to which experiments this refers. It says “the same as in Section 5.1”, but I thought G.2 is where Section 5.1 gets explained. When you write “We also double the number of hidden units in each layer, [...]” Doubled compared to what?

10. Ablation studies would be helpful. What if all methods had more hidden layers? What if all methods were calibrated?

### Questions
1. **What base models are used for Split-CP and CQR?** The paper states all methods share the same two-layer MLP backbone. Have you tried different models as in [3]?

2. **Why does RMSE differ between Split-CP and CQR?** RMSE is a point prediction metric, and one would assume that both methods should use the same underlying point predictor. It would help to be explicit about what point estimators were used when computing RMSE in the main paper or to explicitly reference the appendix.

3. **Where are standard UQ metrics?** Why no interval width analysis, quantile loss, or normalized calibrated interval width? These are standard metrics in the UQ literature.

4. **Reproducibility materials were mentioned to be provided, but were not accessible** This is a bit unfair for other submissions that provide code/supplementary materials. Can you please clarify why this was missing?

5. **How is this different from [1] or [2]?** What is the key novelty of your approach?

6. **Why 1000 diffusion steps for scalar outputs?** Have you tried any acceleration methods (e.g., DDIM, DPM-Solver)? This seems computationally wasteful for 1D regression.

7. **How do you bound $\mathcal{T}(x_{\text{new}})$ in practice?** The bound scales with an unbounded shift coefficient  $\mathcal{T}(x_{\text{new}})$; absent assumptions or diagnostics to control it, practical coverage under distribution shift may be weak.

8. Why submit to this track? The paper might fit more naturally in the probabilistic-methods track (e.g., variational inference, causal inference, Gaussian processes) rather than generative models.

9. Minor comment: Maybe you also want to compare against SQR [1], which is, from my perspective, the most common benchmark when using QR with neural networks. Standard CQR is usually done with non-DL models.

10. **What assumptions do you make on $\mathcal{F}$ in Theorem 4.5?** I think you assume that $\mathcal{F}$ grows in a very specific way with the number of training datapoints? You should be more explicit about this

11. **IMPORTANT QUESTION: Figure 6**: What is the difference between CDM PIs (in-domain) and CDM PIs (OOD)? Why do both get wider outside of the range of the training data? Is this plot cherry-picked such that the uncertainty gets wider out of sample? Or is this a general pattern? I would be very surprised if pure CDM without explicitly modelling epistemic uncertainty results in bounds that consistently widen up OOD. (Typo in caption of Figure 6: should be “as in Figure 1” instead of “in Figure 1”.)

12. What do these variable names stand for in Line 335? $M_t, W, \kappa, L, K$?

13. Which value of alpha is used in the experiment? 5%?

References:

[1]: Chan, M., Molina, M., & Metzler, C. (2024). Estimating epistemic and aleatoric uncertainty with a single model. Advances in Neural Information Processing Systems, 37, 109845-109870. ,https://arxiv.org/pdf/2406.07658

[2]: Beltran Velez, N., Grande, A. A., Nazaret, A., Kucukelbir, A., & Blei, D. (2024). Treeffuser: probabilistic prediction via conditional diffusions with gradient-boosted trees. Advances in Neural Information Processing Systems, 37, 118296-118325.

[3]: Tagasovska, N., & Lopez-Paz, D. (2019). Single-model uncertainties for deep learning. Advances in neural information processing systems, 32. https://arxiv.org/abs/1811.00908

### Soundness
3

### Presentation
2

### Contribution
2
