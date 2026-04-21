# Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization

- Avg Score: 7.60
- Decision: Accept (Oral)
- Scores: 8, 8, 6, 8, 8

## Abstract
A long-standing belief holds that Bayesian Optimization (BO) with standard Gaussian processes (GP) --- referred to as standard BO --- underperforms in high-dimensional optimization problems. While this belief seems plausible, it lacks both robust empirical evidence and theoretical justification. To address this gap, we present a systematic investigation. First, through a comprehensive evaluation across twelve benchmarks, we found that while the popular Square Exponential (SE) kernel often leads to poor performance, using Mat\'ern kernels enables standard BO to consistently achieve top-tier results, frequently surpassing methods specifically designed for high-dimensional optimization. Second, our theoretical analysis reveals that the SE kernel’s failure primarily stems from improper initialization of the length-scale parameters, which are commonly used in practice but can cause gradient vanishing in training. We provide a probabilistic bound to characterize this issue, showing that Mat\'ern kernels are less susceptible and can robustly handle much higher dimensions. Third, we propose a simple robust initialization strategy that dramatically improves the performance of the SE kernel, bringing it close to state-of-the-art methods, without requiring additional priors or regularization. We prove another probabilistic bound that demonstrates how the gradient vanishing issue can be effectively mitigated with our method. Our findings advocate for a re-evaluation of standard BO’s potential in high-dimensional settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper challenges the widely held notion that Bayesian optimization, using standard kernels, does not scale beyond problems with a, say, single-digit number of independent input variables $d$. The central hypothesis is that standard-kernel GP models _can_ indeed model empirically relevant high-dimensional optimization problems, but are hampered by a problem of hyperparameter tuning. Simply put, if the length-scales $\ell_0$ of isotropic kernels are initialized of order 1, independent of $d$, then the implied covariance between a fixed number of function evaluations drops exponentially with $d$, and so do the gradients of the marginal likelihood, quickly reaching a regime in which hyperparameter optimization becomes numerically impossible. 

Instead, the authors propose to initialize kernel length scales as $\ell_0 \sim \sqrt{d}$, and show that doing so effectively counters the exponential decay in correlation between datapoints, and the associated gradients of the marginal likelihood. They argue further that this surprisingly simple change actually suffices to make standard-kernel GP Bayesian Optimization competitive with methods specifically designed to address high-dimensional global optimization problems.

### Strengths
This work makes a nice, compact analytical point about about an algorithmic detail, and provides a host of empirical evidence to support the hypothesis. The theoretical analysis appears sound, although the overarching argument can perhaps be questioned (more below). Prior work is extensively reviewed, and the text itself is an enjoyable read.

### Weaknesses
Overall I think this is a fine paper. I am not entirely sure I follow the whole argument, though. Here is why:

From my understanding, the argument was never quite that standard kernels are particularly bad as modelling high-dimensional functions, but that general high-dimensional functions are (in the worst case) just exponentially hard to optimize, period. One can "hide" a global optimum among an exponentially growing number of local extrema in, e.g., a sample from a $\ell_0=1$ SE kernel GP (indeed, the analysis provided in this paper indirectly shows that this is true). So, while I completely agree with the authors that the gradients of the marginal likelihood become exponentially suppressed with growing $d$ if the kernel is initialized with $\ell_0=1$, and that this can be alleviated by initializing $\ell_0\sim \sqrt[d}$, a second point also needs to be true for the paper's argument to work: **Practically relevant** objective functions must have some property that constrains their shape such that global optima remain identifiable in feasible sample complexity, *and* that property must be at least approximately captured by the isotropic kernels when initialized in the way proposed in the paper. I am not entirely convinced that the paper provides sufficient evidence in this regard, but I realise that this is a debatable point. Hence I'm trying to phrase a question below.

On a minor note, the authors admit quite clearly that the recent work by Hvarfner et al. achieves pretty much the same effect as their method, but just uses an ever so slightly more involved way to initialize $\ell_0$ by drawing from a LogNormal distribution. Personally, I don't think this is a major issue, since the authors deal with it quite openly. I guess one could argue that this limits the novelty of the work somewhat, though.


### mini issues, easy to fix: 
* Section 2, line 091: It's not technically true that BO aims to find new points that are "ideally closer to the optimum". They should just provide helpful information about the location and/or value of the optimium. 
* Proposition 4.3 starts with a typo ("Given $\forall$")
* Eq. 6: the parentheses around the argument of $\exp$ are too small. Use `\bigg` or `\left`

### Questions
For the moment I have marked the paper as marginally above threshold, but I would be quite willing to raise my score if the point I raised under "weaknesses" is addressed convincingly in the discussion. To re-state it as a question: Are you claiming that practically relevant optimization problems do not have an exponentially growing (in $d$) number of local optima? Or perhaps less explicitly, that practically relevant problems have some other property that means they can be globally optimized with a sample complexity that grows sub-exponentially with $d$? Which part of your work supports this claim? (For example, would you say that the problems listed in Figure 4 are a good surrogate for "practically relevant" problems? How so? 

As a counter-example, have you run `SBO-SE (RI)` on a genuine sample of a Matérn $\ell_0=1$ kernel? (Such samples _do_ have an exponentially growing number of local extrema). What happens in such a case?  

Thanks in advance for your clarifying reply!


---

Update post discussion:

The authors have addressed my questions. One could always discuss further, and I do see some open questions left here. However, I am convinced this paper provides an empirically highly useful technique, reasonably motivated and supported by theory.

The presence of a related paper does not seem problematic to me, since the related work is amply cited, and the present paper provides a further simplification that users will find highly usable.

Overall, the results presented here will find direct, valuable use with a significant subset of the ICLR community. I thus recommend accepting this paper.

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
The common assumption is that Gaussian Process-based Bayesian Optimization inherently scales poorly with the dimensionality (D) of the search space, such that the search is virtually uninformed for D surpassing 15-20. This paper investigates the validity of this assumption and the reason for the typically observed poor performance on search spaces of high dimensionality. They show that, using the popular SE-kernel, the gradient of the log likelihood with respect to the lengthscales vanishes (with high probability) in the case of high dimensionality and typical initial lengthscales, preventing effective gradient-based training of the lengthscales. While Matérn kernels also experience vanishing gradients, the problem occurs at much larger D compared to SE, which aligns with their experimental performance results. Furthermore, the authors show that initialising the lengthscales to \sqrt{D} is sufficient to overcome the vanishing gradient issue and achieving performance that matches or exceeds that of more advanced Bayesian Optimization methods.

### Strengths
The overall story of the paper is easy to follow, the investigated topic is important, and the proposed approach is sound. The theoretical analysis (Section 4) of the vanishing gradients is interesting and, as far as I am aware, presents a novel perspective on the curse of dimensionality in this context.

### Weaknesses
The central thesis of this paper is that a standard "no bells and whistles" implementation of Gaussian Process (GP)-based Bayesian Optimization (BO) can perform competitively with state-of-the-art methods for high-dimensional BO, requiring only a straightforward adjustment to common practices. However, this thesis closely mirrors that of the existing work (1) "Vanilla Bayesian Optimization Performs Great in High Dimensions" (ICML 2024), which presents a similar argument.

In the paper under review, the authors propose scaling the lengthscales proportionally to \sqrt{D}, and they achieve this by initialising the lengthscales with \sqrt{D}. The existing work (1) similarly advocates for scaling the lengthscales in proportion to \sqrt{D}, as discussed in Section 5.1 of their paper. They suggest implementing this adjustment by setting the hyperparameters in the prior on the lengthscale appropriately, noting that using a maximum a posteriori (MAP) estimate for a hyperprior is a straightforward and widely-used approach.

While the reviewed paper does cite the existing work, its claim that the proposed method is "even simpler" (see Section 5 "Alternative Method") is not entirely convincing. Both papers recommend favouring lengthscales near \sqrt{D}; the difference lies in how they implement this: the reviewed paper achieves it solely through initialisation, whereas the earlier work suggests adjusting the prior on the lengthscales. As it stands, the reviewed paper presents the main thesis as though it were novel (at least in the abstract, introduction and conclusion), despite its overlap with the prior work.

Nevertheless, there are valuable contributions in the reviewed paper, such as the analysis of vanishing gradients, which offers an insightful perspective on the consequences of using inappropriately small (but typically used) lengthscales. To improve the work, the authors could clearly position it as an expansion upon the thesis established in the earlier paper, emphasising the novel aspects of their analysis and findings.

### Questions
I do not have any questions. Please see the suggestions on how the work could be improved under "Weaknesses".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper thoroughly investigates the performance of standard GP in high-dimensional BO. Specifically, the paper focuses on the use of SE and Matern kernels in standard GP, along with analysis on their gradient values during GP training. The authors point out the failure of using SE kernel is mainly due to the vanishing gradient problem and propose a new initialization strategy for kernel length-scale that mitigates the problem and improves performance. Comprehensive experiments are carried out in comparing performance of the proposed standard BO (SBO) against other state-of-the-art BO algorithms on various optimization tasks.

### Strengths
1. The authors claim the vanishing gradient in the SE kernel training is the main cause of BO failure in high-dimensional cases and provide detailed theoretical analysis on the gradient values of SE kernel. The theoretical works are easy to follow and provide insights on how likely the vanishing gradient problem can occur when initial values of length-scale are fixed in high-dimensional setting. The proposed initialization strategy is also supported by a probabilistic bound.

2. The authors perform adequate experiments to validate their statement. Both high-dimensional synthetic functions and real-world applications are considered, and the chosen BO algorithms are representative and classic in the field of high-dimensional BO. In addition to theoretical analysis, the authors also provide several experiments results to confirm both the existence of vanishing gradient problem and its relationship with initial length-scale and dimensionalities.

### Weaknesses
1. This paper mainly focuses on evaluating existing method. Vanishing gradient problem has been widely discussed in gradient-based training. I think the novelty of this submission only comes from the proposed initialization method, but the authors spend only limited paragraph highlighting the performance of this proposed method. In the real-world problems, the proposed SBO-SE(RI) method even worsens the performance of the original SBO-SE in most cases (namely every case except for the SVM(388) according to Figure 4).

2.  I'm not convinced by the experiments design in Section 4, where the authors try to further validate their theoretical results by varying $l_0$ and $d$. (1) The evaluation metric used to confirm the occurrence of vanishing gradient lacks generality. To my understanding neither the $L_2$ difference nor the average gradient norm at the first training step reveals the presence of vanishing gradient throughout the training process. The number of training steps is not specified by the authors but it is safe to say there are more than 1 step of kernel parameter training. Therefore, the numerical difference in $L_2$ norm of length-scale before and after training only proves $l$ was updated in some steps but vanishing gradient could still have occurred in other steps. The latter gradient norm metric is only considered for the first step of training without a generalization to the whole training process. Vanishing gradient could still occur in the following steps. Step-wise gradient/length-scale evaluation would be more convincing. (2) The values of length-scale initialization ($l_0$) is not robustly tested. In line 199-200, the authors state fixed $l_0$ will lead to vanishing gradient and in the later numerical verification (line 239-240) several fixed values of $l_0$ are tested. However, given the minimal chosen value of $d$ is $50$, $\sqrt{d} > 7$ which is much larger than other choices always. For better generality, I think some larger fixed value of $l_0$ (e.g. $5, 15, 25$) should also be considered as a stronger support for the adaptive $l_0 = c\sqrt{d}$ setting. 

3. The readability of this submission needs to improve. Starting from line 233, I keep being instructed to jump between sections to get details and results of the experiments. For example, in line 236 I need to go to section 6.1 to get definitions of the benchmark notations. Similarly in line 264 I'm referred to Figure 3 and 4 which are at the very end of the paper. It would be much easier to read if the authors could gather all the numerical experiments into Section 6 instead of alternating between theoretical presentation and experiments evaluation (e.g. move line 233-324 to the start of section 6).

### Questions
1.  In equation (1) and (2), both ARD SE kernel and ARD Matern kernel contain the amplitude $a$. Since in line 405 a Gamma prior is placed on this parameter, I assume the authors also include $a$ as the trainable parameter in GP training. How did the authors initialize $a$ in GP training? In equation (6), is the amplitude $a$ considered as a constant during theoretical analysis? In line 209-212, why is $a$ not presented in the gradient expression of the Matern kernel?

2.  In Figure 2, the relative length-scale differences for both GP-Matern($l_0=0.693$) and GP-SE($l_0=\sqrt{d}$) are flat lines. Since $l_0$ is initialized at the same value independent of the number of iteration, does this mean the value of length-scale is all the same after training even for different GP (since the GP at each iteration is modeled on different data sets)?

3.  Consider cases: Ackley(150, 150), Mopta08(124), Rover(60), NAS201(30), DNA(180) in Figure 3 and 4 (5 out of the 11 benchmarks), it seems adding the robust initialization to SE kernel worsens its performance. How would the author justify this observation supports their statement in line 345-346 ("the performance across all the benchmarks is dramatically improved")?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate vanishing gradient problems associated with the squared exponential and Matern kernel functions, finding them both to exhibit gradient vanishing at common default initializations. They propose a dimension-dependent initialization and advocate for the Matern kernel, and demonstrate on numerical experiments that this can actually be more important than all the bells and whistles we have been putting on GPs for the last decade.

### Strengths
The aim of this work, to challenge a basic, un-sexy component of the BO framework as the lengthscale initialization/covariance kernel and to attempt to pin down the influence on model performance is great. 

At first, I found it difficult to fully take in the implications. But if my quick refreshing/skim of them is to be trusted, the articles proposing TuRBO, Tree UCB, RDUCB, HESBO, ALEBO, SaasBO do not have any "vanilla BO" baseline: they compare only to other sophisticated methods. This article may have revealed that the emperor has no clothes: this thread of literature may indeed have been missing the basic Matern kernel as a suitable approach to high dimensional BO. If correct, this finding will have a significant impact on the future of BO research in high dimension. It should probably also prompt some soul-searching among us who have been doing work in this area.

### Weaknesses
---

Insufficient replicates in the numerical experiments. 

Can a study with 10 replicates really be said to be a "comprehensive evaluation"? Particularly when there is such a large variance in the results: maybe half of the method-problem combinations seem to span the entire plot. Given how influential the conclusions of this article would be if correct, it is really disappointing that they are based on estimates of performance with such high standard error. Running (significantly) more experiments would solidify the important findings of this article. Really, there should probably be closer to 50 replicates or more.


---
Criticism of the squared exponential and advocacy of the Matern kernel is not new. 

Stein, M. L. in Interpolation of Spatial Data makes this argument in 1999, and that is referenced in the covariance chapter of Rasmussen and Williams. 

It is probably worth citing at least these documents, and trying to locate other discussion of this topic.

---

While there are indeed (surprisingly) many software platforms which use a constant lengthscale initialization (including GLflow, GPJax, Spearmint, GPStuff, GPML, scikit-learn and GPy, based on my quick skim of their source code), this is not universally the case. 

There are also packages that do indeed use the idea of involving the expected distances between points when choosing the lengthscales.
These packages actually do so empirically instead of assuming a uniform input distribution, which is surely violated in, say, Bayesian Optimization settings. The ones I found based on a quick search were:
i) DiceKriging uses multiple random initializations between 1e-10 and twice the range of the distances.
ii) hetGP initializes the lengthscale based on the quantiles of the observed distances.
iii) GPVecchia seems to use 1/4 times the mean distance between training points.

It is probably worth mentioning these and situating them relative to your proposal.

### Questions
Why not use a Matern kernel with the proposed initialization in your numerical experiments?

Under the assumption of uniform sampling, does any of the procedures of DiceKriging, hetGP or GPVecchia satisfy your robust initialization?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the problem of high-dimensional Bayesian Optimization with Gaussian Processes, in particular the selection of a kernel. It argues that the poor performance of the standard SE kernel in high dimensions stems from the vanishing gradient problem, and provides theoretical and experimental evidence to this effect. It then proposes a method to initialize the parameters of the SE kernel (setting the length-scale parameters to $O(\sqrt{d})$) to tackle this problem. Experimental results on synthetic and real datasets show that this initialization brings up the performance of the SE kernel to be comparable to the state-of-the-art.

### Strengths
- The paper is well-written, clear, and straightforward to understand.
- Hyperparameter optimization is a problem with wide-ranging applications.
- As far as I know, the consideration of the gradient vanishing problem for Bayesian Optimization with Gaussian Processes is novel. It is quite interesting, and indicates that it might be a pertinent direction of analysis for other applications due to the popularity of gradient descent.
- The paper provides both experimental and learning theoretic evidence of the gradient vanishing problem as the problem dimension increases.
- The proposed fix brings up the performance of standard BO with GPs and classic SE kernel to be comparable to the state-of-the art, including on NAS201.

### Weaknesses
- The dimensionality of the problems in the experiments are fairly high, but some problems in neural architecture search have even higher dimensionality.
- The proposed method does not outperform the baselines. However, I don't see this as a major weakness since I feel the main message of the paper is not the proposed method, but the importance of considering the gradient vanishing problem.

### Questions
Have you done experiments on problems with even higher dimensions, e.g. in the thousands?

### Soundness
4

### Presentation
3

### Contribution
3
