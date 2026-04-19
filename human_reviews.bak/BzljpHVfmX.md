# An Asymptotic Theory of Random Search for Hyperparameters in Deep Learning

- Decision: Reject
- Scores: 3, 5, 5, 3

## Abstract
Scale is essential in modern deep learning; however, greater scale brings a greater need to make experiments efficient. Often, most of the effort is spent finding good hyperparameters, so we should consider exactly how much to spend searching for them&mdash;unfortunately this requires a better understanding of hyperparameter search, and how it converges, than we currently have. An emerging approach to such questions is *the tuning curve*, or the test score as a function of tuning effort. In theory, the tuning curve predicts how the score will increase as search continues; in practice, current estimators use nonparametric assumptions that, while robust, can not extrapolate beyond the current search step. Such extrapolation requires stronger assumptions&mdash;realistic assumptions designed for hyperparameter tuning. Thus, we derive an asymptotic theory of random search. Its central result is a new limit theorem that explains random search in terms of four interpretable quantities: the effective number of hyperparameters, the variance due to random seeds, the concentration of probability around the optimum, and the best hyperparameters' performance. These four quantities parametrize a new probability distribution, *the noisy quadratic*, which characterizes the behavior of random search. We test our theory against three practical deep learning scenarios, including pretraining in vision and fine-tuning in language. Based on 1,024 iterations of search in each, we confirm our theory achieves excellent fit. Using the theory, we construct the first confidence bands that extrapolate the tuning curve. Moreover, once fitted, each parameter of the noisy quadratic answers an important question&mdash;such as what is the best possible performance. So others may use these tools in their research, we make them available at (URL redacted).

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, the probability function relationship between random search and model performance is analyzed theoretically, and the corresponding parameterized distribution is designed. The effect of parameter estimation of this distribution is better than that of nonparametric estimation. With this distribution, you can have a good understanding of the impact of parameter adjustment under the task. It helps researchers to evaluate the self-designed method and modify the corresponding strategy without eliminating the influence of parameter adjustment.

### Strengths
In this paper, a new parameter distribution is proposed, which theoretically fits the model performance changes under random search parameters. The fitted curves can help researchers to do the next step, such as judging whether their model can solve the task based on the best predictions.

The three parameters in the new parameter distribution correspond to the actual parameter meanings, and the influence of the parameters can be roughly understood directly through the estimated distribution.

### Weaknesses
The task model in the experiment is slightly thin and not comprehensive. For example, in line 120, the authors mentioned "Which architecture", but in experiments, no choice of ResNet has been considered. The authors use ResNet18 directly.

The new parameter distribution proposed by the author is compared only with the non-parametric distribution, not with other simple parameter distributions, whether the simple parameter model is sufficient to approximate the ground truth.

The author claims to propose an asymptotic theory of random search, but in fact the author relies only on analysis rather than proof to provide an approximation, without any theoretical guarantee like asymptotic convergence or probably approximately correct (PAC) learning theory.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper describes a theory of tuning curves for random hyper-parameter tuning near the optimum under a smoothness assumption.
A parametric form of the tuning curve is described based on a novel description of the distribution of outcomes, and confirmed on three deep learning models.

### Strengths
Hyper-parameter tuning is still a critical aspect of deep learning research and practice, and is understudied in the ML community.
The paper proposes a concise methodology to understand the asymptotics of randomized search, with clear predictive capabilities.

The paper is well structured and clearly written.

### Weaknesses
I'm skeptical of a core assumption of the paper, which is whether the asymptotic regime is relevant in practice. If the function is  smooth, and well-approximated by a quadratic locally, then clearly random search is not the right tool. Using a GP would provide immense benefits if the local smoothness assumption holds, and the search is "near the optimum".
The main reason that random search is so successful is that in practice, many areas of the search space are not smooth, and jumps are common.
I'm quite surprised by how smooth the tuning curves in figure 1 and 3 are, and they are very unlike tuning curves I have seen with random search, which often stay constant for a long time, and don't progress for 10 or more iterations.

This might be due to the architectures used being extremely well understood, and, potentially, as the authors point out, easy to tune.
I would be quite curious to see if these results hold when tuning an MLP on, say the AutoML benchmark, or TabZilla.

Given the smoothness observed in the experiments given in this paper, I would be very interested to see how the tuning curve for TPE or a GP would look in these cases.

I did not study all the mathematics in detail. Given the assumptions the formulation seems reasonable; my main concern is about the assumptions and practical utility of the tool.

### Questions
How are the empirical confidence bands estimated in Figures 1 and 3?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a simplified but accurate statistical model of hyperparameter tuning under random search in the "asymptotic" regime. Here the term "asymptotic" refers to hyperparameter settings which are "close" to optimal, for which a second-order taylor expansion around the set of optimal hyperparameters is informative. Fore this regime the authors heuristically propose the "quadratic" distribution to model the performance of random search in expectation over training randomness. To model the noisy effect from training randomness the authors propose a homoskedastic additive gaussian noise process which results in the "noisy quadratic" distribution. Over a variety of tasks the authors demonstrate the efficacy of this distribution for modeling random hyperparameter search.

### Strengths
The authors provide a clean and empirically compelling model of random hyperparameter search using a heuristic, first-principles based approach. The paper is written clearly, with a large amount of statistical and empirical validations.

### Weaknesses
It is quite unclear what the implications of these observations are. Also the framework only applies to random search as opposed to other randomized search methods such as Bayesian optimization. Currently the results seems like a few nice observations, but not a substantially impactful contribution.

### Questions
Is it possible to fit a model of H and perform a type of PCA procedure to determine the effective hyperparameters? Such an insight might help reduce the effective search space for certain classes of problems / hyperparameters. Are there any implications for other non-uniform random search methods such as Bayesian optimization? Or when doing muTransfer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors propose to parametrize the performance of random search (tuning curve) with a noisy quadratic distribution. The authors test the fit and extrapolation of the proposed work in three experimental settings with diverse deep learning models.

### Strengths
-

### Weaknesses
- No baselines are considered.
- Limited number of experiments conducted. To really validate the claims of the paper one must consider diverse search spaces and models.
- The code for the work is not provided.
- The related work section is outdated.

### Questions
- **Line 180, "Thus as the search continues, the region of relevant hyperparameters converges about the optimum"**

   I do not agree with the above statement, random search, as the name gives, samples hyperparameters randomly. It is not a model-based method that incorporates the results into it's sampling stategy. So the region of relevant hyperparameters is the same search space (except maybe what was sampled before), it is not constrained in any manner. Additionally, being close to the optimum, would require a very very large number of trials in a continuous search space of $D$ dimensions.

- **The work defines the asymptotic regime as the hyperparameters that we care about the most, those close to the optimum (Line 81).** 

   Looking at Figures 3 and 4, this perspective does not correspond to the explanation provided by the authors. For example, at the bottom of Figure 4 (the ResNet model), the region pointed out as the asymptotic regime in my perspective, would be somewhere at iteration 8-10. Random search there seems to be close to finding an optimum solution. While the asymptotic regime pointed out by the authors is around iteration 1-2.

- At the bottom of Figure 3 and Figure 4, did the authors order the random search trials by performance? Because the performance over the iterations seems to follow a power law. Given that it is random search, I would expect some flat regions given by hyperparameter configurations that are not optimal. Based on the figures it seems that the performance is constantly improving which is very surprising. The curve looks like a curve that is generated from training a model.

- Throughout the manuscript, the authors mention that they use 1024 iterations for each considered model/search space combination, however, on the plots the number of iterations is up to 100 for Figure 3 and up to 70 for Figure 4. Do the authors consider the number of repetitions too, how exactly is the number 1024 devised? How exactly are the 48 subsamples collected, part of the beginning of the "tuning curve" or randomly from the full data?

- **Line 502, "however, random search remains a strong baseline, with variants near state-of-the-art (Li et al.,2018;2020)."**

   The authors do not accurately reflect the current state of the domain. HyperBand and ASHA are not the state-of-the-art in multi-fidelity hyperparameter optimization. There have been several advancements that combined the schedule of HyperBand with model-based surrogates[1][2] and more recently, the current state-of-the-art [3][4] approaches that use an adaptive schedule with model-based algorithms. I would urge the authors to incorporate the provided citations into the manuscript to provide accurate information about the current state-of-the-art in multi-fidelity optimization.

- While the authors advocate the use of their proposed work with deep learning, the deep learning tasks are expensive and achieving $x$ trials is computationally demanding. In these scenarios, practitioners tend to use multi-fidelity based methods that are model-based. Random search is not a very promising algorithm.

- How many data points (HPO trials) are needed for the provided distribution to accurately reflect the tuning curve?

[1] Falkner, S., Klein, A., & Hutter, F. (2018, July). BOHB: Robust and efficient hyperparameter optimization at scale. In International conference on machine learning (pp. 1437-1446). PMLR.

[2] Awad, N., Mallik, N., & Hutter, F. DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization.

[3] Wistuba, M., Kadra, A., & Grabocka, J. (2022). Supervising the multi-fidelity race of hyperparameter configurations. Advances in Neural Information Processing Systems, 35, 13470-13484.

[4] Kadra, A., Janowski, M., Wistuba, M., & Grabocka, J. (2024). Scaling laws for hyperparameter optimization. Advances in Neural Information Processing Systems, 36.

### Soundness
2

### Presentation
2

### Contribution
1
