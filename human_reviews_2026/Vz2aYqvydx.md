# Cooperative variance estimation and Bayesian neural networks disentangle aleatoric and epistemic uncertainties

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Real-world data contains aleatoric uncertainty - irreducible noise arising from imperfect measurements or from incomplete knowledge about the data generation process. Mean variance estimation (MVE) networks can learn this type of uncertainty but require ad-hoc regularization strategies to avoid overfitting and are unable to predict epistemic uncertainty (model uncertainty). Conversely, Bayesian neural networks predict epistemic uncertainty but are notoriously difficult to train due to the approximate nature of Bayesian inference. We propose to cooperatively train a variance network with a Bayesian neural network and empirically demonstrate that the resulting model disentangles aleatoric and epistemic uncertainties while improving the mean estimation. We demonstrate the effectiveness and scalability of this method across a diverse range of datasets, including a time-dependent heteroscedastic regression dataset we created where the aleatoric uncertainty is known. The proposed method is straightforward to implement, robust, and adaptable to various model architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on Bayesian heteroscedastic regression with the goal to disentangle the estimate for the variance (aleatoric) and the model uncertainty. The proposed method consists of three steps in which first the mean is estimated and subsequently the variance and the BNN are updated iteratively for K steps. The marginal likelihood is approximated with the MAP estimate of a Gaussian likelihood and a Gaussian prior. The main novelty of the method lies in the loss function for the variance estimation that is based on the squared residuals, which the authors show are distributed according to a Gamma distribution.

### Strengths
- The proposed approach is evaluated on a large range of tabular and image regression datasets.
- An interesting synthetic dataset based on plasticity laws is introduced, however, it is unclear if this is a contribution of the submitted paper, or of Anonymous (2025).

### Weaknesses
- The authors write that “no current method achieves reliable uncertainty disentanglement”, referring to Amini et al. (2020) and Immer et al. (2023) by citing Mucsányi et al. (2024). After briefly checking the latter paper, it appears, however, that it does not cite Amini and Immer. Thus, this statement is incorrect in this context. The same argument applies for the statement made at the end of Sec. 3.2, which is confusing. Immer et al. (2023) estimate both the aleatoric and epistemic uncertainty without introducing a new hyperparameter. This approach is also omitted as a baseline although it achieves stronger results on the UCI regression tasks, e.g., on concrete, energy, boston, wine, and yacht.
- The formal justification of the method is weak (see questions). 
- Seitzer et al should be cited after Eq. (3).
- In Sec 3.3.1 the authors should state a formal proposition or lemma that they proof.

### Questions
What theoretical advantage is there to train three instead of a single network? Since the authors advocate to disentangle the networks for the two parameters (mean and variance), how does this justify having to learn two output parameters for the variance network alone? Does the corresponding loss have desirable properties?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel approach to separate aleatoric and epistemic uncertainty in MVE networks. The method proceeds in three stages: first train the network to estimate the mean; then train a separate variance network to model the aleatoric (data) uncertainty; and finally train a Bayesian neural network to estimate epistemic (model) uncertainty. The authors demonstrate the approach on a toy heteroscedastic regression problem, several UCI benchmark datasets, and large-scale image regression tasks.

### Strengths
1. The method is straightforward to implement and scalable to different architectures.

2. The paper is well-written and easy to follow.

### Weaknesses
1. The paper introduces no novel theoretical contribution. The proof of assumption 3.1 is a known result. 

2. The experimental results are confusingly presented. It is unclear whether the values in parentheses in Table 1 represent standard deviations. If so, the reported overlaps between methods suggest that performance differences are not statistically significant, calling into question the claimed improvements. Additionally, some acronyms in the results are undefined such as TLL.

3. The model architectures (two hidden layers with 256 neurons each) seem prone to overfitting, particularly on small or low-complexity datasets such as the toy sinusoidal example. This raises doubts about the robustness and generalizability of the proposed method.

4. The paper only addresses Gaussian (homoscedastic) aleatoric noise and evaluates a single type of neural architecture. This narrow focus limits the applicability and generality of the proposed uncertainty separation compared to related works [1, 2].

5. Several plots in Figure 2 and 5 are missing lines from the legend. These presentation issues make it difficult to interpret the qualitative results and reproduce the experiments.

6. The paper reports RMSE and TLL but omits standard uncertainty quality measures such as AUROC for out-of-distribution detection, limiting interpretability of the claimed uncertainty improvements.


[1] Berry, Lucas, and David Meger. "Normalizing flow ensembles for rich aleatoric and epistemic uncertainty modeling." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.

[2] Berry, Lucas, and David Meger. "Efficient epistemic uncertainty estimation in regression ensemble models using pairwise-distance estimators." arXiv preprint arXiv:2308.13498 (2023).

### Questions
1. Why was method not applied to Ensembles (MVE)? I imagine this would work as it did for MC Dropout.

2. Do you have the RMSE as a value instead of percentage? It would be nice to see this in the paper, maybe in the appendix.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The submission suggests novel procedural improvements on how to estimate aleatoric and epistemic uncertainty for regression by modifying training algorithms of Bayesian Neural Networks (BNN). Extensive empirical experiments are presented. These procedural improvements sometimes lead to significantly better performance than plain BNNs.

These procedural improvements are:

* A: Step 1: a MAP as a mean predictor, then Step 2: fit the aleatoric uncertainty to the residuals (keeping the mean from Step 1 fixed for computing these residuals). Then Step 3: train a BNN with fixed aleatoric uncertainty from step 2. 
* B: They modify Step 2 of A by using a Gamma distribution
* C: They modify Step 2 of A by using a more restricted (i.e., smaller) architecture for Step 2 than for the other steps.
* D: In step 3, they warm-start the BNN with the MAP from step A (not novel, but a reasonable thing to do)

### Strengths
Evaluating the methods on 18 datasets (including multiple real-world datasets) and adding some ablation studies is very valuable. Not only is it interesting to see the performance of your procedural improvements, but for me, it was also interesting to see the performance comparison between multiple other methods.

The improvement achieved with your procedural improvements is very impressive for some of the datasets.

Uncertainty quantification is an important problem for many applications. In particular, epistemic and aleatoric uncertainty.

Training BNNs can be very challenging, so improving their training with some procedural improvements is very valuable. 

The material plasticity datasets with known aleatoric uncertainty could be very valuable, also for benchmarking other methods that disentangle aleatoric and epistemic uncertainty for regression. Usually, real-world datasets are more important than synthetic ones, but this one seems like a realistic (practical?) simulated dataset. I have no expertise in material science, but at first sight it seems legit. Do any of the other reviewers better understand the relevance of this dataset?

I like Figure 2. In particular, the Wasserstein distance between the true aleatoric and the estimated aleatoric. This is something that wouldn’t be possible for most real-world datasets.

### Weaknesses
1. The performance of the results varies quite a lot. While for some combinations of datasets and models, your procedural improvements improve the performance, for others, your procedural improvements worsen the performance. Overall, I think your procedural improvements have, on average, a rather positive impact, and therefore I think it is a very valuable contribution, but I am not sure if it meets the bar of ICLR. My impression is that for the materials, your procedural improvements are very consistently helpful; for the UCI datasets, the procedural improvements are, on average, more helpful than harmful, but for the deep learning image datasets, the results are much harder to interpret. For the image regression datasets, the classical ensembles are also very strong, and often better than your BNNs.

2. Metrics: Why do you show different metrics for different datasets? You should also show TLL (and VLL) for the deep learning datasets (in the appendix). You should also show interval length and coverage for the UCI datasets (in the appendix). Pinball loss would also be an interesting addition of another suitable metric, but your choice of metrics is already OK if you apply them across all datasets. Maybe you should also show the test interval length? (The validation coverage is equal for all methods due to calibration on the validation dataset?) I think the test interval lengths when calibration on the validation dataset would be interesting, as this is what you would obtain in practice, but also the test interval length when calibrated on the test dataset would be interesting, to make the numbers comparable. Otherwise, if one method has worse coverage but better length, they are hard to compare. But I think I see your point that the test interval lengths might be hard to compare, as the test coverages of different methods vary a lot.

3. Calibration: In practice, uncertainty should (almost) always be calibrated on the validation dataset or some separate calibration dataset. The current version of the submission is quite unclear about when you are calibrating your uncertainties. Please be very precise about this in the revision. In the main paper, you never mention any form of calibration. In Eq.~(33) on page 17 in the appendix, you mention that you are using a constant additive calibration constant that you calibrate on the validation dataset. However, it seems as if you only calibrate your uncertainty for the coverage/length metrics? Are you using the uncalibrated uncertainties for the TLL experiments? How do you choose κ? κ strongly affects the scale of the epistemic uncertainty. Especially if you don’t do any hyperparameter optimization (HPO), you really should use some calibration. E.g., you can do classical calibration (https://arxiv.org/abs/1706.04599, https://arxiv.org/abs/1807.00263, https://www.mdpi.com/1424-8220/22/15/5540) by multiplying your total predictive variance from Eq. (11) by a constant $c$, either such that it maximizes the VLL (Validation Log Likelihood) or such that the 95% predictive interval covers 95% of the validation data. (Alternatively, you could try something more recent like https://arxiv.org/abs/2507.08150 where you use  2 calibration constants: multiply $c_1$ on aleatoric variance and $c_2$ on the epistemic variance, and optimize these two constants by maximizing the VLL (constrained on your predictive interval covering $1-\alpha$ of the validation data).) Maybe better calibration (e.g., multiplicative instead of additive, or 2 constants instead of 1) would also improve your test coverage? Multiplicative calibration has the advantage that it is compatible with metrics such as TLL (scaling up the standard deviation of a Gaussian by a constant $c$ still results in a Gaussian). I think there is quite a high chance that the TLL can be improved by calibrating via the VLL. Maybe this would mitigate the seemingly random differences in performance rankings of the methods across different datasets.

4. Math: When I try to derive a formula for the NLL of a Gamma Distribution, I obtain a different result. Are you also using this density for the Gamma distribution: $f(x)={\frac {\lambda ^{\alpha }}{\Gamma (\alpha )}}x^{\alpha -1}e^{-\lambda x}$? When I compute the logarithm of this I obtain  α Log[λ] -Log[Gamma[α])] + (α - 1)Log[x] - λx, which I have checked with WolframAlpha: https://www.wolframalpha.com/input?i=Log%5B%28%28%CE%BB%5E%CE%B1%2FGamma%5B%CE%B1%5D%29+x%5E%28%CE%B1+-+1%29%29%2FE%5E%28%CE%BB+x%29%5D+-+%28++%CE%B1+Log%5B%CE%BB%5D+-Log%5BGamma%5B%CE%B1%5D%29%5D+%2B+%28%CE%B1+-+1%29Log%5Bx%5D+-+%CE%BBx++%29 
I don’t understand, for example, why you divide by $r_n$ in the last term instead of multiplying by $r_n$? The first term also seems incorrect to me. Can you please clarify?

5. Why don’t you apply your procedural improvements to deep ensembles as well? They can also be seen as a simple approximation of BNNs.

6. There are also some issues with the presentation, which is sometimes not clear (see Questions for detailed feedback).

### Questions
Q1: Lines 36-37: The sentence “In such cases, an outcome with good mean performance but large aleatoric uncertainty may be unacceptable.” feels confusing. Maybe write something weaker, like “undesirable” instead of "unacceptable"? And/or maybe formulate it the other way around: “In such cases, an outcome with bad mean performance but low aleatoric and low epistemic uncertainty might be unacceptable.”? What exactly do you mean by “mean performance”? Or maybe skip the sentence?

Q2: Lines 37-38: The next sentence, “Therefore, the principle of reducing epistemic uncertainty behind active learning or decision-making needs to be balanced by the respective prediction of aleatoric uncertainty.” feels even more confusing. How does it need to be balanced? What does this mean? Or maybe skip the sentence?

Q3: Lines 37-38: Why not cite (Lakshminarayanan et al., 2017) for deep ensembling?

Q4: Lines 128-129: Maybe the sentence: “Still, a recent investigation (Mucsányi et al., 2024) has shown that no current method achieves reliable uncertainty disentanglement.” should be weakened? Maybe “suggests” instead of “has shown”. Is it a theoretical result or an empirical one? “No current method” sounds very hard to prove.

Q5: Section 3.1: I think the more typical notation would switch the variable names $s$ and $\varepsilon$.

Q6: Line 162: “with of” is a typo? Should be “of”?

Q7: Line 186: Important question: In Step 2, when you are using the fixed mean from Step 3, are you using the MAP or the mean posterior from Step 3?.

Q8: Line 191: Instead of $\theta^*$, the estimated posterior distribution over $\theta$ is meant? Or actually only the MAP?

Q9: Line 194: Concretely, you are setting constant=1 for Step 1?

Q10: Line 200: Instead of writing: “There is, however, an important detail” write "There are, however, multiple important details"

Q11: Line 203: Instead of writing “the variance network outputs the residual” I would write something like, “the variance network estimates the distribution of the squared residuals” or “the variance network gets the squared residuals […] as training-labels”.

Q12: Lines 203-204:The Gamma distribution falls a bit from the sky. I would add “, as explained in Lemma 3.1.” at the end of the sentence.

Q13: Lines 208-210: The structure is very confusing. A proof should always follow a lemma/theorem/proposition/corollary. Therefore, I would write “Lemma 3.1. We aim to demonstrate that from Assumption 3.1 and for y following a Gaussian distribution, the squared residual r = (μ(x; θ) − y)² follows a Gamma distribution.
Proof. Since [...]”

Q14: Eq.~(6-7) seems mathematically wrong to me. Please clarify.

Q15: Lines 256-257: For “From experience, preconditioned Stochastic Gradient Langevin Dynamics (pSGLD) (Li et al., 2016) is expected to be a good choice for this BNN because the aleatoric uncertainty is fixed, and the likelihood and prior are both Gaussian, leading to a Gaussian posterior.” a short explanation of “preconditioned” might be nice if you have space left? More importantly, what do you mean by Gaussian posterior? Gaussian predictive distribution (or the posterior of µ is Gaussian)? Or the posterior on the parameters is Gaussian? Why is this Gaussian? Do you mean the true posterior of the theoretical BNN prior or the posterior approximated by pSGLD?

Q16: Line 301: The abbreviation "TLL" is only introduced in the appendix.

Q17: Line 318: In the last column, “-0.64 (0.28)” should be bold.

Q18: Line 323: “The conclusions are similar to those obtained from the UCI regression datasets.” is too simplified, I think. The results are less clear there, I would say.

Q19: Line 332: I don’t agree with the end of the sentence: “This is very encouraging because training is faster for  VeBNN (pSGLD) when compared to MVE (Ensembles), as the former collects samples in the same training procedure, while ensembling requires collecting one sample per training of a single MVE (i.e., training restarts many times).”. Deep Ensembles can be parallelized much more easily, as each ensemble member can be trained in parallel. For Deep Ensembles, usually fewer epochs are needed compared to pSGLD. Therefore, I definitely would not say that pSGLD is faster. I think Deep Ensembles is faster if you parallelize. Maybe you can say that pSGLD has lower computational costs, but even there, I am not sure. VeBNN also adds sequential computational costs, which cannot be parallelized. This also makes it slower, especially when K increases.

Q20: Line 368-369: In “(upper column to bottom column)”, it should be “row” instead of “column”.

Q21: Lines 403-405: I have no idea what “for 800” refers to in “To further explore this, we show the trajectory of all essential components of VeBNN with respect to the iteration K from 1 to 5 for 800 in Figure 18.”. Can you clarify?

Q22: Lines 483-484: Important question: What do you mean by “As a  Bayesian method, it does not require validation data.”? Aren’t you doing HPO on the validation dataset? Aren’t you calibrating your uncertainty on the calibration dataset? I personally think that for basically any ML task, a validation dataset is strongly recommended. Your method has many hyperparameters, so I really think you should use a validation dataset, and I think, in fact, you are using one?

Q23: Line 915: “Should it be “predictive interval” instead of “confidence interval'? Also later, when you write “predictive confidence interval”, most people would call this a “predictive interval”. I think “confidence intervals” are about inference rather than prediction.

Q24: Figure 14: I think the results are very inconclusive, as the test coverage varies a lot, and not using the procedural improvement consistently performs better than using your procedural improvements in this metric, except for Aerial.

Q25: Where do you report the TLL for the image regression datasets?

Q26: Table 3: “For each problem, we highlight the best method in green, but we  also highlight in yellow methods with similar performance.” How do you define “best method” and how do you define “similar performance”? Do you have at least some vague rule of thumb to describe this?

Q27: Table 3: You should mention here that you are calibrating for validation coverage here.

Q28: Line 1459: training deep ensembles for 20k epochs is a lot, I think. How do you avoid overfitting? Do you select for each ensemble member the epoch where it had the lowest validation LL? Usually, a quite small number of epochs is selected, I guess?

Q29: Lines 1464-1465: “10000 epochs with an early stopping of 100 epochs”. What does “with an early stopping of 100 epochs” mean? Patience?

Q30: Line 1469: How do you calibrate your uncertainty if you use “all training data” for retraining?

Q31: Do you use any form of early stopping for the Image Regression Dataset?

My current score is ~5, and I am willing to change my score based on your answers. E.g., if you add the missing metrics.

### Soundness
2

### Presentation
2

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
This paper presents a cooperative training strategy to jointly estimate *aleatoric* and *epistemic* uncertainties in regression problems. The authors argue that mean–variance networks (MVEs) poorly estimate heteroscedastic variance and that standard Bayesian neural networks (BNNs) are unstable when directly modeling aleatoric uncertainty.

The proposed method, **VeBNN (Variance-estimating BNN)**, introduces a two-stage iterative scheme:
1. Train a deterministic mean network.
2. Train a **variance network** on squared residuals using a **Gamma likelihood**, theoretically justified under Gaussian noise.
3. Freeze the variance and perform Bayesian sampling (via pSGLD) to model epistemic uncertainty.

This cooperative process repeats for \(K\) iterations, selecting the best parameters through an estimated log marginal likelihood. The Gamma residual modeling provides numerical stability and better calibration.

Experiments include **UCI regression benchmarks**, **image-based regression under distribution shift**, and a **new physics-based dataset (material plasticity)** with *known aleatoric uncertainty*, enabling quantitative disentanglement analysis.

### Strengths
- **Mathematical rigor:** Correct derivation of Gamma residual likelihood; strong connection to heteroscedastic regression theory.
- **Stability:** Cooperative schedule avoids the gradient imbalance issues of Gaussian NLL.
- **Empirical depth:** Includes a dataset with known aleatoric uncertainty, allowing quantitative disentanglement.
- **Reproducibility:** Clear training details and ablations; implementation reproducible.
- **Trustworthy AI relevance:** Promotes calibrated and interpretable uncertainty estimation, valuable for safe AI systems.

### Weaknesses
1. **Assumption of unimodality:** The Gamma residual model presumes Gaussian, symmetric residuals. It may not generalize to multi-modal or skewed \(p(y|x)\).  
   *Suggestion:* Add stress tests under Student-t or mixture noise.

2. **Incomplete baselines:** The study omits Mixture Density Networks (MDN) and Mixture-of-Experts (MoE), which directly model multi-modal uncertainties.

3. **No convergence analysis:** The cooperative iterations (\(K\)) are empirically motivated but lack theoretical justification.

4. **Limited empirical diversity:** Datasets are appropriate but not exhaustive—no explicit OOD or multi-modal scenarios tested.

5. **Moderate novelty:** The work refines existing ideas rather than introducing a new learning paradigm.

### Questions
1. How does the Gamma residual assumption behave under **non-Gaussian or multi-modal noise**?
2. Do multiple cooperative cycles (\(K>1\)) provably improve marginal likelihood, or is convergence purely empirical?
3. Would the results hold using other inference methods (e.g., VI, SGHMC) instead of pSGLD?
4. Have you measured **epistemic calibration** or coverage (ECE, TLL) under domain shift?
5. Will the **plasticity dataset** with known aleatoric uncertainty be released publicly for reproducibility?

### Soundness
2

### Presentation
3

### Contribution
3
