# Conformal Prediction for Long-Tailed Classification

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Many real-world classification problems, such as plant identification, have extremely long-tailed class distributions. In order for prediction sets to be useful in such settings, they should (i) provide good class-conditional coverage, ensuring that rare classes are not systematically omitted from the prediction sets, and (ii) be a reasonable size, allowing users to easily verify candidate labels. Unfortunately, existing conformal prediction methods, when applied to the long-tailed setting, force practitioners to make a binary choice between small sets with poor class-conditional coverage or sets that have very good class-conditional coverage but are extremely large. We propose methods with marginal coverage guarantees that smoothly trade off set size and class-conditional coverage. First, we introduce a new conformal score function called prevalence-adjusted softmax that optimizes for macro-coverage, defined as the average class-conditional coverage across classes. Second, we propose a new procedure that interpolates between marginal and class-conditional conformal prediction by linearly interpolating their conformal score thresholds. We demonstrate our methods on Pl@ntNet-300K and iNaturalist-2018, two long-tailed image datasets with 1,081 and 8,142 classes, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of constructing useful prediction sets using conformal prediction (CP) in long-tailed classification settings, such as plant identification, where rare classes are underrepresented. Standard CP achieves marginal coverage but often fails to cover rare classes adequately, while class-conditional CP methods produce excessively large sets. The authors propose two approaches: (1) a new conformal score function called prevalence-adjusted softmax (PAS) and its weighted variant (WPAS) to target macro-coverage (the unweighted average of class-conditional coverages), derived from oracle-optimal sets that balance set size and coverage; (2) INTERP-Q, a simple interpolation between standard and classwise CP quantiles to trade off set size and class-conditional coverage. They evaluate these methods on long-tailed image datasets, demonstrating improved trade-offs compared to baselines.

### Strengths
The paper introduces novel conformal score functions (PAS and WPAS) specifically tailored for macro-coverage in long-tailed settings, grounded in theoretical derivations of oracle-optimal prediction sets. This extends prior work on CP beyond marginal or basic class-conditional guarantees, addressing a practical gap in applications like biodiversity monitoring where rare classes are critical. The paper is clear, with well-defined notation, algorithms, and metrics, making the contributions accessible.

### Weaknesses
The interpolation in INTERP-Q is simple but lacks deeper analysis of why linear weighting works well, and why it can only guarantee a conservative coverage bound ($1-2\alpha$) while being close to $1-\alpha$ in practice. 

Additionally, comparisons could include more recent long-tail methods (e.g., Ding et al. (2023)).

### Questions
1. Can this method be combined with other scores, such as RAPS and SAPS?
2. In figure 2, there are cases where test label distribution is different from training and validation distribution, how this affects the results, especially considering PAS that is based on class normalization.
3. Can this method be extended to regression problems with class imbalance, where the continuous target variable has a highly skewed distribution with certain value ranges, much more frequent than others?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies conformal prediction for very imbalanced multi class problems where a few classes have many examples and many classes have very few. The goal is to produce prediction sets that keep marginal coverage guarantees while improving coverage for rare classes without exploding set size. The authors offer two main ideas. First, a new score called prevalence adjusted softmax and its weighted version that aim directly at macro coverage, with an option to upweight classes of interest such as endangered species. Second, an interpolation procedure called INTERP Q that blends classwise and standard conformal thresholds to trade off set size and class conditional coverage, with a conservative marginal guarantee. They test on Pl@ntNet 300K and iNaturalist 2018 and report better coverage for tail classes at small or moderate set sizes, plus a study with simple human decision models.

### Strengths
Targeting macro coverage with a simple change to the score is a neat idea. It connects the oracle form of the optimal set for macro coverage to a practical score based on p hat of y given x divided by the estimated prevalence. The weighted version lets users push coverage toward special subsets like at risk species.

The paper is easy to follow. The problem is well motivated with plant identification. The two approaches are separated and labeled. Table 1 is a good map of methods and guarantees.

Selective set prediction in long tailed regimes is common in biodiversity and medicine and also in open world recognition. A method that improves tail coverage while keeping sets short is directly useful for real labeling workflows and can slow collapse of rare labels in human in the loop systems.

### Weaknesses
PAS relies on p hat of y given x and an estimate of label prevalence. In real systems there is often label shift between train, calibration, and test. The paper does not test robustness under such shift, even though label shift directly changes the p of y term that PAS divides by.

The 1 minus 2 alpha lower bound is likely conservative, as the authors note, but the paper does not quantify the realized marginal coverage gap across settings or give a simple correction to hit a target level.

Most results use softmax scores from a standard ResNet trained with cross entropy, with one mention of focal loss in the appendix. Since the approach is driven by score quality, the work would benefit from a broader check across stronger long tail learners such as logit adjusted training and from alternative scores such as APS or label ranking scores, even if set sizes grow

The human models are an expert verifier and a random guesser, plus mixtures. That is a useful first look, but real users are neither.

### Questions
How sensitive is PAS to misspecified prevalence. If the true p of y differs from the training estimate, can you still expect macro coverage gains.

Can you provide a practical scheme to choose tau on the calibration fold to meet a specific marginal coverage while improving class conditional coverage.

Have you tried teaching a small correctness predictor on the calibration set and using that as the score inside PAS or as a rank corrector.

Macro coverage can look good while a handful of classes are still far below target. Can you add plots of the full distribution of per class coverage, not just the fraction below fifty percent.

For a practitioner who wants sets of average size at most three while lifting average coverage of at risk species above a target, can you give a small recipe that picks alpha and lambda and, for INTERP Q, tau.

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
The paper applies conformal prediction (CP) to an extremely long-tailed classification setting. The standard CP tends to ensure coverage of common classes and under/miss coverage of the rare ones, whereas classwise CP covers each class but produces unusable prediction sets. The authors address this challenge by 2 approaches: a) PAS / WPAS -> the idea here is to, instead of trusting the raw model probabilities, recalibrate label scores by their rarity, so that rare classes get a boost. The authors claim this results in smaller prediction sets compared to classwise CP, and is also fairer to rare classes compared to standard CP. b) INTERP-Q -> instead of score recalibration, this approach uses a controllable parameter that provides a cutoff threshold that balances between standard CP and classwise CP thresholds. It gives the user a controllable knob that can decide how much to protect tail classes versus how large a prediction set can be.
Empirically, they compare against standard baselines and the evidence supports their claims.

### Strengths
Strength

a) The main paper is well motivated, mostly clear and easy to follow.

b) It tackles conformal prediction in the extreme long-tailed scenario, which is practically important.

c) The class coverage vs prediction set size tradeoff as a problem formulation itself seems novel. The two proposed approaches also appear reasonably original. 

d) Empirical studies are convincing, and their human decision maker simulation experiment seems interesting to me.

### Weaknesses
Weaknesses

a) I could understand the working of PAS/WPAS and INTERP-Q, but I couldn't clearly find the motivation for utilizing either/or both of them. The two methods seem to address different parts of the pipeline, but the paper does not clearly explain a practical guideline when a practitioner should pick PAS/WPAS, INTERP-Q, or use them together.

b) The experimental details in the appendix mention utilizing a truncated version with n-core filtering with n = 101. I am curious: doesn't this contradict the problem the authors are trying to solve? The paper is motivated by the extreme long tail, but the analysis is done only on this truncated subset. It would be helpful to clarify whether the main conclusions still hold for the truly rare classes that were filtered out.

c) Based on my understanding, PAS / WPAS is motivated by the hypothesis that reweighting by class prevalence is optimal for trading off average per-class coverage and set size. However, in the implementation, PAS simply recalibrates standard split CP scores, which only guarantees marginal coverage, not class-conditional coverage. So the claim that there are fewer under-covered long-tailed classes seems to be empirical rather than a strict guarantee to me. Similarly, INTERP-Q appears to be only loosely bound to the standard finite-sample marginal guarantee. Neither approach provides a strict class-conditional guarantee, which makes the strength of claims less clear.

d) The paper only evaluates using splits where calibration and test share the same long-tailed label frequencies. It would be interesting to see whether the proposed methods hold up when the test distribution differs from calibration.  I believe stress-testing robustness to even basic shifts would better support the practicality claims of the paper.

e) Comparison with standard CP, classwise CP, and the clustered variant is good. But I would like to see how the method fares against some recent methods that seem to have similar motivation/methodologies [1][2].


[1] Liu, Shuqi & Huang, Jianguo & Ong, Luke. (2025). Conformal Prediction Meets Long-tail Classification.

[2] Yuanjie Shi, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao Doppa, and Yan Yan. (2024). Conformal prediction for class-wise coverage via augmented label rank calibration (RC3P). NeurIPS 2024.

### Questions
Please refer weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing conformal prediction methods often trade off between poor class-conditional coverage and overly large prediction sets. This paper proposes a new non-conformity score, PAS (and its weighted variant WPAS), which aims to interpolate between marginal and class-conditional conformal prediction. The goal is to achieve better coverage–efficiency balance under long-tailed label distributions.

### Strengths
1.	Addresses the practically relevant problem of conformal prediction under long-tailed label distributions.

2.	The proposed PAS/WPAS scores are simple and easy to implement.

3.	The experiments offer preliminary evidence that the proposed method improves coverage fairness under long tailed setting.

### Weaknesses
1.	Truncated dataset setup: The test datasets are balanced by truncating rare classes and retaining those with more than 100 samples per class. The choice of this threshold is not explained. Why 100? In more realistic scenarios where both calibration and test sets are long tailed, how would the proposed method perform?

2.	Motivation: The new non-conformity score (PAS) is motivated by an oracle analysis showing that the optimal set depends on p(y|x)/p(y), but this only characterizes an ideal solution rather than a provably better practical formulation. Overall, the motivation appears ad-hoc, and it remains unclear what concrete limitation of existing scores (e.g., APS, LAC) PAS resolves under long-tailed distributions. 

3.	Bassline: The paper does not include [1], which also targets class-conditional coverage under imbalanced settings, which appears highly relevant to this work. In addition, standard non-conformity scores such as APS [2] and LAC [3] are omitted. 

4.	Score comparison: Since APS emphasizes X-conditional coverage and LAC minimizes the expected set size, including them would provide a more informative comparison of the adaptiveness–efficiency tradeoff that PAS/WPAS aims to improve. It would also be helpful to clarify whether a weighted combination of APS and LAC could already achieve a similar trade-off, and how PAS/INTERP-Q compares in that respect.

5.	Evaluation Metric: The study reports FracBelow50% and UCG but omits the more intuitive Under-Coverage Ratio (UCR) used in [1], which reflects the fraction of classes that fail to meet the target coverage and would provide a more informative evaluation.

[1] Yuanjie Shi, Subhankar Ghosh, Taha Belkhouja, Jana Doppa, and Yan Yan. Conformal prediction for class-wise coverage via augmented label rank calibration. NeurIPS 2024.

[2] Yaniv Romano, Matteo Sesia, and Emmanuel J. Candes. Classification with valid and adaptive coverage. NeurIPS, 2020.

[3] Mauricio Sadinle, Jing Lei, and Larry Wasserman. Least ambiguous set-valued classifiers with bounded error levels. J. Amer. Statist. Assoc 2019.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
