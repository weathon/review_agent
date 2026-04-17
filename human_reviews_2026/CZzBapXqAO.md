# Is it worth it to collect missing values?: The Missing Value Uncertainty Problem

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
In high-stakes domains like healthcare, operators often face the critical decision of whether to act on incomplete information or collecting missing values is likely to change the prediction Existing methods typically focus on imputing missing data or quantifying model uncertainty, but they do not directly assess the stability of a prediction if missing features were to be revealed. To address this gap, we introduce a framework for Missing Value Uncertainty (MVU), which is the distribution of predictions induced by incomplete inputs. We formalize the problem by defining hard confidence: the probability that a prediction will not change after collecting the missing data. We propose a novel Direct Missing Value (DMV) to efficiently estimate the MVU distribution, bypassing the need for expensive Monte Carlo sampling. Second, we introduce the Missing Value Calibration Error (MVCE), a new metric specifically designed to evaluate the calibration of hard confidence values, and a post-hoc calibration procedure to improve MVU estimation. We showcase our method and metric on synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This study formalises Missing Value Uncertainty (MVU), which is the epistemic uncertainty introduced by not observing all input features for a given sample at test time. They then propose Direct Missing Value (DMV) as an efficient way to estimate the MVU and introduce the Missing Value Calibration Error (MVCE) as a new metric to quantify the added value of collecting currently missing values. The MVCE is evaluated in a synthetic data experiment, whereas the DMV is compared to a diffusion model on CelebA and two simple baselines on MNIST, CIFAR10, and StarcraftCIFAR10.

### Strengths
The paper is well written and introduces a novel notion of quantifying the value of information of missing data at test time based on probabilistic predictors and hard voting.

### Weaknesses
**Novelty** : The authors seem to ignore a rich existing literature on active feature acquisition at test time such as Ma et al. (2019) or Li et al. (2021). These methods seem to address the same question posed by the authors: "Is it worth it to collect missing values (for this specific test sample)?" Existing methods need to be discussed in the paper and must be included as baselines in the experiments (or a strong case must be made why they in fact should not be included as baselines) to show the relativ merit of the proposed approach. 

Ma C et al. EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE. ICML. 2019.
Li Y et al. Active Feature Acquisition with Generative Surrogate Models. ICML. 2021. 

**Flexibility**: In the current version, the introduced methods only dicuss an all-or-nothing setting in which either no additional information is collected (sticking with the initial $X_O$) or all missing values are obtained (resulting in $X$). However, it may often be interesting to collect only the most informative one or two additional values or to sequentially obtain values. 

**Experiments**: The included experiments are limited to dropping continuous regions of common image datasets. Following previous literature like EDDI, experiments with additional modalities like tabular data would be beneficial. The authors also don't discuss how their derived consistency or MVCE should be used in practice to make a decision to collect additional data, and what benefit the user would obtain by doing so.

### Questions
I did not fully understand the methods used in the synthetic data experiment and Table 1, which don't seem to be described in the paper or appendix.

### Soundness
2

### Presentation
3

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
The paper aims to provide a tool to estimate, for each sample, the probability that a decision based on a probabilistic classifier would change if more data were available. This can guide decision-makers on whether additional data collection is worthwhile. The targeted uncertainty is a form of epistemic uncertainty, but orthogonal to the usual notion: rather than uncertainty due to model suboptimality (reducible by more training data), it captures uncertainty due to unobserved variables (reducible by completing missing data).
The authors define **Missing Value Uncertainty (MVU)** as the distribution of a model’s probabilistic predictions when only a subset of features is observed — i.e., the variability in predictions induced by the distribution of missing values. They propose an efficient estimator, the **Direct Missing Value estimator (DMV)**, which bypasses sampling in potentially high-dimensional spaces or imputation. In practice, MVU is modeled using a Dirichlet distribution.

They also introduce the notions of **hard prediction** (majority vote) and **hard confidence** (the proportion of predictions agreeing with the majority), along with a new evaluation metric, the **Missing Value Calibration Error (MVCE)**, which mirrors Expected Calibration Error (ECE) but measures how well hard confidence reflects the empirical probability of prediction changes between partial and complete data. A recalibration method tailored to this setting is also proposed.

Experiments on synthetic and image-based datasets compare DMV to sampling (via diffusion) and imputation baselines in terms of MVCE, as well as the recalibration quality.

### Strengths
**Novelty**: The authors introduce and formalize a new class of uncertainty quantification problems, addressing a question of clear practical relevance: whether one should collect additional data before making a decision. To my knowledge, no prior work has explicitly or rigorously formulated this problem, making this contribution useful and original.

**Clarity of formalization**: The underlying intuition is articulated very clearly, and the resulting problem formulation is precise and easy to follow.

**Efficient estimator**: The proposed estimator avoids the need for costly and difficult sampling of missing values given observed ones, offering a more elegant and practical solution for estimatingMVU.

### Weaknesses
**Clarity of the practical aspects** :  The paper lacks sufficient detail to fully understand how the proposed method is implemented in practice. This is an important weakness, as it prevents a thorough evaluation of the approach’s validity and effectiveness. Specific issues are detailed in the questions section.

**Evaluation of the MVU estimator**: Estimating MVU is a very challenging task—typically harder than learning the optimal probabilistic prediction under arbitrary missingness—because it requires estimating the full predictive distribution rather than just the conditional expectation. I am therefore wondering how well DMV can approximate this distribution, and the current experiments do not convince me that it is indeed a good estimator.

For example, Table 2 shows that DMV achieves an MVCE roughly comparable to a diffusion-based sampling baseline using only 3 samples (in 2 out of 3 tasks). It is questionable whether 3 samples suffice to accurately approximate the MVU distribution, which raises concerns about whether DMV captures the distribution’s quality well enough. 

Validating DMV on a controlled synthetic setting (e.g., a multivariate Gaussian) would allow comparison against a ground-truth baseline, exploring how performance varies with dimensionality, sample size, and other factors. It would also allow to compare to a sampling baseline in a controlled setting where many samples can be drawn, and see what the estimation quality/computational tradeoffs are more clearly.


**Assumptions limiting practical applicability**

- **MAR assumption**: According to line 363, the proposed method is valid only under the MAR (missing at random) assumption. While common, this assumption is restrictive, as real-world datasets with missing values often violate MAR. The authors should clarify precisely where and why this assumption is required in their method.

- **Complete training data**: The method requires complete training data, with missing values handled only at inference time. This limits its applicability in settings where comlpete training data is available.


**Conclusion** - I think the theoretical and formalization contribution is very interesting and impactful. Yet the experiments (and lack of details thereof) do not convince me that the proposed estimator is effective (and in which scenarios).

### Questions
1 -  In practice in the experiments, how does one go from the predicted Dirichlet distribution to the hard prediction/confidences? Indeed, in equations (2) and (4), $y_{hard}$ and $c_{hard}$ involve expectations over the distribution of probabilistic prediction $p(\Phi)$. Let’s say I am given an incomplete test sample $x_{obs}$. I deduce the MVU $p(\phi|x_{obs})$ using DMV. And then? Is sampling from the learned Dirichlet perfromed? If yes, how many samples? And what would the effect of this hyperparameter (the number of samples) be? 

2 - I did not completely understand the imputation baseline. Detailing the formula would be necessary in Appendix.

3 - In the synthetic experiment, what is the technique used to estimate MVU?

4 - What neural network exactly is used to predict the Dirichlet parameters in DMV? How does it handle the missing values? What amount of training data is necessary to obtain a good fit?

5 - How many bins to compute MVCE?

6 - Where is the MAR assumption used in the proposed method?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors pose the question as to whether it is worth collecting missing data, but miss important criteria in determining an answer kike whether the data is missing at random, the amount of missing data and any tangible cost estimation approach.

### Strengths
The problem is important and not enough people consider the missing data factor in model computation. It is also important that the authors bring in the idea of worth/cost.

### Weaknesses
The authors ignore critical aspects of missing data including whether the data is missing at random and how much data is missing for a variable of interest in terms of its overall impact on outcomes. In addition, it is not clear why synthetic data was used, as this would be easy to manipulate to guarantee outcomes. Lastly, the determination of worth is critical but glossed over in this paper. For example, missing data in a diabetes data set could have significant ramifications on model quality but missing data in a dataset aimed at predicting what a person is likely to buy next is not. 

The paper is very light on references to missing data literature, I encourage the authors to read :

L Joel, W Doorsamy, A review of missing data handling techniques for machine learning. Int.
J. Innov. Technol. Interdiscip. Sci. 5, 971–1005 (2022).

6S Nijman, et al., Missing data is poorly handled and reported in prediction model studies
using machine learning: a literature review. J. clinical epidemiology 142, 218–229 (2022).

TE Raghunathan, What do we do with missing data? some options for analysis of
incomplete data. Annu. Rev. Public Heal. 25, 99–117 (2004).

T Emmanuel, et al., A survey on missing data in machine learning. J. Big data 8, 1–37
(2021).

M Van Ness, TM Bosschieter, R Halpin-Gregorio, M Udell, The missing indicator method:
From low to high dimensions in Proceedings of the 29th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining. pp. 5004–5015 (2023).

### Questions
How would random vs. non-random patterns of missingness affect your outcomes?

How would data set sizes and real data change your outcomes?

Given the focus on whether it woudl be worth collecting data, it would have been good to see a cost-benefit approach to demonstrate what is meant by worth

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Paper proposes a framework for missing value uncertainty of classification methods in the case of missing input values at inference time. This can be used in decision making to collect more information, if required. Computationally efficient estimator called direct missing value is derived to overcome computational challenges of Monte Carlo (MC) sampling, as well as missing value calibration error metric for evaluation and post-hoc calibration of the model. Proposed methodologies are evaluated and compared to a few baselines on synthetic and real data with promising results.

### Strengths
To my knowledge, paper proposes novel approach to uncertainty quantification of missing input data on a test-time, more rarely studied compared to first-order uncertainty and the calibration of soft confidence, and could provide interesting addition and complementary information to uncertainty modelling in trustworthy ML and active decision making in critical application areas. The more specific contributions are 1) the formulation of the framework, 2) computational improvements against MC baseline with new estimator for missing value uncertainty, and 3) new error metric for this particular setting (inspired by expected calibration error (ECE) typically used to evaluate and estimate the output calibration). The empirical evaluations are done to support and justify the most of the claims.

Summary of strengths:
- Novel and interesting approach
- Computational improvements against baseline
- New metric and estimator proposed
- Experiments show some usefulness

### Weaknesses
Although paper is generally sound and the most of the claims are supported by the empirical evaluations, there are some weaknesses related to experiments to better justify and support the findings and the significance of the work. First, there could be more versatile evaluation baselines and discussion of prior work of why particular simple baselines (e.g., imputation and missing robust models) are chosen and how would more advanced methods affect the evaluation. Second, analysis of proposed error metrics (MVCE) could be examined more detailed and its relation to other metrics in the literature. Third, masked image dataset are used in experiments, but it  is unclear how the proposed approach could perform in other modalities such as tabular data (with missing inputs) or data from medical and sensor network domains mentioned in the paper. Clarification for these could improve the quality and significance of the work (see also specific questions below).

Summary of Weaknesses
- Insufficient connection to prior work and utilised baselines in the discussion and experiments (i.e., why particular simple baselines are chosen)
- Limited analysis of proposed error metric (MVCE) against other metrics in the literature (and its properties/parameters)
- Limitations on evaluated datasets (e.g., missing of modalities such as real tabular data or those critical application domains mentioned in the paper)

### Questions
- Paper considers image datasets. How would it apply e.g., in medical diagnosis or sensor network data mentioned in the paper? 
- How MVCE parameters (i.e., bin size) affect the error metrics in different settings?
- What are the computational efficiency/times of proposed and baseline methods with other than CelebA dataset?

### Soundness
3

### Presentation
2

### Contribution
3
