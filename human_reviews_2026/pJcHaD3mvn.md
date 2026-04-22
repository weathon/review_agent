# Extrapolating Large Models from the Small: Optimal Learning of Scaling Laws

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
As large language models (LLMs) continue to scale to billions of parameters, training them becomes increasingly expensive, making it infeasible to exhaustively explore the vast design space including model architectures, parameter sizes, and compute budgets. Scaling laws have therefore emerged as an essential tool for predicting the performance of larger models by extrapolating from smaller ones, enabling practitioners to make informed design choices without full‑scale training. However, existing approaches lack formal guarantees on the predicted results and overlook the out-of-distribution nature of such extrapolation, leading to high instability. We address these challenges with three key contributions. First, we introduce Equivalent Sample Size (ESS), a natural and principled metric that quantifies prediction uncertainty by translating it into the number of test samples required for direct, in-distribution evaluation. Second, we analyze how extrapolation amplifies prediction variance and develop an efficient algorithm that optimally allocates smaller-model evaluations to maximize ESS under compute budgets. Third, experiments on both simulated and real datasets show that ESS and our algorithm guide the design of scaling-law learning, cut evaluation cost, and deliver reliable LLM performance predictions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of uncertainty in scaling prediction in LLM evaluations and proposes a framework that cuts the evaluation costs when using smaller models to predict the larger model's performance.

### Strengths
- The problem of scaling prediction seems to be correctly introduced and motivated in the introduction 
- General organization of the paper is in a good state.

### Weaknesses
- **W1) Unclear Definition 1.** The definition of the equivalent sample size/ESS (Definition 1) is central to this paper, but the definition is not clear. It does not clearly introduce the "equivalent sample size". The definition just says that a distribution has an equivalent sample size if the number of samples $n$ fulfills a certain condition. Is $n$ the equivalent sample size or what do you mean specifically? If your goal is to introduce a metric for quantifying prediction uncertainty you have to explicitly introduce this metric (this is not sufficiently clear in the current state).  A clear introduction would be especially critical for researchers / practitioners to understand how to adapt your measure for prediction uncertainty.

- **W2) Missing justifications.** The derivation in line 180 suggests that Hoeffding's inequality can be applied, however, applying this inequality requires justification. The problem is also that you do not clearly introduce the random variable for which you want to apply this inequality. Do you want to bound the expectation of the random variable $\ell(f(S_i), R_i)$, where $\ell$ is a loss function? The problem is that we would require this random variable to be bounded in order to apply Hoeffding's inequality (in my understanding), which would require an additional assumption on the loss function. It would also be helpful to cite a source where this inequality is introduced, e.g. (Hoeffding, 1963).

- **W3) General writing-quality is low.** In line 178 $ \hat{P}\_n $ is introduced as the empirical performance, but in Definition 1 it is denoting a distribution (line 184). First $X\_\*$ is a non-negative value (line 212), but later you consider $X_* \in [x_l, x_u]$ (line 254) and it remains unclear why that changes exactly. Section 4 and 5 refer to a Theorem 4.1 but there is no such theorem (just a proposition). $\overline{XY}_M$ in line 216 is never used. The main section 4 aims at quantifying "reliability" instead of uncertainty (line 169) and it remains unclear what is meant with reliability. The paper requires significant polishing.

**Minor weaknesses**
- The paper would generally benefit from a first figure.
- Typo in line 180. Do you mean $\epsilon_n$ instead of $\epsilon$ in the confidence interval?
- The example 1 (lines 231-237) is very loaded with notation and rather challenging to follow.

Overall, the paper requires major revisions throughout the entire manuscript, I cannot recommend it for acceptance in its current state.

### Questions
What do we need the "effective sample size" for (line 192)? This is never used again (unless I'm missing something).

Further questions see weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies using small models to fit scaling laws and then extrapolate to large models to cut evaluation cost, but finds that such extrapolation is out-of-distribution and often unstable. The main contribution of the work is proposing an interpretable ESS metric, explaining the variance-amplification mechanism of out-of-range extrapolation, and providing an optimal selection method for small models.

### Strengths
1. The proposed definition and framework are novel and workable.

2. In implementing the algorithm, the objective function and constraints are clearly defined, and the experiments support the theoretical claims.

### Weaknesses
1. Reframing the Problem Motivation: The paper compellingly frames the problem around the "prohibitively expensive" cost of evaluation. This is a valid point. However, in the context of SOTA model development, the cost of training is often the dominant bottleneck by several orders of magnitude. The primary industrial use of scaling laws is typically for pre-training decision support (e.g., comparing architectures or data/model allocations) rather than saving post-training evaluation costs. The paper's impact could be significantly strengthened by discussing how the proposed ESS metric and optimal selection algorithm could be adapted for this (arguably more critical) pre-training decision-making scenario.

2. On the 'Model Family' and 'Universal' Link Function Assumption: The framework's practicality hinges on the assumptions made about 'model families.' In practice, scaling laws are often most needed to compare models with subtle but critical differences (e.g., aspect ratios, data mixes), which may already constitute different 'families.' The paper's real-world experiment (Section 7.2) and Remark 1 assume a 'largely universal' link function by fitting it across multiple, distinct model architectures (LLaMA, Qwen, Bloom, etc.). This assumption feels very strong and may not hold in the very scenarios where scaling prediction is most valuable. It would be helpful for the authors to provide more justification for this universality or discuss the sensitivity of their method to this assumption.

3. Reliance on a Correctly Specified Model Form: The paper's core theoretical contributions, such as the variance characterization (Proposition 5.1) and the optimal selection algorithm (Theorem 6.1) , are derived from a specific power-law model (Eq. 1) assuming Gaussian noise. This framework primarily addresses aleatoric uncertainty (noise) and extrapolation variance, assuming the model form itself is correct. As the authors thoughtfully acknowledge in Section 8, this is a key limitation. In practice, model misspecification (i.e., the true scaling relationship is not captured by Eq. 1) is often the largest source of error. This creates a risk that the proposed algorithm could lead to a solution that is "optimally confident" but potentially "incorrect" (i.e., high ESS for a biased prediction). A discussion on how ESS might also help detect or quantify this model form uncertainty would be a valuable addition.

4. Distinguishing Prediction 'Confidence' from 'Correctness': Following the point above, the paper's focus is on minimizing variance to maximize ESS. This is an important and non-trivial contribution. However, it would be beneficial to more clearly delineate this from the challenge of correctness (i.e., bias). The current framework does not seem to penalize a model that is "confidently wrong." It would strengthen the paper to discuss this distinction and whether the optimal selection strategy might inadvertently shift if the goal was to minimize total error (Bias + Variance) rather than just variance.

5. Simplification of Scaling Factors and Interaction Terms: The analysis in Section 5 simplifies the design factor $X$ to a single dimension (log model size) for illustrative purposes. While the model $Y=\alpha+\beta X$ could in principle handle a multi-dimensional $X$, it remains a linear model. This may not be sufficient to capture the complex, non-linear interaction terms between scaling factors [1] have shown to be critical. It is unclear how the optimal selection algorithm (Theorem 6.1) would behave if the true underlying scaling law had such cross-terms.

[1] Houyi Li, Wenzhen Zheng, Qiufeng Wang, Zhenyu Ding, Haoying Wang, Zili Wang, Shijie Xuyang, Ning Ding, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang“Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models.” arXiv:2506.10972 [cs.LG]. https://arxiv.org/abs/2506.10972.

### Questions
1. Regarding the problem motivation on evaluation cost: Could the authors elaborate on the practical scenarios where this cost is the primary bottleneck, especially in relation to the much larger training costs that scaling laws are typically used to predict? We are interested in how the framework might be repositioned to address the pre-training decision problem.

2. Regarding the 'universal' link function: This assumption is central to the experimental setup . How sensitive is the method to this assumption? In practice, one often needs to compare two very similar, but distinct, model families. How would a practitioner validate whether they can use a shared link function, or how would the method be adapted if they cannot?

3. Regarding the ESS metric (Definition 1) : This provides a valuable measure of confidence based on prediction variance. However, if the underlying scaling model (Eq. 1) is misspecified, ESS might be high for a prediction that is, in fact, highly biased. Could the authors comment on this potential trade-off? Is there a way to use ESS to also help diagnose model misspecification, rather than only quantifying variance?

4. Regarding interaction terms: The theoretical analysis relies on a linear relationship between the critical quantity $Y$ and the design factors $X$ (Eq. 1). How would the theoretical results (change if the true scaling law involved non-linear interaction terms between factors, as suggested by other recent work on scaling laws [e.g., Farseer, Li et al., 2025]?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the statistical guarantees of the observational scaling law. It introduces a metric called equivalent sample size to quantify prediction uncertainty. The authors also develop an efficient algorithm to maximize ESS within computational budgets. Synthetic and real-world experiments demonstrate the proposed method’s effectiveness.

### Strengths
- The paper tackles an important problem regarding formal guarantees for scaling laws.
- The proposed efficient model-selection method is technically sound. Some conclusions are counterintuitive and therefore noteworthy.

### Weaknesses
- The proposed framework largely relies on the functional form derived from observational scaling laws. This raises the risk of model misspecification, and the Gaussian assumption may not hold in that case.
- The computation saved by the proposed algorithm seems modest to me in experiments. 
 - Minor issue: It would be better to reference the appendix contents in the main paper. Currently, none of the proofs are cited.

### Questions
Please refer to those in the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the confidence intervals of scaling law. It mainly puts forward a series of optimization strategies to address the problem that large model evaluation consumes excessive resources.​

The authors' discussion revolves around two key perspectives:​
1. As the model scales up, the confidence intervals widen;​
2. As the volume of test data increases, the confidence intervals narrow.​

Accordingly, the authors propose that more evaluations should be conducted at critical junctures, with the aim of reducing the widening of confidence intervals during the model scaling process.

### Strengths
The authors' discussion angles demonstrate remarkable novelty. 

Moreover, they indeed possess considerable practical value—specifically, they can help pre-training teams proactively predict model performance.

### Weaknesses
- Some studies have proposed that scaling laws may not follow a simple logarithmic curve—especially with an increase in data repetition. This could lead to biases in the widening of the authors' confidence intervals, greatly undermining the effectiveness of the authors' method.​
- The authors' discussion lacks consideration of learning rates. In fact, most model training processes may involve adjustments to learning rates during multi-stage training, which exerts a significant impact on model performance. Meanwhile, the usability of the authors' method is also compromised.

Chen, Zhengyu, et al. "Revisiting scaling laws for language models: The role of data quality and training strategies." Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.

Muennighoff, Niklas, et al. "Scaling data-constrained language models." Advances in Neural Information Processing Systems 36 (2023): 50358-50376.

Hernandez, Danny, et al. "Scaling laws and interpretability of learning from repeated data." arXiv preprint arXiv:2205.10487 (2022).

### Questions
How do the authors plan to mitigate the aforementioned weaknesses?

### Soundness
4

### Presentation
3

### Contribution
2
