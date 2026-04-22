# Diagnosing and Improving Diffusion Models by Estimating the Optimal Loss Value

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Diffusion models have achieved remarkable success in generative modeling. Despite more stable training, the loss of diffusion models is not indicative of absolute data-fitting quality, since its optimal value is typically not zero but unknown, leading to the confusion between large optimal loss and insufficient model capacity. In this work, we advocate the need to estimate the optimal loss value for diagnosing and improving diffusion models. We first derive the optimal loss in closed form under a unified formulation of diffusion models, and develop effective estimators for it, including a stochastic variant scalable to large datasets with proper control of variance and bias. With this tool, we unlock the inherent metric for diagnosing training quality of mainstream diffusion model variants, and develop a more performant training schedule based on the optimal loss. Moreover, using models with 120M to 1.5B parameters, we find that the power law is better demonstrated after subtracting the optimal loss from the actual training loss, suggesting a more principled setting for investigating the scaling law for diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets a known limitation of diffusion training losses—namely, that the theoretical optimum is generally non-zero and unknown—making the raw loss an unreliable proxy for data-fitting quality. The authors (i) derive a closed-form expression for the optimal loss under a unified formulation of diffusion models, (ii) propose practical estimators, including a stochastic variant with variance/bias control scalable to large datasets, and (iii) use these tools to diagnose training quality and design an improved training schedule.

### Strengths
Theoretical derivations appear rigorous.
Writing quality and notation are clear, consistent, and professionally presented.

### Weaknesses
1. Loss–performance alignment under matched settings. The paper argues it resolves the mismatch between diffusion loss (with unknown non-zero optimum) and generative performance. To substantiate this claim more convincingly, I suggest including a correlation analysis between training loss (after subtracting the optimal term) and FID under identical settings and budgets, compared to standard loss formulations. Concretely, report Pearson/Spearman correlations over training, rank consistency across checkpoints, and side-by-side training-curve plots for your method vs. baselines on the same architectures/datasets.

2. Over-reliance on FID. FID alone cannot fully capture generative performance. Consider adding complementary metrics and qualitative evidence. A small human preference study (even limited-scale) would help validate perceptual quality beyond distributional statistics.

### Questions
See the weaknesses.

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
3

### Summary
This paper introduce the optimal loss in the parameter designs of diffussion models. The authors propose an estimator of optimal loss. The loss gap idea is then used in the design of noise schedules and loss weights in training objectives. The scaling law of diffusion model is also discussed based on the loss gap other than just the loss value.

### Strengths
The authors propose the use of optimal loss in training schedule design, and design practical estimation of the optimal loss. They also proposed a new schedule which seems to be some refined version of EDM schedules but focuses more about the nature of loss gap between the true loss and the optimal loss, and the practical performance of their schedule is validated.

### Weaknesses
The proposed schedule is still vague, both in writing and in the actual design (see Questions).

### Questions
- In section 2, sometimes the data distribution is $p_0(x_0)$, sometimes is $p(x_0)$. Are they different distributions?
- Please specify the noise schedule $p(t)$ and weight $w_t$ in line 293 using the notation from Section 2 (especially which superscript should be used).
- In Line 382, please specify what is v, F prediction.
- In Line 386-387, is the loss weight discontinuous, which contradicts the continuity of actual loss gap? If so, then why this is a good choice?
- In Line 396, why $p(\sigma)$ is chosen to be proportional to $w_{\sigma}(J-J*)$ instead of for example $(w_{\sigma} (J-J^*))^a$ where $a\ge 0$ ? Is there a theoretical/empirical reason for this? What precise noice schedule do you use in the experiments? Is this changing at each training iteration due to different $J_\sigma(\theta)$? If there is no explicit expression, please show some figures of $p(\sigma)$.
- The strategy of choosing weights and noise schedule is quite confusing to me. In loss weight, you would like to emphasize left side of the threshold $\sigma*$. Btw, the authors state in the paper but I'm still confused why such pattern (the emphasis of positive correlation) is desired. In noise schedule, you would like to emphasize  $w_\sigma (J-J*)$ and therefore in total, combining loss weight, noise schedule, and the l2 distance, the loss looks something like $w_\sigma^2 (J-J*) J^2$ (or maybe $w_\sigma^2 (J-J*) (J^2-J^{*2})$ since constant term does not affect the optimization). Can the author explain the overall intuition of the schedule?
- What is the schedule (timestep, discretization) of $\sigma$? Is this also adapted in some way?


Minor:
- font size in figures is too small

### Soundness
3

### Presentation
3

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
The paper investigates how to estimate the optimal loss values in diffusion models’ training objectives. It first introduces several diffusion losses (corresponding to different prediction targets) and presents them under a unified formalism and notation. Since the optimal predictors for these objectives are given by conditional expectations, their minimal achievable losses still depend on the conditional variances. Based on this, Theorem 1 derives an analytical expression for the optimal loss of the clean-data prediction objective, showing that it decomposes into two terms: a simple dataset-average term and a second, computationally expensive nested expectation. One of the paper’s main contributions is to propose efficient approximations for this second term—the DOL and cDOL estimators—which make estimation feasible on large datasets. The authors validate these estimators on smaller datasets, demonstrating that the approximated optimal losses closely match their true values across diffusion times. They then use the estimated optimal losses to analyze how existing diffusion models approach these limits as noise varies, revealing distinct behaviors across diffusion steps. These observations motivate a combination of an improved training schedule and a loss-weighting scheme yielding better FID scores. Finally, the paper revisits scaling laws for diffusion models, showing that when losses are shifted by their corresponding optimal values, the resulting curves adhere more closely to power-law behavior.

### Strengths
- From the best of the reviewer's knowledge, the characterisation of the diffusion optimal loss values is novel. 
- The paper's results are quite general as the manuscript is structured in such a way that several diffusion setups (i.e., different prediction targets and noising schemes) can be analysed under the same framework.
- The use of optimal loss values as a diagnostic tool is interesting as it allows to identify the noise regimes where models are weaker and ultimately provide adjustment with improved nosing and loss-weighting schemes.

### Weaknesses
- Clarity of exposition and results presentation could be enhanced. The notation is quite heavy and the presentation of the results would benefit from more explanations/interpretation (see Questions). For example, it took me I while to understand what Figure 2(b) is plotting. 
- More fundamentally, it is not clear whether getting a smaller training loss value ultimately results in better performing/generalising models. For example, it would be interesting to understand to what extent the proposed scheduler and loss weight impact memorisation.

### Questions
- Do the authors have an intuition behind the larger loss gap in the intermediate region? This gap appears to be consistent across various model variants.
- Figure 1(c): in the paper the authors state: "The result confirms that the variance increases with C." This does not seem apparent in the plot. could the authors please clarify?
- Could the authors please clarify what is on the y-axis in Figure 2(b)?
- From my understanding, the proposed loss weight tries to assign more importance to the interval to the left of $\sigma*$ based on the results in Figure 2(b). This is because the loss-gap positively correlates with FID in that region. However, the same figure also shows a negative correlation in the proximity of $\sigma*$. Does the proposed loss weight take that into account?
- Figure 3(b): why the choice of $\sigma = 4.38$?
- Figure 3(c): what do you mean by total loss gap?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to estimate the *optimal loss value* of diffusion models — a quantity that depends solely on the dataset and the diffusion setting, not on model capacity. The authors provide a theoretical derivation of this optimal loss under a unified formulation of diffusion models, and develop consistent and scalable estimators that remain stable and “not too greedy.” These estimators allow a principled analysis of model training quality, identifying where diffusion models underfit across noise scales. Leveraging these insights, the authors propose a new training schedule that equalizes performance across diffusion scales and leads to consistent improvements across various datasets (CIFAR-10, ImageNet-64, ImageNet-256). Finally, they show that subtracting the optimal loss from the observed training loss yields a more faithful scaling law, providing a better theoretical understanding of diffusion model scaling.

### Strengths
- The paper answers and analyzes questions that have been so far addressed only empirically — *finally!* The EDM sampling schedule has performed remarkably well for years, and this work explains *why*. Such principled analysis is much needed in a community increasingly driven by empirical results rather than theoretical foundations.  
- Strong and complete theoretical analysis of the proposed estimators, including bias–variance tradeoffs and formal proofs of consistency.  
- Extensive and well-structured experimental validation, showing clear and consistent improvements across multiple datasets and model variants.

### Weaknesses
- The presentation quality is poor. Example given: Figure 1 is hardly readable without zooming, and inline equations significantly hurt readability. Inconsistent line heights (e.g., lines 130–131, or the paragraph 189–200) make the paper visually dense and tiring to read.  
- The alternative formulation of diffusion models in the main text feels unnecessary. A single sentence linking Theorem 1 to the alternative formulation in the appendix would suffice, leaving more space to improve layout and readability.  
- While technically sound, the paper’s exposition could be more accessible, especially in highlighting intuition behind the estimators.

### Questions
- Intuitively, image *quality* tends to be related to predictions at small noise scales, while *recall/diversity* corresponds to large noise scales. It would be interesting to compare the loss gap across these regimes with standard Precision and Recall metrics — this could provide an even more interpretable connection between the optimal loss and generative performance.

### Soundness
4

### Presentation
2

### Contribution
4
