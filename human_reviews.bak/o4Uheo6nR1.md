# Robust prediction under missingness shifts

- Decision: Reject
- Scores: 6, 8, 3, 5

## Abstract
Prediction becomes more challenging with missing data. What method is chosen to handle missing data can greatly affect how models perform. In many real-world problems, the best prediction performance is achieved by models that also leverage the informative nature of a value being missing. However, the reasons why data goes missing can change once a model is deployed in practice. In this case, prediction performance in the development data may no longer be a good selection criterion, and approaches that do not rely on informative missingness may be preferable. To identify the conditions that lead to robust prediction, we formalise the problem of missingness shifts as any change in the conditional probability of a value being missing. We then show that the optimal predictor is only affected by non-ignorable shifts, where the probability of missingness depends on unobserved data. When the optimal predictor is changed due to a non-ignorable shift, we find empirically that even predictors which utilise information encoded in the missingness may still achieve robust predictions, although different methods appear robust to different types of shifts. Disregarding informative missingness was most beneficial when the probability of missingness was influenced by the outcome.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper examines theoretically and empirically the problem of missingness shift. It characterizes missingness shifts to which an optimal Bayes model would be robust, introduces an iterative imputation strategy for joint imputation and outcome modelling, and concludes with an empirical assessment of performance under missingness shifts.

### Strengths
This paper presents an excellent yet simple theoretical characterization of the missingness shift problem. The Bayes optimal formalisation is clearly described, and the connection with current approaches to missingness is appreciated. The paper is impeccably written, clear and well-motivated.

### Weaknesses
While the paper presents a strong and clear theoretical case, I believe addressing the following points would strengthen the paper, particularly concerning the empirical aspects:

- Existing works have studied the problem of missingness/observation shifts [1, 2]. One paper formalizes missingness shifts, while the other focuses on empirical studies. The authors should consider discussing the distinctions with their proposed formalization and approach.
- Equation (2) relies on the assumption: $\mathbb{E}[\epsilon \mid X_{obs}, M] = 0$. Authors should detail this assumption, its meaning and its real-world relevance. Particularly, does it not imply some independence of Y upon the missing data?
- In Section 4, there is an implicit assumption of no covariate or concept shift. Making this explicit would enhance clarity for readers.
- The paper would be strengthened by detailing NeuMISE and why this modelling would better approximate the conditional expectations and a Bayes optimal model. 
- It would be beneficial to delve further into the discussion of Appendix F, particularly in addressing the unclear aspects of why the model performs less effectively when uncertainty increases.
- While the empirical results employ real-world covariates, the analysis lacks a study of real-world missingness shifts. Considering a dataset with changing missingness processes would strengthen the experimental analysis.
- Comparing the proposed approach with an end-to-end neural network that uses zero imputed data and mask as input seems a natural comparison to demonstrate the superiority of the proposed NeuMICE.


[1] Zhou, H., Balakrishnan, S., & Lipton, Z. (2023, April). Domain adaptation under missingness shift. In International Conference on Artificial Intelligence and Statistics (pp. 9577-9606). PMLR.
[2] Jeanselme, V., Martin, G., Peek, N., Sperrin, M., Tom, B., & Barrett, J. (2022). Deepjoint: Robust survival modelling under clinical presence shift. In NeurIPS Learning from Time Series for Health Workshop.

### Questions
I consider the paper to be a valuable theoretical contribution, my current rating is only hurt by the limitations outlined earlier.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies the problem of developing a robust predictive model in cases where the pattern through which covariates are missing changes between a source and target domain. The theoretical contributions of the work are a formalization of the problem, a proof that the optimal predictor is stable in case that the missingness pattern is ignorable in both environments (e.g. the missingness pattern depends only observed covariates), and an argument that predictors that leverage informative missingness can still be robust under missingness shift. They also introduce a new neural network architecture NeuMISE that builds off of the NeuMiss architecture for learning with missing data. Experiments with synthetic and real-world and data are conducted with injected missingess shift.

### Strengths
* This work tackles an important but under-emphasized problem with simple but powerful theoretical results. The core contribution regarding the formalization of missingness shift and discussion of ignorable shifts is a strong contribution and has potential for broad use in applications.
* The experiments are broad and cover a number of data generating processes, shift mechanisms, and comparator methods.

### Weaknesses
* The motivation for the NeuMISE method is not presented clearly enough or with enough detail to tie it to the rest of the core claims of the work. It is primarily not clear why modifying the masking of NeuMiss is well-motivated to address the issue of generalizing across unobserved missingness patterns.
* I have several concerns regarding the clarity of the work, which are elaborated on in the Questions section below.

### Questions
* Related to clarity:
  * Important aspects of the experiments are not presented clearly enough or with enough detail in the main text. For example, it is not explained what “low correlation” and “high correlation” corresponds to in the experiments.
  * Section 5 on the role of Y is interesting, but is not presented particularly clearly. In particular, please elaborate on how adjusting for Y in a source domain can induce missingness shift, but omitting Y results in a stable estimator.
  * The discussion focuses strongly on the comparing methods that leverage informative missingness vs. “unbiased” estimators that do not. This seems to be a critical point for the paper overall (e.g. related to the second of the three contributions listed in the introduction section), but it did not come through clearly to me in the writing. Furthermore, it is not clear how (or if) this was evaluated in the experiments. This could perhaps be improved with additional exposition earlier in the paper that sets up this argument more clearly with specific hypotheses to be evaluated.
* Can the results be generalized to binary outcomes? Naively, it seems like the additive noise model limits the direct applicability of this framework to binary outcomes.
* Is the assumption that the error term in equation (2) depends only on the observed X and M limiting the generalizability of this work? Which aspects of the results would no longer hold if the error term were to depend on the full X?
* Please comment on the relationship of the results of this work to Zhou et al 2023 “Domain Adaptation under Missingness Shift” and adjust claims regarding novelty and prior work, if appropriate.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work is motivated by the observation that the source of missingness (missing values in covariates) may differ in the train and deployment populations. The paper studies conditions under which the optimal predictor does not change in the presence of missingness shifts. It also analyzes the extent to which methods that utilize informative missingness generalize well in the presence of missingness shifts. They introduce a method called NeuMISE that aims to be robust across a range of missingness mechanisms.

### Strengths
1. Finding ways to cope with non-ignorable distribution shifts (shifts in the conditional distribution of Y|X) is an important and challenging problem and has not received as much attention as covariate distribution shift, so it’s great that this work points out that methods for dealing with missingness and ignorable missingness shift are insufficient in the presence of non-ignorable missingness shift.

2. The paper is clear and well-written.

### Weaknesses
1. The main weakness of this work is that it does not cite or discuss its connection to [1], which is another work that studies robustness to missingness shift. This paper describes that when missing data indicators are available, domain adaptation under missingness shift reduces to a covariate shift problem. This finding seems to be related to one of the central contributions of this paper, which is that the optimal predictor remains unchanged if missingness only depends on observables in both the training and test environment.


2. It’s not clear to me what advantage NeuMISE (the authors’ proposed method) has compared to the existing baseline in the presence of non-ignorable missingness shift. While the authors have some empirical results that NeuMISE performs outperforms other methods in the presence of non-ignorable missingness shift, I’m skeptical that such a result holds in general. To my understanding, generalizing well to non-ignorable missingness shift should only be possible if the model is in some sense robust to a variety of non-ignorable missingness shift, and I would presume that such a model may trade off some performance on the source data for better generalization across target environments. Is that the type of result that we observe for NeuMISE? What is the reason that NeuMISE is more robust? Furthermore, what benefit does NeuMISE offer compared to existing baselines.

3. It would be helpful to add a few concrete examples where missingness shifts occur in the real-world to motivate the research.

Improvements:

1. It would be helpful if the authors emphasize in their abstract/introduction that they focus on missingness in covariates, not labels. There is an extensive literature on learning with missing labels and it is somewhat unclear what type of missingness the authors are focusing on until the problem definition in Section 3.

2. It would be nice to draw a connection between ignorable / non-ignorable missingness to ignorable / non-ignorable sample selection. 

[1] Zhou, Helen, Sivaraman Balakrishnan, and Zachary Lipton. "Domain adaptation under missingness shift." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

### Questions
1. It would be helpful if the authors add a line after Equation 2 that explains what the assumption $\mathbb{E}[ \epsilon \mid X_{obs}, M] = 0$ means concretely – i.e., to what extent can the noise $\epsilon$ depend on the missingness $M$? The current presentation does not require $\epsilon$ to be independent of $M$ – is that the desired interpretation? To the best of my understanding, in the current presentation, the variance of the noise $\epsilon$ could depend on the missingness mechanism.

2. Could the authors explain why the following is true: ``If $Y$ only influences missingness in the source environment, shifts may still be ignorable.”

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a prediction problem with missing shift settings, which is an important practical task.  The paper first provides an overall review of missing mechnisms and related literature. It further discusses the equivalence of Bayes predictors under ignorable missing shift and the effects of shifts in Y-dependence. It also proposes a NeuMISE to handle this challenging task.

### Strengths
The paper is well-written and has a very clear organization. The proposed method, NeuMISE, seems to be simple but effective and outperform other baselines. The results are relatively complete and solid.

### Weaknesses
The paper uses quite a lot space to discuss the missingness shift. Although such descriptions are complete and clear, it seems to be relatively elementary and do not provide enough new intelletucal insights. Under ignorable condition, Theorem 1 "equivalence" is also straightforward and hence is not surprising, at least to me.

Last few sentences in Section 5.1 confuse me. what is "adjusting Y", "omitting Y" and definition of "stable estimator"?

Section 6 is rather short. It should be expanded to explain why NeuMISE is more effective from a deeper viewpoint.

### Questions
See weakness points.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
