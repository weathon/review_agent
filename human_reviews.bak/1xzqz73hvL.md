# High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws

- Decision: Accept (Spotlight)
- Scores: 8, 6, 6, 8

## Abstract
A growing number of machine learning scenarios rely on knowledge distillation where one uses the output of a surrogate model as labels to supervise the training of a target model. In this work, we provide a sharp characterization of this process for ridgeless, high-dimensional regression, under two settings: *(i)* model shift, where the surrogate model is arbitrary, and *(ii)* distribution shift, where the surrogate model is the solution of empirical risk minimization with out-of-distribution data. In both cases, we characterize the precise risk of the target model through non-asymptotic bounds in terms of sample size and data distribution under mild conditions. As a consequence, we identify the form of the optimal surrogate model, which reveals the benefits and limitations of discarding weak features in a data-dependent fashion. In the context of weak-to-strong (W2S) generalization, this has the interpretation that *(i)* W2S training, with the surrogate as the weak model, can provably outperform training with strong labels under the same data budget, but *(ii)* it is unable to improve the data scaling law. We validate our results on numerical experiments both on ridgeless regression and on neural network architectures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the problem of knowledge distillation under the setting of linear ridge-less regression in the teacher student scenario.

The setting considered in this paper is the the proportional regime where $p,n\to\infty$ and their ratio $\kappa_t = p/n$ is kept fixed. They consider $\kappa_t > 1$ in the overparametrised regime.

The models considered in the paper are three:
* _The Surrogate-to-Target model_ where the data is generated from a dataset $\mathcal{D}$ with input $x\in\mathbb{R}^d$ and output $y = x^\top \beta_\star + z$ with $ \beta_\star$ a teacher vector. This data is used to estimate a min norm estimator called $\beta^s$ and generate a second data set $y^s = x^\top \beta^s + z$ and the final estimation is done as $\beta^{s2t}$ from $(x,y^s)$.
* _The Standard Target model_ where the model is evaluated on the generated data $(x, y)$
* _The Covariance Shift model_ here the dataset is generated with a certain choice of covariance and then the population risk evaluated on a different covariance model.

The first part of the paper is devoted to finding the performance conditioned on a specific teacher while the second to last section considers the full _Surrogate-to-Target_ setup.
The authors also consider the procedure of Masking for the surrogate model. In this case the surrogate model has been trained on a masked version of the data and the new labels are generated from the original inputs and the labels of the surrogate model.

The main technical results presented in the main are the characterisation of the population risk for the model conditioned on $\beta^s$ and then for the _Surrogate-to-Target_ model.

For the case conditioned on the target the authors are able to precisely derive the effect of the surrogate model on the final student, showing specific conditions (depending on the covariates and $\beta^\star$) under which a $\beta^{s2t}$ performs better than a _The Standard Target model_. The same is true for the masking.

### Strengths
This paper is mathematically sound and considers an interesting setting. The main strengths are
* The derivations of the different transition values for the covariates is sharp and to my knowledge is a novel finding.
* The theory match with simulations also work at finite dimension even if the result is high-dimensional.
* The model introduced and studied is expressive enough to show different behaviour that characterise performances.

### Weaknesses
* Right now the mathematical result is introduced in generality without explaining the idea behind the proof. The authors could briefly explain that to derive the results one should apply the theory from [Han&Xu2023] that relies on the use of of the convex gordon min max theorem.
* The authors provide some numerical simulations on ResNet-50 on a CIFAR10 classification showing a result that qualitatively differs from the theory. Either this limitation is explained in detail or I don't think it is necessary to be shown.
* Is there any reason why the authors consider in their technical results the ridgeless estimator instead of the ridge one? A long series of works (e.g. [Hastie2020, Louriero2022]) considers general loss and provides similar bounds.
* _(Minor)_ Section 4 is presented unclearly. The settings for the propositions are not well explained and need to be introduced more clearly.

[Hastie 2020] Surprises in High-Dimensional Ridgeless Least Squares Interpolation. Annals of Stats 2020

[Loureiro2022] Learning curves of generic features maps for realistic datasets with a teacher-student model. Neurips 2022

### Questions
* I would love to see Figure 1b and Figure 2b in LogLog Scale. One of the main points of the authors in Section 4 (Proposition 5) concerns the learning rates in the high-dimensional limit. It would be nice to see them in Figure 1b.
* Is it possible to generalise the result of Section 3.1 to the case where the chosen features are chosen with a matrix $A$ which has a non zero kernel? The masking seems a specific case of this.


* There is a broken citation on page 3.
* On line 334 is it "omniscent test risk estimate"?

### Soundness
3

### Presentation
3

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
The paper provides a sharp characterization for knowledge distillation in the high-dimensional regression setting, including both model shift and distribution shift, cases. Concretely, the paper characterizes the precise risk of the target model in both cases through non-asymptotic bounds in terms of sample size and data distribution under mild conditions. As a consequence, the paper identifies the form of the optimal surrogate model, which reveals the benefits and limitations of such processes. Finally, the paper validates the results by numerical experiments.

### Strengths
1. Knowledge distillation and weak-to-strong generalization are significant topics today, and their theory is very poor. Therefore, this is a meaningful paper for me.
2. The theory is complete and well-written.
3. The derived bounds seem tight because they are matched with empirical results.

### Weaknesses
1. The theory only focuses the high-dimensional linear regression setting, which is well-studied in the literature. Besides, the results can not be extended to neural networks directly.
2. A typo in line 134.

### Questions
1. Can you give me some insights to extend the theory to neural networks (even if two-layer neural network)? I think the authors should also discuss this in the refined version.

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
This paper studies the problem of knowledge distillation under linear regression. In the first stage, data are collected from a surrogate model. In the second stage, a target model is trained using the data generated in the first stage. The authors characterize the non-asymptotic excess risk of the target model under "model shift" setting and "distribution shift" setting. Numerical results are provided, justifying their theory on ridgeless regression and on neural network architectures.

### Strengths
The authors provide comprehensive theoretical results for weak-to-strong generalization, giving a exact characterization of the excess risk of the weak-to-strong estimator. This knowledge distillation problem is important in modern machine learning, indicating the significance of this work.

### Weaknesses
The presentation of this paper is in general not very satisfactory, in the sense that this paper is lack of necessary intuition and explanation. For example, how to interpret the non-asymptotic bounds and what does each term stand for? Why it is possible that weak-to-strong estimator is even better than purely using the strong model?

### Questions
1. Can you provide intuition why the risk of the surrogate-to-target model under the optimal selection of the parameters scales the same as that of the target model (even though there is a strict improvement in the risk)? I am wondering why improvement is possible. Is it because, for example, if the tail components of the covariance is zero, then features on these components are essentially useless, therefore an surrogate that omits those components will be better? 

2. your equation (8) involves $\beta^{s2t}$. Does it mean that your asymptotic risk estimate (9) also involves $\beta^{s2t}$ and thus can not be directly computed? I think in the final bound $\beta^{s2t}$ should not appear; otherwise I can just claim the definition of the excess risk of $\beta^{s2t}$ is already an exact characterization of itself.

3. In observation1, you assume jointly diagonalizability. Is there fundamental hardness to remove this assumption?

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
3

### Summary
In this paper, the authors propose a precise characterization of the benefits of knowledge distillation, mostly in the context of gaussian linear regression. In particular, the main set-up considers the excess risk of linear regression on a given distribution, but with the learner only able to access pseudolabels generated by a surrogate model instead of the true labels. Notably, the authors show that under a covariance-shift model (i.e. the distribution of covariates $x$ may change between the surrogate and target stages, but the underlying predictor $\beta_\star$ remains the same in between), then the optimal surrogate predictor minimizing the (asymptotic) excess risk on the target distribution is a weighted version of the ground-truth predictor $\beta_\star$, which which amplifies entries corresponding to large eigenvalues of the (diagonal) covariance matrix above a certain threshold, and shrinks entries below a threshold. Furthermore, in a masked setting, where the surrogate model is restricted to selecting a subset of the full set of features, then similarly the optimal surrogate predictor selects predictor entries above a certain threshold of covariance eigenvalues. Lastly, the authors show that in a certain asymptotic regime, an optimal surrogate-to-target model (i.e. a model trained on target distribution covariates with surrogate model pseudolabels) has the same excess risk as the least-squares target model trained with the true labels, demonstrating that knowledge distillation in a sense cannot beat out ERM with access to true labels.

### Strengths
This paper is rather well-written and contains quite a few interesting theoretical insights. The results draw clear delineations on how knowledge distillation through a surrogate model can help. As someone who hasn't thought about overparameterized linear regression in a while, Proposition 1 and Corollary 1 were rather surprising results, demonstrating that for a dataset size proportional (but smaller) than the number of parameters, the optimal surrogate predictor to use for generating pseudolabels is actually not the ground truth predictor, and that there is (in theory) always room for benefit as long as the covariance is non-isotropic, which implies that a learner benefits from using something other than the actual distribution of labels. 

In addition to the theory, the numerical results on CIFAR-10 also counterintuitively support that a learner only trained on surrogate pseudolabels on the target domain actually outperform the surrogate model itself, which has access to true labels (albeit with a different covariate distribution...?).

### Weaknesses
Though the theoretical results are interesting, there are a few aspects that are worth clarifying. Notably, even though it is demonstrated that there can exist a surrogate model that induces better risk on the target model than using the true labels, in general a surrogate model is typically not trained with foreknowledge of the task distribution. I believe this is what Section 5 is trying to convey, but it is not clear to me after reading that section how to interpret the result therein. In particular, it should be explained how this relates to, for example, a target model that is trained using strong labels to demonstrate the marginal gain (or loss).

In general, the paper accrues a lot of jargon and notation; it would be very helpful to either create a table containing the different set-ups/regimes considered and the summary conclusion of the relative gain/suboptimality of knowledge distillation and/or a notation table that summarizes what the various risks denote. This would help clarify how to place the scaling law (Proposition 5) and Section 5 with respect to the results of the prior sections.

### Questions
In addition to the points above that should be clarified, I have one following question: how is the CIFAR-10 experiment performed? Notably, how are the surrogate and target distributions generated? This is worth expanding in the paper.

### Soundness
4

### Presentation
3

### Contribution
3
