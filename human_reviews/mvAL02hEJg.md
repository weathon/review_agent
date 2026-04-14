# Revisiting Covariate and Hypothesis Roles in ITE Estimation: A New Approach Using Laplacian Regularization

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
The recent surge in data availability across many fields, such as medicine, social science, and marketing, has brought to the forefront the problem of estimating Individual Treatment Effect (ITE) from observational data to effectively tailor treatment to personalized characteristics. ITE estimation is known to be a challenging task because we can only observe the outcome with or without treatment, but never both. Moreover, observational datasets exhibit selection bias induced by the treatment assignment policy. In this paper, we present a new approach consisting of two novel aspects. First, we depart from conventional approaches that minimize the covariate shift. Instead, we incorporate it as a crucial element in ITE estimation, recognizing that it stems from highly predictive features that exhibit significant imbalance in observational data. Second, unlike existing methods, our approach utilizes hypothesis functions to directly estimate outcomes under covariate shift, enhancing reliability across observed and unobserved outcomes. To support this approach theoretically, we derive a new upper bound of the expected ITE loss and show that it explicitly depends on the discrepancy between the hypothesis functions, which are absent from the objectives of existing methods. Based on this new approach, we present LITE: Laplacian Individual Treatment Effect, a novel method that leverages Laplacian-regularized representation and incorporates both the covariate shift and the hypothesis functions for ITE estimation, effectively bridging observed and unobserved outcomes. We demonstrate LITE on illustrative simulations and two leading benchmarks, where we show superior results compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper revisits the role of covariates in the estimation of individualized treatment effects (ITE). It argues that instead of minimizing covariate shifts, the focus should be on differences in the hypothesis functions.

Building on this perspective, the paper proposes a new upper bound for ITE estimation and introduces a graph-based approach to model covariate representation, allowing for direct estimation of ITE under covariate shifts.

### Strengths
- This paper addresses an important issue: learning a balanced representation in individualized treatment effect (ITE) estimation might impair a model's predictive ability for outcomes.

- The motivation for the proposed method is sound. To mitigate the negative impact of the IPM term on outcome prediction, the approach seeks to impose constraints from the perspective of the hypothesis function, aiming to fit the function without being affected by distribution shifts.

- The experiments conducted in this paper are thorough.

### Weaknesses
- My main concern lies in the theoretical part of the paper, particularly regarding the constraint term in Theorem 3: $\int_{\mathcal{R}}\left|\left(\ell_{h, \Phi}(\Psi(r), 1)-\ell_{h, \Phi}(\Psi(r), 0)\right)\right| d r $. From my perspective, this term merely substitutes the overall distribution constraint of the IPM from [1] with a constraint on expected distributions. Essentially, it does not differ significantly from the IPM term; it still aligns the expected covariates for T=0 and T=1, failing to achieve the paper's stated motivation of avoiding covariate shift while focusing on hypothesis functions. This is unfortunate; it would be more effective if the paper directly constrained Equation (22) in Appendix A. Consequently, my concerns regarding this constraint lead me to be skeptical about the method designed based on it.

- Additionally, there is a minor writing issue: Section 3.2 is titled "THE SELECTION BIAS," yet lines 184-194 primarily discuss ITE learning. Only lines 195-198 briefly address selection bias, making the content misaligned with the section title.

- The discussion of related work overlooks several significant studies. For instance, [2] introduces sample weights for weighted regression to address the negative impact of the IPM on outcome prediction. Study [3] incorporates a TR constraint based on representation learning to enhance robustness in outcome prediction. Additionally, [4] presents the concept of prognostic scores to directly ensure information about outcomes. Studies [5] and [6] introduce the double balancing score/representation concept, aiming to model shifts in covariate distributions while preserving predictive power for Y. Research on sufficient dimension reduction, such as [7], focuses on learning identifiable representations from covariates to retain essential information. Given these contributions, it is important to clarify the unique advantages of this paper in modeling outcome prediction information compared to these prior works.

---

[1] Shalit U, Johansson F D, Sontag D. Estimating individual treatment effect: generalization bounds and algorithms[C]//International conference on machine learning. PMLR, 2017: 3076-3085.

[2] Johansson F D, Shalit U, Kallus N, et al. Generalization bounds and representation learning for estimation of potential outcomes and causal effects[J]. Journal of Machine Learning Research, 2022, 23(166): 1-50.

[3] Shi C, Blei D, Veitch V. Adapting neural networks for the estimation of treatment effects[J]. Advances in neural information processing systems, 2019, 32.

[4] Hansen B B. The prognostic analogue of the propensity score[J]. Biometrika, 2008, 95(2): 481-488.

[5] Hu Z, Follmann D A, Wang N. Estimation of mean response via the effective balancing score[J]. Biometrika, 2014, 101(3): 613-624.

[6] Zhu M, Wu A, Li H, et al. Contrastive balancing representation learning for heterogeneous dose-response curves estimation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(15): 17175-17183.

[7] Zhang Y, Berrevoets J, Van Der Schaar M. Identifiable energy-based representations: an application to estimating heterogeneous causal effects[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2022: 4158-4177.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper offers a new approach for ITE estimation. They rather incorporate covariant shift, by assuming that it arises from highly predictive features. Unlike existing methods, they do not regularize explicitly over covariant shift, but in turn work on regularizing laplacian type quantity on similarity graph between features.

### Strengths
(1) The paper is tackling an important problem of ITE estimation.

### Weaknesses
The key weakness about this paper is technical novelty of the paper. I think the contribution is only to come up with Eq. (13), which is not technically very challenging to come up with. It is indeed true that no paper worked this regularization but, in my opinion, this is not enough contribution. Algorithm 1 is adding little value--- it is just about training their model.

Re. regularization, I think the regularizer is using more of a similarity kernel rather than graph. The authors should look into this paper:
Continuous Treatment Effect Estimation Using Gradient Interpolation and Kernel Smoothing [Lokesh et al AAAI 2024], for a discussion.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel method call LITE to incorporate covariate shift and hypothesis function in ITE (CATE) estimation. The method replaces the IPM metrics in classical counterfactual representation learning framework with a Laplacian regularization term to impose smoothness on outcome predictions across different samples. Theories are derived to explain the motivation of such regularization, and experiments show that it generally outperforms previous representation learning methods.

### Strengths
The paper draws our eyes on alternative ways apart from direct balancing representation distributions in representation learning, and emphasizes the role of hypothesis function class in effect prediction. The theory is a direct yet interesting extension from Shalit’s work. The paper is overall well-organized and well-written, with theories and experiments backing up the arguments.

### Weaknesses
* It is kindly reminded that from general terminology in causal inference, the paper is actually estimating CATE but not ITE, although early representation learning works do use the name ITE. The typical ITE definition can be found in [1].

* There are few flaws in the theory, and the association between algorithm and theory should be further clarified. See questions for details.

* Since Theorem 1 and 2 are directly borrowed from Shalit’s work, I suggest using Proposition rather than Theorem to highlight the novel results.

### Questions
* Equation (23) in the proof of Theorem 3 is based on the assumption than $p_{\Phi}^{t=1}(r)<1$. However, when $\Phi(x)$ is a continuous variable, the probability density may not be bounded by 1.
* Although extensively discussed in the paper, I am still not clear on why imposing smoothness on prediction helps bound the distance term in Theorem 3. Some mathematical explanations would help a lot.
* When potential outcome is not continuous, will imposing smoothness harms the outcome prediction?

**References**

[1] Lei and Candès, Conformal Inference of Counterfactuals and Individual Treatment Effects, JRSSB 2021.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper revisits a fundamental problem in causal inference: the estimation of the individual treatment effect (ITE). Many previous studies have been proposed to estimate the ITE and eliminate the selection bias. While most of them solved this issue by aligning the latent feature space and minimizing the covariate shift, this paper asserted the imbalance would be inherent and stem from the highly predictive features -- it directly impacts the causal treatment effect and should not be simply mitigated. This paper proposed a novel method to estimate the ITE by incorporating the covariate shift as a crucial element in ITE estimation and utilizing hypothesis functions to directly estimate the outcomes under the covariate shift. The experiments show great improvement in the performance of ITE estimation and comparable results in ATE estimation.

### Strengths
S1: The paper is well presented, with every detail clearly described. 

S2: The paper is well motivated, not incremental, but completely novel, with totally different approaches to estimating ITE.

S3: The experiment results are outstanding, which significantly reduces the PEHE compared with the previous SOTA (dramatic performance improvement).

### Weaknesses
W1: I am afraid the motivation for why the authors think their proposed LITE can obtain the goal is unclear. In Section 4, the proposed LITE is developed to enforce $h_t[i]$ and $h_t[l]$ are similar when the two samples are close in representation space, but it doesn't directly address the covariate shift, which should address the connections of $h_1(\cdot)$ and $h_0(\cdot)$, but I didn't see that in your method. 

W2: The illustration of the proposed LITE with Theorem 3 is weak. Within line 384 -400, the authors illustrate that their methods can achieve the proposed goal (controlling the PEHE based on Eq. 9) based on some arguments. However, these arguments are subjective and lack theoretical promise. 

W3: The practical use is somehow confusing. The method requires a proper $\alpha$. Improper $\alpha$ will lead to dramatically large errors (Figure 3 (a)). However, in practical use, there is no validation set and how to choose a proper $\alpha$ is confusing.

### Questions
Q1: There is no explicit illustration of how minimizing the penalty Eq. 13 will reflect the covariate shift. The covariate shift should construct connections between $h_1$ and $h_0$. However they are separated in Eq. 13. The proposed method enforces the predictions $h_t$ to be close if the samples are close in representation space, but it doesn't address the covariate shift. 

Q2: The connections between the proposed methods and Theorem 3 are insufficient, based on arguments while lacking theoretical promise. 

Q3: In experiments, the authors only compare their methods with the naive method in simulation studies (Section 5.1), what if other baseline methods are applied to the simulation studies? can the authors illustrate more why the baseline methods cannot be applied to the simulation studies or supplement some studies to compare the performance? 

Q4: How to choose the proper $\alpha$ in real-world application as there is no validation set (in observational studies, no counterfactual outcomes are available)? Do we need the RCT as validation studies (which would be expensive), or do the authors have additional solutions?

### Soundness
2

### Presentation
3

### Contribution
3
