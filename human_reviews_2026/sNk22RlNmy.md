# DoubleGen: Debiased Generative Modeling of Counterfactuals

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 8, 4, 6

## Abstract
Generative models for counterfactual outcomes face two key sources of bias. Confounding bias arises when approaches fail to account for systematic differences between those who receive the intervention and those who do not. Misspecification bias arises when methods attempt to address confounding through estimation of an auxiliary model, but specify it incorrectly. We introduce DoubleGen, a doubly robust framework that modifies generative modeling training objectives to mitigate these biases. The new objectives rely on two auxiliaries---a propensity and outcome model---and successfully address confounding bias even if only one of them is correct. We provide finite-sample guarantees for this robustness property. We further establish conditions under which DoubleGen achieves oracle optimality---matching the convergence rates standard approaches would enjoy if interventional data were available---and minimax rate optimality. We illustrate DoubleGen with three examples: diffusion models, flow matching, and autoregressive language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new counterfactual generation framework that ensures doubly-robustness. Theoretical properties of the robustness and optimality of the estimates were derived. Empirical experiments on the framework were presented and analyzed.

### Strengths
The paper is the first to propose a generative counterfactual framework that is doubly robust. Previous works typically only focus on at most one aspect.

The framework is completely general-purpose and can be used with any generative models, making it highly flexible to deploy in different practical tasks.

The theories derived in the paper, including finite-sample PAC bounds and minimax rates, are technically highly nontrivial. The proof techniques are valuable for existing statistical learning and semiparametric theory literature.

The theories provide a clear interpretation of the role of "doubly robustness" in the framework.

### Weaknesses
Framework novelty: the proposed framework (Algorithm 1 and Algorithm 2) seems to take the form of a common framework for doubly-robust estimation. The novelty of the framework is unclear.

Assumptions: the theories hinge on a series of assumptions (C3-C8), involving bounded losses, curvature constants, entropy integrals, and Lipschitz properties of transport maps. These assumptions seem to be stylized, and their practical verifiability remains unknown.

Baselines: the baselines used in the experiments are naive (plug-in, IPW), and no state-of-the-art baselines were used for comparison.

### Questions
Framework novelty: how is the algorithm novel compared to existing doubly robust frameworks?

Assumptions: could the authors justify the reasonability and verifiability of the assumptions that were used in the paper?

Baselines: I suggest that the authors should do an ablation study by comparing different types of generative models other than the diffusion model used in DoubleGen.

### Soundness
2

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
2

### Summary
The paper proposes a counterfactual generation framework/loss called DoubleGen with doubly robustness properties. That is, the framework can properly estimate the counterfactual risk if either the inverse propensity or the conditional transport map is estimated consistently. The authors then give some examples of adapting DoubleGen to different types of generative models, such as diffusion and autoregressive models. They then provide a theoretical analysis that shows that the resulting generalization error between the estimated and true risk is small with high probability. Finally, the authors conduct numerical experiments on two datasets, demonstrating that DoubleGen has more robust performance under misspecification settings.

### Strengths
•	The proposed framework can be generalized to a number of generative models with different architectures, highlighting its flexibility and potential applicability across diverse domains.

•	The authors provide rigorous proofs to validate the generalization and minimax lower bounds for the counterfactual generation problem.

•	The conditions/assumptions used in Section 5.2 mostly make sense and allow for nonparametric nuisance estimation while still providing finite-sample guarantees.

•	The celebA example serves as an intuitive and straightforward illustration for confounding bias.

### Weaknesses
•	The notation/symbols in the introduction section needs more clarification to improve readability. For instance, if $P$ represents the factual distribution, then it would be better to use $P^*$ rather than $\mathbb{P}$ to represent the counterfactual distribution.

•	Similarly, the symbols should be better clarified in Sections 4 and 5. For example, it should be noted that $Law(Y)$ means “the distribution of $Y$”. Also, it is recommended to use consistent fonts for variables (e.g., $X$, $Y$, $A$), distributions (e.g., $\mathbb{P}$), and spaces (e.g., $\mathcal{Y}$, $\mathcal{U}$).

### Questions
Is it possible to extend the framework to situations where there exist unmeasured confounding and/or interference (i.e., the treatment on one unit can affect the outcome of another)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper address the problem of training generative models of counterfactuals using observational data, in which the intervention is not assigned randomly, but rather depends on some underlying attributes of the population. The authors define a general class of generative models (OracleGen) and then propose how the loss function for these models can be adjusted to allow for doubly robust generation using samples from a confounded distribution. The authors provide some results about the theoretical guarantees of their proposed approach, although verifying whether these are correct is beyond my expertise. The authors then showcase some empirical results showing that their approach works better than naive approaches and IPW.

The theoretical results presented in this paper falls outside of my area of expertise, hence in what follows I will focus my comments only on the proposed setup and experimental evaluation.

### Strengths
The most important contributions of this paper seem to lie in the theoretical guarantees it derives for the consider counterfactual generation setting. However, I do not feel qualified to evaluate whether these results are correct and/or novel.

### Weaknesses
- The experimental results in Table 3 are presented for a single dataset in each modality only. Further, the results seem to be run over a single seed only, making it difficult to establish any notion of statistical significance of the presented results.
- For section 6.1, from the description of the performance metrics in the appendix (l. 2089) it seems to me that the test set was obtained from the confounded dataset in a way very analogous to the way that the outcome model was constructed (using doubly robust generation, under the same models). If this is the case, one could expect some notion of "congeniality bias" in the results (there results are good because the model was fitted using the same principles and assumptions used to construct the test set). Have the authors considered other methods for constructing the test set?
- For the experiments in section 6.2, I am not sure to what extent PPL is a good evaluation metric in this context, as it seems to measure how "confident" the LLM was in its own prediction (unless an external LLM was used to evaluate the quality of the generations, which is not stated in the paper). That does not seem to account for whether the generated samples match the counterfactual distribution or not.
- It would be great if the descriptions of the evaluation metrics used in the paper were a bit more self-contained.

### Questions
Some things which were not clear to me when reading the paper include:
- What is the space of the interventions $A$? Are the authors only considering binary interventions $A \in \{0, 1\}$? In line 144, why do we consider $A=1$?
- In Algorithm 2, why is the empirical risk computed over two disjoint sets $\mathcal{Z}_n^1$ and $\mathcal{Z}_n^2$? In a standard doubly-robust estimators we usually want to estimate the propensity score function and the outcome model using disjoint subsets of the data, however it does not seem to me from the formulation of $R_n(\theta)$ which of $\mathcal{Z}_n^j$ is used for estimating the propensity and which is used for estimating the outcome model. Also, why do we need two outcome models $\psi_n^1$ and $\psi_n^2$?

### Soundness
2

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
This paper studies confounding and misspecification in generative models that aim to sample from counterfactual distributions. The authors propose DoubleGen, a doubly robust training framework that learns to generate counterfactual outcomes from observational data with confounding. The paper provides theoretical support and demonstrates the method on diffusion models and language models.

### Strengths
1.The paper proposes a doubly robust training framework for counterfactual generation called DoubleGen and claims that the generator remains consistent as long as either the propensity model or the outcome model is correct, without requiring both to be correct.

2.The paper provides theoretical support, including a finite sample generalization bound for the DoubleGen risk, a guarantee that links excess risk to closeness to the target interventional distribution, and a result that in the diffusion setting the method achieves a near minimax optimal rate.

3.The paper motivates the task using clear counterfactual scenarios such as what would each person look like if everyone smiled and frames this as causal counterfactual generation rather than ordinary attribute editing or style transfer.

### Weaknesses
1. In the CelebA experiment, the paper highlights specific confounding attributes such as lipstick, female, and no beard, and argues that naive generation entangles these attributes with smiling. However, the quantitative evaluation later focuses on global realism and diversity metrics such as FAD, KID, precision, and recall. The paper does not report whether the attribute distributions themselves were corrected toward the intended counterfactual target, so it is unclear if the method actually removes the identified confounding.


2. The theoretical results rely on a large set of conditions (C1-C20), which are distributed across the main paper and the appendix. The paper does not sufficiently discuss the practical meaning of these assumptions or how a practitioner could verify them. For example, the localized bound depends on condition C15, which assumes that at least one of the nuisance models is not estimated poorly.

### Questions
Q1. The theory shows bounds in terms of the error of the estimated propensity model and the distance between the estimated outcome model and the oracle outcome model, but in experiments you only present two regimes described as well specified and misspecified. Could the authors quantify nuisance quality more explicitly, for example by reporting calibration error, KL divergence, mean squared error, conditional log-likelihood, FID, or perplexity for each nuisance model?

Q2. The method assumes positivity, namely that every covariate profile has some non-zero chance to receive the intervention of interest. CelebA is relatively balanced for many attributes and the text experiment is semi-controlled, so violations of this assumption may be limited in the reported settings. Could the authors investigate what happens when the treatment probability is near zero for some subpopulation? 

Q3. Algorithm 2 involves first fitting the nuisance models, then performing a cross-fit style data split, and then sampling minibatches together with latent noise in order to obtain unbiased gradient estimates of the proposed objective. This appears more expensive than training a single diffusion model or a single language model with a standard loss. Could the authors quantify the additional wall-clock time and GPU memory required by DoubleGen compared to naive finetuning and compared to an IPW-only baseline? Could the authors clarify whether the dominant cost comes from training the nuisance models or from optimizing the DoubleGen objective itself? For the language modeling setup with LoRA finetuning, could the authors report the approximate GPU-hours? 

Q4. Table 1 highlights explicit confounding attributes such as lipstick, makeup, and female. However, Table 3 reports global distribution-quality metrics such as FAD and KID rather than directly measuring whether those specific confounding attributes were corrected in the generated counterfactual samples. Could the authors provide a more direct evaluation by reporting, for Naive, IPW, Plug-in, and DoubleGen, the marginal frequencies of the attributes listed in Table 1 within the generated samples, and comparing those frequencies to the target counterfactual distribution estimated via reweighting on the test set?

### Soundness
3

### Presentation
4

### Contribution
4
