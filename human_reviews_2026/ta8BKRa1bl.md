# On the identifiability of causal graphs with multiple environments

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 10, 2, 8, 6

## Abstract
Causal discovery from i.i.d. observational data is known to be generally ill-posed. We demonstrate that if we have access to the distribution induced by a structural causal model, and additional data from (in the best case)  *only two* environments that sufficiently differ in the noise statistics, the unique causal graph is identifiable. Notably, this is the first result in the literature that guarantees the entire causal graph recovery with a constant number of environments and arbitrary nonlinear mechanisms. Our only constraint is the Gaussianity of the noise terms; however, we propose potential ways to relax this requirement. Of interest on its own, we expand on the well-known duality between independent component analysis (ICA) and causal discovery; recent advancements have shown that nonlinear ICA can be solved from multiple environments, at least as many as the number of sources: we show that the same can be achieved for causal discovery while having access to much less auxiliary information.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the condition for the unique acyclic causal graph identification from multiple environments.  Under assumptions around nonlinear ICA, the authors show that only two sufficiently different environments is enough for the identifiability.  The technique is by probing the Jacobian mixing functions' support.

### Strengths
1. The **main theorem is well motivated and looks correct to me;** I didn't check for details though.

2. The authors **do a good job in discussing the connection between causal discovery and independent component analysis (ICA),** and explain why causal discovery can be easier than full ICA recovery -- one cares the support of the mixing matrix at one point, while another cares the independent sources recovery at each point.

3. **The overall presentation is good.**  The assumptions are clearly listed, and the writing is still clear with heavy notations.

### Weaknesses
1. **My major concern is that the assumptions are way too strong, untestable, and goes a bit against the core goal interventional causal discovery:**
- Though the conclusion "only two environments needed; this number does not depend to the number of vertices in graph" may sound appealing at first, it comes with the price that the causal mechanisms (the entire mixing function) must remain fixed across all environments, the noise components are required to be Gaussian, and only variance rescaling is allowed between interventions.
- These conditions are too strong merely for the goal of "two environments needed".  **This goal itself somewhat goes against the goals for both interventional causal discovery and the characterization of environments needed:**
   - **For interventional causal discovery,** the main idea is to fully use the available information from multiple environments' data (while allow the changes across environments to be arbitrarily flexible).  It is good as long as we can identify more than from only one environment (e.g.,  the CPDAG).  The goal is not the exact DAG recovery at any cost, but rather to make full use of available information in practice. Otherwise, if the goal is just exact DAG recovery even with model misspecification, why bother to use the difficult estimations involved in this work, instead of just running LiNGAM on one domain?
   - **For characterization of environments needed** for exact DAG recovery, or other characterizations to the sizes of the equivalence class under interventions: the main purpose is to characterize the internal randomness from data induced by the causal graph, and to help design experiments -- e.g., about how many interventions and which targets are needed for the exact DAG recovery, so as to do the exact effect estimation.  The dependence of this number to the graph size is not a bad thing.  And again, experiments in real world need to be flexible and the strict assumptions in this work are not desired.

2. **Assumptions can be stated more clearly:**

For instance, in Assumption 2 about noise rescaling:
 - Is mean shift also allowed, as long as the variance changes?
 - Does L have to be diagonal, or is it enough to be orthogonal against Si's diagonal variances?

3. **Experiments are limited to bivariate case with synthetic data.**  This is minor and acceptable though, given the theoretical focus of this paper.  The authors also acknowledge this in the appendix, which is fair.

### Questions
Could the authors elaborate more on the role of acyclicity in their framework?

Given the close connection between causal discovery and ICA, I am curious whether acyclicity is truly essential for the results in this work.  For instance, In the linear non-Gaussian case, as shown in https://arxiv.org/abs/1206.3273, the presence of cycles does not create technical difficulties.  One can simply run ICA and interpret the resulting demixing matrix differently (not necessarily to be lower triangular;  as long as it's nonzero entries on diagonal). 

Is the same reasoning applicable in the nonlinear ICA setting here, or does acyclicity play some more essential roles here?


And minor, at L133 notations, "Also, we use [d] := {1,...,d}." should be stated before being used at "supp(M) := {(i,j)|i∈[m],j ∈[n] ".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper proposes novel results for causal discovery from multi-environmental data, by relying on the relationship between representational identifiability and the identifiability of the causal graph. Their resulst provides novel insights into this relationship by characterizing the complexity of the causal discovery task in terms of the sufficient variability conditions prevalent in the ICA and CRL literatures.

Although the paper uses similar proof techniques as prior work (relying on the score and the support of the Jacobian), I believe their insight is extremely important (not to mention that it works for Gaussian sources, which is often a barraier to identifiability). The authors provide synethetic experiments to corroborate their findings (that causal discovery is independent of the number of performance, when their assumptions are met).

I have a few minor remarks for improvement (see below); nonetheless, **I am strongly in favor of the acceptance of the paper, and would recommend it for an oral if my concerns are addressed.**

### Strengths
- The paper is generally well-written
- The assumptions are clearly stated
- The theoretical results are well explained and contextualized (the "relation with ICA identifiability paragraph is very instructive")
- The experiments back up the theoretical findings (and even provide insights into how robust the findings are, i.e., when some of the assumptions are violated - I especially liked the ones with gamma distribution and vanishing/not vanishing gradients in E.4)
- The topic is important, especially that it shows a fundamental (and quantifiable) connection between representation identifiability (ICA) and causal discovery - in terms of the number of environments

### Weaknesses
My concerns are mainly about phrasing and writing; thus, minor (also see my clarifying questions below).

- abstract: please make it clear *two environments* is your best-case scenario. As of now, the current phrasing in the abstract might be misleading.
- L226: please put a reference where you discuss why Assm.4 is needed
	- In general, I'd consider putting all assumptions into a single listing
- Lem. 1.: for insights about the score function, there was previous work by Burak Varici and collaborators, I strongly recommend citing their papers.

### Questions
- What does "distribution of a structural causal model" mean in the abstract?
- L046: How do you prove that your results holds for arbitrary SCMs?
- L363: Could you please explain why the SCMs you used cannot be reparametrized as PNL or location-scale models?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers the problem of recovering the causal graph in nonlinear structural causal model based on nonlinear independent analysis. Specifically, unlike previous results on causal discovery using nonlinear ICA, where the number of environments is required to be proportional to the number of sources for identifiability of the full model, the authors show that, if the task is to only recover the causal relations, under certain assumptions on the data generating mechanism, only three environments are required. Further, the authors provide an algorithm for graph identifiability based on estimation of the Jacobin matrix of the mixing function, and demonstrated the effectiveness of the proposed method through simulations based on synthetic data.

### Strengths
1. The problem formulation is clear. Specifically, the authors clearly state that the task is to identify the causal graph, which is different from causal discovery where the task is to identify the full causal model.
2. The theoretical results are clearly explained with detailed proofs.

### Weaknesses
1. Assumption 2 is very restrictive, making the theoretical results not novel. Specifically, Assumption 2 requires that the noises across the environments are "rescaled", i.e., $s_i^{k}=\lambda_i^{(k)} s_i^{0}$ for all $i,k$, where $s_i^k$ represents the noise term $s_i$ in environment $k$. This is a very strong assumption and asserts dependencies among the noises across environments. On the contrary, most of the existing results on nonlinear ICA only assumes that the noises are jointly gaussian with environemnt-specific covariance matrix.
2. The setting in the simulation results is over-simplified, which only considers the graph with two variables. In this case, the graph structure only has three possible choices, making the recovery task much easier. It would be better if the author could test the performance with more observed variables (say around 5). Also, it lacks comparison with baseline methods that use nonlinear ICA for causal discovery.

### Questions
1. How are $L_i$ selected in the numerical simulation? Specifically, in the settings with 6 and 9 environments, are $L_i$ different across all environments?
2. Since the task is to recover the graph structure instead of the causal model, can the theoretical results provided in this paper be extended to the case where the causal graphs across environments are the same but the functional mechanisms are different? Note that some of the the existing works allow for different mechanisms across environments, such as Jaber et al. (2020).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors establish that "the full causal graph of an invertible SCM with arbitrary nonlinear mechanisms is identifiable from the model’s distribution and data gathered from only two sufficiently variant environments", providing the first theoretical guarantee for complete causal graph recovery under a constant number of environments, in contrast to prior nonlinear identifiability results which require #environmental which scales linearly with data dimension. The analysis assumes Gaussian noise, yet the authors discuss how to relax this condition to a more general class of noise distribution (having at least one point with zero gradient of the likelihood, not necessarily at the mean as for Gaussian) . Their proof introduces a novel application of the recently highlighted duality between (non-linear) SCM and ICA identifiability, showing that causal discovery requires far less auxiliary information than nonlinear ICA. Empirical validation on synthetic data corroborates the theory, showing successful causal direction recovery in previously non-identifiable bivariate cases.

Overall assessment: The paper is well written and generally interesting. It builds on the recent duality between identifiability in non-linear SCM and ICA and suggests promising directions for advancing identifiability theory and scalable algorithms for high-dimensional, multi-environment causal discovery. While I enjoyed reading the paper, I would have appreciated more comments/remarks/analysis on limitations, for example for unobserved confounders and a more general class of distributions. Nevertheless, I do believe the paper brings useful insight to ICLR, and more specifically causality, research communities.

### Strengths
This work provides a first theoretical result toward identifiability of the full causal structure with as few as two environments with sufficient variability with respect to a base environment as specified in Assumption 5. The novel proof technique of exploiting the established duality of SCMs and ICA brings interesting methodology for subsequent analysis. The paper is well written and manifest a good balance between main body details and further explanations in the appendix. Empirical validation on synthetic data complements the theory, showing successful causal direction recovery in previously non-identifiable bivariate cases.

### Weaknesses
- Recent work shows that overcomplete ICA is beneficial in scenarios with unobserved confounders and a single environment. I think a discussion about the limitation of "extending this work with multiple environments" to causal diagrams with unobserved confounders could be of great interest to the community. For instance, what are the fundamental limitations to extend proof techniques to over complete non-linear ICA (more sources than variables)?

- The authors acknowledged the limitation of "Gaussian noise assumption" and commented on what other noise distributions to which their method extends (at least one point with zero gradient). I would have appreciated also a discussion about the fundamental limitation of the method to wider classes of distributions (i.e., from the perspective of necessary as opposed to sufficient conditions).

### Questions
- I think there is a typo in the equation of Proposition 1: The RHS should be $j\notin {\text{PA}}_{I}$, no?
- I think using an index for environment (say t or r)  different than that of variables (i) could make your presentation less confusing; see for example definition 4 and how it references equation (1). Also, a superscript for environment versus a subscript for variable could be defined in the notation section at the beginning of Section 3.1. 
- I think that a closer mathematical explanation in the last part of Section 4.1 (Theorem 1 beyond Gaussianity) could be beneficial to inexperienced reader.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the identifiability of causal graphs when data come from multiple environments. The authors prove that for structural causal models (SCMs) with arbitrary nonlinear mechanisms and Gaussian noise, the full causal graph can be uniquely identified using data from only two distinct environments. The work builds on the theoretical duality between Independent Component Analysis (ICA) and causal discovery, offering a new identifiability result that links second-order statistics (Hessians of log-likelihoods) across environments to the Jacobian structure of the causal model. Synthetic experiments on bivariate models validate the theoretical claims.

### Strengths
I appreciate the authors’ clear and well-organized presentation of their work. The paper carefully states its assumptions, proofs, and detailed algorithms, demonstrating both theoretical rigor and practical insight. Most notably, it shows that full causal graph identifiability can be achieved without the number of required environments scaling with dimensionality—a highly appealing property for real-world applications, which I consider the paper’s most significant strength.

### Weaknesses
So far, the largest weakness of the paper is the narrowness of the experiment. They only test on simple two-variable synthetic examples. Although the authors acknowledge this as a limitation, I believe that experiments on multivariate settings are necessary. Without trying it on higher-dimensional problems or actual datasets, it's hard to gauge whether this really works in practice.

Another potential weakness lies in the assumption of Gaussian noise variables. At first glance, this may not seem overly restrictive for two reasons.

1. Noise is often regarded as the sum of many small unobserved factors, which should approximate normality according to the central limit theorem.

2. Even if the true noise is non-Gaussian, it can theoretically be expressed as an invertible transformation of a Gaussian variable, and such a mapping could be absorbed into the causal mechanisms without affecting the identifiability of the causal graph.

However, the non-Gaussian experiments reported in the appendix raise some concerns—the results are not entirely convincing. This suggests that, in practice, the Gaussianity assumption may indeed be a substantive limitation of the current work.

### Questions
1. For proposition~1, I believe there is a typo that should be $J_f^{-1}(x)_{ij} \not=0$.

2. Can the authors discuss the potential reasons why including more domains causes performance to decrease? I agree that in theory, two domains are enough. However, for nearly all CRL papers, we benefited from additional domains. Is that an optimization issue?

### Soundness
3

### Presentation
3

### Contribution
3
