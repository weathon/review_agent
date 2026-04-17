# Using  maximal information auxiliary variables to improve synthetic data generation based on TabPFN foundation models

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 8

## Abstract
Synthetic data generation for tabular datasets is shifting toward the use of large, general-purpose foundation models. TabPFN, a state-of-the-art example, uses in-context learning to generate probabilistic predictions conditioned on observed examples in a single forward pass. However, when variables are only weakly associated with others, the model's ability to generate realistic synthetic data deteriorates, as the context examples provide little predictive signal. To address this, we introduce the maximal information auxiliary variable (MIAV) strategy, which increases context information with auxiliary variables constructed by rank-matching random noise variables to real data. We establish theoretical properties of the approach which explain its good performance for weakly associated variables. Additional practical advantages of the MIAV approach include improved computational efficiency and invariance to variable order during the synthetic data generation process. Empirical evaluations, on simulated and real datasets, illustrate how the MIAV strategy improves data generation when compared to direct application of TabPFN, and is competitive against other baselines. To illustrate the generality of the MIAV approach we also present an implementation based on the TabICL model (a more scalable tabular foundation model restricted to classification tasks) for performing synthetic data generation on categorical datasets. Overall, MIAV offers an effective foundation model–based alternative to bespoke synthetic data generators.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper highlights a limitation of current tabular foundations models (and in particular of TabPFN): namely, it does not perform in features that are poorly correlated with the others. 

To overcome this limitation they propose to leverage maximal information auxiliary variables (MIAV). These variables are build from random noise and retain the information about how the datapoints are ordered (from lowest to highest in case of continuous features) with respect to the value of a feature. The authors show that following this construction, each variable $X_j$ conditional on its MIAV is independent of all other variables and the MIAV retains maximal information about $X_j$ in a non-conditional way.

### Strengths
The paper identifies a limitation of a well established model. 

The authors propose a sensible solution that seems to work.

### Weaknesses
1) The paper needs a lot of rewriting to make it clearer and more structured. I give some pointers to some improvements below: 
   - The section Notation and Related Work should be two sections: it is not clear why this is a single one
   - Calling vectors of random variables in slanted boldface and matrices as simple boldface makes it difficult to distinguish between the two of them (the space of the matrices of size $mxn$ is isomorphic wrt to the space of vectors of length $mn$, why not treating everything as vectors for example? it might help wrt the notation_
   - In the notation the symbol $X_{-j}$ is used at line 119 but only introduced at line 122
   - In $q_\theta(x_j^{ts} \mid X_{-j}^ts, X^tr)$ it would be clearer if you just define it as $q_\theta(x_j^{ts} \mid X_{-j}, X^tr)$ as you are removing the feature $j$ from both the training at testing
   - $p$ in equation (1) is not defined (I guess it is the number of features)
   - equation (2), the small $x$ might be missing the $\hat$
   - page 4 the authors talk about the permutation sampling approximation of Janossy pooling: explaining what this means takes 2 lines and would make the paper more accessible 
   - the experimental analysis is just a big wall of text right now and could use some structuring: the authors could consider adding paragraphs titled "Metrics", "Experimental Setup","Datasets" etc. 

2) It is not clear to me why the authors are sorting in line 2 of the algorithm if then the noise needs anyway to be sorted on the ground of the values of the features

3) in both the algorithm and the theorem the a variable $Z$ and vector $z$ appear, which I think are used as $M$ and $m$ but they are not defined anywhere

4) it is not clear how to read the colours in the charts in the last column of Figure 3 

5) At page 6 the authors write that $X_j$ is *completely determined* by $M_j$. I am not sure how that is possible if $M_j$ is essentially just modelling the ordering 

6) In page 6 Equation (4) I do not understand why the authors suddenly use $X_{<j}$ - I thought they were just considering $X_{-j}$.

7) In the experimental analysis  is missing  common metrics used in tabular data generation like utility and sample generation time. Also, in order to measure the statistical fidelity why didn't the authors use the Wasserstein distance for continuous features? That ia very commonly used. 

8) The authors did not consider more recent baselines like GReaT which are transformer based.

Overall, I believe the paper presents an interesting solution but it needs more work before publication.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper identifies a problem with using tabular foundation models like TabPFN to generate tabular data, viz., that their performance degrades significantly when variables have weak statistical associations. The paper then proposes a novel method called the Maximal Information Auxiliary Variable (MIAV) strategy to deal with this problem. The MIAV strategy creates an auxiliary variable for every variable through a rank-matching procedure between the real data and a random noise vector. The core insight is that this auxiliary variable is perfect (in the rank-based sense) predictor of the corresponding real variable. A concise theoretical justification for MIAV is provided and its practical benefits, including computational efficiency (reducing complexity from cubic to linear in the number of features) and invariance to column order are demonstrated through extensive empirical evaluations.

### Strengths
+ The paper clearly identifies the "weak associations" problem.  
+ The MIAV strategy is simple and effective.
+ The strategy reduces computational complexity, which is a practically significant contribution.
+ The Theorem 1 provides strong theoretical grounding for the MIAV strategy. It is also well supported by the extensive evaluations.

### Weaknesses
- The MIAV strategy is an "add on" in the sense that its performance of the overall generator  will  always be limited by the underlying model (TabPFN, TabICL etc.). This could be more explictly stated as a limitation
- The method requires generating MIAV for the entire dataset, including the test set. This could cause information leakage if someone were to apply this method for prediction. Although, footnote 1 on page 7 clarifies that this meant for generation not classification task, it should probably be stated even more emphatically in the paper, to prevent misapplication of this method.

### Questions
1. A uniform distribution is used as the random noise generator for the rank-matching. What happens if the distribution were changed, say to a Gaussian? How sensitive is the method to this choice of distribution?
2. SMOTE seems to be performing well on  fidelity metrics alone, although MIAV is often better when fidelity-privacy trade-off is considered. Can you elaborate on when a practitioner would prefer MIAV over the much simpler SMOTE?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an approach to improve tabular data generation models like TabPFN on the synthetic data generation task. Specifically, the authors develop a maximal information auxiliary variable (MIAV) based framework capable of producing accurate synthetic data even in the context of variables with weak associations, overcoming the scalability issues of TabPFN. Experiments performed on numerous synthetic and real-world datasets demonstrate that the proposed MIAV conditioned TabPFN model is comparable to or better than state-of-the-art synthetic data generation approaches. Overall, the paper develops a novel method that is consistent with the claims made in the paper and will contribute to the development of efficient, high-quality synthetic data generation for tabular data especially with weakly associated and small datasets.

### Strengths
1. The paper is well written and easy to follow. Further, the problem of synthetic data generation of tabular data comprising continuous and categorical variables, that is being tackled, is of prime importance.

2. The proposed MIAV method is simple yet effective in accounting for weak associations along with improving generation scalability. The method is capable of being applied to both continuous and categorical variables. 

3. The paper also develops theory to support its claims of improved scalability and demonstrates state-of-the-art results on numerous synthetic and real-world data compared with multiple baselines.

### Weaknesses
1. Out-of-Distribution generalization is alluded to the the introduction but it is unclear if it has been demonstrated in the paper. 

2. Run-time experiments have been run on small synthetic datasets (e.g., <= 3000 samples).

### Questions
1. How does the proposed model perform in the context of out-of-distribution synthetic-data generation task compared to TabPFN, especially owing to its ability to perform well in the context of variables with weak associations? 

2. Scalability is an important aspect of such models and this is somewhat of a concern for me at the moment. Why have synthetic datasets only been generated with <=3000 instances in the scenario evaluating runtime analysis? Is it difficult to scale the model over these to truly large scale data (e.g., 10000 rows or greater)?  If scaling is difficult, what are the bottlenecks? 

3. How does the current proposed approach scale with the number of columns in the datasets?

### Soundness
3

### Presentation
3

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
This paper focuses on tabular foundation models (like TabPFN) and explores the use of these models for synthetic data generation (as opposed to tabular classification). The authors claim that when such models are used in an ICL (in context learning) mode weak associations between columns can cause poor performance. As a result they propose a so-called maximal information auxiliary variable (MIAV) strategy, which enriches the model’s context using auxiliary variables that they derive through a rank-matching process.

### Strengths
- Focuses on synthetic data generation which appears to be understudied in the tabular foundation model space
- Generalizes across multiple tabular foundation models
- The proposed MIAV appears to be theoretically grounded

### Weaknesses
-  The proposed framework seems highly reminiscent of Bayesian structure discovery algorithms which also need to learn an ordering of variables to support better sampling (e.g., for MCMC methods). Can you comment on the relationships (or lack thereof) to that literature?
- The idea of creating auxiliary variables is good but it seems orthogonal to seeing whether a different ordering of variables will work. For instance, see this work: https://arxiv.org/abs/2406.14541. Can you comment on when we will need a different ordering vs when your approach is needed?

### Questions
- Please address the above questions from the Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3
