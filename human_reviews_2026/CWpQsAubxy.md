# ActiveCQ: Active Estimation of Causal Quantities

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Estimating causal quantities (CQs) typically requires large datasets, which can be expensive to obtain, especially when measuring individual outcomes is costly. This challenge highlights the importance of sample-efficient active learning strategies. To address the narrow focus of prior work on the conditional average treatment effect, we formalize the broader task of Active Estimation of Causal Quantities (ActiveCQ) and propose a unified framework for this general problem. Built upon the insight that many CQs are integrals of regression functions, our framework models the regression function with a Gaussian process. For the distribution component, we explore both a baseline using explicit density estimators and a more integrated method using conditional mean embeddings in a reproducing kernel Hilbert space. This latter approach offers key advantages: it bypasses explicit density estimation, operates within the same function space as the GP, and adaptively refines the distributional model after each update. Our framework enables the principled derivation of acquisition strategies from the CQ's posterior uncertainty; we instantiate this principle with two utility functions based on information gain and total variance reduction. A range of simulated and semi-synthetic experiments demonstrate that our proposed framework significantly outperforms relevant baselines, achieving substantial gains in sample efficiency across a variety of CQs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Strengths:This work makes both remarkable conceptual and methodological contributions: it delineates the new task of Active Estimation of Multiple Causal Quantities in a form that is immediately applicable to the real-world challenge of costly outcome measurement and begins closing the theoretical gap in CQ-aware active learning. On a methodological front, it would be difficult to find a more innovative way of marrying GP and CME than this paper, producing a coherent framework for uncertainty modeling in the RKHS, where analytic integration over target distributions replaces Monte Carlo sampling noise. Meanwhile,the paper sets a good basis in theory for consolidation and better portability to further acquisition strategies by summarizing the various causal quantities ATE, CATE, ATT, and DS into one integral formulation.Weaknesses:The paper remains thin on empirical validation; most experiments are run on simulated and, at best, semi-synthetic datasets (IHDP, LaLonde) without any observational reality, so it is challenging to talk about the paper with practical implications. Besides, the "compatibility of the GP kernel space with the conditional embedding" is claimed but not empirically tested, so this essential assumption is not empirically validated. Third, while the author blames the distributional mismatch for the poor performance of the methods reviewed, such a mismatch is not quantified, nor is any evidence provided that the performance boost observed for ActiveCQ results from addressing such an issue. Therefore, there is a logical gap in the causal reasoning

### Strengths
na

### Weaknesses
na

### Questions
na

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the problem of active learning when the goal is estimation of causal quantities like the average treatment effect (ATE) or conditional average treatment effect (CATE). More precisely, the authors consider settings in which there is available a (generally small) dataset containing outcomes $y$, treatments $a$, confounders $s$, and effect modifiers $z$, and also a larger dataset containing the same observables other than outcomes. One can obtain outcomes for selected observations in the second larger data set but at some cost. The problem is then to choose the data-point (or set of points) in the large sample, so that obtaining outcome information for that points leads to the greatest reduction in uncertainty in the target causal quantity of interest.

The approach of the authors is Bayesian, with Guassian process priors on the relevant features of the joint distribution of the observables and the outcomes assumed to be conditionally Gaussian with fixed conditional variance. The use of RKHS leads to tractable formulations for the posterior distribution of the causal quantities of interest. Points can be selected to achieve the greatest expected reduction in posterior uncertainty, as measured by either information gain or a total variance reduction criterion. A greedy algorithm is proposed for tractable selection of batches of points (selecting the optimal batch of $m$ points out of a large sample is computationally infeasible when $m$ is anything other than very small, for obvious combinatoric reasons). The performance of the method is assessed against benchmarks on semi-synthetic and simulated data and appears to perform very favourably.

In my reading, the key technical contribution here is as follows. Causal quantities depend on both the regression function ($E[y|a,z,s]$ in the authors' notation) and often also on the joint/conditional distribution of $z$, $s$ and (possibly) $a$. Two examples of causal quantities that depend on the latter are the ATE and CATE conditional only on $z$ (i.e., $E[y|1,z]-E[y|0,z]$). For standard active learning, it generally suffices to target the posterior of the regression function. The authors' approach is premised on the observation that many causal quantities of interest can be written as integrals of the regression function and the joint distribution or related objects. A key insight of the paper is then that, by expressing the joint distribution in terms of conditional kernel mean embeddings (CME) (an approach employed in some other recent papers on causal learning in RKHS), an analytic expression for the posterior of the causal quantities can still be obtained. The authors do consider an alternative based on kernel density estimation and Monte Carlo sampling, but the CME approach appears more tractable and elegant.

### Strengths
I enjoyed reading the paper. I found it to be well-written. The discussion of the analysis and approach is thorough. The question is novel and the methods are innovative and practically relevant. The analysis is very elegant. I found myself convinced that the methods proposed by the authors are effective.

### Weaknesses
The could be a little more detail given regarding possible real-world applications. The figures are too small. The grammar in some parts of the appendices needs fixing.

### Questions
I wondered which if any of the simulated/semi-synthetic datasets violate the assumption that the outcome is conditionally Gaussian with fixed variance? While the likelihood is misspecified if this is the case, the comparison on such potentially less favorable datasets might be quite informative as in the real world conditionally heteroskedastic errors may be very common. 

One reason that the CATE may be of interest is that knowledge of the CATE can allow researchers to optimally allocate treatment in some new population on the basis of the covariates $z$. More precisely, one may hope to choose a treatment assignment rule $\pi$, (where $\pi(z)\in \lbrace 0,1 \rbrace $) that achieves high expected utility compared to the baseline of no treatment in the new population. For example, equating the ITE with the utility gain from treatment, one would hope to achieve a large value of the objective  $\ell[\pi]=\int E[y|a=\pi(z),z]-E[y|a=0,z]d \tilde{P}_z (z)$, where $\tilde{P}_z$ is the distribution of $z$ in the new population which one may take as known. While this objective depends on the unknown CATE, given sample data, the posterior distribution of this object can be evaluated, and one could find the treatment allocation rule (perhaps in some restricted class) that maximizes the posterior expectation of this object (this is the approach in Bayesian statistical decision theory). This suggests an alternative (to information gain and the total variance) objective one could use to select the points whose outcomes one should sample, namely the supremum over $\pi$ of the posterior expectation of $\ell[\pi]$. I wondered if the methods in the paper would still apply for this objective?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ActiveCQ, a pool-based active learning approach tailored towards estimating causal quantities of interest. The approach is observational, selecting “unlabeled” items from a pool to query their outcomes (labels), as opposed to carrying out interventions.  ActiveCQ follows a Bayesian active learning approach, where uncertainty about the causal quantity of interest is minimized. A key insight of the proposed approach is that if the outcome is modeled as a Gaussian process (GP), the causal quantity of interest is also a GP, with mean and covariance functions derived in equations (4), along with two sample-based estimators based on conditional density estimation and conditional kernel mean embeddings.  This insight allows utilizing acquisition functions from active learning with Gaussian processes, along with a (suitably adapted) theoretical analysis, to this novel setting.  The approach is evaluated in a quite extensive set of experiments.

### Strengths
- The specific instance of active learning considered in this paper is interesting, practically relevant and has not been studied yet to the best of my knowledge
- The approach is principled, and accompanied by a theoretical analysis
- The experiments and baselines are quite comprehensive
- The paper is quite well written (except some acronyms, such as SUTVA, should be defined to be more self-contained)

### Weaknesses
- The paper is primarily a natural composition of ideas that have individually been fairly well explored, thus perhaps not exceedingly surprising to researchers in the area.  Nevertheless, I find it makes a relevant contribution.

### Questions
- The figures are tiny (especially Fig 3)... perhaps it would make sense to move some to the appendix?

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
This paper proposes the first unified framework for active estimation of multiple causal estimands, including ATE, ATT, CATE and ATE under distribution shift. The framework is a Bayesian framework with uncertainty quantification, which is supported by solid theoretical grounding.   Empirical validation on several synthetic or semi-synthetic datasets shows the effectiveness of the framework.

### Strengths
1. This paper is novel in task formalization, unified modeling of causal estimands, and creative integration of Gaussian Processes and Conditional Mean Embeddings to balance uncertainty quantification with computational efficiency in active learning.
2. The paper demonstrates exceptional quality through theoretical rigor, robust experimental design, and transparent reproducibility practices.
3. The paper is exceptionally clear, making its technical content accessible to both experts in causal inference and researchers new to the field.

### Weaknesses
1. This paper acknowledges standard GPs’ cubic complexity and mentions efficient alternatives (sparse variational GPs, Random Fourier Features, Nyström). However, no empirical evaluation of these methods is provided.
2. The CME’s regularization parameter $\lambda=0.01$ is fixed with no sensitivity analysis.
3. If I am right, the greedy strategy for batch selection still requires iteratively re-computing the posterior for all remaining pool points after each selection, which is not efficient enough.

### Questions
I would suggest using a alternative term for the word "causal quantity". In potential outcome framework, a more common definition is "causal estimand", indicating that we aim to estimate the value.

### Soundness
4

### Presentation
3

### Contribution
4
