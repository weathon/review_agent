# The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures

- Avg Score: 5.25
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 3

## Abstract
The assumption of Gaussian or Gaussian mixture data has been extensively exploited in a long series of precise performance analyses of machine learning (ML) methods, on large datasets having comparably numerous samples and features. 
To relax this restrictive assumption, subsequent efforts have been devoted to establish "Gaussian equivalent principles" by studying scenarios of Gaussian universality where the asymptotic performance of ML methods on non-Gaussian data remains unchanged when replaced with Gaussian data having the *same mean and covariance*.
Beyond the realm of Gaussian universality, there are few exact results on how the data distribution affects the learning performance. 

In this article, we provide a precise high-dimensional characterization of empirical risk minimization, for classification under a general mixture data setting of *linear factor models* that extends Gaussian mixtures. 
The Gaussian universality is shown to break down under this setting, in the sense that the asymptotic learning performance depends on the data distribution *beyond* the class means and covariances.
To clarify the limitations of Gaussian universality in the classification of mixture data and to understand the impact of its breakdown, we specify conditions for Gaussian universality and discuss their implications for the choice of loss function.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the Gaussian universality and its breakdown in classification problems in the setting of high-dimensional linear factor mixture models (LFMM). LFMMs is an extension of the Gaussian Mixture Model in the classification problem. The study demonstrates that the Gaussian universality does not hold for LFMM in general. Furthermore, the paper considers the relationship with the squared loss and Gaussian universality, and clarify the importance of selecting an appropriate loss function. The theoretical analysis provides a new perspective on high-dimensional data analysis. Several numerical experiments are conducted to confirm the theoretical findings based on asymptotic theory. More comprehensive numerical experiments and verification using real data are needed.

### Strengths
- Regarding the breakdown of Gaussian universality:
Many previous studies have investigated data that follow a Gaussian distribution and scenarios in which the Gaussian universality holds. Since these assumptions do not necessarily hold for real data, it is necessary to analyse more general situations. In the paper, the authors reveal the conditions under which the Gaussian universality is violated for LFMMs. The results can provide new insights into high-dimensional data analysis. 

- Evaluation of the optimality of the squared loss
It is known that the squared loss is the optimal loss function under the Gaussian mixture models. In contrast to that, this paper discusses the possibility that the squared loss is sub-optimal under a more general LFMMs. It suggests that existing works do not necessarily apply in cases where the Gaussian universality breaks down, and presents a new challenge of selecting an appropriate loss function.

- Numerical experiments:
In order to verify the theoretical results, several numerical experiments are demonstrated. These experiments examine the behaviour of the classifier when the Gaussian universality is violated. In Figure 2, the performance of the square loss and the square hinge loss is compared, and one can see that the numerical result is consistent with the theoretical results. Similarly, Figure 3 suggests that the squared loss may not be the optimal solution, and this is also consistent with the theoretical results.

### Weaknesses
This paper deals with the important problem of the breakdown of Gaussian universality in the classification problem of high-dimensional LFMMs. The following are some of the concerns that should be considered.

- In some theorems and corollaries, is it possible to evaluate the convergence speed? Although the proof in the supplementary material seems to evaluate the convergence speed, it would be better for the reader to understand the paper more deeply if the speed were explicitly stated in the main text.

- Theoretical claims are supported by some numerical experiments. However, these experiments are only verified using synthetic data. It is unclear how much the real-world data deviates from the Gaussian universality. Additional experiments using real data are needed to strengthen the paper's arguments. In addition, in line 53 of the introduction, the possibility of observing Gaussian universality in real data is mentioned. I think a supplementary explanation is needed to relate this statement to the results of the paper.

- This paper clarifies the conditions under which the Gaussian universality breaks down, but does not discuss the extent of this breakdown in detail. Can the extent of the deviation from the Gaussian universality be evaluated as a statistical bias? If the impact of this bias on learning performance can be quantified, it would be very useful in practical applications.

### Questions
See Strengths and Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Given the relevance of Gaussian or Gaussian mixtures for dataset modeling in theoretical machine learning, a recent line of works investigated under which conditions the Gaussian assumption can be justified in the high-dimensional regime. This line of works resulted in a series of contributions showing that a *Gaussian Universality Principle* can be indeed safely invoked in many cases. To show the limitations of such an assumption, the authors propose a classification task on a linear factor mixture model, in which two clouds in a $p$-dimensional space have to be classified. The clouds are constructed by using a linear factor model each so that the signal (i.e., the cloud mean) is determined by $q<p$ independent directions only. It is then proven that in this setting Gaussian universality might break down, and the statistical properties of the estimators obtained by convex ERM are asymptotically characterized.

### Strengths
The paper presents an interesting model, that is completely characterizable in the asymptotic regime, to test and break down Gaussian universality (GU). Amongst the relevant features observed by the authors, the directions in which the signal is absent play no role in the statistics of the preactivations (i.e., the in-distribution performance), and square loss exhibits special robustness with respect to GU. The results in the paper are rigorously proven and the problem is attacked by adopting a leave-one-out strategy. Proofs are given in the Appendix. Finally, I found both the manuscript and the appendix very clear and well-written.

### Weaknesses
The paper claims to be ```the first to characterize the learning performance when the Gaussian universality``` (GU) ```breaks down```. However, in one of the cited papers by [Adomaityte et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html) a similar (in spirit) analysis has been recently performed: the classification problem of two non-Gaussian clouds has been analyzed by using the heuristic replica method, analytically showing that Gaussian universality can indeed break down quite simply: the statistical properties of the estimators are also given therein. Although the type of dataset model adopted is in general different, my impression is that the results are on the same line as the one published in this manuscript: unless I am mistaken, the LFMM includes as special case a subfamily of the setup considered therein. If for example $\mathbf V=\mathbf I_p$ and $e_k=a\eta_k$ with $\eta_k\sim\mathcal N(0,1)$ and $a$ positive random quantity having $\mathbb E[a^2]<+\infty$, one recovers the finite-variance model in [Adomaityte et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html) where GU breaking is observed (note however that in this model $\mathbb E[a^4]=\mathbb E[e_i^2e_j^2]\neq \mathbb E[e_i^2]\mathbb E[e_j^2]=\mathbb E[a^2]^2$ in general).

One might argue that, beyond classification, similar Gaussian universality-breaking results were present in the line of works of [El Karoui et al. (2013)](https://www.pnas.org/doi/10.1073/pnas.1307842110) on regression (see also a different paper by [Adomaityte et al. (2024)](https://arxiv.org/pdf/2309.16476)) although the focus in these works was more on robustness. One weakness of the paper is therefore that it might appear as a complement to the existing contributions in which a lack of GU is observed and characterized in the same asymptotic regime.

### Questions
I list below a series of observations and questions.

- I found a little bit misleading the sentence at lines 139-141: the one dimensional CLT in Eq. (2) is independent of any ERM solution $\hat{\boldsymbol\beta}$.
- The notation in Eq. (3) is not very transparent with respect to the average over $\mathbf g$. It would be maybe more clear to just consider $\mathbb E[f(\mathbf g^\top\boldsymbol\beta)]$ and then specify that $\mathbf g\sim\mathcal N(\mathbb E[\mathbf x|y_{\mathbf x}=C],\mathrm{Cov}[\mathbf x|y_{\mathbf x}=C])$.
- In the discussion of the elliptical distribution case, the quantity $a$ is random (this should maybe specified when introduced). It is not clear to me what the authors refer to when commenting about the application of the CGEP to this case: from what I understand *each data point in the dataset* has a different $a$ from a given distribution, so for example in the paper by [Adomaityte et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html) conditioning on the class does not lead to a distribution that satisfies a CGEP, so elliptic distributions can be seen as another example of a simple setup that ``breaks'' Gaussian universality (which is I think the point of the cited paper).
- From what I understand, the noise terms in Eq. (5) are supposed to be independent: is this correct?
- The possibility of accessing an explicit form for $\tilde{\boldsymbol\beta}$ in Eq. (16) strongly depends on the fact that ridge regularization has been chosen. Unless I am mistaken, it might be worth clarifying this point.
- Is the discrepancy appearing in Fig 1 middle due to finite-size effects? In this case, is this numerically verifiable? Also, in the label, I suppose that 'normal' has to follow 'Middle'.
- Corollary 3 and the following Remark 2 are interesting results, although I was wondering how much they depend on the specific LFMM form of the dataset. Again, in the aforementioned paper by [Adomaityte et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html) a breaking of Gaussian universality is indeed observed also in the classification of two elliptic clouds *with square loss* (see Figure 1 therein). My impression therefore is that the 'robustness' of square loss with respect to GU in Corollary 3 might stem more from the specific model considered in the paper than from some intrinsic property of square loss.
- In line 535, ```anlaysis``` to be changed in ```analysis```.
- In the Appendix, at line 870 what does the notation $\partial_-$ mean? Also, lines 887-888 should be checked. A square bracket is missing at the end of line 1075. At line 1131, $E$ has to be replaced with $\mathbb E$. I think an expectation has to be added under the square root in the definition of $\gamma$ at line 1395.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies on the issue of 'Gaussian universality' for classification of high dimensional data drawn from a Linear Factor Model (LFMM) that constitutes a GMM generalization. It focuses on LFMM classifiers trained using squared loss and ridge regression. The main theoretical result is that, if LFMMs with non-Gaussian informative factors are used, 'Gaussian universality' breaks down.

### Strengths
S1. The paper presents theoretical results related to the Gaussian Universality of classification mixtures for high dimensional data. 

S2. It identifies a case namely, LFMMs with non-Gaussian  informative factors trained using squared loss ridge regression, where Gaussian universality breaks.  

S3. Rigorous proofs are provided.

### Weaknesses
W1. Although the paper is well-structured, it is rather difficult to follow due to its theoretical character. 
Nevertheless, I think that the main theoretical conclusions could be explained in a more comprehensive way.

W2. The authors should also comment on the importance of the presented results from a practical point of view.

### Questions
Q1. See comment W2 above.

Q2. The paper proves a negative theoretical result regarding the squared loss, however a suggestion on the type of loss that could
be considered would add value to the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Paper shows that a mixture of linear factors model with non-Gaussian noise doesn't satisfy "Gaussian universality" in terms of asymptotic analysis. This is what I'd call a negative result, and the authors argue this is significant because previously researchers have shown that in various cases you can get away with swapping in Gaussians when doing asymptotic analysis using some central limit theorem arguments.

### Strengths
Focus: There focus on mixtures of linear factor models is quite reasonable and practical and theoretically satisfying in that they generalize GMMs.

Correctness: I didn't see any problems in the math and it all smelled clean, but I didn't fine tooth comb. 

This seems like a perfectly nice bit of results that I'd like to see published somewhere someday, but I just can't personally argue that it meets the high bar for ICLR in terms of significance.

### Weaknesses
They have to make a non-Gaussian noise assumption, and then the number of non-Gaussian noise variables goes to infinity whereas the number of informative features $q$ stays fixed. Now CLT says that even in the presence of a lot of non-Gaussian noise one can end up with Gaussians, so some might argue it's still surprising that this failed a CLT like story, but this isn't the usual CLT set-up, so perhaps not so surprising that infinite non-Gaussian noise leads to non-Gaussian-swappable asymptotics.

### Questions
The authors declare that they have shown squared error isn't an optimal loss *in this setting* and state that as an important contribution. Why is that surprising though, I mean it's easy to conjure up scenarios where that's true (e.g. Laplacian noise -> L1 loss).  So why should we be surprised if you assume non-Gaussian noise on your features and then tell us squared loss is not optimal?

### Soundness
4

### Presentation
4

### Contribution
2
