# Subsampled Ensemble Can Improve Generalization Tail Exponentially

- Decision: Reject
- Scores: 8, 6, 6, 3

## Abstract
Ensemble learning is a popular technique to improve the accuracy of machine learning models. It hinges on the rationale that aggregating multiple weak models can lead to better models with lower variance and hence higher stability, especially for discontinuous base learners. In this paper, we provide a new perspective on ensembling. By selecting the best model trained on subsamples via majority voting, we can attain exponentially decaying tails for the excess risk, even if the base learner suffers from slow (i.e., polynomial) decay rates. This tail enhancement power of ensembling is agnostic to the underlying base learner and is stronger than variance reduction in the sense of exhibiting rate improvement. We demonstrate how our ensemble methods can substantially improve out-of-sample performances in a range of examples involving heavy-tailed data or intrinsically slow rates.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents an asymptotic theory of ensembling that focuses on the tail distribution of the excess error, that is, the probability that the generalization error exceeds the best possible one by more than a specified constant.  The paper shows that the proposed ensembles can have exponential tails even if the base algorithm has polynomial tails.  Experiments show that ensembles do not only improve the variance, but can have a dramatic effect on the tails.



/* Raised score after discussion */

### Strengths
The paper is clearly written.  I learned something from this paper (but maybe not what the authors claim.)

### Weaknesses
I believe that the asymptotic viewpoint undermines the theoretical result. Even without going to the refined empirical process literature, we know from Vapnik (Vapnik 1998, Wiley) that if the function L(theta,z) has a finite capacity (VC dim for classification, extensions of VC dim for continuous losses with either bounded range or bounded p-th moments), we have relative uniform convergence results with asymptotically exponential tails.  Hence in that setup, any reasonable base learner will also have exponential tails.  On the other hand, if the VC dim is infinite, we know that it will be very difficult to get any guarantee at all (see also JMLR 11 (2010) 2635-2670). 

I can see therefore two ways to interpret the results of the paper:

* We can consider that the author prove that ensembling alone is as good as empirical risk optimization and variants in the sense that it is going to give exponential tails using possibly bad base algorithms.  However this amounts to saying that averaging is optimization, and we know that averaging can be superior to optimization. 

* We can also recall that the VC exponential bounds are often meaningless for practical data sizes. Okay they're exponential in the very large sample limit, but for anything practical, they don't look exponential.  Ensembling can be a way to improve dramatically these practical tails (the experiments support this point of view). But that also means that the asymptotic perspective of the theory proposed in the paper is not the right way to look at this.

### Questions
1) Can you give a simple self-contained example of a "reasonable" base learner that provably does not have exponential tails but has polynomial ones?  It seems to me that this means that you either have to consider infinite VC cases, or losses with unbounded moments, or algorithms that are far inferior to ERM in the limit.  There might be a way to make a convincing case for the last option, that is, inferior to ERM in the limit, but nevertheless practically reasonable in practical settings.

2) A brief review of the proofs suggests that they might repurposed to argue that ensembling can considerably improve the tails long before the asymptotic regime.  Is this correct?

### Soundness
2

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
3

### Summary
This paper presents a method for ensemble learning. After training B models on subsampled data from the training set, the authors propose two strategies (MoVE and ROVE / ROVEs) to select the best model from the B ensemble members. The authors proved that their method can achieve an exponentially decaying rate for excess risk. Experiments results on synthetic data and six datasets from UCI repository validate the proposed method’s performance in out-of-sample testing.

### Strengths
The authors propose to study an important problem in ensemble learning.

The idea of using majority voting to select the best model in ensemble learning seems interesting.

### Weaknesses
=============

After rebuttal: 

Thank you to the authors for their further response. I appreciate the linear regression example provided in the discussion with Reviewer aq9J. The derivation on the polynomial tail in the excess risk of the base learner and the exponential tail of the proposed method helps me better understand the technical contribution of the work and addresses my previous concerns about the base learner. Additionally, the discussion on mode estimation in models sampling is insightful. I would recommend including this discussion in the final paper, as it strengthens the rationale behind the proposed ensemble method. Taking all this into account, I believe the theoretical contributions in this work outweigh limitations in experiments. I am happy to raise my score.

=============

I have the following concerns/questions regarding the work:

1. Practicality of Theorem Assumptions: in Theorem 1, Eq. (9) is under the assumption that $n_{k, \delta} > 4/5$, which indicate that $p_k^{\text{max}} > 4/5$. This assumption appears quite strong, as it requires the most frequently subsampled model to have a subsample rate above 80%. Could the authors clarify the rationale behind this requirement?

2. In contrast to MoVE which seems to be based on a strong assumption, ROVE/ROVEs presents a more practical strategy for selecting the best model. However, the practical insights of Theorem 2 regarding the excess risk of ROVE/ROVEs remain somewhat unclear. Could the authors provide more detailed practical insights into Theorem 2?

Regarding the implementation details and evaluation, I have the following additional questions:

3. In line 193, the authors mention that "Therefore, $n_{k, \delta}$ taking large values signals the situation where the base learner already generalizes well." However, this statement depends not only on the base learner but also on the choice of $\delta$, i.e., a larger $\delta$ leads to a smaller $\mathcal{E}\_{k,\delta}$, thus a larger $n_{k, \delta}$. Could the authors provide further clarification on this claim?

4. In line 271, the authors discussed the choice of $\epsilon$ as "In our experiments, we find it a good strategy to choose an ε that leads to a maximum likelihood around 1/2." Could the authors elaborate further on this?

5. Details of the Base Learning Algorithm $\mathcal{A}$: The paper assumes that the ensemble members are trained using a generic learning algorithm $\mathcal{A}$ based on the subsampled data. Could the authors clarify details about $\mathcal{A}$? For instance, would the settings in $\mathcal{A}$ (such as learning rate, batch size in gradient-based methods) impact the performance of the proposed framework?

6. Experimental Validation and Comparisons: The current experimental results may be insufficient to support the claims fully. Firstly, one of the main contributions is the proof of an exponential decay rate of excess risk; could the authors present numerical results to validate this aspect? Secondly, the current experiments focus only on comparisons between the proposed method (MoVE and ROVE / ROVEs) and the base model. Including additional ensemble methods, as discussed in Section 4, would strengthen the experimental comparisons. Finally, the appendix mentions that the ensemble members use MLPs with 2-8 hidden layers; it would be helpful to explore whether the framework is applicable to other architectures, such as CNNs or transformers.

### Questions
It would be helpful if the authors could address my questions regarding the practicality, implementation, and evaluation of the proposed method, as detailed above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents an ensemble learning technique that enhances the generalization performance by selecting the best models trained on subsamples via majority voting, resulting in exponentially decaying tails ($C_{2}\gamma^{n/k}$) for excess risk, even when base laerners have slow (polynomial i.e. $C_{1}n^{-\alpha}$) decay rates. The method is agnostic to the underlying base learner  and offers a stronger improvement than traditional variance reduction, providing rate improvement rather than just a constant factor improvement. Experimetns show that the method also effective in scenarios with heavy-tailed data and slow convergence rates.

### Strengths
The paper presents a fresh view on ensemble learning, focusing on exponential improvement in generalization tail bounds rather than the traditional variance reduction approach. 

The paper offers valuable theoretical contributions, e.g. developing a sharper concentration result for U-statistics with binary kernels, learners, performing a sensitivity analysis on the regret, and developing a uniform law of large numbers (LLN) for the class of events of being $\epsilon$-optimal. 

Theoretical insights are complemented by experiments on both real and synthetic data using neural networks and stochastic programs, demonstrating relevance of the theory.

### Weaknesses
**Only one baseline used**: The paper could benefit from including additional baselines beyond just the 'base' model. Since the authors mention variance reduction, it would be useful to incorporate it as a baseline, as well as to provide more detailed comparisons with state-of-the-art ensemble methods across a wider range of datasets and problem types.

**Section 3.1**: The paper only addresses regression problems, using synthetic and real data from UCI. Why not include classification tasks like CIFAR-10, CIFAR-100, or ImageNet? Does the proposed method work for these? If it doesn’t, could you explain why that wasn’t mentioned?

The models used appear to be quite small, with only a few-layer MLPs employed. Exploring more advanced architectures, such as ResNet or WideResNet, could enhance the practical applicability of the findings.

**Real data is too old**:  UCI Machine Learning Repository was launched in 1987 and is too old a benchmark. How about considering latest more challenging regression benchmarks e.g. OpenML Benchmark Datasets (2013), PMLB (Penn Machine Learning Benchmark) (2018) etc.?

**Section 3.2**: The choice using SSA isn’t clearly justified. What is the reason behind this choice? How about doing an abaltion on other methods e.g.  Distributionally Robust Optimization (DRO), RO etc.?

**Computational cost not discussed**: The paper doesn’t clearly address the computational cost of the proposed method, which could be a major concern for large-scale applications.

### Questions
**1)** What does "majority vote" mean for the models? What does it imply in practice? How can models trained on different subsets of data end up producing the same outputs that allow for voting, as seen in the final lines of Alg-1 and Alg-2? In standard ensemble methods, majority voting is based on the predictions of the models. I might not be familiar with this specific approach to ensembling, so I would appreciate a clearer explanation.

**2)** In Algorithm 2 line 231-233, how to get $\theta_{\prime}$ in practice?

**3)** For more questions, see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a novel approach to ensemble learning called MoVE (Majority Vote Ensembling) and ROVE (Retrieval and ϵ-Optimality Vote Ensembling). The approach leverages subsampling to create ensembles that can produce exponentially decaying tails for excess risk, overcoming the polynomial decay limitations typically faced in heavy-tailed distributions. By iteratively training on subsets of the data and combining models through majority voting or likelihood-based selection, the method aims to provide stronger generalization than traditional ensembling techniques, especially in contexts with heavy-tailed data distributions. Empirical evaluations demonstrate improved out-of-sample performance across synthetic and real datasets and various stochastic programming problems.

### Strengths
- Provide a new approach to ensemble learning with exponentially decaying tails for excess risk. 
- The paper presents experimental evaluations to support the theoretical claims.

### Weaknesses
- The theoretical results are quite dense and hard to parse. Need to provide example instantiations to allow the reader to get a better sense of them.
- The proposed algorithms look quite computationally expensive and yet underperform the baseline on real-world tasks.

### Questions
- What happens in the case that the data is not heavy-tailed? Do the gains from MoVE and ROVE become constant factor as before? Can you provide the bounds and the correct selection of k for this case?
- It appears that for the method to work, k (sub-sampled data) has to be selected at a lower rate than n which implies that the ensemble size (B = O(n/k)) will be very large. How do existing generalization bounds for ensembles scale with the ensemble size and what are the consequences for them when B is not constant but rather a function of n?
- It seems like in the real-world dataset settings, ROVE is underperforming the base method. Can you detail out why this is the case and what assumptions seem to be breaking?
- Would it be possible to test out this method with slightly more real world datasets (like MNIST, CIFAR and converting them to a regression problem)? I would like to see the convergence rate of ROVE when compared with traditional ensembles as a function of the number of learners in the ensemble.
- Theorems 1 and 2 are quite dense and hard to parse. Would it be possible to instantiate both the theorems for a few example function classes and loss functions and have them as corollaries?
- In the discrete setting, what is the dependence of the generalization error on the size of the parameter class for the traditional ensembling methods?

### Soundness
2

### Presentation
1

### Contribution
2
