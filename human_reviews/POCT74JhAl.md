# Discriminative Estimation of Total Variation Distance: A Fidelity Auditor for Generative Data

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5

## Abstract
With the proliferation of generative AI and the increasing volume of generative data (also called as synthetic data), assessing the fidelity of generative data has become a critical concern. In this paper, we propose a discriminative approach to estimate the total variation (TV) distance between two distributions as an effective measure of generative data fidelity. Our method quantitatively characterizes the relation between the Bayes risk in classifying two distributions and their TV distance. Therefore, the estimation of total variation distance reduces to that of the Bayes risk. In particular, this paper establishes theoretical results regarding the convergence rate of the estimation error of TV distance between two Gaussian distributions. We demonstrate that, with a specific choice of hypothesis class in classification, a fast convergence rate in estimating the TV distance can be achieved. Specifically, the estimation accuracy of the TV distance is proven to inherently depend on the separation of two Gaussian distributions: smaller estimation errors are achieved when the two Gaussian distributions are farther apart. This phenomenon is also validated empirically through extensive simulations. In the end, we apply this discriminative estimation method to rank fidelity of synthetic image data using the MNIST/CIFAR-10 dataset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper shows how to efficiently compute the total variation (TV) distance between two distributions by deriving an equivalence between the TV distance and the Bayes-Optimal classifier. This immediately leads to a lower bound on the TV distance by choosing a non-optimal classifier that can be learned from data. The paper then proves that under a low noise condition, the lower bound for two Gaussians (which is efficiently computed) converges with a relatively rapid convergence rate. The experiments show that the proposed method outperforms existing methods for computing the TV distance.

### Strengths
The paper is clearly written and the derivation seems novel. According to [7]  the question of how to compute the TV between two Gaussians with different means is an open problem and this paper  presents a partial solution to the problem.

### Weaknesses
The main weakness is significance to a ML audience. The paper attempts to motivate the paper by appealing to the problem of estimating the fidelity of generative models but it is not at all clear that TV is an appropriate distance for such a task. In particular, (1)  modern generative models are not Gaussian distributions and (2) for a Gaussian distribution the estimation of other divergences (in particular the Wasserstein-2 distance and the KL divergence) can be computed in closed form. 

Another weakness is that a more recent version of [7] claims that the open problem has now been solved (see footnote 1 in
https://arxiv.org/pdf/1810.08693). The solution appears in the appendix of:
https://arxiv.org/pdf/2303.04288

### Questions
Can you give an alternative motivation for using TV in ML?

How does your approximation differ from the tight bound of Arbas et al?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the evaluation of generative AI models using the total variation (TV) distance between the distributions of real and synthetic data. The proposed method utilizes the variational representation of the TV distance, which necessitates finding the Bayes classifier. The analytical results are based on the assumption that both distributions are multivariate Gaussian, as specified in Assumption 3.1. The proxy optimal classifier is determined by solving Equation (4), and Theorem 3.4 establishes a bound between the true and estimated values of the TV distances. The paper also presents numerical results using synthetic Gaussian distributions and the MNIST dataset to demonstrate the feasibility of estimating TV distance for the evaluation of generative models.

### Strengths
1- The paper investigates the ranking and evaluation of generative models using total variation distance, which is an interesting approach.

2- Although I have some comments on the theorems in the paper, the theoretical result in Theorem 3.4 on estimating TV distance from samples looks interesting.

### Weaknesses
1- The authors’ approach of using total variation distance to evaluate generated samples is restricted to Gaussian and exponential family models. This assumption might not hold for the distribution of generated data. Even when the samples are processed through an embedding model, the resulting distribution will probably have multiple modes due to the variety of samples and cannot be represented as a unimodal Gaussian or exponential family model. Therefore, the authors’ method may not be capable of accurately estimating the underlying TV distance and can only provide a lower-bound which may be loose in practice.

2- The literature review in the introduction (the second and third paragraphs) misses several existing frameworks for evaluating generative models that are based on established distance measures between probability distributions. The maximum mean discrepancy (MMD), an established distance, has been used in evaluating generative models, and the Kernel Inception Distance is a metric that estimates the MMD between two distributions. In addition, the discriminative approach applied to the Wasserstein distances has also been used in the literature to evaluate generative models. Another note, the FID (Wasserstein distance between Gaussians fitted to real and fake distributions) also represents the variational representation of the Wasserstein distance where the function set in the dual optimization is the set of quadratic functions. These relevant ideas have not been acknowledged in the introduction.

3- Using Assumption 3.1 in Theorem 3.4 does not make much sense, because the authors have already supposed the Gaussian distributions P and Q. Therefore, constants $C_0$ and $\gamma$ should follow from the Gaussian distribution parameters $\mu_1,\mu_2,\Sigma_1, \Sigma_2$. The theorem is supposed to bound these constants $C_0$ and $\gamma$ in terms of the distribution parameters, and should not define them as two general constants without any further analysis.

4- Theorem 3.6 (and also Lemma 3.2) seems an obvious statement, because the Bayes classifier of two distributions $P,Q$ is well-known to be the indicator of $P(x)/Q(x) \leq 1$.      

5- The numerical experiments on real data are insufficient. MNIST is considered a relatively simple dataset. The study could benefit from extending these experiments to more complex datasets such as CelebA, LSUN, or ImageNet.

### Questions
1- Do the authors have numerical results on more complex image datasets than MNIST?

2- Can the authors bound the constants $C,\gamma$ in Assumption 3.1 for multivariate Gaussians in Theorem 3.4?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper addresses the challenge of assessing the fidelity of generative data. It introduces a discriminative method for estimating the total variation (TV) distance between two distributions, which serves as a measure of fidelity. The approach links the estimation of TV distance to the Bayes risk in classification. The authors present theoretical results on the convergence rate of estimation errors for TV distance between Gaussian distributions. It shows that estimation accuracy improves as the separation between the distributions increases. This relationship is supported by empirical simulations. The method is applied to rank the fidelity of synthetic image data using the MNIST dataset.

### Strengths
1.	Total variation is a well-established metric in statistics and can potentially provide a quantitative way to assess the fidelity of generative data. 
2.	The authors link the estimation of TV distance to Bayes risk in classification and provides an effective estimation method based on finite samples. 
3.	The paper shows a fast convergence rate when the two Gaussian distributions are well-separated.

### Weaknesses
1.	It is not clear why focusing on using TV instead of other metrics like Wasserstein distance for evaluating the fidelity of the generated data. For two Gaussians, Wasserstein distance has closed form solution. The paper only listed $f$-divergence in the related work.  
2.	The experimental results are very limited. Is it possible to add some natural image data results, such as CIFAR 10? 
3.	It is not clear how to effectively select the appropriate hypothesis class and classifier in practice to achieve accurate TV distance estimation.
4.	On page 3 below eq. (3), the authors mentioned that “Intuitively, if none of classifies yields  a large lower bound, then the synthetic data can be considered similar to the real data, indicating that their total variation distance is small.” I feel that it is not true, since each classifier only offers a lower bound. Even if all lower bounds are small, it doesn’t guarantee that the TV is small.

### Questions
Could the authors provide more experimental results in real application? For example, using CIFAR10.

### Soundness
3

### Presentation
3

### Contribution
2
