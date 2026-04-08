## Human Reviewer 1

### Summary
This work extends the recent trend of information theoretic diffusion based estimators to discrete data.
The approach is "decomposable" in the sense that a single training run to estimate the mutual information of a vector allows for the estimation of that of any subvector.
Empirical evaluation is extensive, both accuracy of estimation and quality of learned representations are evaluated in multiple environment of increasing scale.

### Strengths
+ The paper is well written.
+ Advanced stochastic calculus concepts are explained simply, making the paper accessible to wider audience.
+ Estimation of mutual information on discrete data is promising and useful.
+ Amortized inference on subsequence is bound to be useful in many situations.
+ Evaluation on genomic data is a beautiful application of the estimator.

### Weaknesses
Most of the weaknesses have to do with the empirical evaluation.
+ Table 1: Missing error bars.
+ Evaluations are mostly presented as a graph which can be misleading.
+ Table 2: Correlation with human metrics only report INFO-SEDD and no baselines.
+ Overall evaluation feels somewhat indirect. While having the merit of being grounded in real life problems,  the empirical evaluation could be made stronger by evaluating the estimator itself as an objective for a clearly defined machine learning downstream tasks.

### Questions
Can do author suggest more direct application of Info-SEDD to machine learning tasks?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper proposes a new algorithm for estimating the KL divergence (and consequently, entropy and mutual information) in high-dimensional discrete distributions. The method expresses the KL divergence as a conditional expectation and applies Dynkin’s formula to derive a KL estimator that depends on two density ratios. Each density ratio is then approximated using a diffusion-weighted denoising score entropy loss. The method is evaluated across a range of mutual information and entropy estimation tasks.

### Strengths
The proposed method for estimating mutual information in discrete data is novel. The reviewer is not aware of any previous works that apply discrete (CTMC-based) diffusion to the mutual information estimation task.
The method shows potential practical utility. The results on estimating the entropy of Ising models are promising, and the reviewer believes the approach could be applied to study phase transitions in more complex statistical physics systems with discrete microstate space. As another practical utility one can apply the proposed method to learning neural sufficient statistics and representation learning.

### Weaknesses
It is unclear whether the proposed mutual information estimator is consistent and unbiased. Additionally, although the error analysis is included in the supplementary material, it is not discussed in the main text.

In general, the theoretical contribution of the proposed method is limited and primarily based on a reformulation of the results by Lou et al [1]. For example, the proposed KL estimator has the same form as the diffusion-weighted denoising score entropy (Equation 10 in Lou et al.), with the main difference being the replacement of the density ratio. Moreover, the use of Dynkin’s formula was already presented in Lou et al. (Theorem 3.6).

### Questions
Could you provide a proof that the proposed estimator is consistent and unbiased? For example using a similar strategy as the proof of consistency of the score entropy loss in Lou et. al. (Proposition 3.2).

What is the computational complexity of the proposed estimator?
Regarding the broader impact, the paper would benefit from a discussion on sufficient statistics learning (e.g., see [2]). Including a computational complexity analysis would clarify whether the method can be feasibly used as a loss function in downstream tasks.

[1] Lou et. al. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution, 2020.

[2] Y Chen et. al. Neural Approximate Sufficient Statistics for Implicit Models, 2020.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors propose a discrete diffusion-based approach that mitigates the problem of continuous space MI estimators in high data dimensionality. The proposed approach builds upon Dynkin’s lemma and can be used to estimate mutual information or entropy. Numerical experiments validate the proposed estimator compared to other state-of-the-art estimators. The authors design consistency tests for real-world textual and genomics data.

### Strengths
- Good theoretical contributions (including Appendix E)
- Promising experimental results
- The proposed approach differs from many continuous existing MI estimators
- The proposed estimator appear to be converging faster than other sota estimators

### Weaknesses
- The paper is not easy to read
- In line 200 you write that expressing the MI as the KL divergence requires training two score models. However, many MI estimators target the estimation of the density ratio. So this sentence should be rewritten.
- Line 1211 typo: $\mathcal{X}|$
- The code is not provided

### Questions
- What is $\delta()$ in equation 1? It is not defined
- Table 1 does not report standard deviation information. Can the authors provide some results that also quantify the variance of the proposed estimator compared to the others?
- The authors claim that INFO-SEDD is lightweight (line 21), but I do not see any result reporting computational times in the main part of the paper. Can the authors provide a table with the computational time comparison of the different estimators? In Appendix C.1.3 the authors report the runtime for one experiment, but it would be meaningful to see these results in the main part of the paper and summarized over multiple scenarios. The authors report that INFO-SEDD uses less memory than SMILE and GAN-DIME, so is “lightweight” referred to this?
- In Appendix C.1.4, Table 3 shows that MINDE overestimates MI for low values. Is this result generalizable for different scenarios? Could you claim that this happens to many generative MI estimators except INFO-SEDD? Or is it just a coincidence?
- How do the other methods perform in the extreme dimensionality experiments in C.1.7?
- In Table 5, increasing $|\mathcal{X}|$ should lead to a better performance of the other estimators, right? Why does the trend appear to be opposite?
- Why did the authors decide to not provide the code?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper focuses on mutual information for high-dimensional discrete data. The score function of a diffusion model is used as a key component to estimate mutual information and entropy from a trained diffusion model. In comparison, prior work on diffusion models for mutual information estimation focuses on continuous random variables. The proposed method INFO-SEDD is implemented in two variants, one that approximates MI as the KL divergence between the product distribution and the marginals and one that compares the conditional $p_{Y|X}$ to $p_Y$. The approaches are benchmarked on synthetic discrete data pairs with high dimensions. Further, the authors consider case studies for text and DNA sequence data.

### Strengths
- The authors consider a rich set of benchmark tasks and potential real-world applications.
- The performance of the proposed method compared to prior work on continuous data seems promising.
- Connecting the continuous time Markov chains and diffusion models to compute MI for discrete data seems new to the reviewer.

### Weaknesses
- The paper is very dense and hard to follow since many parts are pushed to the appendix (see questions and comments below).
- It is not clear what the exact differences to other diffusion-based MI estimators are, e.g., MINDE, and how the authors manage to model discrete distributions compared to prior work that did not cover this case.
- Experiments:
	- It is not clear how the baselines are implemented for Sec. 4.1. Don’t they all focus on continuous data? As a simple baseline, using the plug-in estimator for discrete MI, and some more elaborate simple estimators (e.g., Goebel et al. (2005)) would improve the evaluation.
	- Real-world examples: The examples on text summarization and genomics are interesting, but they do not allow clear inferences about the performance of the MI estimator. The consistency test on the genomics dataset seems to not go beyond classification.
	- Typically, papers on discrete MI discuss the issue of bias in the plug-in estimator, which occurs under high dimensionality (in comparison to the sample size). It would be interesting to see if the proposed estimator can effectively mitigate such issues (see, e.g., Goebel et al. (2005)). The authors do evaluate susceptibility to dimensionality, but adopt very large sample sizes ($10^6$) for D=256 and 1024 and employ a model with 2M parameters. Here again, a comparison to the plug-in estimator and improvements of it to high-dimensions should be compared to.

Minor points:
- To which data is the GP fitted (line 356)?
- What is meant by “simple and uninteresting scenarios” (line 124/125)?

Reference:
- Goebel et al. (2005). An approximation to the distribution of finite sample size mutual information estimates.

### Questions
- The training of the method is not clear. Do the authors use a pre-trained diffusion model? 
- The authors state that MI can be used for model selection. How does it compare to other metrics like the model perplexity?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3