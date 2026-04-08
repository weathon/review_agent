## Human Reviewer 1

### Summary
The paper studies the problem of support recovery from noisy observations in a linear regression setting with Gaussian design. The design is assumed to be centered and isotropic, while the noise is Gaussian with zero mean and a diagonal covariance matrix that takes two possible values: $\sigma_1$ and $\sigma_2$.

The paper focuses on two types of results: first, sufficient conditions for the maximum likelihood estimator (computationally inefficient) to consistently recover the support, considering both the ill-specified and well-specified noise covariance matrix cases; second, sufficient conditions ensuring that the Lasso estimator consistently identifies both the support and the sign of the regression vector.

### Strengths
- The problem of recovering the support of an unknown vector is a fundamental challenge in statistics, frequently regarded as a prototypical case for mathematical results on variable selection. Exploring the information-theoretic and algorithmic limitations of this problem holds, in my opinion, significant interest for the ICLR audience. 

 - The mathematical proofs, as far as I could check, appear to be correct. 

 - The proof of Theorem 3 overcomes some non-trivial technical challenges.

### Weaknesses
- I suspect that **Theorem 1** is not sharp, for two main reasons:
  - When $n_2 = 0$ (i.e., no low-quality data is present), the theorem's condition requires $n_1$ to exceed a quantity involving $\sigma_2$. Intuitively, $\sigma_2$ should not appear in this condition under these circumstances.
  - As noted in lines 265–269, if $\sigma_2 = o(s)$, the condition becomes independent of $\sigma_1$, which seems counterintuitive.

- The conclusion at the top of page 6, *"one unit of high-quality data is worth at most 2 units of low-quality data,"* does not seem justified by **Theorem 1**. The theorem provides only a **sufficient condition** for recovery, but it does not rule out the possibility that the maximum likelihood estimator could detect the support with a smaller value of $n_1$.

- The conditions under which the results are proved, although similar to those required in many papers on this topic, are quite restrictive.

### Questions
- Since in Thm 3 it is required that $s=o(p)$, can we replace $n_{Alg}$ by its asymptotic equivalent $2s\log (p)$?

- Can we generalize Theorem 2 to arbitrary matrices $\Sigma$, just by replacing the LHS of (15) by $\sum_{i=1}^n \log(1+\frac{\delta s}{2\lambda_i(\Sigma)})$?

 - What would happen if, in **Theorems 1 and 2**, we replaced the condition that nonzero elements are equal to one with the condition that they are either $1$ or $-1$? Would sign recovery still hold under the same conditions?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper studies the problem of sparse signal recovery in the mixed-quality data setting: where a subset of the data is observed with low noise, and the complement observed with different (higher, w.l.o.g) noise. They study the information theoretic thresholds of the problem, and demonstrate that the maximum likelihood estimator in the gaussian noise setting has significantly different behaviour depending on whether one is agnostic or not to the noise variances. They also analyze the LASSO and prove that it is robust under this model.

### Strengths
The paper is original, they study a timely problem and do so with an original look at a new variable called the price of quality. I must mention that such problems were already considered in the heterogeneous statistics literature, but they were moreso concerned about inferring the heterogeneities themselves, rather than quantifying downstream implications of standard models on mixed quality data. In this regard, I find this work novel. 

The work is clear, and is of high quality as the authors investigate various aspects of the problem thoroughly. 

The work is significant, especially in the era of AI, where models can often be trained using a mix of real and synthetic data. In particular, the authors clarify their findings and present them as important and non-trivial rules of thumb for machine learning, such as that in the agnostic setting, where one high-quality sample is never worth more than two low-quality samples.

### Weaknesses
I found no major weaknesses in the paper. Following were minor comments/questions: 

- The information-theoretic and algorithmic thresholds in (2) are mainly for Gaussian noise right? Or have these been proven for additive noise more broadly? Please clarify this, and maybe state earlier on in the work that you consider additive independent Gaussian noise.
- Typos: First paragraph after 1.2.1, “In the first part of [our] work ([S]ection 3) …”. In the next paragraph: “, the condition requires that a linear combination of [?] has the form …”
- A bit more clarification as to how \sigma_1, \sigma_2 scale w.r.t other parameters, and remind this throughout the paper.
- Your results in Theorem 1 are mostly relevant for additive (sub)-Gaussian noise, please clarify this and that your sufficient conditions (and therefore findings) are not universal over general additive noise.
- End of page 7, you mention “OGP” but do not explain/state the full name or what it is. It would be good to get more background on that.
- Could you comment on how this relates to existing statistical work on heteroscedastic regression, and whether you expect improvements for algorithms which first estimate the heterogeneities, then proceed to weight samples accordingly?

### Questions
Questions asked in the “Weaknesses” section

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
In this paper, the authors study sparse recovery from noisy linear observations of mixed quality. To be more precise, they seek to recover an $s$-sparse ground-truth $\beta_* \in \mathbb R^p$ from $n$ observations

$Y = X\beta_* + Z \in \mathbb R^n,$

where the measurement process is modelled by a Gaussian matrix $X$, and the entries of $Z$ model random noise and are zero-mean Gaussians. For $n_1+n_2=n$, the first $n_1$ observations $Y_1,…,Y_{n_1}$ are of high quality (modelled by noise variance $\sigma_1^2$) whereas the remaining $n_2$ observations $Y_{n_1+1},…,Y_n$ are of low quality (modelled by noise variance $\sigma_2^2 > \sigma_1^2$). The authors consider the agnostic setting in which the recovery method has no information on the data quality, and the informed setting in which the recovery method knows which noise level applies to which observation.

Theorem 1 provides sufficient conditions on $n_1,n_2,n$ for asymptotic support recovery in the agnostic setting. The result shows that asymptotically a high-quality observation is never worth more than two low-quality observations.

Theorem 2 is the counterpart of Theorem 1 in the informed setting and shows that depending on the ratio between $\sigma_1$ and $\sigma_2$ a high-quality observation can be worth more than any number of low-quality observations.

Finally, Theorem 3 analyzes sufficient and necessary conditions on $n$ to achieve support recovery algorithmically.

### Strengths
+ Rigorous results
+ Study well-motivated via mixed quality data in learning problems

### Weaknesses
- Algorithmic recovery is not analyzed in the informed setting

### Questions
Since the results are interesting and the derivation appears to be rigorous, I do recommend this paper to be accepted. Due to the limited reviewing time, I have only loosely screened parts of the supplementary material, so I cannot confirm correctness of the presented results. They appear to be reasonable though.

While the paper is well-written and easy to read, the notation can be improved at certain points and there are some points that should be corrected:

- X,Y,Z are all upper-case although X is a matrix and Y,Z are vectors. In contrast $\beta$ is lower case. This is confusing.
- l. 126: „… a linear combination of ??? has the …“
- l. 199: Maybe I missed it, but I think Z_1,Z_2 are used without being defined
- l. 235: Can you comment on the fact that your required size of n_* is smaller for s=p than for s=0.5p? Clearly, there is less ambiguity in the possible supports, nevertheless, this is rather counter-intuitive and should be properly discussed in the paper.
- l. 251: „…Chernoff bound to the LHS…“ -> Shouldn’t this be RHS?
- Eq (11): $\sigma_4^2$ -> $\sigma_2^4$
- Remark 3.2, second bullet point: It would make sense to mention that the proposed estimator is motivated by the MLE discussed afterwards in Section 3.2
- Eq (20): This should be argmin and not min
- l. 604: $\hat S$ is only an element of the argmin not necessarily the only solution.
- l. 725: If I'm not mistaken, you apply Markov’s inequality here, and not a Chernoff bound. This would need to be adapted all over the document.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
Sparse recovery has typically been discussed under the assumption that noise variance is constant (homogeneous noise). However, real data often contains a mixture of a small number of "high-quality (low-noise)" observations (e.g., expert measurements) and a large number of "low-quality (high-noise)" observations (e.g., crowdsourcing or LLM labels).
This paper theoretically analyzes the support recovery problem for a sparse signal $\beta$ under mixed-quality data and defines two bounds:
i) Information-theoretic bounds: What sample size (n₁, n₂) is theoretically necessary for recovery?
ii) Algorithmic bounds: To what extent can LASSO recover the signal?

### Strengths
Originality: This paper  newly introduced the viewpoint that the Kronecker product structure is equivalent to "measuring a multidimensional signal at multiple stages." Based on this idea, it developed a signal recovery algorithm that operates with low computational complexity.
Quality: The developed algorithm reduces the run time for the recovery by $O(10^2)--O(10^3)$. In addition, theoretical guarantees are presented mathematically. The performance is also validated numerically. In summary, this paper is of high quality. 
Clarity: The writing is well structured and the results are clear. 
Significance: The results obtained in this paper are expected to bring significant advances to the problem of compressed sensing with Kronecker structures.

### Weaknesses
For problems where compressed sensing with a Kronecker structure is useful, the sparsity structure must be independent for each dimension. In the introduction, this paper mentions that there are examples of this in radar imaging and wireless communications, but does not provide specific examples. It is unclear how useful it is for practical applications.

### Questions
I understand that this paper is theoretical and not intended the practical usefulness. But, I would like to ask about its practical relevance. I could not imagine example problems for which the setup assumed in this paper are realistic. Can you raise up some?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3