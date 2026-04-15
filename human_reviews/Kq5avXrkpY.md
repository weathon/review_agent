# Federated Optimization Algorithms with Random Reshuffling and Gradient Compression

- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
Gradient compression is a popular technique for improving communication complexity of stochastic first-order methods in distributed training of machine learning models. However, the existing works consider only with-replacement sampling of stochastic gradients. In contrast, it is well-known in practice and recently confirmed in theory that stochastic methods based on without-replacement sampling, e.g., Random Reshuffling (RR) method, perform better than ones that sample the gradients with-replacement. In this work, we close this gap in the literature and provide the first analysis of methods with gradient compression and without-replacement sampling. We first develop a naïve combination of random reshuffling with gradient compression (Q-RR). Perhaps surprisingly, but the theoretical analysis of Q-RR does not show any benefits of using RR. Our extensive numerical experiments confirm this phenomenon. This happens due to the additional compression variance. To reveal the true advantages of RR in the distributed learning with compression, we propose a new method called DIANA-RR that reduces the compression variance and has provably better convergence rates than existing counterparts with with-replacement sampling of stochastic gradients. Next, to have a better fit to Federated Learning applications, we incorporate local computation, i.e., we propose and analyze the variants of Q-RR and DIANA-RR -- Q-NASTYA and DIANA-NASTYA that use local gradient steps and different local and global stepsizes. Finally, we conducted several numerical experiments to illustrate our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces an innovative method to enhance communication efficiency in distributed machine learning model training. The proposed method incorporates without-replacement sampling and gradient compression, leading to improved performance in comparison to existing algorithms. The paper provides theoretical analysis and experimental results to support the effectiveness of the proposed approach.

### Strengths
1.The paper introduces an innovative method to enhance communication efficiency in distributed machine learning model training. This approach incorporates without-replacement sampling and gradient compression, leading to improved performance compared to existing algorithms.

2.The paper offers a comprehensive validation of the proposed approach by providing both theoretical analysis and empirical results. These findings illustrate the superiority of the proposed method over existing algorithms in terms of convergence rate and communication efficiency.

3.In addition to its contributions, the paper conscientiously addresses the limitations and challenges associated with the proposed approach. It also suggests potential avenues for future research in this area.

### Weaknesses
The DIANA-NASTYA algorithm's theoretical analysis is conducted without the need for a strongly convex assumption. A strongly convex assumption often allows for more precise and efficient convergence guarantees. To investigate the impact of such an assumption, further analysis would be necessary to determine whether the algorithm's performance improves, and if so, to what extent.

The experiments primarily revolve around deep learning applications, which are typically considered non-convex problems. However, it's important to note that the paper lacks theoretical analysis specifically tailored to these non-convex problem settings.

This work appears to be an incremental extension of DIANA. While it introduces some additional techniques, the improvements achieved may not be substantial or readily discernible.

### Questions
see the weaknesses.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper combines several existing techniques to accelerate the communication complexity of distributed stochastic optimization. It is known that random reshuffling gives better convergence rate of stochastic gradient descent, and gradient compression can save communication bandwidth by sending fewer bits over the network. It is quite natural to consider a distributed learning algorithm with both random reshuffling and gradient compression. However, the noise introduced by the gradient compression might cancel out the improvements of convergence of random reshuffling, thus it is not a priori clear if the combination is actually useful. This paper proves several theoretical guarantees and that random reshuffling can indeed improve upon some existing algorithms with gradient compression. This paper also provide experiments that demonstrate the results.

### Strengths
The theorems and proofs are stated very clearly. I don't see any major flaws in the proofs. The paper discusses the improvements over previous results thoroughly.

### Weaknesses
The results in this paper are not suprising. The proofs only utilize existing methods and techniques and are more or less routine.

### Questions
I only have one major question. I notice that the authors prove some non-strongly convex results in the appendix B.2 and C.2. Can the authors provide some discussion on this matter? How does the non-strongly convex setting change the results and the improvements? Have the authors considered nonconvex setting? I really like to see more discussion for alternative assumptions.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this work, authors study the behavior of Federated learning with gradient compression and without-replacement sampling. Authors first develop a distributed variant of random reshuffling with gradient compression (Q-RR) and show that the compression variance will overwhelm the variance of the gradient. Next, authors propose a variant
of Q-RR called Q-NASTYA, which uses local gradient steps and different local and global stepsizes.

Thanks for the authors' feedback, I have changed my score accordingly.

### Strengths
The idea sounds interesting to me. In federated learning, the communication cost can be the bottleneck for the scalability of the training system, especially on edge devices. Moreover, without-replacement has attracted lots of interest in recent studies. The motivation of this work is solid.

### Weaknesses
There are several questions that need to be addressed:
1) I see no experiment results in the main draft, although authors put those details in the supplementary, I still believe it shall be put in the main part.
2) The communication compression is at most 50% since only half of the rounds are compressed. This is not the optimal design since there should be a better way for compressing the second round of communication, since you have already assumed that all workers participate the updating of the global x_t at each round.
3) About the compression, why not use the error-compensated compression (e.g. DoubleSqueeze and PowerSGD) since it is the SOTA method for incorporating compression in optimization? With error compensation, the variance of the compression introduced in the training will be greatly reduced and you even do not need the compression to be unbiased.
4) The paper is hard to follow, I still cannot get a high level comparison of your algorithms with other works and fail to find a clue about why your design works.

### Questions
Please refer to my question about the weakness part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the convergence of federated learning algorithms with gradient compression and random reshuffling.

### Strengths
The authors show that the naive combination of compression and reshuffling (called Q-RR) doesn’t outperform compressed SGD without reshuffling. To alleviate this issue, they develop an algorithm combining Q-RR with DIANA, hence reducing the compression variance. They then introduce a version of the algorithm supporting the local steps.

### Weaknesses
Also questions included below:

-- The notation for $\pi$ in section 1.3 doesn’t match the rest of the paper.

-- Algorithm 2, line 3: is $\pi_m$ sampled for each machine $m$? Do machines have different permutations? Same for other algorithms.

-- Algorithm 3: Are lines 6 and 7 preformed on the server? Then they are not performed in parallel

-- Definition 2.1 - What is the meaning of the sequence? What is the “scope” of the sequence (you have different $\pi_m^i$ for different 
$t$)? If this sequence is different for each $t$, how do you aggregate $\sigma_{rad}^2$ over different $t$ (do you take a maximum)? Also, $x_\star$ is undefined

-- I think that derivations (e.g. on page 25) are rather hard to parse. I would introduce auxiliary notation for $f_m^{\pi_m^i}$ and for $h \cdots$.

### Questions
-- Page 4 - “Finally, to illustrate our theoretical findings we conduct experiments on federated linear regression tasks.” - Did you mean logistic regression? Also, I believe that this line makes your experimental results look weaker than what they actually are.

-- Since you use $\zeta_\star$ and $\sigma_\star$ in multiple results, they should be defined in an Assumption, instead of being defined in Theorem 2.1

-- Page 7 - “we can do enable…”

-- Page 15, after Equation (9) - broken math

-- Page 24: $f^{i, \pi}$

-- “mygreen” in bookmark names, e.g. “Algorithm mygreen Q-RR”

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
