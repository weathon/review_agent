# Faster Algorithms for Structured Linear and Kernel Support Vector Machines

- Decision: Accept (Poster)
- Scores: 5, 8, 8, 6

## Abstract
Quadratic programming is a ubiquitous prototype in convex programming. Many machine learning problems can be formulated as quadratic programming, including the famous Support Vector Machines (SVMs). Linear and kernel SVMs have been among the most popular models in machine learning over the past three decades, prior to the deep learning era.

Generally, a quadratic program has an input size of $\Theta(n^2)$, where $n$ is the number of variables. Assuming the Strong Exponential Time Hypothesis ($\textsf{SETH}$), it is known that no $O(n^{2-o(1)})$ time algorithm exists when the quadratic objective matrix is positive semidefinite (Backurs, Indyk, and Schmidt, NeurIPS'17). However, problems such as SVMs usually admit much smaller input sizes: one is given $n$ data points, each of dimension $d$, and $d$ is oftentimes much smaller than $n$. Furthermore, the SVM program has only $O(1)$ equality linear constraints. This suggests that faster algorithms are feasible, provided the program exhibits certain structures.

In this work, we design the first nearly-linear time algorithm for solving quadratic programs whenever the quadratic objective admits a low-rank factorization, and the number of linear constraints is small. Consequently, we obtain results for SVMs:

* For linear SVM when the input data is $d$-dimensional, our algorithm runs in time $\widetilde O(nd^{(\omega+1)/2}\log(1/\epsilon))$ where $\omega\approx 2.37$ is the fast matrix multiplication exponent;
   
* For Gaussian kernel SVM, when the data dimension $d = O(\log n)$ and the squared dataset radius is sub-logarithmic in $n$, our algorithm runs in time $O(n^{1+o(1)}\log(1/\epsilon))$. We also prove that when the squared dataset radius is at least $\Omega(\log^2 n)$, then $\Omega(n^{2-o(1)})$ time is required. This improves upon the prior best lower bound in both the dimension $d$ and the squared dataset radius.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies fast algorithms for structured linear and kernel support vector machines (SVMs). SVMs are a type of machine learning model used for tasks such as classification and regression. The main contribution of the paper is to provide an algorithm with almost linear time for solving SVM problems with low tree width or low-rank structure. This is related to sparse patterns in the Cholesky decomposition of a matrix, etc. The paper also considers how to apply the proposed method to SVMs using Gaussian kernels, and provides an estimate of the computational complexity. No numerical investigation is given into the effectiveness of the method for real-world problems, etc.

### Strengths
- The authors provide intensive mathematical analysis of robust interior point methods for quadratic optimization problems.

### Weaknesses
- The paper states that the dimensionality constraint is d=O(log(n)), but this seems like a very strict assumption for real-world data analysis. It would be good to consider how the computational complexity changes when this constraint is relaxed.
- The effectiveness of the proposed method is unclear because there are no numerical experiments. It would be useful for practitioners to see things like implementation ideas and estimates of the data size that can actually be handled.
- As only the Gaussian kernel is used as the kernel, the range of applicability is narrow. Is it possible to apply the algorithm to other kernels, while maintaining the computational efficiency? How does the computation time of the algorithm relate to the kernel width of the Gaussian kernel? 
- For kernel methods, low-rank approximation methods such as the Nystrom approximation have been developed. In addition, it is known that when there are a large number of data points, it is more computationally efficient to use deep neural networks, which have developed significantly in recent years. Given this situation, it is necessary to explain the situations in which the proposed method in this paper is effective based on real-world problems.

### Questions
see strengths and weaknesses.

**Comment after the rebuttal period**: the authors' sincere responses resolved some of my concerns. I would like to express my gratitude for their efforts. Therefore, I decided to raise my score. However, I still believe that adding numerical experiments is necessary to ensure acceptance.

### Soundness
3

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
3

### Summary
The paper proposed a nearly-linear time algorithm for solving quadratic programming problems, especially designed for kernel SVMs. It combines the robust interior point method, low-treewidth setting, low-rank factorization and low-rank kernel matrix approximation techniques to enable the almost-linear time convergence. Though the assumptions to achieve this convergence rate is relatively strong, it is the first almost-linear convergence algorithm for quadratic programming problems or kernel SVMs.

### Strengths
Novel Approach: The paper proposes an innovative nearly-linear time algorithm for solving quadratic programming problems with applications to kernel SVMs. The use of robust interior point method, low-rank factorization and low-rank matrix approximation make the approach computationally appealing.

Relevance: Given the broad usage of SVMs in machine learning, the proposed methodology could provide substantial computational improvements if the assumptions of the method are satisfied.

Complexity Analysis: The paper includes a rigorous complexity analysis and explores different scenarios for kernel SVMs under various dataset radius conditions. These insights provide a clear understanding of the expected performance under different cases. It also adheres to previous literature.

### Weaknesses
Empirical Validation: While the theoretical contribution is significant, empirical results demonstrating the practical runtime improvements and accuracy on real datasets would be beneficial. This would help validate the theoretical findings and highlight the algorithm’s scalability and precision.

Assumptions and Constraints: The assumptions, especially regarding the squared radius, may limit applicability in some scenarios. The paper could benefit from discussing the practical implications of these assumptions on dataset characteristics.

Comparison with Existing Methods: Though the theoretical basis is sound, a more detailed comparison with existing SVM solvers (e.g., LIBSVM, SVM-Light) in terms of computational complexity and memory usage could strengthen the contribution by illustrating the specific benefits and trade-offs of the proposed algorithm.

### Questions
1.	In section 1, the authors denote $B$ as the squared radius, then in section 2, the authors redefine $B = Q + t H$. Please make sure the notation is constant.

2.	Though the theoretical results of this paper are very strong, it is highly recommended that the authors could do some experiments to verify the theoretical results.

3.	It seems the assumption that square radius $B = O(log(n)/loglog(n))$ is very strong. It would be great if the authors could share some kernel SVM problems which satisfy this assumption. 

4.	Overall, in section 2, The technical details are very heavy but not clear. It would great if the authors could give more intuitions about these approaches. For example, consider a more detailed discussion on how the low-rank factorization impacts specific types of SVM problems or datasets.

5.	Small typos: for example, in line 1540, is it “memebers” or “members”? Please double check.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors develop faster quadratic program solvers using interior point methods for problems where the quadratic form matrix has low rank. The technique is then applied to yield SVM solvers with complexity linearly/nearly-linearly in the number of samples for linear/kernel SVMs.

### Strengths
SVMs are classical subjects in machine learning, and thoroughly understanding the time complexity of optimizing SVMs is important. This paper takes an important step toward this. The technique for solving QP faster using IPM with the efficient Hessian updates could be beneficial for other problems in machine learning.

### Weaknesses
While this paper addresses a theoretical problem in complexity of SVMs, it'd still be interesting to see some actual experimental results.

### Questions
The $O(\epsilon^{-2})$ time copmlexity mentioned on Line 121 was sharpened to $O(\epsilon^{-1})$ in [1]. Of course the point that the author makes about the logarithmic dependency still stands, but I think the improvement in [1] should be mentioned since the issue raised on line 124 is alleviated to some degree.

The complexity lower bound on the SVMs considered also implies complexity lower bound on QPs, correct? How does this compare to existing complexity lower bounds for QPs, if any?

Is there a reference for the result on $T$ on line 291?

What is the graph structure on the bags on line 310?

Can the authors give references for the sparse Cholesky factory result on Line 318?

What happens in the kernel SVM case when $d$ is constant (both for the upper bound and lower bound)? Why must $d$ grow with $n$ in order for the result to hold? For the lower bound, I presume it has to do with leveraging hardness results in [2]. Can the authors explain more about the $d = \Theta (\log n)$ assumption?

---

[1] Shalev-Shwartz, Shai, Yoram Singer, and Nathan Srebro. "Pegasos: Primal estimated sub-gradient solver for svm." Proceedings of the 24th international conference on Machine learning. 2007.

[2] Alman, Josh, and Ryan Williams. "Probabilistic polynomials and hamming nearest neighbors." 2015 IEEE 56th Annual Symposium on Foundations of Computer Science. IEEE, 2015.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors design the first nearly-linear time algorithm for solving SVM problems via quadratic programs whenever the quadratic objective admits a low-rank factorization, and the number of linear constraints is small. They have derived the algorithm complexity analysis with different cases, which improves the existing result.

### Strengths
The paper clearly states the contribution, the story is easy to follow. This is a solid work on solving SVM dual problem based on quadratic programming.

### Weaknesses
1. No experiment to demonstrate the superiority to validate the new claimed contributions.
2. The result works for vanilla SVM and Gaussian Kernel, however, the analysis on other kernel options such as polynomial is blank, thus, whether it can be generalized into broader applications remains unknown. 
3. What if the kernel is a combination of kernels, say K_new=K1+K2 where K1 and K2 are Guassian kernel?

### Questions
1. I know Eq. (1) is quadratic programming, but I believe the inequality constraint should be generalized into $Ax \le 0$, instead of simple $x\le 0$, though it is a special case.
2. For example, when ε is set to be 10^{−3} to account for the usual machine precision errors, these algorithms would require at least 10^6 iterations to converge. I don't understand this, as the complexity is O(d/ε), I thought it should be 1000*d. 
3. I know this is a theoretic paper, but I still hope the authors can compare the convergence with others which will be more convincing to reviewers and audiences. My experience told me that though libsvm is old, the speed is amazingly fast. I definitely hope the proposed algorithm can outperform libsvm.

### Soundness
3

### Presentation
3

### Contribution
3
